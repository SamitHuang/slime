"""
UpdateVLLMWeightFromTensor
==========================

Update vLLM rollout engines using CUDA IPC (HTTP mode) when colocated on the
same GPU(s) as the trainer.  This follows the vLLM RLHF HTTP IPC example:
https://docs.vllm.ai/en/stable/examples/rl/rlhf_http_ipc/

The flow:
1. Megatron params → TP all-gather → HF conversion (via HfWeightIteratorBase)
2. Send HF-named tensors to the vLLM server using
   ``IPCWeightTransferEngine.trainer_send_weights()`` with
   ``IPCTrainerSendWeightsArgs(mode="http", url=<vllm_url>)``.
3. For any overflow (non-colocated) engines, fall back to NCCL distributed
   broadcast identical to ``UpdateWeightFromDistributed``.

The API is compatible with ``MegatronTrainRayActor`` — same ``__init__``,
``connect_rollout_engines``, and ``update_weights`` signatures as
``UpdateWeightFromTensor`` / ``UpdateWeightFromDistributed``.
"""

from __future__ import annotations

import logging
import os
import time
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence

import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from ray.actor import ActorHandle

from slime.utils.distributed_utils import get_gloo_group

from .common import all_gather_param, named_params_and_buffers
from .hf_weight_iterator_base import HfWeightIteratorBase
from .update_weight_from_distributed import (
    connect_rollout_engines_from_distributed,
    disconnect_rollout_engines_from_distributed,
    post_process_weights,
    update_weights_from_distributed,
)

logger = logging.getLogger(__name__)


class UpdateVLLMWeightFromTensor:
    """
    Update colocated vLLM engines from tensor via CUDA IPC (HTTP mode).

    Colocated path:
        Megatron weights → TP all-gather → HF conversion → CUDA IPC to vLLM
        server via ``IPCWeightTransferEngine.trainer_send_weights()``.

    Non-colocated overflow path (optional):
        Falls back to NCCL distributed broadcast via ``_NcclBridge``.
    """

    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        *,
        model_name: str,
        quantization_config: dict[str, int | str | list[str]] | None,
    ) -> None:
        self.args = args
        self.model = model
        self.weights_getter = weights_getter
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.weight_version = 0

        self._hf_weight_iterator = HfWeightIteratorBase.create(
            args=args, model=model, model_name=model_name, quantization_config=quantization_config,
        )

        # Populated by connect_rollout_engines
        self.rollout_engines: list[ActorHandle] = []
        self._colocated_engines: list[ActorHandle] = []
        self._colocated_vllm_urls: list[str] = []
        self._distributed_engines: list[ActorHandle] = []
        self._model_update_groups = None
        self._is_distributed_src_rank = False
        self._group_name = "slime"
        self._ipc_initialized = False

    # ------------------------------------------------------------------
    # connect / disconnect
    # ------------------------------------------------------------------

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
    ) -> None:
        """
        Split colocated / distributed engines.

        For colocated engines we resolve their vLLM base URLs (needed by the
        IPC weight transfer HTTP mode).  For overflow distributed engines we
        create the NCCL bridge just like ``UpdateWeightFromDistributed``.
        """
        self.rollout_engines = list(rollout_engines)
        self.rollout_engine_lock = rollout_engine_lock

        if engine_gpu_counts is None:
            engine_gpu_counts = [self.args.rollout_num_gpus_per_engine] * len(rollout_engines)
        if engine_gpu_offsets is None:
            engine_gpu_offsets = []
            offset = 0
            for c in engine_gpu_counts:
                engine_gpu_offsets.append(offset)
                offset += c

        # Determine colocated vs distributed engines
        total_actor_gpus = self.args.actor_num_nodes * self.args.actor_num_gpus_per_node
        colocate_engine_nums = 0
        for gpu_offset, gpu_count in zip(engine_gpu_offsets, engine_gpu_counts, strict=True):
            if gpu_offset + gpu_count > total_actor_gpus:
                break
            colocate_engine_nums += 1

        self._colocated_engines = list(rollout_engines[:colocate_engine_nums])
        self._distributed_engines = list(rollout_engines[colocate_engine_nums:])

        # Resolve vLLM base URLs for colocated engines (blocking Ray call)
        if dist.get_rank() == 0 and self._colocated_engines:
            url_refs = [engine.get_vllm_url.remote() for engine in self._colocated_engines]
            self._colocated_vllm_urls = ray.get(url_refs)
            logger.info("Colocated vLLM URLs for IPC weight transfer: %s", self._colocated_vllm_urls)

        # Set up NCCL bridge for distributed overflow engines
        if self._distributed_engines:
            distributed_gpu_counts = engine_gpu_counts[colocate_engine_nums:]
            self._is_distributed_src_rank = (
                mpu.get_data_parallel_rank(with_context_parallel=True) == 0
                and mpu.get_tensor_model_parallel_rank() == 0
                and mpu.get_pipeline_model_parallel_rank() == 0
            )
            if self._is_distributed_src_rank:
                if self._model_update_groups is not None:
                    disconnect_rollout_engines_from_distributed(
                        self.args, self._group_name, self._model_update_groups, self._distributed_engines,
                    )
                self._model_update_groups = connect_rollout_engines_from_distributed(
                    self.args,
                    self._group_name,
                    self._distributed_engines,
                    engine_gpu_counts=distributed_gpu_counts,
                )

    # ------------------------------------------------------------------
    # weight update
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update_weights(self) -> None:
        """
        Main entry-point called by ``MegatronTrainRayActor.update_weights()``.

        Pause → flush → init IPC (once) → send weights via IPC → resume.
        """

        self.weight_version += 1
        rank = dist.get_rank()

        # Pause generation and flush KV cache on all engines
        if rank == 0:
            all_engines = self._colocated_engines + self._distributed_engines
            ray.get([engine.pause_generation.remote() for engine in all_engines])
            ray.get([engine.flush_cache.remote() for engine in all_engines])
            if self.quantization_config and self.quantization_config.get("quant_method") in ["compressed-tensors"]:
                post_process_weights(
                    restore_weights_before_load=True,
                    post_process_quantization=False,
                    rollout_engines=all_engines,
                )
        dist.barrier(group=get_gloo_group())

        # Initialize IPC weight transfer engines (first time only)
        if rank == 0 and self._colocated_engines and not self._ipc_initialized:
            self._init_ipc_weight_transfer()
            self._ipc_initialized = True

        # Get megatron weights and iterate HF chunks
        megatron_local_weights = self.weights_getter()

        for hf_named_tensors in self._hf_weight_iterator.get_hf_weight_chunks(megatron_local_weights):
            # Send to colocated engines via IPC
            if rank == 0 and self._colocated_engines:
                self._send_via_ipc(hf_named_tensors)

            # Send to distributed engines via NCCL
            if self._distributed_engines and self._is_distributed_src_rank:
                refs = update_weights_from_distributed(
                    self._group_name,
                    self._model_update_groups,
                    self.weight_version,
                    self._distributed_engines,
                    hf_named_tensors,
                    use_vllm=True,
                    packed=True,
                )
                if refs:
                    ray.get(refs)

        dist.barrier(group=get_gloo_group())

        # Post-process and resume
        if rank == 0:
            all_engines = self._colocated_engines + self._distributed_engines
            if self.quantization_config and self.quantization_config.get("quant_method") in ["compressed-tensors"]:
                post_process_weights(
                    restore_weights_before_load=False,
                    post_process_quantization=True,
                    rollout_engines=all_engines,
                )
            # Bump weight version on sidecar for colocated engines
            for engine in self._colocated_engines:
                try:
                    ray.get(engine.set_weight_version.remote(self.weight_version))
                except Exception as exc:
                    logger.warning("Failed to set weight version on engine: %s", exc)

            ray.get([engine.continue_generation.remote() for engine in all_engines])
        dist.barrier(group=get_gloo_group())

    # ------------------------------------------------------------------
    # IPC helpers
    # ------------------------------------------------------------------

    def _init_ipc_weight_transfer(self) -> None:
        """
        Call ``/init_weight_transfer_engine`` on each colocated vLLM server.
        For IPC backend this is a no-op on the server side but is still required
        by vLLM's weight transfer protocol.
        """
        import requests

        for url in self._colocated_vllm_urls:
            try:
                resp = requests.post(
                    f"{url}/init_weight_transfer_engine",
                    json={"init_info": {}},
                    timeout=60,
                )
                resp.raise_for_status()
                logger.info("Initialized IPC weight transfer on %s", url)
            except Exception as exc:
                logger.error("Failed to init IPC weight transfer on %s: %s", url, exc)
                raise

    def _send_via_ipc(self, hf_named_tensors: list[tuple[str, torch.Tensor]]) -> None:
        """
        Send HF-named tensors to all colocated vLLM engines via CUDA IPC.

        Uses ``vllm.distributed.weight_transfer.ipc_engine.IPCWeightTransferEngine``
        with ``mode="http"`` so the IPC handles are sent via HTTP to the vLLM
        server's ``/update_weights`` endpoint.
        """
        # Allow insecure serialization for IPC handle serialization
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

        from vllm.distributed.weight_transfer.ipc_engine import (
            IPCTrainerSendWeightsArgs,
            IPCWeightTransferEngine,
        )

        for url in self._colocated_vllm_urls:
            trainer_args = IPCTrainerSendWeightsArgs(mode="http", url=url)
            IPCWeightTransferEngine.trainer_send_weights(
                iterator=iter(hf_named_tensors),
                trainer_args=trainer_args,
            )
            logger.debug("IPC weight transfer completed to %s", url)
