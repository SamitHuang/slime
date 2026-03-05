"""VLLMEngine: Ray actor that launches and manages a vLLM server."""

import logging
import os
import subprocess
import tempfile
import time

import requests

from slime.ray.ray_actor import RayActor
from slime.utils.http_utils import get_host_info
from slime.utils.misc import get_free_port

logger = logging.getLogger(__name__)


class VLLMEngine(RayActor):
    """Ray actor that runs vLLM server with same interface as SGLangEngine for weight sync."""

    def __init__(self, args, rank: int, base_gpu_id: int | None = None, gpu_ids: list[int] | None = None, **kwargs):
        self.args = args
        self.rank = rank
        self.base_gpu_id = base_gpu_id or 0
        self.gpu_ids = gpu_ids or [self.base_gpu_id]
        self.server_host = None
        self.server_port = None
        self.process = None
        self._log_file = None

    def init(self, port=None, host=None, **kwargs):
        self.server_host = host or get_host_info()[1]
        self.server_port = port or get_free_port(15000)

        model = getattr(self.args, "vllm_model", None) or self.args.hf_checkpoint
        tp = self.args.rollout_num_gpus_per_engine
        gpu_ids = self.gpu_ids[:tp]
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cvd:
            visible = [int(x) for x in cvd.split(",") if x.strip()]
            dev_str = ",".join(str(visible[gid]) if gid < len(visible) else str(gid) for gid in gpu_ids)
        else:
            dev_str = ",".join(str(g) for g in gpu_ids)

        seed = getattr(self.args, "seed", 1234) + self.rank
        cmd = [
            "vllm", "serve", model,
            "--tensor-parallel-size", str(tp),
            "--port", str(self.server_port),
            "--host", "0.0.0.0",
            "--weight-transfer-config", '{"backend": "nccl"}',
            "--seed", str(seed),
            "--trust-remote-code",
        ]
        if getattr(self.args, "offload_rollout", False):
            cmd.append("--enable-sleep-mode")
        if getattr(self.args, "vllm_enforce_eager", True):
            cmd.append("--enforce-eager")
        if getattr(self.args, "fp16", False):
            cmd.extend(["--dtype", "float16"])

        env = os.environ.copy()
        env["VLLM_SERVER_DEV_MODE"] = "1"
        env["CUDA_VISIBLE_DEVICES"] = dev_str
        env.setdefault("NCCL_DEBUG", "INFO")
        env.setdefault("NCCL_DEBUG_SUBSYS", "ALL")
        env["NCCL_P2P_DISABLE"] = "1"
        env.setdefault("NCCL_IB_DISABLE", "1")

        self._log_file = tempfile.NamedTemporaryFile(
            prefix="vllm_engine_", suffix=".log", delete=False, mode="w"
        )
        logger.info("Launching vLLM: cmd=%s, CUDA_VISIBLE_DEVICES=%s, log=%s",
                     " ".join(cmd), dev_str, self._log_file.name)
        self.process = subprocess.Popen(
            cmd,
            env=env,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
        )
        self._wait_healthy()

    def _wait_healthy(self, timeout=300):
        base = f"http://{self.server_host}:{self.server_port}"
        start = time.time()
        while time.time() - start < timeout:
            try:
                r = requests.get(f"{base}/health", timeout=5)
                if r.status_code == 200:
                    logger.info("vLLM server healthy at %s:%s", self.server_host, self.server_port)
                    return
            except Exception:
                pass
            if self.process and self.process.poll() is not None:
                log_tail = self._read_log_tail()
                raise RuntimeError(f"vLLM process exited with code {self.process.returncode}.\n{log_tail}")
            time.sleep(2)
        log_tail = self._read_log_tail()
        raise TimeoutError(f"vLLM server failed to become healthy within {timeout}s.\n{log_tail}")

    def _read_log_tail(self, n=200):
        if not self._log_file:
            return ""
        try:
            self._log_file.flush()
            with open(self._log_file.name) as f:
                lines = f.readlines()
            return "".join(lines[-n:])
        except Exception:
            return ""

    def _post(self, path: str, json_data: dict | None = None, params: dict | None = None):
        url = f"http://{self.server_host}:{self.server_port}{path}"
        kwargs = {"timeout": 120}
        if json_data is not None:
            kwargs["json"] = json_data
        if params is not None:
            kwargs["params"] = params
        r = requests.post(url, **kwargs)
        if not r.ok:
            body = r.text[:2000] if r.text else "(empty)"
            log_tail = self._read_log_tail(50)
            logger.error(
                "vLLM %s returned %s: %s\n--- vLLM log tail ---\n%s",
                path, r.status_code, body, log_tail,
            )
        r.raise_for_status()
        try:
            return r.json()
        except Exception:
            return None

    def health_generate(self, timeout: float = 5.0) -> bool:
        try:
            r = requests.get(
                f"http://{self.server_host}:{self.server_port}/health",
                timeout=timeout,
            )
            r.raise_for_status()
            return True
        except requests.RequestException:
            return False

    def pause_generation(self):
        self._post("/pause", params={"mode": "abort"})

    def flush_cache(self):
        pass

    def init_weights_update_group(self, master_address, master_port, rank_offset, world_size, group_name=None, backend=None):
        logger.info(
            "Initializing NCCL weight transfer: master=%s:%s, rank_offset=%d, "
            "world_size=%d, vllm_url=http://%s:%s, vllm_log=%s",
            master_address, master_port, rank_offset, world_size,
            self.server_host, self.server_port,
            self._log_file.name if self._log_file else "<none>",
        )
        self._post("/init_weight_transfer_engine", json_data={
            "init_info": {
                "master_address": master_address,
                "master_port": master_port,
                "rank_offset": rank_offset,
                "world_size": world_size,
            }
        })
        log_tail = self._read_log_tail(30)
        logger.info("vLLM log after init_weight_transfer_engine:\n%s", log_tail)

    def update_weights_from_distributed(
        self,
        names,
        dtypes,
        shapes,
        group_name=None,
        flush_cache=False,
        weight_version=None,
        packed: bool = True,
    ):
        dtype_names = [str(d).replace("torch.", "") for d in dtypes]
        shape_lists = [list(s) for s in shapes]
        self._post("/update_weights", json_data={
            "update_info": {
                "names": names,
                "dtype_names": dtype_names,
                "shapes": shape_lists,
                "packed": packed,
            }
        })

    def continue_generation(self):
        self._post("/resume")

    def destroy_weights_update_group(self, group_name):
        pass

    def release_memory_occupation(self):
        try:
            self._post("/sleep", params={"level": "1", "mode": "abort"})
        except requests.RequestException as e:
            logger.warning("vLLM sleep failed (need --enable-sleep-mode?): %s", e)

    def resume_memory_occupation(self, tags: list[str] | None = None):
        try:
            params = {}
            if tags:
                params["tags"] = tags
            self._post("/wake_up", params=params)
        except requests.RequestException as e:
            logger.warning("vLLM wake_up failed: %s", e)

    def get_weight_version(self):
        return None

    def check_weights(self, action: str):
        pass

    def post_process_weights(self, **kwargs):
        pass

    def shutdown(self):
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
        if self._log_file:
            try:
                self._log_file.close()
            except Exception:
                pass
