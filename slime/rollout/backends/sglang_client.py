import logging

import numpy as np
import pybase64
import sglang_router
from packaging.version import parse

from slime.rollout.backends.base_client import BackendCapabilities, RolloutBackendClient
from slime.rollout.base_types import RolloutBackendRequest, RolloutBackendResponse
from slime.utils.http_utils import get, post

logger = logging.getLogger(__name__)


class SGLangClient(RolloutBackendClient):
    def __init__(self, args):
        self.args = args

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supports_abort=True,
            supports_routed_experts=bool(getattr(self.args, "use_rollout_routing_replay", False)),
            supports_prompt_logprobs=True,
        )

    async def generate(
        self,
        request: RolloutBackendRequest,
        base_url: str,
        headers: dict | None = None,
    ) -> RolloutBackendResponse:
        payload = {
            "sampling_params": request.sampling_params,
            "return_logprob": request.return_logprob,
            "return_routed_experts": request.return_routed_experts,
        }
        # For multimodal: send raw text so SGLang's server-side processor can
        # expand image placeholders. Otherwise send pre-tokenized input_ids.
        if request.image_data and request.text is not None:
            payload["text"] = request.text
            payload["image_data"] = request.image_data
        else:
            payload["input_ids"] = request.input_ids
            if request.image_data:
                payload["image_data"] = request.image_data

        url = f"{base_url.rstrip('/')}/generate"
        output = await post(url, payload, headers=headers)

        meta = output.get("meta_info", {})
        logprobs = meta.get("output_token_logprobs", [])
        output_token_ids = [item[1] for item in logprobs]
        output_token_logprobs = [item[0] for item in logprobs]

        finish_reason = meta.get("finish_reason", {}).get("type", "stop")
        routed_experts = None
        if "routed_experts" in meta and self.capabilities.supports_routed_experts:
            num_layers = getattr(self.args, "num_layers", 0)
            moe_topk = getattr(self.args, "moe_router_topk", 1)
            if num_layers and moe_topk:
                routed_experts = np.frombuffer(
                    pybase64.b64decode(meta["routed_experts"].encode("ascii")),
                    dtype=np.int32,
                ).reshape(
                    len(request.input_ids) + len(output_token_ids) - 1,
                    num_layers,
                    moe_topk,
                )

        return RolloutBackendResponse(
            text=output.get("text", ""),
            output_token_ids=output_token_ids,
            output_token_logprobs=output_token_logprobs,
            finish_reason=finish_reason,
            prompt_tokens=meta.get("prompt_tokens", len(request.input_ids)),
            completion_tokens=meta.get("completion_tokens", len(output_token_ids)),
            backend_raw=output,
            routed_experts=routed_experts,
        )

    async def abort(self) -> list[str]:
        base = f"http://{self.args.sglang_router_ip}:{self.args.sglang_router_port}"
        if parse(sglang_router.__version__) <= parse("0.2.1") or getattr(self.args, "use_slime_router", False):
            r = await get(f"{base}/list_workers")
            return r.get("urls", [])
        r = await get(f"{base}/workers")
        return [w["url"] for w in r.get("workers", [])]
