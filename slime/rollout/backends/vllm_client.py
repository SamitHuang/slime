import logging

from slime.rollout.backends.base_client import BackendCapabilities, RolloutBackendClient
from slime.rollout.base_types import RolloutBackendRequest, RolloutBackendResponse
from slime.utils.http_utils import post

logger = logging.getLogger(__name__)

_FINISH_REASON_MAP = {
    "stop": "stop",
    "length": "length",
    "abort": "abort",
    "end_turn": "stop",
    "max_tokens": "length",
}


class VLLMClient(RolloutBackendClient):
    def __init__(self, args):
        self.args = args
        self._max_retries = getattr(args, "vllm_max_retries", 3)

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supports_abort=False,
            supports_routed_experts=False,
            supports_prompt_logprobs=False,
        )

    async def generate(
        self,
        request: RolloutBackendRequest,
        base_url: str,
        headers: dict | None = None,
    ) -> RolloutBackendResponse:
        sp = request.sampling_params
        payload = {
            "prompt": request.input_ids,
            "max_tokens": sp.get("max_new_tokens", 1024),
            "temperature": sp.get("temperature", 1.0),
            "top_p": sp.get("top_p", 1.0),
            "top_k": sp.get("top_k", -1),
            "stop": sp.get("stop"),
            "stop_token_ids": sp.get("stop_token_ids"),
            "skip_special_tokens": sp.get("skip_special_tokens", True),
            "spaces_between_special_tokens": sp.get("spaces_between_special_tokens", False),
            "logprobs": 1 if request.return_logprob else None,
            "include_stop_str_in_output": sp.get("no_stop_trim", False),
            "return_token_ids": True,
            "seed": sp.get("sampling_seed"),
        }
        payload = {k: v for k, v in payload.items() if v is not None}

        url = f"{base_url.rstrip('/')}/v1/completions"
        output = await post(url, payload, headers=headers)

        choice = output.get("choices", [{}])[0]
        text = choice.get("text", "")
        finish = choice.get("finish_reason", "stop")
        finish_reason = _FINISH_REASON_MAP.get(finish, "stop")

        # vLLM /v1/completions response format:
        #   choice.token_ids: list[int]  (output token IDs)
        #   choice.logprobs.token_logprobs: list[float|None]
        #   choice.logprobs.tokens: list[str]  (token text, not IDs)
        output_token_ids = choice.get("token_ids") or []
        logprobs_obj = choice.get("logprobs") or {}
        raw_logprobs = logprobs_obj.get("token_logprobs") or []
        output_token_logprobs = [float(lp) if lp is not None else 0.0 for lp in raw_logprobs]

        # If token_ids not in response, fall back to tokenizer
        if not output_token_ids and text:
            logger.warning("vLLM response missing token_ids, falling back to tokenizer")
            from slime.utils.processing_utils import load_tokenizer
            tokenizer = load_tokenizer(self.args.hf_checkpoint, trust_remote_code=True)
            output_token_ids = tokenizer.encode(text, add_special_tokens=False)

        # Ensure logprobs list matches token count
        if len(output_token_logprobs) < len(output_token_ids):
            output_token_logprobs.extend([0.0] * (len(output_token_ids) - len(output_token_logprobs)))

        usage = output.get("usage", {})
        return RolloutBackendResponse(
            text=text,
            output_token_ids=output_token_ids,
            output_token_logprobs=output_token_logprobs,
            finish_reason=finish_reason,
            prompt_tokens=usage.get("prompt_tokens", len(request.input_ids)),
            completion_tokens=usage.get("completion_tokens", len(output_token_ids)),
            backend_raw=output,
            routed_experts=None,
        )
