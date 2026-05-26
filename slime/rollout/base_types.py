from dataclasses import dataclass
from typing import Any

from slime.utils.types import Sample


@dataclass
class RolloutBackendRequest:
    """Backend-agnostic rollout request."""

    input_ids: list[int]
    sampling_params: dict[str, Any]
    return_logprob: bool = True
    return_routed_experts: bool = False
    image_data: list[str] | None = None
    session_id: str | None = None


@dataclass
class RolloutBackendResponse:
    """Backend-agnostic rollout response."""

    text: str
    output_token_ids: list[int]
    output_token_logprobs: list[float]
    finish_reason: str  # "stop" | "length" | "abort"
    prompt_tokens: int
    completion_tokens: int
    backend_raw: dict
    routed_experts: Any = None


@dataclass
class RolloutFnTrainOutput:
    samples: list[list[Sample]]
    metrics: dict[str, Any] = None


@dataclass
class RolloutFnEvalOutput:
    data: dict[str, dict[str, Any]]
    metrics: dict[str, Any] = None


def call_rollout_fn(fn, *args, evaluation: bool, **kwargs):
    output = fn(*args, **kwargs, evaluation=evaluation)

    # compatibility for legacy version
    if not isinstance(output, (RolloutFnTrainOutput, RolloutFnEvalOutput)):
        output = RolloutFnEvalOutput(data=output) if evaluation else RolloutFnTrainOutput(samples=output)

    return output
