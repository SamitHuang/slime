from abc import ABC, abstractmethod
from dataclasses import dataclass

from slime.rollout.base_types import RolloutBackendRequest, RolloutBackendResponse


@dataclass
class BackendCapabilities:
    supports_abort: bool
    supports_routed_experts: bool
    supports_prompt_logprobs: bool


class RolloutBackendClient(ABC):
    @property
    @abstractmethod
    def capabilities(self) -> BackendCapabilities:
        ...

    @abstractmethod
    async def generate(
        self,
        request: RolloutBackendRequest,
        base_url: str,
        headers: dict | None = None,
    ) -> RolloutBackendResponse:
        ...

    async def abort(self) -> list[str]:
        """Return worker URLs for abort. Empty for backends without worker-level abort."""
        return []
