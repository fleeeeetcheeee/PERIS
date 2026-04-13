from __future__ import annotations

from typing import Any

from .base import BaseAgent


class SourcingAgent(BaseAgent):
    def __init__(self, llm: Any | None = None) -> None:
        """Initialize the sourcing agent."""
        super().__init__(llm=llm)

    def build_chain(self) -> Any:
        """Build the sourcing workflow chain."""
        raise NotImplementedError

    def run(self, input: dict[str, Any]) -> dict[str, Any]:
        """Run the sourcing workflow for incoming deal data."""
        raise NotImplementedError

    def identify_targets(self, market_map: dict[str, Any]) -> dict[str, Any]:
        """Identify potential investment targets from market data."""
        raise NotImplementedError

    def prioritize_targets(self, candidates: list[dict[str, Any]]) -> dict[str, Any]:
        """Rank candidate targets for downstream review."""
        raise NotImplementedError
