"""The tool registry (PL43): every tool the engine may run, validated when the engine starts.

This replaces the earlier ToolDef stub, which nothing populated or read. A registry is built from
declarative records (app/local/tools/spec.py); a record that lacks a field, or names one twice,
raises ToolSpecError and the engine does not start. The engine builds its registry once in
create_app() and keeps it on app.state.tool_registry.

It answers one question deterministically: which tools may this session use. A tool is offered
when the session's product mode is in its `modes` and, for a tool that leaves the device
(`network: remote`), the network decision (app/local/tools/network.py) allows it. permits() is the
same rule for one tool, and the executor (app/local/tools/runtime.py) asks it before every call,
so a tool that is not offered cannot be run by calling it directly either.

The registry itself is pure (no I/O); with spec.py it is shaped to move into voxscribe-core (PL58).
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping

from app.local.tools.network import NetworkDecision
from app.local.tools.spec import ToolSpec, ToolSpecError

REASON_UNKNOWN_TOOL = "unknown_tool"
REASON_MODE_NOT_ALLOWED = "mode_not_allowed"
REASON_NETWORK_UNKNOWN = "network_unknown"

_MODE_ALIASES = {"seniorcare": "senior_care"}


def canonical_mode(mode: str) -> str:
    return _MODE_ALIASES.get(mode, mode)


class ToolRegistry:
    def __init__(self, records: Iterable[Mapping]) -> None:
        specs: dict[str, ToolSpec] = {}
        for record in records:
            spec = ToolSpec.from_record(record)
            if spec.name in specs:
                raise ToolSpecError(spec.name, "name", "declared twice")
            specs[spec.name] = spec
        self._specs = specs

    def get(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(self._specs.values())

    @staticmethod
    def permits(spec: ToolSpec, mode: str,
                network: NetworkDecision | None) -> str | None:
        """None when `spec` may run in `mode` under `network`; otherwise the refusal code.

        A remote tool is judged by the network decision first, so its refusal carries the
        decision's reason (mode_restricted, profile_network_off, privacy_offline)."""
        if spec.network == "remote":
            if network is None:
                return REASON_NETWORK_UNKNOWN
            if not network.allowed:
                return network.reason
        if canonical_mode(mode) not in spec.modes:
            return REASON_MODE_NOT_ALLOWED
        return None

    def offered(self, mode: str, network: NetworkDecision | None) -> tuple[ToolSpec, ...]:
        """The tools this session may use, in declaration order."""
        return tuple(s for s in self._specs.values() if self.permits(s, mode, network) is None)

    def needs_network_decision(self, mode: str) -> bool:
        """True when some remote tool is declared for `mode`, so offered() needs the decision."""
        mode = canonical_mode(mode)
        return any(s.network == "remote" and mode in s.modes for s in self._specs.values())


def default_registry() -> ToolRegistry:
    """The first-party tools (app/local/tools/builtin.py)."""
    from app.local.tools.builtin import FIRST_PARTY

    return ToolRegistry(FIRST_PARTY)


def tool_registry(app) -> ToolRegistry:
    """The engine's registry, from app.state; built (and validated) on first use when the app
    was not made by create_app()."""
    registry = getattr(app.state, "tool_registry", None)
    if registry is None:
        registry = default_registry()
        app.state.tool_registry = registry
    return registry
