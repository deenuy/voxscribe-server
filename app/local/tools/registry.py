"""Typed tool registry for VoxScribe.

Each tool is a dataclass with a JSON schema for Ollama's tool-calling API.
Tools must declare a timeout, an untrusted-data flag, and a call() coroutine.

Only allow-listed tools may be called; no dynamic registration.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Awaitable, Callable
from typing import Any


@dataclasses.dataclass(frozen=True)
class ToolDef:
    name: str
    description: str
    parameters: dict          # JSON Schema for the function arguments
    timeout_s: float          # hard deadline for the tool call
    untrusted: bool           # True = result is user-controlled data; wrap before LLM

    # call signature: async (arguments: dict) -> str
    call: Callable[[dict], Awaitable[str]] = dataclasses.field(compare=False, hash=False)

    def to_ollama(self) -> dict:
        """Return the Ollama-format tool spec for this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# Global registry — populated by register_default_tools()
_REGISTRY: dict[str, ToolDef] = {}


def register(tool: ToolDef) -> None:
    _REGISTRY[tool.name] = tool


def get(name: str) -> ToolDef | None:
    return _REGISTRY.get(name)


def all_tools() -> list[ToolDef]:
    return list(_REGISTRY.values())


def register_default_tools() -> None:
    """Register all built-in tools. Called once at engine startup."""
    if _REGISTRY:
        return
    from app.local.tools.web_search import web_search_tool
    register(web_search_tool())
