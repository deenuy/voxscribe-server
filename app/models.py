"""Request models for session endpoints.

Recording happens in the Electron client, which holds mic consent. This
backend never captures audio. It receives the recorded session the client
uploads and runs LLM processing. Ideas are stored and returned as plain dicts
in the exact canonical camelCase shape so the client contract is matched byte
for byte. These models exist for request validation and the OpenAPI docs.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class SessionStart(CamelModel):
    """Optional pre-allocation. The client calls this when the user starts
    recording so it can subscribe to progress before the upload exists."""

    title: str = "Untitled session"
    participants: int = 1
    context: str = ""
    project_id: Optional[str] = None
    source_type: str = "recording"  # recording | file_upload | youtube


class SessionFromUrl(CamelModel):
    url: str
    title: str = "Untitled session"
    participants: int = 1
    context: str = ""
    project_id: Optional[str] = None
