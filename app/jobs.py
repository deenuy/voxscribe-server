"""A processing job.

One job per session. The job id is derived from the session id so enqueueing
the same session twice is a no-op, which keeps the pipeline idempotent under
client retries and crash recovery.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Job:
    session_id: str
    idea_id: str
    tenant_id: str
    workspace_id: str
    project_id: Optional[str]
    session_title: str
    source_type: str
    source_ref: str
    participants: int
    context: str
    audio_path: Optional[str]
    attempts: int = 0

    @property
    def job_id(self) -> str:
        return f"job-{self.session_id}"

    def to_row(self, status: str, error: str | None = None) -> dict:
        payload = {
            "session_id": self.session_id,
            "idea_id": self.idea_id,
            "tenant_id": self.tenant_id,
            "workspace_id": self.workspace_id,
            "project_id": self.project_id,
            "session_title": self.session_title,
            "source_type": self.source_type,
            "source_ref": self.source_ref,
            "participants": self.participants,
            "context": self.context,
            "audio_path": self.audio_path,
            "attempts": self.attempts,
        }
        return {
            "job_id": self.job_id,
            "session_id": self.session_id,
            "tenant_id": self.tenant_id,
            "workspace_id": self.workspace_id,
            "status": status,
            "attempts": self.attempts,
            "error": error,
            "payload": json.dumps(payload),
        }

    @classmethod
    def from_row(cls, row: dict) -> "Job":
        p = json.loads(row["payload"])
        return cls(**p)
