"""Sequential idea id allocation: idea-NNNN, zero padded.

Next id is one past the highest stored id, floored by IDEASCRIBE_ID_FLOOR so a
fresh backend can start above the client seed range (the client ships seeds up
to idea-0052) and avoid collisions.
"""

from __future__ import annotations

import os
import re
import threading

from .store import store

_FLOOR = int(os.environ.get("IDEASCRIBE_ID_FLOOR", "53"))
_lock = threading.Lock()
_PATTERN = re.compile(r"idea-(\d+)")


def _num(idea_id: str) -> int:
    m = _PATTERN.fullmatch(idea_id)
    return int(m.group(1)) if m else 0


def next_idea_id() -> str:
    with _lock:
        highest = max((_num(i) for i in store.all_idea_ids()), default=_FLOOR - 1)
        return f"idea-{max(highest + 1, _FLOOR):04d}"
