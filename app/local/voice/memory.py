"""Rule-based fact extraction and in-session memory for the reply path (T19).

Facts are extracted from user speech by regex patterns; they are retrieved before every LLM
reply as a [Memory] block prepended to the system prompt.  No persistence across sessions:
MemoryStore is created at session start and discarded on session close.

No ML; no LLM; no database (session scope only).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

MEMORY_BLOCK_HEADER = "[Memory]"


@dataclass
class Memory:
    key: str            # normalised fact label, e.g. "user_name"
    value: str          # extracted value, e.g. "Alex"
    turn_n: int         # which turn the fact was extracted from
    forgotten: bool = False


# ---------------------------------------------------------------------------
# Extraction rules
# Each entry is (key, compiled_pattern).  The pattern must have a named group
# "value" that captures the fact.  Rules are tried in order; a later turn for
# the same key updates the existing entry.

_EXTRACTION_RULES: list[tuple[str, re.Pattern[str]]] = [
    # user_name: "my name is <Name>" takes priority over "I'm <Name>"
    (
        "user_name",
        re.compile(r"\bmy name is (?P<value>[A-Z][a-z]+(?:\s[A-Z][a-z]+)*)", re.I),
    ),
    (
        "user_name",
        re.compile(r"\bi[' ]?m (?P<value>[A-Z][a-z]+)(?=[\s,.]|$)", re.I),
    ),
    # user_location: "I'm from <X>" / "I live in <X>"
    (
        "user_location",
        re.compile(r"\bi[' ]?m from (?P<value>[A-Z][a-zA-Z\s]+?)(?=\.|,|$)", re.I),
    ),
    (
        "user_location",
        re.compile(r"\bi live in (?P<value>[A-Z][a-zA-Z\s]+?)(?=\.|,|$)", re.I),
    ),
    # user_role: "I work as <X>" / "I'm a <X>"
    (
        "user_role",
        re.compile(r"\bi work as (?P<value>[a-z][a-zA-Z\s]+?)(?=\.|,|$)", re.I),
    ),
    (
        "user_role",
        re.compile(r"\bi[' ]?m a (?P<value>[a-z][a-zA-Z\s]+?)(?=\.|,|$)", re.I),
    ),
    # user_preference: "I prefer <X>"
    (
        "user_preference",
        re.compile(r"\bi prefer (?P<value>[a-zA-Z\s]+?)(?=\.|,|$)", re.I),
    ),
    # user_note: "remember that <X>"
    (
        "user_note",
        re.compile(r"\bremember that (?P<value>.+?)(?=\.|$)", re.I),
    ),
]

# Forget command: "forget [my] <key-word>" or "forget that"
_FORGET_PATTERN = re.compile(
    r"\bforget(?:\s+that|\s+my\s+(?P<key_word>[a-zA-Z]+)|\s+(?P<key_bare>[a-zA-Z]+))",
    re.I,
)

# Map spoken key-words to stored keys.
_KEY_ALIASES: dict[str, str] = {
    "name": "user_name",
    "location": "user_location",
    "place": "user_location",
    "city": "user_location",
    "role": "user_role",
    "job": "user_role",
    "occupation": "user_role",
    "preference": "user_preference",
    "prefer": "user_preference",
    "note": "user_note",
}


def _resolve_key(word: str) -> str | None:
    return _KEY_ALIASES.get(word.lower())


# ---------------------------------------------------------------------------


class MemoryStore:
    """In-session memory.  Not persisted across sessions (T19 scope only).

    Usage in the reply path:
        store.extract(user_text, turn_n=turn_n)   # call every turn
        mems = store.retrieve()                   # call before building system prompt
        block = memory_block(mems)                # inject into system prefix
    """

    def __init__(self) -> None:
        self._facts: dict[str, Memory] = {}   # key -> Memory

    # ---- writes ----

    def extract(self, text: str, turn_n: int) -> list[Memory]:
        """Extract facts from text and store them.  Returns newly added or updated memories.

        Forget commands in the same utterance are processed first so that
        "forget my name, my name is Sam" correctly updates after forgetting.
        """
        # 1. Forget commands.
        for m in _FORGET_PATTERN.finditer(text):
            kw = m.group("key_word") or m.group("key_bare")
            if kw:
                key = _resolve_key(kw)
                if key:
                    self.forget(key)
            elif "that" in m.group(0).lower():
                # "forget that" with no key-word -- forget the most recent user_note
                self.forget("user_note")

        # 2. Extraction rules; first match per key wins within a single utterance.
        extracted: list[Memory] = []
        seen_keys: set[str] = set()
        for key, pattern in _EXTRACTION_RULES:
            if key in seen_keys:
                continue
            match = pattern.search(text)
            if match:
                value = match.group("value").strip()
                if value:
                    mem = Memory(key=key, value=value, turn_n=turn_n)
                    self._facts[key] = mem
                    seen_keys.add(key)
                    extracted.append(mem)

        return extracted

    def forget(self, key: str) -> bool:
        """Mark the memory for key as forgotten.  Returns True if it existed."""
        if key in self._facts:
            self._facts[key].forgotten = True
            return True
        return False

    def update(self, key: str, value: str) -> Memory:
        """Update or create the memory for key.  Clears the forgotten flag."""
        existing = self._facts.get(key)
        turn_n = existing.turn_n if existing else 0
        mem = Memory(key=key, value=value, turn_n=turn_n, forgotten=False)
        self._facts[key] = mem
        return mem

    # ---- reads ----

    def retrieve(self) -> list[Memory]:
        """Return all non-forgotten memories, sorted by key for a stable prompt block."""
        return sorted(
            (m for m in self._facts.values() if not m.forgotten),
            key=lambda m: m.key,
        )


def memory_block(memories: list[Memory]) -> str:
    """Format retrieved memories as a [Memory] block for the system prompt.

    Returns an empty string when the list is empty so the caller can guard with:
        if block := memory_block(store.retrieve()): ...
    """
    if not memories:
        return ""
    lines = [MEMORY_BLOCK_HEADER]
    for m in memories:
        lines.append(f"- {m.key}: {m.value}")
    return "\n".join(lines)
