"""Clause splitter: breaks sentences into speakable clauses for streaming TTS.

The rule: split at comma, semicolon or colon only when at least MIN_CLAUSE_WORDS
words precede the boundary (measured from the last cut point). Hard-split at
MAX_CLAUSE_WORDS when a clause grows too long with no boundary in sight.

This keeps short interjections ("Yes, go ahead.") as single clauses while
allowing "I found three results, and here they are." to be split so the first
clause can be synthesised and queued for playback before the LLM has finished
producing the sentence.
"""
from __future__ import annotations

import re

MIN_CLAUSE_WORDS = 4         # do not split before this many words
MAX_CLAUSE_WORDS = 25        # hard-split if no valid boundary is found before this
CLAUSE_BUFFER_SIZE = 2       # synthesise at most this many clauses ahead of playback
CLAUSE_GAP_MAX_MS = 300      # silence between clauses above this is audibly noticeable

# Comma, semicolon or colon followed by one or more whitespace characters.
_BOUNDARY_RE = re.compile(r"[,;:]\s+")


def split_into_clauses(text: str) -> list[str]:
    """Split a sentence into speakable clauses for streaming TTS.

    Returns a non-empty list; the input is returned as a single clause when no
    valid split point exists. Punctuation (comma, semicolon, colon) is kept at
    the end of its clause, not moved to the start of the next one.
    """
    text = text.strip()
    if not text:
        return []

    # Find all valid split positions: boundaries where enough words precede them.
    cuts: list[tuple[int, int]] = []   # (punc_pos, after_whitespace_pos) in text
    last_cut_pos = 0

    for m in _BOUNDARY_RE.finditer(text):
        words_since_last_cut = len(text[last_cut_pos:m.start()].split())
        if words_since_last_cut >= MIN_CLAUSE_WORDS:
            cuts.append((m.start(), m.end()))
            last_cut_pos = m.end()

    # Build clause list from the cut positions.
    clauses: list[str] = []
    pos = 0
    for punc_start, space_end in cuts:
        clause = text[pos : punc_start + 1].strip()  # include the punctuation mark
        if clause:
            clauses.append(clause)
        pos = space_end

    # Handle tail (text after the last cut) with hard-split when it is too long.
    tail = text[pos:].strip()
    if tail:
        words = tail.split()
        while len(words) > MAX_CLAUSE_WORDS:
            clauses.append(" ".join(words[:MAX_CLAUSE_WORDS]))
            words = words[MAX_CLAUSE_WORDS:]
        if words:
            clauses.append(" ".join(words))

    return clauses
