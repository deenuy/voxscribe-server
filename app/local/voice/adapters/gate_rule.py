"""Gate over the fixed-rule safety gate (app.local.gate): the mode rules, then G7.

The content rules (classroom, senior care) see the user's text and the sentence. G7 replaces a
sentence that claims a completed action (reminder, alarm, timer, calendar) unless the session
has a tool row of that type; the host fetches tool rows only when `needs_session_tools` says so.
"""
from __future__ import annotations

from collections.abc import Sequence

from voxscribe.session.ports import GATE_PASS, GATE_REFUSE, GATE_REPLACE, GateDecision

from app.local import gate


class RuleGate:
    fail_closed_phrase = gate.DEFAULT_REFUSAL_PHRASE

    def __init__(self, mode: str = "default") -> None:
        self.mode = mode

    def needs_session_tools(self, sentence: str) -> bool:
        return gate.find_action_claim(sentence) is not None

    def check(self, sentence: str, session_tools: Sequence[str] = (), *,
              user_text: str = "") -> GateDecision:
        decision, category = gate.check(user_text, sentence, mode=self.mode)
        if decision != GATE_PASS:
            return GateDecision(GATE_REFUSE, category, gate.refusal_phrase(category, self.mode))
        if self.needs_session_tools(sentence):
            decision, reason = gate.check_action_claim(sentence, session_tools)
            if decision == GATE_REPLACE:
                return GateDecision(GATE_REPLACE, gate.ACTION_CLAIM_REASON,
                                    gate.ACTION_CLAIM_PHRASE)
        return GateDecision(GATE_PASS)
