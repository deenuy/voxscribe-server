"""Live transcription support: secrets kept out of logs, and the summary of a recording finishes cleanly."""

import logging
from types import SimpleNamespace

from app.local import summary_service
from app.local.log_redaction import RedactSecretsFilter, redact
from app.local.summary_service import NO_SPEECH_OVERVIEW, SummaryService


def test_redact_hides_token_and_identity_values():
    line = '127.0.0.1:1 - "WebSocket /v1/meetings/m1/live?token=abc.def&identity=eyJx.y-z_" [accepted]'
    out = redact(line)
    assert "abc.def" not in out and "eyJx" not in out
    assert "token=[redacted]&identity=[redacted]" in out
    assert redact("no secrets here") == "no secrets here"


def test_filter_rewrites_formatted_records():
    record = logging.LogRecord("uvicorn.error", logging.INFO, __file__, 1, '%s - "WebSocket %s" %s', ("1.2.3.4", "/x?token=SECRET", "[accepted]"), None)
    assert RedactSecretsFilter().filter(record)
    assert "SECRET" not in record.getMessage()


class _Store:
    def __init__(self, texts):
        self.texts = texts
        self.saved = None

    async def get_transcript(self, meeting_id):
        return [SimpleNamespace(id=f"s{i}", text=t, startMs=0, endMs=1, source="mic") for i, t in enumerate(self.texts)]

    async def get_notes(self, meeting_id):
        return []

    async def get_enhanced(self, meeting_id):
        return {"blocks": []}

    async def set_summary(self, meeting_id, overview, details, generated_at):
        self.saved = (overview, details)


async def test_no_speech_skips_the_model(monkeypatch, caplog):
    calls = []
    monkeypatch.setattr(summary_service, "summarize_meeting", lambda *a: calls.append(a))
    store = _Store(["", "  "])
    with caplog.at_level(logging.ERROR):
        await SummaryService(store, llm=object()).generate_summary("m1")
    assert calls == []
    assert store.saved == (NO_SPEECH_OVERVIEW, [])
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


async def test_an_unusable_summary_is_tried_once_more(monkeypatch):
    attempts = []

    def summarize(*args):
        attempts.append(1)
        if len(attempts) == 1:
            raise ValueError("the model did not return a usable summary: ''")
        return SimpleNamespace(overview="We talked.", details=[SimpleNamespace(title="A", body="B")])

    monkeypatch.setattr(summary_service, "summarize_meeting", summarize)
    store = _Store(["hello there"])
    await SummaryService(store, llm=object()).generate_summary("m1")
    assert len(attempts) == 2
    assert store.saved == ("We talked.", [{"title": "A", "body": "B"}])
