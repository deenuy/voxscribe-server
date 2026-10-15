# HARNESS-CHANGES.md

Harness defects logged per VOICE-AUTOPILOT.md section 4.

## HC-01  test_live_quality.py: _Store mock missing set_summary_status

**Round:** R9 (CK2 P0 fix)
**File:** tests/test_live_quality.py
**Defect:** `_Store` mock class did not implement `set_summary_status(meeting_id, status)`.
  `summary_service.generate_summary` calls this method to update the summary status in Postgres.
  The test mock's omission caused `AttributeError` on every call, preventing
  `test_an_unusable_summary_is_tried_once_more` from exercising the retry logic.
  The defect was masked by a prior import error (PROMPT_VERSION missing from voxscribe-core).
**Fix:** Added `async def set_summary_status(self, meeting_id, status): pass` to `_Store`.
**Companion fix:** Added `PROMPT_VERSION = "v1"` and `generate_meeting_title()` to
  `voxscribe-core/voxscribe/meeting/summary.py` (backport from main commit 2deb318 + 187ac6b).

## HC-02  test_live_quality.py: _Store.set_summary missing MIN_WORDS_FOR_SUMMARY bypass + prompt_version kwarg

**Round:** R9 (CK2 P0 fix)
**File:** tests/test_live_quality.py
**Defect A:** `_Store` transcript `["hello there"]` (2 words) was below `MIN_WORDS_FOR_SUMMARY=20`,
  causing `generate_summary` to return early via the "too short" path before calling `summarize_meeting`.
  The retry test never exercised the code it was testing.
**Defect B:** `_Store.set_summary()` lacked the `prompt_version` kwarg added when PROMPT_VERSION was
  backported; production `summary_service.py` passes `prompt_version=PROMPT_VERSION` on every `set_summary` call.
**Fix A:** Added `monkeypatch.setattr(summary_service, "MIN_WORDS_FOR_SUMMARY", 0)` in the test.
**Fix B:** Changed `set_summary(self, meeting_id, overview, details, generated_at)` to accept `**kwargs`.
