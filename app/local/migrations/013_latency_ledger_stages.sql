-- Add per-stage timing columns to latency_ledger for the five-segment breakdown:
-- turn_end_ms  → asr_final_ms  → llm_request_sent_ms  → llm_first_token_ms  → tts_first_audio_ms  → playback_end_ms
ALTER TABLE latency_ledger ADD COLUMN IF NOT EXISTS asr_final_ms         INTEGER;
ALTER TABLE latency_ledger ADD COLUMN IF NOT EXISTS llm_request_sent_ms  INTEGER;
