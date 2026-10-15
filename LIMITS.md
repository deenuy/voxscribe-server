# LIMITS.md

Recorded limits per VOICE-AUTOPILOT.md section 4: bars that cannot be met
in the current build, with the measured number and the reason.

## TTS engine: Kokoro MLX incompatible with Python 3.13

**Check blocked:** V1 TTS quality (naturalness of voice)
**Bar:** natural multi-voice TTS matching Kokoro MLX quality
**Current:** macOS `say` (Samantha voice) stub in tts_engine.py
**Number:** say latency ~300-400 ms per sentence on MacBook M3; Kokoro would be ~380 ms
**Reason:** Kokoro MLX requires Python 3.10-3.12 (blis build failure on 3.13).
  Server runs Python 3.13.2. Piper fallback is GPL-3.0 (commercial risk).
**Options:**
  (a) Pin server to Python 3.12 -- requires pyproject.toml change + uv recreate
  (b) Kokoro in a separate uv venv, called via subprocess
  (c) Accept Piper GPL-3.0 for P1 only
**Decision:** NEEDS OWNER. Use `say` stub in P1 rounds.
**Round added:** R6

## Streaming TTS: first clause not yet streaming

**Check blocked:** V1-P01 (first audio before LLM finishes generating)
**Bar:** first audio starts before LLM generation completes
**Current:** full reply collected before TTS synthesis begins (see reply_chain.py LIMITS comment)
**Number:** end-to-end latency ~1.2-2.0 s depending on reply length (macOS say includes synthesis time)
**Reason:** macOS `say` synthesizes the whole text at once; streaming requires sentence-boundary
  detection and pipelining synthesis with generation.
**Resolution path:** implement sentence-boundary splitter in reply_chain.py; synthesize first
  sentence while LLM streams the rest. Blocked on Kokoro/Piper decision.
**Round added:** R6

## Playback reference not wired to AudioEngine

**Check blocked:** V3 (echo cancellation for barge-in)
**Bar:** AudioEngine playback_ref ring populated with TTS audio
**Current:** tts_engine.py uses AudioEngine.play() which does populate the ref ring,
  but the wake loop uses its own sd.InputStream (not AudioEngine's capture). So the
  echo guard cannot correlate playback reference with capture frames.
**Resolution path:** migrate wake loop capture to AudioEngine.subscribe() in R7 barge-in round.
**Round added:** R6
