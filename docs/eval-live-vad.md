# Live turn detection evaluation (phase S2 exit gate)

Evidence for whether the live path's default turn detector should change from the adaptive energy gate to
core's shared speech capability (ADR 0001). The default has NOT been changed by this evaluation; see the
recommendation at the end. Numbers come from `results/eval-live-vad.json`, produced by
`scripts/eval_live_vad.py` on 2026-10-12 (22:39 to 22:57 local) on the development Mac.

## Setup

- Driver: core's `LiveTranscriber` driven directly the way the live socket drives it: PCM16 16 kHz frames of
  100 ms, pushed in order, no socket, no database, no wall-clock pacing. Partial decodes off, so the ASR seconds
  measure the stable path alone. Long speech cut every 15 s (the server's `live_max_segment_ms` default).
- ASR: the server's own `LocalASRProvider`, faster-whisper `small.en`, CPU, int8, beam 5, with faster-whisper's
  own VAD filter on, exactly as in production. That inner filter already drops pure noise inside each call, so
  the live gate's value shows in what reaches the recognizer and in the time it costs, not in hallucinated lines.
- Modes: `adaptive` (today's default), `silero` and `heuristic` (core detectors under the SEGMENTATION policy:
  start at p >= 0.4 for 60 ms, pre-roll 320 ms, hangover 700 ms).
- Inputs, 16 kHz mono:
  - `clean`: `voxscribe-baseline/fixtures/fixture-10min.wav` (681.7 s, macOS Samantha, 1611 reference words,
    46 paragraphs read back to back with 0.6 s gaps; sha256 `2b2d5ece...`), scored against
    `fixture-10min.reference.txt`
  - `fan20`, `fan10`: the same mixed with fan-like noise (low-passed hiss with a 17 Hz flutter, seed 2026) at
    20 dB and 10 dB SNR measured over the active speech frames
  - `noise`: 180 s with no speech: the fan at the 10 dB level, keyboard-like clicks every 0.3 to 1.5 s, and a
    60 Hz hum with harmonics from 90 s on; about -25 dBFS overall
- Scoring (the P0 harness rules): lowercase, non-word characters to space, unique words and unique 4-grams.
  A "noise segment" is a stable segment that shares no 4-gram with the reference; the table splits those into
  segments of four or more words (the candidates for hallucination) and shorter fragments, which have no 4-gram
  and are counted whatever they say.

## Results

| Input | Mode | Word recall % | 4-gram recall % | Noise segments (4+ words) | Short segments (< 4 words) | Segments | Utterances | Speech s | ASR s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | adaptive | 95.5 | 84.0 | 4 | 9 | 144 | 46 | 681.5 | 107.8 |
| clean | silero | 95.3 | 84.3 | 5 | 20 | 185 | 46 | 681.7 | 135.5 |
| clean | heuristic | 96.5 | 84.9 | 7 | 15 | 168 | 2 | 681.4 | 128.8 |
| fan20 | adaptive | 95.3 | 83.0 | 7 | 10 | 135 | 45 | 681.1 | 132.5 |
| fan20 | silero | 95.0 | 82.8 | 7 | 16 | 168 | 46 | 681.6 | 152.1 |
| fan20 | heuristic | 93.2 | 79.3 | 5 | 12 | 145 | 14 | 681.4 | 98.0 |
| fan10 | adaptive | 94.3 | 79.7 | 6 | 7 | 132 | 42 | 680.5 | 109.2 |
| fan10 | silero | 94.3 | 79.9 | 8 | 15 | 154 | 46 | 681.6 | 103.0 |
| fan10 | heuristic | 94.6 | 81.0 | 8 | 7 | 149 | 15 | 681.0 | 90.5 |
| noise | adaptive | | | 0 | 0 | 0 | 1 | 108.9 | 0.2 |
| noise | silero | | | 0 | 0 | 0 | 0 | 0.0 | 0.0 |
| noise | heuristic | | | 0 | 0 | 0 | 0 | 0.0 | 0.0 |

On the noise-only clip the adaptive gate opened once at the start (the fan is far above its initial floor of
0.005) and handed 108.9 s of noise to the recognizer in 8 calls before its floor caught up; faster-whisper's
inner filter then discarded all of it, which is why the ASR time stayed at 0.2 s. Silero and the heuristic never
opened: no utterance, no recognizer call.

## Reading the numbers

1. The fixture is wall-to-wall synthetic speech. Every mode hands all 681 s of the speech inputs to the
   recognizer, so recall differences between modes come from where utterance boundaries fall, not from
   audio that was skipped. This fixture cannot show a gate losing quiet words after a long pause; a recording
   with real pauses and a real room is still needed for that.
2. No mode hallucinated on the speech inputs. Every "noise segment" of four or more words is a reference
   sentence the recognizer spelled differently (for example "Topic loop-back detection" became "Topic loopback
   detection" and "Phase 0" became "face-zero"), so the 4-gram match fails. The counts (4 to 8 per run) are
   within run-to-run variation and do not separate the modes.
3. Silero produces more, shorter stable lines than the adaptive gate on the same audio (185 against 144 on
   clean) with the same number of utterances (46): its boundaries (320 ms pre-roll, 700 ms hangover) make the
   recognizer split the same text into more chunks. Recall is unchanged; the transcript simply has more lines.
4. The heuristic kept whole paragraphs together on clean audio (2 utterances, closed only by the 15 s cut)
   because its probability stays up through the 0.6 s gaps, and it lost 3.7 points of 4-gram recall at 20 dB
   SNR (79.3 against 83.0): under the fan it drops words. At 10 dB it recovered (81.0), which says its
   behaviour in noise is not stable enough to rely on.
5. ASR seconds are wall-clock on a shared, loaded machine (CPU seconds in the JSON confirm the load varied
   between runs); within one input they differ by up to 25 percent for the same amount of audio. Treat them
   as a cost ceiling, not as a ranking.

## Against the S2 exit gate

The gate asks for recall at least equal to adaptive within 0.5 points and fewer noise segments.

| Mode | Recall within 0.5 of adaptive | Fewer noise segments | Verdict |
| --- | --- | --- | --- |
| silero | yes on all three inputs (+0.3, -0.2, +0.2 points of 4-gram recall; word recall -0.2, -0.3, 0.0) | yes on the noise-only clip (0 utterances and 0 recognizer calls against 1 utterance, 109 s and 8 calls); equal within variation on the speech inputs, where none are hallucinations | meets the gate |
| heuristic | no (-3.7 points at 20 dB SNR) | yes on the noise-only clip | does not meet the gate |

## Recommendation

Make `silero` the live default, in a separate commit with the owner's approval, and keep `adaptive` available
as a fallback setting. Silero matches the adaptive gate's recall on every input, is the only candidate besides
the heuristic that never opens on steady noise, and runs on the model faster-whisper already ships (about 10 ms
of CPU per second of audio). Two things to check before the switch, neither of which this fixture can answer:
a recording with real pauses and a real room (the gate's reason to exist is quiet onsets after silence), and
whether the larger number of shorter stable lines (item 3) reads well in the app's transcript view; if it does
not, the SEGMENTATION preset's hangover is the knob, to be changed with `voxscribe speech-eval` against the
golden vectors first (ADR 0001, invariant 4).

The heuristic stays what the ADR says it is: diagnostics, display and a second opinion, not the segmenter.

## Reproducing

```bash
cd voxscribe-server
nice -n 15 .venv/bin/python scripts/eval_live_vad.py                # all modes, all inputs, about 18 minutes
.venv/bin/python scripts/eval_live_vad.py --markdown                 # the table from the saved report
.venv/bin/python scripts/eval_live_vad.py --limit-seconds 20         # a smoke run
```

The generated inputs land in `results/eval-live-vad-inputs/` (ignored by git); the report in
`results/eval-live-vad.json` keeps every stable segment's text, so the scores can be recomputed with `--rescore`.
