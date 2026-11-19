# voxscribe-server

The local voice assistant engine for VoxScribe products. Runs on-device, never sends audio to the cloud.

Say **"hey vox"** and the assistant wakes, listens, replies by voice, and goes back to standby -- all from one local process, one PostgreSQL database.

## What it does

- **Wakes on "hey vox"** -- a two-stage gate (keyword detector + Silero VAD + Whisper confirm) so random speech, background noise, and other wake phrases never open a session.
- **Understands natural speech** -- Whisper ASR, turn detection with pause-band patience, barge-in during replies.
- **Replies by voice** -- Kokoro TTS with clause-level streaming (second clause starts synthesizing while the first plays).
- **Never speaks uninvited** -- every spoken output passes a speech guard; chime transcripts, sub-500ms turns, and low-ASR-confidence audio are discarded silently.
- **Background tasks** -- searches and long lookups run in a delegate lane while the assistant keeps answering simpler follow-ups.
- **Reminders and timers** -- "remind me at 3pm" creates a local reminder; no false confirmation when no tool backs it.
- **Spoken controls** -- pacing, volume, hold, repeat, shorter/longer by voice.
- **Session memory** -- names and preferences extracted from each turn, injected into the next reply.
- **Runtime-mutable profile** -- name, wake phrase, voice, speed, and system prompt change at runtime with no restart.
- **Echo cancellation** -- WebRTC AEC3 so the assistant does not hear its own replies through the mic.

## Architecture

A ports-and-adapters voice host (`app/local/voice/host.py`) wires ten typed session ports to swappable adapters. The host runs a five-stage cascade:

```
Capture → Recognize → Turn → Reply → Playback
```

No audio library, database driver, or platform call appears above the port boundary. Swap any adapter without touching the cascade logic.

```
Microphone
    │
    ▼
 PortAudioCapture  (pyaudio ring buffer, AEC reference signal)
    │  raw PCM frames
    ▼
 WhisperRecognize  (faster-whisper, finalize timeout, partial fallback)
    │  TranscriptChunk
    ▼
 TurnDetector      (pause bands, speech guard, silent close, barge-in)
    │  TurnResult
    ▼
 OllamaReply       (spoken controls, delegation routing, memory injection)
    │  reply text
    ▼
 KokoroPlayback    (clause streaming, end cue off hot path, echo guard)
```

## Quick start

```bash
# Install (Python 3.13 recommended, 3.10+ supported)
uv sync --extra wake --extra dev

# Fetch the wake detector model (once)
uv run tools/fetch_wake_models.py

# Start the engine with the voice assistant
uv run --extra wake python -m app.local.engine --wake

# Development (no auth, debug trigger enabled)
uv run --extra wake python -m app.local.engine --wake --no-auth --debug-trigger
```

The engine listens on a free port (printed at startup). Connect ideascribe-app to it via `VOXSCRIBE_ENGINE_URL` in the app's `.env`.

## Running with Docker

```bash
make build      # build the container image
make up         # build and start in the background
make health     # GET /v1/health (from outside the container: see note)
make down       # stop and remove
```

The engine binds to `127.0.0.1` (loopback) inside the container by design, so the port
exposed by Docker (`8000:8000`) is not reachable from the host or from other containers.
The `HEALTHCHECK` runs from inside the container and succeeds. External bind address and
authentication are owned by task PL08/PL27. `VOXSCRIBE_DATABASE_URL` must be set via
`env_file` or `-e` before the engine will start.

## Running with the app

```bash
cd ../ideascribe-app
make start          # starts PostgreSQL, API, engine and app together
make health         # checks all components can reach each other
make logs           # follows all logs
make stop
```

## Key configuration

```bash
# Wake detector: open-vocabulary keyword spotter (default) or trained OWW model
VOXSCRIBE_WAKE_DETECTOR=openvocab     # or: oww
VOXSCRIBE_WAKE_PHRASE="hey vox"       # phrase for the open-vocabulary detector

# Two-stage gate thresholds
VOXSCRIBE_WAKE_PATIENCE=2             # consecutive frames required
VOXSCRIBE_WAKE_VAD_THRESHOLD=0.3      # Silero speech ratio in 1s pre-roll
VOXSCRIBE_WAKE_CONFIRM_DURATION_S=2.0 # Whisper confirm window

# Echo cancellation (requires pywebrtc-audio)
VOXSCRIBE_AEC_ENABLED=true            # false to disable WebRTC AEC3 (default: true)
VOXSCRIBE_NOISE_SUPPRESS_ENABLED=false # noise suppression (raises WER on fan fixture)

# Finalize timing
FINALIZE_TIMEOUT_MS=2000              # max wait for Whisper final pass

# Barge-in (only active when AEC is on)
BARGE_IN_ENERGY_THRESHOLD=0.03
BARGE_IN_MIN_FRAMES=4                 # 160ms

# Background delegation
STALE_RESULT_S=300                    # how long a delegate result stays fresh
THINKER_TIMEOUT_S=30                  # max time for a background thinker

# Reminders
REMINDER_LOOKAHEAD_H=24               # how far ahead to scan for due reminders

# Clause streaming
MIN_CLAUSE_WORDS=4
MAX_CLAUSE_WORDS=25

# Quiet hours (comma-separated HH:MM ranges, engine timezone)
VOXSCRIBE_QUIET_HOURS="22:00-07:00"

# Auth
VOXSCRIBE_NO_AUTH=false               # true in development to skip JWT
VOXSCRIBE_DEBUG_TRIGGER=false         # true to allow POST /v1/wake/trigger
```

Full reference: `app/local/config.py` (server-level) and `voxscribe/profile/schema.py` (assistant profile).

## API

### Voice assistant

```
GET  /v1/assistant/profile          read the current assistant profile
PATCH /v1/assistant/profile         update name, wake phrase, voice, speed, system prompt, end phrases
POST /v1/assistant/profile/reset    reset to defaults
GET  /v1/assistant/voices           list available TTS voice IDs

GET  /v1/wake/state                 current engine state (armed / session / hold / …)
POST /v1/wake/power                 notify of system sleep/wake events
POST /v1/wake/trigger               open a debug session (requires --debug-trigger)
```

### Meetings / scribes

```
POST /v1/meetings                   create a meeting record
POST /v1/meetings/{id}/start        begin recording a meeting (voice session)
POST /v1/meetings/{id}/stop         end a meeting recording
GET  /v1/meetings                   list meetings
GET  /v1/meetings/{id}              one meeting with transcript, summary, tasks
PATCH /v1/meetings/{id}             edit title, notes, project
POST /v1/meetings/{id}/edits        atomic edit commit (transcript + summary + tasks)
DELETE /v1/meetings/{id}            delete a meeting
POST /v1/meetings/import            import audio, text, or YouTube URL
```

### Live transcription

```
WS  /v1/meetings/{id}/live          WebSocket; streams ASR chunks and voice events
GET /v1/health                      engine status, model states, speech activity
```

Errors return `{ error, detail }` with correct HTTP status. The database name, internal paths, and stack traces are never included.

## Database

PostgreSQL 16. The engine manages its own schema via numbered migration scripts.

```bash
make db-migrate    # apply all pending migrations
make db-shell      # psql into the development database
```

Migrations: numbered files in `app/local/migrations/`, one number each, no gaps. See `CHANGELOG.md` for which sprint introduced each.

### Migrations only add: expand now, contract later

Decision (PL32, 2026-11-18). A migration only adds: tables, columns, indexes, views, triggers, rows. Removing or
renaming anything the previous release reads or writes happens in a later migration, shipped in a later release,
once no supported version of the code uses it (the contract step).

Why: the schema outlives the code that wrote it. A person who reinstalls the previous version, a second engine
instance still on older code, and a feature branch run against a shared database all meet a schema migrated by newer
code, and each must keep working. Migrations run automatically against whatever database an engine is pointed at, so
there is no moment at which a breaking change could be coordinated by hand. On 2026-11-18 a test booted an engine on
the developer's database and applied an unmerged migration that dropped `gate_decisions`, a table the running, older
engine still wrote to.

How it is enforced: `tests/test_migration_files.py` fails on `DROP TABLE`, `DROP COLUMN`, `DROP VIEW` or `RENAME` in a
migration unless the file carries a marker naming the earlier migration that started the contract:

```sql
-- contract: completes 021 (gate_decisions has had no writer since 021's release)
```

The same file checks that numbers are unique and have no gap. The runner (`app/local/migrations.py`) applies every
file whose number is not recorded in that schema's `schema_migrations`, in ascending order, and logs a warning naming
any number it applies after a higher one, so a migration that arrives late from another branch still runs. Two
engines or two requests migrating one schema exclude each other with the advisory lock earlier releases also take.

Open contract steps:

- `gate_decisions` (expanded in 021). Since 021 the engine writes the gate's ledger to `gate_records` and never writes
  `gate_decisions`; the old table keeps its rows with `input_text` cleared, and a trigger clears any text an older
  engine still writes. Per-reply outcomes are read from the view `gate_outcomes`. Contract, in a release after the one
  that ships 021: copy into `gate_records` any `gate_decisions` rows written by older engines after 021 (as 021 copied
  the earlier ones), then drop `gate_decisions` and its trigger function, with the marker above.

Locally: `ideascribe_dev`. Production: set `DATABASE_URL`.

## voxscribe-core dependency

The server depends on `voxscribe-core`, the shared intelligence kernel. The dependency is pinned to an exact version and resolved from a wheel tracked in `wheels/`.

```bash
make core-wheel                        # rebuild from ../voxscribe-core at the current tag
make core-wheel CORE_TAG=v0.5.1        # or a specific tag
make core-dev                          # switch to editable sibling for core development
make core-pinned                       # switch back to the tracked wheel
```

Current pin: `voxscribe-core==0.5.0` (`wheels/voxscribe_core-0.5.0-py3-none-any.whl`).

## Tests

```bash
uv run --extra dev pytest tests/voice/unit/ -q     # unit tests, no mic, no DB
uv run --extra dev pytest tests/ -q                # full suite
uv run --extra dev ruff check app/                 # lint
```

No test touches a real microphone or the network, and no test touches the engine's database: tests use their own
database (`VOXSCRIBE_TEST_DATABASE_URL`, default `ideascribe_test`), created once with `make test-db`, and refuse to
run when it is the engine's database. See `tests/README.md`.

## Project structure

```
app/
  local/
    engine.py              CLI entry point (--wake, --no-auth, --debug-trigger)
    app.py                 FastAPI application factory
    voice/
      host.py              Five-stage cascade, ten session ports
      barge_in.py          Barge-in detector, HeardReply, interruption_note
      clause_splitter.py   Clause-level TTS streaming
      controls.py          Spoken controls (ControlIntent, speed, volume)
      delegate_lane.py     Background delegation (DelegateLane, TaskStatus)
      memory.py            Session memory (MemoryStore, 7 extraction patterns)
      reminder_store.py    Reminders and timers
      profile_store.py     AssistantProfileStore (thread-safe)
    routers/
      assistant.py         GET/PATCH /v1/assistant/profile, GET /v1/assistant/voices
      wake.py              GET /v1/wake/state, POST /v1/wake/power, POST /v1/wake/trigger
      meetings.py          /v1/meetings/*
      live.py              WebSocket /v1/live
    migrations/            numbered SQL migrations (expand only; see Database)
tools/
  fetch_wake_models.py     Download keyword spotter and OWW models (run once)
  train_wake_model.py      Train a custom OWW model on synthetic clips
  wake_synth_clips.py      Synthesize "hey vox" training clips
  voice_bench/             Latency and WER benchmarking harness
tests/
  voice/unit/              301 unit tests (no hardware)
  voice/room/              ROOM-level hardware verification scenarios (25 owed)
wheels/                    Tracked voxscribe-core wheel
results/archive/           Sprint evaluation artifacts
tools/archive/             Pre-sprint tool scripts
```

## Changelog

See `CHANGELOG.md`. Current version: **0.9.0** (voice assistant GA, T00-T21).
