# Changelog

## Unreleased

### New Features
- Added a spoken register, `VOXSCRIBE_VOICE_EXPRESSIVENESS` = `plain` | `natural` (PL66). The default follows the product mode the gate runs under: `natural` for personal and IdeaScribe, `plain` (the replies as before) for senior care and classroom. In `natural` the reply model is asked to talk as people talk, and deterministic rules hold on what it wrote before the gate: at most one reaction, only at the start and only from the interjections the voice pronounces, never the previous reply's opener, no stock assistant sentences, nothing added to scripted lines. A pronunciation table rewrites what the voice misreads (Mm-hmm, Mhm, Mm, Aww) after the gate; captions and the stored transcript keep the original spelling. The lists live in `app/local/voice/expression.py`.

- Added thinking lines for a late reply, in `natural` only (PL67): when a reply's speech has not started `VOXSCRIBE_VOICE_THINKING_AFTER_S` (1.5 s) after the turn closed, one short line is said first ("Let me see.", "One moment.", ...), never the same twice running, through the gate, muted from capture like any playback and stopped by a person speaking. None when the reply starts in time, after a yes, a no or a correction, or before a scripted route such as a lookup's announcement. The reply follows the line in the playback queue with no gap. The lines are rendered in the reply voice into the phrase cache. In `natural` the immediate "hm" before every reply is gone; `plain` keeps it.

- Added ending rules with a spoken farewell (PL61). Plain closings end a session ("that's all", "that's it, thanks", "okay, I'm done", "goodbye", "bye", "stop", "never mind, bye") as the whole short turn or its ending clause, never inside a longer sentence, and the assistant says one farewell through the gate before closing: varied and never the device's last one in `natural`, "Goodbye." in `plain`. "Stop", "wait" or "hold on" said over the assistant stops the speech and the session listens on; "stop" said while it listens ends the session. End closes at once and an idle end says nothing.

- Added `python -m app.local.voice.audition --out DIR [--play]` (PL68): every reaction, thinking line and farewell, synthesised with the configured voice, written as a WAV each, listed by name and played in turn. Each line is first put to the gate of the chosen mode. Strike a line by deleting its name in `app/local/voice/expression.py`.

### Bug Fixes
- Fixed a turn the engine cannot use (no words, one word, noise) ending the whole session (PL61). It is now ignored and the wait goes on as it was. A session that never has a usable turn still closes as silent and keeps nothing, and its captions are no longer published before that is known, so the app is never shown words of a conversation that was deleted. Waits are measured in listening time, only recognised speech extends them (loudness no longer does), and `VOXSCRIBE_VOICE_NO_SPEECH_LIMIT_S` (120 s) ends a session with no recognised speech however loud the room.
- Fixed the first turn of a voice session ending at the first-turn wait instead of on the person's pause (PL59). The turn detector settled on digital silence before every turn, so the room after the person stopped still read as speech and the closing pause started a second or more late; it now settles on the room's own sound. The first-turn wait now applies only while nothing has been recognised, and a cue is muted from capture for its whole length (the cues are about 1.6 s; the mute was 1.1 s).

## What's New in VoxScribe Server 0.10.0

Baseline release of 2026-11-19. Uses voxscribe-core 0.6.0. Starting this version applies migrations
020 to 026 to the database it is pointed at; all seven only add.

### New Features
- Added tenant isolation to the voice path: a device belongs to the person who arms it, and a voice session, its conversation, its tool calls and its reminders are stored in that person's schema.
- Added `POST /v1/tickets`: a short-lived, single-path ticket for opening a socket or an event stream, so a session token is no longer sent in an address. Access tokens must carry `typ: "access"`, and browser origins are checked.
- Added one safety path for speech: the host's gate delegates to core's policy gate, and every decision is kept in the append-only `gate_records` table.
- Added a typed event envelope (`v`, `id`, `type`, `sessionId`, `seq`, `data`) with one stream per tenant, replay from the last event received, and `stream.reset` when replay is not possible.
- Added a new first stage for the wake phrase: a speech-onset detector with a recognizer pass primed with the phrase, beside the keyword spotter. "hey vox" must begin the utterance.
- Added `POST /v1/wake/start`, which starts a voice session with no wake phrase, for a button in an app.
- Added the conversational pace as a setting: the pause that closes a turn comes from the product profile's `end_of_turn_pause_s`, or `VOXSCRIBE_END_OF_TURN_PAUSE_S`.
- Added a tool registry: one declarative record per tool, checked at start-up, with an audit row for every call and a receipt for every executed write. A reply may claim an action only when that turn has a receipt for it.
- Added one network rule for every remote tool: refused in senior care and classroom, refused when the profile turns the network off, and refused unless the tenant's privacy setting allows connectors.
- Added reminders and timers by voice: a reminder is read back and saved only on "yes", a timer starts at once, and both can be listed and cancelled by voice. Times are parsed and read back by `voice/spoken_time.py`.
- Added a scheduler that delivers due reminders and timers with no session open, after a restart, and without sound during quiet hours; `GET /v1/reminders`, `DELETE /v1/reminders/{id}` and the `reminder.due` event.
- Added the `meeting.pipeline` event and `GET /v1/meetings/{id}/pipeline`: the transcript, summary and tasks steps after Stop, each working, ready, failed or skipped with a reason code. Added the `meeting_steps` table (migration 023).
- Added `POST /v1/wake/stop` (End) and `POST /v1/wake/mute` (Mute) for a running voice session; `GET /v1/wake/state` reports `muted`.
- Added a machine-readable contract in `contract/` (`openapi.json`, `events.json`, `websocket.json`, `errors.json`), written by `python -m contract.export` and checked by the tests.

### Bug Fixes
- Fixed "Ask about this scribe" answering from a search over every source instead of the scribe's transcript, and chat searching through a connection of its own outside the tenant pool.
- Fixed the chat reply stream breaking on backslashes and control characters, and holding the event loop while the model generated.
- Fixed regenerating a summary, which never ran the model and left the summary pending; a regenerate while a summary is being written now attaches to that run.
- Fixed a task drafting timeout being logged with no reason; it is now `model_timeout` on the pipeline and can be retried with Generate.
- Fixed a saved reminder being followed by "I couldn't set that", a failed save being confirmed, and "timer set" being said without a timer.
- Fixed the assistant not answering after waking: a turn that begins with speech is now seen, and a first turn with words heard is answered when its wait runs out.
- Fixed the health route failing under isolation when no tenant was in scope.
- Fixed the delegate lane spinning when every pending write had already finished.
- Fixed a search warning that could log the query.

### Changed
- Changed the test suite to run in a dedicated `ideascribe_test` database and to refuse the development databases. An engine or a test run migrates the database it is pointed at.
- Changed migrations to expand-only: a migration may add, never drop or rename. `tests/test_migration_files.py` enforces it.
- Changed which privacy setting governs web search by voice: the tenant's stored setting, not the environment variable.
- Changed the gate's replacement line for an unbacked action claim to "I haven't set anything." (floor version `host-floor-v3`).

### Known Issues
- The first turn of a session does not close on the pause; it runs to its eight-second wait.
- A one-word turn outside a pending question ends the session with no reply, and in the first turn deletes the conversation after its captions were published. "stop" is not an ending phrase.
- The idle wait after a reply is extended by loud sound, not only by recognised speech, so the end of a session is not predictable.
- No event says that a session ended or why, and there is no single session state: `wake_state` knows off, armed, listening and error only.
- `wake_state` and the `/v1/wake/*` responses use `snake_case` fields (`meeting_id`); the other events and routes use `camelCase`.
- A refused web search ends the turn with the refusal line; the reply model is not asked instead.
- A failed refresh of the identity service's keys is retried on every request with no back-off, and logs no reason when the cause is a timeout.
- Reminders, timers and the scheduler have not been run with real audio. "set an alarm" is not recognised. The time zone is the machine's.
- The Dockerfile has never been built. The workflow in `.github/` runs the tests (which include the contract check) and the undefined-name lint, and has not yet run on GitHub; a full lint (173 findings) and a type check are not part of it.

## What's New in VoxScribe Server 0.9.0

### New Features
- Changed the wake phrase to "hey vox" behind a two-stage gate: a session starts only when the detector holds the phrase for two consecutive frames, at least 30% of the last second is voice, and a short Whisper pass confirms the phrase. "Hey jarvis" no longer wakes the assistant.
- Added two wake detectors behind one interface: an open-vocabulary keyword spotter that takes the phrase as plain text (the default) and an openWakeWord model trained on synthetic "hey vox" clips, selectable with `VOXSCRIBE_WAKE_DETECTOR=oww`.
- Added a capture watchdog that reopens the microphone stream when it stalls, delivers only digital silence, or the default input device changes, making the assistant recover automatically from hardware interruptions.
- Added quiet hours so the assistant stays silent during configured overnight windows, recording each rejected wake without playing a sound or opening a session.
- Added `POST /v1/wake/power` so the app can notify the engine of system sleep and wake events, keeping the microphone stream in sync with the device power state.
- Added a profile system with per-deployment configuration for wake sensitivity, reply style, guardrail level, network access, and enrollment requirements, loadable from YAML with no code changes.
- Added `GET /v1/assistant/profile`, `PATCH /v1/assistant/profile` and `POST /v1/assistant/profile/reset` so the assistant name, wake phrase, voice, speed, and system prompt can be changed at runtime with no restart.
- Added `GET /v1/assistant/voices` listing the available TTS voices for use in the settings screen.
- Added acoustic echo cancellation using the WebRTC AEC3 algorithm, so the assistant no longer hears its own replies through the microphone, making self-triggered sessions significantly less likely. `VOXSCRIBE_AEC_ENABLED=true` to enable; noise suppression is `VOXSCRIBE_NOISE_SUPPRESS_ENABLED` and off by default (raises WER on fan fixture).
- Added a background delegation lane that routes web searches and long lookups to a background task while the assistant keeps answering simpler follow-up questions immediately.
- Added native reminder and timer support: saying "remind me to call Alex at 3pm" creates a local reminder that fires at the next turn after the scheduled time, with no false confirmation when no action was taken.
- Added spoken controls so pacing, volume, reply length, hold, and repeat can be adjusted by voice during a session without interrupting the conversation.
- Added a session memory store that extracts facts from each turn and injects them as context for the next reply, so the assistant can use the person's name and preferences throughout a session.
- Added clause-level TTS streaming: synthesis of the second clause starts while the first clause is playing, reducing the gap between the end of the LLM reply and the start of the spoken answer.
- Added a tray state indicator that updates every two seconds to show the engine state (listening, speaking, follow-up, hold, and others) as a tooltip on the menu bar icon.
- Added an assistant settings screen in the app for changing the wake phrase, name, voice, and speed, wired to the new profile API.

- Added a delegate lane (`app/local/voice/delegate_lane.py`): a request that needs a web search is answered with "I'm looking that up, it'll take a moment" and runs as a background task, so the next question is answered while the search is still going. The result is spoken after the reply to a later turn, or when asked "what did you find?". A result that has waited longer than the stale limit (`STALE_RESULT_S`) is offered ("want me to run it again?") instead of read out.
- Added spoken cancel and change for a background task: "cancel that", "never mind", "forget it" and "stop searching" stop it and nothing is spoken later; "actually, ..." and "search for ... instead" replace it and only the new result is spoken.
- Added the `delegate_tasks` table (migration 017), one row per background task with its status and result, and a `task_id` column on `tool_calls` so every tool call names the task that made it.

### Bug Fixes
- Fixed the acknowledgement chime being transcribed as the user's first words, which caused the assistant to reply to its own sound at night; the chime is now muted in the capture signal for a window after it plays.
- Fixed sessions opening from a single noisy frame with no actual speech; the two-stage gate now requires patience, voice activity, and a phrase match before a session starts.
- Fixed the turn detector closing turns early by measuring pause length against engine uptime instead of stream time; pauses are now measured against the audio clock, so long mid-sentence pauses no longer cut off the speaker.
- Fixed the assistant claiming to set reminders or alarms when it had no tool for them; spoken sentences that claim to have taken an action without a matching tool call are replaced by a fixed clarification phrase.
- Fixed the assistant speaking at night and during other quiet periods by enforcing configurable quiet hours and checking the system power state before arming.
- Prevented sessions from persisting when no real speech was detected: a turn that the speech guard blocks on the first attempt silently discards the meeting row and returns the assistant to standby.

### Changed
- Replaced the wake loop with a ports-and-adapters voice host that wires ten typed session ports to swappable adapters, isolating the core voice logic from sound libraries, database drivers, and platform calls.
- Changed the debug trigger endpoint (`POST /v1/wake/trigger`) to require a `--debug-trigger` flag at startup, preventing accidental wake events from local processes in production.
- Removed the browser-based speech recognition fallback from the app; the Python engine handles all voice input.
- Requires voxscribe-core 0.5.0. The `wake` extra includes `pywebrtc-audio` for AEC; without it the engine runs with the time-based echo guard only.
- The reply model no longer has a tool path: web search moved from the reply turn to the delegate lane's thinker. The reply model's prompt asks it to emit `[DELEGATE: <request>]` for questions it cannot answer without a search; the marker is never spoken.

## What's New in VoxScribe Server 0.8.0

### New Features
- Added two live turn detection modes, `VOXSCRIBE_LIVE_VAD_MODE=silero` and `heuristic`, which segment speech with core's shared speech activity capability under its segmentation policy; the default stays `adaptive` and `silero` refuses to start with the reason when its model or runtime is missing.
- Added `at_ms`, `p` and `detector` to the live socket's `vad` message, so old clients keep reading `type` and `speech` while new ones get richer timing and probability data.
- Added an opt-in `vad.level` message (`levels=1` on the live URL, at most 10 per second of audio) that carries the detector's latest speech probability for a live level meter.
- Added `speech_activity` to `/v1/health`: the detectors this machine can run, the configured live mode and the spec version.
- Added a live turn detection evaluation script (`scripts/eval_live_vad.py`) that scores every mode on the baseline fixture with fan noise and noise-only clips and writes `results/eval-live-vad.json`.
- Added multi-turn conversations to the voice assistant: after answering, it stays listening for the next question for up to 20 seconds, so a full back-and-forth exchange needs only one wake phrase.
- Added stop-phrase recognition so saying "thank you vox", "bye vox", or similar phrases ends the conversation and returns the assistant to standby without waiting for the idle timeout.
- Added speculative response start that begins generating the reply while speech recognition is still finishing its final pass, reducing first-audio latency from around 4.7 seconds to around 2 seconds on a warm device.
- Added a no-authentication mode for development and testing, letting the engine accept WebSocket and trigger connections without requiring a token.

### UX Improvements
- Improved engine log output to show a structured timestamp, log level, and component label on every line, making it faster to identify which part of the voice pipeline each message came from.

### Bug Fixes
- Fixed voice-session recordings left open by a server restart appearing as interrupted sessions, which prompted a spurious resume dialog on the next app launch.
- Fixed the wake trigger endpoint returning before the meeting identifier was available, causing callers to receive an empty meeting ID.
- Fixed voice reply events not being forwarded to open WebSocket connections, so live session listeners now receive spoken response text as it arrives.

### Changed
- Changed the wake-word model setup so it loads only when the listener first activates, removing openwakeword and onnxruntime as declared install extras.
- Changed the live connection to log one summary per recording (seconds of audio and speech, utterances, recognizer time, mode and detector) at the end of each session.


## What's New in VoxScribe Server 0.7.1

### Changed
- Changed the voxscribe-core dependency to resolve from a tracked wheel file rather than a sibling source directory, so installs are reproducible without the separate repository being checked out.

## What's New in VoxScribe Server 0.7.0

### New Features
- Added multi-turn conversations to the voice assistant: after answering, it stays listening for the next question for up to 20 seconds, so a full back-and-forth exchange needs only one wake phrase.
- Added stop-phrase recognition so saying "thank you vox", "bye vox", or similar phrases ends the conversation and returns the assistant to standby without waiting for the idle timeout.
- Added speculative response start that begins generating the reply while speech recognition is still finishing its final pass, reducing first-audio latency from around 4.7 seconds to around 2 seconds on a warm device.
- Added a no-authentication mode for development and testing, letting the engine accept WebSocket and trigger connections without requiring a token.

### UX Improvements
- Improved engine log output to show a structured timestamp, log level, and component label on every line, making it faster to identify which part of the voice pipeline each message came from.

### Bug Fixes
- Fixed voice-session recordings left open by a server restart appearing as interrupted sessions, which prompted a spurious resume dialog on the next app launch.
- Fixed the wake trigger endpoint returning before the meeting identifier was available, causing callers to receive an empty meeting ID.
- Fixed voice reply events not being forwarded to open WebSocket connections, so live session listeners now receive spoken response text as it arrives.

### Changed
- Changed the wake-word model setup so it loads only when the listener first activates, removing openwakeword and onnxruntime as declared install extras.

## What's New in VoxScribe Server 0.6.1

### Bug Fixes
- Fixed audio frames sent during the final seconds of a recording being silently dropped by adding an end-of-audio handshake that holds the connection open until the engine confirms all frames are queued.
- Fixed per-session ring statistics being invisible in logs by promoting the end-of-session stat line to warning level.

## What's New in VoxScribe Server 0.6.0

### New Features
- Added real transcript import: a pasted text or a transcript file (.txt, .md, .srt, .vtt) becomes a finished scribe with a line for every cue or paragraph, keeping times and speaker names where the file has them.
- Added importing a YouTube link from the video's captions, with the video's title, when connectors are allowed.
- Added reading web pages added as sources: they are now fetched and indexed, and a page that cannot be read ends in an error with the reason.

### Changed
- Changed importing an audio file to answer that it is not available yet, instead of saving made-up text.
- Changed an import that fails to remove the scribe it started, so no empty scribe is left behind.

### Fixed
- Fixed imports of files and YouTube links saving the placeholder text "Imported file content" and "Transcribed YouTube video".
- Fixed web page sources being marked ready without ever being read.
- Fixed ingestion and retrieval in a throwaway test schema reading and writing the shared one.

## What's New in VoxScribe Server 0.5.0

### New Features
- Added separation between people: with an identity service configured, the engine verifies who is calling and keeps each person's data in a schema of their own, so one person can never see or change another's scribes, sources, ideas, projects, tasks, chats or settings.
- Added one atomic edit commit for a scribe (`POST /v1/meetings/{id}/edits`): its title, project, transcript lines, summary and tasks are changed together or not at all, each with the revision the editor saw, and a retry with the same Idempotency-Key is applied once.
- Added a revision to the scribe, the summary and each task so a stale edit is refused instead of overwriting, and the ability to edit a summary by hand, which marks it as edited.
- Added tasks: a to-do list for each scribe with a status (To do, In progress, Done), an owner, a due date and a project, written by hand or drafted by the AI from the transcript and the scribe's project, with a list across scribes that can be filtered by status or project.
- Added the ability to move a scribe into a project, or out of it, from the scribe's own update.
- Added `scripts/adopt_legacy_data.py` to copy data stored before separation into one person's own schema, and a script that puts two sample scribes and their projects into the development database.

### Changed
- Changed the live event stream, the live meeting events, the Google access token, background jobs, source ingestion and the chosen models to belong to one person each, so nothing is shared between people.
- Changed regenerating a summary that a person edited to need an explicit confirmation.
- Changed the delete-all-data call to cover every table, and to act only on the caller's own data.
- Changed the fake embedder to the width the database expects, which fixes ingestion in the built-in test engine.

### Fixed
- Fixed the AI giving no answer on models that reason before replying, by allowing them room to finish.
- Fixed ingestion and retrieval opening their own database connection that ignored any per-person setting.

### Prevented
- Prevented a database call from running with no person in scope when separation is on.
- Prevented tokens signed with another algorithm, another issuer or audience, or a refresh token from being accepted as identity.

## What's New in VoxScribe Server 0.4.0

### New Features
- Added a details report for the engine (version, running time, AI runtime, database connection and schema version) on its own address, so the frequently polled status stays small.

- Added a script that puts two sample scribes and their projects into the development database, and removes them again.

### Changed
- Changed the status report so each model has its own state (ready, on demand, not downloaded, error, starting), and the whole is the worst of them.
- Changed the status report to say "error" instead of "degraded", and to stop sending the engine version, running time, runtime and database.
- Changed the health report to name the kind of database instead of the database itself.

### Fixed
- Fixed the status report not showing an error when the database was down while using the built-in test engine.

### Prevented
- Prevented any database or environment name from leaving the engine in the status, details or health reports.

## What's New in VoxScribe Server 0.3.0

### New Features
- Added the engine's own health to the status report (its version, how long it has been running, and the database), all read from the running engine.
- Added a status report of the on-device AI that lists what is installed and what is in memory right now, so the app can show whether it is ready.
- Added a way to start the on-device AI from the app when it is not running, without ever starting a second copy.

### Bug Fixes
- Fixed the engine reporting a made-up version number instead of its real one.
- Fixed timestamps that apps could not read, which broke the Scribes list.
- Fixed an installed embedding model being shown as not installed.
- Fixed no model being shown as selected for the main and live work, and a chosen model being forgotten after a restart.
- Prevented choosing a new main model from leaving the old one also marked as main.

## What's New in VoxScribe Server 0.2.0

### New Features
- Added the ability to rename a label, delete a chat thread, list the sources attached to a meeting, and open a single project.

### Bug Fixes
- Fixed requests with unusual characters or oversized page sizes crashing the engine instead of being refused politely.
- Fixed unknown addresses and wrong methods answering in a different format from every other error.
- Fixed deleting a project or an idea answering with a body that browsers and proxies are not allowed to expect.
- Fixed idea extraction claiming to run when it could not, and it now says clearly that it is not available yet.
- Prevented blank or oversized names and titles, unknown idea types, ideas pointing at missing projects, and negative note times from being saved.
- Prevented an engine with a broken part from starting and looking healthy.

## What's New in VoxScribe Server 0.1.1

### Bug Fixes
- Fixed every save of notes erasing the enhanced notes, including ones you had edited.
- Fixed meetings that stopped automatically after silence staying in processing forever.
- Fixed meetings left in processing by a restart by finishing them at startup.
- Fixed stopping a meeting twice changing its end time or sending a finished meeting back to processing.
- Prevented two meetings from recording at the same time, and prevented a finished meeting from being started again.
- Prevented a late recognizer update from replacing a line you had edited.
- Prevented live connections from being cut by idle timeouts with a regular heartbeat.

## What's New in VoxScribe Server 0.1.0

### New Features
- Added a local meeting engine that records, transcribes, and stores every meeting on your own computer, making it possible to keep conversations private.
- Added live transcription with provisional lines that settle as you speak, so the transcript keeps up with the conversation.
- Added shorthand notes that are enhanced into clear notes using the live transcript, with a link from each line back to where it was said.
- Added automatic summaries when a meeting ends, and a way to update a summary on demand.
- Added transcript editing for published meetings, with a saved history of every change and protection against two people overwriting each other.
- Added a stored project on each meeting, making it easier to organize work by project.
- Added paging through very long transcripts, making hours-long meetings quick to open.
- Added searching words across all transcripts.
- Added meeting details such as origin, duration, word count, number of lines, and publish time, making each record complete.
- Added import of meetings from text, files, audio, and YouTube links.
- Added sources you can search and ask questions about, with answers that cite the passages they used.
- Added idea extraction with projects, readiness scoring, and PRD generation from your conversations.
- Added Google Drive and Calendar connections that keep their access on your device.
- Added model management with download progress and usage figures showing what ran locally.
- Added export of a meeting as Markdown with its notes, summary, and transcript.

### Bug Fixes
- Fixed transcripts keeping half-finished lines after a meeting was published.
- Prevented a transcript word count from going out of date when an edit and a publish happened at the same time.
- Prevented one person's older edit from silently replacing a newer one.
- Prevented unknown projects from being attached to a meeting.
