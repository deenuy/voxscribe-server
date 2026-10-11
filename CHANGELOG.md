# Changelog

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
