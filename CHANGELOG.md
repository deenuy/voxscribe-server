# Changelog

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
