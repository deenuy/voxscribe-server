# Changelog

## Unreleased

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
