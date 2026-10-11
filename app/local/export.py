"""Export meetings to markdown format."""

from .store_meetings import MeetingStore


async def export_meeting_markdown(store: MeetingStore, meeting_id: str) -> str:
    """Export a meeting to markdown format."""
    meeting = await store.get_meeting(meeting_id)
    notes = await store.get_notes(meeting_id)
    enhanced = await store.get_enhanced(meeting_id)
    summary = await store.get_summary(meeting_id)
    segments = await store.get_transcript(meeting_id)

    # Title and date
    lines = [f"# {meeting.title}"]
    if meeting.startedAt:
        date_str = meeting.startedAt.split("T")[0]
        lines.append(f"*{date_str}*")
    lines.append("")

    # Summary section
    if summary["status"] == "ready":
        lines.append("## Summary")
        if summary["overview"]:
            lines.append(summary["overview"])
        lines.append("")
        if summary["details"]:
            for detail in summary["details"]:
                lines.append(f"### {detail.get('title', 'Detail')}")
                lines.append(detail.get("body", ""))
                lines.append("")

    # Shorthand notes
    if notes:
        lines.append("## Notes")
        for note in notes:
            lines.append(f"- {note.text}")
        lines.append("")

    # Enhanced notes (with citations)
    if enhanced["blocks"]:
        lines.append("## Expanded Notes")
        for block in enhanced["blocks"]:
            lines.append(f"### {block.text[:50]}...")
            lines.append(block.text)
            if block.citations:
                lines.append("*Citations:*")
                for cite in block.citations:
                    if cite.kind == "transcript":
                        lines.append(f"  - Transcript [{cite.segmentId}]")
                    elif cite.kind == "source":
                        lines.append(f"  - Source: {cite.title} ({cite.locator})")
            lines.append("")

    # Full transcript
    if segments:
        lines.append("## Full Transcript")
        for segment in segments:
            speaker = f"{segment.speaker}: " if segment.speaker else ""
            lines.append(f"{speaker}{segment.text}")
        lines.append("")

    return "\n".join(lines)
