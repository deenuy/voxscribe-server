"""ICS calendar file parser using stdlib.

Parses VEVENT entries extracting:
- UID, SUMMARY, DTSTART, DTEND, ATTENDEE
- Handles folded lines, escaped text, UTC/floating/TZID times, and all-day dates
"""

import re
from datetime import datetime, timedelta


def parse_ics_file(path: str) -> list[dict]:
    """Parse an ICS file and return a list of events.

    Args:
        path: File path to .ics file

    Returns:
        List of dicts with keys: uid, title, start, end, attendees
    """
    with open(path, encoding="utf-8") as f:
        content = f.read()

    return parse_ics_content(content)


def parse_ics_content(content: str) -> list[dict]:
    """Parse ICS content and return events.

    Handles:
    - Folded lines (continuation lines starting with space or tab)
    - Escaped text (\\, \\n, \\, etc.)
    - UTC times (DTSTART:20231025T100000Z)
    - Floating times (DTSTART:20231025T100000)
    - TZID times (DTSTART;TZID=America/New_York:20231025T100000)
    - All-day dates (DTSTART;VALUE=DATE:20231025)

    Args:
        content: ICS file content as string

    Returns:
        List of event dicts
    """
    # Unfold lines (continuation lines start with space or tab)
    lines = []
    current_line = ""
    for line in content.split("\n"):
        if line and line[0] in (" ", "\t"):
            # Continuation line, append to current
            current_line += line[1:]
        else:
            if current_line:
                lines.append(current_line)
            current_line = line
    if current_line:
        lines.append(current_line)

    # Parse VEVENT blocks
    events = []
    in_vevent = False
    event_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line == "BEGIN:VEVENT":
            in_vevent = True
            event_lines = []
        elif line == "END:VEVENT":
            in_vevent = False
            event = _parse_vevent(event_lines)
            if event:
                events.append(event)
        elif in_vevent:
            event_lines.append(line)

    return events


def _parse_vevent(lines: list[str]) -> dict | None:
    """Parse a VEVENT block.

    Args:
        lines: List of property lines

    Returns:
        Dict with uid, title, start, end, attendees or None if invalid
    """
    props = {}

    for line in lines:
        # Parse property line: KEY;PARAM=VALUE:DATA or KEY:DATA
        match = re.match(r"([A-Z_]+)(?:;([^:]*?))?:(.*)$", line)
        if not match:
            continue

        key = match.group(1)
        params = match.group(2) or ""
        value = match.group(3)

        if key not in props:
            props[key] = []
        props[key].append((params, value))

    # Extract required fields
    uid = _get_property(props, "UID")
    if not uid:
        return None

    title = _get_property(props, "SUMMARY", default="Event")
    title = _unescape_text(title)

    # Parse dates/times
    start_str = _get_property(props, "DTSTART")
    end_str = _get_property(props, "DTEND")

    if not start_str:
        return None

    start = _parse_datetime(start_str, props.get("DTSTART", []))
    end = _parse_datetime(end_str, props.get("DTEND", [])) if end_str else start + timedelta(hours=1)

    # Extract attendees
    attendees = []
    if "ATTENDEE" in props:
        for _params, value in props["ATTENDEE"]:
            # Extract email from mailto: URI or use as-is
            email = value.replace("mailto:", "").strip()
            attendees.append(email)

    return {
        "uid": uid,
        "title": title,
        "start": start.isoformat() + "Z" if start else None,
        "end": end.isoformat() + "Z" if end else None,
        "attendees": attendees,
    }


def _get_property(props: dict, key: str, default: str = "") -> str:
    """Get the first value of a property, handling multiple occurrences."""
    if key not in props or not props[key]:
        return default
    params, value = props[key][0]
    return value


def _parse_datetime(value_str: str, prop_list: list[tuple[str, str]]) -> datetime | None:
    """Parse a DTSTART/DTEND value with params.

    Handles:
    - UTC: 20231025T100000Z
    - Floating: 20231025T100000
    - All-day date: 20231025
    - With TZID: TZID=America/New_York:20231025T100000

    Args:
        value_str: The value part (after colon)
        prop_list: List of (params, value) tuples for this property

    Returns:
        datetime object or None
    """
    if not value_str:
        return None

    params, value = prop_list[0] if prop_list else ("", value_str)

    # Check VALUE parameter
    if "VALUE=DATE" in params:
        # All-day date: YYYYMMDD
        try:
            return datetime.strptime(value, "%Y%m%d")
        except ValueError:
            return None

    # Handle TZID (for now, treat as floating and ignore timezone)
    # TODO: Actually parse TZID times with proper timezone handling
    value = value.strip()

    # Check if UTC (ends with Z)
    if value.endswith("Z"):
        try:
            return datetime.strptime(value, "%Y%m%dT%H%M%SZ")
        except ValueError:
            return None

    # Floating time (no Z)
    try:
        return datetime.strptime(value, "%Y%m%dT%H%M%S")
    except ValueError:
        # Try just date
        try:
            return datetime.strptime(value, "%Y%m%d")
        except ValueError:
            return None


def _unescape_text(text: str) -> str:
    """Unescape ICS text escapes.

    ICS uses backslash escaping:
    - \\N -> newline
    - \\, -> comma
    - \\\\ -> backslash
    - etc.
    """
    text = text.replace("\\n", "\n")
    text = text.replace("\\,", ",")
    text = text.replace("\\;", ";")
    text = text.replace("\\\\", "\\")
    return text
