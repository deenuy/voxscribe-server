"""Tests for calendar endpoints."""

import pytest
from pathlib import Path
import tempfile

from app.local import ics


@pytest.mark.asyncio
async def test_calendar_ics_parse():
    """Test ICS parsing with various formats."""
    ics_content = """BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//Test//Test//EN
BEGIN:VEVENT
UID:event1
SUMMARY:Meeting
DTSTART:20231025T100000Z
DTEND:20231025T110000Z
ATTENDEE:user1@example.com
ATTENDEE:user2@example.com
END:VEVENT
BEGIN:VEVENT
UID:event2
SUMMARY:All Day Event
DTSTART;VALUE=DATE:20231026
DTEND;VALUE=DATE:20231027
END:VEVENT
END:VCALENDAR"""

    events = ics.parse_ics_content(ics_content)
    assert len(events) == 2

    # Check first event
    e1 = events[0]
    assert e1["uid"] == "event1"
    assert e1["title"] == "Meeting"
    assert len(e1["attendees"]) == 2
    assert "user1@example.com" in e1["attendees"]

    # Check all-day event
    e2 = events[1]
    assert e2["uid"] == "event2"
    assert e2["title"] == "All Day Event"


@pytest.mark.asyncio
async def test_calendar_import_ics(client):
    """Test POST /calendar/import with ICS file."""

    # Create a temp ICS file
    ics_content = """BEGIN:VCALENDAR
VERSION:2.0
BEGIN:VEVENT
UID:test-event-1
SUMMARY:Test Meeting
DTSTART:20231025T100000Z
DTEND:20231025T110000Z
END:VEVENT
END:VCALENDAR"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".ics", delete=False) as f:
        f.write(ics_content)
        temp_path = f.name

    try:
        resp = await client.post(
            "/v1/calendar/import",
            json={"kind": "ics", "path": temp_path},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["imported"] == 1
        assert "lastImportedAt" in data
    finally:
        Path(temp_path).unlink()


@pytest.mark.asyncio
async def test_calendar_import_missing_file(client):
    """Test importing from missing file fails."""
    resp = await client.post(
        "/v1/calendar/import",
        json={"kind": "ics", "path": "/nonexistent/file.ics"},
    )
    assert resp.status_code == 400
    assert resp.json()["error"] == "invalid_request"


@pytest.mark.asyncio
async def test_calendar_import_no_path_for_ics(client):
    """Test that ICS import requires path."""
    resp = await client.post(
        "/v1/calendar/import",
        json={"kind": "ics"},
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_calendar_status(client):
    """Test GET /calendar/status."""
    resp = await client.get("/v1/calendar/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "lastImportedAt" in data
    assert "connectedAccounts" in data
    assert isinstance(data["connectedAccounts"], list)


@pytest.mark.asyncio
async def test_calendar_get_events(client):
    """Test GET /calendar/events."""
    resp = await client.get("/v1/calendar/events")
    assert resp.status_code == 200
    data = resp.json()
    assert "items" in data
    assert isinstance(data["items"], list)
