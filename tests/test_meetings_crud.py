"""Tests for meetings CRUD endpoints."""

import uuid

import pytest

from app.local.db import create_pool
from app.local.store_meetings import MeetingStore
from app.local.models import Block, Segment, Citation
from app.local.errors import NotFoundError


@pytest.mark.asyncio
async def test_create_meeting(test_db_url):
    """Test creating a meeting."""
    db_url, schema = test_db_url
    pool = await create_pool(db_url, schema=schema)
    store = MeetingStore(pool)
    
    meeting = await store.create_meeting(
        meeting_id=str(uuid.uuid4()),
        title="Test Meeting",
        context="Test context",
    )
    
    assert meeting.title == "Test Meeting"
    assert meeting.context == "Test context"
    assert meeting.status == "scheduled"
    assert meeting.createdAt is not None
    assert meeting.updatedAt is not None


@pytest.mark.asyncio
async def test_create_meeting_default_title(test_db_url):
    """Test creating a meeting with default title."""
    db_url, schema = test_db_url
    pool = await create_pool(db_url, schema=schema)
    store = MeetingStore(pool)
    
    meeting = await store.create_meeting(meeting_id=str(uuid.uuid4()))
    assert meeting.title == "New meeting"


@pytest.mark.asyncio
async def test_get_meeting(test_db_url):
    """Test getting a meeting by ID."""
    db_url, schema = test_db_url
    pool = await create_pool(db_url, schema=schema)
    store = MeetingStore(pool)
    
    meeting_id = str(uuid.uuid4())
    created = await store.create_meeting(meeting_id=meeting_id, title="Test")
    
    retrieved = await store.get_meeting(meeting_id)
    assert retrieved.id == created.id
    assert retrieved.title == created.title


@pytest.mark.asyncio
async def test_get_nonexistent_meeting(test_db_url):
    """Test getting a nonexistent meeting raises NotFoundError."""
    db_url, schema = test_db_url
    pool = await create_pool(db_url, schema=schema)
    store = MeetingStore(pool)
    
    with pytest.raises(NotFoundError):
        await store.get_meeting("nonexistent")


@pytest.mark.asyncio
async def test_list_meetings(test_db_url):
    """Test listing meetings."""
    db_url, schema = test_db_url
    pool = await create_pool(db_url, schema=schema)
    store = MeetingStore(pool)
    
    # Create several meetings
    for i in range(3):
        await store.create_meeting(
            meeting_id=str(uuid.uuid4()),
            title=f"Meeting {i}",
        )
    
    meetings, cursor = await store.list_meetings(limit=10)
    assert len(meetings) >= 3


@pytest.mark.asyncio
async def test_list_meetings_with_query(test_db_url):
    """Test listing meetings with query filter."""
    db_url, schema = test_db_url
    pool = await create_pool(db_url, schema=schema)
    store = MeetingStore(pool)
    
    await store.create_meeting(
        meeting_id=str(uuid.uuid4()),
        title="Team Standup",
    )
    await store.create_meeting(
        meeting_id=str(uuid.uuid4()),
        title="Planning Session",
    )
    
    meetings, _ = await store.list_meetings(query="team", limit=10)
    assert any("Team" in m.title for m in meetings)


@pytest.mark.asyncio
async def test_update_meeting(test_db_url):
    """Test updating a meeting."""
    db_url, schema = test_db_url
    pool = await create_pool(db_url, schema=schema)
    store = MeetingStore(pool)
    
    meeting_id = str(uuid.uuid4())
    await store.create_meeting(meeting_id=meeting_id, title="Original")
    
    updated = await store.update_meeting(
        meeting_id,
        title="Updated",
        enhance_style="concise",
    )
    
    assert updated.title == "Updated"
    assert updated.enhanceStyle == "concise"


@pytest.mark.asyncio
async def test_start_meeting(test_db_url):
    """Test starting a meeting."""
    db_url, schema = test_db_url
    pool = await create_pool(db_url, schema=schema)
    store = MeetingStore(pool)
    
    meeting_id = str(uuid.uuid4())
    await store.create_meeting(meeting_id=meeting_id)
    
    started = await store.start_meeting(meeting_id)
    assert started.status == "recording"
    assert started.startedAt is not None


@pytest.mark.asyncio
async def test_stop_meeting(test_db_url):
    """Test stopping a meeting."""
    db_url, schema = test_db_url
    pool = await create_pool(db_url, schema=schema)
    store = MeetingStore(pool)
    
    meeting_id = str(uuid.uuid4())
    await store.create_meeting(meeting_id=meeting_id)
    await store.start_meeting(meeting_id)
    
    stopped = await store.stop_meeting(meeting_id)
    assert stopped.status == "processing"
    assert stopped.endedAt is not None


@pytest.mark.asyncio
async def test_delete_meeting(test_db_url):
    """Test deleting a meeting."""
    db_url, schema = test_db_url
    pool = await create_pool(db_url, schema=schema)
    store = MeetingStore(pool)
    
    meeting_id = str(uuid.uuid4())
    await store.create_meeting(meeting_id=meeting_id)
    
    await store.delete_meeting(meeting_id)
    
    with pytest.raises(NotFoundError):
        await store.get_meeting(meeting_id)


@pytest.mark.asyncio
async def test_put_notes(test_db_url):
    """Test replacing notes for a meeting."""
    db_url, schema = test_db_url
    pool = await create_pool(db_url, schema=schema)
    store = MeetingStore(pool)
    
    meeting_id = str(uuid.uuid4())
    await store.create_meeting(meeting_id=meeting_id)
    
    blocks = [
        Block(blockId=str(uuid.uuid4()), text="First note", offsetMs=0),
        Block(blockId=str(uuid.uuid4()), text="Second note", offsetMs=1000),
    ]
    
    count = await store.put_notes(meeting_id, blocks)
    assert count == 2
    
    notes = await store.get_notes(meeting_id)
    assert len(notes) == 2
    assert notes[0].text == "First note"


@pytest.mark.asyncio
async def test_transcript_segments(test_db_url):
    """Test adding and retrieving transcript segments."""
    db_url, schema = test_db_url
    pool = await create_pool(db_url, schema=schema)
    store = MeetingStore(pool)
    
    meeting_id = str(uuid.uuid4())
    await store.create_meeting(meeting_id=meeting_id)
    
    segment = Segment(
        id=str(uuid.uuid4()),
        startMs=0,
        endMs=1000,
        source="mic",
        text="Hello world",
        state="stable",
    )
    
    await store.append_segment(meeting_id, segment)
    
    segments = await store.get_transcript(meeting_id)
    assert len(segments) == 1
    assert segments[0].text == "Hello world"


@pytest.mark.asyncio
async def test_attachments(test_db_url):
    """Test attaching and detaching sources."""
    db_url, schema = test_db_url
    pool = await create_pool(db_url, schema=schema)
    store = MeetingStore(pool)

    # Create a source first (required for FK constraint)
    source_id = str(uuid.uuid4())
    async with pool.connection() as conn:
        await conn.execute(
            """INSERT INTO sources (id, kind, uri, name, status)
            VALUES (%s, %s, %s, %s, %s)""",
            (source_id, "file", "/path/to/file", "Test Source", "ready"),
        )
        await conn.commit()

    meeting_id = str(uuid.uuid4())
    await store.create_meeting(meeting_id=meeting_id)

    await store.attach_source(meeting_id, source_id)

    attachments = await store.get_attachments(meeting_id)
    assert source_id in attachments

    await store.detach_source(meeting_id, source_id)
    attachments = await store.get_attachments(meeting_id)
    assert source_id not in attachments


@pytest.mark.asyncio
async def test_enhanced_blocks(test_db_url):
    """Test creating and retrieving enhanced blocks."""
    db_url, schema = test_db_url
    pool = await create_pool(db_url, schema=schema)
    store = MeetingStore(pool)
    
    meeting_id = str(uuid.uuid4())
    await store.create_meeting(meeting_id=meeting_id)
    
    block_id = str(uuid.uuid4())
    block = Block(blockId=block_id, text="Test note", offsetMs=0)
    await store.put_notes(meeting_id, [block])
    
    # Create enhanced block
    citations = [
        Citation(kind="transcript", segmentId="seg_1", startMs=0, endMs=100)
    ]
    await store.set_enhanced_block(
        meeting_id,
        block_id,
        "Expanded test note",
        citations,
        status="stable",
    )
    
    enhanced = await store.get_enhanced(meeting_id)
    assert len(enhanced["blocks"]) == 1
    assert enhanced["blocks"][0].text == "Expanded test note"
    assert len(enhanced["blocks"][0].citations) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
