"""Tests for sources endpoints."""

import pytest
import tempfile
from pathlib import Path

from app.local.store import Store


@pytest.mark.asyncio
async def test_create_and_list_sources(test_db_url):
    """Test creating and listing sources."""
    db_url, schema = test_db_url
    store = await Store.create(db_url, schema=schema)

    # Create a file source
    source = await store.create_source(
        source_id="test-file-1",
        kind="file",
        uri="/tmp/test.txt",
        name="test.txt",
    )

    assert source.id == "test-file-1"
    assert source.kind == "file"
    assert source.uri == "/tmp/test.txt"
    assert source.status == "queued"

    # List sources
    sources, cursor, total, counts = await store.list_sources()
    assert len(sources) == 1
    assert total == 1
    assert "file" in counts

    # Get source
    fetched = await store.get_source("test-file-1")
    assert fetched.id == "test-file-1"

    await store.close()


@pytest.mark.asyncio
async def test_labels(test_db_url):
    """Test creating and managing labels."""
    db_url, schema = test_db_url
    store = await Store.create(db_url, schema=schema)

    # Create a label
    label = await store.create_label(label_id="label-1", name="Important", color="#FF0000")
    assert label.name == "Important"

    # List labels
    labels = await store.list_labels()
    assert len(labels) == 1
    assert labels[0].name == "Important"

    # Delete label
    await store.delete_label("label-1")
    labels = await store.list_labels()
    assert len(labels) == 0

    await store.close()


@pytest.mark.asyncio
async def test_source_labels_association(test_db_url):
    """Test associating labels with sources."""
    db_url, schema = test_db_url
    store = await Store.create(db_url, schema=schema)

    # Create label and source
    label = await store.create_label(label_id="label-1", name="Important")
    source = await store.create_source(
        source_id="source-1",
        kind="file",
        uri="/tmp/test.txt",
        name="test.txt",
    )

    # Associate label with source (need to use the source store module)
    from app.local.store_sources import SourceStore
    source_store = SourceStore(store.pool)
    await source_store.update_source_labels("source-1", ["label-1"])

    # Fetch and verify
    updated = await store.get_source("source-1")
    assert len(updated.labels) == 1
    assert updated.labels[0].id == "label-1"

    await store.close()


@pytest.mark.asyncio
async def test_chat_threads(test_db_url):
    """Test creating and listing chat threads."""
    from app.local.store_chat import ChatStore
    from app.local.models import ChatScope

    db_url, schema = test_db_url
    store = await Store.create(db_url, schema=schema)
    chat_store = ChatStore(store.pool)

    # Create thread
    thread = await chat_store.create_thread(
        thread_id="thread-1",
        label_ids=["label-1"],
        meeting_id=None,
        source_ids=["source-1"],
    )

    assert thread.id == "thread-1"
    assert thread.scope.labelIds == ["label-1"]

    # List threads
    threads = await chat_store.list_threads()
    assert len(threads) == 1

    # Get thread
    fetched = await chat_store.get_thread("thread-1")
    assert fetched.id == "thread-1"

    await store.close()
