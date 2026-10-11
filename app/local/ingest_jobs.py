"""Background ingestion worker for sources.

Processes queued sources asynchronously, updates status, publishes events.
"""

import asyncio
import logging
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from voxscribe.knowledge import ingest_path, scan_folder

from app.local import events, tenancy
from app.local.pg_sync_index import PostgresKnowledgeIndexSync, SourceBoundIndex
from app.local.store_sources import SourceStore

log = logging.getLogger(__name__)


class IngestionWorker:
    """Background worker for ingesting sources."""

    def __init__(self, store_pool, database_url: str, embedder, num_workers: int = 2):
        self.store_pool = store_pool
        self.database_url = database_url
        self.knowledge_index = PostgresKnowledgeIndexSync(database_url, getattr(store_pool, "base_schema", None))
        self.embedder = embedder
        self.num_workers = num_workers
        self.queue: asyncio.Queue = asyncio.Queue()
        self.running = False
        self.executor = ThreadPoolExecutor(max_workers=num_workers)

    async def start(self):
        """Start the worker pool."""
        self.running = True
        # Start a single monitor task that processes the queue
        self.monitor_task = asyncio.create_task(self._monitor_queue())

    async def stop(self):
        """Stop the worker pool."""
        self.running = False
        await self.queue.join()
        self.monitor_task.cancel()
        self.executor.shutdown(wait=True)

    async def queue_source(self, source_id: str):
        """Queue a source for ingestion. The queue is shared by everyone, so each item remembers whose it is."""
        await self.queue.put((tenancy.current(), source_id))

    async def _monitor_queue(self):
        """Monitor queue and dispatch tasks to executor."""
        while self.running:
            try:
                scope, source_id = await asyncio.wait_for(self.queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            try:
                # This task was started at startup, with no tenant. Work on the item inside its owner's scope.
                if scope is None:
                    await self._ingest_source(source_id)
                else:
                    with tenancy.tenant_scope(scope):
                        await self._ingest_source(source_id)
            finally:
                self.queue.task_done()

    async def _ingest_source(self, source_id: str):
        """Ingest a single source."""
        store = SourceStore(self.store_pool)

        try:
            source = await store.get_source(source_id)
            if not source:
                return

            # Mark as indexing
            await store.update_source_status(source_id, "indexing", error=None)
            source = await store.get_source(source_id)
            events.publish("source.update", source.model_dump(exclude_none=True))

            # Ingest the source in thread pool
            if source.kind == "file":
                path = Path(source.uri)
                _source_id, chunk_count = await asyncio.to_thread(
                    ingest_path,
                    SourceBoundIndex(self.knowledge_index, source_id),
                    self.embedder,
                    path,
                )
                await store.update_source_chunk_count(source_id, chunk_count)

            elif source.kind == "url":
                # A web page: fetched (with the same safety checks as everywhere: no private addresses, size and type
                # limits), then indexed like a file. Before this a web page was accepted, marked ready and never read.
                from app.local.url_fetcher import fetch_url

                page_text = await fetch_url(source.uri)
                if not page_text or not page_text.strip():
                    raise ValueError("the page has no text to index")
                with tempfile.TemporaryDirectory() as tmp:
                    page = Path(tmp) / "page.txt"
                    page.write_text(page_text, encoding="utf-8")
                    _source_id, chunk_count = await asyncio.to_thread(
                        ingest_path,
                        SourceBoundIndex(self.knowledge_index, source_id),
                        self.embedder,
                        page,
                    )
                await store.update_source_chunk_count(source_id, chunk_count)

            elif source.kind == "folder":
                path = Path(source.uri)
                files = scan_folder(path)
                total_chunks = 0

                for file_path in files:
                    child_id = await store.find_child_id(source_id, str(file_path))
                    if child_id is None:
                        child_id = str(uuid.uuid4())
                        await store.create_source(
                            source_id=child_id,
                            kind="file",
                            uri=str(file_path),
                            name=file_path.name,
                            mime=None,
                            parent_id=source_id,
                        )
                    try:
                        await store.update_source_status(child_id, "indexing")
                        _unused, chunk_count = await asyncio.to_thread(
                            ingest_path,
                            SourceBoundIndex(self.knowledge_index, child_id),
                            self.embedder,
                            file_path,
                        )
                        await store.update_source_chunk_count(child_id, chunk_count)
                        await store.update_source_status(child_id, "ready")
                        total_chunks += chunk_count
                    except Exception as e:
                        log.error(f"Failed to ingest {file_path}: {e}", exc_info=True)
                        await store.update_source_status(child_id, "error", error=str(e))
                    child_source = await store.get_source(child_id)
                    events.publish("source.update", child_source.model_dump(exclude_none=True))

                await store.update_source_chunk_count(source_id, total_chunks)

            # Mark as ready
            await store.update_source_status(source_id, "ready", error=None)
            source = await store.get_source(source_id)
            events.publish("source.update", source.model_dump(exclude_none=True))

        except Exception as e:
            error_msg = str(e)
            await store.update_source_status(source_id, "error", error=error_msg)
            source = await store.get_source(source_id)
            events.publish("source.update", source.model_dump(exclude_none=True))
