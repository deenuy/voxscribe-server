"""Job queue with logging of every state transition.

The JobQueue protocol is the seam for a Kafka or Temporal adapter. Day one
uses InProcessJobQueue: asyncio worker pool, SQLite durability, exponential
backoff retry, dead-letter on exhaustion.
"""

from __future__ import annotations

import asyncio
import os
from typing import Awaitable, Callable, Protocol

from .jobs import Job
from .logging_config import (
    clear_session_context,
    get_logger,
    set_session_context,
)
from .sessions import hub
from .store import store

log = get_logger(__name__)

Handler = Callable[[Job], Awaitable[None]]

QUEUE_BACKEND = os.environ.get("QUEUE_BACKEND", "memory")
CONCURRENCY = int(os.environ.get("WORKER_CONCURRENCY", "2"))
MAX_ATTEMPTS = int(os.environ.get("QUEUE_MAX_ATTEMPTS", "3"))
BASE_BACKOFF = float(os.environ.get("QUEUE_BACKOFF", "2.0"))


class JobQueue(Protocol):
    async def enqueue(self, job: Job) -> None: ...
    async def start(self) -> None: ...
    async def stop(self) -> None: ...


class InProcessJobQueue:
    def __init__(
        self,
        handler: Handler,
        concurrency: int = CONCURRENCY,
        max_attempts: int = MAX_ATTEMPTS,
        base_backoff: float = BASE_BACKOFF,
    ):
        self._handler = handler
        self._concurrency = concurrency
        self._max_attempts = max_attempts
        self._base_backoff = base_backoff
        self._q: asyncio.Queue[Job] = asyncio.Queue()
        self._workers: list[asyncio.Task] = []
        self._retries: set[asyncio.Task] = set()
        self._stopping = False

    async def start(self) -> None:
        recovered = store.pending_jobs()
        if recovered:
            log.info("queue.recovery", extra={"count": len(recovered)})
        for row in recovered:
            job = Job.from_row(row)
            store.update_job_status(job.job_id, "queued")
            await self._q.put(job)
        self._workers = [
            asyncio.create_task(self._run(i)) for i in range(self._concurrency)
        ]
        log.info("queue.started", extra={"workers": self._concurrency, "backend": QUEUE_BACKEND})

    async def enqueue(self, job: Job) -> None:
        existing = store.get_job(job.job_id)
        if existing and existing["status"] in ("complete", "processing", "queued"):
            log.debug("queue.duplicate_skipped", extra={"job_id": job.job_id, "status": existing["status"]})
            return
        store.put_job(job.to_row("queued"))
        await self._q.put(job)
        log.info(
            "queue.enqueued",
            extra={"job_id": job.job_id, "session_id": job.session_id,
                   "queue_depth": self._q.qsize()},
        )

    async def _run(self, idx: int) -> None:
        log.debug("worker.ready", extra={"worker": idx})
        while not self._stopping:
            try:
                job = await self._q.get()
            except asyncio.CancelledError:
                return
            token = set_session_context(job.session_id, job.tenant_id)
            try:
                log.info("worker.picked_up", extra={"worker": idx, "job_id": job.job_id, "attempt": job.attempts + 1})
                store.update_job_status(job.job_id, "processing", attempts=job.attempts)
                await self._handler(job)
                store.update_job_status(job.job_id, "complete")
                log.info("worker.complete", extra={"worker": idx, "job_id": job.job_id})
            except asyncio.CancelledError:
                store.update_job_status(job.job_id, "queued", attempts=job.attempts)
                clear_session_context(token)
                return
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "worker.failure",
                    extra={"worker": idx, "job_id": job.job_id, "attempt": job.attempts + 1, "error": str(exc)},
                    exc_info=True,
                )
                self._handle_failure(job, exc)
            finally:
                self._q.task_done()
                clear_session_context(token)

    def _handle_failure(self, job: Job, exc: Exception) -> None:
        job.attempts += 1
        if job.attempts < self._max_attempts:
            delay = self._base_backoff * (2 ** (job.attempts - 1))
            store.update_job_status(job.job_id, "retrying", attempts=job.attempts, error=str(exc))
            hub.emit(
                job.session_id,
                {"step": f"transient error, retry {job.attempts}/{self._max_attempts - 1} in {delay:.0f}s",
                 "pct": 10, "done": False, "error": str(exc)},
            )
            log.warning(
                "queue.retry_scheduled",
                extra={"job_id": job.job_id, "attempt": job.attempts,
                       "delay_s": delay, "max": self._max_attempts},
            )
            task = asyncio.create_task(self._requeue_after(job, delay))
            self._retries.add(task)
            task.add_done_callback(self._retries.discard)
        else:
            store.update_job_status(job.job_id, "dead", attempts=job.attempts, error=str(exc))
            store.set_session_status(job.session_id, "failed")
            hub.emit(
                job.session_id,
                {"step": f"failed after {job.attempts} attempts: {exc}",
                 "pct": 100, "done": True, "error": str(exc)},
            )
            log.error(
                "queue.dead_lettered",
                extra={"job_id": job.job_id, "attempts": job.attempts, "error": str(exc)},
            )

    async def _requeue_after(self, job: Job, delay: float) -> None:
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            return
        if not self._stopping:
            store.update_job_status(job.job_id, "queued", attempts=job.attempts)
            await self._q.put(job)
            log.info("queue.retried", extra={"job_id": job.job_id, "attempt": job.attempts})

    async def stop(self) -> None:
        log.info("queue.stopping", extra={"pending": self._q.qsize()})
        self._stopping = True
        for w in self._workers:
            w.cancel()
        for r in list(self._retries):
            r.cancel()
        await asyncio.gather(*self._workers, *self._retries, return_exceptions=True)
        log.info("queue.stopped")


def make_queue(handler: Handler) -> JobQueue:
    if QUEUE_BACKEND == "memory":
        return InProcessJobQueue(handler)
    if QUEUE_BACKEND in ("kafka", "temporal"):
        raise NotImplementedError(
            f"QUEUE_BACKEND={QUEUE_BACKEND} is a future adapter. Implement the "
            f"JobQueue protocol and return it here. Routers and the handler are unchanged."
        )
    raise ValueError(f"unknown QUEUE_BACKEND: {QUEUE_BACKEND}")
