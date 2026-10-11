"""In-process job runner for async tasks like extraction and PRD generation."""

import asyncio
import logging
import uuid
from collections.abc import Callable
from typing import Any

from app.local import events, tenancy

log = logging.getLogger(__name__)


class JobRunner:
    """Manages async jobs with event publishing."""

    def __init__(self):
        self.jobs: dict[str, dict] = {}
        self._lock = asyncio.Lock()

    async def submit(
        self,
        kind: str,
        work_fn: Callable[[], Any],
        detail: dict = None,
    ) -> str:
        """Submit a job to run in the background.

        Returns the job ID. The job will publish job.update events with state
        transitions: queued -> running -> done (or error).

        Args:
            kind: Job type (extract_ideas, generate_prd, rescore_idea, transcribe_meeting, etc)
            work_fn: Async callable that does the work
            detail: Optional context data to include in job.update events

        Returns:
            Job ID as a string
        """
        job_id = f"job_{uuid.uuid4().hex[:12]}"

        async with self._lock:
            self.jobs[job_id] = {
                "id": job_id,
                "kind": kind,
                "state": "queued",
                "detail": detail or {},
                "tenant": tenancy.current_tenant(),
            }

        # Publish initial queued event
        events.publish("job.update", {
            "jobId": job_id,
            "kind": kind,
            "state": "queued",
            "detail": detail or {},
        })

        # Schedule the work
        asyncio.create_task(self._run_job(job_id, kind, work_fn, detail))

        return job_id

    async def _run_job(self, job_id: str, kind: str, work_fn: Callable, detail: dict):
        """Execute a job and publish state transitions."""
        try:
            # Publish running event
            async with self._lock:
                if job_id in self.jobs:
                    self.jobs[job_id]["state"] = "running"

            events.publish("job.update", {
                "jobId": job_id,
                "kind": kind,
                "state": "running",
                "detail": detail or {},
            })

            # Run the work
            await work_fn()

            # Publish done event
            async with self._lock:
                if job_id in self.jobs:
                    self.jobs[job_id]["state"] = "done"

            events.publish("job.update", {
                "jobId": job_id,
                "kind": kind,
                "state": "done",
                "detail": detail or {},
            })

        except Exception as e:
            log.exception(f"Job {job_id} failed: {e}")
            async with self._lock:
                if job_id in self.jobs:
                    self.jobs[job_id]["state"] = "error"

            events.publish("job.update", {
                "jobId": job_id,
                "kind": kind,
                "state": "error",
                "detail": {**(detail or {}), "error": str(e)},
            })

    async def get_job(self, job_id: str) -> dict | None:
        """Get job status."""
        async with self._lock:
            job = self.jobs.get(job_id)
        if job is None or job.get("tenant") != tenancy.current_tenant():
            return None  # someone else's job looks exactly like a job that does not exist
        return {k: v for k, v in job.items() if k != "tenant"}


# Global job runner instance
_runner: JobRunner | None = None


def get_runner() -> JobRunner:
    """Get or create the global job runner."""
    global _runner
    if _runner is None:
        _runner = JobRunner()
    return _runner
