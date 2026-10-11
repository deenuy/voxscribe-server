"""SQLite storage, tenant scoped.

Ideas, sessions, jobs, and device claims are stored here. Every idea and
session read goes through a helper that enforces the tenant and workspace
filter, so a caller cannot accidentally read across tenants. This is the lean
Phase 1 equivalent of the ArangoDB tenant rule in the IdeaScribe doc.

The jobs table gives the queue durability: jobs survive a restart. The devices
table enforces one active recording per device while allowing the same user to
record on another device at the same time.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from typing import Any, Optional

from .errors import DeviceBusy

DATA_DIR = os.environ.get("DATA_DIR", "./data")
DB_PATH = os.environ.get("DB_PATH", os.path.join(DATA_DIR, "ideascribe.db"))
RECORDING_TTL = float(os.environ.get("RECORDING_TTL_SECONDS", "3600"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS ideas (
  id TEXT PRIMARY KEY,
  tenant_id TEXT NOT NULL,
  workspace_id TEXT NOT NULL,
  doc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY,
  tenant_id TEXT NOT NULL,
  workspace_id TEXT NOT NULL,
  idea_id TEXT,
  status TEXT NOT NULL,
  doc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
  job_id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  tenant_id TEXT NOT NULL,
  workspace_id TEXT NOT NULL,
  status TEXT NOT NULL,
  attempts INTEGER DEFAULT 0,
  error TEXT,
  payload TEXT NOT NULL,
  updated_at REAL
);
CREATE TABLE IF NOT EXISTS devices (
  device_key TEXT PRIMARY KEY,
  tenant_id TEXT NOT NULL,
  device_id TEXT NOT NULL,
  session_id TEXT,
  status TEXT NOT NULL,
  claimed_at REAL
);
"""


class Store:
    def __init__(self, db_path: str = DB_PATH):
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()
        self._lock = threading.Lock()

    # ---- ideas ----

    def put_idea(self, idea: dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO ideas (id, tenant_id, workspace_id, doc) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET doc=excluded.doc",
                (idea["id"], idea["tenantId"], idea["workspaceId"], json.dumps(idea)),
            )
            self._conn.commit()

    def get_idea(self, tenant: str, workspace: str, idea_id: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT doc FROM ideas WHERE id=? AND tenant_id=? AND workspace_id=?",
            (idea_id, tenant, workspace),
        ).fetchone()
        return json.loads(row["doc"]) if row else None

    def list_ideas(self, tenant: str, workspace: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT doc FROM ideas WHERE tenant_id=? AND workspace_id=?",
            (tenant, workspace),
        ).fetchall()
        return [json.loads(r["doc"]) for r in rows]

    def all_idea_ids(self) -> list[str]:
        rows = self._conn.execute("SELECT id FROM ideas").fetchall()
        return [r["id"] for r in rows]

    # ---- sessions ----

    def put_session(self, sess: dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO sessions (id, tenant_id, workspace_id, idea_id, status, doc) "
                "VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET idea_id=excluded.idea_id, "
                "status=excluded.status, doc=excluded.doc",
                (
                    sess["sessionId"],
                    sess["tenantId"],
                    sess["workspaceId"],
                    sess.get("ideaId"),
                    sess.get("status", "processing"),
                    json.dumps(sess),
                ),
            )
            self._conn.commit()

    def get_session(self, tenant: str, workspace: str, session_id: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT doc FROM sessions WHERE id=? AND tenant_id=? AND workspace_id=?",
            (session_id, tenant, workspace),
        ).fetchone()
        return json.loads(row["doc"]) if row else None

    def list_sessions(self, tenant: str, workspace: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT doc FROM sessions WHERE tenant_id=? AND workspace_id=?",
            (tenant, workspace),
        ).fetchall()
        return [json.loads(r["doc"]) for r in rows]

    def set_session_status(self, session_id: str, status: str) -> None:
        with self._lock:
            row = self._conn.execute(
                "SELECT doc FROM sessions WHERE id=?", (session_id,)
            ).fetchone()
            if not row:
                return
            doc = json.loads(row["doc"])
            doc["status"] = status
            self._conn.execute(
                "UPDATE sessions SET status=?, doc=? WHERE id=?",
                (status, json.dumps(doc), session_id),
            )
            self._conn.commit()

    # ---- jobs ----

    def put_job(self, row: dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO jobs (job_id, session_id, tenant_id, workspace_id, status, "
                "attempts, error, payload, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(job_id) DO UPDATE SET status=excluded.status, "
                "attempts=excluded.attempts, error=excluded.error, "
                "payload=excluded.payload, updated_at=excluded.updated_at",
                (
                    row["job_id"], row["session_id"], row["tenant_id"], row["workspace_id"],
                    row["status"], row["attempts"], row.get("error"), row["payload"], time.time(),
                ),
            )
            self._conn.commit()

    def get_job(self, job_id: str) -> Optional[dict]:
        row = self._conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return dict(row) if row else None

    def update_job_status(self, job_id: str, status: str, attempts: int | None = None,
                          error: str | None = None) -> None:
        with self._lock:
            if attempts is None:
                self._conn.execute(
                    "UPDATE jobs SET status=?, error=?, updated_at=? WHERE job_id=?",
                    (status, error, time.time(), job_id),
                )
            else:
                self._conn.execute(
                    "UPDATE jobs SET status=?, attempts=?, error=?, updated_at=? WHERE job_id=?",
                    (status, attempts, error, time.time(), job_id),
                )
            self._conn.commit()

    def pending_jobs(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM jobs WHERE status IN ('queued', 'processing', 'retrying')"
        ).fetchall()
        return [dict(r) for r in rows]

    def job_status_for_session(self, session_id: str) -> Optional[str]:
        row = self._conn.execute(
            "SELECT status FROM jobs WHERE session_id=?", (session_id,)
        ).fetchone()
        return row["status"] if row else None

    # ---- device recording lock ----

    def claim_device(self, tenant: str, device_id: str, session_id: str) -> None:
        """One active recording per device. Raises DeviceBusy if the device
        already holds a non-stale claim. Stale claims (older than the TTL) are
        reclaimed so a crashed client cannot block a device forever."""
        key = f"{tenant}:{device_id}"
        now = time.time()
        with self._lock:
            row = self._conn.execute(
                "SELECT session_id, status, claimed_at FROM devices WHERE device_key=?",
                (key,),
            ).fetchone()
            if row and row["status"] == "recording" and row["claimed_at"] and \
                    (now - row["claimed_at"]) < RECORDING_TTL:
                raise DeviceBusy(device_id, row["session_id"])
            self._conn.execute(
                "INSERT INTO devices (device_key, tenant_id, device_id, session_id, status, claimed_at) "
                "VALUES (?, ?, ?, ?, 'recording', ?) "
                "ON CONFLICT(device_key) DO UPDATE SET session_id=excluded.session_id, "
                "status='recording', claimed_at=excluded.claimed_at",
                (key, tenant, device_id, session_id, now),
            )
            self._conn.commit()

    def release_device(self, tenant: str, device_id: str) -> None:
        key = f"{tenant}:{device_id}"
        with self._lock:
            self._conn.execute(
                "UPDATE devices SET status='idle', session_id=NULL WHERE device_key=?",
                (key,),
            )
            self._conn.commit()


store = Store()
