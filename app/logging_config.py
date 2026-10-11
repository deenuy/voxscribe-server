"""Structured logging with per-session trace context.

Two formatters ship:
  text  colourised, human-readable for local dev
  json  machine-readable for production log aggregators (Datadog, CloudWatch)

Session context is injected via a ContextVar so every log line produced during
a session carries the same session_id, tenant_id, and device_id. Set it in the
queue worker at job start and clear it in finally so context never leaks between
jobs. The ContextVar copies correctly into asyncio tasks and to_thread calls.

Usage:

    from app.logging_config import get_logger, set_session_context, clear_session_context

    logger = get_logger(__name__)

    token = set_session_context("sess-abc", "tnt-acme", "mac-electron")
    try:
        logger.info("step.started", extra={"step": "transcribing", "pct": 35})
    finally:
        clear_session_context(token)
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
import time
from contextvars import ContextVar, Token
from typing import Optional

# Per-session trace context. Set once per job in the queue worker.
_ctx: ContextVar[dict] = ContextVar("session_ctx", default={})


def set_session_context(
    session_id: str,
    tenant_id: str = "",
    device_id: str = "",
) -> Token:
    return _ctx.set({"session_id": session_id, "tenant_id": tenant_id, "device_id": device_id})


def clear_session_context(token: Token) -> None:
    _ctx.reset(token)


def get_session_context() -> dict:
    return _ctx.get()


class _SessionContextFilter(logging.Filter):
    """Injects session_id / tenant_id / device_id into every LogRecord."""

    def filter(self, record: logging.LogRecord) -> bool:
        ctx = _ctx.get()
        record.session_id = ctx.get("session_id", "-")
        record.tenant_id = ctx.get("tenant_id", "-")
        record.device_id = ctx.get("device_id", "-")
        return True


# ANSI colour helpers for the text formatter.
_RESET = "\033[0m"
_BOLD = "\033[1m"
_LEVEL_COLOURS = {
    "DEBUG": "\033[36m",    # cyan
    "INFO": "\033[32m",     # green
    "WARNING": "\033[33m",  # yellow
    "ERROR": "\033[31m",    # red
    "CRITICAL": "\033[35m", # magenta
}


class _TextFormatter(logging.Formatter):
    """Colourised single-line format for local development.

    Example:
    10:31:05 INFO  [sess-abc][tnt-acme] app.sessions | step=transcribing pct=35
    """

    USE_COLOUR = sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        ts = time.strftime("%H:%M:%S", time.localtime(record.created))
        lvl = record.levelname[:5].ljust(5)
        if self.USE_COLOUR:
            colour = _LEVEL_COLOURS.get(record.levelname, "")
            lvl = f"{colour}{_BOLD}{lvl}{_RESET}"
        sid = getattr(record, "session_id", "-")
        tid = getattr(record, "tenant_id", "-")
        ctx_tag = f"[{sid}][{tid}]" if sid != "-" else ""
        msg = record.getMessage()
        # Append any structured extras passed via extra={}.
        extras = {
            k: v for k, v in record.__dict__.items()
            if k not in logging.LogRecord.__dict__
            and k not in {"session_id", "tenant_id", "device_id", "message",
                          "asctime", "args", "exc_info", "exc_text", "stack_info"}
            and not k.startswith("_")
        }
        if extras:
            msg += " " + " ".join(f"{k}={v}" for k, v in extras.items())
        return f"{ts} {lvl} {ctx_tag} {record.name} | {msg}"


class _JsonFormatter(logging.Formatter):
    """JSON-per-line format for log aggregators."""

    def format(self, record: logging.LogRecord) -> str:
        doc = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "session_id": getattr(record, "session_id", "-"),
            "tenant_id": getattr(record, "tenant_id", "-"),
            "device_id": getattr(record, "device_id", "-"),
            "msg": record.getMessage(),
        }
        # Merge structured extras.
        for k, v in record.__dict__.items():
            if k not in logging.LogRecord.__dict__ \
                    and k not in doc \
                    and k not in {"args", "exc_info", "exc_text", "stack_info",
                                  "message", "asctime"} \
                    and not k.startswith("_"):
                doc[k] = v
        if record.exc_info:
            doc["exc"] = self.formatException(record.exc_info)
        return json.dumps(doc)


def configure(
    level: str = "INFO",
    fmt: str = "text",
    log_file: Optional[str] = None,
    max_bytes: int = 10_485_760,
    backup_count: int = 5,
) -> None:
    """Configure root logger. Call once from the FastAPI lifespan."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove any handlers added by a previous call or by uvicorn.
    for h in root.handlers[:]:
        root.removeHandler(h)

    ctx_filter = _SessionContextFilter()
    formatter: logging.Formatter = _JsonFormatter() if fmt == "json" else _TextFormatter()

    # Stdout handler (always present).
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    sh.addFilter(ctx_filter)
    root.addHandler(sh)

    # Rotating file handler (optional).
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        fh.setFormatter(_JsonFormatter())  # always JSON in files for grep/parsing
        fh.addFilter(ctx_filter)
        root.addHandler(fh)

    # Suppress noisy libs.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
