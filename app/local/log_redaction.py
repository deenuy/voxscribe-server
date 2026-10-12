"""Keeps credentials out of log lines.

A browser cannot set headers on a WebSocket, so the live socket carries the engine token and the account's
identity token in its address, and the server's connection log line prints that address. This filter rewrites
`token=` and `identity=` values (and similar query parameters) to `[redacted]` before any handler sees them.
"""

import logging
import re

_SECRET_PARAM = re.compile(r"\b((?:token|identity|access_token|id_token)=)[^&\s\"']+", re.IGNORECASE)


def redact(text: str) -> str:
    """The text with secret query parameter values replaced."""
    return _SECRET_PARAM.sub(r"\1[redacted]", text)


class RedactSecretsFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True
        cleaned = redact(message)
        if cleaned != message:
            record.msg = cleaned
            record.args = ()
        return True


def install(*logger_names: str) -> None:
    """Attach the filter to the named loggers (records are filtered where they are created)."""
    for name in logger_names or ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        if not any(isinstance(f, RedactSecretsFilter) for f in logger.filters):
            logger.addFilter(RedactSecretsFilter())
