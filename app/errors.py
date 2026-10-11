"""Typed errors with HTTP status codes.

Routers raise these; main.py registers handlers that turn them into clean JSON
responses. Keeps error handling consistent and production grade instead of
scattering raw HTTPExceptions.
"""

from __future__ import annotations


class AppError(Exception):
    status_code = 400
    code = "bad_request"

    def __init__(self, detail: str):
        super().__init__(detail)
        self.detail = detail


class NotFoundError(AppError):
    status_code = 404
    code = "not_found"


class DeviceBusy(AppError):
    status_code = 409
    code = "device_busy"

    def __init__(self, device_id: str, active_session: str):
        super().__init__(
            f"device {device_id} already has an active recording ({active_session}). "
            f"Finish or cancel it before starting another on this device."
        )
        self.device_id = device_id
        self.active_session = active_session


class ValidationFailed(AppError):
    status_code = 422
    code = "validation_failed"


class QueueUnavailable(AppError):
    status_code = 503
    code = "queue_unavailable"
