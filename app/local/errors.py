"""Error types for the engine contract."""



class EngineError(Exception):
    """Base error for the engine API."""

    def __init__(self, code: str, detail: str, status: int, extra: dict | None = None):
        self.code = code
        self.detail = detail
        self.status = status
        self.extra = extra or {}  # more fields for the response body, e.g. which items conflicted
        super().__init__(f"{code}: {detail}")


class NotFoundError(EngineError):
    def __init__(self, detail: str = "not found"):
        super().__init__("not_found", detail, 404)


class InvalidRequestError(EngineError):
    def __init__(self, detail: str = "invalid request"):
        super().__init__("invalid_request", detail, 400)


class ConflictError(EngineError):
    def __init__(self, detail: str = "conflict"):
        super().__init__("conflict", detail, 409)


class ProviderUnavailableError(EngineError):
    def __init__(self, detail: str = "provider unavailable"):
        super().__init__("provider_unavailable", detail, 503)


class ConsentRequiredError(EngineError):
    def __init__(self, detail: str = "consent required"):
        super().__init__("consent_required", detail, 400)


class NetworkDisabledError(EngineError):
    def __init__(self, detail: str = "network disabled"):
        super().__init__("network_disabled", detail, 403)


class DatabaseUnavailableError(EngineError):
    def __init__(self, detail: str = "database unavailable"):
        super().__init__("database_unavailable", detail, 503)


class ConnectorNotConnectedError(EngineError):
    def __init__(self, detail: str = "connector not connected"):
        super().__init__("connector_not_connected", detail, 409)


class ConnectorTokenExpiredError(EngineError):
    def __init__(self, detail: str = "connector token expired"):
        super().__init__("connector_token_expired", detail, 409)
