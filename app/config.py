"""Settings loader.

Reads config/config.yaml for base configuration, then applies environment
variable overrides. After loading, canonical values are pushed back into
os.environ so every module that reads os.environ directly still works without
changes. Explicit environment variables always win over the YAML file.

Import pattern (call load() once at startup from main.py, then read anywhere):

    from app.config import settings
    print(settings.queue.worker_concurrency)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

try:
    import yaml  # type: ignore
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

CONFIG_PATH = os.environ.get("CONFIG_PATH", "config/config.yaml")


@dataclass
class ServerSettings:
    host: str = "0.0.0.0"
    port: int = 8000


@dataclass
class QueueSettings:
    backend: str = "memory"
    worker_concurrency: int = 2
    max_attempts: int = 3
    base_backoff: float = 2.0
    recording_ttl_seconds: int = 3600


@dataclass
class ProcessingSettings:
    extractor: str = "simulated"
    asr_backend: str = "stub"
    cleanup: bool = False


@dataclass
class OllamaSettings:
    host: str = "http://host.docker.internal:11434"
    model: str = "gemma3:4b"


@dataclass
class ClaudeSettings:
    model: str = "claude-sonnet-4-6"
    api_key: str = ""


@dataclass
class OpenAISettings:
    api_key: str = ""
    whisper_model: str = "base"


@dataclass
class ModelsSettings:
    ollama: OllamaSettings = field(default_factory=OllamaSettings)
    claude: ClaudeSettings = field(default_factory=ClaudeSettings)
    openai: OpenAISettings = field(default_factory=OpenAISettings)


@dataclass
class StorageSettings:
    data_dir: str = "./data"
    db_path: str = "./data/ideascribe.db"
    id_floor: int = 53


@dataclass
class AuthSettings:
    default_tenant: str = "tnt-acme"
    default_workspace: str = "ws-discovery"


@dataclass
class LoggingSettings:
    level: str = "INFO"
    format: str = "text"        # text | json
    log_file: str = "./logs/voxscribe.log"
    max_bytes: int = 10_485_760  # 10 MB per file
    backup_count: int = 5


@dataclass
class CorsSettings:
    origins: List[str] = field(default_factory=lambda: ["http://localhost:3000"])


@dataclass
class Settings:
    server: ServerSettings = field(default_factory=ServerSettings)
    queue: QueueSettings = field(default_factory=QueueSettings)
    processing: ProcessingSettings = field(default_factory=ProcessingSettings)
    models: ModelsSettings = field(default_factory=ModelsSettings)
    storage: StorageSettings = field(default_factory=StorageSettings)
    auth: AuthSettings = field(default_factory=AuthSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)
    cors: CorsSettings = field(default_factory=CorsSettings)


def _get(yaml_section: dict, key: str, env_key: str, default, cast=str):
    """Read from env first, then YAML, then default. Cast to the right type."""
    yaml_val = yaml_section.get(key, default) if yaml_section else default
    raw = os.environ.get(env_key, yaml_val)
    if cast is bool:
        return str(raw).lower() in ("1", "true", "yes", "on")
    try:
        return cast(raw)
    except (TypeError, ValueError):
        return default


def _push_env(key: str, value) -> None:
    """Push a resolved value into os.environ if not already set externally."""
    os.environ.setdefault(key, str(value))


def load() -> Settings:
    cfg: dict = {}
    if _HAS_YAML and Path(CONFIG_PATH).exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}

    s = cfg.get("server", {})
    q = cfg.get("queue", {})
    p = cfg.get("processing", {})
    m = cfg.get("models", {})
    mo = m.get("ollama", {})
    mc = m.get("claude", {})
    mw = m.get("openai", {})
    st = cfg.get("storage", {})
    a = cfg.get("auth", {})
    lg = cfg.get("logging", {})
    co = cfg.get("cors", {})

    settings = Settings(
        server=ServerSettings(
            host=_get(s, "host", "HOST", "0.0.0.0"),
            port=_get(s, "port", "PORT", 8000, int),
        ),
        queue=QueueSettings(
            backend=_get(q, "backend", "QUEUE_BACKEND", "memory"),
            worker_concurrency=_get(q, "worker_concurrency", "WORKER_CONCURRENCY", 2, int),
            max_attempts=_get(q, "max_attempts", "QUEUE_MAX_ATTEMPTS", 3, int),
            base_backoff=_get(q, "base_backoff", "QUEUE_BACKOFF", 2.0, float),
            recording_ttl_seconds=_get(q, "recording_ttl_seconds", "RECORDING_TTL_SECONDS", 3600, int),
        ),
        processing=ProcessingSettings(
            extractor=_get(p, "extractor", "EXTRACTOR", "simulated"),
            asr_backend=_get(p, "asr_backend", "ASR_BACKEND", "stub"),
            cleanup=_get(p, "cleanup", "CLEANUP", False, bool),
        ),
        models=ModelsSettings(
            ollama=OllamaSettings(
                host=_get(mo, "host", "OLLAMA_HOST", "http://host.docker.internal:11434"),
                model=_get(mo, "model", "VOXSCRIBE_OLLAMA_MODEL", "gemma3:4b"),
            ),
            claude=ClaudeSettings(
                model=_get(mc, "model", "CLAUDE_MODEL", "claude-sonnet-4-6"),
                api_key=_get(mc, "api_key", "ANTHROPIC_API_KEY", ""),
            ),
            openai=OpenAISettings(
                api_key=_get(mw, "api_key", "OPENAI_API_KEY", ""),
                whisper_model=_get(mw, "whisper_model", "WHISPER_MODEL", "base"),
            ),
        ),
        storage=StorageSettings(
            data_dir=_get(st, "data_dir", "DATA_DIR", "./data"),
            db_path=_get(st, "db_path", "DB_PATH", "./data/ideascribe.db"),
            id_floor=_get(st, "id_floor", "IDEASCRIBE_ID_FLOOR", 53, int),
        ),
        auth=AuthSettings(
            default_tenant=_get(a, "default_tenant", "IDEASCRIBE_TENANT_ID", "tnt-acme"),
            default_workspace=_get(a, "default_workspace", "IDEASCRIBE_WORKSPACE_ID", "ws-discovery"),
        ),
        logging=LoggingSettings(
            level=_get(lg, "level", "LOG_LEVEL", "INFO"),
            format=_get(lg, "format", "LOG_FORMAT", "text"),
            log_file=_get(lg, "log_file", "LOG_FILE", "./logs/voxscribe.log"),
            max_bytes=_get(lg, "max_bytes", "LOG_MAX_BYTES", 10_485_760, int),
            backup_count=_get(lg, "backup_count", "LOG_BACKUP_COUNT", 5, int),
        ),
        cors=CorsSettings(
            origins=os.environ.get(
                "CORS_ORIGINS",
                ",".join(co.get("origins", ["http://localhost:3000"])),
            ).split(","),
        ),
    )

    # Push resolved values into os.environ so all existing modules that
    # read os.environ directly pick up YAML-configured defaults.
    _push_env("QUEUE_BACKEND", settings.queue.backend)
    _push_env("WORKER_CONCURRENCY", settings.queue.worker_concurrency)
    _push_env("QUEUE_MAX_ATTEMPTS", settings.queue.max_attempts)
    _push_env("QUEUE_BACKOFF", settings.queue.base_backoff)
    _push_env("RECORDING_TTL_SECONDS", settings.queue.recording_ttl_seconds)
    _push_env("EXTRACTOR", settings.processing.extractor)
    _push_env("ASR_BACKEND", settings.processing.asr_backend)
    _push_env("CLEANUP", "on" if settings.processing.cleanup else "off")
    _push_env("OLLAMA_HOST", settings.models.ollama.host)
    _push_env("VOXSCRIBE_OLLAMA_MODEL", settings.models.ollama.model)
    _push_env("CLAUDE_MODEL", settings.models.claude.model)
    _push_env("DATA_DIR", settings.storage.data_dir)
    _push_env("DB_PATH", settings.storage.db_path)
    _push_env("IDEASCRIBE_ID_FLOOR", settings.storage.id_floor)
    _push_env("IDEASCRIBE_TENANT_ID", settings.auth.default_tenant)
    _push_env("IDEASCRIBE_WORKSPACE_ID", settings.auth.default_workspace)
    _push_env("CORS_ORIGINS", ",".join(settings.cors.origins))
    _push_env("LOG_LEVEL", settings.logging.level)
    _push_env("LOG_FORMAT", settings.logging.format)
    _push_env("LOG_FILE", settings.logging.log_file)

    return settings


# Loaded once at import time.
settings: Settings = load()
