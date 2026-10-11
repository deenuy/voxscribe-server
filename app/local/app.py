"""FastAPI application setup."""

import hmac
import time
import os
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import psycopg
from fastapi.exceptions import RequestValidationError
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from psycopg_pool import PoolTimeout
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.local import tenancy
from app.local.errors import EngineError
from app.local.identity import IdentityError, tenant_for
from app.local.version import engine_version
from app.local.pg_index import PostgresKnowledgeIndex
from app.local.providers import select_providers
from app.local.store import Store


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager."""
    # Startup
    print("Starting up engine...", file=sys.stderr, flush=True)

    # Start ingestion worker if available
    if hasattr(app.state, "ingestion_worker"):
        await app.state.ingestion_worker.start()
        print("Ingestion worker started", file=sys.stderr, flush=True)

    # The models the person chose in an earlier run keep their roles.
    try:
        from app.local.model_service import restore_selected_roles

        pool = app.state.store.pool
        if getattr(pool, "enforced", False):
            for key in await pool.provisioner.known_tenants():
                with tenancy.tenant_scope(key):
                    await restore_selected_roles(pool)
        else:
            await restore_selected_roles(pool)
    except Exception as exc:  # the defaults from configuration still apply
        print(f"Restoring chosen models failed: {exc}", file=sys.stderr, flush=True)

    # A crash during finalisation leaves a meeting 'processing': finish it now.
    try:
        from app.local.finalize import recover_unfinished

        await recover_unfinished(app)
    except Exception as exc:  # never block startup on recovery
        print(f"Recovery of unfinished meetings failed: {exc}", file=sys.stderr, flush=True)

    yield

    # Shutdown
    print("Shutting down engine...", file=sys.stderr, flush=True)
    if hasattr(app.state, "ingestion_worker"):
        await app.state.ingestion_worker.stop()
        print("Ingestion worker stopped", file=sys.stderr, flush=True)
    if hasattr(app.state, "store"):
        await app.state.store.close()


async def create_app(
    settings,
    auth_disabled: bool = False,
    schema: str | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        settings: Configuration object
        auth_disabled: If True, skip authentication
        schema: Optional database schema name for test isolation
    """
    from app.local.config import Config

    if not isinstance(settings, Config):
        raise RuntimeError("settings must be a Config object")

    # Create data directory
    data_dir = Path(settings.data_dir) if settings.data_dir else Path("/tmp/voxscribe")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Connect to database
    try:
        store = await Store.create(settings.database_url, schema=schema, enforced=settings.tenancy_enforced)
    except Exception as e:
        raise RuntimeError(f"Failed to connect to database: {e}") from e

    # Initialize providers
    providers = select_providers(settings.providers, settings)

    # Create knowledge index
    knowledge_index = PostgresKnowledgeIndex(store.pool)

    # Create ingestion worker
    try:
        from app.local.ingest_jobs import IngestionWorker
        ingestion_worker = IngestionWorker(
            store.pool,
            settings.database_url,
            providers["embedding_provider"],
            num_workers=2,
        )
    except Exception as e:
        print(f"Failed to initialize ingestion worker: {e}", file=sys.stderr, flush=True)
        ingestion_worker = None

    # Create FastAPI app
    app = FastAPI(
        title="VoxScribe Engine",
        version=engine_version(),
        lifespan=lifespan,
    )

    # Store objects in app state
    app.state.store = store
    app.state.data_dir = data_dir
    app.state.providers = providers
    app.state.knowledge_index = knowledge_index
    app.state.ingestion_worker = ingestion_worker
    app.state.auth_disabled = auth_disabled
    app.state.auth_token = os.environ.get("VOXSCRIBE_TOKEN", str(uuid.uuid4()))
    app.state.settings = settings
    app.state.started_at = time.time()
    # Tenant isolation: with an identity service configured, every request names a person and each person's data is
    # kept in a schema of their own. Without one, nothing changes (one person at a time).
    app.state.identity = None
    if settings.tenancy_enforced:
        from app.local.identity import IdentityVerifier

        app.state.identity = IdentityVerifier(
            issuer=settings.identity_issuer,
            audience=settings.identity_audience,
            jwks_url=settings.identity_jwks_url,
            jwks_file=settings.identity_jwks_file or None,
        )

    # Auth middleware
    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        if request.url.path == "/health":
            return await call_next(request)

        if not app.state.auth_disabled:
            # Check Authorization header
            auth_header = request.headers.get("Authorization", "")
            token = None
            if auth_header.startswith("Bearer "):
                token = auth_header[7:]
            elif request.url.path.endswith("/live") or request.url.path == "/v1/events":
                # Query token only where a browser cannot set a header (WebSocket and SSE).
                token = request.query_params.get("token")

            if not token:
                return JSONResponse(
                    status_code=401,
                    content={"error": "unauthorized", "detail": "token required"},
                )

            # Check token (constant time comparison)
            if not _constant_time_compare(token, app.state.auth_token):
                return JSONResponse(
                    status_code=401,
                    content={"error": "unauthorized", "detail": "invalid token"},
                )

            request.state.token = token

        if app.state.identity is None:
            return await call_next(request)  # tenant isolation is off: one shared scope

        # Tenant isolation is on. Facts about the machine (health, the AI runtime's state) touch no one's data;
        # everything else needs a verified person, and runs inside that person's scope.
        path = request.url.path
        if path == "/v1/health" or path.startswith("/v1/engine/"):
            with tenancy.system_scope():
                return await call_next(request)
        identity_token = request.headers.get("X-Identity-Token")
        if not identity_token and path == "/v1/events":
            identity_token = request.query_params.get("identity")  # an EventSource cannot set headers
        try:
            scope = await tenant_for(app, identity_token)
        except IdentityError as exc:
            code = "identity_required" if not identity_token else "identity_invalid"
            return JSONResponse(status_code=401, content={"error": code, "detail": exc.detail})
        request.state.tenant = scope
        with tenancy.tenant_scope(scope):
            return await call_next(request)

    # CORS is added after the auth middleware so it is the OUTERMOST layer: preflight requests
    # (which cannot carry a token) and 401 responses must still get CORS headers, or a browser
    # reports every auth failure as an opaque network error.
    allowed_origins = settings.allowed_origins.split(",") if settings.allowed_origins else []
    allowed_origins = [o.strip() for o in allowed_origins if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins or ["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Exception handlers
    @app.exception_handler(EngineError)
    async def engine_error_handler(request: Request, exc: EngineError):
        return JSONResponse(
            status_code=exc.status,
            content={"error": exc.code, "detail": exc.detail, **exc.extra},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        # Same {error, detail} shape as every other failure, instead of FastAPI's default.
        first = exc.errors()[0] if exc.errors() else {}
        where = ".".join(str(p) for p in first.get("loc", ()) if p != "body")
        return JSONResponse(
            status_code=400,
            content={"error": "invalid_request", "detail": f"{where}: {first.get('msg', 'invalid value')}".strip(": ")},
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        # Unknown addresses and wrong methods get the same {error, detail} shape as every other failure.
        code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
        detail = {404: "no such address", 405: "this method is not allowed here"}.get(exc.status_code, str(exc.detail))
        return JSONResponse(status_code=exc.status_code, content={"error": code, "detail": detail}, headers=getattr(exc, "headers", None))

    @app.exception_handler(psycopg.errors.DataError)
    async def bad_value_handler(request: Request, exc: psycopg.errors.DataError):
        # A value PostgreSQL cannot store (a NUL byte, a number out of range) is the caller's mistake, not ours.
        return JSONResponse(status_code=400, content={"error": "invalid_request", "detail": "a value in the request is not acceptable"})

    @app.exception_handler(psycopg.errors.IntegrityError)
    async def integrity_handler(request: Request, exc: psycopg.errors.IntegrityError):
        # A constraint said no. Unique and foreign-key violations are conflicts with what is stored (it already
        # exists, it does not exist, or something still depends on it); anything else is a bad value.
        if isinstance(exc, (psycopg.errors.UniqueViolation, psycopg.errors.ForeignKeyViolation)):
            detail = "this already exists" if isinstance(exc, psycopg.errors.UniqueViolation) else "it refers to something that does not exist, or something still depends on it"
            return JSONResponse(status_code=409, content={"error": "conflict", "detail": detail})
        return JSONResponse(status_code=400, content={"error": "invalid_request", "detail": "a value in the request is not acceptable"})

    @app.exception_handler(psycopg.OperationalError)
    @app.exception_handler(PoolTimeout)
    async def database_down_handler(request: Request, exc: Exception):
        print(f"Database unavailable: {exc}", file=sys.stderr, flush=True)
        return JSONResponse(status_code=503, content={"error": "database_unavailable", "detail": "the database is not reachable, try again shortly"}, headers={"Retry-After": "5"})

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        print(f"Unhandled exception: {exc}", file=sys.stderr, flush=True)
        return JSONResponse(
            status_code=500,
            content={"error": "internal_error", "detail": "An unexpected error occurred"},
        )

    # Health endpoint
    @app.get("/v1/health")
    async def health():
        import httpx
        settings = app.state.settings

        # Database health
        try:
            db_ok, schema_version, _ = await store.health_check()
            db_full_name = "postgresql"  # the kind of database, never its name: a name such as a dev database must not reach a screen
        except Exception as exc:
            print(f"health: database check failed: {exc}", file=sys.stderr, flush=True)
            db_ok = False
            db_full_name = "postgresql"
            schema_version = 0

        # ASR health
        asr_ok = False
        asr_name = "unavailable"
        if settings.providers == "fake":
            asr_ok = True
            asr_name = "fake:deterministic"
        else:
            # Check if faster-whisper can be imported
            try:
                import faster_whisper  # noqa: F401
                asr_ok = True
                asr_name = f"faster-whisper:{settings.asr_model}"
            except ImportError:
                pass

        # LLM health (main model for enrichment)
        llm_ok = False
        llm_name = "unavailable"
        if settings.providers == "fake":
            llm_ok = True
            llm_name = "fake:deterministic"
        else:
            # Check Ollama for configured LLM model
            try:
                async with httpx.AsyncClient(timeout=2) as client:
                    response = await client.get(f"{settings.ollama_host}/api/tags")
                    if response.status_code == 200:
                        models = response.json().get("models", [])
                        model_names = [m.get("name", "") for m in models]
                        if settings.llm_model in model_names:
                            llm_ok = True
                            llm_name = f"ollama:{settings.llm_model}"
                        else:
                            llm_name = f"ollama:{settings.llm_model}"
            except Exception:
                llm_name = f"ollama:{settings.llm_model}"

        # Live LLM health
        live_llm_ok = False
        live_llm_name = "unavailable"
        if settings.providers == "fake":
            live_llm_ok = True
            live_llm_name = "fake:deterministic"
        else:
            # Check Ollama for configured live model
            try:
                async with httpx.AsyncClient(timeout=2) as client:
                    response = await client.get(f"{settings.ollama_host}/api/tags")
                    if response.status_code == 200:
                        models = response.json().get("models", [])
                        model_names = [m.get("name", "") for m in models]
                        if settings.live_model in model_names:
                            live_llm_ok = True
                            live_llm_name = f"ollama:{settings.live_model}"
                        else:
                            live_llm_name = f"ollama:{settings.live_model}"
            except Exception:
                live_llm_name = f"ollama:{settings.live_model}"

        # Embedding health
        embedding_ok = False
        embedding_name = "unavailable"
        if settings.providers == "fake":
            embedding_ok = True
            embedding_name = "fake:deterministic"
        else:
            # Check Ollama for configured embedding model
            try:
                async with httpx.AsyncClient(timeout=2) as client:
                    response = await client.get(f"{settings.ollama_host}/api/tags")
                    if response.status_code == 200:
                        models = response.json().get("models", [])
                        model_names = [m.get("name", "") for m in models]
                        # Check if model exists; treat missing tag as :latest
                        configured = settings.embed_model
                        configured_with_latest = f"{configured}:latest"
                        if configured in model_names or configured_with_latest in model_names:
                            embedding_ok = True
                            embedding_name = f"ollama:{configured}"
                        else:
                            embedding_name = f"ollama:{configured}"
            except Exception:
                embedding_name = f"ollama:{settings.embed_model}"

        status = "ok" if (db_ok and asr_ok and llm_ok and embedding_ok) else "degraded"

        result = {
            "status": status,
            "version": engine_version(),
            "profile": "ideascribe",
            "capabilities": [
                "transcribe",
                "enhance",
                "summary",
                "knowledge",
                "chat",
                "assist",
                "models",
                "stats",
                "calendar",
                "google",
                "ideas",
            ],
            "offline": True,
            "database": {
                "ok": db_ok,
                "name": db_full_name,
                "schemaVersion": int(schema_version) if schema_version else 0,
            },
            "providers": {
                "asr": {
                    "ok": asr_ok,
                    "name": asr_name,
                },
                "llm": {
                    "ok": llm_ok,
                    "name": llm_name,
                },
                "liveLlm": {
                    "ok": live_llm_ok,
                    "name": live_llm_name,
                },
                "embedding": {
                    "ok": embedding_ok,
                    "name": embedding_name,
                },
            },
        }
        return result

    # Auto-discover and register routers from app/local/routers/
    import importlib
    import pkgutil

    from app.local import routers

    routers_module_path = routers.__path__
    for _importer, modname, _ispkg in pkgutil.iter_modules(routers_module_path):
        if modname.startswith("_"):
            continue
        full_module_name = f"app.local.routers.{modname}"
        try:
            module = importlib.import_module(full_module_name)
            if hasattr(module, "router"):
                app.include_router(module.router)
        except (ImportError, AttributeError) as exc:
            # A router that cannot load must stop the engine from starting. Skipping it would leave every one of
            # its addresses answering 404 while health still says ok.
            raise RuntimeError(f"router {full_module_name} failed to load: {exc}") from exc

    return app


def _constant_time_compare(a: str, b: str) -> bool:
    """Constant-time string comparison to prevent timing attacks."""
    return hmac.compare_digest(a.encode(), b.encode())
