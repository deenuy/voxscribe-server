# VoxScribe Local Engine

## Running the Engine

Start the engine with:

```bash
uv run voxscribe-engine --data-dir /tmp/voxscribe --port 0
```

The engine will:
- Run migrations on startup
- Bind to 127.0.0.1 on a free port (with `--port 0`)
- Print exactly one JSON line to stdout: `{"event":"ready","port":<int>,"version":"<semver>"}`
- Log everything else to stderr
- Exit cleanly on SIGTERM

## Configuration

Configuration is loaded in precedence order:
1. Environment variables (highest priority)
2. Dotenv file (if `VOXSCRIBE_ENV_FILE` is set)
3. Built-in defaults

See `config.py` for all available keys. Invalid values cause exit code 2 with a clear error message on stderr.

## Testing

Test database isolation uses temporary schemas named `t_<hex>`:

```bash
# Run all tests
uv run --extra dev pytest -q

# Run with coverage
uv run --extra dev pytest --cov=app/local tests/
```

Each test gets its own schema created from the migrations. The fixture must drop it on teardown.

## Store Architecture

Database access is split by area:
- `store_meetings.py` - meetings, notes, transcripts, enhanced blocks, summaries, attachments
- `store_sources.py` - sources, labels, source-label associations, chunks
- `store_chat.py` - chat threads and messages
- `store_models.py` - models and downloads
- `store_misc.py` - settings, calendar events, usage metrics, auth events, connector accounts

This allows parallel agents to own one module each.

## Router Convention

Routers are auto-discovered from `app/local/routers/`. Any module with a `router` object is included in the FastAPI app. Add new routers without editing `app.py`.

## Health Endpoint

`GET /v1/health` checks:
- Database connectivity and schema version
- ASR provider (faster-whisper or fake)
- LLM provider (Ollama models or fake)
- Embedding provider (Ollama models or fake)

Database password in connection URL is redacted in health responses.

## Database Migrations

Migrations live in `app/local/migrations/` as numbered SQL files:
- `001_init.sql` - initial schema
- `002_add_auth_and_metrics.sql` - auth events, connector accounts, usage metrics

Migrations run automatically on startup and are recorded in the `schema_migrations` table.
