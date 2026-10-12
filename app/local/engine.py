"""Entry point for the voxscribe-engine."""

import argparse
import asyncio
import json
import os
import sys

import uvicorn

from app.local.app import create_app
from app.local.config import Config
from app.local.db import redact_password
from app.local.log_redaction import install as redact_secrets_in_logs
from app.local.migrations import run_migrations

VERSION = "0.1.0"


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="VoxScribe local engine")
    parser.add_argument("--port", type=int, default=0, help="Port to bind to (0 = pick free)")
    parser.add_argument("--data-dir", type=str, default=None, help="Data directory")
    parser.add_argument(
        "--database-url",
        type=str,
        default=None,
        help="PostgreSQL database URL",
    )
    parser.add_argument(
        "--providers",
        type=str,
        default=None,
        help="Provider mode (fake or local)",
    )
    parser.add_argument(
        "--no-auth",
        action="store_true",
        help="Disable authentication (dev only)",
    )

    args = parser.parse_args()

    # Load config from CLI > env > dotenv > defaults
    # This validates the config BEFORE we do anything else
    settings = _load_and_validate_config(args)

    # Run migrations
    try:
        print(f"Running migrations on {redact_password(settings.database_url)}...", file=sys.stderr)
        schema_version = await run_migrations(settings.database_url)
        print(f"Migrations complete. Schema version: {schema_version}", file=sys.stderr)
    except Exception as e:
        print(f"Migration failed: {e}", file=sys.stderr)
        print(json.dumps({"error": "database", "detail": str(e)}), file=sys.stderr)
        sys.exit(1)

    # Create app
    try:
        app = await create_app(
            settings=settings,
            auth_disabled=args.no_auth,
        )
    except Exception as e:
        print(f"Failed to create app: {e}", file=sys.stderr)
        print(json.dumps({"error": "startup", "detail": str(e)}), file=sys.stderr)
        sys.exit(1)

    # Create and configure uvicorn. The ready line is printed once the socket is
    # bound, so the port is the real one even with --port 0.
    uvicorn_config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=args.port,
        lifespan="on",
        access_log=False,
        log_level=settings.log_level.lower(),
    )
    # uvicorn sets up its loggers in Config; the live socket's address carries tokens, so redact them in log lines.
    redact_secrets_in_logs()

    class ReadyServer(uvicorn.Server):
        async def startup(self, sockets=None):
            await super().startup(sockets=sockets)
            if self.started and self.servers:
                port = self.servers[0].sockets[0].getsockname()[1]
                # stdout carries exactly this one line; everything else goes to stderr.
                print(json.dumps({"event": "ready", "port": port, "version": VERSION}), flush=True)

    # uvicorn installs its own SIGINT and SIGTERM handlers and shuts down cleanly.
    await ReadyServer(uvicorn_config).serve()


def _load_and_validate_config(args) -> Config:
    """Load configuration with precedence: CLI > env > dotenv > defaults.

    Validates config BEFORE returning. Invalid values exit with code 2.
    """
    # Build an env dict with CLI overrides
    cli_overrides = {}
    if args.database_url:
        cli_overrides["VOXSCRIBE_DATABASE_URL"] = args.database_url
    if args.data_dir:
        cli_overrides["VOXSCRIBE_DATA_DIR"] = args.data_dir
    if args.providers:
        cli_overrides["VOXSCRIBE_PROVIDERS"] = args.providers

    # Temporarily set overrides in os.environ so Config.from_env() sees them
    saved_env = {}
    for key, value in cli_overrides.items():
        saved_env[key] = os.environ.get(key)
        os.environ[key] = value

    try:
        config = Config.from_env()
    finally:
        # Restore original environment
        for key, original_value in saved_env.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value

    return config


def run():
    """Entry point for the console script."""
    asyncio.run(main())


if __name__ == "__main__":
    run()
