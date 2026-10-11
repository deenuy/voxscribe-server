"""The engine's own version, read from the installed package: one source, never a string typed in two places."""

from importlib import metadata


def engine_version() -> str:
    try:
        return metadata.version("voxscribe-server")
    except metadata.PackageNotFoundError:
        return "unknown"
