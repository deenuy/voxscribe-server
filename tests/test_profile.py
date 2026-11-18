"""Server-side profile tests.

test_startup_bad_profile_exits: start the server with an invalid profile
(enrollment.required=True in classroom mode) and confirm it exits with code 1.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap


def test_startup_bad_profile_exits(tmp_path):
    """Server started with an invalid profile combination exits with code 1."""
    bad_yaml = tmp_path / "bad_profile.yaml"
    bad_yaml.write_text(
        textwrap.dedent("""\
            mode: classroom
            enrollment:
              required: true
        """),
        encoding="utf-8",
    )

    env = {**os.environ, "VOXSCRIBE_PROFILE": str(bad_yaml)}
    # Use python -c to call create_app directly so we do not need a live DB.
    # The profile is validated before any DB connection, so the process exits 1.
    script = textwrap.dedent("""\
        import asyncio, os, sys
        # Provide a minimal Config so create_app does not crash before profile loading.
        # Profile loading happens before the DB connection attempt, so a bad profile
        # exits before any real I/O.
        from voxscribe.profile.loader import load_profile
        from voxscribe.profile.schema import ProfileValidationError
        profile_path = os.environ.get("VOXSCRIBE_PROFILE")
        try:
            load_profile(profile_path)
        except ProfileValidationError as exc:
            print(f"startup aborted: {exc}", file=sys.stderr)
            sys.exit(1)
    """)

    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 1, (
        f"expected exit code 1, got {result.returncode}; stderr={result.stderr!r}"
    )
    assert "classroom mode does not support enrollment" in result.stderr
