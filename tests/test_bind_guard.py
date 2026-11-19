"""The engine refuses to serve with authentication off on an address other than loopback (PL08, gap T4).

Tested at the seam where the bind address is decided (engine.bind_host) and through the engine's entry point.
Neither opens a database: the refusal comes before migrations, and the test fails if they are reached.
"""
import asyncio
import sys

import pytest

from app.local import engine


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.20", "10.0.0.5", "example.com"])
def test_authentication_off_on_a_non_loopback_address_exits_non_zero(host):
    with pytest.raises(SystemExit) as exc:
        engine.bind_host(auth_disabled=True, requested=host)
    assert exc.value.code not in (0, None)


@pytest.mark.parametrize("host", ["127.0.0.1", "127.0.0.2", "::1", "localhost"])
def test_authentication_off_on_loopback_is_allowed(host):
    assert engine.bind_host(auth_disabled=True, requested=host) == host


def test_authentication_on_any_address_is_allowed_and_the_default_is_loopback():
    assert engine.bind_host(auth_disabled=False, requested="0.0.0.0") == "0.0.0.0"
    assert engine.bind_host(auth_disabled=True) == engine.BIND_HOST == "127.0.0.1"


def test_the_engine_entry_point_exits_non_zero_before_touching_the_database(monkeypatch, tmp_path):
    def must_not_run(*_a, **_k):
        raise AssertionError("the engine went on to the database after refusing its bind address")

    monkeypatch.setattr(engine, "BIND_HOST", "0.0.0.0")
    monkeypatch.setattr(engine, "run_migrations", must_not_run)
    monkeypatch.setattr(engine, "create_app", must_not_run)
    monkeypatch.setenv("VOXSCRIBE_ENV_FILE", str(tmp_path / "no.env"))
    monkeypatch.setattr(sys, "argv", ["voxscribe-engine", "--no-auth", "--providers", "fake"])
    with pytest.raises(SystemExit) as exc:
        asyncio.run(engine.main())
    assert exc.value.code not in (0, None)
