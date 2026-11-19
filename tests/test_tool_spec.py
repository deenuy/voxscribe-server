"""PL43: the tool specification record, its validator and the registry (section 2.3).

Pure tests (no database, no model) except the start-up test, which builds the real engine on a
throwaway schema to show that a bad record stops it.
"""
from __future__ import annotations

import copy

import pytest

from app.local.tools import builtin
from app.local.tools.network import decide
from app.local.tools.registry import (
    REASON_MODE_NOT_ALLOWED,
    ToolRegistry,
    default_registry,
)
from app.local.tools.spec import FIELDS, ArgsError, ToolSpec, ToolSpecError


def _record(name: str = builtin.REMINDER_CREATE) -> dict:
    return copy.deepcopy(next(r for r in builtin.FIRST_PARTY if r["name"] == name))


def _spec(**over) -> ToolSpec:
    record = _record()
    record.update(over)
    return ToolSpec.from_record(record)


ALLOWED = decide(mode="personal", profile_allowed=True, privacy_mode="connectors")
OFFLINE = decide(mode="personal", profile_allowed=True, privacy_mode="offline")


# ── records ──────────────────────────────────────────────────────────────────


def test_every_first_party_record_validates_with_its_own_version():
    registry = default_registry()
    names = [s.name for s in registry.specs()]
    assert names == [builtin.WEB_SEARCH, builtin.REMINDER_CREATE, builtin.TIMER_START,
                     builtin.REMINDER_LIST]
    versions = [s.version for s in registry.specs()]
    assert all(len(v) == 16 for v in versions) and len(set(versions)) == len(versions)
    assert default_registry().get(builtin.WEB_SEARCH).version == versions[0]   # stable


@pytest.mark.parametrize("missing", FIELDS)
def test_a_record_missing_a_field_is_refused_naming_the_field(missing):
    record = _record()
    del record[missing]
    with pytest.raises(ToolSpecError) as err:
        ToolSpec.from_record(record)
    assert err.value.field == missing and repr(missing) in str(err.value)


def test_a_changed_record_has_a_different_version():
    assert _spec(timeout_ms=2_001).version != _spec().version


def test_a_name_twice_is_refused():
    with pytest.raises(ToolSpecError) as err:
        ToolRegistry([_record(), _record()])
    assert err.value.field == "name"


@pytest.mark.parametrize("field_name,value", [
    ("name", "Reminder"),
    ("side_effect", "delete"),
    ("confirm", "none"),                  # a write must announce or ask
    ("modes", ["personal", "everyone"]),
    ("modes", []),
    ("network", "internet"),
    ("lane", "slow"),
    ("timeout_ms", 0),
    ("timeout_ms", True),
    ("untrusted_result", "no"),
    ("summary", " "),
])
def test_bad_values_are_refused_by_field(field_name, value):
    with pytest.raises(ToolSpecError) as err:
        _spec(**{field_name: value})
    assert err.value.field == field_name


def test_a_remote_tool_must_be_untrusted_and_on_the_delegate_lane():
    record = _record(builtin.WEB_SEARCH)
    with pytest.raises(ToolSpecError) as err:
        ToolSpec.from_record({**record, "lane": "fast"})
    assert err.value.field == "lane"
    with pytest.raises(ToolSpecError) as err:
        ToolSpec.from_record({**record, "untrusted_result": False})
    assert err.value.field == "untrusted_result"


@pytest.mark.parametrize("schema", [
    {"type": "array", "items": {"type": "string"}},
    {"type": "object", "properties": {"x": {"type": "array", "maxLength": 3}},
     "additionalProperties": False},
    {"type": "object", "properties": {"x": {"type": "string"}}, "additionalProperties": False},
    {"type": "object", "properties": {"x": {"type": "string", "maxLength": 9}}},
    {"type": "object", "properties": {"x": {"type": "string", "maxLength": 9,
                                            "format": "date"}},
     "additionalProperties": False},
    {"type": "object", "properties": {"x": {"type": "integer", "minimum": "1"}},
     "additionalProperties": False},
    {"type": "object", "properties": {}, "required": ["x"], "additionalProperties": False},
])
def test_schemas_outside_the_subset_are_refused(schema):
    with pytest.raises(ToolSpecError) as err:
        ToolSpec.from_record({**_record(builtin.REMINDER_LIST), "args": schema})
    assert err.value.field.startswith("args")


def test_templates_are_filled_only_from_arguments_and_the_receipt():
    record = _record()
    record["templates"]["success"] = "Done, {who} will be reminded."
    with pytest.raises(ToolSpecError) as err:
        ToolSpec.from_record(record)
    assert err.value.field == "templates.success"

    record = _record()
    record["templates"]["failure"] = "Could not save {reminder_id}."    # no receipt on failure
    with pytest.raises(ToolSpecError) as err:
        ToolSpec.from_record(record)
    assert err.value.field == "templates.failure"

    record = _record()
    del record["templates"]["timeout"]
    with pytest.raises(ToolSpecError) as err:
        ToolSpec.from_record(record)
    assert err.value.field == "templates.timeout"


def test_an_asking_tool_needs_a_readback_line():
    with pytest.raises(ToolSpecError) as err:
        _spec(confirm="ask")
    assert err.value.field == "templates.readback"


def test_the_audit_list_must_cover_every_argument_exactly_once():
    for audit in ({"store": ["fire_at"], "hash": []},
                  {"store": ["fire_at", "text"], "hash": ["text"]},
                  {"store": ["fire_at"], "hash": ["text", "who"]},
                  {"keep": ["fire_at"]}):
        with pytest.raises(ToolSpecError) as err:
            _spec(audit=audit)
        assert err.value.field == "audit"


def test_hashed_arguments_are_kept_as_a_digest_and_a_length():
    spec = _spec()
    args = {"text": "call the dentist", "fire_at": "2026-11-19T15:00:00+00:00"}
    kept = spec.audit_args(args)
    assert kept["fire_at"] == args["fire_at"]
    assert kept["text"]["len"] == len(args["text"]) and "dentist" not in str(kept)


# ── arguments ────────────────────────────────────────────────────────────────


GOOD = {"text": "call the dentist", "fire_at": "2026-11-19T15:00:00+00:00"}


@pytest.mark.parametrize("args,code,field_name", [
    ({**GOOD, "priority": "high"}, "unknown_field", "priority"),
    ({"text": "x"}, "missing_field", "fire_at"),
    ({**GOOD, "text": 7}, "wrong_type", "text"),
    ({**GOOD, "text": "x" * 501}, "too_long", "text"),
    ({**GOOD, "fire_at": "tomorrow at nine"}, "pattern_mismatch", "fire_at"),
    ("call the dentist", "not_an_object", ""),
])
def test_arguments_are_refused_with_a_code_and_never_repaired(args, code, field_name):
    with pytest.raises(ArgsError) as err:
        _spec().validate_args(args)
    assert (err.value.code, err.value.field) == (code, field_name)
    assert "dentist" not in str(err.value)


def test_integers_ranges_and_booleans_are_strict():
    spec = ToolSpec.from_record(_record(builtin.REMINDER_LIST))
    result = {"type": "object", "properties": {"n": {"type": "integer", "minimum": 1,
                                                     "maximum": 5}},
              "required": ["n"], "additionalProperties": False}
    from app.local.tools.spec import validate_value

    assert validate_value(result, {"n": 3}) == {"n": 3}
    for value, code in ((0, "out_of_range"), (6, "out_of_range"), (True, "wrong_type"),
                        (2.5, "wrong_type"), ("3", "wrong_type")):
        with pytest.raises(ArgsError) as err:
            validate_value(result, {"n": value})
        assert err.value.code == code
    assert spec.validate_args({}) == {}


# ── the registry ─────────────────────────────────────────────────────────────


def _names(specs) -> list[str]:
    return [s.name for s in specs]


def test_which_tools_a_session_may_use():
    registry = default_registry()
    everything = _names(registry.specs())
    assert _names(registry.offered("personal", ALLOWED)) == everything
    assert builtin.WEB_SEARCH not in _names(registry.offered("personal", OFFLINE))
    for mode in ("senior_care", "classroom", "seniorcare"):
        decision = decide(mode=mode, profile_allowed=True, privacy_mode="connectors")
        offered = _names(registry.offered(mode, decision))
        assert builtin.WEB_SEARCH not in offered and builtin.REMINDER_CREATE in offered
    assert registry.offered("dictation", ALLOWED) == ()


def test_permits_names_why_a_tool_is_refused():
    registry = default_registry()
    search = registry.get(builtin.WEB_SEARCH)
    assert registry.permits(search, "personal", ALLOWED) is None
    assert registry.permits(search, "personal", OFFLINE) == "privacy_offline"
    assert registry.permits(search, "senior_care",
                            decide(mode="senior_care", profile_allowed=True,
                                   privacy_mode="connectors")) == "mode_restricted"
    record = {**_record(builtin.TIMER_START), "modes": ["personal"]}
    timer = ToolRegistry([record]).get(builtin.TIMER_START)
    assert registry.permits(timer, "classroom", ALLOWED) == REASON_MODE_NOT_ALLOWED


# ── start-up ─────────────────────────────────────────────────────────────────


async def test_a_bad_record_stops_the_engine_naming_the_field(monkeypatch, throwaway_schema,
                                                             throwaway_data_dir):
    from app.local.app import create_app
    from app.local.config import Config
    from tests.conftest import _DB_URL

    broken = _record()
    del broken["timeout_ms"]
    monkeypatch.setattr(builtin, "FIRST_PARTY", (*builtin.FIRST_PARTY[:1], broken))
    settings = Config(database_url=_DB_URL, data_dir=str(throwaway_data_dir), providers="fake")
    with pytest.raises(ToolSpecError, match="timeout_ms"):
        await create_app(settings=settings, auth_disabled=True, schema=throwaway_schema)
