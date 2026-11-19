"""The tool specification (PL43, TOOLS-CONNECTORS.md section 2.3): one declarative record per tool.

A record is plain data (a mapping). ToolSpec.from_record() validates every field and refuses a
record that lacks one, naming the field, so a bad record stops the engine at start-up. Argument
and result schemas are a JSON Schema subset and nothing else is accepted:

  object    properties, required, additionalProperties: false (mandatory)
  string    maxLength (mandatory: no unbounded text reaches a tool), pattern, enum
  integer   minimum, maximum
  number    minimum, maximum
  boolean
  every property may carry a description

validate_args() checks arguments against `args` strictly: an unknown field, a missing field, a wrong
type, a value out of range, an over-long string, a pattern or enum miss is an ArgsError with a code.
There is no repair and no coercion.

Pure, with no I/O and no engine imports: shaped to move into voxscribe-core unchanged (PL58).
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import string
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

SIDE_EFFECTS = ("read", "write", "irreversible")
CONFIRMS = ("none", "announce", "ask", "ask_strict")
MODES = ("personal", "ideascribe", "senior_care", "classroom")
NETWORKS = ("none", "local", "remote")
LANES = ("fast", "delegate")

# Every record carries these, in this order. `version` is not authored: it is the record's hash.
FIELDS = ("name", "summary", "args", "result", "side_effect", "confirm", "modes", "network",
          "lane", "timeout_ms", "templates", "audit", "untrusted_result")

# The scripted lines every tool has. `readback` is required when the tool asks before it runs.
TEMPLATE_KEYS = ("success", "failure", "timeout", "refused", "invalid")
READBACK_KEY = "readback"

MAX_TIMEOUT_MS = 120_000
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$")
_KEYWORDS = {
    "string": {"type", "description", "maxLength", "pattern", "enum"},
    "integer": {"type", "description", "minimum", "maximum"},
    "number": {"type", "description", "minimum", "maximum"},
    "boolean": {"type", "description"},
    "object": {"type", "description", "properties", "required", "additionalProperties"},
}


class ToolSpecError(ValueError):
    """A record that cannot be a tool. `field` names what is wrong."""

    def __init__(self, tool: str, field_name: str, reason: str) -> None:
        super().__init__(f"tool spec {tool or '<unnamed>'}: field {field_name!r}: {reason}")
        self.tool, self.field, self.reason = tool, field_name, reason


class ArgsError(ValueError):
    """Arguments refused by the schema. Carries a code and a field name, never the value."""

    def __init__(self, code: str, field_name: str = "") -> None:
        super().__init__(f"{code}:{field_name}" if field_name else code)
        self.code, self.field = code, field_name


# ── schema subset ────────────────────────────────────────────────────────────


def _check_schema(tool: str, where: str, schema: Any, *, top: bool = False) -> None:
    def bad(reason: str):
        raise ToolSpecError(tool, where, reason)

    if not isinstance(schema, Mapping):
        bad("must be a schema object")
    kind = schema.get("type")
    if kind not in _KEYWORDS:
        bad(f"type must be one of {sorted(_KEYWORDS)}")
    if top and kind != "object":
        bad("must be an object schema")
    unknown = set(schema) - _KEYWORDS[kind]
    if unknown:
        bad(f"keywords not in the accepted subset: {sorted(unknown)}")
    if kind == "object":
        props = schema.get("properties")
        if not isinstance(props, Mapping):
            bad("an object needs properties")
        if schema.get("additionalProperties") is not False:
            bad("an object needs additionalProperties: false")
        required = schema.get("required", [])
        if not isinstance(required, list) or not set(required) <= set(props):
            bad("required must list declared properties")
        for name, sub in props.items():
            if not isinstance(name, str) or not name:
                bad("property names must be strings")
            _check_schema(tool, f"{where}.{name}", sub)
    elif kind == "string":
        limit = schema.get("maxLength")
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            bad("a string needs a positive maxLength")
        if "pattern" in schema:
            try:
                re.compile(schema["pattern"])
            except (re.error, TypeError):
                bad("pattern does not compile")
        if "enum" in schema and (not isinstance(schema["enum"], list) or not schema["enum"]
                                 or not all(isinstance(v, str) for v in schema["enum"])):
            bad("enum must be a non-empty list of strings")
    elif kind in ("integer", "number"):
        for bound in ("minimum", "maximum"):
            if bound in schema and not _is_number(schema[bound]):
                bad(f"{bound} must be a number")


def _is_number(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def validate_value(schema: Mapping, value: Any, field_name: str = "") -> Any:
    """`value` checked against `schema`; returns it unchanged or raises ArgsError."""
    kind = schema["type"]
    if kind == "object":
        if not isinstance(value, Mapping):
            raise ArgsError("wrong_type", field_name)
        props = schema["properties"]
        for key in value:
            if key not in props:
                raise ArgsError("unknown_field", _join(field_name, str(key)))
        for key in schema.get("required", []):
            if key not in value:
                raise ArgsError("missing_field", _join(field_name, key))
        return {key: validate_value(props[key], value[key], _join(field_name, key))
                for key in value}
    if kind == "string":
        if not isinstance(value, str):
            raise ArgsError("wrong_type", field_name)
        if len(value) > schema["maxLength"]:
            raise ArgsError("too_long", field_name)
        if "enum" in schema and value not in schema["enum"]:
            raise ArgsError("not_in_enum", field_name)
        if "pattern" in schema and re.fullmatch(schema["pattern"], value) is None:
            raise ArgsError("pattern_mismatch", field_name)
        return value
    if kind == "boolean":
        if not isinstance(value, bool):
            raise ArgsError("wrong_type", field_name)
        return value
    if kind == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
        raise ArgsError("wrong_type", field_name)
    if kind == "number" and not _is_number(value):
        raise ArgsError("wrong_type", field_name)
    if "minimum" in schema and value < schema["minimum"]:
        raise ArgsError("out_of_range", field_name)
    if "maximum" in schema and value > schema["maximum"]:
        raise ArgsError("out_of_range", field_name)
    return value


def _join(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


# ── the record ───────────────────────────────────────────────────────────────


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    summary: str
    args: Mapping
    result: Mapping
    side_effect: str
    confirm: str
    modes: tuple[str, ...]
    network: str
    lane: str
    timeout_ms: int
    templates: Mapping
    audit: Mapping              # {"store": (fields...), "hash": (fields...)}
    untrusted_result: bool
    version: str = field(default="", compare=False)

    @classmethod
    def from_record(cls, record: Mapping) -> ToolSpec:
        """Validate a declarative record. Raises ToolSpecError naming the first bad field."""
        if not isinstance(record, Mapping):
            raise ToolSpecError("", "record", "must be a mapping")
        name = record.get("name") if isinstance(record.get("name"), str) else ""
        for key in FIELDS:
            if key not in record:
                raise ToolSpecError(name, key, "missing")
        extra = set(record) - set(FIELDS)
        if extra:
            raise ToolSpecError(name, sorted(extra)[0], "not a spec field")

        def bad(key: str, reason: str):
            raise ToolSpecError(name, key, reason)

        if not name or not _NAME_RE.match(name):
            bad("name", "must be namespaced lower case, like reminder.create")
        if not isinstance(record["summary"], str) or not record["summary"].strip():
            bad("summary", "must be one sentence")
        _check_schema(name, "args", record["args"], top=True)
        _check_schema(name, "result", record["result"], top=True)
        side_effect, confirm = record["side_effect"], record["confirm"]
        if side_effect not in SIDE_EFFECTS:
            bad("side_effect", f"must be one of {SIDE_EFFECTS}")
        if confirm not in CONFIRMS:
            bad("confirm", f"must be one of {CONFIRMS}")
        if side_effect == "read" and confirm != "none":
            bad("confirm", "a read confirms nothing: none")
        if side_effect != "read" and confirm == "none":
            bad("confirm", "a write must announce or ask")
        if side_effect == "irreversible" and confirm not in ("ask", "ask_strict"):
            bad("confirm", "an irreversible tool must ask")
        modes = record["modes"]
        if (not isinstance(modes, (list, tuple)) or not modes
                or not set(modes) <= set(MODES) or len(set(modes)) != len(modes)):
            bad("modes", f"must be a non-empty list from {MODES}")
        if record["network"] not in NETWORKS:
            bad("network", f"must be one of {NETWORKS}")
        if record["lane"] not in LANES:
            bad("lane", f"must be one of {LANES}")
        if record["network"] == "remote" and record["lane"] != "delegate":
            bad("lane", "a remote tool runs on the delegate lane")
        timeout = record["timeout_ms"]
        if (not isinstance(timeout, int) or isinstance(timeout, bool)
                or not 0 < timeout <= MAX_TIMEOUT_MS):
            bad("timeout_ms", f"must be an integer in 1..{MAX_TIMEOUT_MS}")
        if not isinstance(record["untrusted_result"], bool):
            bad("untrusted_result", "must be true or false")
        if record["network"] == "remote" and not record["untrusted_result"]:
            bad("untrusted_result", "a result from outside the engine is untrusted")
        _check_templates(name, record)
        _check_audit(name, record)

        spec = cls(name=name, summary=record["summary"].strip(), args=_freeze(record["args"]),
                   result=_freeze(record["result"]), side_effect=side_effect, confirm=confirm,
                   modes=tuple(modes), network=record["network"], lane=record["lane"],
                   timeout_ms=timeout, templates=_freeze(record["templates"]),
                   audit=_freeze({"store": tuple(record["audit"].get("store", ())),
                                  "hash": tuple(record["audit"].get("hash", ()))}),
                   untrusted_result=record["untrusted_result"])
        digest = hashlib.sha256(canonical_json(spec.record()).encode()).hexdigest()
        object.__setattr__(spec, "version", digest[:16])
        return spec

    def record(self) -> dict:
        """The declarative record, as plain data (what `version` hashes)."""
        return {key: _plain(getattr(self, key)) for key in FIELDS}

    # ── using the record ──

    def validate_args(self, raw: Any) -> dict:
        """The arguments, strictly checked. Raises ArgsError."""
        if not isinstance(raw, Mapping):
            raise ArgsError("not_an_object")
        return validate_value(self.args, raw)

    def validate_result(self, raw: Any) -> dict:
        return validate_value(self.result, raw)

    def render(self, key: str, values: Mapping[str, Any]) -> str:
        """A scripted line filled only from `values` (validated arguments and the receipt)."""
        return self.templates[key].format_map({k: values[k] for k in _placeholders(
            self.templates[key])})

    def audit_args(self, args: Mapping[str, Any]) -> dict:
        """The arguments as the audit row keeps them: stored fields as they are, hashed fields as
        a digest and a length, nothing else."""
        out: dict[str, Any] = {}
        for key in self.audit["store"]:
            if key in args:
                out[key] = args[key]
        for key in self.audit["hash"]:
            if key in args:
                text = canonical_json(args[key])
                out[key] = {"sha256": hashlib.sha256(text.encode()).hexdigest()[:16],
                            "len": len(args[key]) if isinstance(args[key], str) else len(text)}
        return out


def args_digest(args: Any) -> str:
    return hashlib.sha256(canonical_json(args).encode()).hexdigest()


def _placeholders(template: str) -> list[str]:
    return [f for _, f, _, _ in string.Formatter().parse(template) if f is not None]


def _check_templates(name: str, record: Mapping) -> None:
    templates = record["templates"]
    if not isinstance(templates, Mapping):
        raise ToolSpecError(name, "templates", "must be a mapping of scripted lines")
    keys = list(TEMPLATE_KEYS)
    if record["confirm"] in ("ask", "ask_strict"):
        keys.append(READBACK_KEY)
    for key in keys:
        line = templates.get(key)
        if not isinstance(line, str) or not line.strip():
            raise ToolSpecError(name, f"templates.{key}", "missing")
    extra = set(templates) - set(keys)
    if extra:
        raise ToolSpecError(name, f"templates.{sorted(extra)[0]}", "not a template key")
    fillable = set(record["args"]["properties"]) | set(record["result"]["properties"])
    for key in keys:
        try:
            fields = _placeholders(templates[key])
        except ValueError:
            raise ToolSpecError(name, f"templates.{key}", "malformed placeholder") from None
        for f in fields:
            if f not in fillable or (key != "success" and f not in record["args"]["properties"]):
                # Only a success line may use the result: nothing else has a receipt.
                raise ToolSpecError(name, f"templates.{key}",
                                    f"{{{f}}} is not a field it can be filled from")


def _check_audit(name: str, record: Mapping) -> None:
    audit = record["audit"]
    if not isinstance(audit, Mapping) or set(audit) - {"store", "hash"}:
        raise ToolSpecError(name, "audit", "must be {store: [...], hash: [...]}")
    store, hashed = list(audit.get("store", ())), list(audit.get("hash", ()))
    props = set(record["args"]["properties"])
    if set(store) & set(hashed):
        raise ToolSpecError(name, "audit", "a field is either stored or hashed, not both")
    missing = props - set(store) - set(hashed)
    if missing:
        raise ToolSpecError(name, "audit", f"says nothing about {sorted(missing)}")
    unknown = (set(store) | set(hashed)) - props
    if unknown:
        raise ToolSpecError(name, "audit", f"names fields that are not arguments: {sorted(unknown)}")
