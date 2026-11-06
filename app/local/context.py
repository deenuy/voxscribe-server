"""Context header injected into the system prompt on every reply turn.

build_context_header(now, cfg) returns a single line such as:
    Today is Saturday 17 October 2026, 8:48 pm, Toronto (America/Toronto). Mode: default. User: Deenu.

The caller passes datetime.now(tz) where tz comes from cfg.vox_timezone.
Time is read from the OS at the start of each turn, not once per session.
"""

from __future__ import annotations

from datetime import datetime


def _local_now(tz_name: str) -> datetime:
    """Return the current local time in the given IANA timezone."""
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo(tz_name))
    except Exception:
        from datetime import UTC
        return datetime.now(UTC)


def build_context_header(now: datetime, cfg) -> str:
    """Return a single-line context string for the given moment and config.

    Args:
        now: Current datetime (timezone-aware, in the local timezone).
        cfg: Config object with vox_timezone, vox_location, vox_user_name, voice_mode.
    """
    tz_name = getattr(cfg, "vox_timezone", "UTC") or "UTC"
    location = getattr(cfg, "vox_location", "") or ""
    user_name = getattr(cfg, "vox_user_name", "") or ""
    mode = getattr(cfg, "voice_mode", "default") or "default"

    # Format: "Today is Saturday 17 October 2026, 8:48 pm, Toronto (America/Toronto). Mode: default. User: Deenu."
    date_part = now.strftime("%-d %B %Y")           # 17 October 2026
    weekday   = now.strftime("%A")                   # Saturday
    time_part = now.strftime("%-I:%M %p").lower()   # 8:48 pm

    location_clause = ""
    if location:
        location_clause = f", {location} ({tz_name})"
    elif tz_name != "UTC":
        location_clause = f" ({tz_name})"

    parts = [f"Today is {weekday} {date_part}, {time_part}{location_clause}."]
    parts.append(f"Mode: {mode}.")
    if user_name:
        parts.append(f"User: {user_name}.")

    return " ".join(parts)


def current_context_header(cfg) -> str:
    """Convenience: build the header using the current OS clock in cfg.vox_timezone."""
    tz_name = getattr(cfg, "vox_timezone", "UTC") or "UTC"
    return build_context_header(_local_now(tz_name), cfg)
