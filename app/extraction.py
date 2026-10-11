"""Idea extraction: turn a transcript into a canonical Idea document.

This is the LLM processing the backend exists for. Two backends:

  EXTRACTOR=simulated  deterministic, zero deps, no API key. Builds a plausible
                       Idea with rubric scores that sum to statusScore. This is
                       the Phase 1 default and matches the client _buildSimulatedIdea.
  EXTRACTOR=ollama     local model via Ollama, produces real structured JSON
  EXTRACTOR=claude     Anthropic API, highest quality (needs ANTHROPIC_API_KEY)

Both return a dict in the exact canonical camelCase shape.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request

from .vocab import (
    IDEA_TYPES,
    RUBRIC_MAX,
    display_time,
    now_iso,
    seeded_fraction,
    status_for_score,
)

EXTRACTOR = os.environ.get("EXTRACTOR", "simulated")


def extract_idea(
    *,
    idea_id: str,
    tenant_id: str,
    workspace_id: str,
    project_id: str | None,
    session_id: str,
    session_title: str,
    source_type: str,
    source_ref: str,
    participants: int,
    context: str,
    transcript: list[dict],
) -> dict:
    if EXTRACTOR in ("ollama", "claude"):
        try:
            return _llm_extract(
                idea_id=idea_id,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                project_id=project_id,
                session_id=session_id,
                session_title=session_title,
                source_type=source_type,
                source_ref=source_ref,
                transcript=transcript,
            )
        except Exception:
            # Fall back to simulated so a session never fails to produce an idea.
            pass
    return _simulated(
        idea_id=idea_id,
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        project_id=project_id,
        session_id=session_id,
        session_title=session_title,
        source_type=source_type,
        source_ref=source_ref,
        participants=participants,
        context=context,
        transcript=transcript,
    )


def _rubric_for(seed: str) -> tuple[dict, int]:
    scores = {dim: round(mx * seeded_fraction(seed, dim)) for dim, mx in RUBRIC_MAX.items()}
    return scores, sum(scores.values())


def _simulated(**kw) -> dict:
    seed = kw["session_title"] + kw["idea_id"]
    rubric, total = _rubric_for(seed)
    full_text = " ".join(seg["text"] for seg in kw["transcript"]).strip()
    type_idx = int(seeded_fraction(seed, "type") * len(IDEA_TYPES)) % len(IDEA_TYPES)
    status = status_for_score(total)
    evidence = [
        {"time": seg.get("start", 0.0), "speaker": seg.get("speaker", "SPEAKER_0"), "text": seg["text"]}
        for seg in kw["transcript"][:5]
    ]
    next_step = {
        "raw": "Run a discovery interview to define the problem and persona.",
        "needs_discovery": "Validate the problem with two customer conversations.",
        "promising": "Draft a one-page solution sketch and size the value.",
        "prd_candidate": "Generate the PRD and review feasibility with engineering.",
        "delivery_ready": "Hand off to delivery and open the implementation epic.",
    }[status]
    return {
        "id": kw["idea_id"],
        "tenantId": kw["tenant_id"],
        "workspaceId": kw["workspace_id"],
        "projectId": kw["project_id"],
        "sourceType": kw["source_type"],
        "sourceRef": kw["source_ref"],
        "sessionId": kw["session_id"],
        "sessionTitle": kw["session_title"],
        "title": kw["session_title"],
        "type": IDEA_TYPES[type_idx],
        "description": full_text[:600] or "No transcript content captured.",
        "customer": "",
        "problem": "",
        "currentWorkaround": "",
        "businessValue": "",
        "statusScore": total,
        "rubricScores": rubric,
        "evidence": evidence,
        "challenges": [
            {"category": "discovery", "type": "gap", "question": "Who is the primary persona?"}
        ],
        "contradictions": [],
        "assumptions": [],
        "gaps": ["persona", "businessValue"],
        "lineage": {
            "evolvedFrom": [],
            "mergedWith": [],
            "originatedFromSession": kw["session_id"],
            "linkedToFeature": None,
        },
        "originator": "",
        "contributors": [],
        "challengers": [],
        "capturedAt": display_time(),
        "createdAt": now_iso(),
        "recommendedNextStep": next_step,
    }


_SYSTEM = """You are a product management analyst. Extract one structured idea
from the transcript. Return only a JSON object, no preamble, no markdown fences.
Score each rubric dimension honestly and penalize missing data, vague claims,
and unsupported assumptions. The rubric dimensions and their max points are:
problem 15, persona 10, value 10, strategy 10, differentiation 5, feasibility 10,
data_ai 10, scope 10, risk 10, execution 10. statusScore must equal the sum of
the rubric scores. Use these idea types only: %s.""" % ", ".join(IDEA_TYPES)


def _llm_extract(**kw) -> dict:
    transcript_text = "\n".join(
        f"[{seg.get('speaker','SPEAKER_0')}] {seg['text']}" for seg in kw["transcript"]
    )
    prompt = (
        "Transcript:\n" + transcript_text + "\n\n"
        "Return JSON with keys: title, type, description, customer, problem, "
        "currentWorkaround, businessValue, rubricScores (object with the 10 "
        "dimensions), evidence (array of {time, speaker, text}), challenges "
        "(array of {category, type, question}), contradictions, assumptions "
        "(array of strings), gaps (array of strings), recommendedNextStep."
    )
    raw = _call_model(_SYSTEM, prompt)
    parsed = _coerce_json(raw)

    rubric = parsed.get("rubricScores", {})
    rubric = {dim: int(rubric.get(dim, 0)) for dim in RUBRIC_MAX}
    total = sum(rubric.values())
    return {
        "id": kw["idea_id"],
        "tenantId": kw["tenant_id"],
        "workspaceId": kw["workspace_id"],
        "projectId": kw["project_id"],
        "sourceType": kw["source_type"],
        "sourceRef": kw["source_ref"],
        "sessionId": kw["session_id"],
        "sessionTitle": kw["session_title"],
        "title": parsed.get("title", kw["session_title"]),
        "type": parsed.get("type") if parsed.get("type") in IDEA_TYPES else "new_feature",
        "description": parsed.get("description", ""),
        "customer": parsed.get("customer", ""),
        "problem": parsed.get("problem", ""),
        "currentWorkaround": parsed.get("currentWorkaround", ""),
        "businessValue": parsed.get("businessValue", ""),
        "statusScore": total,
        "rubricScores": rubric,
        "evidence": parsed.get("evidence", []),
        "challenges": parsed.get("challenges", []),
        "contradictions": parsed.get("contradictions", []),
        "assumptions": parsed.get("assumptions", []),
        "gaps": parsed.get("gaps", []),
        "lineage": {
            "evolvedFrom": [],
            "mergedWith": [],
            "originatedFromSession": kw["session_id"],
            "linkedToFeature": None,
        },
        "originator": "",
        "contributors": [],
        "challengers": [],
        "capturedAt": display_time(),
        "createdAt": now_iso(),
        "recommendedNextStep": parsed.get("recommendedNextStep", ""),
    }


def _call_model(system: str, prompt: str) -> str:
    if EXTRACTOR == "claude":
        key = os.environ["ANTHROPIC_API_KEY"]
        body = {
            "model": os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6"),
            "max_tokens": 2048,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        }
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(body).encode(),
            headers={
                "Content-Type": "application/json",
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
            },
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read().decode())
        return "".join(b.get("text", "") for b in payload.get("content", []))

    # ollama
    host = os.environ.get("OLLAMA_HOST", "http://host.docker.internal:11434")
    body = {
        "model": os.environ.get("VOXSCRIBE_OLLAMA_MODEL", "gemma3:4b"),
        "system": system,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.1},
    }
    req = urllib.request.Request(
        f"{host.rstrip('/')}/api/generate",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode()).get("response", "")


def _coerce_json(raw: str) -> dict:
    raw = raw.strip()
    raw = re.sub(r"^```(json)?|```$", "", raw).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        return json.loads(match.group(0)) if match else {}
