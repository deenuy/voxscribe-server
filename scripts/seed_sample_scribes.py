"""Put two sample scribes (and the projects that hold them) into the development database, or take them out again.

    uv run python scripts/seed_sample_scribes.py            # add them (safe to repeat: nothing is added twice)
    uv run python scripts/seed_sample_scribes.py --remove   # remove exactly what this script added

They simulate what the app stores after a recording of each short video is stopped and published: a transcript, shorthand notes
and a summary. The words are written for demonstration, on the topic of each video. They are NOT the videos' own transcripts,
and every title and context line says so. Every id starts with `sample_`, which is how --remove finds them.
The database is the engine's configured one (VOXSCRIBE_DATABASE_URL, default the local development database).
"""
import asyncio
import os
import sys
from datetime import UTC, datetime

sys.path.insert(0, os.getcwd())
from app.local.config import Config  # noqa: E402
from app.local.models import Block, Segment  # noqa: E402
from app.local.store import Store  # noqa: E402
from app.local.store_ideas import ProjectStore  # noqa: E402
from app.local.store_meetings import MeetingStore  # noqa: E402

NOTE = "Sample data for demonstration. Written on the topic of the video, not its transcript."

SAMPLES = [
    {
        "project": ("sample_proj_agents", "AI agents", "Notes on building and running AI agents"),
        "id": "sample_scribe_agent_memory",
        "title": "Sample: Giving an AI agent perfect memory",
        "context": f"Simulates a recording of the short video 'I gave my Hermes Agent perfect memory' (https://youtube.com/shorts/202DccqHiv0). {NOTE}",
        "lines": [
            "Most agents forget everything the moment a conversation ends, and that is the real problem to solve.",
            "So the first thing I did was split memory into two parts: what happened in this session, and what should last.",
            "Every session ends with the agent writing a short summary of the decisions it made and the facts it learned.",
            "Those summaries go into a store the agent can search, so it can recall a fact from last week by meaning, not by exact words.",
            "The trick is to save less, not more. If you store every message, retrieval returns noise and the agent gets confused.",
            "I keep a small file of stable facts, like my preferences and my projects, and load it at the start of every run.",
            "When the context window fills up, the agent compacts the older turns into a summary instead of dropping them.",
            "The result feels like talking to someone who remembers you. It picks up where we stopped without being reminded.",
            "One more rule: memory must be editable. When a fact changes, the agent updates it instead of piling up contradictions.",
        ],
        "notes": ["Two kinds of memory: this session, and what should last", "Save less, retrieve better", "Stable facts file loaded at start", "Compact old turns instead of dropping them", "Memory must be editable"],
        "overview": "A walkthrough of giving an AI agent durable memory by separating session context from long-term facts, saving short summaries instead of whole conversations, and keeping the memory editable.",
        "details": [
            ("Two layers of memory", "Session context lives for one run. Facts worth keeping are written to a searchable store at the end of each session."),
            ("Save less, retrieve better", "Storing every message makes retrieval noisy. Short summaries of decisions and learned facts recall well by meaning."),
            ("Stable facts and compaction", "A small file of stable preferences loads on every run, and old turns are compacted into a summary when the window fills."),
            ("Keep it editable", "When a fact changes the agent updates the record, so memory does not collect contradictions."),
        ],
    },
    {
        "project": ("sample_proj_architecture", "System design", "Patterns and trade-offs for building software that scales"),
        "id": "sample_scribe_design_patterns",
        "title": "Sample: Six system design patterns every developer needs",
        "context": f"Simulates a recording of the short video '6 System Design Patterns Every Dev Needs' (https://youtube.com/shorts/SHrSDgsVJUE). {NOTE}",
        "lines": [
            "If you build anything that has to scale, six patterns will carry you a long way.",
            "First, caching. Keep hot data close to the user so most requests never reach the database.",
            "Second, load balancing. Spread requests across several servers so no single machine is the bottleneck or the single point of failure.",
            "Third, sharding. When one database is too big, split the data by a key so each piece stays small and fast.",
            "Fourth, a message queue. Let slow work happen in the background so the user is not left waiting.",
            "Fifth, the circuit breaker. When a service keeps failing, stop calling it for a while so one failure does not take down everything.",
            "Sixth, CQRS. Separate the model you write to from the model you read from, so each can be shaped for its job.",
            "None of these is free. Each one trades simplicity for scale, so add them when the numbers ask for them, not before.",
        ],
        "notes": ["1 Caching", "2 Load balancing", "3 Sharding", "4 Message queue", "5 Circuit breaker", "6 CQRS", "Each trades simplicity for scale"],
        "overview": "Six patterns for systems that have to scale (caching, load balancing, sharding, message queues, circuit breakers and CQRS), with the reminder that each adds complexity and should be added when the load requires it.",
        "details": [
            ("Serving more users", "Caching keeps hot data near the user. Load balancing spreads requests so no one server is the bottleneck."),
            ("Handling more data", "Sharding splits a large database by a key. CQRS separates the write model from the read model."),
            ("Staying up under failure", "A message queue moves slow work to the background. A circuit breaker stops calls to a failing service."),
            ("Trade-offs", "Every pattern costs simplicity. Add one when measured load asks for it."),
        ],
    },
]


def segments(lines):
    out, t = [], 0
    for i, text in enumerate(lines):
        dur = max(2500, len(text.split()) * 380)
        out.append(Segment(id=f"{{}}_seg_{i + 1:03d}", startMs=t, endMs=t + dur, source="mic", speaker="Speaker 1", text=text, state="stable"))
        t += dur + 400
    return out


async def main(remove: bool):
    store = await Store.create(Config().database_url)
    meetings, projects = MeetingStore(store.pool), ProjectStore(store.pool)
    try:
        for s in SAMPLES:
            pid, name, desc = s["project"]
            if remove:
                await meetings.delete_meeting(s["id"])
                try:
                    await projects.delete_project(pid)
                except Exception:
                    pass
                print("removed", s["id"])
                continue
            try:
                await projects.get_project(pid)
            except Exception:
                await projects.create_project(pid, name, desc)
            try:
                await meetings.get_meeting(s["id"])
                print("already there", s["id"])
                continue
            except Exception:
                pass
            await meetings.create_meeting(s["id"], title=s["title"], context=s["context"], project_id=pid)
            await meetings.start_meeting(s["id"])
            for seg in segments(s["lines"]):
                await meetings.append_segment(s["id"], seg.model_copy(update={"id": seg.id.format(s["id"])}))
            await meetings.put_notes(s["id"], [Block(blockId=f"{s['id']}_note_{i + 1}", text=t, offsetMs=i * 12000) for i, t in enumerate(s["notes"])])
            await meetings.mark_done(s["id"])
            await meetings.set_summary(s["id"], s["overview"], [{"title": t, "body": b} for t, b in s["details"]], datetime.now(UTC).isoformat().replace("+00:00", "Z"))
            print("added", s["id"], f"({len(s['lines'])} segments)")
    finally:
        await store.pool.close()


asyncio.run(main("--remove" in sys.argv))
