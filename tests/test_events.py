import asyncio

import pytest

from app.local import events


@pytest.mark.asyncio
async def test_publish_reaches_all_subscribers_and_cleans_up():
    async with events.subscribe() as a, events.subscribe() as b:
        events.publish("source.update", {"id": "s1"})
        assert (await asyncio.wait_for(a.get(), 1)) == {"type": "source.update", "data": {"id": "s1"}}
        assert (await asyncio.wait_for(b.get(), 1))["data"] == {"id": "s1"}
        assert events.subscriber_count() == 2
    assert events.subscriber_count() == 0


@pytest.mark.asyncio
async def test_slow_subscriber_drops_oldest_instead_of_blocking():
    async with events.subscribe() as q:
        for i in range(events._MAX_QUEUE + 10):
            events.publish("model.progress", {"n": i})
        assert q.qsize() == events._MAX_QUEUE
        assert (await q.get())["data"]["n"] == 10
