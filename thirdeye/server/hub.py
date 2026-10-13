"""Fan-out of session messages to WebSocket clients, from any thread.

A session runs in a worker thread (a replay, or the live camera loop) while clients
are asyncio tasks, so the hub is the one thread-safe seam between them: `publish` may
be called from anywhere and each subscriber's queue is fed on the event loop.

A late client is not left blind: the hub keeps the latest `status`, the latest `state`
and every alert so far, and replays them on subscribe, so a browser opened mid-game
shows the game as it stands rather than an empty page.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

from thirdeye.server.messages import msg

Message = dict[str, Any]


class Hub:
    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._subs: set[asyncio.Queue[Message]] = set()
        self._lock = threading.Lock()
        self.status: Message = msg("status", running=False, procedure="", source="")
        self.state: Message | None = None
        self.alerts: list[Message] = []

    def bind(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def reset(self) -> None:
        """A new session: forget the last one's state and alerts."""
        with self._lock:
            self.state, self.alerts = None, []

    def publish(self, message: Message) -> None:
        with self._lock:
            kind = message["type"]
            if kind == "status":
                self.status = message
            elif kind == "state":
                self.state = message
            elif kind == "alert":
                self.alerts.append(message)
            subs = list(self._subs)
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        for q in subs:
            loop.call_soon_threadsafe(q.put_nowait, message)

    def subscribe(self) -> asyncio.Queue[Message]:
        q: asyncio.Queue[Message] = asyncio.Queue()
        with self._lock:
            backlog = [self.status, *([self.state] if self.state else []), *self.alerts]
            self._subs.add(q)
        for m in backlog:
            q.put_nowait(m)
        return q

    def unsubscribe(self, q: asyncio.Queue[Message]) -> None:
        with self._lock:
            self._subs.discard(q)
