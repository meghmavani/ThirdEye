"""FastAPI + WebSocket server and the browser UI (FR-UI-6, FR-UI-7, IF-WS-1, IF-WS-2).

    python -m thirdeye.server.app [--port 8000]

Serves one page (`static/index.html`): procedure selection and start/stop (FR-UI-6),
then the checklist or the live legality panel, the lattice or piece grid, and the alerts
(FR-UI-1..3) as they arrive over `/ws`. The socket is outbound-only (IF-WS-1); the
message schema is `messages.py`.

Two kinds of session feed it:

*A scripted replay* started from the page (`POST /api/session`) -- the full
tracker-and-checker path with no hardware, paced so the page can be watched. This is
the dry run of the demo, and what the tests drive.

*A live loop* (`thirdeye.live_jigsaw --serve PORT`, `thirdeye.live_jenga --serve PORT`)
that reads its cameras in its own process (FR-UI-4) and pushes state and alerts through
a `Publisher`; only overlays and alerts ever cross the socket, never video.
"""

from __future__ import annotations

import argparse
import asyncio
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel

from thirdeye.explain.report import plain_report
from thirdeye.replay import Replay
from thirdeye.server.hub import Hub
from thirdeye.server.messages import alert_message, msg, state_message
from thirdeye.session import Session
from thirdeye.sessionlog import SessionRecord

STATIC = Path(__file__).resolve().parent / "static"
SCENARIOS = Path(__file__).resolve().parents[2] / "scripts" / "scenarios"


class Publisher:
    """What a session loop calls to put its state on the socket.

    `update` after each pump publishes the state and any alerts not yet sent;
    `finish` after `session.finish()` publishes the closing summary and report. The
    loop owns the session; the publisher only reads it.
    """

    def __init__(self, hub: Hub) -> None:
        self.hub = hub
        self._sent = 0

    def start(self, procedure: str, source: str) -> None:
        self.hub.reset()
        self._sent = 0
        self.hub.publish(msg("status", running=True, procedure=procedure, source=source))

    def update(self, session: Session, t: float) -> None:
        for alert in session.alerts[self._sent:]:
            self.hub.publish(alert_message(alert))
        self._sent = len(session.alerts)
        self.hub.publish(state_message(session, t))

    def finish(self, session: Session, t: float, source: str = "") -> None:
        self.update(session, t)
        record = SessionRecord.of(session)
        self.hub.publish(msg("done", summary=session.summary(), report=plain_report(record)))
        self.hub.publish(msg("status", running=False, procedure=session.spec.procedure,
                             source=source))


class StopSession(Exception):
    """Raised inside a replay tick to end it early."""


class StartRequest(BaseModel):
    script: str
    #: Replay speed-up over scene time; 1.0 is real time.
    speed: float = 6.0


class Runner:
    """At most one replay at a time, on a worker thread."""

    def __init__(self, hub: Hub) -> None:
        self.hub = hub
        self.publisher = Publisher(hub)
        self.thread: threading.Thread | None = None
        self._stop = threading.Event()

    @property
    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def start(self, script: Path, speed: float) -> None:
        replay = Replay.load(script)
        self._stop.clear()
        self.publisher.start(replay.session.spec.procedure, script.stem)
        last = [0.0]

        def tick(t: float) -> None:
            if self._stop.is_set():
                raise StopSession
            self.publisher.update(replay.session, t)
            if speed > 0:
                time.sleep(max(t - last[0], 0.0) / speed)
            last[0] = t

        replay.on_tick = tick

        def work() -> None:
            try:
                try:
                    replay.run()
                except StopSession:
                    replay.session.finish(last[0])
                self.publisher.finish(replay.session, last[0], script.stem)
            except Exception as exc:  # a dead session must not leave clients waiting
                self.hub.publish(msg("status", running=False, procedure="", source=script.stem,
                                     error=f"{type(exc).__name__}: {exc}"))
                raise

        self.thread = threading.Thread(target=work, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self.thread is not None:
            self.thread.join(timeout=5)


def scenarios(directory: Path = SCENARIOS) -> list[dict[str, str]]:
    out = []
    for path in sorted(directory.glob("*.yaml")):
        script = yaml.safe_load(path.read_text(encoding="utf-8"))
        out.append({"name": path.stem, "procedure": Path(script["spec"]).stem})
    return out


def create_app(hub: Hub | None = None, scenario_dir: Path = SCENARIOS) -> FastAPI:
    hub = hub or Hub()
    runner = Runner(hub)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        hub.bind(asyncio.get_running_loop())  # the loop the sockets live on
        yield
        runner.stop()

    app = FastAPI(title="ThirdEye", lifespan=lifespan)
    app.state.hub, app.state.runner = hub, runner

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/api/procedures")
    def procedures() -> dict:
        return {"scenarios": scenarios(scenario_dir)}

    @app.get("/api/session")
    def status() -> dict:
        return {**hub.status, "running": runner.running}

    @app.post("/api/session")
    def start(req: StartRequest) -> dict:
        if runner.running:
            raise HTTPException(409, "a session is already running; stop it first")
        path = scenario_dir / f"{req.script}.yaml"
        if not path.is_file() or path.parent.resolve() != scenario_dir.resolve():
            raise HTTPException(404, f"no scenario named {req.script!r}")
        runner.start(path, req.speed)
        return {"started": req.script}

    @app.delete("/api/session")
    def stop() -> dict:
        runner.stop()
        return {"stopped": True}

    @app.websocket("/ws")
    async def ws(socket: WebSocket) -> None:
        await socket.accept()
        await socket.send_json(msg("hello", procedures=scenarios(scenario_dir)))
        queue = hub.subscribe()
        try:
            while True:
                await socket.send_json(await queue.get())
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            hub.unsubscribe(queue)

    return app


def serve_in_thread(port: int = 8000) -> Publisher:
    """Start the server on a background thread and return the publisher a live loop uses."""
    import uvicorn

    hub = Hub()
    app = create_app(hub)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    return Publisher(hub)


def main(argv: list[str] | None = None) -> int:
    import uvicorn

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args(argv)
    uvicorn.run(create_app(), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
