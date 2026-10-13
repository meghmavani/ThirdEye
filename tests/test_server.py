"""The FastAPI server and its WebSocket schema (FR-UI-6, FR-UI-7, IF-WS-1, IF-WS-2)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from thirdeye.replay import Replay
from thirdeye.server.app import Publisher, create_app
from thirdeye.server.hub import Hub
from thirdeye.server.messages import PROTOCOL_VERSION, overlay, state_message

SCENARIOS = Path(__file__).resolve().parents[1] / "scripts" / "scenarios"


def drain_until(ws, kind: str, limit: int = 400) -> list[dict]:
    seen = []
    for _ in range(limit):
        m = ws.receive_json()
        seen.append(m)
        if m["type"] == kind:
            return seen
    raise AssertionError(f"no {kind!r} message; saw {[m['type'] for m in seen[-5:]]}")


def test_every_message_carries_the_protocol_version_and_the_page_is_served() -> None:
    with TestClient(create_app()) as client:
        assert "ThirdEye" in client.get("/").text
        with client.websocket_connect("/ws") as ws:
            hello = ws.receive_json()
            assert hello["v"] == PROTOCOL_VERSION and hello["type"] == "hello"
            assert any(p["name"] == "jenga_fouls" for p in hello["procedures"])
            assert ws.receive_json()["type"] == "status"


def test_a_scripted_jenga_session_streams_state_alerts_and_a_report() -> None:
    with TestClient(create_app()) as client, client.websocket_connect("/ws") as ws:
        r = client.post("/api/session", json={"script": "jenga_two_handed", "speed": 0})
        assert r.status_code == 200
        msgs = drain_until(ws, "done")
        kinds = [m["type"] for m in msgs]
        assert "state" in kinds and "alert" in kinds
        alert = next(m for m in msgs if m["type"] == "alert")
        assert alert["code"] == "two_handed_move" and alert["rule"] == "c2"
        state = [m for m in msgs if m["type"] == "state"][-1]
        assert state["kind"] == "constraints" and state["checklist"] == []
        assert state["legality"]["rules"] and state["overlay"]["model"] == "tower_lattice"
        assert state["counts"]["alerts"] == 1
        done = msgs[-1]
        assert "two_handed_move" in done["report"] and "jenga_game" in done["summary"]


def test_a_scripted_jigsaw_session_streams_a_checklist_and_the_grid() -> None:
    with TestClient(create_app()) as client, client.websocket_connect("/ws") as ws:
        client.post("/api/session", json={"script": "jigsaw_forest_12_correct", "speed": 0})
        msgs = drain_until(ws, "done")
        state = [m for m in msgs if m["type"] == "state"][-1]
        assert len(state["checklist"]) == 12
        assert all(r["status"] == "done" for r in state["checklist"])
        assert state["overlay"]["model"] == "piece_grid" and len(state["overlay"]["cells"]) == 12
        assert state["legality"] is None and state["outcome"] is None


def test_a_late_client_is_brought_up_to_date() -> None:
    with TestClient(create_app()) as client:
        with client.websocket_connect("/ws") as ws:
            client.post("/api/session", json={"script": "jenga_two_handed", "speed": 0})
            drain_until(ws, "done")
        with client.websocket_connect("/ws") as late:
            late.receive_json()  # hello
            backlog = drain_until(late, "alert")
            assert {m["type"] for m in backlog} >= {"status", "state", "alert"}


def test_two_sessions_at_once_are_refused_and_an_unknown_script_is_a_404() -> None:
    with TestClient(create_app()) as client:
        assert client.post("/api/session", json={"script": "nope"}).status_code == 404
        assert client.post("/api/session", json={"script": "../pyproject"}).status_code == 404
        client.post("/api/session", json={"script": "jenga_clean", "speed": 0.5})
        assert client.post("/api/session", json={"script": "jenga_clean"}).status_code == 409
        client.delete("/api/session")
        deadline = time.time() + 5
        while client.get("/api/session").json()["running"] and time.time() < deadline:
            time.sleep(0.05)
        assert not client.get("/api/session").json()["running"]


def test_state_message_for_a_rulebook_has_a_legality_panel_and_no_checklist() -> None:
    replay = Replay.load(SCENARIOS / "jenga_clean.yaml")
    replay.run()
    m = state_message(replay.session, 9.0)
    assert m["legality"]["top_complete_layer"] >= 17 and m["checklist"] == []
    assert overlay(replay.session)["rows"][0] == [1, 1, 1]


def test_the_publisher_sends_each_alert_once() -> None:
    hub = Hub()
    seen: list[dict] = []
    hub.publish = seen.append  # type: ignore[method-assign]
    pub = Publisher(hub)
    replay = Replay.load(SCENARIOS / "jenga_two_handed.yaml")
    replay.run()
    pub.update(replay.session, 1.0)
    pub.update(replay.session, 2.0)
    assert sum(m["type"] == "alert" for m in seen) == 1


@pytest.mark.parametrize("script", ["model_a_correct", "model_a_swap"])
def test_the_lego_path_streams_too(script: str) -> None:
    with TestClient(create_app()) as client, client.websocket_connect("/ws") as ws:
        client.post("/api/session", json={"script": script, "speed": 0})
        msgs = drain_until(ws, "done")
        assert [m for m in msgs if m["type"] == "state"][-1]["overlay"]["model"] == "stud_grid"
