"""Saved briefs.

The interesting assertions are all the same assertion from different sides: what
comes back on a brief was computed here, from the programs, and not carried over
from whatever the caller had on screen. The caller has nowhere to put a figure, and
the fixture's provider is a different one from the program being sent, so an answer
that matched the browser rather than the program would be visible immediately.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from app.api import deps
from app.api.routes import briefs as brief_routes
from app.core.config import get_settings
from app.main import create_app
from app.services.llm import Completion
from app.services.reasoning import Orchestrator
from fastapi.testclient import TestClient

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "finqa_sample.json"
DOCUMENT = "sample-2023-page-10-pdf"
QUESTION = "What was the change in operating income?"


class ScriptedProvider:
    """A provider that would answer with a program of its own, if asked."""

    name = "scripted"
    model = "scripted-model"

    def complete(self, messages, *, temperature: float = 0.0, max_tokens=None) -> Completion:
        return Completion(
            text='{"reasoning": "because", "program": "subtract(150, 120)", "evidence": []}',
            model=self.model,
            provider=self.name,
        )


class ExplodingProvider:
    """A provider that fails the test if a saved brief asks it anything."""

    name = "exploding"
    model = "none"

    def complete(self, *args, **kwargs):
        raise AssertionError("saving a brief called a model")


class MemoryStore:
    """The repository, in a dictionary.

    Stands in for Postgres so these tests are about the route and the replay path.
    The real queries are exercised in tests/integration.
    """

    def __init__(self) -> None:
        self.briefs: dict[str, dict] = {}
        self.unavailable = False

    def save(self, document_id: str, filename: str, rows: list[dict]) -> str:
        if self.unavailable:
            raise brief_routes.store.BriefsUnavailable("the brief store is not reachable")
        brief_id = f"brief{len(self.briefs) + 1:04d}"
        self.briefs[brief_id] = {
            "id": brief_id,
            "document_id": document_id,
            "filename": filename,
            "rows": rows,
            "created_at": "2026-10-20T09:00:00+00:00",
        }
        return brief_id

    def load(self, brief_id: str) -> dict | None:
        if self.unavailable:
            raise brief_routes.store.BriefsUnavailable("the brief store is not reachable")
        return self.briefs.get(brief_id)


@pytest.fixture
def store(monkeypatch):
    fake = MemoryStore()
    monkeypatch.setattr(brief_routes.store, "save", fake.save)
    monkeypatch.setattr(brief_routes.store, "load", fake.load)
    return fake


@pytest.fixture
def client(tmp_path, monkeypatch, store):
    data = tmp_path / "finqa"
    data.mkdir()
    shutil.copy(FIXTURE, data / "dev.json")

    monkeypatch.setenv("FINQA_DATA_DIR", str(data))
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.setattr(brief_routes, "get_orchestrator", lambda: Orchestrator(ScriptedProvider()))

    get_settings.cache_clear()
    deps.get_library.cache_clear()
    deps.get_provider.cache_clear()
    deps.get_orchestrator.cache_clear()

    with TestClient(create_app()) as test_client:
        yield test_client

    get_settings.cache_clear()
    deps.get_library.cache_clear()
    deps.get_provider.cache_clear()
    deps.get_orchestrator.cache_clear()


def rows(*programs: str | None) -> list[dict]:
    return [{"question": QUESTION, "program": program} for program in programs]


class TestSave:
    def test_a_brief_comes_back_with_what_the_programs_produce(self, client):
        response = client.post(
            "/api/briefs",
            json={"document_id": DOCUMENT, "rows": rows("subtract(150, 120)")},
        )

        assert response.status_code == 201
        body = response.json()
        assert body["id"]
        assert body["filename"] == "SAMPLE/2023/page_10.pdf"
        row = body["rows"][0]
        assert row["answer"] == "30"
        assert row["verdict"] == "grounded"
        assert row["labels"] == ["table_1"]
        assert row["note"] is None

    def test_the_figure_follows_the_program_in_the_request(self, client):
        # The fixture's provider answers subtract(150, 120). This program adds, and
        # 270 is what comes back, which it can only be if the program was run.
        body = client.post(
            "/api/briefs",
            json={"document_id": DOCUMENT, "rows": rows("add(150, 120)")},
        ).json()

        assert body["rows"][0]["answer"] == "270"

    def test_an_invented_figure_is_refused_on_the_saved_page_too(self, client):
        body = client.post(
            "/api/briefs",
            json={"document_id": DOCUMENT, "rows": rows("subtract(998, 999)")},
        ).json()

        row = body["rows"][0]
        assert row["answer"] is None
        assert row["verdict"] == "ungrounded"
        assert row["unsourced"] == ["998", "999"]
        assert row["note"].startswith("UngroundedAnswer")

    def test_a_partly_grounded_answer_keeps_its_figure_and_its_warning(self, client):
        body = client.post(
            "/api/briefs",
            json={"document_id": DOCUMENT, "rows": rows("subtract(150, 999)")},
        ).json()

        row = body["rows"][0]
        assert row["answer"] == "-849"
        assert row["verdict"] == "partly_grounded"
        assert row["unsourced"] == ["999"]

    def test_a_line_with_no_program_is_kept_rather_than_dropped(self, client):
        # The model sometimes returns nothing runnable. Leaving that line out of the
        # saved brief would make the session look better than it was.
        body = client.post(
            "/api/briefs",
            json={"document_id": DOCUMENT, "rows": rows(None)},
        ).json()

        row = body["rows"][0]
        assert row["program"] is None
        assert row["answer"] is None
        assert row["verdict"] is None
        assert row["note"] == "the model returned no program for this question"

    def test_it_does_not_ask_the_model(self, client, monkeypatch):
        monkeypatch.setattr(
            brief_routes, "get_orchestrator", lambda: Orchestrator(ExplodingProvider())
        )

        response = client.post(
            "/api/briefs",
            json={"document_id": DOCUMENT, "rows": rows("subtract(150, 120)")},
        )

        assert response.status_code == 201
        assert response.json()["rows"][0]["answer"] == "30"

    def test_an_unknown_report_is_a_404(self, client):
        response = client.post(
            "/api/briefs",
            json={"document_id": "not-a-report", "rows": rows("subtract(150, 120)")},
        )

        assert response.status_code == 404

    def test_a_brief_with_no_lines_is_refused(self, client):
        response = client.post("/api/briefs", json={"document_id": DOCUMENT, "rows": []})

        assert response.status_code == 422

    def test_a_brief_longer_than_the_limit_is_refused(self, client):
        response = client.post(
            "/api/briefs",
            json={"document_id": DOCUMENT, "rows": rows(*(["subtract(150, 120)"] * 41))},
        )

        assert response.status_code == 422

    def test_a_store_that_is_not_configured_says_so(self, client, store):
        store.unavailable = True

        response = client.post(
            "/api/briefs",
            json={"document_id": DOCUMENT, "rows": rows("subtract(150, 120)")},
        )

        assert response.status_code == 503
        assert "not reachable" in response.json()["detail"]


class TestRead:
    def test_a_saved_brief_can_be_read_back_by_its_id(self, client):
        saved = client.post(
            "/api/briefs",
            json={"document_id": DOCUMENT, "rows": rows("subtract(150, 120)", "add(150, 120)")},
        ).json()

        body = client.get(f"/api/briefs/{saved['id']}").json()

        assert body == saved
        assert [row["answer"] for row in body["rows"]] == ["30", "270"]

    def test_an_unknown_id_is_a_404(self, client):
        assert client.get("/api/briefs/nosuchbrief1").status_code == 404

    def test_an_id_that_is_not_an_id_is_refused_before_the_store(self, client):
        # Ids are minted here as URL-safe strings with no separators in them, so
        # anything else is rejected by the path pattern rather than looked up.
        assert client.get("/api/briefs/short").status_code == 422
        assert client.get("/api/briefs/" + "a" * 65).status_code == 422
        assert client.get("/api/briefs/not%20an%20id").status_code == 422

    def test_a_store_that_is_down_is_a_503_not_a_500(self, client, store):
        saved = client.post(
            "/api/briefs",
            json={"document_id": DOCUMENT, "rows": rows("subtract(150, 120)")},
        ).json()
        store.unavailable = True

        response = client.get(f"/api/briefs/{saved['id']}")

        assert response.status_code == 503
