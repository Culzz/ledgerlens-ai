"""Tests for the HTTP layer.

The one that matters is the last: the reference answer is looked up for display
and never reaches the code that computes the answer. That boundary is easy to
cross by accident and impossible to notice once crossed, so it gets a test of its
own rather than being left to a code review.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from app.api import deps
from app.api.routes import ask as ask_routes
from app.core.config import get_settings
from app.main import create_app
from app.services.llm import Completion
from app.services.reasoning import Orchestrator
from fastapi.testclient import TestClient

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "finqa_sample.json"
DOCUMENT = "sample-2023-page-10-pdf"


class ScriptedProvider:
    """A provider that replies with one canned string."""

    name = "scripted"
    model = "scripted-model"

    def __init__(self, program: str) -> None:
        self.program = program

    def complete(self, messages, *, temperature: float = 0.0, max_tokens=None) -> Completion:
        reply = json.dumps(
            {"reasoning": "because", "program": self.program, "evidence": ["table_1"]}
        )
        return Completion(text=reply, model=self.model, provider=self.name)


@pytest.fixture
def client(tmp_path, monkeypatch):
    data = tmp_path / "finqa"
    data.mkdir()
    shutil.copy(FIXTURE, data / "dev.json")

    monkeypatch.setenv("FINQA_DATA_DIR", str(data))
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.setattr(
        ask_routes, "get_orchestrator", lambda: Orchestrator(ScriptedProvider("subtract(150, 120)"))
    )

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


class TestDocuments:
    def test_health_reports_the_provider_and_the_documents(self, client):
        body = client.get("/api/health").json()

        assert body["status"] == "ok"
        assert body["documents_loaded"] is True
        # The fixture holds six questions across four reports.
        assert body["documents"] == 4

    def test_documents_are_grouped_by_report(self, client):
        body = client.get("/api/documents").json()

        assert body["total"] == 4
        page_10 = next(d for d in body["documents"] if d["id"] == DOCUMENT)
        assert len(page_10["questions"]) == 3
        assert page_10["table_rows"] == 3

    def test_the_filter_narrows_the_list(self, client):
        body = client.get("/api/documents", params={"q": "page_11"}).json()

        assert body["total"] == 1
        assert body["documents"][0]["id"] == "sample-2023-page-11-pdf"

    def test_a_report_carries_its_table_and_narrative(self, client):
        body = client.get(f"/api/documents/{DOCUMENT}").json()

        assert body["filename"] == "SAMPLE/2023/page_10.pdf"
        assert body["table"][0] == ["", "2023", "2022"]
        assert body["table"][1] == ["Operating income", "150", "120"]
        assert len(body["context"]) > 0

    def test_an_unknown_report_is_a_404(self, client):
        assert client.get("/api/documents/not-a-report").status_code == 404


class TestAsk:
    def test_the_answer_is_the_engines(self, client):
        body = client.post(
            "/api/ask",
            json={"document_id": DOCUMENT, "question": "What was the change in operating income?"},
        ).json()

        assert body["answer"] == "30"
        assert body["program"] == "subtract(150, 120)"
        assert body["steps"] == ["#0 = subtract(150, 120) = 30"]
        assert body["failure"] is None
        assert body["provider"] == "scripted"
        assert body["model"] == "scripted-model"

    def test_the_evidence_marks_what_the_answer_rested_on(self, client):
        body = client.post(
            "/api/ask",
            json={"document_id": DOCUMENT, "question": "What was the change in operating income?"},
        ).json()

        assert body["evidence"], "the prompt always carries the table header and the retrieved rows"
        cited = [item for item in body["evidence"] if item["cited"]]
        assert [item["label"] for item in cited] == ["table_1"]

    def test_the_reference_is_display_only(self, client, monkeypatch):
        # The provider computes 900 / 3 = 300 while the annotation for this
        # question says 30. Whatever comes back has to be 300: the reference is
        # looked up afterwards, to show, and never enters the computation.
        monkeypatch.setattr(
            ask_routes, "get_orchestrator", lambda: Orchestrator(ScriptedProvider("divide(900, 3)"))
        )

        body = client.post(
            "/api/ask",
            json={
                "document_id": DOCUMENT,
                "question": "What was the change in operating income from 2022 to 2023?",
            },
        ).json()

        assert body["reference_answer"] == "30"
        assert body["answer"] == "300"

    def test_a_question_that_came_with_the_report_shows_its_reference(self, client):
        body = client.post(
            "/api/ask",
            json={
                "document_id": DOCUMENT,
                "question": "What was the change in operating income from 2022 to 2023?",
            },
        ).json()

        assert body["reference_answer"] == "30"

    def test_a_question_typed_by_hand_has_no_reference(self, client):
        body = client.post(
            "/api/ask",
            json={"document_id": DOCUMENT, "question": "How many rows does this table have?"},
        ).json()

        assert body["reference_answer"] is None

    def test_an_unknown_report_is_a_404(self, client):
        response = client.post(
            "/api/ask", json={"document_id": "not-a-report", "question": "anything"}
        )
        assert response.status_code == 404

    def test_an_empty_question_is_rejected(self, client):
        response = client.post("/api/ask", json={"document_id": DOCUMENT, "question": ""})
        assert response.status_code == 422


class TestWithoutDocuments:
    def test_a_missing_split_says_how_to_get_it(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FINQA_DATA_DIR", str(tmp_path / "empty"))
        get_settings.cache_clear()
        deps.get_library.cache_clear()
        deps.get_provider.cache_clear()

        with TestClient(create_app()) as test_client:
            assert test_client.get("/api/health").json()["documents_loaded"] is False

            response = test_client.get("/api/documents")
            assert response.status_code == 503
            assert "download_data" in response.json()["detail"]

        get_settings.cache_clear()
        deps.get_library.cache_clear()
        deps.get_provider.cache_clear()
