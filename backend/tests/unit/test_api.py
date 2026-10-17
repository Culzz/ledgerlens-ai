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
        assert body["failure"] is None
        assert body["provider"] == "scripted"
        assert body["model"] == "scripted-model"
        assert body["repair_count"] == 0

    def test_the_audit_trail_says_where_each_figure_came_from(self, client):
        body = client.post(
            "/api/ask",
            json={"document_id": DOCUMENT, "question": "What was the change in operating income?"},
        ).json()

        audit = body["audit"]
        assert audit["verdict"] == "grounded"
        assert audit["labels"] == ["table_1"]
        assert audit["unsourced"] == []
        assert audit["steps"] == [
            {
                "index": 0,
                "operation": "subtract",
                "result": "30",
                "render": "#0 = subtract(150 [table_1], 120 [table_1]) = 30",
                "operands": [
                    {
                        "text": "150 [table_1]",
                        "role": "sourced",
                        "value": "150",
                        "labels": ["table_1"],
                    },
                    {
                        "text": "120 [table_1]",
                        "role": "sourced",
                        "value": "120",
                        "labels": ["table_1"],
                    },
                ],
            }
        ]

    def test_an_invented_figure_is_refused_and_says_so(self, client, monkeypatch):
        # None of these figures is in the report - not 150, not the table. The
        # arithmetic is fine and every input is made up, which is the one failure
        # the engine alone cannot see.
        monkeypatch.setattr(
            ask_routes,
            "get_orchestrator",
            lambda: Orchestrator(ScriptedProvider("subtract(998, 999)"), max_repairs=0),
        )

        body = client.post(
            "/api/ask",
            json={"document_id": DOCUMENT, "question": "What was the change in operating income?"},
        ).json()

        assert body["answer"] is None
        assert body["failure"].startswith("UngroundedAnswer")
        # The trail is still there. It is the reason there is no answer, so hiding
        # it would hide the explanation.
        assert body["audit"]["verdict"] == "ungrounded"
        assert body["audit"]["unsourced"] == ["998", "999"]

    def test_a_figure_the_report_does_not_state_alone_does_not_refuse_the_answer(
        self, client, monkeypatch
    ):
        # 999 is nowhere in the report and 150 is in it. One unreadable figure is a
        # weaker answer rather than an invented one, so the answer stands and the
        # trail says which figure was the weak point.
        monkeypatch.setattr(
            ask_routes,
            "get_orchestrator",
            lambda: Orchestrator(ScriptedProvider("subtract(150, 999)"), max_repairs=0),
        )

        body = client.post(
            "/api/ask",
            json={"document_id": DOCUMENT, "question": "What was the change in operating income?"},
        ).json()

        assert body["answer"] == "-849"
        assert body["audit"]["verdict"] == "partly_grounded"
        assert body["audit"]["unsourced"] == ["999"]

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


class TestRecompute:
    """The same program with a figure changed, answered without a model.

    This is the other half of the answer being a program rather than a sentence. A
    follow-up about a number is a change of input, and the engine answers it in the
    time it takes to do the arithmetic - no provider, no tokens, no waiting.
    """

    def edit(self, client, program, edits, document=DOCUMENT):
        return client.post(
            "/api/recompute",
            json={"document_id": document, "program": program, "edits": edits},
        )

    def test_changing_a_figure_reruns_the_program(self, client):
        response = self.edit(
            client, "subtract(150, 120)", [{"step": 0, "operand": 0, "value": "200"}]
        )

        assert response.status_code == 200
        body = response.json()
        assert body["program"] == "subtract(200, 120)"
        assert body["answer"] == "80"
        assert body["failure"] is None

    def test_it_does_not_ask_the_model(self, client, monkeypatch):
        # The client is wired to a provider that answers subtract(150, 120). This is a
        # different program and the answer follows the program, which it only can if
        # no model was asked. Belt and braces: the orchestrator factory is replaced
        # with something that fails loudly.
        def refuse():
            raise AssertionError("recompute must not reach the model path")

        monkeypatch.setattr(ask_routes, "get_orchestrator", refuse)

        body = self.edit(
            client, "multiply(150, 2)", [{"step": 0, "operand": 0, "value": "160"}]
        ).json()

        assert body["answer"] == "320"

    def test_a_figure_the_report_does_not_state_is_computed_anyway(self, client):
        # A what-if has changed a figure the report states, so the new figure is not
        # in the report by definition. The verdict says so and the answer still
        # stands. Refusing it would make a what-if impossible; returning it without
        # the verdict would let it look grounded.
        body = self.edit(
            client, "multiply(150, 2)", [{"step": 0, "operand": 0, "value": "200"}]
        ).json()

        assert body["answer"] == "400"
        assert body["audit"]["verdict"] == "ungrounded"
        assert "200" in body["audit"]["unsourced"]

    def test_the_figures_left_alone_keep_their_citation(self, client):
        body = self.edit(
            client, "subtract(150, 120)", [{"step": 0, "operand": 0, "value": "200"}]
        ).json()

        assert body["audit"]["verdict"] == "partly_grounded"
        operands = [o for step in body["audit"]["steps"] for o in step["operands"]]
        changed = next(o for o in operands if o["text"].startswith("200"))
        untouched = next(o for o in operands if o["text"].startswith("120"))
        assert changed["role"] == "unsourced"
        assert changed["labels"] == []
        assert untouched["role"] == "sourced"
        assert untouched["labels"]

    def test_the_trail_shows_the_program_that_ran(self, client):
        body = self.edit(
            client, "subtract(150, 120)", [{"step": 0, "operand": 0, "value": "200"}]
        ).json()

        render = body["audit"]["steps"][0]["render"]
        # The step is the edited one, the figure that came from the report keeps its
        # citation, and the figure that did not is marked as such on the same line.
        assert "subtract(" in render
        assert "200 (unverified)" in render
        assert "120 [table_1]" in render

    def test_a_thousands_separator_is_rewritten_as_a_figure(self, client):
        body = self.edit(
            client, "subtract(150, 120)", [{"step": 0, "operand": 0, "value": "1,234"}]
        ).json()

        assert body["program"] == "subtract(1234, 120)"
        assert body["answer"] == "1114"

    def test_an_accounting_negative_is_understood(self, client):
        body = self.edit(
            client, "add(150, 120)", [{"step": 0, "operand": 1, "value": "(500)"}]
        ).json()

        assert body["program"] == "add(150, -500)"
        assert body["answer"] == "-350"

    def test_two_figures_at_once(self, client):
        body = self.edit(
            client,
            "subtract(150, 120)",
            [
                {"step": 0, "operand": 0, "value": "1000"},
                {"step": 0, "operand": 1, "value": "250"},
            ],
        ).json()

        assert body["program"] == "subtract(1000, 250)"
        assert body["answer"] == "750"

    def test_a_label_cannot_be_smuggled_in_as_a_figure(self, client):
        # The value is parsed and printed again rather than pasted into the program,
        # so nothing sent here arrives as program syntax.
        response = self.edit(
            client, "subtract(150, 120)", [{"step": 0, "operand": 0, "value": "Revenue"}]
        )

        assert response.status_code == 422
        assert "cannot parse" in response.json()["detail"]

    def test_a_reference_cannot_be_smuggled_in_as_a_figure(self, client):
        response = self.edit(
            client, "subtract(150, 120)", [{"step": 0, "operand": 0, "value": "#0"}]
        )

        assert response.status_code == 422

    def test_a_position_the_program_does_not_have_is_refused(self, client):
        response = self.edit(
            client, "subtract(150, 120)", [{"step": 4, "operand": 0, "value": "1"}]
        )

        assert response.status_code == 422
        assert "no step 4" in response.json()["detail"]

    def test_a_row_selection_has_no_figure_to_change(self, client):
        response = self.edit(
            client,
            "table_sum(Operating income, none)",
            [{"step": 0, "operand": 0, "value": "1"}],
        )

        assert response.status_code == 422
        assert "selects a row by its label" in response.json()["detail"]

    def test_a_program_that_will_not_parse_is_refused(self, client):
        response = self.edit(client, "subtract(150 120)", [{"step": 0, "operand": 0, "value": "1"}])

        assert response.status_code == 422

    def test_a_what_if_with_no_answer_says_so_rather_than_failing(self, client):
        # Setting a divisor to zero is a fair thing to try. It comes back as a failure
        # against the program that actually ran, rather than as a bad request.
        response = self.edit(client, "divide(150, 120)", [{"step": 0, "operand": 1, "value": "0"}])

        assert response.status_code == 200
        body = response.json()
        assert body["program"] == "divide(150, 0)"
        assert body["answer"] is None
        assert "division by zero" in body["failure"]

    def test_an_unknown_report_is_a_404(self, client):
        response = self.edit(
            client,
            "subtract(150, 120)",
            [{"step": 0, "operand": 0, "value": "1"}],
            document="not-a-report",
        )

        assert response.status_code == 404

    def test_editing_nothing_is_refused(self, client):
        assert self.edit(client, "subtract(150, 120)", []).status_code == 422


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
