"""Tests for the LLM adapter.

Everything here runs offline. The HTTP tests go through httpx's MockTransport,
so retries, backoff and error handling are exercised without a network, a key or
a cent of spend.
"""

from __future__ import annotations

import json

import httpx
import pytest
from app.core.config import Settings
from app.services.llm import (
    COMPATIBLE_ENDPOINTS,
    MODEL_LIST_HINTS,
    BudgetExceeded,
    BudgetLedger,
    Connection,
    FakeLLM,
    InvalidResponse,
    Message,
    OpenAICompatibleProvider,
    ProviderNotConfigured,
    ProviderRateLimited,
    ProviderRequestFailed,
    ProviderUnavailable,
    RateLimiter,
    ScriptExhausted,
    Usage,
    build_provider,
    estimate_cost,
    mask_key,
    price_for,
    resolve_connection,
    system,
    user,
)
from app.services.llm.fake import estimate_tokens

from evaluation.check_provider import KEY_PREFIXES, NO_PREFIX_CHECK

# --- helpers -----------------------------------------------------------------


class SleepRecorder:
    """Stands in for time.sleep so tests never actually wait."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


def stepping_clock(step: float = 0.001):
    state = {"now": 0.0}

    def clock() -> float:
        state["now"] += step
        return state["now"]

    return clock


class Responder:
    """A MockTransport handler that replays a script of responses."""

    def __init__(self, responses: list[httpx.Response]) -> None:
        self.responses = responses
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        index = min(len(self.requests) - 1, len(self.responses) - 1)
        return self.responses[index]


def body(text: str = "hello", *, prompt_tokens: int = 10, completion_tokens: int = 5) -> dict:
    return {
        "model": "test-model",
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }


def ok(text: str = "hello", **usage) -> httpx.Response:
    return httpx.Response(200, json=body(text, **usage))


def make_provider(
    responder: Responder,
    *,
    sleeps: SleepRecorder | None = None,
    **overrides,
) -> OpenAICompatibleProvider:
    kwargs = {
        "name": "test",
        "model": "test-model",
        "base_url": "https://example.invalid/v1",
        "api_key": "secret-key",
        "max_retries": 2,
        "free_tier": False,
        "transport": httpx.MockTransport(responder),
        "sleep": sleeps if sleeps is not None else SleepRecorder(),
        "clock": stepping_clock(),
        "jitter": lambda: 1.0,
    }
    kwargs.update(overrides)
    return OpenAICompatibleProvider(**kwargs)


def settings(**overrides) -> Settings:
    base = {
        "llm_provider": "fake",
        "llm_model": "",
        "llm_base_url": "",
        "llm_api_key": "",
        "openai_api_key": "",
        "groq_api_key": "",
        "gemini_api_key": "",
        "mistral_api_key": "",
        "nvidia_api_key": "",
        "cerebras_api_key": "",
        "openrouter_api_key": "",
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)


# --- types -------------------------------------------------------------------


class TestTypes:
    def test_usage_adds_up(self):
        assert Usage(3, 4) + Usage(1, 1) == Usage(4, 5)

    def test_total_tokens(self):
        assert Usage(3, 4).total_tokens == 7

    def test_reported_total_beats_the_visible_counts(self):
        # A reasoning model bills for thinking that never reaches the message.
        usage = Usage(prompt_tokens=100, completion_tokens=20, reported_total=900)
        assert usage.total_tokens == 900
        assert usage.reasoning_tokens == 780

    def test_without_a_reported_total_the_visible_counts_are_summed(self):
        usage = Usage(prompt_tokens=3, completion_tokens=4)
        assert usage.total_tokens == 7
        assert usage.reasoning_tokens == 0

    def test_addition_keeps_a_reported_total_only_when_both_sides_have_one(self):
        both = Usage(1, 1, reported_total=9) + Usage(1, 1, reported_total=10)
        assert both.reported_total == 19
        half = Usage(1, 1, reported_total=9) + Usage(1, 1)
        assert half.reported_total is None

    def test_reasoning_tokens_are_never_negative(self):
        # Some providers report a total smaller than prompt + completion.
        assert Usage(prompt_tokens=10, completion_tokens=10, reported_total=5).reasoning_tokens == 0

    def test_message_serialises_to_the_wire_shape(self):
        assert system("be brief").as_dict() == {"role": "system", "content": "be brief"}

    def test_user_helper(self):
        assert user("hi").role == "user"
        assert isinstance(user("hi"), Message)


# --- pricing -----------------------------------------------------------------


class TestPricing:
    def test_known_model(self):
        assert price_for("gpt-4o-mini").input_per_million == 0.15

    def test_vendor_prefix_is_stripped(self):
        assert price_for("openai/gpt-4o") == price_for("gpt-4o")

    def test_unknown_model_has_no_price(self):
        assert price_for("some/rotated-out-model") is None

    def test_empty_model_has_no_price(self):
        assert price_for("") is None

    def test_estimate_is_none_for_unpriced_models(self):
        assert estimate_cost("who-knows", 1000, 1000) is None

    def test_estimate_scales_with_tokens(self):
        one = estimate_cost("gpt-4o-mini", 1_000_000, 0)
        assert one == pytest.approx(0.15)


# --- fake provider -----------------------------------------------------------


class TestFakeProvider:
    def test_echo_mode_is_deterministic(self):
        provider = FakeLLM()
        first = provider.complete([user("what is 2 + 2?")])
        second = provider.complete([user("what is 2 + 2?")])
        assert first.text == second.text

    def test_script_is_consumed_in_order(self):
        provider = FakeLLM(responses=["one", "two"])
        assert provider.complete([user("q")]).text == "one"
        assert provider.complete([user("q")]).text == "two"

    def test_running_out_of_script_raises(self):
        provider = FakeLLM(responses=["only"])
        provider.complete([user("q")])
        with pytest.raises(ScriptExhausted):
            provider.complete([user("q")])

    def test_responder_sees_the_messages(self):
        seen: list[str] = []

        def responder(messages):
            seen.append(messages[-1].content)
            return "42"

        provider = FakeLLM(responder=responder)
        assert provider.complete([user("first")]).text == "42"
        assert provider.complete([user("second")]).text == "42"
        assert seen == ["first", "second"]

    def test_calls_are_recorded(self):
        provider = FakeLLM(responder=lambda messages: "ok")
        provider.complete([user("q")], temperature=0.7, max_tokens=128)
        assert provider.call_count == 1
        assert provider.calls[0]["temperature"] == 0.7
        assert provider.calls[0]["max_tokens"] == 128

    def test_usage_is_estimated_not_zero(self):
        provider = FakeLLM(responder=lambda messages: "a short answer")
        completion = provider.complete([user("a question of some length")])
        assert completion.prompt_tokens > 0
        assert completion.completion_tokens > 0

    def test_free_tier_reports_zero_cost(self):
        provider = FakeLLM(responder=lambda messages: "ok")
        assert provider.complete([user("q")]).cost_usd == 0.0

    def test_paid_mode_prices_a_known_model(self):
        provider = FakeLLM(model="gpt-4o-mini", responder=lambda m: "ok", free_tier=False)
        assert provider.complete([user("q")]).cost_usd is not None

    def test_estimate_tokens_never_returns_zero(self):
        assert estimate_tokens("") == 1


# --- rate limiter ------------------------------------------------------------


class TestRateLimiter:
    def test_no_interval_never_sleeps(self):
        sleeps = SleepRecorder()
        limiter = RateLimiter(0, clock=stepping_clock(), sleep=sleeps)
        assert limiter.wait() == 0.0
        assert limiter.wait() == 0.0
        assert sleeps.delays == []

    def test_second_call_waits_out_the_interval(self):
        sleeps = SleepRecorder()
        # A clock that stands still, so the full interval is still owed.
        limiter = RateLimiter(2.0, clock=lambda: 0.0, sleep=sleeps)
        limiter.wait()
        limiter.wait()
        assert sleeps.delays == [2.0]

    def test_no_wait_when_enough_time_passed(self):
        sleeps = SleepRecorder()
        state = {"now": 0.0}
        limiter = RateLimiter(2.0, clock=lambda: state["now"], sleep=sleeps)
        limiter.wait()
        state["now"] = 5.0
        assert limiter.wait() == 0.0
        assert sleeps.delays == []


# --- budget ledger -----------------------------------------------------------


class TestBudgetLedger:
    def test_records_and_persists(self, tmp_path):
        path = tmp_path / "budget.json"
        ledger = BudgetLedger(path, limit_usd=10.0)
        ledger.record(cost_usd=0.25, prompt_tokens=100, completion_tokens=50)
        assert ledger.spent_usd == 0.25
        assert ledger.calls == 1
        assert ledger.total_tokens == 150

        reopened = BudgetLedger.open(path, limit_usd=10.0)
        assert reopened.spent_usd == 0.25
        assert reopened.calls == 1
        assert reopened.total_tokens == 150

    def test_unpriced_calls_are_counted_separately(self, tmp_path):
        ledger = BudgetLedger(tmp_path / "budget.json", limit_usd=10.0)
        ledger.record(cost_usd=None, prompt_tokens=10, completion_tokens=10)
        assert ledger.unpriced_calls == 1
        assert ledger.spent_usd == 0.0
        assert ledger.calls == 1

    def test_check_raises_once_the_ceiling_is_reached(self, tmp_path):
        ledger = BudgetLedger(tmp_path / "budget.json", limit_usd=0.10)
        ledger.record(cost_usd=0.10)
        with pytest.raises(BudgetExceeded):
            ledger.check()

    def test_check_passes_below_the_ceiling(self, tmp_path):
        ledger = BudgetLedger(tmp_path / "budget.json", limit_usd=0.10)
        ledger.record(cost_usd=0.05)
        ledger.check()

    def test_no_limit_means_no_ceiling(self, tmp_path):
        ledger = BudgetLedger(tmp_path / "budget.json", limit_usd=None)
        ledger.record(cost_usd=999.0)
        ledger.check()
        assert ledger.remaining_usd is None

    def test_remaining_tracks_the_limit(self, tmp_path):
        ledger = BudgetLedger(tmp_path / "budget.json", limit_usd=1.0)
        ledger.record(cost_usd=0.25)
        assert ledger.remaining_usd == pytest.approx(0.75)

    def test_missing_file_starts_clean(self, tmp_path):
        ledger = BudgetLedger.open(tmp_path / "nothing-here.json", limit_usd=1.0)
        assert ledger.spent_usd == 0.0

    def test_corrupt_file_does_not_crash_the_run(self, tmp_path):
        path = tmp_path / "budget.json"
        path.write_text("{not json", encoding="utf-8")
        assert BudgetLedger.open(path, limit_usd=1.0).spent_usd == 0.0

    def test_summary_shape(self, tmp_path):
        ledger = BudgetLedger(tmp_path / "budget.json", limit_usd=2.0)
        ledger.record(cost_usd=0.5, prompt_tokens=10, completion_tokens=5)
        summary = ledger.summary()
        assert summary["total_tokens"] == 15
        assert summary["remaining_usd"] == pytest.approx(1.5)


# --- openai-compatible provider ----------------------------------------------


class TestRequestShape:
    def test_sends_the_expected_json(self):
        responder = Responder([ok()])
        provider = make_provider(responder)
        provider.complete([system("be terse"), user("2+2?")], temperature=0.2, max_tokens=64)

        request = responder.requests[0]
        assert str(request.url) == "https://example.invalid/v1/chat/completions"
        payload = json.loads(request.content)
        assert payload["model"] == "test-model"
        assert payload["messages"][0] == {"role": "system", "content": "be terse"}
        assert payload["temperature"] == 0.2
        assert payload["max_tokens"] == 64

    def test_authorisation_header_carries_the_key(self):
        responder = Responder([ok()])
        make_provider(responder).complete([user("q")])
        assert responder.requests[0].headers["authorization"] == "Bearer secret-key"

    def test_default_max_output_tokens_is_applied(self):
        responder = Responder([ok()])
        provider = make_provider(responder, max_output_tokens=256)
        provider.complete([user("q")])
        assert json.loads(responder.requests[0].content)["max_tokens"] == 256

    def test_max_output_tokens_is_omitted_when_unset(self):
        responder = Responder([ok()])
        make_provider(responder).complete([user("q")])
        assert "max_tokens" not in json.loads(responder.requests[0].content)

    def test_trailing_slash_on_base_url_is_tolerated(self):
        responder = Responder([ok()])
        provider = make_provider(responder, base_url="https://example.invalid/v1/")
        provider.complete([user("q")])
        assert str(responder.requests[0].url) == "https://example.invalid/v1/chat/completions"


class TestSuccessfulResponse:
    def test_text_and_usage_are_parsed(self):
        provider = make_provider(Responder([ok("the answer is 42", prompt_tokens=30, completion_tokens=7)]))
        completion = provider.complete([user("q")])
        assert completion.text == "the answer is 42"
        assert completion.usage == Usage(30, 7)
        assert completion.attempts == 1
        assert completion.provider == "test"

    def test_model_is_taken_from_the_response(self):
        provider = make_provider(Responder([ok()]))
        assert provider.complete([user("q")]).model == "test-model"

    def test_latency_is_recorded(self):
        provider = make_provider(Responder([ok()]), clock=stepping_clock(step=0.5))
        assert provider.complete([user("q")]).latency_seconds > 0

    def test_priced_model_reports_a_cost(self):
        provider = make_provider(
            Responder([ok(prompt_tokens=1_000_000, completion_tokens=0)]),
            model="gpt-4o-mini",
        )
        assert provider.complete([user("q")]).cost_usd == pytest.approx(0.15)

    def test_unpriced_model_reports_unknown_cost_not_zero(self):
        provider = make_provider(Responder([ok()]), model="rotated-out")
        assert provider.complete([user("q")]).cost_usd is None

    def test_free_tier_reports_zero_regardless_of_model(self):
        provider = make_provider(Responder([ok()]), model="gpt-4o", free_tier=True)
        assert provider.complete([user("q")]).cost_usd == 0.0

    def test_missing_usage_block_is_unknown_when_paid(self):
        response = httpx.Response(200, json={"model": "m", "choices": [{"message": {"content": "hi"}}]})
        provider = make_provider(Responder([response]))
        assert provider.complete([user("q")]).cost_usd is None

    def test_finish_reason_is_kept(self):
        """It is the only signal that an answer was cut off rather than wrong."""
        response = httpx.Response(
            200,
            json={
                "model": "m",
                "choices": [
                    {"message": {"content": "the total is 1"}, "finish_reason": "length"}
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 700},
            },
        )
        completion = make_provider(Responder([response])).complete([user("q")])
        assert completion.finish_reason == "length"
        assert completion.usage.total_tokens == 700
        assert completion.usage.reasoning_tokens == 686

    def test_reasoning_tokens_are_priced_as_output(self):
        # 60k reasoning tokens, billed at gpt-4o-mini's $0.60 per million output.
        response = httpx.Response(
            200,
            json={
                "model": "gpt-4o-mini",
                "choices": [{"message": {"content": "1"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 60_000},
            },
        )
        completion = make_provider(Responder([response]), model="gpt-4o-mini").complete(
            [user("q")]
        )
        assert completion.cost_usd == pytest.approx(0.036)


class TestRetries:
    def test_rate_limit_then_success(self):
        sleeps = SleepRecorder()
        responder = Responder([httpx.Response(429, json={"error": "slow down"}), ok("finally")])
        provider = make_provider(responder, sleeps=sleeps)
        completion = provider.complete([user("q")])
        assert completion.text == "finally"
        assert completion.attempts == 2
        assert len(sleeps.delays) == 1

    def test_retry_after_header_wins_over_backoff(self):
        sleeps = SleepRecorder()
        responder = Responder(
            [httpx.Response(429, json={}, headers={"retry-after": "7"}), ok()]
        )
        make_provider(responder, sleeps=sleeps).complete([user("q")])
        assert sleeps.delays == [7.0]

    def test_retry_delay_in_the_body_is_honoured(self):
        """Gemini reports its wait in the body rather than in a header."""
        sleeps = SleepRecorder()
        response = httpx.Response(429, json={"error": {"details": [{"retryDelay": "7s"}]}})
        make_provider(Responder([response, ok()]), sleeps=sleeps).complete([user("q")])
        assert sleeps.delays == [7.0]

    def test_the_header_wins_when_both_are_present(self):
        sleeps = SleepRecorder()
        response = httpx.Response(
            429,
            json={"error": {"details": [{"retryDelay": "99s"}]}},
            headers={"retry-after": "7"},
        )
        make_provider(Responder([response, ok()]), sleeps=sleeps).complete([user("q")])
        assert sleeps.delays == [7.0]

    def test_a_daily_quota_stops_the_run_instead_of_sleeping_hours(self):
        sleeps = SleepRecorder()
        response = httpx.Response(
            429,
            json={
                "error": {
                    "code": 429,
                    "message": "You exceeded your current quota",
                    "details": [{"retryDelay": "34484.99s"}],
                }
            },
        )
        provider = make_provider(Responder([response]), sleeps=sleeps)
        with pytest.raises(ProviderRateLimited, match=r"9h34m"):
            provider.complete([user("q")])
        assert sleeps.delays == [], "it must not sit through a nine hour wait"

    def test_a_daily_quota_wrapped_in_a_list_is_still_read(self):
        """Gemini's OpenAI-compatible layer returns errors inside a one-element list."""
        sleeps = SleepRecorder()
        response = httpx.Response(
            429,
            json=[
                {
                    "error": {
                        "code": 429,
                        "details": [
                            {
                                "@type": "type.googleapis.com/google.rpc.RetryInfo",
                                "retryDelay": "540s",
                            }
                        ],
                    }
                }
            ],
        )
        provider = make_provider(Responder([response]), sleeps=sleeps)
        with pytest.raises(ProviderRateLimited, match=r"9m00s"):
            provider.complete([user("q")])
        assert sleeps.delays == []

    def test_backoff_grows_when_no_header_is_given(self):
        sleeps = SleepRecorder()
        responder = Responder(
            [httpx.Response(503, json={}), httpx.Response(503, json={}), ok()]
        )
        make_provider(responder, sleeps=sleeps).complete([user("q")])
        # jitter is pinned to 1.0 in the helper, so the delays are the ceilings.
        assert sleeps.delays == [2.0, 4.0]

    def test_giving_up_on_rate_limits(self):
        responder = Responder([httpx.Response(429, json={})])
        provider = make_provider(responder, max_retries=1)
        with pytest.raises(ProviderRateLimited):
            provider.complete([user("q")])
        assert len(responder.requests) == 2

    def test_giving_up_on_server_errors(self):
        responder = Responder([httpx.Response(500, json={})])
        provider = make_provider(responder, max_retries=1)
        with pytest.raises(ProviderUnavailable):
            provider.complete([user("q")])

    def test_bad_request_is_not_retried(self):
        responder = Responder([httpx.Response(400, json={"error": "bad model"})])
        provider = make_provider(responder)
        with pytest.raises(ProviderRequestFailed):
            provider.complete([user("q")])
        assert len(responder.requests) == 1

    def test_unauthorised_is_not_retried(self):
        responder = Responder([httpx.Response(401, json={})])
        provider = make_provider(responder)
        with pytest.raises(ProviderRequestFailed):
            provider.complete([user("q")])
        assert len(responder.requests) == 1

    def test_transport_error_is_retried_then_reported(self):
        attempts = {"count": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["count"] += 1
            raise httpx.ConnectError("no route to host", request=request)

        provider = make_provider(Responder([]), max_retries=1, transport=httpx.MockTransport(handler))
        with pytest.raises(ProviderUnavailable):
            provider.complete([user("q")])
        assert attempts["count"] == 2

    def test_timeout_is_reported_as_unavailable(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("too slow", request=request)

        provider = make_provider(Responder([]), max_retries=0, transport=httpx.MockTransport(handler))
        with pytest.raises(ProviderUnavailable):
            provider.complete([user("q")])

    def test_retry_after_ignores_a_non_numeric_value(self):
        sleeps = SleepRecorder()
        responder = Responder(
            [httpx.Response(429, json={}, headers={"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}), ok()]
        )
        make_provider(responder, sleeps=sleeps).complete([user("q")])
        # Ignored, so the wait falls back to the rate-limit backoff rather than
        # being parsed into something wrong.
        assert sleeps.delays == [15.0]

    def test_a_rate_limit_waits_longer_than_a_server_error(self):
        # A 503 is a capacity blip that clears in seconds. A 429 with no stated
        # delay is usually a per-minute window, which is not.
        limit = SleepRecorder()
        make_provider(
            Responder([httpx.Response(429, json={}), ok()]), sleeps=limit
        ).complete([user("q")])

        blip = SleepRecorder()
        make_provider(
            Responder([httpx.Response(503, json={}), ok()]), sleeps=blip
        ).complete([user("q")])

        assert limit.delays[0] > blip.delays[0]


class TestMalformedResponses:
    def test_non_json_body(self):
        responder = Responder([httpx.Response(200, text="<html>not json</html>")])
        with pytest.raises(InvalidResponse):
            make_provider(responder).complete([user("q")])

    def test_missing_choices(self):
        responder = Responder([httpx.Response(200, json={"model": "m"})])
        with pytest.raises(InvalidResponse):
            make_provider(responder).complete([user("q")])

    def test_empty_choices(self):
        responder = Responder([httpx.Response(200, json={"choices": []})])
        with pytest.raises(InvalidResponse):
            make_provider(responder).complete([user("q")])

    def test_non_string_content(self):
        responder = Responder(
            [httpx.Response(200, json={"choices": [{"message": {"content": ["a", "b"]}}]})]
        )
        with pytest.raises(InvalidResponse):
            make_provider(responder).complete([user("q")])


class TestLifecycle:
    def test_context_manager_closes_the_client(self):
        with make_provider(Responder([ok()])) as provider:
            provider.complete([user("q")])
            assert not provider.closed
        assert provider.closed

    def test_redacted_key_hides_all_but_the_tail(self):
        provider = make_provider(Responder([ok()]))
        assert provider.redacted_key == "****-key"
        assert "secret" not in provider.redacted_key

    def test_redacted_key_when_unset(self):
        provider = make_provider(Responder([ok()]), api_key="")
        assert provider.redacted_key == "(unset)"


# --- factory -----------------------------------------------------------------


class TestFactory:
    def test_fake_is_the_default(self):
        assert isinstance(build_provider(settings()), FakeLLM)

    def test_fake_needs_no_key_or_model(self):
        provider = build_provider(settings(llm_provider="fake"))
        assert provider.model == "fake-model"

    def test_unknown_provider_is_rejected(self):
        with pytest.raises(ProviderNotConfigured, match="unknown provider"):
            build_provider(settings(llm_provider="some-new-startup"))

    def test_missing_key_is_reported(self):
        with pytest.raises(ProviderNotConfigured, match="needs an API key"):
            build_provider(settings(llm_provider="groq", llm_model="m"))

    def test_missing_model_points_at_the_model_list(self):
        with pytest.raises(ProviderNotConfigured, match="console.groq.com"):
            build_provider(settings(llm_provider="groq", llm_api_key="k"))

    def test_generic_key_is_used(self):
        provider = build_provider(
            settings(llm_provider="groq", llm_model="m", llm_api_key="generic-key")
        )
        assert provider.redacted_key == "****-key"

    def test_provider_specific_key_is_the_fallback(self):
        provider = build_provider(
            settings(llm_provider="groq", llm_model="m", groq_api_key="groq-specific")
        )
        assert provider.redacted_key == "****ific"

    def test_generic_key_wins_over_the_specific_one(self):
        provider = build_provider(
            settings(
                llm_provider="groq",
                llm_model="m",
                llm_api_key="generic-key",
                groq_api_key="groq-specific",
            )
        )
        assert provider.redacted_key == "****-key"

    def test_known_endpoints(self):
        expected = {
            "openai": "https://api.openai.com/v1",
            "groq": "https://api.groq.com/openai/v1",
            "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
            "cerebras": "https://api.cerebras.ai/v1",
            "openrouter": "https://openrouter.ai/api/v1",
        }
        for name, base_url in expected.items():
            provider = build_provider(settings(llm_provider=name, llm_model="m", llm_api_key="k"))
            assert provider.base_url == base_url

    def test_gemini_reaches_the_documented_chat_endpoint(self):
        provider = build_provider(
            settings(llm_provider="gemini", llm_model="gemini-3.8-flash", gemini_api_key="k")
        )
        assert provider._endpoint.endswith("/v1beta/openai/chat/completions")

    def test_gemini_key_is_read_from_its_own_variable(self):
        provider = build_provider(
            settings(llm_provider="gemini", llm_model="m", gemini_api_key="AIzaSyExample")
        )
        assert provider.redacted_key == "****mple"

    def test_base_url_override_wins(self):
        provider = build_provider(
            settings(
                llm_provider="openai",
                llm_model="m",
                llm_api_key="k",
                llm_base_url="https://my-gateway.internal/v1",
            )
        )
        assert provider.base_url == "https://my-gateway.internal/v1"

    def test_settings_flow_through_to_the_client(self):
        provider = build_provider(
            settings(
                llm_provider="groq",
                llm_model="m",
                llm_api_key="k",
                llm_max_retries=5,
                llm_request_timeout_seconds=12.0,
                llm_free_tier=False,
            )
        )
        assert provider.max_retries == 5
        assert provider.free_tier is False
        provider.close()


class TestMaskKey:
    def test_hides_everything_but_the_tail(self):
        assert mask_key("gsk_abcdefgh1234") == "****1234"

    def test_empty(self):
        assert mask_key("") == "(unset)"


class TestResolveConnection:
    def test_resolves_endpoint_key_and_model(self):
        connection = resolve_connection(
            settings(llm_provider="gemini", llm_model="gemini-3.8-flash", gemini_api_key="k")
        )
        assert isinstance(connection, Connection)
        assert connection.name == "gemini"
        assert connection.model == "gemini-3.8-flash"
        assert connection.base_url.startswith("https://generativelanguage")

    def test_redacts_the_key(self):
        connection = resolve_connection(
            settings(llm_provider="gemini", llm_model="m", gemini_api_key="AIzaSecret999")
        )
        assert connection.redacted_key == "****t999"

    def test_unknown_provider(self):
        with pytest.raises(ProviderNotConfigured, match="unknown provider"):
            resolve_connection(settings(llm_provider="nope"))

    def test_fake_has_no_endpoint_to_resolve(self):
        with pytest.raises(ProviderNotConfigured, match="offline"):
            resolve_connection(settings(llm_provider="fake"))

    def test_missing_key(self):
        with pytest.raises(ProviderNotConfigured, match="needs an API key"):
            resolve_connection(settings(llm_provider="gemini", llm_model="m"))

    def test_missing_model_names_the_right_hint(self):
        with pytest.raises(ProviderNotConfigured, match="aistudio.google.com"):
            resolve_connection(settings(llm_provider="gemini", gemini_api_key="k"))

    def test_a_provider_reads_its_own_key_variable(self):
        # {name}_api_key is looked up by name, so a provider added to the endpoint
        # table without a matching settings field would send no key at all.
        connection = resolve_connection(
            settings(llm_provider="mistral", llm_model="mistral-large-latest", mistral_api_key="mk")
        )
        assert connection.api_key == "mk"
        assert connection.base_url == "https://api.mistral.ai/v1"

    def test_every_compatible_provider_has_a_model_list_hint(self):
        assert set(COMPATIBLE_ENDPOINTS) <= set(MODEL_LIST_HINTS)


class TestKeyPrefixes:
    """The diagnostic must not cry wolf about a key that is simply a new format."""

    def test_every_provider_has_prefixes_to_check(self):
        # Every provider is either matched on a prefix or explicitly recorded as
        # having none, so adding one cannot silently skip the check.
        assert set(COMPATIBLE_ENDPOINTS) <= set(KEY_PREFIXES) | NO_PREFIX_CHECK

    def test_a_provider_is_not_both_matched_and_exempt(self):
        assert not set(KEY_PREFIXES) & NO_PREFIX_CHECK

    def test_prefixes_are_non_empty_strings(self):
        for name, prefixes in KEY_PREFIXES.items():
            assert prefixes, f"{name} has nothing to match against"
            assert all(isinstance(prefix, str) and prefix for prefix in prefixes)

    def test_gemini_accepts_both_key_formats(self):
        # AI Studio issues "AQ." authentication keys now; "AIza" traffic keys came
        # before, and both are still in circulation.
        assert "AQ." in KEY_PREFIXES["gemini"]
        assert "AIza" in KEY_PREFIXES["gemini"]

