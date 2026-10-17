"""The diagnostic's parts that can be tested without reaching the network.

Argument handling and the provider override live here rather than alongside the
adapter's tests, because they belong to the check command. The one call to
``check`` uses the offline provider, so the suite still runs with no key.
"""

from __future__ import annotations

from app.core.config import Settings

from evaluation.check_provider import check, parse_args, settings_for


def settings(**overrides) -> Settings:
    base = {
        "llm_provider": "gemini",
        "llm_model": "gemini-3.6-flash",
        "llm_base_url": "",
        "llm_api_key": "",
        "openai_api_key": "",
        "groq_api_key": "",
        "gemini_api_key": "AIzaSecret9999",
        "mistral_api_key": "",
        "nvidia_api_key": "",
        "cerebras_api_key": "",
        "openrouter_api_key": "",
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)


class TestSettingsFor:
    def test_no_override_leaves_the_settings_alone(self):
        original = settings()
        assert settings_for(original, None, None) is original

    def test_provider_swaps_without_taking_the_model_with_it(self):
        changed = settings_for(settings(), "mistral", None)
        assert changed.llm_provider == "mistral"
        assert changed.llm_model == "gemini-3.6-flash"

    def test_model_swaps_without_taking_the_provider_with_it(self):
        changed = settings_for(settings(), None, "mistral-large-latest")
        assert changed.llm_provider == "gemini"
        assert changed.llm_model == "mistral-large-latest"

    def test_provider_name_is_normalised(self):
        # The factory lowercases as well, but the check prints the name it was
        # given, and " Mistral" reads like a mistake when it is not one.
        assert settings_for(settings(), " Mistral ", None).llm_provider == "mistral"

    def test_a_swapped_provider_brings_its_own_key(self):
        # The point of the override: try a candidate while the project stays
        # pointed at whatever it was pointed at.
        changed = settings_for(
            settings(mistral_api_key="mk"), "mistral", "mistral-large-latest"
        )
        assert changed.gemini_api_key == "AIzaSecret9999"
        assert changed.mistral_api_key == "mk"

    def test_the_original_settings_are_not_mutated(self):
        original = settings()
        settings_for(original, "mistral", "mistral-large-latest")
        assert original.llm_provider == "gemini"


class TestParseArgs:
    def test_defaults(self):
        args = parse_args([])
        assert args.provider is None
        assert args.model is None
        assert args.list_only is False

    def test_flags(self):
        args = parse_args(["--provider", "mistral", "--model", "m", "--list-only"])
        assert args.provider == "mistral"
        assert args.model == "m"
        assert args.list_only is True


class TestCheck:
    def test_the_offline_provider_costs_nothing_and_asks_nothing(self, capsys):
        code = check(settings(llm_provider="fake", llm_model=""), list_only=False)
        assert code == 0
        assert "nothing to spend" in capsys.readouterr().out
