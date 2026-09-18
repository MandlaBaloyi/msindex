import pandas as pd
import pytest

from src import briefing
from src.briefing import active_provider, build_facts, fallback_briefing, write_briefing
from src.pricing import PolicyTerms

KEYS = ["ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "GEMINI_API_KEY", "GOOGLE_API_KEY"]


@pytest.fixture
def no_keys(monkeypatch):
    for key in KEYS:
        monkeypatch.delenv(key, raising=False)


def test_no_keys_means_no_provider(no_keys):
    assert active_provider() is None


def test_gemini_key_selects_gemini(no_keys, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    assert active_provider() == "gemini"


def test_google_key_also_works(no_keys, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    assert active_provider() == "gemini"


def test_claude_takes_precedence(no_keys, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    assert active_provider() == "anthropic"


def sample_facts():
    df = pd.DataFrame([{
        "season": 2025, "district": "Bothaville", "province": "Free State", "Data split": "Current season (forecast)",
        "rain_mm": 380.0, "water_balance_mm": -420.0, "dry_spell_days": 21, "heat_stress_days": 40,
        "ndvi_anomaly": -0.08, "pred_yield_anomaly": -0.22, "yield_anomaly": -0.23, "Risk band": "Severe",
        "pred_yield_tha": 4.1, "yield_tha": 4.0, "yield_trend": 5.2, "yield_status": "preliminary (CEC)",
    }])
    return build_facts(df, 2025, PolicyTerms(), {})


def test_fact_sheet_carries_only_computed_aggregates():
    facts = sample_facts()
    assert facts["season"] == "2025/26"
    assert facts["districts"][0]["district"] == "Bothaville"
    assert facts["region_summary"]["districts_total"] == 1
    assert facts["caveats"], "the fact sheet must carry its own caveats to the model"


def test_template_briefing_reads_from_the_facts():
    text = fallback_briefing(sample_facts())
    for heading in ("**Conditions.**", "**Model view.**", "**Exposure.**", "**Watch items.**"):
        assert heading in text
    assert "380 mm" in text


def test_without_keys_the_page_still_gets_a_briefing(no_keys):
    text, source = write_briefing(sample_facts())
    assert source == "template"
    assert "**Conditions.**" in text


def test_questions_say_so_when_no_model_is_configured(no_keys):
    answer, source = write_briefing(sample_facts(), question="Which district is worst?")
    assert source == "unavailable"
    assert "GEMINI_API_KEY" in answer


def test_a_failing_provider_falls_back_to_the_template(no_keys, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    def boom(*args, **kwargs):
        raise RuntimeError("quota exhausted")

    monkeypatch.setattr(briefing, "gemini_briefing", boom)
    text, source = write_briefing(sample_facts())
    assert "**Conditions.**" in text
    assert "gemini unavailable" in source
