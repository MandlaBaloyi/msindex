"""Smoke tests: every page renders and the headline controls work, headlessly."""
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / "data" / "district_seasons.csv").exists(),
                                reason="dataset not built; run python -m src.pipeline")
PAGES = ["app_pages/overview.py", "app_pages/portfolio.py", "app_pages/basis_risk.py", "app_pages/pricing.py",
         "app_pages/audit.py", "app_pages/briefing.py", "app_pages/data_methods.py", "app_pages/security.py",
         "app_pages/business.py"]


def run(page: str) -> AppTest:
    at = AppTest.from_file(str(ROOT / page), default_timeout=120).run()
    assert not at.exception, at.exception
    return at


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    at = run(page)
    assert len(at.markdown) + len(at.metric) + len(at.dataframe) > 0


def test_map_has_satellite_overlay_and_columns():
    at = run("app_pages/portfolio.py")
    spec = json.loads(at.get("deck_gl_json_chart")[0].proto.json)
    kinds = [layer["@@type"] for layer in spec["layers"]]
    assert "TileLayer" in kinds and "ColumnLayer" in kinds
    if (ROOT / "static" / "overlays").exists():
        assert "BitmapLayer" in kinds


def test_3d_terrain_toggle_swaps_the_basemap():
    at = run("app_pages/portfolio.py")
    [t for t in at.toggle if "3D terrain" in t.label][0].set_value(True).run()
    assert not at.exception, at.exception
    kinds = [l["@@type"] for l in json.loads(at.get("deck_gl_json_chart")[0].proto.json)["layers"]]
    assert "TerrainLayer" in kinds


def test_season_selection_changes_the_overlay():
    at = run("app_pages/portfolio.py")
    at.select_slider[0].set_value(2015).run()
    assert not at.exception, at.exception
    images = [l.get("image") for l in json.loads(at.get("deck_gl_json_chart")[0].proto.json)["layers"]
              if l["@@type"] == "BitmapLayer"]
    assert not images or "2015" in images[0]


def test_tamper_and_rebuild_cycle():
    at = run("app_pages/audit.py")
    assert any("verified" in s.value.lower() for s in at.success)
    [b for b in at.button if "tampering" in b.label][0].click().run()
    assert not at.exception, at.exception
    assert any("Tampering detected" in e.value for e in at.error)
    [b for b in at.button if "Rebuild" in b.label][0].click().run()
    assert not at.exception, at.exception
    assert any("verified" in s.value.lower() for s in at.success)


def test_pricing_premium_covers_expected_loss():
    at = run("app_pages/pricing.py")
    labels = {m.label: m.value for m in at.metric}
    expected = float(labels["Expected loss / ha"].replace("R", "").replace(",", "").strip())
    gross = float(labels["Gross premium / ha"].replace("R", "").replace(",", "").strip())
    assert gross >= expected


def test_widening_the_ladder_raises_the_premium():
    at = run("app_pages/pricing.py")
    before = float({m.label: m.value for m in at.metric}["Gross premium / ha"].replace("R", "").replace(",", "").strip())
    at.sidebar.slider("ladder").set_range(-0.40, -0.05).run()  # (exhaustion, attachment): cover starts earlier
    assert not at.exception, at.exception
    after = float({m.label: m.value for m in at.metric}["Gross premium / ha"].replace("R", "").replace(",", "").strip())
    assert after >= before  # cover starting at a smaller shortfall cannot be cheaper
