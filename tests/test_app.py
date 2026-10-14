"""Headless smoke test of the Streamlit UI using Streamlit's own test harness."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).parents[1] / "app.py")


def test_app_renders_and_compares():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    at.text_area(key="original").set_value("Candidates must submit the form before Friday.")
    at.text_area(key="rewritten").set_value("Candidates may submit the form before Friday.")
    at.button[0].click().run()
    assert not at.exception
    html = " ".join(m.value for m in at.markdown)
    assert "Potential meaning drift detected" in html
    assert "MOD-WEAKENED" in html
