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


def test_passage_and_batch_tabs():
    at = AppTest.from_file(APP, default_timeout=60).run()
    next(b for b in at.button if b.label == "Load sample passage").click().run()
    assert not at.exception
    html = " ".join(m.value for m in at.markdown)
    assert "PSG-SENTENCE-DROPPED" in html and "COND-EXCEPTION-REMOVED" in html

    next(b for b in at.button if b.label == "Use bundled benchmark").click().run()
    assert not at.exception
    html = " ".join(m.value for m in at.markdown)
    assert "152 pairs from data/benchmark.csv" in html


def test_swap_and_clear():
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.text_area(key="original").set_value("A must pay.")
    at.text_area(key="rewritten").set_value("A may pay.")
    next(b for b in at.button if b.label == "Swap").click().run()
    assert at.text_area(key="original").value == "A may pay."
    next(b for b in at.button if b.label == "Clear").click().run()
    assert at.text_area(key="original").value == "" and at.session_state["result"] is None
