import json

import pytest

from src import Status, analyze


@pytest.mark.parametrize("original,rewritten", [("", "Something."), ("Something.", "   "), ("\n\t", "")])
def test_empty_or_whitespace_input(original, rewritten):
    result = analyze(original, rewritten)
    assert result.status is Status.REVIEW
    assert [f.rule_id for f in result.findings] == ["INPUT-EMPTY"]


def test_unchanged_pair_triggers_nothing():
    text = "Students must submit the form before 15 November unless the HOD approves."
    result = analyze(text, text)
    assert result.findings == []
    assert result.status is Status.NONE


def test_whitespace_differences_are_ignored():
    assert analyze("Pay  the fee\nby Friday.", "Pay the fee by Friday.").findings == []


def test_harmless_lexical_variation():
    result = analyze("Submit the application before 15 November.",
                     "The application must be submitted before 15 November.")
    assert result.status is Status.NONE


def test_status_never_claims_equivalence():
    result = analyze("Pay the fee.", "Pay the fee.")
    assert "equivalent" not in result.status.value.lower()
    assert any("not proof" in note for note in result.notes)


def test_findings_serialise_to_json():
    result = analyze("Applicants must have at least 75% attendance.",
                     "Applicants may have at least 70% attendance.")
    data = json.loads(result.to_json())
    assert data["status"] == "Potential meaning drift detected"
    assert data["severity_counts"]["HIGH"] >= 1
    first = data["findings"][0]
    for key in ("rule_id", "category", "severity", "original_evidence", "rewritten_evidence",
                "explanation", "reason", "suggested_action"):
        assert key in first
    assert first["severity"] in {"HIGH", "MEDIUM", "REVIEW"}


def test_findings_sorted_by_severity():
    result = analyze("Applicants must have at least 75% attendance.",
                     "Applicants may have at least 70% attendance.")
    order = [f.severity.value for f in result.findings]
    assert order == sorted(order, key=["HIGH", "MEDIUM", "REVIEW"].index)


def test_examples_file_runs_through_engine():
    from pathlib import Path
    examples = json.loads((Path(__file__).parents[1] / "data" / "examples.json").read_text())
    assert len(examples) >= 5
    for ex in examples:
        analyze(ex["original"], ex["rewritten"])  # must not raise
