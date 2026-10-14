import json

from src.passage import analyze_passage, split_sentences, similarity


def test_split_respects_abbreviations_times_and_bullets():
    text = "Pay Rs. 500 by 5 p.m. on Friday. No. 4 applies. Dr. Rao approves leave!\n- First item\n2) Second item"
    assert split_sentences(text) == ["Pay Rs. 500 by 5 p.m. on Friday.", "No. 4 applies.",
                                     "Dr. Rao approves leave!", "First item", "Second item"]


def test_similarity_bounds():
    assert similarity(set(), {"a"}) == 0.0
    assert similarity({"a", "b"}, {"a", "b"}) == 1.0


ORIGINAL = ("Students must submit the form before Friday. Late forms are accepted only if the HOD approves. "
            "Parents will be informed of any delay.")


def test_identical_passages_align_one_to_one():
    result = analyze_passage(ORIGINAL, ORIGINAL)
    assert [s.kind for s in result.segments] == ["aligned", "aligned", "aligned"]
    assert result.findings == []
    assert result.status.value == "No configured checks triggered"


def test_dropped_and_changed_sentences():
    rewritten = "Students may submit the form before Friday. Late forms are accepted if the HOD approves."
    result = analyze_passage(ORIGINAL, rewritten)
    kinds = [s.kind for s in result.segments]
    assert kinds == ["aligned", "aligned", "dropped"]
    rules = {f.rule_id for f in result.findings}
    assert {"MOD-WEAKENED", "COND-ONLYIF-CHANGED", "PSG-SENTENCE-DROPPED"} <= rules


def test_added_sentence_with_constraint_is_medium():
    result = analyze_passage("Submit the form before Friday.",
                             "Submit the form before Friday. A late fee of Rs. 200 applies.")
    added = [s for s in result.segments if s.kind == "added"]
    assert len(added) == 1 and added[0].finding.severity.value == "MEDIUM"


def test_merged_sentences_are_grouped():
    result = analyze_passage("Students must carry ID cards. ID cards must be shown at the gate.",
                             "Students must carry ID cards and show them at the gate.")
    assert result.segments[0].kind == "merged"


def test_passage_json_and_empty_input():
    data = json.loads(analyze_passage(ORIGINAL, ORIGINAL).to_json())
    assert data["mode"] == "passage" and len(data["segments"]) == 3
    assert analyze_passage("", "x").status.value == "Manual review recommended"
