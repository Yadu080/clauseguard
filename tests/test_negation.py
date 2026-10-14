from src.extractors import analyze_sentence
from tests.helpers import drift_categories, rule_ids, severities


def labels(text):
    return [m.label for m in analyze_sentence(text).markers_in("negation")]


def test_missing_negation_is_high():
    sev = severities("Students who have not paid the fee are ineligible.",
                     "Students who have paid the fee are ineligible.")
    assert sev.get("NEG-REMOVED") == "HIGH"


def test_missing_negation_points_at_unnegated_word():
    from src import analyze
    result = analyze("Students who have not paid the fee are ineligible.",
                     "Students who have paid the fee are ineligible.")
    finding = next(f for f in result.findings if f.rule_id == "NEG-REMOVED")
    assert finding.original_evidence.text == "not"
    assert finding.rewritten_evidence.text == "paid"


def test_added_negation_is_high():
    assert "NEG-ADDED" in rule_ids("Admission will be cancelled if the documents are fake.",
                                   "Admission will not be cancelled if the documents are fake.")


def test_contraction_counts_as_same_negation():
    assert "negation" not in drift_categories("Students who have not paid the fee are ineligible.",
                                              "Students who haven't paid the fee are ineligible.")


def test_antonym_swap_detected():
    assert "NEG-ANTONYM" in rule_ids("Hostel fees are refundable on withdrawal.",
                                     "Hostel fees are non-refundable on withdrawal.")


def test_lexical_paraphrase_is_only_review():
    sev = severities("Late submissions will not be accepted.", "Late submissions will be rejected.")
    assert sev == {"NEG-REPHRASED": "REVIEW"}


def test_negation_moved_between_verbs():
    assert "NEG-MOVED" in rule_ids("Students who have not paid can attend.",
                                   "Students who have paid cannot attend.")


def test_bounds_are_not_negations():
    assert labels("Submit no later than Friday and write no more than 5 pages.") == []
    assert labels("Reports must not exceed 10 pages.") == []


def test_not_only_and_whether_or_not_are_not_negations():
    assert labels("Not only students but also staff must register.") == []
    assert labels("Fees apply whether or not you attend.") == []


def test_neither_nor_counts_once():
    assert labels("Neither phones nor calculators are allowed.") == ["NEITHER"]


def test_abbreviation_no_dot_is_not_negation():
    assert labels("Refer to Circular No. 12 for details.") == []
