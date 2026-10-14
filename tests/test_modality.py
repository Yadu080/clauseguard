from src.extractors import analyze_sentence
from tests.helpers import drift_categories, rule_ids, severities


def modal_labels(text):
    return [m.label for m in analyze_sentence(text).markers_in("modality")]


def test_obligation_weakened_to_permission():
    sev = severities("Candidates must submit the form before Friday.",
                     "Candidates may submit the form before Friday.")
    assert sev.get("MOD-WEAKENED") == "MEDIUM"


def test_permission_strengthened_to_obligation():
    assert "MOD-STRENGTHENED" in rule_ids("Students may appeal within 15 days.",
                                          "Students must appeal within 15 days.")


def test_prohibition_to_permission_is_high():
    assert severities("Visitors are not allowed in the hostel.",
                      "Visitors are allowed in the hostel.").get("MOD-REVERSED") == "HIGH"


def test_mandatory_to_optional_is_reversal():
    assert "MOD-REVERSED" in rule_ids("Attendance is mandatory.", "Attendance is optional.")


def test_negated_modals_are_prohibitions():
    for text in ("Students must not leave.", "Students cannot leave.", "Students may not leave.",
                 "Students are not permitted to leave.", "Students will not be allowed to leave.",
                 "Phones are prohibited.", "No student may leave."):
        assert modal_labels(text) == ["PROHIBITION"], text


def test_same_force_different_words_is_not_flagged():
    # Not every pair of obligation words is interchangeable in every context,
    # but these pairs are listed under the same label on purpose.
    assert "modality" not in drift_categories("Students must carry ID cards.",
                                              "Students are required to carry ID cards.")
    assert "modality" not in drift_categories("Phones are not permitted.", "Phones are prohibited.")


def test_imperative_counts_as_obligation():
    assert modal_labels("Submit the application before 15 November.") == ["OBLIGATION"]
    assert analyze_sentence("Submit the form.").markers_in("modality")[0].implicit


def test_noun_phrase_start_is_not_imperative():
    assert modal_labels("Use of phones is prohibited.") == ["PROHIBITION"]
    assert modal_labels("Leave applications must be approved.") == ["OBLIGATION"]


def test_month_may_is_not_a_modal():
    assert modal_labels("The results were declared on 15 May 2025.") == []


def test_should_not_exceed_is_recommendation():
    assert modal_labels("The report should not exceed 40 pages.") == ["RECOMMENDATION"]
