from src import Severity, analyze
from tests.helpers import drift_categories, rule_ids, severities

ORIGINAL = "Students below 75% attendance cannot appear for the examination unless the HOD approves."


def test_missing_exception_is_high():
    result = analyze(ORIGINAL, "Students below 75% attendance cannot appear for the examination.")
    finding = next(f for f in result.findings if f.rule_id == "COND-EXCEPTION-REMOVED")
    assert finding.severity is Severity.HIGH
    assert finding.original_evidence.text == "unless"
    assert finding.original_evidence.context == "unless the HOD approves"
    assert finding.rewritten_evidence is None


def test_missing_condition_is_high():
    assert severities("Late work is accepted if a certificate is attached.",
                      "Late work is accepted.").get("COND-CONDITION-REMOVED") == "HIGH"


def test_only_if_to_if():
    assert "COND-ONLYIF-CHANGED" in rule_ids("Leave is granted only if a certificate is submitted.",
                                             "Leave is granted if a certificate is submitted.")


def test_unless_to_if_reverses():
    assert "COND-EXCEPTION-TO-CONDITION" in rule_ids("Fees are charged unless waived.",
                                                     "Fees are charged if waived.")


def test_equivalent_exception_words_not_flagged():
    assert drift_categories("Laptops are allowed except during exams.",
                            "Laptops are allowed unless it is exam week.") == set()


def test_unless_rephrased_as_without_is_review_only():
    result = analyze("Students cannot enter the lab unless the instructor approves.",
                     "Students cannot enter the lab without the instructor's approval.")
    assert [f.rule_id for f in result.findings] == ["COND-REPHRASED"]
    assert result.status.value == "Manual review recommended"


def test_added_exception_is_medium():
    assert severities("Phones are not allowed.",
                      "Phones are not allowed except in emergencies.").get("COND-EXCEPTION-ADDED") == "MEDIUM"
