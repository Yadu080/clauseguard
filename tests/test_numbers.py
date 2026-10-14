import pytest

from src.extractors import extract_numbers
from tests.helpers import rule_ids, severities


def keys(text):
    return [n.normalized for n in extract_numbers(text)]


def test_changed_percentage_is_high():
    assert severities("Applicants must have at least 75% attendance.",
                      "Applicants must have at least 70% attendance.").get("NUM-CHANGED") == "HIGH"


def test_changed_date():
    assert "NUM-CHANGED" in rule_ids("Submit before 15 November.", "Submit before 25 November.")


def test_changed_weekday_and_time():
    assert "NUM-CHANGED" in rule_ids("Return by Friday.", "Return by Monday.")
    assert "NUM-CHANGED" in rule_ids("Residents must return by 9:30 pm.", "Residents must return by 10:30 pm.")


def test_missing_value_is_high_and_new_value_is_medium():
    assert severities("Pay within 7 days.", "Pay promptly.").get("NUM-MISSING") == "HIGH"
    assert severities("Pay promptly.", "Pay within 7 days.").get("NUM-ADDED") == "MEDIUM"


def test_unit_change_detected():
    assert "NUM-UNIT-CHANGED" in rule_ids("File within 7 days.", "File within 7 weeks.")


@pytest.mark.parametrize("text,expected", [
    ("The fee is Rs. 1,500.", ["currency:INR:1500"]),
    ("The fee is ₹1500.", ["currency:INR:1500"]),
    ("The fee is 1500 rupees.", ["currency:INR:1500"]),
    ("A grant of $2.5 million.", ["currency:USD:2500000"]),
    ("A stipend of Rs. 1.2 lakh.", ["currency:INR:120000"]),
    ("A fine of Rs. 10 per day.", ["currency:INR/day:10"]),
    ("A CGPA of 7.25 is needed.", ["number:7.25"]),
    ("At least 82.5% marks.", ["percent:82.5"]),
    ("Seventy five percent equals 75 per cent.", ["percent:75", "percent:75"]),
    ("At least seventy-five percent attendance.", ["percent:75"]),
])
def test_currency_and_decimal_parsing(text, expected):
    assert keys(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("Due on 15 November 2025.", ["date:2025-11-15"]),
    ("Due on November 15th, 2025.", ["date:2025-11-15"]),
    ("Due on 2025-11-15.", ["date:2025-11-15"]),
    ("Due on 15/11/2025.", ["date:2025-11-15"]),
    ("Due on 05/06/2025.", ["date_raw:5/6/2025"]),  # ambiguous day/month: not normalised
    ("Gates close at 5 p.m. and open at 08:30.", ["time:17:00", "time:08:30"]),
    ("Closed at midnight.", ["time:00:00"]),
])
def test_date_and_time_parsing(text, expected):
    assert keys(text) == expected


def test_equivalent_formats_match():
    assert rule_ids("Due on 15 November 2025.", "Due on November 15, 2025.") == set()
    assert rule_ids("The fee is Rs. 500.", "The fee is ₹500.") == set()
    assert rule_ids("Reply within 7 days.", "Reply within seven days.") == set()


def test_ambiguous_numeric_date_is_not_assumed_equal():
    # 05/06/2025 may be 5 June or May 6: one reading matches, so it is only a REVIEW
    # note, never silently treated as equal.
    assert severities("Due on 05/06/2025.", "Due on 5 June 2025.") == {"NUM-DATE-AMBIGUOUS": "REVIEW"}
    # Neither reading matches 7 June, so it is a real change.
    assert rule_ids("Due on 05/06/2025.", "Due on 7 June 2025.") == {"NUM-CHANGED"}


def test_equivalent_durations_are_review_only():
    assert severities("Submit within 2 weeks.", "Submit within 14 days.") == {"NUM-EQUIVALENT": "REVIEW"}
    assert severities("Reply within 48 hours.", "Reply within 2 days.") == {"NUM-EQUIVALENT": "REVIEW"}
    # 'working days' are not calendar days, so no conversion is assumed.
    assert "NUM-EQUIVALENT" not in rule_ids("Reply within 1 week.", "Reply within 7 working days.")


def test_military_time():
    assert keys("Report at 0900 hrs.") == ["time:09:00"]
    assert rule_ids("Report at 0900 hrs.", "Report at 9 am.") == set()


def test_year_only_in_one_sentence_is_review():
    assert severities("Due on 15 November.", "Due on 15 November 2025.") == {"NUM-DATE-PARTIAL": "REVIEW"}


def test_number_words_need_context():
    assert keys("No one may enter.") == []
    assert keys("Each team needs at least two members.") == ["number:2 member"]
    assert keys("Teams of at least two are allowed.") == ["number:2"]
