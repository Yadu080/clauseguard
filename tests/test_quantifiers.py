from tests.helpers import drift_categories, rule_ids, severities


def test_universal_to_partial():
    assert severities("All students must carry ID cards.",
                      "Some students must carry ID cards.").get("QNT-SCOPE-CHANGED") == "MEDIUM"


def test_only_removed():
    assert "QNT-REMOVED" in rule_ids("Only final-year students are eligible.", "Final-year students are eligible.")


def test_bound_direction_reversed_is_high():
    assert severities("Applicants need at least 75% attendance.",
                      "Applicants need at most 75% attendance.").get("QNT-BOUND-REVERSED") == "HIGH"


def test_inclusive_to_strict_bound():
    assert "QNT-INCLUSIVENESS" in rule_ids("Students need at least 75% attendance.",
                                           "Students need more than 75% attendance.")


def test_before_to_after():
    assert "QNT-TIME-REVERSED" in rule_ids("Submit before Friday.", "Submit after Friday.")


def test_equivalent_bounds_not_flagged():
    assert drift_categories("Borrow no more than 2 books.", "Borrow at most 2 books.") == set()
    assert drift_categories("Every student must register.", "All students must register.") == set()


def test_dropping_all_from_bare_plural_is_review_only():
    assert severities("All students must register.", "Students must register.") == {"QNT-REMOVED": "REVIEW"}


def test_by_is_a_deadline_only_before_a_date():
    assert rule_ids("The form is signed by the HOD.", "The form is signed by the principal.") == set()
