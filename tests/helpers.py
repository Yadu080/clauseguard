"""Small helpers shared by the test modules."""

from src import analyze


def rule_ids(original: str, rewritten: str) -> set[str]:
    return {f.rule_id for f in analyze(original, rewritten).findings}


def drift_categories(original: str, rewritten: str) -> set[str]:
    return analyze(original, rewritten).predicted_categories()


def severities(original: str, rewritten: str) -> dict[str, str]:
    return {f.rule_id: f.severity.value for f in analyze(original, rewritten).findings}
