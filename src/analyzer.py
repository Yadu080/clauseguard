"""Public entry point: analyze(original, rewritten) -> AnalysisResult.

Pipeline
    1. normalise both texts and reject empty input
    2. describe each sentence (tokens, markers, numbers)      -> extractors.py
    3. run every comparison rule                              -> rules.py
    4. merge findings that together form a known paraphrase   -> this file
    5. sort by severity and decide the overall status
"""

from __future__ import annotations

from .extractors import analyze_sentence, normalize_text
from .rules import RULES, SUGGESTED_ACTION
from .schemas import SEVERITY_RANK, AnalysisResult, Finding, Severity, Status

LIMITATION_NOTE = (
    "ClauseGuard compares lexical markers and values. A warning is a recommendation to "
    "review, not proof that the meaning changed; the absence of warnings is not proof "
    "that the meaning is preserved."
)


def _is_negating_cue(text: str) -> bool:
    t = text.lower()
    return t in {"not", "never"} or t.endswith("n't") or t.replace(" ", "") == "cannot"


# (rule that fires on 'unless', side holding 'unless', rule that fires on the
#  negation, side holding the negation, test for the negation word)
_UNLESS_PATTERNS = [
    # 'unless X' -> 'without X'
    ("COND-EXCEPTION-REMOVED", "original_evidence", "NEG-ADDED", "rewritten_evidence",
     lambda t: t.lower() == "without"),
    # 'unless X' -> 'if not X'
    ("COND-EXCEPTION-TO-CONDITION", "original_evidence", "NEG-ADDED", "rewritten_evidence",
     _is_negating_cue),
    # 'without X' -> 'unless X'
    ("COND-EXCEPTION-ADDED", "rewritten_evidence", "NEG-REMOVED", "original_evidence",
     lambda t: t.lower() == "without"),
    # 'if not X' -> 'unless X'
    ("COND-CONDITION-TO-EXCEPTION", "rewritten_evidence", "NEG-REMOVED", "original_evidence",
     _is_negating_cue),
]


def _first(findings: list[Finding], rule_id: str, side: str, test) -> Finding | None:
    for f in findings:
        ev = getattr(f, side)
        if f.rule_id == rule_id and ev is not None and test(ev.text):
            return f
    return None


def _merge_unless_paraphrases(findings: list[Finding]) -> list[Finding]:
    """Validated pattern: 'unless X' ~ 'without X' ~ 'if ... not X'.

    'cannot appear unless the HOD approves' vs 'cannot appear without the HOD's
    approval' triggers two HIGH findings (exception removed, negation added) that
    cancel each other out. When both are present they are replaced by a single
    REVIEW finding. The reverse direction is handled the same way.
    """
    findings = list(findings)
    for exc_rule, exc_side, neg_rule, neg_side, neg_test in _UNLESS_PATTERNS:
        while True:
            exc = _first(findings, exc_rule, exc_side, lambda t: t.lower() == "unless")
            neg = _first(findings, neg_rule, neg_side, neg_test)
            if exc is None or neg is None:
                break
            findings = [f for f in findings if f is not exc and f is not neg]
            o_ev = exc.original_evidence if exc_side == "original_evidence" else neg.original_evidence
            r_ev = exc.rewritten_evidence if exc_side == "rewritten_evidence" else neg.rewritten_evidence
            findings.append(Finding(
                "COND-REPHRASED", "condition", Severity.REVIEW,
                f"The exception appears to be rephrased: '{o_ev.context}' vs '{r_ev.context}'.",
                "'unless X' is usually equivalent to 'without X' or 'if not X'. The two "
                "individual changes (exception and negation) cancel out, so this is only "
                "flagged for review.",
                SUGGESTED_ACTION["condition"], o_ev, r_ev))
    return findings


def overall_status(findings: list[Finding]) -> Status:
    if any(f.severity in (Severity.HIGH, Severity.MEDIUM) for f in findings):
        return Status.DRIFT
    if findings:
        return Status.REVIEW
    return Status.NONE


def analyze(original: str, rewritten: str) -> AnalysisResult:
    original_clean, rewritten_clean = normalize_text(original), normalize_text(rewritten)

    if not original_clean or not rewritten_clean:
        missing = " and ".join(name for name, text in (("original", original_clean),
                                                       ("rewritten", rewritten_clean)) if not text)
        finding = Finding(
            "INPUT-EMPTY", "input", Severity.REVIEW,
            f"The {missing} sentence is empty, so no comparison was made.",
            "Checks need two non-empty sentences.",
            "Enter both the original and the rewritten sentence.")
        return AnalysisResult(original_clean, rewritten_clean, Status.REVIEW, [finding],
                              None, None, notes=[LIMITATION_NOTE])

    o, r = analyze_sentence(original_clean), analyze_sentence(rewritten_clean)
    findings: list[Finding] = []
    for rule in RULES:
        findings.extend(rule(o, r))
    findings = _merge_unless_paraphrases(findings)

    def position(f: Finding) -> int:
        ev = f.original_evidence or f.rewritten_evidence
        return ev.start if ev and ev.start is not None else 10**6

    findings.sort(key=lambda f: (SEVERITY_RANK[f.severity], position(f)))

    notes = [LIMITATION_NOTE]
    if not o.markers and not o.numbers:
        notes.append("No configured markers or values were found in the original sentence, "
                     "so the checks have very little to compare.")
    if original_clean == rewritten_clean:
        notes.append("The two sentences are identical after whitespace normalisation.")
    return AnalysisResult(original_clean, rewritten_clean, overall_status(findings), findings, o, r, notes)
