"""Comparison rules: two SentenceAnalysis objects -> list of Findings.

Each `check_*` function handles one category and is independent of the others.
Severity is assigned by small, readable tables in this file:

    HIGH    likely change to negation, a condition/exception, a deontic
            reversal (e.g. prohibition -> permission) or a numerical value.
    MEDIUM  likely change to obligation strength, quantified scope, a bound
            or another important constraint.
    REVIEW  a marker changed or disappeared but the change is often a
            harmless paraphrase; the context is insufficient to judge.

Severity is a heuristic. It says how much attention a reviewer should give a
warning; it is not a probability that the meaning changed.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from .extractors import content_stems, phrase_around, stem
from .lexicon import DURATION_MINUTES, VERBAL_NEGATION_LABELS
from .schemas import Evidence, Finding, Marker, NumericValue, SentenceAnalysis, Severity

H, M, R = Severity.HIGH, Severity.MEDIUM, Severity.REVIEW

SEVERITY_POLICY = {
    "HIGH": "Likely change to negation, a condition or exception, a reversal between "
            "prohibition and permission/obligation, or a numerical value.",
    "MEDIUM": "Likely change to obligation strength, quantified scope, a numerical bound, "
              "or another important constraint.",
    "REVIEW": "A marker changed or disappeared, but such changes are often harmless "
              "paraphrases; the surrounding context is needed to judge.",
}

SUGGESTED_ACTION = {
    "negation": "Check whether the rewrite still says what is NOT allowed or NOT true in the original.",
    "modality": "Check whether the action is still required, recommended, permitted or prohibited as in the original.",
    "condition": "Check that every condition and exception of the original is preserved in the rewrite.",
    "quantifier": "Check that the scope (who/what is covered) and the bounds are unchanged.",
    "numeric": "Compare the values against the source document; restore the original value if it was changed.",
    "coverage": "Read both sentences in full; automatic checks may not have aligned the content.",
}

LABEL_NAMES = {
    # negation
    "NOT": "negation", "CANNOT": "negation", "NO": "negative determiner", "NEVER": "negation",
    "FAIL_TO": "negative verb", "WITHOUT": "negative preposition", "NEITHER": "negation", "NOR": "negation",
    "NEG_PRONOUN": "negative pronoun", "LEXICAL_NEG": "negative word",
    # modality
    "OBLIGATION": "obligation", "PROHIBITION": "prohibition", "PERMISSION": "permission",
    "RECOMMENDATION": "recommendation", "NON_OBLIGATION": "non-obligation (optional)",
    # condition
    "CONDITION": "condition", "CONDITION_ONLY": "necessary condition ('only if')",
    "EXCEPTION": "exception", "TEMPORAL": "temporal condition", "CONCESSION": "concession",
    # quantifier
    "LOWER_INCL": "inclusive lower bound", "LOWER_STRICT": "strict lower bound",
    "UPPER_INCL": "inclusive upper bound", "UPPER_STRICT": "strict upper bound",
    "BEFORE_INCL": "inclusive deadline", "BEFORE_STRICT": "strict deadline",
    "AFTER": "start time / after", "WITHIN": "time window",
    "UNIVERSAL": "universal quantifier", "ONLY": "restrictive 'only'", "PARTIAL": "partial quantifier",
}


def _name(label: str) -> str:
    return LABEL_NAMES.get(label, label.lower())


def _ev(item: Marker | NumericValue) -> Evidence:
    context = item.context
    if isinstance(item, Marker) and item.implicit:
        context = f"{context} (implicit imperative)"
    return Evidence(item.text, item.start, item.end, context)


def _finding(rule_id: str, category: str, severity: Severity, explanation: str, reason: str,
             orig: Evidence | None = None, rew: Evidence | None = None) -> Finding:
    return Finding(rule_id, category, severity, explanation, reason,
                   SUGGESTED_ACTION[category], orig, rew)


def _multiset_diff(old: Iterable, new: Iterable, key: Callable) -> tuple[list, list]:
    """Items of `old` with no equal-keyed partner in `new`, and vice versa.
    Order is preserved so evidence is reported left to right."""
    pool = list(new)
    removed = []
    for item in old:
        match = next((x for x in pool if key(x) == key(item)), None)
        if match is not None:
            pool.remove(match)
        else:
            removed.append(item)
    return removed, pool


# A classifier returns (rule_id, severity, explanation_template, reason) or None.
Spec = tuple[str, Severity, str, str]


def _compare_labelled(category: str, old: list[Marker], new: list[Marker],
                      on_change: Callable[[str, str], Spec | None],
                      on_removed: Callable[[str], Spec],
                      on_added: Callable[[str], Spec]) -> list[Finding]:
    """Shared logic for modality, condition and quantifier markers.

    1. Markers whose normalised label occurs in both sentences are matched and
       ignored (so 'unless' vs 'except when' is not a change).
    2. A removed marker is paired with an added marker when `on_change`
       recognises the pair (e.g. 'must' -> 'may').
    3. Anything left over is reported as removed or added.
    """
    removed, added = _multiset_diff(old, new, key=lambda m: m.label)
    findings = []
    for o in removed:
        partner = next((a for a in added if on_change(o.label, a.label)), None)
        if partner is not None:
            added.remove(partner)
            rule_id, sev, text, reason = on_change(o.label, partner.label)
            explanation = text.format(old=o.text, new=partner.text,
                                      old_name=_name(o.label), new_name=_name(partner.label))
            findings.append(_finding(rule_id, category, sev, explanation, reason, _ev(o), _ev(partner)))
        else:
            rule_id, sev, text, reason = on_removed(o.label)
            explanation = text.format(old=o.text, old_name=_name(o.label))
            findings.append(_finding(rule_id, category, sev, explanation, reason, _ev(o), None))
    for a in added:
        rule_id, sev, text, reason = on_added(a.label)
        explanation = text.format(new=a.text, new_name=_name(a.label))
        findings.append(_finding(rule_id, category, sev, explanation, reason, None, _ev(a)))
    return findings


# =============================================================================
# 1. Negation and polarity
# =============================================================================

def _same_focus(a: Marker, b: Marker) -> bool:
    return bool(a.focus and b.focus and stem(a.focus) == stem(b.focus))


def _loosely_compatible(a: Marker, b: Marker) -> bool:
    """'without' only stands in for another 'without'; other cues
    ('not', 'no', 'never', 'cannot') may stand in for each other."""
    return (a.label == "WITHOUT") == (b.label == "WITHOUT")


def _token_evidence(analysis: SentenceAnalysis, word: str | None) -> Evidence | None:
    if not word:
        return None
    for t in analysis.tokens:
        if t.text[0].isalpha() and stem(t.text) == stem(word):
            return Evidence(t.text, t.start, t.end, phrase_around(analysis.text, t.start, t.end))
    return None


def _negating_cue(cues: list[Marker], word: str) -> Marker | None:
    return next((c for c in cues if c.focus and stem(c.focus) == stem(word)), None)


def check_negation(o: SentenceAnalysis, r: SentenceAnalysis) -> list[Finding]:
    exp_o = [m for m in o.markers_in("negation") if m.label != "LEXICAL_NEG"]
    exp_r = [m for m in r.markers_in("negation") if m.label != "LEXICAL_NEG"]
    lex_o = [m for m in o.markers_in("negation") if m.label == "LEXICAL_NEG"]
    lex_r = [m for m in r.markers_in("negation") if m.label == "LEXICAL_NEG"]

    # Step 1: pair explicit cues that negate the same word ('not paid' ~ "haven't paid").
    pool = list(exp_r)
    unpaired = []
    for cue in exp_o:
        match = next((x for x in pool if _same_focus(cue, x)), None)
        if match:
            pool.remove(match)
        else:
            unpaired.append(cue)
    # Step 2: pair remaining cues one-to-one regardless of focus ('No student may
    # leave' ~ 'Students may not leave'). Count-preserving rewrites end up here.
    removed, loose_pairs = [], []
    for cue in unpaired:
        match = next((x for x in pool if _loosely_compatible(cue, x)), None)
        if match:
            pool.remove(match)
            loose_pairs.append((cue, match))
        else:
            removed.append(cue)
    added = pool

    findings: list[Finding] = []
    lex_removed, lex_added = _multiset_diff(lex_o, lex_r, key=lambda m: m.focus)

    def rephrased(cue: Marker, word: Marker, cue_in_original: bool) -> Finding:
        orig, rew = (cue, word) if cue_in_original else (word, cue)
        return _finding(
            "NEG-REPHRASED", "negation", R,
            f"Negation appears to be expressed differently: '{orig.context}' vs '{rew.context}'.",
            "An explicit negation was replaced by a negative word (or vice versa). This is "
            "often a faithful paraphrase, but polarity should be confirmed by a reader.",
            _ev(orig), _ev(rew))

    # Step 3: lexical negatives, e.g. 'eligible' -> 'ineligible'.
    def lexical_pass(lex_new: list[Marker], other: SentenceAnalysis, other_cues: list[Marker],
                     lex_in_original: bool) -> list[Marker]:
        leftovers = []
        for word in lex_new:
            bases = [b for b in (word.detail or "").split("|") if b]
            base_ev = next((e for e in (_token_evidence(other, b) for b in bases) if e), None)
            if base_ev is None:
                leftovers.append(word)
                continue
            cue = _negating_cue(other_cues, base_ev.text)
            if cue is not None:  # 'not accepted' <-> 'rejected'
                other_cues.remove(cue)
                findings.append(rephrased(cue, word, cue_in_original=not lex_in_original))
                continue
            orig_ev, rew_ev = (_ev(word), base_ev) if lex_in_original else (base_ev, _ev(word))
            findings.append(_finding(
                "NEG-ANTONYM", "negation", H,
                f"Polarity reversed by an antonym: '{orig_ev.text}' in the original vs "
                f"'{rew_ev.text}' in the rewrite.",
                "A word and its negative counterpart (e.g. eligible/ineligible) appear on "
                "opposite sides, which usually reverses the statement.",
                orig_ev, rew_ev))
        return leftovers

    lex_added = lexical_pass(lex_added, o, removed, lex_in_original=False)
    lex_removed = lexical_pass(lex_removed, r, added, lex_in_original=True)

    # Step 4: an explicit negation traded for an unrelated negative word.
    for word in lex_added:
        if removed:
            findings.append(rephrased(removed.pop(0), word, cue_in_original=True))
        else:
            findings.append(_finding(
                "NEG-LEXICAL", "negation", R,
                f"The negative word '{word.text}' appears only in the rewrite.",
                "Negative-polarity words can reverse a statement, but the context is needed to judge.",
                None, _ev(word)))
    for word in lex_removed:
        if added:
            findings.append(rephrased(added.pop(0), word, cue_in_original=False))
        else:
            findings.append(_finding(
                "NEG-LEXICAL", "negation", R,
                f"The negative word '{word.text}' from the original does not appear in the rewrite.",
                "Negative-polarity words can reverse a statement, but the context is needed to judge.",
                _ev(word), None))

    # Step 5: explicit negations that are simply missing or new.
    for cue in removed:
        rew_ev = _token_evidence(r, cue.focus)
        tail = (f" '{rew_ev.text}' appears in the rewrite without this negation." if rew_ev else "")
        findings.append(_finding(
            "NEG-REMOVED", "negation", H,
            f"The original negation '{cue.text}' ('{cue.context}') has no counterpart in the rewrite.{tail}",
            "Removing a negation usually reverses the polarity of the clause it belongs to.",
            _ev(cue), rew_ev))
    for cue in added:
        orig_ev = _token_evidence(o, cue.focus)
        tail = (f" '{orig_ev.text}' is not negated in the original." if orig_ev else "")
        findings.append(_finding(
            "NEG-ADDED", "negation", H,
            f"The rewrite introduces the negation '{cue.text}' ('{cue.context}').{tail}",
            "Adding a negation usually reverses the polarity of the clause it belongs to.",
            orig_ev, _ev(cue)))

    # Step 6: same number of negations, but attached to different verbs
    # ('who have not paid can attend' -> 'who have paid cannot attend').
    o_stems, r_stems = content_stems(o.tokens), content_stems(r.tokens)
    for a, b in loose_pairs:
        if (a.label in VERBAL_NEGATION_LABELS and b.label in VERBAL_NEGATION_LABELS
                and a.focus and b.focus and stem(a.focus) != stem(b.focus)
                and stem(a.focus) in r_stems and stem(b.focus) in o_stems):
            findings.append(_finding(
                "NEG-MOVED", "negation", H,
                f"The negation moved from '{a.focus}' ('{a.context}') to '{b.focus}' ('{b.context}').",
                "Both words occur in both sentences, but a different one is negated, which "
                "changes what is being denied.",
                _ev(a), _ev(b)))
    return findings


# =============================================================================
# 2. Obligation, permission, recommendation
# =============================================================================

MODAL_STRENGTH = {"OBLIGATION": 3, "RECOMMENDATION": 2, "PERMISSION": 1, "NON_OBLIGATION": 1}


def _modal_change(old: str, new: str) -> Spec:
    if "PROHIBITION" in (old, new) or {old, new} == {"OBLIGATION", "NON_OBLIGATION"}:
        return ("MOD-REVERSED", H,
                "The {old_name} '{old}' became a {new_name} '{new}'.",
                "Switching between prohibition and permission/obligation, or between "
                "'required' and 'not required', can reverse what the reader may do.")
    if {old, new} == {"PERMISSION", "NON_OBLIGATION"}:
        return ("MOD-SHIFTED", R,
                "The {old_name} '{old}' became a {new_name} '{new}'.",
                "Both forms leave the action optional, but the emphasis differs.")
    if MODAL_STRENGTH[new] < MODAL_STRENGTH[old]:
        return ("MOD-WEAKENED", M,
                "The {old_name} '{old}' was weakened to a {new_name} '{new}'.",
                "A weaker modal can turn a requirement into a choice.")
    return ("MOD-STRENGTHENED", M,
            "The {old_name} '{old}' was strengthened to a {new_name} '{new}'.",
            "A stronger modal can turn a choice into a requirement.")


def _modal_removed(label: str) -> Spec:
    if label == "PROHIBITION":
        return ("MOD-REMOVED", H, "The {old_name} '{old}' has no counterpart in the rewrite.",
                "Dropping a prohibition can make a forbidden action appear allowed.")
    if label == "PERMISSION":
        return ("MOD-REMOVED", R, "The {old_name} '{old}' has no counterpart in the rewrite.",
                "Permission is often implicit in a plain statement; the context is needed to judge.")
    return ("MOD-REMOVED", M, "The {old_name} '{old}' has no counterpart in the rewrite.",
            "Without an explicit modal the rewrite may read as a description rather than a rule.")


def _modal_added(label: str) -> Spec:
    if label == "PROHIBITION":
        return ("MOD-ADDED", H, "The rewrite introduces the {new_name} '{new}'.",
                "A new prohibition can forbid something the original allowed.")
    return ("MOD-ADDED", R, "The rewrite introduces the {new_name} '{new}'.",
            "Explicit modals are often added to make an implicit rule explicit; "
            "check that the force matches the original intent.")


def check_modality(o: SentenceAnalysis, r: SentenceAnalysis) -> list[Finding]:
    return _compare_labelled("modality", o.markers_in("modality"), r.markers_in("modality"),
                             lambda a, b: _modal_change(a, b), _modal_removed, _modal_added)


# =============================================================================
# 3. Conditions and exceptions
# =============================================================================

_CONDITIONAL = {"CONDITION", "CONDITION_ONLY"}


def _condition_change(old: str, new: str) -> Spec | None:
    if {old, new} == _CONDITIONAL:
        return ("COND-ONLYIF-CHANGED", M,
                "The {old_name} '{old}' became a {new_name} '{new}'.",
                "'only if' states a necessary condition; 'if' states a sufficient one.")
    if old == "EXCEPTION" and new in _CONDITIONAL:
        return ("COND-EXCEPTION-TO-CONDITION", H,
                "The exception '{old}' became the condition '{new}'.",
                "'unless X' means 'if not X'; replacing it with 'if X' usually reverses the rule.")
    if old in _CONDITIONAL and new == "EXCEPTION":
        return ("COND-CONDITION-TO-EXCEPTION", H,
                "The condition '{old}' became the exception '{new}'.",
                "'if X' and 'unless X' have opposite polarity.")
    if {old, new} == {"CONDITION", "TEMPORAL"}:
        return ("COND-TEMPORAL-SWAP", R,
                "The {old_name} '{old}' became a {new_name} '{new}'.",
                "'if' and 'when' are often interchangeable in policies, but not always.")
    return None


def _condition_removed(label: str) -> Spec:
    if label == "EXCEPTION":
        return ("COND-EXCEPTION-REMOVED", H,
                "The exception '{old}' has no counterpart in the rewrite.",
                "Dropping an exception makes the rule apply more broadly than the original.")
    if label in _CONDITIONAL:
        return ("COND-CONDITION-REMOVED", H,
                "The {old_name} '{old}' has no counterpart in the rewrite.",
                "Dropping a condition can turn a conditional rule into an unconditional one.")
    return ("COND-CLAUSE-REMOVED", R,
            "The {old_name} '{old}' has no counterpart in the rewrite.",
            "This clause type is often rephrased, but it may carry a constraint.")


def _condition_added(label: str) -> Spec:
    if label == "EXCEPTION":
        return ("COND-EXCEPTION-ADDED", M, "The rewrite introduces the exception '{new}'.",
                "A new exception can create a loophole that the original did not have.")
    if label in _CONDITIONAL:
        return ("COND-CONDITION-ADDED", M, "The rewrite introduces the {new_name} '{new}'.",
                "A new condition can restrict a rule that was unconditional in the original.")
    return ("COND-CLAUSE-ADDED", R, "The rewrite introduces the {new_name} '{new}'.",
            "This clause type is often added during rephrasing, but it may add a constraint.")


def check_conditions(o: SentenceAnalysis, r: SentenceAnalysis) -> list[Finding]:
    return _compare_labelled("condition", o.markers_in("condition"), r.markers_in("condition"),
                             _condition_change, _condition_removed, _condition_added)


# =============================================================================
# 4. Quantifiers, bounds and scope
# =============================================================================

_FAMILY = {
    "LOWER_INCL": "lower", "LOWER_STRICT": "lower",
    "UPPER_INCL": "upper", "UPPER_STRICT": "upper",
    "BEFORE_INCL": "before", "BEFORE_STRICT": "before",
    "AFTER": "after", "WITHIN": "within",
    "UNIVERSAL": "scope", "ONLY": "scope", "PARTIAL": "scope",
}


def _quantifier_change(old: str, new: str) -> Spec | None:
    fo, fn = _FAMILY[old], _FAMILY[new]
    if fo == fn and fo in {"lower", "upper", "before"}:
        return ("QNT-INCLUSIVENESS", M,
                "The {old_name} '{old}' became a {new_name} '{new}'.",
                "Inclusive and strict bounds differ exactly at the threshold "
                "(e.g. 'at least 75%' includes 75%, 'more than 75%' does not).")
    if {fo, fn} == {"lower", "upper"}:
        return ("QNT-BOUND-REVERSED", H,
                "The {old_name} '{old}' became an {new_name} '{new}'.",
                "Reversing the direction of a bound inverts who satisfies the rule.")
    if {fo, fn} in ({"before", "after"}, {"within", "after"}):
        return ("QNT-TIME-REVERSED", H,
                "The time expression '{old}' became '{new}'.",
                "Before/after reversal changes when the action is allowed or required.")
    if {fo, fn} == {"within", "before"}:
        return ("QNT-TIME-CHANGED", R,
                "The time expression '{old}' became '{new}'.",
                "'within' and 'before' can describe the same deadline; check the reference point.")
    if fo == fn == "scope":
        return ("QNT-SCOPE-CHANGED", M,
                "The {old_name} '{old}' became a {new_name} '{new}'.",
                "Changing the quantifier changes who or what the rule covers.")
    return None


def _quantifier_removed(label: str) -> Spec:
    if label == "UNIVERSAL":
        return ("QNT-REMOVED", R, "The {old_name} '{old}' has no counterpart in the rewrite.",
                "A bare plural ('Students must ...') often keeps the universal reading.")
    return ("QNT-REMOVED", M, "The {old_name} '{old}' has no counterpart in the rewrite.",
            "Dropping a bound or scope word can widen or narrow the rule.")


def _quantifier_added(label: str) -> Spec:
    if label == "UNIVERSAL":
        return ("QNT-ADDED", R, "The rewrite introduces the {new_name} '{new}'.",
                "Adding 'all/every' often only makes a generic statement explicit.")
    return ("QNT-ADDED", M, "The rewrite introduces the {new_name} '{new}'.",
            "A new bound or scope word can narrow or widen the rule.")


def check_quantifiers(o: SentenceAnalysis, r: SentenceAnalysis) -> list[Finding]:
    return _compare_labelled("quantifier", o.markers_in("quantifier"), r.markers_in("quantifier"),
                             _quantifier_change, _quantifier_removed, _quantifier_added)


# =============================================================================
# 5. Numbers, percentages, amounts, dates and times
# =============================================================================

def _without_year(key: str) -> str:
    return key[:len("date:")] + key[len("date:????"):] if key.startswith("date:") else key


def _date_readings(key: str) -> set[str]:
    """Both possible readings of an ambiguous numeric date key 'date_raw:5/6/2025'."""
    if not key.startswith("date_raw:"):
        return {key}
    a, b, year = key[len("date_raw:"):].split("/")
    return {f"date:{int(year):04d}-{int(b):02d}-{int(a):02d}",   # day first
            f"date:{int(year):04d}-{int(a):02d}-{int(b):02d}"}   # month first


def _minutes(n: NumericValue) -> float | None:
    """Duration in minutes when the unit converts exactly ('2 weeks' -> 20160)."""
    if n.kind != "number" or n.value is None or n.unit not in DURATION_MINUTES:
        return None
    return n.value * DURATION_MINUTES[n.unit]


def check_numbers(o: SentenceAnalysis, r: SentenceAnalysis) -> list[Finding]:
    removed, added = _multiset_diff(o.numbers, r.numbers, key=lambda n: n.normalized)
    findings: list[Finding] = []

    def take(pred: Callable[[NumericValue, NumericValue], bool]):
        for a in list(removed):
            b = next((x for x in added if pred(a, x)), None)
            if b is not None:
                removed.remove(a)
                added.remove(b)
                yield a, b

    # (a0) An ambiguous numeric date ('01/02/2025') with one reading that matches.
    for a, b in take(lambda a, b: a.kind == b.kind == "date" and a.normalized != b.normalized
                     and "date_raw:" in (a.normalized + b.normalized)
                     and bool(_date_readings(a.normalized) & _date_readings(b.normalized))):
        raw = a if a.normalized.startswith("date_raw:") else b
        findings.append(_finding(
            "NUM-DATE-AMBIGUOUS", "numeric", R,
            f"'{raw.text}' can be read day-first or month-first; one reading matches "
            f"'{(b if raw is a else a).text}'.",
            "Numeric dates such as 01/02/2025 are not normalised automatically because the "
            "day/month order is ambiguous. Confirm which convention the source uses.",
            _ev(a), _ev(b)))
    # (a1) The same duration in different units ('2 weeks' -> '14 days').
    for a, b in take(lambda a, b: _minutes(a) is not None and _minutes(a) == _minutes(b)
                     and a.unit != b.unit):
        findings.append(_finding(
            "NUM-EQUIVALENT", "numeric", R,
            f"{a.text!r} and {b.text!r} describe the same duration.",
            "The units convert exactly (minutes, hours, days, weeks), so the value is "
            "probably unchanged; flagged only so that a reader can confirm.",
            _ev(a), _ev(b)))
    # (a) Same date, but the year is given in only one sentence.
    for a, b in take(lambda a, b: a.kind == b.kind == "date" and "????" in (a.normalized + b.normalized)
                     and _without_year(a.normalized) == _without_year(b.normalized)):
        findings.append(_finding(
            "NUM-DATE-PARTIAL", "numeric", R,
            f"The date '{a.text}' and '{b.text}' agree on day and month, but only one gives a year.",
            "The dates probably refer to the same day, but this cannot be confirmed.",
            _ev(a), _ev(b)))
    # (b) Same number, different unit ('7 days' -> '7 weeks', 'Rs. 10 per day' ->
    #     'Rs. 10 per week'). For amounts the unit is the currency plus optional rate.
    for a, b in take(lambda a, b: a.kind == b.kind and a.kind in ("number", "currency")
                     and a.value == b.value):
        if a.kind == "currency":
            rates_differ = "/" in a.unit and "/" in b.unit
            currencies_differ = a.unit.split("/")[0] != b.unit.split("/")[0]
            both = rates_differ or currencies_differ
        else:
            both = bool(a.unit and b.unit)
        findings.append(_finding(
            "NUM-UNIT-CHANGED" if both else "NUM-UNIT-PARTIAL", "numeric", H if both else R,
            f"The value {a.text!r} appears as {b.text!r} in the rewrite.",
            "The number is the same but its unit differs, which changes the quantity."
            if both else "The unit is stated in only one sentence.",
            _ev(a), _ev(b)))
    # (c) Same value written as a different kind ('Rs. 500' vs '500').
    for a, b in take(lambda a, b: a.kind != b.kind and a.value is not None and a.value == b.value):
        findings.append(_finding(
            "NUM-FORMAT", "numeric", R,
            f"The value {a.text!r} ({a.kind}) appears as {b.text!r} ({b.kind}) in the rewrite.",
            "The figure is the same but its type or unit is expressed differently.",
            _ev(a), _ev(b)))
    # (d) Same kind of value, different value ('75%' -> '70%').
    for a, b in take(lambda a, b: a.kind == b.kind):
        findings.append(_finding(
            "NUM-CHANGED", "numeric", H,
            f"The {a.kind} {a.text!r} in the original appears as {b.text!r} in the rewrite.",
            "A changed threshold, amount, date or time directly changes who or what the rule applies to.",
            _ev(a), _ev(b)))
    for a in removed:
        findings.append(_finding(
            "NUM-MISSING", "numeric", H,
            f"The {a.kind} {a.text!r} ('{a.context}') does not appear in the rewrite.",
            "A value from the source was dropped, so the rewrite may no longer state the limit.",
            _ev(a), None))
    for b in added:
        findings.append(_finding(
            "NUM-ADDED", "numeric", M,
            f"The rewrite introduces the {b.kind} {b.text!r} ('{b.context}').",
            "A value that is not in the source may be invented or come from elsewhere.",
            None, _ev(b)))
    return findings


# =============================================================================
# 6. Coverage: are the two sentences even about the same thing?
# =============================================================================

LOW_OVERLAP_THRESHOLD = 0.25


_NOT_CONTENT = {"am", "pm", "hr", "hrs", "a.m", "p.m"}


def _word_stems(analysis: SentenceAnalysis) -> set[str]:
    """Content-word stems without numbers or time abbreviations; values are
    compared by check_numbers, so they should not count as shared vocabulary."""
    return {w for w in content_stems(analysis.tokens) if w[0].isalpha() and w not in _NOT_CONTENT}


def check_coverage(o: SentenceAnalysis, r: SentenceAnalysis) -> list[Finding]:
    a, b = _word_stems(o), _word_stems(r)
    if len(a) < 3 or len(b) < 3:
        return []
    overlap = len(a & b) / len(a | b)
    if overlap >= LOW_OVERLAP_THRESHOLD:
        return []
    return [_finding(
        "COV-LOW-OVERLAP", "coverage", R,
        f"The sentences share only {overlap:.0%} of their content words (Jaccard overlap).",
        "With little shared vocabulary the marker comparison may be comparing unrelated "
        "clauses, and paraphrases outside the lexicon will not be recognised.")]


RULES = [check_negation, check_modality, check_conditions, check_quantifiers, check_numbers, check_coverage]
