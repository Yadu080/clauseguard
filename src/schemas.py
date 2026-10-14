"""Data structures shared by the engine, the evaluation script and the UI.

Everything here is a plain dataclass so that results can be inspected in a
debugger, compared in tests and serialised to JSON without extra libraries.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum

ENGINE_VERSION = "1.1.0"

# The five linguistic categories ClauseGuard checks. Findings outside these
# categories (e.g. "coverage", "input") are informational REVIEW notes.
CATEGORIES = ("negation", "modality", "condition", "quantifier", "numeric")


class Severity(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    REVIEW = "REVIEW"


SEVERITY_RANK = {Severity.HIGH: 0, Severity.MEDIUM: 1, Severity.REVIEW: 2}


class Status(str, Enum):
    DRIFT = "Potential meaning drift detected"
    NONE = "No configured checks triggered"
    REVIEW = "Manual review recommended"


@dataclass(frozen=True)
class Token:
    text: str
    start: int
    end: int

    @property
    def lower(self) -> str:
        return self.text.lower()


@dataclass(frozen=True)
class Marker:
    """A lexical marker found in a sentence (e.g. 'unless', 'must', 'not')."""

    category: str  # negation | modality | condition | quantifier
    label: str  # normalised class, e.g. OBLIGATION, EXCEPTION, LOWER_INCL
    text: str  # exact surface text as it appears in the sentence
    start: int
    end: int
    context: str = ""  # short phrase around the marker, used as evidence
    focus: str | None = None  # for negation: the word the cue attaches to
    detail: str | None = None  # extra info (e.g. positive base of 'ineligible')
    implicit: bool = False  # True for markers inferred from structure (imperatives)


@dataclass(frozen=True)
class NumericValue:
    """A number, percentage, amount, date, weekday or time found in a sentence."""

    kind: str  # percent | currency | date | weekday | time | number
    text: str  # exact surface text
    start: int
    end: int
    value: float | None  # numeric value when meaningful (None for dates/weekdays)
    unit: str | None  # '%', currency code, or a unit noun such as 'day'
    normalized: str  # comparison key, e.g. 'percent:75' or 'date:????-11-15'
    context: str = ""


@dataclass
class SentenceAnalysis:
    text: str
    tokens: list[Token]
    markers: list[Marker]
    numbers: list[NumericValue]

    def markers_in(self, category: str) -> list[Marker]:
        return [m for m in self.markers if m.category == category]


@dataclass(frozen=True)
class Evidence:
    text: str  # the exact matched expression
    start: int | None
    end: int | None
    context: str  # the surrounding phrase shown to the user


@dataclass
class Finding:
    rule_id: str
    category: str
    severity: Severity
    explanation: str  # what changed
    reason: str  # why the rule considers it important (severity rationale)
    suggested_action: str
    original_evidence: Evidence | None = None
    rewritten_evidence: Evidence | None = None


@dataclass
class AnalysisResult:
    original: str
    rewritten: str
    status: Status
    findings: list[Finding]
    original_analysis: SentenceAnalysis | None
    rewritten_analysis: SentenceAnalysis | None
    notes: list[str] = field(default_factory=list)
    engine_version: str = ENGINE_VERSION

    def severity_counts(self) -> dict[str, int]:
        counts = {s.value: 0 for s in Severity}
        for f in self.findings:
            counts[f.severity.value] += 1
        return counts

    def predicted_categories(self, include_review: bool = False) -> set[str]:
        """Categories with at least one warning at the chosen severity level."""
        allowed = {Severity.HIGH, Severity.MEDIUM}
        if include_review:
            allowed.add(Severity.REVIEW)
        return {
            f.category for f in self.findings
            if f.severity in allowed and f.category in CATEGORIES
        }

    def has_drift(self, include_review: bool = False) -> bool:
        allowed = {Severity.HIGH, Severity.MEDIUM}
        if include_review:
            allowed.add(Severity.REVIEW)
        return any(f.severity in allowed for f in self.findings)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["status"] = self.status.value
        data["severity_counts"] = self.severity_counts()
        for f in data["findings"]:
            f["severity"] = f["severity"].value if isinstance(f["severity"], Severity) else f["severity"]
        # Tokens are bulky and add nothing for a reader of the JSON export.
        for side in ("original_analysis", "rewritten_analysis"):
            if data[side] is not None:
                data[side].pop("tokens", None)
        return data

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
