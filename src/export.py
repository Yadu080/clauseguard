"""Human-readable Markdown reports for an analysis (sentence or passage mode)."""

from __future__ import annotations

from .passage import PassageResult
from .rules import SEVERITY_POLICY
from .schemas import AnalysisResult, Evidence, Finding

_DISCLAIMER = ("_ClauseGuard reports potential meaning drift. Warnings are review recommendations, "
               "not proof; no warnings is not proof that the meaning is preserved._")


def _ev(ev: Evidence | None) -> str:
    if ev is None:
        return "_no counterpart found_"
    return f"**{ev.text}** — “{ev.context}”"


def _finding_md(f: Finding, n: int) -> list[str]:
    return [
        f"{n}. **{f.severity.value}** · {f.category} · `{f.rule_id}`  ",
        f"   {f.explanation}  ",
        f"   - Original evidence: {_ev(f.original_evidence)}",
        f"   - Rewritten evidence: {_ev(f.rewritten_evidence)}",
        f"   - Why flagged: {f.reason}",
        f"   - Review: {f.suggested_action}",
        "",
    ]


def _findings_md(findings: list[Finding]) -> list[str]:
    if not findings:
        return ["No configured check fired.", ""]
    lines: list[str] = []
    for i, f in enumerate(findings, start=1):
        lines += _finding_md(f, i)
    return lines


def result_to_markdown(result: AnalysisResult) -> str:
    c = result.severity_counts()
    lines = [
        "# ClauseGuard report", "",
        f"**Status:** {result.status.value}  ",
        f"**Warnings:** {c['HIGH']} high · {c['MEDIUM']} medium · {c['REVIEW']} review  ",
        f"**Engine:** {result.engine_version}", "",
        "| | Sentence |", "|---|---|",
        f"| Original | {result.original} |",
        f"| Rewritten | {result.rewritten} |", "",
        "## Warnings", "",
    ]
    lines += _findings_md(result.findings)
    lines += ["## Severity levels", ""] + [f"- **{k}** — {v}" for k, v in SEVERITY_POLICY.items()]
    lines += ["", _DISCLAIMER, ""]
    return "\n".join(lines)


def passage_to_markdown(result: PassageResult) -> str:
    c = result.severity_counts()
    lines = [
        "# ClauseGuard passage report", "",
        f"**Status:** {result.status.value}  ",
        f"**Warnings:** {c['HIGH']} high · {c['MEDIUM']} medium · {c['REVIEW']} review  ",
        f"**Sentences:** {len(result.original_sentences)} original · {len(result.rewritten_sentences)} rewritten", "",
    ]
    for n, seg in enumerate(result.segments, start=1):
        lines += [f"## Group {n} — {seg.kind}", ""]
        for s in seg.original_sentences:
            lines.append(f"> **Original:** {s}  ")
        for s in seg.rewritten_sentences:
            lines.append(f"> **Rewritten:** {s}  ")
        lines.append("")
        lines += _findings_md(seg.findings)
    lines += result.notes + ["", _DISCLAIMER, ""]
    return "\n".join(lines)
