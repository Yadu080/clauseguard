"""Turn an analysed sentence into HTML with highlighted markers.

Kept separate from the UI so it can be unit-tested and reused. The output is
plain HTML with CSS classes; the styles live in app.py.

    cg-mark cg-<category>   a detected marker or value
    cg-flag-<severity>      the marker is evidence for a warning
"""

from __future__ import annotations

import html

from .schemas import AnalysisResult, Finding, SentenceAnalysis

Span = tuple[int, int, str, str]  # start, end, css classes, tooltip


def _flag_index(findings: list[Finding], side: str) -> dict[tuple[int, int], str]:
    """(start, end) of every evidence span on one side -> highest severity."""
    flags: dict[tuple[int, int], str] = {}
    for f in findings:  # findings are sorted most severe first
        ev = f.original_evidence if side == "original" else f.rewritten_evidence
        if ev is not None and ev.start is not None:
            flags.setdefault((ev.start, ev.end), f.severity.value.lower())
    return flags


def collect_spans(analysis: SentenceAnalysis, findings: list[Finding], side: str) -> list[Span]:
    flags = _flag_index(findings, side)
    spans: list[Span] = []
    used: set[tuple[int, int]] = set()
    for m in analysis.markers:
        flag = flags.get((m.start, m.end))
        used.add((m.start, m.end))
        cls = f"cg-mark cg-{m.category}" + (f" cg-flag-{flag}" if flag else "")
        spans.append((m.start, m.end, cls, f"{m.category}: {m.label}"))
    for n in analysis.numbers:
        flag = flags.get((n.start, n.end))
        used.add((n.start, n.end))
        cls = "cg-mark cg-numeric" + (f" cg-flag-{flag}" if flag else "")
        spans.append((n.start, n.end, cls, f"{n.kind}: {n.normalized}"))
    # Evidence that is not itself a marker (e.g. the word 'paid' that lost its negation).
    for (start, end), flag in flags.items():
        if (start, end) in used:
            continue
        spans.append((start, end, f"cg-mark cg-context cg-flag-{flag}", "evidence for a warning"))
    return spans


def render(text: str, spans: list[Span]) -> str:
    """Wrap spans in <mark> tags. Overlapping spans are resolved by keeping
    flagged spans first, then longer spans; the rest are dropped."""
    ordered = sorted(spans, key=lambda s: ("cg-flag" not in s[2], -(s[1] - s[0]), s[0]))
    chosen: list[Span] = []
    for span in ordered:
        if all(span[1] <= c[0] or span[0] >= c[1] for c in chosen):
            chosen.append(span)
    chosen.sort(key=lambda s: s[0])

    out, pos = [], 0
    for start, end, cls, title in chosen:
        out.append(html.escape(text[pos:start]))
        out.append(f'<mark class="{cls}" title="{html.escape(title)}">{html.escape(text[start:end])}</mark>')
        pos = end
    out.append(html.escape(text[pos:]))
    return "".join(out)


def highlight(result: AnalysisResult, side: str) -> str:
    analysis = result.original_analysis if side == "original" else result.rewritten_analysis
    if analysis is None:
        return html.escape(result.original if side == "original" else result.rewritten)
    return render(analysis.text, collect_spans(analysis, result.findings, side))


def missing_in_rewrite(result: AnalysisResult) -> list[Finding]:
    """Warnings whose evidence exists only in the original (nothing to highlight
    on the rewritten side), shown as 'missing' chips in the UI."""
    return [f for f in result.findings if f.original_evidence and not f.rewritten_evidence]
