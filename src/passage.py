"""Passage mode: compare two multi-sentence texts.

The sentence-level engine (analyzer.py) is unchanged. This module only
  1. splits each passage into sentences,
  2. aligns original sentences with rewritten sentences by shared vocabulary,
  3. runs analyze() on every aligned group, and
  4. reports sentences that have no counterpart (dropped or added).

Alignment allows one-to-one, merged (two originals -> one rewrite) and split
(one original -> two rewrites) groups: every sentence links to its single most
similar partner, and linked sentences form a group.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field

from .analyzer import analyze, overall_status
from .extractors import analyze_sentence, content_stems, tokenize
from .schemas import AnalysisResult, Finding, Severity, Status

# Abbreviations whose trailing period does not end a sentence.
ABBREVIATIONS = {
    "rs", "no", "nos", "dr", "mr", "mrs", "ms", "prof", "st", "vs", "etc", "eg", "e.g", "ie", "i.e",
    "approx", "dept", "sec", "fig", "govt", "univ", "jan", "feb", "mar", "apr", "jun", "jul", "aug",
    "sep", "sept", "oct", "nov", "dec", "a.m", "p.m", "am", "pm", "max", "min", "ref", "art", "cl",
}
_BULLET = re.compile(r"^\s*(?:[-•*·]|\(?\d{1,2}[.)]|\(?[a-zA-Z][.)]|\([ivx]+\))\s+")
_BOUNDARY = re.compile(r"[.!?;](?=\s+[\"“(]?[A-Z0-9])")

# A sentence links to its best partner only if they are at least this similar.
LINK_THRESHOLD = 0.3


def split_sentences(text: str) -> list[str]:
    """Split a passage into sentences. Line breaks and list bullets also separate
    items, because policies are often written as bullet lists."""
    sentences: list[str] = []
    for line in (text or "").splitlines():
        line = _BULLET.sub("", line).strip()
        if not line:
            continue
        start = 0
        for m in _BOUNDARY.finditer(line):
            before = line[start:m.start()].split()
            last = before[-1].lower().rstrip(".") if before else ""
            # 'Rs. 500', 'No. 4', 'a.m. on' and initials ('J. Smith') do not end a sentence.
            if last in ABBREVIATIONS or re.fullmatch(r"[a-z]", last) or re.fullmatch(r"(?:[a-z]\.)+[a-z]", last):
                continue
            piece = line[start:m.end()].strip()
            if piece:
                sentences.append(piece)
            start = m.end()
        tail = line[start:].strip()
        if tail:
            sentences.append(tail)
    return sentences


def similarity(a: set[str], b: set[str]) -> float:
    """Mean of Jaccard overlap and overlap coefficient. The overlap coefficient
    (|A∩B| / min(|A|,|B|)) keeps a short sentence that was merged into a long one
    close to it; Jaccard stops tiny sentences from matching everything."""
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return 0.5 * inter / len(a | b) + 0.5 * inter / min(len(a), len(b))


@dataclass
class Segment:
    kind: str  # aligned | merged | split | dropped | added
    original_sentences: list[str]
    rewritten_sentences: list[str]
    similarity: float | None
    result: AnalysisResult | None = None
    finding: Finding | None = None  # for dropped / added sentences

    @property
    def findings(self) -> list[Finding]:
        if self.result is not None:
            return self.result.findings
        return [self.finding] if self.finding else []


@dataclass
class PassageResult:
    original_sentences: list[str]
    rewritten_sentences: list[str]
    segments: list[Segment]
    status: Status
    notes: list[str] = field(default_factory=list)

    @property
    def findings(self) -> list[Finding]:
        return [f for s in self.segments for f in s.findings]

    def severity_counts(self) -> dict[str, int]:
        counts = {s.value: 0 for s in Severity}
        for f in self.findings:
            counts[f.severity.value] += 1
        return counts

    def to_dict(self) -> dict:
        segments = []
        for seg in self.segments:
            segments.append({
                "kind": seg.kind,
                "original_sentences": seg.original_sentences,
                "rewritten_sentences": seg.rewritten_sentences,
                "similarity": seg.similarity,
                "result": seg.result.to_dict() if seg.result else None,
                "finding": json.loads(json.dumps(asdict(seg.finding), default=str)) if seg.finding else None,
            })
        return {"mode": "passage", "status": self.status.value, "severity_counts": self.severity_counts(),
                "original_sentences": self.original_sentences,
                "rewritten_sentences": self.rewritten_sentences,
                "segments": segments, "notes": self.notes}

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


def _has_constraints(sentence: str) -> bool:
    a = analyze_sentence(sentence)
    return bool(a.markers or a.numbers)


def _unmatched_finding(sentence: str, dropped: bool) -> Finding:
    constrained = _has_constraints(sentence)
    severity = Severity.MEDIUM if constrained else Severity.REVIEW
    if dropped:
        return Finding(
            "PSG-SENTENCE-DROPPED", "coverage", severity,
            f"No sentence in the rewrite corresponds to: “{sentence}”",
            "The sentence contains rule markers or values, so dropping it may remove a constraint."
            if constrained else "A sentence without detected rule markers was left out; it may be unimportant.",
            "Check whether the rule in this sentence is still stated somewhere in the rewrite.")
    return Finding(
        "PSG-SENTENCE-ADDED", "coverage", severity,
        f"The rewrite adds a sentence with no counterpart in the original: “{sentence}”",
        "The new sentence contains rule markers or values that the source may not support."
        if constrained else "A new sentence without detected rule markers was added.",
        "Check that the added sentence is supported by the original text.")


def align(orig: list[str], rew: list[str]) -> list[tuple[list[int], list[int], float]]:
    """Group sentence indices. Each sentence links to its most similar partner on
    the other side (if similar enough); connected sentences form one group."""
    o_stems = [content_stems(tokenize(s)) for s in orig]
    r_stems = [content_stems(tokenize(s)) for s in rew]
    sim = [[similarity(a, b) for b in r_stems] for a in o_stems]

    parent = {("o", i): ("o", i) for i in range(len(orig))} | {("r", j): ("r", j) for j in range(len(rew))}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def link(x, y):
        parent[find(x)] = find(y)

    for i in range(len(orig)):
        if rew:
            j = max(range(len(rew)), key=lambda j: sim[i][j])
            if sim[i][j] >= LINK_THRESHOLD:
                link(("o", i), ("r", j))
    for j in range(len(rew)):
        if orig:
            i = max(range(len(orig)), key=lambda i: sim[i][j])
            if sim[i][j] >= LINK_THRESHOLD:
                link(("o", i), ("r", j))

    groups: dict = {}
    for node in parent:
        groups.setdefault(find(node), []).append(node)
    out = []
    for members in groups.values():
        o_idx = sorted(i for side, i in members if side == "o")
        r_idx = sorted(j for side, j in members if side == "r")
        best = max((sim[i][j] for i in o_idx for j in r_idx), default=0.0)
        out.append((o_idx, r_idx, round(best, 3)))
    # Present groups in the order of the original passage; added sentences by position.
    out.sort(key=lambda g: (g[0][0] if g[0] else (g[1][0] + 0.5 if g[1] else 0), g[1][:1]))
    return out


def analyze_passage(original: str, rewritten: str) -> PassageResult:
    orig, rew = split_sentences(original), split_sentences(rewritten)
    segments: list[Segment] = []
    if not orig or not rew:
        result = analyze(" ".join(orig), " ".join(rew))  # yields the INPUT-EMPTY finding
        segments.append(Segment("aligned", orig, rew, None, result=result))
        return PassageResult(orig, rew, segments, result.status, notes=list(result.notes))

    for o_idx, r_idx, best in align(orig, rew):
        o_text, r_text = [orig[i] for i in o_idx], [rew[j] for j in r_idx]
        if o_text and r_text:
            kind = "aligned" if len(o_text) == len(r_text) == 1 else ("merged" if len(o_text) > 1 else "split")
            if len(o_text) > 1 and len(r_text) > 1:
                kind = "regrouped"
            segments.append(Segment(kind, o_text, r_text, best,
                                    result=analyze(" ".join(o_text), " ".join(r_text))))
        elif o_text:
            for s in o_text:
                segments.append(Segment("dropped", [s], [], None, finding=_unmatched_finding(s, dropped=True)))
        else:
            for s in r_text:
                segments.append(Segment("added", [], [s], None, finding=_unmatched_finding(s, dropped=False)))

    all_findings = [f for s in segments for f in s.findings]
    notes = [
        "Passage mode aligns sentences by shared vocabulary before comparing them. A wrong "
        "alignment produces misleading warnings, so check the pairing shown for each group.",
    ]
    return PassageResult(orig, rew, segments, overall_status(all_findings), notes)
