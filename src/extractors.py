"""Preprocessing and extraction: text -> tokens, lexical markers and numbers.

Nothing in this module compares two sentences. It only describes one sentence
so that `rules.py` can compare two descriptions.
"""

from __future__ import annotations

import re

from . import lexicon as lx
from .schemas import Marker, NumericValue, SentenceAnalysis, Token

# --- preprocessing -------------------------------------------------------------

_QUOTE_MAP = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"',
                            "–": "-", "—": "-", " ": " "})

# Words, contractions ('can't'), hyphenated words, numbers with , or . inside,
# currency symbols and single punctuation characters.
_TOKEN_RE = re.compile(r"\d+(?:[.,:/]\d+)*%?|[A-Za-z]+(?:[-'][A-Za-z]+)*|[₹$€£%]|[^\w\s]")

STOPWORDS = {
    "a", "an", "the", "of", "to", "in", "on", "for", "at", "by", "with", "and", "or",
    "is", "are", "was", "were", "be", "been", "being", "it", "its", "this", "that",
    "these", "those", "who", "which", "their", "they", "them", "his", "her", "he",
    "she", "as", "from", "has", "have", "had", "will", "would", "do", "does", "did",
    "so", "than", "then", "there", "into", "per", "your", "you", "our", "we",
}
# Words skipped when looking for the word a negation cue attaches to.
_FOCUS_SKIP = STOPWORDS | {"any", "yet", "also", "always", "even", "ever", "get", "got",
                           "own", "able", "please", "kindly", "currently", "necessarily"}


def normalize_text(text: str) -> str:
    """Unify quotes/dashes and collapse runs of whitespace."""
    if text is None:
        return ""
    return re.sub(r"\s+", " ", text.translate(_QUOTE_MAP)).strip()


def tokenize(text: str) -> list[Token]:
    return [Token(m.group(0), m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]


def stem(word: str) -> str:
    """A deliberately tiny suffix stripper, sufficient for matching
    'submit/submitted' or 'student/students'. It is not a linguistic stemmer."""
    w = word.lower()
    for suffix in ("ing", "ed", "es", "s"):
        if w.endswith(suffix) and len(w) - len(suffix) >= 3 and not w.endswith("ss"):
            w = w[: -len(suffix)]
            break
    if len(w) > 3 and w[-1] == w[-2] and w[-1] not in "aeiouls":
        w = w[:-1]  # submitt -> submit
    if len(w) > 3 and w.endswith("e"):
        w = w[:-1]  # approve -> approv, matching approv(es)
    return w


def content_stems(tokens: list[Token]) -> set[str]:
    return {stem(t.text) for t in tokens if t.text[0].isalnum() and t.lower not in STOPWORDS}


def phrase_after(text: str, start: int, end: int, max_tokens: int = 6) -> str:
    """The marker plus the words that follow it, up to clause punctuation."""
    tail = text[end:]
    stop = re.search(r"[,;:.!?](?:\s|$)", tail)
    if stop:
        tail = tail[: stop.start()]
    words = tail.split()[:max_tokens]
    return (text[start:end] + (" " + " ".join(words) if words else "")).strip()


def phrase_around(text: str, start: int, end: int, before: int = 3, after: int = 3) -> str:
    left = text[:start].split()[-before:]
    right = text[end:].split()[:after]
    phrase = " ".join(left + [text[start:end]] + right)
    return re.sub(r"\s+([,;:.!?])", r"\1", phrase).strip(" ,;:.!?")


def _overlaps(span: tuple[int, int], taken: list[tuple[int, int]]) -> bool:
    s, e = span
    return any(s < te and ts < e for ts, te in taken)


# --- marker extraction -----------------------------------------------------------

def _focus_after(tokens: list[Token], end: int) -> str | None:
    """First content word after a negation cue: 'not paid the fee' -> 'paid'."""
    for tok in tokens:
        if tok.start < end:
            continue
        if tok.text in ",;:.!?":
            return None  # the clause ended before a content word appeared
        if tok.text[0].isalpha() and tok.lower not in _FOCUS_SKIP:
            return tok.lower
    return None


def _match_patterns(text: str, patterns: list[lx.Pattern], category: str) -> list[Marker]:
    taken: list[tuple[int, int]] = []
    found: list[Marker] = []
    for pat in patterns:
        for m in pat.regex.finditer(text):
            span = m.span()
            if span[0] == span[1] or _overlaps(span, taken):
                continue
            taken.append(span)
            found.append(Marker(category, pat.label, m.group(0), span[0], span[1],
                                context=phrase_after(text, *span)))
    return found


def extract_negation(text: str, tokens: list[Token]) -> list[Marker]:
    markers = _match_patterns(text, lx.NEGATION_PATTERNS, "negation")
    # 'neither A nor B' is a single negation; drop the 'nor' that follows 'neither'.
    neither_at = [m.start for m in markers if m.label == "NEITHER"]
    markers = [m for m in markers
               if not (m.label == "NOR" and any(n < m.start for n in neither_at))]
    with_focus = [
        Marker(m.category, m.label, m.text, m.start, m.end, m.context,
               focus=_focus_after(tokens, m.end))
        for m in markers
    ]
    for m in lx.NON_PREFIX_PATTERN.finditer(text):
        word = m.group(0).lower()
        if word in lx.LEXICAL_NEGATIVES:
            continue  # listed explicitly below
        with_focus.append(Marker("negation", "LEXICAL_NEG", m.group(0), m.start(), m.end(),
                                 context=phrase_around(text, m.start(), m.end()),
                                 focus=word, detail=m.group(1).lower()))
    for m in lx.LEXICAL_NEGATIVE_PATTERN.finditer(text):
        word = re.sub(r"\s+", " ", m.group(0).lower())
        with_focus.append(Marker("negation", "LEXICAL_NEG", m.group(0), m.start(), m.end(),
                                 context=phrase_around(text, m.start(), m.end()),
                                 focus=word, detail=lx.LEXICAL_NEGATIVES[word]))
    return with_focus


def _clause_start(tokens: list[Token]) -> list[Token]:
    """Tokens where the main clause starts. Usually the sentence start, but after
    a fronted adverbial clause ('If you are absent, inform the HOD') it is the
    first word after the comma."""
    if tokens and tokens[0].lower in lx.FRONTED_CLAUSE_STARTERS:
        for i, tok in enumerate(tokens):
            if tok.text == ",":
                return tokens[i + 1:]
    return tokens


def _imperative_marker(text: str, tokens: list[Token]) -> Marker | None:
    words = [t for t in _clause_start(tokens) if t.text[0].isalpha()]
    if words and words[0].lower in {"please", "kindly"}:
        words = words[1:]
    if not words or words[0].lower not in lx.IMPERATIVE_VERBS:
        return None
    if any(w.lower in lx.NOT_IMPERATIVE_NEXT for w in words[1:3]):
        return None
    first = words[0]
    return Marker("modality", "OBLIGATION", first.text, first.start, first.end,
                  context=phrase_after(text, first.start, first.end),
                  detail="implicit imperative", implicit=True)


def extract_modality(text: str, tokens: list[Token]) -> list[Marker]:
    markers = _match_patterns(text, lx.MODALITY_PATTERNS, "modality")
    imperative = _imperative_marker(text, tokens)
    # An explicit modal later in the sentence ('Submit ... if you can') does not
    # cancel the imperative force of the main verb, so both are kept.
    if imperative and not _overlaps((imperative.start, imperative.end),
                                    [(m.start, m.end) for m in markers]):
        markers.append(imperative)
    return markers


def extract_markers(text: str, tokens: list[Token] | None = None) -> list[Marker]:
    tokens = tokens if tokens is not None else tokenize(text)
    markers = (
        extract_negation(text, tokens)
        + extract_modality(text, tokens)
        + _match_patterns(text, lx.CONDITION_PATTERNS, "condition")
        + _match_patterns(text, lx.QUANTIFIER_PATTERNS, "quantifier")
    )
    return sorted(markers, key=lambda m: (m.start, m.category))


# --- numeric extraction ----------------------------------------------------------

_NUM = r"\d+(?:,\d{2,3})*(?:\.\d+)?"
_MONTH_RE = "(?:" + "|".join(sorted(lx.MONTHS, key=len, reverse=True)) + r")(?![a-z])"
_WEEKDAY_RE = "(?:" + "|".join(lx.WEEKDAYS) + ")"

_RATE = r"(?:\s+" + lx.RATE_REGEX + r")?"
_CURRENCY_PREFIX = re.compile(
    r"(₹|\brs\.?|\binr\b|\$|\busd\b|€|\beur\b|£|\bgbp\b)\s?(" + _NUM + r")"
    r"(?:\s?(lakhs?|crores?|thousand|million|k)\b)?" + _RATE, re.IGNORECASE)
_CURRENCY_SUFFIX = re.compile(
    r"\b(" + _NUM + r")\s?(lakhs?|crores?|thousand|million)?\s?"
    r"(rupees?|inr|dollars?|usd|euros?|eur|pounds?|gbp)\b" + _RATE, re.IGNORECASE)
_PERCENT = re.compile(r"\b(" + _NUM + r")\s?(%|percent\b|per\s?cent\b|percentage\b)", re.IGNORECASE)
_DATE_ISO = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_DATE_NUMERIC = re.compile(r"\b(\d{1,2})([/.-])(\d{1,2})\2(\d{4}|\d{2})\b")
_DATE_DMY = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(" + _MONTH_RE + r")(?:,?\s+(\d{4})\b)?", re.IGNORECASE)
_DATE_MDY = re.compile(r"\b(" + _MONTH_RE + r")\.?\s+(\d{1,2})(?:st|nd|rd|th)?(?!\d)(?:,?\s+(\d{4})\b)?", re.IGNORECASE)
_DATE_MY = re.compile(r"\b(" + _MONTH_RE + r")\s+(\d{4})\b", re.IGNORECASE)
_WEEKDAY = re.compile(r"\b(" + _WEEKDAY_RE + r")s?\b", re.IGNORECASE)
_TIME_12 = re.compile(r"\b(\d{1,2})(?:[:.](\d{2}))?\s?([ap])\.?\s?m\b", re.IGNORECASE)
_TIME_24 = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b(?:\s?(?:hrs|hours)\b)?", re.IGNORECASE)
_TIME_MILITARY = re.compile(r"\b([01]\d|2[0-3])([0-5]\d)\s?(?:hrs|hours|h)\b", re.IGNORECASE)
_TIME_WORD = re.compile(r"\b(noon|midday|midnight)\b", re.IGNORECASE)
_PLAIN_NUMBER = re.compile(r"(?<![\w.])(" + _NUM + r")(?:st|nd|rd|th)?(?!\w)(?:\s+(" + lx.UNIT_REGEX + r"))?", re.IGNORECASE)
_TENS = ("twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
_UNITS = ("one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
# One capture group for the whole number word: 'seventy-five', 'seventy five' or 'twelve'.
_WORD_NUMBER_RE = (r"((?:" + "|".join(_TENS) + r")[-\s](?:" + "|".join(_UNITS) + r")"
                   r"|(?:" + "|".join(sorted(lx.NUMBER_WORDS, key=len, reverse=True)) + r"))\b")
_BOUNDED_WORD_NUMBER = re.compile(lx.BOUND_BEFORE_WORD_NUMBER + r"\b" + _WORD_NUMBER_RE, re.IGNORECASE)
_WORD_PERCENT = re.compile(r"\b" + _WORD_NUMBER_RE + r"\s?(?:percent|per\s?cent)\b", re.IGNORECASE)
_WORD_NUMBER = re.compile(r"\b" + _WORD_NUMBER_RE + r"\s+(" + lx.UNIT_REGEX + r")", re.IGNORECASE)
_WORD_PERCENT = re.compile(r"\b" + _WORD_NUMBER_RE + r"\s?(?:percent|per\s?cent)\b", re.IGNORECASE)
_WORD_NUMBER = re.compile(r"\b" + _WORD_NUMBER_RE + r"\s+(" + lx.UNIT_REGEX + r")", re.IGNORECASE)


class _Span:
    """Minimal stand-in for re.Match when only part of a match is the value."""

    def __init__(self, start: int, end: int):
        self._span = (start, end)

    def span(self, group: int = 0) -> tuple[int, int]:
        return self._span


def _word_value(raw: str) -> float:
    """'seventy-five' -> 75.0, 'twelve' -> 12.0"""
    return float(sum(lx.NUMBER_WORDS[w] for w in re.split(r"[-\s]+", raw.lower())))


def _to_float(raw: str) -> float:
    return float(raw.replace(",", ""))


def _fmt(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def _unit(raw: str | None) -> str | None:
    """Normalise a unit noun: 'Working Days' -> 'working day', 'hrs' -> 'hour'."""
    if not raw:
        return None
    words = raw.lower().split()
    head = words[-1]
    head = {"hrs": "hour", "hr": "hour", "mins": "minute", "min": "minute",
            "copies": "copy", "classes": "class"}.get(head, head)
    if head.endswith("s") and head not in {"class"}:
        head = head[:-1]
    return " ".join(words[:-1] + [head])


def _date_key(year: int | None, month: int | None, day: int | None) -> str:
    y = f"{year:04d}" if year else "????"
    m = f"{month:02d}" if month else "??"
    d = f"{day:02d}" if day else "??"
    return f"date:{y}-{m}-{d}"


def _year(raw: str | None) -> int | None:
    if not raw:
        return None
    y = int(raw)
    return y + 2000 if y < 100 else y


def extract_numbers(text: str) -> list[NumericValue]:
    taken: list[tuple[int, int]] = []
    found: list[NumericValue] = []

    def add(m: re.Match, kind: str, value: float | None, unit: str | None, key: str,
            group: int = 0) -> None:
        start, end = m.span(group)
        if _overlaps((start, end), taken):
            return
        taken.append((start, end))
        found.append(NumericValue(kind, text[start:end], start, end,
                                  value, unit, key, phrase_around(text, start, end)))

    def add_currency(m: re.Match, code: str, amount: str, multiplier: str | None, rate: str | None) -> None:
        value = _to_float(amount) * lx.MULTIPLIERS.get((multiplier or "").lower(), 1)
        unit = f"{code}/{rate.lower()}" if rate else code  # 'INR/day' for 'Rs. 10 per day'
        add(m, "currency", value, unit, f"currency:{unit}:{_fmt(value)}")

    for m in _CURRENCY_PREFIX.finditer(text):
        add_currency(m, lx.CURRENCY_CODES[m.group(1).lower()], m.group(2), m.group(3), m.group(4))
    for m in _CURRENCY_SUFFIX.finditer(text):
        add_currency(m, lx.CURRENCY_CODES[m.group(3).lower()], m.group(1), m.group(2), m.group(4))
    for m in _PERCENT.finditer(text):
        value = _to_float(m.group(1))
        add(m, "percent", value, "%", f"percent:{_fmt(value)}")

    for m in _DATE_ISO.finditer(text):
        add(m, "date", None, None, _date_key(int(m.group(1)), int(m.group(2)), int(m.group(3))))
    for m in _DATE_NUMERIC.finditer(text):
        a, b, year = int(m.group(1)), int(m.group(3)), _year(m.group(4))
        # Only normalise when the day/month order is unambiguous. '05/06/2025'
        # could be 5 June or May 6, so it is compared by its digits only.
        if a > 12 >= b:
            key = _date_key(year, b, a)
        elif b > 12 >= a:
            key = _date_key(year, a, b)
        else:
            key = f"date_raw:{a}/{b}/{year}"
        add(m, "date", None, None, key)
    for m in _DATE_DMY.finditer(text):
        month = lx.MONTHS[m.group(2).lower()]
        add(m, "date", None, None, _date_key(_year(m.group(3)), month, int(m.group(1))))
    for m in _DATE_MDY.finditer(text):
        # 'may 15' after a modal verb is ambiguous; require a capital M for May.
        if m.group(1).lower() == "may" and not m.group(1)[0].isupper():
            continue
        month = lx.MONTHS[m.group(1).lower()]
        add(m, "date", None, None, _date_key(_year(m.group(3)), month, int(m.group(2))))
    for m in _DATE_MY.finditer(text):
        add(m, "date", None, None, _date_key(int(m.group(2)), lx.MONTHS[m.group(1).lower()], None))
    for m in _WEEKDAY.finditer(text):
        add(m, "weekday", None, None, f"weekday:{m.group(1).lower()}")

    for m in _TIME_12.finditer(text):
        hour, minute = int(m.group(1)) % 12, int(m.group(2) or 0)
        if m.group(3).lower() == "p":
            hour += 12
        add(m, "time", hour * 60 + minute, None, f"time:{hour:02d}:{minute:02d}")
    for m in _TIME_24.finditer(text):
        hour, minute = int(m.group(1)), int(m.group(2))
        add(m, "time", hour * 60 + minute, None, f"time:{hour:02d}:{minute:02d}")
    for m in _TIME_MILITARY.finditer(text):  # '0900 hrs'
        hour, minute = int(m.group(1)), int(m.group(2))
        add(m, "time", hour * 60 + minute, None, f"time:{hour:02d}:{minute:02d}")
    for m in _TIME_WORD.finditer(text):
        hour = 0 if m.group(1).lower() == "midnight" else 12
        add(m, "time", hour * 60, None, f"time:{hour:02d}:00")

    for m in _PLAIN_NUMBER.finditer(text):
        value, unit = _to_float(m.group(1)), _unit(m.group(2))
        add(m, "number", value, unit, f"number:{_fmt(value)}" + (f" {unit}" if unit else ""))
    for m in _WORD_PERCENT.finditer(text):  # 'seventy-five percent'
        value = _word_value(m.group(1))
        add(m, "percent", value, "%", f"percent:{_fmt(value)}")
    for m in _WORD_NUMBER.finditer(text):
        value = _word_value(m.group(1))
        unit = _unit(m.group(2))
        add(m, "number", value, unit, f"number:{_fmt(value)} {unit}")
    for m in _BOUNDED_WORD_NUMBER.finditer(text):
        # Only the number word is the value; the bound itself is a quantifier marker.
        value = _word_value(m.group(1))
        add(_Span(m.start(1), m.end(1)), "number", value, None, f"number:{_fmt(value)}")

    return sorted(found, key=lambda n: n.start)


# --- one-stop sentence description ------------------------------------------------

def analyze_sentence(text: str) -> SentenceAnalysis:
    clean = normalize_text(text)
    tokens = tokenize(clean)
    return SentenceAnalysis(clean, tokens, extract_markers(clean, tokens), extract_numbers(clean))
