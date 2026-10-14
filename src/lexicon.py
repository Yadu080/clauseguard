"""Hand-written lexicons and regular-expression patterns.

Every pattern maps a surface expression to a *normalised label*. Surface words
that play the same role (e.g. 'unless' and 'except when') share a label, so a
rewrite that swaps one for the other is not reported as a change.

Patterns inside one category are tried in the order listed. Once a span of
text is claimed by a pattern, later patterns of the *same* category cannot
match inside it. That is why longer or negated forms ('must not', 'only if')
are listed before the shorter words they contain ('must', 'if').

All patterns are compiled with re.IGNORECASE.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Pattern:
    label: str
    regex: re.Pattern


def _p(label: str, regex: str) -> Pattern:
    return Pattern(label, re.compile(regex, re.IGNORECASE))


# --- shared fragments -------------------------------------------------------

WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sept": 9, "sep": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}
_FULL_MONTHS = [m for m in MONTHS if len(m) > 4 or m in ("may", "june", "july")]

# Something that looks like the start of a date/time expression. Used so that
# ambiguous words such as 'by' are only treated as deadlines before a date.
DATEISH = (
    r"(?:the\s+)?(?:end\s+of\s+|close\s+of\s+)?(?:\d|"
    + "|".join(WEEKDAYS + tuple(_FULL_MONTHS))
    + r"|noon|midnight|today|tomorrow|tonight)"
)
# A following number, optionally preceded by a currency sign.
NUMBER_AHEAD = (r"(?=\s+(?:rs\.?\s?|₹\s?|\$\s?|inr\s)?(?:\d|(?:one|two|three|four|five|six|seven|eight|"
                r"nine|ten|eleven|twelve|fifteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)\b))")

# Things that look like negation but express a bound or an addition instead:
# 'no more than', 'not later than', 'not only', 'must not exceed'.
_NOT_A_NEGATION = (
    r"(?!\s+(?:only\b|more\s+than|less\s+than|fewer\s+than|later\s+than|"
    r"earlier\s+than|exceed))"
)

# --- negation ---------------------------------------------------------------

NEGATION_PATTERNS = [
    _p("NEG_PRONOUN", r"\b(?:no\s+one|no-one|nobody|nothing|none|nowhere)\b"),
    _p("CANNOT", r"\bcan\s*not\b" + _NOT_A_NEGATION),
    _p("NOT", r"\b[a-z]+n't\b" + _NOT_A_NEGATION),
    # 'whether or not' is concessive, not a negation of the clause.
    _p("NOT", r"(?<!whether or )\bnot\b" + _NOT_A_NEGATION),
    # 'No. 5' is an abbreviation for 'number'.
    _p("NO", r"\bno\b(?!\s*\.\s*\d)" + _NOT_A_NEGATION),
    _p("NEVER", r"\bnever\b"),
    # 'students who fail to register' negates 'register' like 'do not register'.
    _p("FAIL_TO", r"\bfail(?:s|ed|ure)?\s+to\b"),
    _p("WITHOUT", r"\bwithout\b"),
    _p("NEITHER", r"\bneither\b"),
    _p("NOR", r"\bnor\b"),
]

# Words whose meaning is negative through a prefix or lexical choice. The
# value is the positive counterpart (alternatives separated by '|'), used to
# spot direct antonym swaps such as 'eligible' -> 'ineligible'.
LEXICAL_NEGATIVES = {
    "ineligible": "eligible",
    "invalid": "valid",
    "unpaid": "paid",
    "unauthorized": "authorized",
    "unauthorised": "authorised",
    "unable": "able",
    "incomplete": "complete",
    "insufficient": "sufficient",
    "unacceptable": "acceptable",
    "inadmissible": "admissible",
    "disqualified": "qualified",
    "impermissible": "permissible",
    "unavailable": "available",
    "non-refundable": "refundable",
    "nonrefundable": "refundable",
    "unapproved": "approved",
    "unregistered": "registered",
    "unsigned": "signed",
    "absent": "present",
    "prohibited": "permitted|allowed",
    "forbidden": "permitted|allowed",
    "barred": "allowed|permitted",
    "disallowed": "allowed",
    "banned": "allowed|permitted",
    "rejected": "accepted",
    "denied": "granted",
    "closed": "open",
}
LEXICAL_NEGATIVE_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(w).replace(r"\ ", r"\s+")
                        for w in sorted(LEXICAL_NEGATIVES, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

# Negation cues that negate a verb phrase; used for the 'negation moved' check.
VERBAL_NEGATION_LABELS = {"NOT", "CANNOT", "NEVER", "FAIL_TO"}

# Morphological negation with 'non-': 'non-teaching' is the negative of 'teaching'.
NON_PREFIX_PATTERN = re.compile(r"\bnon-?([a-z]{3,})\b", re.IGNORECASE)

# --- modality (obligation / permission / recommendation) ---------------------

MODALITY_PATTERNS = [
    # 'must not exceed 2 pages' is an upper bound, i.e. an obligation to stay
    # within a limit, not a prohibition of the action itself.
    _p("OBLIGATION", r"\b(?:must|shall|may)\s+not\s+exceed\b|\b(?:cannot|can't)\s+exceed\b"),
    _p("RECOMMENDATION", r"\bshould\s+not\s+exceed\b"),
    # prohibition (negated obligation / negated permission)
    _p("PROHIBITION", r"\b(?:must|shall|may|should)\s+not\b|\b(?:mustn't|shan't|shouldn't)\b"),
    _p("PROHIBITION", r"\bcan\s*not\b|\bcan't\b"),
    _p("PROHIBITION", r"\b(?:is|are|was|were|be|been)\s+not\s+(?:permitted|allowed|authori[sz]ed)\b"),
    _p("PROHIBITION", r"\bnot\s+(?:be\s+)?(?:permitted|allowed|authori[sz]ed)\b"),
    _p("PROHIBITION", r"\b(?:is|are)\s+not\s+to\b"),
    _p("PROHIBITION", r"\b(?:prohibited|forbidden|barred|disallowed|banned)\b"),
    _p("PROHIBITION", r"^(?:please\s+|kindly\s+)?(?:do\s+not|don't|never)\b"),
    # 'No student may leave' / 'Nobody can enter' = prohibition, not permission.
    _p("PROHIBITION", r"\b(?:no\s+(?:[a-z]+\s+){1,3}?|nobody\s+|no\s+one\s+|none\s+(?:of\s+(?:the\s+)?[a-z]+\s+)?)(?:may|can|shall)\b"),
    # absence of obligation
    _p("NON_OBLIGATION", r"\bneed\s+not\b|\bneedn't\b"),
    _p("NON_OBLIGATION", r"\b(?:do|does|did)\s+not\s+(?:need|have)\s+to\b|\b(?:don't|doesn't|didn't)\s+(?:need|have)\s+to\b"),
    _p("NON_OBLIGATION", r"\bnot\s+(?:required|mandatory|compulsory|obligatory|necessary)\b"),
    _p("NON_OBLIGATION", r"\boptional\b"),
    # obligation
    _p("OBLIGATION", r"\b(?:is|are|be|being|was|were)\s+required\s+to\b|\brequired\s+to\b"),
    _p("OBLIGATION", r"\b(?:needs?|have|has|had)\s+to\b"),
    _p("OBLIGATION", r"\bobli(?:ged|gated)\s+to\b"),
    _p("OBLIGATION", r"\b(?:is|are)\s+to\b"),
    _p("OBLIGATION", r"\b(?:must|shall)\b"),
    _p("OBLIGATION", r"\b(?:required|requires?|mandatory|compulsory|obligatory|necessary)\b"),
    # recommendation
    _p("RECOMMENDATION", r"\bought\s+to\b|\bshould\b"),
    _p("RECOMMENDATION", r"\b(?:recommended|advised|encouraged)\b"),
    # permission ('may' is skipped when it is the month in '15 May')
    _p("PERMISSION", r"\b(?:permitted|allowed|authori[sz]ed|entitled|free|eligible)\s+to\b"),
    _p("PERMISSION", r"\b(?:permitted|allowed)\b"),
    _p("PERMISSION", r"(?<!\d )(?<!\dst )(?<!\dnd )(?<!\drd )(?<!\dth )(?<!of )\bmay\b(?!\s+\d)"),
    _p("PERMISSION", r"\bcan\b"),
]

# Verbs that, when they start a sentence, mark an imperative ("Submit the form").
# An imperative is treated as an implicit OBLIGATION; 'Do not ...' as PROHIBITION.
IMPERATIVE_VERBS = {
    "submit", "pay", "ensure", "attach", "complete", "register", "bring", "report",
    "return", "carry", "follow", "keep", "attend", "apply", "upload", "fill", "sign",
    "obtain", "contact", "read", "wear", "switch", "maintain", "inform", "collect",
    "deposit", "send", "provide", "include", "mention", "avoid", "make", "take",
    "verify", "enroll", "enrol", "renew", "clear", "produce", "present", "seek",
    "get", "check", "notify", "remember", "use", "display", "show", "leave",
}
# A sentence starting with one of these may open with an adverbial clause, so the
# imperative verb can follow the first comma ("If you are absent, inform the HOD").
FRONTED_CLAUSE_STARTERS = {
    "if", "when", "whenever", "unless", "before", "after", "once", "upon", "while",
    "during", "for", "in", "on", "at", "in case", "where", "wherever",
}

# Exact conversions between duration units, in minutes. Months and years are left
# out on purpose: '1 month' is not a fixed number of days.
DURATION_MINUTES = {"minute": 1, "hour": 60, "day": 1440, "week": 10080}

# If one of the next two words is one of these, the first word is a noun, not an
# imperative ("Use of phones is prohibited", "Leave applications must be ...").
NOT_IMPERATIVE_NEXT = {
    "of", "is", "are", "was", "were", "will", "would", "must", "should", "can",
    "may", "has", "have", "shall", "need", "needs",
}

# --- conditions and exceptions -----------------------------------------------

CONDITION_PATTERNS = [
    _p("CONDITION_ONLY", r"\bif\s+and\s+only\s+if\b|\bonly\s+(?:if|when|after|upon)\b"),
    _p("CONCESSION", r"\beven\s+(?:if|though|when)\b"),
    _p("EXCEPTION", r"\bexcept\s+(?:when|if|for|where|that|in)\b|\bexcept\b"),
    _p("EXCEPTION", r"\bunless\b|\bwith\s+the\s+exception\s+of\b|\bother\s+than\b|\bexcluding\b|\bsave\s+for\b|\bbarring\b"),
    _p("CONDITION", r"\bprovid(?:ed|ing)\s+that\b|(?<=, )provided\b|(?<=,)provided\b"),
    _p("CONDITION", r"\b(?:as|so)\s+long\s+as\b|\bin\s+the\s+event\s+(?:that|of)\b|\bin\s+case(?:\s+of)?\b"),
    _p("CONDITION", r"\bon\s+(?:the\s+)?condition\s+that\b|\bsubject\s+to\b|\bassuming\s+that\b"),
    _p("CONDITION", r"\bcontingent\s+(?:up)?on\b|\bdepending\s+on\b"),
    _p("CONDITION", r"\bif\b"),
    # 'once approved' is temporal; 'only once' / 'once a week' is a frequency.
    _p("TEMPORAL", r"\bwhen(?:ever)?\b|(?<!only )\bonce\b(?!\s*(?:[.,;]|$|a\b|an\b|per\b|every\b|more\b|again\b))|\bupon\b"),
]

# --- quantifiers, bounds and scope -------------------------------------------

QUANTIFIER_PATTERNS = [
    # upper bounds
    _p("UPPER_INCL", r"\b(?:not|cannot|can't|mustn't|shouldn't|won't|doesn't|don't)\s+exceed(?:ing)?\b"),
    _p("UPPER_INCL", r"\bat\s+most\b|\b(?:no|not)\s+more\s+than\b|\bup\s+to\b|\bmaximum(?:\s+of)?\b|\bor\s+(?:less|fewer|below|lower)\b|\bless\s+than\s+or\s+equal\s+to\b"),
    # lower bounds
    _p("LOWER_INCL", r"\bat\s+least\b|\b(?:no|not)\s+(?:less|fewer)\s+than\b|\bminimum(?:\s+of)?\b|\bor\s+(?:more|above|higher|greater)\b|\band\s+above\b|\bgreater\s+than\s+or\s+equal\s+to\b"),
    # deadlines (inclusive 'by Friday' vs strict 'before Friday')
    _p("BEFORE_INCL", r"\b(?:no|not)\s+later\s+than\b|\bon\s+or\s+before\b|\buntil\b|\btill\b"),
    _p("BEFORE_INCL", r"\bby\b(?=\s+" + DATEISH + r")"),
    _p("AFTER", r"\b(?:no|not)\s+earlier\s+than\b|\bon\s+or\s+after\b|(?<!only )\bafter\b"),
    _p("BEFORE_STRICT", r"\bbefore\b|\bprior\s+to\b"),
    _p("WITHIN", r"\bwithin\b"),
    # strict numeric bounds
    _p("LOWER_STRICT", r"\b(?:more|greater|higher)\s+than\b|\b(?:over|above|exceeding|exceeds?)\b" + NUMBER_AHEAD),
    _p("UPPER_STRICT", r"\b(?:less|fewer|lower)\s+than\b|\b(?:below|under)\b" + NUMBER_AHEAD),
    # scope
    _p("ONLY", r"(?<!not )\b(?:only|solely|exclusively)\b(?!\s+(?:if|when|after|upon)\b)"),
    _p("ONLY", r"\b(?:restricted|limited)\s+to\b|\breserved\s+for\b"),
    _p("UNIVERSAL", r"\b(?:all|every|each|any|everyone|everybody|anyone|anybody)\b"),
    _p("PARTIAL", r"\b(?:some|several|most|many|a\s+few|few|majority\s+of|minority\s+of)\b"),
]

# --- numbers ------------------------------------------------------------------

CURRENCY_CODES = {
    "₹": "INR", "rs": "INR", "rs.": "INR", "inr": "INR", "rupee": "INR", "rupees": "INR",
    "$": "USD", "usd": "USD", "dollar": "USD", "dollars": "USD",
    "€": "EUR", "eur": "EUR", "euro": "EUR", "euros": "EUR",
    "£": "GBP", "gbp": "GBP", "pound": "GBP", "pounds": "GBP",
}
MULTIPLIERS = {
    "k": 1e3, "thousand": 1e3, "lakh": 1e5, "lakhs": 1e5,
    "crore": 1e7, "crores": 1e7, "million": 1e6,
}
NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
}
# Unit nouns that may follow a number. A number word ('seven') is only
# extracted when followed by one of these, to avoid 'no one' or 'one of'.
# 'Rs. 10 per day': the period a rate refers to.
RATE_REGEX = r"(?:per|a|an|every|each)\s+(day|week|month|year|hour|semester|term|course|credit|subject|student|page|book)\b"
# 'at least two', 'more than three': a number word after a bound is numeric.
BOUND_BEFORE_WORD_NUMBER = (
    r"\b(?:at\s+least|at\s+most|(?:no\s+|not\s+)?(?:more|less|fewer)\s+than|up\s+to|"
    r"minimum\s+of|maximum\s+of|exactly)\s+"
)
UNIT_REGEX = (
    r"(?:(?:working|business|calendar|academic|consecutive|clear)\s+)?"
    r"(?:days?|weeks?|months?|years?|hours?|hrs?|minutes?|mins?|seconds?|credits?|"
    r"marks?|semesters?|attempts?|students?|members?|copies|copy|pages?|times?|"
    r"subjects?|courses?|backlogs?|books?|lectures?|classes|class|sessions?|points?|"
    r"words?|items?|cgpa|sgpa)\b"
)
