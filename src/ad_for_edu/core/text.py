"""Text helpers shared by the generators, the filters and the metric.

Two kinds of string need care.

**Sentinels.** When a source has nothing to offer, the context carries a readable
marker such as `(no slide text detected)` instead of an empty string, so a reader
can tell "nothing was there" from "nothing was looked up". A marker is never slide
content: `strip_sentinels` removes it wherever it occurs, and every function that
grounds a description against the slide goes through `slide_premise`.

**Delivery prefixes.** A description may start with `[before]` or `[after]`, which
says when it is delivered, not what it says. `clean_description` removes the prefix
before any comparison or count.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

NO_SLIDE_TEXT = "(no slide text detected)"
NO_SPEECH = "(no speech in this window)"
NOTHING_DESCRIBED_YET = "(none yet)"

#: Every marker a context field may carry in place of content.
SENTINELS = frozenset({NO_SLIDE_TEXT, NO_SPEECH, NOTHING_DESCRIBED_YET, "(no speech)"})

_SENTINEL = re.compile(
    "|".join(re.escape(marker) for marker in sorted(SENTINELS, key=len, reverse=True)),
    re.IGNORECASE,
)
_DELIVERY_PREFIX = re.compile(r"^\s*\[(before|after)\]\s*", re.IGNORECASE)
_TERM = re.compile(r"[A-Za-z][A-Za-z0-9\-]{2,}")
_WORD = re.compile(r"[a-z0-9]+")

#: Function words that never count as a content term.
STOP_TERMS = frozenset(
    {
        "the", "and", "for", "with", "this", "that", "are", "was", "not", "you", "our",
        "from", "have", "all",
    }
)


def clean_description(text: str | None) -> str:
    """The description without its delivery prefix and surrounding space."""
    if not text:
        return ""
    return _DELIVERY_PREFIX.sub("", text).strip()


def word_count(text: str | None) -> int:
    return len(clean_description(text).split())


def is_sentinel(text: str | None) -> bool:
    """True when the whole string is a marker."""
    return str(text or "").strip().lower() in {marker.lower() for marker in SENTINELS}


def strip_sentinels(text: str | None) -> str:
    """Remove every marker, wherever it sits in the string."""
    return _SENTINEL.sub(" ", str(text or "")).strip()


def slide_premise(context: Mapping[str, Any] | None) -> str:
    """What is on the slide at a moment: slide text plus the on-screen summary, markers removed."""
    context = context or {}
    parts = (context.get("slide_ocr"), context.get("what_on_screen"))
    return strip_sentinels(" ".join(part for part in parts if part))


def content_terms(text: str | None, cap: int = 30) -> list[str]:
    """Distinct content terms in order of first appearance, at most `cap` of them.

    A term starts with a letter and has at least three characters. Comparison is
    case-insensitive; the first spelling met is the one returned.
    """
    cleaned = strip_sentinels(text)
    if not cleaned:
        return []
    terms: list[str] = []
    seen: set[str] = set()
    for token in _TERM.findall(cleaned):
        lowered = token.lower()
        if lowered in STOP_TERMS or lowered in seen:
            continue
        seen.add(lowered)
        terms.append(token)
        if len(terms) >= cap:
            break
    return terms


def following_terms(text: str, limit: int) -> str:
    """The first `limit` term-shaped tokens of `text`, joined by spaces."""
    return " ".join(_TERM.findall(text)[:limit])


def token_set(text: str | None) -> frozenset[str]:
    """Lower-cased alphanumeric runs, as a set. No stemming, no stop-word removal."""
    return frozenset(_WORD.findall((text or "").lower()))


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    """Overlap of two token sets; 0 when either is empty."""
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)
