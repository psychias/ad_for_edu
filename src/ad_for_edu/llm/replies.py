"""Reading a reply.

Two principles hold for every parser here.

**None on ambiguity, never a value.** A parser that falls back to a tie, to the
top of the scale or to zero turns a failure into a plausible answer, and nothing
downstream can tell. A caller given None has to decide; a caller given a value
does not know anything happened.

**The constrained form first, then give up.** The prompts ask for a bare token.
A reply in prose has already broken that contract, and guessing which number in
the prose was meant is how verdicts get inverted. The failure is counted instead,
and the raw reply is kept so that it can be looked at.
"""

from __future__ import annotations

import json
import re
from typing import Any

#: How much of a reply that failed to parse is kept.
RAW_CAP = 2000

_BACKSLASH = chr(92)
_BARE_CHOICE = re.compile(r"^[\s*_'\"`.]*(1|2|tie)[\s*_'\"`.!]*$", re.I)
_TIE = re.compile(r"\btie\b", re.I)
_OPTION = re.compile(r"\b([12])\b")
_FENCE_OPEN = re.compile(r"^```[a-zA-Z]*\s*")
_FENCE_CLOSE = re.compile(r"\s*```$")

#: A reply naming one option is trusted only when it is short. In a long reply a lone
#: digit is as likely to name the option rejected as the option chosen.
SHORT_REPLY_WORDS = 8

#: What a tie is recorded as, everywhere: in a judge's answer and in a rater's label.
#: One value, so that a side read from a reply and a side read from a label compare.
TIE = "tie"


def parse_choice(text: str | None) -> str | None:
    """The option a reply names: `"1"`, `"2"`, `TIE`, or None when it names none.

    The bare form is tried first. `tie` is matched as a word, so that "quantities"
    and "properties" do not vote. Prose that names both options is refused.
    """
    reply = (text or "").strip()
    if not reply:
        return None
    bare = _BARE_CHOICE.match(reply)
    if bare:
        value = bare.group(1).lower()
        return TIE if value == "tie" else value
    options = set(_OPTION.findall(reply))
    says_tie = bool(_TIE.search(reply))
    if says_tie and not options:
        return TIE
    if len(options) == 1 and not says_tie and len(reply.split()) <= SHORT_REPLY_WORDS:
        return options.pop()
    return None


def parse_scale(text: str | None, low: int, high: int) -> int | None:
    """An integer rating in `[low, high]`, or None.

    Mentions of the scale are removed before the search: "4 out of 5", "3/5" and
    "7 on the 0-10 scale" name the scale as well as the score, and the bound must
    not be read as the rating. When more than one distinct value in range remains,
    the reply is refused instead of one being picked by position.
    """
    reply = (text or "").strip()
    if not reply:
        return None
    reply = re.sub(rf"\b(\d+)\s*/\s*{high}\b", r"\1", reply)
    reply = re.sub(rf"\bout of\s+{high}\b", " ", reply, flags=re.I)
    reply = re.sub(
        rf"\b{low}\s*(?:-|to|–)\s*{high}\b(?:\s*(?:scale|range|point))?",
        " ",
        reply,
        flags=re.I,
    )
    reply = re.sub(r"\bscale\b", " ", reply, flags=re.I)
    values = {int(found) for found in re.findall(r"-?\d+", reply) if low <= int(found) <= high}
    if len(values) == 1:
        return values.pop()
    return None


def _balanced_end(text: str, start: int) -> int:
    """Index of the bracket that closes the structure opening at `start`, or -1.

    Brackets inside a string do not count: a description may contain braces.
    """
    closer = "]" if text[start] == "[" else "}"
    depth, in_string, escaped = 0, False, False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == _BACKSLASH:
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "[{":
            depth += 1
        elif char in "]}":
            depth -= 1
            if depth == 0:
                return index if char == closer else -1
    return -1


def extract_json(text: str | None) -> Any:
    """The largest complete JSON array or object in `text`, or None.

    Every opening bracket is tried, because a preamble may itself contain brackets,
    and the largest structure that parses is returned, because the payload is
    larger than any fragment quoted around it.
    """
    reply = text or ""
    found: list[tuple[int, Any]] = []
    for start, char in enumerate(reply):
        if char not in "[{":
            continue
        end = _balanced_end(reply, start)
        if end < 0:
            continue
        try:
            found.append((end - start, json.loads(reply[start : end + 1])))
        except json.JSONDecodeError:
            continue
    if not found:
        return None
    return max(found, key=lambda item: item[0])[1]


def strip_fences(text: str) -> str:
    """The reply without a Markdown code fence around it."""
    return _FENCE_CLOSE.sub("", _FENCE_OPEN.sub("", text.strip()))


def parse_verdict(text: str | None) -> tuple[str | None, str | None]:
    """`(verdict, reasoning)` of a reply `{"verdict": "PASS" | "FAIL", ...}`.

    `(None, None)` when the reply carries no such object.
    """
    if not text:
        return None, None
    body = strip_fences(text)
    match = re.search(r"\{.*\}", body, re.S)
    if match:
        body = match.group(0)
    try:
        parsed = json.loads(body)
    except (json.JSONDecodeError, TypeError):
        return None, None
    if not isinstance(parsed, dict):
        return None, None
    verdict = str(parsed.get("verdict", "")).upper().strip()
    if verdict in ("PASS", "FAIL"):
        return verdict, parsed.get("reasoning", "")
    return None, None


def keep_raw(record: dict[str, Any], text: str | None) -> dict[str, Any]:
    """Attach the raw reply to `record`. Call this whenever a reply failed to parse."""
    reply = text or ""
    record["raw"] = reply[:RAW_CAP]
    if len(reply) > RAW_CAP:
        record["raw_truncated"] = True
    return record
