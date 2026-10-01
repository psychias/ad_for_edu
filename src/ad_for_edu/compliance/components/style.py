"""Style: present tense, active voice, third person, observable content, no meta-language.

The rule names five properties of the wording. Each is detected by a pattern
written from the rule text, and each violated property costs half of the score,
so two violated properties already score zero. Counting properties instead of
matches keeps a single wordy violation from deciding the score alone.

Quoted spans are removed before matching: a description that reads slide text
aloud cites the slide and does not itself break the rule.
"""

from __future__ import annotations

import re

from ...data.schema import MomentContext
from .base import COMPONENTS, Component

SECOND_PERSON = re.compile(r"\byours?\b|\byou\b|\byourself\b", re.I)

PAST_AUXILIARY = re.compile(r"\b(was|were|had|did)\b", re.I)
#: Narrative past forms. A form that doubles as a participle belongs to the voice
#: detector, so the look-behinds keep "is labeled" and the like out of this count.
NARRATIVE_PAST = re.compile(
    r"(?<!\bis )(?<!are )(?<!was )(?<!ere )(?<!een )(?<!ing )"
    r"\b(appeared|showed|became|went|came|drew|wrote|stood|grew|rose|fell|began|got|took|gave|"
    r"changed|moved|opened|closed|turned|vanished|disappeared|faded|zoomed|scrolled)\b",
    re.I,
)

#: Participles that describe something being done. "A circle is drawn" is the passive
#: the rule forbids.
DYNAMIC_PARTICIPLES = (
    r"drawn|given|presented|depicted|illustrated|"
    r"indicated|introduced|added|removed|revealed|erased|replaced|animated|"
    r"overlaid|superimposed|zoomed|scrolled|advanced"
)
#: Participles that describe a state on screen. "Is labeled 2024" is an adjective
#: phrase and is accepted with a present-tense verb.
STATIVE_PARTICIPLES = (
    r"labeled|labelled|titled|named|colored|coloured|numbered|dated|shaped|aligned|centered|"
    r"centred|marked|divided|connected|linked|arranged|placed|positioned|surrounded|enclosed|"
    r"listed|written|underlined|circled|highlighted|described|identified|defined|shown"
)
PASSIVE = re.compile(
    rf"\b(is|are)\s+(\w+ly\s+)?({DYNAMIC_PARTICIPLES})\b"
    rf"|\b(was|were|being)\s+(\w+ly\s+)?"
    rf"({DYNAMIC_PARTICIPLES}|{STATIVE_PARTICIPLES}|shown|displayed|\w{{3,}}ed)\b"
    rf"|\bbeen\s+(\w+ly\s+)?({DYNAMIC_PARTICIPLES})\b",
    re.I,
)

EVALUATIVE = re.compile(
    r"\b(beautiful(ly)?|elegant(ly)?|impressive(ly)?|interesting(ly)?|nice(ly)?|lovely|"
    r"remarkab(le|ly)|striking(ly)?|wonderful(ly)?|excellent(ly)?|great|amazing(ly)?|"
    r"clever(ly)?|neat(ly)?|helpful(ly)?|useful(ly)?|important(ly)?|clear(ly)?\s+(beautiful|"
    r"elegant|impressive)|well[- ](designed|organized|organised|structured|made)|"
    r"easy\s+to\s+(read|see|follow)|hard\s+to\s+(read|see|follow))\b",
    re.I,
)

#: Meta-reference is an open set, so it is matched as a grammar: a definite display
#: noun followed by a display verb. "The slide shows" frames the medium; "a diagram
#: illustrates" introduces an object and is accepted.
_DISPLAY_NOUN = (
    r"(slide|image|figure|screen|visual|diagram|picture|photo|graphic|animation|chart|video|frame)"
)
_DISPLAY_VERB = r"(shows?|showing|depicts?|depicting|displays?|displaying)"
META_GRAMMAR = re.compile(
    rf"\b(the|this|that)\s+{_DISPLAY_NOUN}\s+(now\s+|currently\s+)?{_DISPLAY_VERB}\b", re.I
)
META_FRAMES = re.compile(
    r"\bwhat\s+(appears|is\s+(shown|visible|displayed))\s+here\b|"
    r"\bas\s+(we|you)\s+can\s+see\b|\bhere\s+(we|you)\s+(can\s+)?see\b|"
    r"\b(we|one)\s+can\s+(see|observe|notice)\b|\bis\s+visible\s+on\s+the\s+(slide|screen)\b|"
    r"\ban?\s+(image|picture|visual)\s+of\b|\bthe\s+presenter\b|\bthis\s+slide\b",
    re.I,
)

QUOTED = re.compile(r'"[^"]{0,200}"')

PROPERTIES: tuple[str, ...] = (
    "second_person",
    "past_tense",
    "passive_voice",
    "evaluative",
    "meta_reference",
)


def style_violations(text: str) -> dict[str, int]:
    """Number of matches per property, outside quoted spans."""
    unquoted = QUOTED.sub(" ", text or "")
    return {
        "second_person": len(SECOND_PERSON.findall(unquoted)),
        "past_tense": len(PAST_AUXILIARY.findall(unquoted)) + len(NARRATIVE_PAST.findall(unquoted)),
        "passive_voice": len(PASSIVE.findall(unquoted)),
        "evaluative": len(EVALUATIVE.findall(unquoted)),
        "meta_reference": len(META_GRAMMAR.findall(unquoted)) + len(META_FRAMES.findall(unquoted)),
    }


@COMPONENTS.register("style_properties")
class StyleProperties(Component):
    """One minus half the number of violated style properties, floored at zero."""

    category = "style"

    def __init__(self, cost_per_property: float = 0.5) -> None:
        if not 0 < cost_per_property <= 1:
            raise ValueError(f"cost_per_property must be in (0, 1], got {cost_per_property}")
        self.cost_per_property = cost_per_property

    def score(self, text: str, moment: MomentContext) -> float:
        if not text or not text.strip():
            return 0.0
        violated = sum(1 for count in style_violations(text).values() if count > 0)
        return max(0.0, 1.0 - violated * self.cost_per_property)
