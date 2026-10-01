"""Which rules of the standard each component scores.

A compliance score on its own says a description is worse without saying what is
wrong with it. This module is what turns the mechanical mode into a diagnostic: it
names the rules a description broke.

The mapping is **derived, not transcribed.** The rule table already carries a
`scoring_checks` field naming the check that scores each rule, and the only thing
authored here is which of those check names each component implements. The rule ids
then come from the table, so a rule that moves category or route cannot leave a
stale id behind in this file.

Three facts fall out of that derivation and are worth stating, because each is a
place where the standard and the implementation do not line up:

*The mode and the route are different sets.* Fourteen rules are routed `mechanical`,
but seven of them are scored by the two components that need a learned model, so
they are reachable only in the local mode. The mechanical mode scores five.

*Two routed rules have no check at all.* `rung_reachable` and `transition_effect` are
named by the table and implemented by nothing. They are listed in
`WITHOUT_A_COMPONENT` rather than passed over, so the gap is a stated property.

*The novelty term scores a rule the table calls unscored.* The table names no check
for the rule against re-describing what an earlier description already covered,
which is exactly what the within-lecture novelty term measures. Those two ids are
therefore authored here, not derived, and `SEQUENCE_RULES` says so.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..core.errors import ContractError
from ..standard.applicability import ApplicabilityTable

#: The check names, in the rule table's own vocabulary, that each component implements.
#: This is the only authored mapping; every rule id below is read from the table.
CHECKS_BY_COMPONENT: Mapping[str, tuple[str, ...]] = {
    "style_properties": ("style_meta", "style_pronoun", "style_v2"),
    "terminology_grounding": ("term_grounding",),
    "length_budget_ratio": ("length_budget",),
    "deixis_resolution": ("deixis_referent", "deixis_overlay"),
    "faithfulness_nli": ("faithfulness_nli",),
    "non_redundancy_embedding": ("nonredundancy_now", "nonredundancy_ahead"),
}

#: Checks the table names and no component implements, with the rule each belongs to.
#: Listed so that the gap is a property of this module rather than an absence.
WITHOUT_A_COMPONENT: Mapping[str, str] = {
    "rung_reachable": "the delivery rung a pause can reach is not checked",
    "transition_effect": "a slide transition effect is not checked",
    "slide_copy_verbatim": "copying the slide verbatim is only partly caught, by the novelty term",
}

#: The rules the within-lecture novelty term scores, which no component scores.
#: Authored rather than derived: the table names no check for the first of them.
SEQUENCE_RULES: tuple[str, ...] = ("cross_cutting_006", "slide_text_003")

#: The name the sequence factor is registered under.
SEQUENCE_FACTOR_NAME = "novelty_knee"


@dataclass(frozen=True)
class RuleMap:
    """The rules each component scores, read from the rule table."""

    table: ApplicabilityTable
    by_component: Mapping[str, tuple[str, ...]]
    sequence: tuple[str, ...]

    @classmethod
    def load(cls, table: ApplicabilityTable | None = None) -> RuleMap:
        found = table or ApplicabilityTable.load()
        wanted_to_component = {}
        for component, checks in CHECKS_BY_COMPONENT.items():
            for check in checks:
                if check in wanted_to_component:
                    raise ContractError(
                        f"check {check!r} is claimed by both "
                        f"{wanted_to_component[check]!r} and {component!r}"
                    )
                wanted_to_component[check] = component

        by_component: dict[str, list[str]] = {name: [] for name in CHECKS_BY_COMPONENT}
        named_by_the_table: set[str] = set()
        for rule_id, entry in found.entries.items():
            for check in entry.checks:
                named_by_the_table.add(check)
                component = wanted_to_component.get(check)
                if component is not None and rule_id not in by_component[component]:
                    by_component[component].append(rule_id)

        unclaimed = sorted(named_by_the_table - set(wanted_to_component))
        unexplained = [check for check in unclaimed if check not in WITHOUT_A_COMPONENT]
        if unexplained:
            raise ContractError(
                f"the rule table names checks that no component implements and "
                f"WITHOUT_A_COMPONENT does not explain: {unexplained}"
            )
        promised = sorted(WITHOUT_A_COMPONENT)
        if unclaimed != promised:
            raise ContractError(
                f"WITHOUT_A_COMPONENT lists {promised} but the table leaves {unclaimed} "
                "unimplemented; the two must agree, or the gap is being guessed at"
            )
        for component, ids in by_component.items():
            if not ids:
                raise ContractError(
                    f"component {component!r} claims checks that no rule names: "
                    f"{list(CHECKS_BY_COMPONENT[component])}"
                )
        missing = [rule_id for rule_id in SEQUENCE_RULES if rule_id not in found.entries]
        if missing:
            raise ContractError(f"the novelty term names rules that do not exist: {missing}")
        ordered = {
            component: tuple(sorted(ids, key=lambda rule_id: found[rule_id].number))
            for component, ids in by_component.items()
        }
        return cls(found, ordered, SEQUENCE_RULES)

    # ------------------------------------------------------------------ lookups
    def for_component(self, component: str) -> tuple[str, ...]:
        try:
            return self.by_component[component]
        except KeyError:
            raise ContractError(
                f"no rules are mapped to component {component!r}; "
                f"known: {sorted(self.by_component)}"
            ) from None

    def broken_by(
        self, component: str, moment_type: str | None, *, described: bool = True
    ) -> tuple[str, ...]:
        """The rules a failing component breaks on a moment of this type.

        A component may stand for several rules, and which of them a moment is
        subject to depends on its type: the same faithfulness check answers for the
        figure rule on a figure and the table rule on a table. The type-independent
        rules of the component are always included, so a failure is never attributed
        to nothing.
        """
        applicable = []
        for rule_id in self.for_component(component):
            if self.table[rule_id].applies(moment_type, described):
                applicable.append(rule_id)
        if applicable:
            return tuple(applicable)
        # No rule of this component applies to the type, which means the component
        # should not have scored the moment. Saying so beats attributing to nothing.
        return ()

    def rules_of_mode(self, components: Sequence[str]) -> tuple[str, ...]:
        """Every rule the components of one mode can attribute a failure to."""
        found: list[str] = []
        for component in components:
            for rule_id in self.for_component(component):
                if rule_id not in found:
                    found.append(rule_id)
        return tuple(sorted(found, key=lambda rule_id: self.table[rule_id].number))

    def summary(self) -> dict[str, Any]:
        """What this map covers, for a report or a check."""
        routed_mechanical = sorted(
            rule_id
            for rule_id, entry in self.table.entries.items()
            if entry.route == "mechanical"
        )
        mapped = {rule_id for ids in self.by_component.values() for rule_id in ids}
        return {
            "per_component": {
                component: list(ids) for component, ids in sorted(self.by_component.items())
            },
            "novelty_term": list(self.sequence),
            "routed_mechanical": routed_mechanical,
            "routed_mechanical_without_a_component": [
                rule_id for rule_id in routed_mechanical if rule_id not in mapped
            ],
            "checks_no_component_implements": {
                check: why for check, why in sorted(WITHOUT_A_COMPONENT.items())
            },
        }
