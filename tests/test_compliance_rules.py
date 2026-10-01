"""Which rules each component scores, and what the novelty term answers for.

The sets below are asserted by their members, not by their size. A count would pass
on the wrong rule ids, which is the failure this file exists to prevent: the map is
what turns a score into a statement about the standard, and a wrong id makes that
statement wrong while every number stays plausible.
"""

from __future__ import annotations

import pytest

from ad_for_edu.compliance import (
    CHECKS_BY_COMPONENT,
    SEQUENCE_RULES,
    WITHOUT_A_COMPONENT,
    RuleMap,
    build_scorer,
    build_sequence_factor,
    render_rule_diagnostic,
    rule_diagnostic,
    score_predictions,
)
from ad_for_edu.compliance.scorer import registered_name
from ad_for_edu.core.errors import ContractError
from ad_for_edu.data.schema import MomentContext
from ad_for_edu.standard import RuleBook

from .fixtures.factory import FakeModels

#: The rules the mechanical mode can attribute a failure to.
MECHANICAL_RULES = (
    "cross_cutting_002",
    "cross_cutting_004",
    "length_004",
    "deixis_005",
    "deixis_006",
)
#: The rules the local mode adds, all of them behind a learned model.
LOCAL_ONLY_RULES = (
    "cross_cutting_001",
    "cross_cutting_005",
    "cross_cutting_007",
    "figure_001",
    "table_001",
    "chart_001",
    "math_typeset_001",
)
#: Rules the standard routes as mechanically checkable that nothing here implements.
UNIMPLEMENTED = ("length_003", "slide_transition_001")


@pytest.fixture
def rules() -> RuleMap:
    return RuleMap.load()


# --------------------------------------------------------------- the map itself
def test_every_mapped_rule_exists_in_the_rule_book(rules):
    known = set(RuleBook.load().ids)
    for component, ids in rules.by_component.items():
        unknown = [rule_id for rule_id in ids if rule_id not in known]
        assert not unknown, f"{component} maps rules that are not in the book: {unknown}"
    assert not [rule_id for rule_id in rules.sequence if rule_id not in known]


def test_the_mechanical_mode_scores_exactly_these_rules(rules):
    mechanical = rules.rules_of_mode(
        [
            "style_properties",
            "terminology_grounding",
            "length_budget_ratio",
            "deixis_resolution",
        ]
    )
    assert set(mechanical) == set(MECHANICAL_RULES)


def test_the_local_mode_adds_exactly_these_rules(rules):
    added = rules.rules_of_mode(["faithfulness_nli", "non_redundancy_embedding"])
    assert set(added) == set(LOCAL_ONLY_RULES)


def test_the_two_sets_do_not_overlap(rules):
    assert not set(MECHANICAL_RULES) & set(LOCAL_ONLY_RULES)


def test_seven_of_the_fourteen_routed_rules_need_a_learned_model(rules):
    """The standard's route and the mode are different sets, and this says how."""
    routed = {
        rule_id
        for rule_id, entry in rules.table.entries.items()
        if entry.route == "mechanical"
    }
    assert len(routed) == 14
    assert set(MECHANICAL_RULES) < routed
    assert set(LOCAL_ONLY_RULES) < routed
    assert routed - set(MECHANICAL_RULES) - set(LOCAL_ONLY_RULES) == set(UNIMPLEMENTED)


def test_the_unimplemented_rules_are_named_rather_than_passed_over(rules):
    found = rules.summary()["routed_mechanical_without_a_component"]
    assert set(found) == set(UNIMPLEMENTED)
    assert set(WITHOUT_A_COMPONENT) == {
        "rung_reachable",
        "transition_effect",
        "slide_copy_verbatim",
    }
    for why in WITHOUT_A_COMPONENT.values():
        assert why and why[0].islower()


def test_the_novelty_term_covers_the_two_rules_no_component_does(rules):
    assert rules.sequence == SEQUENCE_RULES == ("cross_cutting_006", "slide_text_003")
    mapped = {rule_id for ids in rules.by_component.values() for rule_id in ids}
    assert not set(rules.sequence) & mapped


def test_the_first_of_those_is_routed_unscored_although_the_term_scores_it(rules):
    """A mismatch between the book and the implementation, asserted so it stays visible."""
    assert rules.table["cross_cutting_006"].route == "unscored"
    assert rules.table["cross_cutting_006"].checks == ()
    assert "re-describe" in rules.table["cross_cutting_006"].summary


def test_every_component_of_the_registry_is_in_the_map(rules):
    from ad_for_edu.compliance import COMPONENTS

    assert set(CHECKS_BY_COMPONENT) == set(COMPONENTS.names())
    assert set(rules.by_component) == set(COMPONENTS.names())


# ------------------------------------------------------------ type-sensitive attribution
@pytest.mark.parametrize(
    ("moment_type", "expected"),
    [
        ("figure", ("cross_cutting_005", "figure_001")),
        ("table", ("cross_cutting_005", "table_001")),
        ("chart", ("cross_cutting_005", "chart_001")),
        ("math", ("cross_cutting_005", "math_typeset_001")),
        ("slide", ("cross_cutting_005",)),
    ],
)
def test_faithfulness_names_the_rule_of_the_moment_it_failed_on(
    rules, moment_type, expected
):
    assert rules.broken_by("faithfulness_nli", moment_type) == expected


def test_deixis_applies_only_where_there_is_something_to_point_at(rules):
    assert rules.broken_by("deixis_resolution", "pointing") == ("deixis_005", "deixis_006")
    assert rules.broken_by("deixis_resolution", "ink") == ("deixis_006",)
    assert rules.broken_by("deixis_resolution", "slide") == ()


def test_an_unmapped_component_is_refused(rules):
    with pytest.raises(ContractError, match="no rules are mapped"):
        rules.for_component("vibes")


# ---------------------------------------------------------- what the scorer records
def context(moment_id: str, kind: str, **fields) -> MomentContext:
    values = {
        "moment_id": moment_id,
        "type": kind,
        "pause_after": 2.0,
        "slide_ocr": "Mitochondrion cristae matrix",
        "what_on_screen": "A labelled organelle",
        "transcript_window": "the energy of the cell is made here",
    }
    values.update(fields)
    return MomentContext(**values)


def test_a_failing_component_names_the_rule_it_broke():
    scorer = build_scorer("mechanical")
    assert scorer.names_rules
    row = scorer.score(
        "You could see that the slide showed a nice picture.",
        context("lec-a#0001", "figure"),
    )
    assert row.overall < 1.0
    assert "cross_cutting_002" in row.broken


def test_a_compliant_description_names_no_rule():
    scorer = build_scorer("mechanical")
    row = scorer.score(
        "Mitochondrion cristae matrix.", context("lec-a#0001", "slide")
    )
    assert row.overall == pytest.approx(1.0)
    assert row.broken == ()


def test_a_scorer_built_without_the_map_names_nothing_and_says_so():
    scorer = build_scorer("mechanical", name_rules=False)
    assert not scorer.names_rules
    row = scorer.score("You could see the slide showed it.", context("lec-a#0001", "figure"))
    assert row.overall < 1.0
    assert row.broken == ()


def test_the_registered_name_is_read_off_the_class():
    scorer = build_scorer("mechanical")
    names = [registered_name(component) for component in scorer.components]
    assert set(names) == {
        "style_properties",
        "terminology_grounding",
        "length_budget_ratio",
        "deixis_resolution",
    }


# ------------------------------------------------------------------- the diagnostic
def scored_rows():
    """Four descriptions of one lecture, the last repeating the one before it."""
    written = [
        ("lec-a#0001", "figure", "You could see that the slide showed a nice diagram."),
        ("lec-a#0002", "pointing", "He points here."),
        ("lec-a#0003", "slide", "Mitochondrion cristae matrix."),
        ("lec-a#0004", "slide", "Mitochondrion cristae matrix."),
    ]
    contexts = {
        moment: context(moment, kind, slide_ocr="Mitochondrion cristae matrix")
        for moment, kind, _text in written
    }
    predictions = [
        {"output_id": f"{moment}::demo", "moment_id": moment, "ad_text": text}
        for moment, _kind, text in written
    ]
    times = {moment: 10.0 * index for index, (moment, *_r) in enumerate(written, start=1)}
    scorer = build_scorer("mechanical")
    rows = score_predictions(
        predictions,
        contexts,
        scorer,
        sequence_factor=build_sequence_factor(),
        times=times,
        system="demo",
    )
    return scorer, rows


def test_the_diagnostic_reports_a_share_per_rule_over_what_it_applies_to(rules):
    scorer, rows = scored_rows()
    found = rule_diagnostic(rows, scorer, rules)
    deixis = found["per_rule"]["deixis_005"]
    # One pointing moment, and it broke the rule.
    assert deixis["applies_to"] == 1 and deixis["broken_by"] == 1
    terminology = found["per_rule"]["cross_cutting_004"]
    assert terminology["applies_to"] == 4


def test_a_rule_this_mode_never_tested_is_absent_rather_than_unbroken(rules):
    scorer, rows = scored_rows()
    found = rule_diagnostic(rows, scorer, rules)
    for rule_id in LOCAL_ONLY_RULES:
        assert rule_id not in found["per_rule"], rule_id
    assert set(found["rules_this_mode_cannot_reach"]) == set(LOCAL_ONLY_RULES)
    rendered = render_rule_diagnostic(found, title="t")
    assert "Not tested here" in rendered


def test_a_repeated_description_names_the_rule_against_repeating(rules):
    scorer, rows = scored_rows()
    repeated = next(row for row in rows if row["moment_id"] == "lec-a#0004")
    assert repeated["novelty_factor"] == 0.0
    assert set(repeated["broken"]) == set(SEQUENCE_RULES)
    found = rule_diagnostic(rows, scorer, rules)
    assert found["per_rule"]["cross_cutting_006"]["broken_by"] == 1


def test_rows_scored_without_the_sequence_factor_cannot_be_diagnosed(rules):
    """The two rules it answers for would otherwise read as unbroken."""
    scorer = build_scorer("mechanical")
    rows = score_predictions(
        [{"output_id": "lec-a#0001::d", "moment_id": "lec-a#0001", "ad_text": "A diagram."}],
        {"lec-a#0001": context("lec-a#0001", "figure")},
        scorer,
    )
    with pytest.raises(ContractError, match="not scored with the sequence factor"):
        rule_diagnostic(rows, scorer, rules)


def test_the_sequence_rules_are_always_among_those_reported(rules):
    scorer, rows = scored_rows()
    found = rule_diagnostic(rows, scorer, rules)
    for rule_id in SEQUENCE_RULES:
        assert rule_id in found["per_rule"], rule_id


def test_a_diagnostic_on_a_scorer_that_names_nothing_is_refused(rules):
    scorer = build_scorer("mechanical", name_rules=False)
    rows = score_predictions(
        [{"output_id": "lec-a#0001::d", "moment_id": "lec-a#0001", "ad_text": "A diagram."}],
        {"lec-a#0001": context("lec-a#0001", "figure")},
        scorer,
    )
    with pytest.raises(ContractError, match="without a rule map"):
        rule_diagnostic(rows, scorer, rules)


def test_a_diagnostic_with_nothing_scored_is_refused(rules):
    scorer = build_scorer("mechanical")
    with pytest.raises(ContractError, match="nothing to diagnose"):
        rule_diagnostic([{"overall": None}], scorer, rules)


def test_a_row_naming_a_rule_the_mode_cannot_reach_is_refused(rules):
    scorer, rows = scored_rows()
    rows[0] = {**rows[0], "broken": ["figure_001"]}
    with pytest.raises(ContractError, match="cannot reach"):
        rule_diagnostic(rows, scorer, rules)


# ------------------------------------------ what the novelty term answers for
def readout_and_description():
    """A slide-OCR readout against a real description, over four moments of one lecture.

    The readout copies the slide, so it repeats itself whenever two moments share a
    deck. That is the shortcut the novelty term exists to close.
    """
    slide = "Mitochondrion cristae matrix"
    moments = [f"lec-a#{index:04d}" for index in range(1, 5)]
    contexts = {
        moment: context(moment, "slide", slide_ocr=slide, transcript_window="and so we go on")
        for moment in moments
    }
    times = {moment: 10.0 * index for index, moment in enumerate(moments, start=1)}
    readout = [
        {"output_id": f"{m}::readout", "moment_id": m, "ad_text": slide + "."} for m in moments
    ]
    real = [
        {
            "output_id": f"{m}::real",
            "moment_id": m,
            "ad_text": f"Mitochondrion cristae matrix, part {index} of the organelle.",
        }
        for index, m in enumerate(moments, start=1)
    ]
    return contexts, times, readout, real


def mean_of(rows):
    values = [float(row["overall"]) for row in rows if row.get("overall") is not None]
    return sum(values) / len(values)


def test_a_system_that_repeats_itself_scores_below_one_that_does_not():
    """The property the metric is built to have, on two systems over one lecture."""
    contexts, times, readout, real = readout_and_description()
    scorer = build_scorer("mechanical")
    repeats = mean_of(
        score_predictions(
            readout, contexts, scorer, sequence_factor=build_sequence_factor(),
            times=times, system="readout",
        )
    )
    varies = mean_of(
        score_predictions(
            real, contexts, scorer, sequence_factor=build_sequence_factor(),
            times=times, system="real",
        )
    )
    assert repeats < varies


def test_the_components_alone_cannot_tell_the_two_apart():
    """Why the sequence factor is in the definition.

    Read one description at a time, the system that copies the slide is as compliant as
    the one that describes it, so the components alone do not separate them. The test
    asserts that about the components, which is a fact about what they measure.
    """
    contexts, _times, readout, real = readout_and_description()
    scorer = build_scorer("mechanical")
    per_description = [
        scorer.score(prediction["ad_text"], contexts[prediction["moment_id"]]).overall
        for prediction in readout
    ]
    assert min(per_description) >= mean_of(
        score_predictions(
            real, contexts, scorer, sequence_factor=build_sequence_factor(),
            times=_times, system="real",
        )
    )


def test_the_factor_reads_the_moment_id_because_that_carries_the_lecture():
    """Why the repository reads the moment id and never a per-row identifier.

    The factor groups by lecture, which it takes from the moment id. A pre-regeneration
    `segment_id` carries no lecture, so each description would land in a lecture of its
    own and nothing could be seen to repeat. The two are asserted apart here so that
    reading the wrong field fails a test rather than quietly changing every number.
    """
    from ad_for_edu.compliance.sequence import SequenceItem
    from ad_for_edu.core.ids import lecture_of

    texts = ["Mitochondrion cristae matrix."] * 4
    proper = [f"lec-a#{index:04d}" for index in range(1, 5)]
    legacy = [f"lec-a_seg{index:04d}" for index in range(1, 5)]
    assert {lecture_of(moment) for moment in proper} == {"lec-a"}
    assert len({lecture_of(moment) for moment in legacy}) == 4

    factor = build_sequence_factor()

    def cuts(ids):
        sequence = [
            SequenceItem(moment, 10.0 * index, text)
            for index, (moment, text) in enumerate(zip(ids, texts, strict=True), start=1)
        ]
        return any(
            effect.factor < 1.0 for effect in factor.effects(sequence).values()
        )

    assert cuts(proper), "the moment id groups by lecture, so repetition is seen"
    assert not cuts(legacy), "a per-row identifier does not, which is why it is unused"


def test_the_knee_is_the_calibrated_one():
    assert build_sequence_factor().knee == pytest.approx(0.4)


def test_settings_without_a_sequence_factor_are_refused(tmp_path, shipped_configs):
    """A compliance score includes the factor, so settings that omit it are not a mode."""
    from ad_for_edu.compliance import load_compliance_settings
    from ad_for_edu.core.errors import SettingsError

    text = (shipped_configs / "compliance.yaml").read_text(encoding="utf-8")
    without = "\n".join(
        line
        for line in text.splitlines()
        if not line.startswith(("sequence_factor:", "  name: novelty_knee", "  params: {knee:"))
    )
    absent = tmp_path / "absent.yaml"
    absent.write_text(without, encoding="utf-8")
    with pytest.raises(SettingsError, match="sequence_factor"):
        load_compliance_settings(absent)

    # Present but empty, which the required-key check cannot catch.
    empty = tmp_path / "empty.yaml"
    empty.write_text(without + "\nsequence_factor:\n", encoding="utf-8")
    with pytest.raises(SettingsError, match="sequence_factor is required"):
        load_compliance_settings(empty)


def test_the_local_mode_reaches_its_rules_as_shortfalls_not_breaches():
    """The two model-backed components are graded, so they report a shortfall.

    Their rules are reached, which is what the local mode adds, but a partial
    contradiction reading is not a contradiction and the row says so.
    """
    scorer = build_scorer("local", models=FakeModels())
    assert scorer.names_rules
    row = scorer.score(
        "The mitochondrion is not shown here, and you could see it earlier.",
        context("lec-a#0001", "figure"),
    )
    assert row.overall < 1.0
    assert set(row.short) & set(LOCAL_ONLY_RULES)
    assert not set(row.broken) & set(LOCAL_ONLY_RULES)


# ------------------------------------------- a shortfall is not a breach
def test_the_three_graded_components_say_a_shortfall_is_not_a_breach():
    from ad_for_edu.compliance import COMPONENTS

    graded = {
        name
        for name in COMPONENTS.names()
        if not COMPONENTS.get(name).shortfall_is_a_breach
    }
    assert graded == {
        "terminology_grounding",
        "faithfulness_nli",
        "non_redundancy_embedding",
    }


def test_a_description_that_adds_a_word_breaks_no_terminology_rule():
    """The case that would otherwise read as ninety-six per cent broken."""
    scorer = build_scorer("mechanical")
    row = scorer.score(
        "Mitochondrion cristae matrix, the organelle that makes the cell's energy.",
        context("lec-a#0001", "slide"),
    )
    assert row.components["terminology"] < 1.0
    assert "cross_cutting_004" in row.short
    assert "cross_cutting_004" not in row.broken


def test_a_style_violation_is_a_breach_not_a_shortfall():
    scorer = build_scorer("mechanical")
    row = scorer.score(
        "You could see that the slide showed it.", context("lec-a#0001", "slide")
    )
    assert "cross_cutting_002" in row.broken
    assert "cross_cutting_002" not in row.short


def test_the_diagnostic_keeps_the_two_columns_apart(rules):
    scorer, rows = scored_rows()
    found = rule_diagnostic(rows, scorer, rules)
    terminology = found["per_rule"]["cross_cutting_004"]
    assert terminology["broken_by"] == 0
    assert terminology["scored_below_full_credit"] > 0
    rendered = render_rule_diagnostic(found, title="t")
    assert "not a breach" in rendered
