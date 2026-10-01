"""The scorer, the modes, the novelty factor, and scoring a system."""

from __future__ import annotations

import pytest

from ad_for_edu.compliance import (
    COMPONENTS,
    ComplianceScorer,
    NoveltyKnee,
    SequenceItem,
    Tier,
    build_scorer,
    build_sequence_factor,
    load_compliance_settings,
)
from ad_for_edu.compliance.components.base import Component
from ad_for_edu.compliance.contexts import (
    InMemoryContexts,
    ReferenceRowContexts,
    load_cursor_facts,
)
from ad_for_edu.compliance.report import render, score_predictions, summarise, system_score
from ad_for_edu.compliance.sequence import within_lecture_novelty
from ad_for_edu.core.errors import ContractError, SettingsError
from ad_for_edu.core.io import write_json, write_jsonl
from ad_for_edu.data.schema import CATEGORIES
from tests.fixtures.factory import FakeModels, context, predictions, reference_rows

MECHANICAL = ("style", "terminology", "length", "deixis")


@pytest.fixture
def settings(shipped_configs):
    return load_compliance_settings(shipped_configs / "compliance.yaml")


# ---------------------------------------------------------------- modes
def test_the_shipped_settings_define_three_modes(settings):
    assert sorted(settings.modes) == ["local", "mechanical", "rubric"]
    assert settings.modes["mechanical"].abstains == {"faithfulness", "non_redundancy"}
    assert settings.modes["local"].abstains == frozenset()
    assert settings.modes["rubric"].paid


def test_the_shipped_knees(settings):
    assert settings.sequence_factor.name == "novelty_knee"
    assert settings.sequence_factor.params == {"knee": 0.4}
    assert settings.kappa_sweep == (0.2, 0.3, 0.4, 0.5, 0.6)


def test_the_mechanical_mode_scores_four_categories(settings):
    row = build_scorer("mechanical", settings).score("A folded membrane.", context())
    assert [c for c in CATEGORIES if row.components[c] is not None] == list(MECHANICAL)
    assert row.abstained == ("faithfulness", "non_redundancy")


def test_the_local_mode_scores_all_six(settings):
    scorer = build_scorer("local", settings, models=FakeModels())
    row = scorer.score("A folded membrane.", context())
    assert all(row.components[c] is not None for c in CATEGORIES)
    assert row.abstained == ()


def test_the_paid_mode_is_not_built_here(settings):
    with pytest.raises(SettingsError):
        build_scorer("rubric", settings)


def test_an_unknown_mode_lists_the_known_ones(settings):
    with pytest.raises(SettingsError) as caught:
        build_scorer("mech", settings)
    assert "mechanical" in str(caught.value)


def test_a_mode_with_a_misspelled_key_is_refused():
    with pytest.raises(SettingsError):
        Tier.from_settings("mechanical", {"component": []})
    with pytest.raises(SettingsError):
        Tier.from_settings("mechanical", {"abstains": ["redundancy"]})


# ---------------------------------------------------------------- scorer
def test_the_score_is_the_unweighted_mean_of_the_applied_components(settings):
    scorer = build_scorer("mechanical", settings)
    moment = context(slide_ocr="Mitochondrion cristae", what_on_screen="", pause_after=0.0)
    # Content terms: slide, shows, mitochondrion, cristae. Two of the four are on the slide.
    row = scorer.score("The slide shows mitochondrion cristae.", moment)
    assert row.components["style"] == 0.5
    assert row.components["terminology"] == 0.5
    assert row.components["length"] == 1.0
    assert row.components["deixis"] == 1.0
    assert row.overall == pytest.approx((0.5 + 0.5 + 1.0 + 1.0) / 4)


def test_a_component_that_does_not_apply_is_left_out_of_the_mean(settings):
    scorer = build_scorer("mechanical", settings)
    moment = context(slide_ocr="Mitochondrion", what_on_screen="", renders_cursor=False)
    row = scorer.score("Look at this one over here.", moment)
    assert row.components["deixis"] is None
    assert "deixis" in row.abstained
    applied = [row.components[c] for c in ("style", "terminology", "length")]
    assert row.overall == pytest.approx(sum(applied) / 3)


def test_the_same_text_scores_lower_when_deixis_applies(settings):
    scorer = build_scorer("mechanical", settings)
    text = "Look at this one over here."
    left_out = scorer.score(text, context(slide_ocr="Mitochondrion", renders_cursor=False))
    scored = scorer.score(text, context(slide_ocr="Mitochondrion", renders_cursor=None))
    assert scored.components["deixis"] == 0.0
    assert scored.overall < left_out.overall


@pytest.mark.parametrize("text", [None, "", "   ", "None"])
def test_an_empty_description_keeps_its_place_unscored(settings, text):
    row = build_scorer("mechanical", settings).score(text, context())
    assert row.overall is None and row.reason == "unfilled_text"
    assert set(row.components) == set(CATEGORIES)


def test_a_scorer_cannot_claim_a_mode_it_does_not_implement(settings):
    local_components = build_scorer("local", settings, models=FakeModels()).components
    with pytest.raises(ContractError):
        ComplianceScorer(settings.modes["mechanical"], local_components)
    mechanical_components = build_scorer("mechanical", settings).components
    with pytest.raises(ContractError):
        ComplianceScorer(settings.modes["local"], mechanical_components)


def test_two_components_for_one_category_are_refused(settings):
    components = list(build_scorer("mechanical", settings).components)
    components.append(COMPONENTS.create("style_properties"))
    with pytest.raises(ContractError):
        ComplianceScorer(settings.modes["mechanical"], components)


def test_a_component_that_stops_applying_is_noticed(settings):
    class SilentStyle(Component):
        category = "style"

        def applies(self, moment):
            return False

        def score(self, text, moment):
            return 1.0

    components = [
        c
        for c in build_scorer("mechanical", settings).components
        if c.category != "style"
    ]
    scorer = ComplianceScorer(settings.modes["mechanical"], [SilentStyle(), *components])
    with pytest.raises(ContractError):
        scorer.score("A folded membrane.", context())


def test_a_score_outside_the_unit_interval_is_refused(settings):
    class Overflowing(Component):
        category = "style"

        def score(self, text, moment):
            return 1.5

    components = [
        c
        for c in build_scorer("mechanical", settings).components
        if c.category != "style"
    ]
    scorer = ComplianceScorer(settings.modes["mechanical"], [Overflowing(), *components])
    with pytest.raises(ContractError):
        scorer.score("A folded membrane.", context())


def test_the_order_of_components_in_the_settings_does_not_change_the_score(settings):
    built = build_scorer("mechanical", settings)
    reversed_scorer = ComplianceScorer(settings.modes["mechanical"], list(built.components)[::-1])
    text = "The slide shows this thing and a mitochondrion."
    assert reversed_scorer.score(text, context()).overall == built.score(text, context()).overall


# ---------------------------------------------------------------- novelty
def items(texts, lecture="lec-a"):
    return [
        SequenceItem(f"{lecture}#{index:04d}", float(index), text)
        for index, text in enumerate(texts, start=1)
    ]


def test_the_first_description_of_a_lecture_has_full_novelty():
    novelty = within_lecture_novelty(items(["Cell structure", "Cell structure"]))
    assert novelty["lec-a#0001"] == 1.0
    assert novelty["lec-a#0002"] == 0.0


def test_a_description_is_compared_with_every_earlier_one():
    novelty = within_lecture_novelty(items(["alpha beta", "gamma delta", "alpha beta"]))
    assert novelty["lec-a#0003"] == 0.0


def test_lectures_do_not_see_each_other():
    sequence = items(["alpha beta"], "lec-a") + items(["alpha beta"], "lec-b")
    assert set(within_lecture_novelty(sequence).values()) == {1.0}


def test_the_order_is_by_time_then_by_moment_id():
    sequence = [
        SequenceItem("lec-a#0002", 5.0, "alpha beta"),
        SequenceItem("lec-a#0001", 5.0, "alpha beta gamma delta"),
        SequenceItem("lec-a#0009", 1.0, "omega"),
    ]
    novelty = within_lecture_novelty(sequence)
    assert novelty["lec-a#0009"] == 1.0
    assert novelty["lec-a#0001"] == 1.0  # nothing in common with "omega"
    assert novelty["lec-a#0002"] == pytest.approx(0.5)


def test_an_empty_description_overlaps_with_nothing():
    novelty = within_lecture_novelty(items(["alpha", "", "..."]))
    assert novelty["lec-a#0002"] == 1.0 and novelty["lec-a#0003"] == 1.0


@pytest.mark.parametrize(
    ("novelty", "factor"), [(1.0, 1.0), (0.4, 1.0), (0.2, 0.5), (0.0, 0.0), (0.39, 0.975)]
)
def test_the_knee_gives_full_credit_at_or_above_it(novelty, factor):
    assert NoveltyKnee(0.4).factor(novelty) == pytest.approx(factor)


@pytest.mark.parametrize("knee", [0, -0.1, 1.5])
def test_the_knee_is_validated(knee):
    with pytest.raises(ValueError):
        NoveltyKnee(knee)


def test_the_knee_can_be_overridden_for_a_sweep(settings):
    assert build_sequence_factor(settings).knee == 0.4
    assert build_sequence_factor(settings, knee=0.2).knee == 0.2


# ---------------------------------------------------------------- scoring a system
def contexts_and_times():
    source = InMemoryContexts(reference_rows())
    return source.contexts(), source.times()


def test_every_moment_asked_for_gets_a_row(settings):
    contexts, times = contexts_and_times()
    given = predictions({"lec-a#0001": "Cell structure outline.", "lec-a#0002": None})
    rows = score_predictions(given, contexts, build_scorer("mechanical", settings), times=times)
    assert len(rows) == len(contexts) == 6
    reasons = {row["moment_id"]: row["reason"] for row in rows}
    assert reasons["lec-a#0001"] is None
    assert reasons["lec-a#0002"] == "unfilled_text"
    assert reasons["lec-b#0003"] == "missing"


def test_a_system_that_reads_the_title_on_every_moment_is_scaled_down(settings):
    contexts, times = contexts_and_times()
    readout = predictions({moment: "Cell structure" for moment in contexts})
    scorer = build_scorer("mechanical", settings)
    plain = score_predictions(readout, contexts, scorer, times=times)
    scaled = score_predictions(
        readout, contexts, scorer, sequence_factor=NoveltyKnee(0.4), times=times
    )
    by_moment = {row["moment_id"]: row for row in scaled}
    assert by_moment["lec-a#0001"]["novelty_factor"] == 1.0
    assert by_moment["lec-a#0002"]["novelty_factor"] == 0.0
    assert by_moment["lec-a#0002"]["overall"] == 0.0
    assert by_moment["lec-b#0001"]["novelty_factor"] == 1.0
    assert system_score(scaled) < system_score(plain)


def test_a_system_that_says_something_new_each_time_keeps_its_score(settings):
    contexts, times = contexts_and_times()
    varied = predictions(
        {
            "lec-a#0001": "An outline lists four topics.",
            "lec-a#0002": "Folded cristae fill the organelle.",
            "lec-a#0003": "A dark nucleolus sits inside the nucleus.",
        }
    )
    scorer = build_scorer("mechanical", settings)
    plain = score_predictions(varied, contexts, scorer, times=times)
    scaled = score_predictions(
        varied, contexts, scorer, sequence_factor=NoveltyKnee(0.4), times=times
    )
    assert system_score(scaled) == pytest.approx(system_score(plain))


def test_an_empty_description_takes_no_part_in_the_ordering(settings):
    contexts, times = contexts_and_times()
    given = predictions(
        {"lec-a#0001": None, "lec-a#0002": "Cell structure", "lec-a#0003": "Cell structure"}
    )
    rows = score_predictions(
        given,
        contexts,
        build_scorer("mechanical", settings),
        sequence_factor=NoveltyKnee(0.4),
        times=times,
    )
    by_moment = {row["moment_id"]: row for row in rows}
    assert "novelty" not in by_moment["lec-a#0001"]
    assert by_moment["lec-a#0002"]["novelty"] == 1.0
    assert by_moment["lec-a#0003"]["novelty"] == 0.0


def test_the_sequence_factor_needs_the_times(settings):
    contexts, _ = contexts_and_times()
    with pytest.raises(ContractError):
        score_predictions(
            predictions({"lec-a#0001": "x y z"}),
            contexts,
            build_scorer("mechanical", settings),
            sequence_factor=NoveltyKnee(0.4),
        )


def test_a_moment_predicted_twice_is_refused(settings):
    contexts, times = contexts_and_times()
    twice = predictions({"lec-a#0001": "One."}) + predictions({"lec-a#0001": "Two."})
    with pytest.raises(ContractError):
        score_predictions(twice, contexts, build_scorer("mechanical", settings), times=times)


def test_a_prediction_for_an_unknown_moment_is_refused(settings):
    contexts, times = contexts_and_times()
    with pytest.raises(ContractError):
        score_predictions(
            predictions({"lec-z#0001": "One."}),
            contexts,
            build_scorer("mechanical", settings),
            times=times,
        )


def test_the_summary_lists_groups_before_the_aggregate(settings):
    contexts, times = contexts_and_times()
    given = predictions({moment: "A labelled diagram appears." for moment in contexts})
    rows = score_predictions(given, contexts, build_scorer("mechanical", settings), times=times)
    summary = summarise(rows)
    assert list(summary["strata"]) == ["chart", "code", "figure", "pointing", "slide"]
    assert summary["aggregate"]["n_total"] == 6
    assert summary["aggregate"]["components"]["faithfulness"]["n_applicable"] == 0
    assert summary["aggregate"]["components"]["style"]["n_applicable"] == 6
    report = render(summary, title="Mechanical mode")
    assert report.index("type: chart") < report.index("all groups together")


def test_the_system_score_is_the_mean_over_scored_descriptions():
    rows = [{"overall": 1.0}, {"overall": 0.5}, {"overall": None}]
    assert system_score(rows) == 0.75
    assert system_score([{"overall": None}]) is None


# ---------------------------------------------------------------- contexts
def test_the_first_row_of_a_moment_supplies_its_context(tmp_path):
    rows = reference_rows()
    rows[1]["slide_ocr"] = "a later row with other text"
    path = tmp_path / "references.jsonl"
    write_jsonl(path, rows)
    source = ReferenceRowContexts(path)
    assert len(source.contexts()) == 6
    assert source.context("lec-a#0001").slide_ocr == "Cell structure"
    assert source.times()["lec-a#0002"] == 47.5


def test_a_moment_where_every_writer_stayed_silent_still_has_a_context():
    rows = [row for row in reference_rows() if row["family"] == "writer-z"]
    assert all(not row["emit"] for row in rows)
    assert len(InMemoryContexts(rows).contexts()) == 6


def test_pointer_facts_reach_the_context(tmp_path):
    facts = tmp_path / "pointer.json"
    write_json(facts, {"lec-a": {"pointing_allowed": False}, "lec-b": True})
    references = tmp_path / "references.jsonl"
    write_jsonl(references, reference_rows())
    source = ReferenceRowContexts(references, cursor_facts=facts)
    assert source.context("lec-a#0001").renders_cursor is False
    assert source.context("lec-b#0001").renders_cursor is True


def test_without_pointer_facts_the_fact_is_unknown_not_false():
    assert InMemoryContexts(reference_rows()).context("lec-a#0001").renders_cursor is None


def test_a_pointer_file_that_omits_the_fact_is_refused(tmp_path):
    facts = tmp_path / "pointer.json"
    write_json(facts, {"lec-a": {"pointing_allowed": False}, "lec-b": {"note": "unknown"}})
    with pytest.raises(ContractError) as caught:
        load_cursor_facts(facts)
    assert "lec-b" in str(caught.value)


def test_an_unknown_moment_has_no_context():
    with pytest.raises(ContractError):
        InMemoryContexts(reference_rows()).context("lec-z#0001")
