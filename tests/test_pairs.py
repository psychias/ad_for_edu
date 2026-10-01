"""Candidates, controlled pairs, pairing, drawing sets and ordering."""

from __future__ import annotations

import random

import pytest

from ad_for_edu.core.errors import ContractError, SettingsError
from ad_for_edu.data.schema import CATEGORIES
from ad_for_edu.pairs import (
    BANDS,
    CANDIDATE_MODES,
    CONTROLLED_AXES,
    ORDER_COMBINATIONS,
    ORDERS,
    Candidate,
    DrawPlan,
    Offered,
    Outcome,
    axes_for,
    band_bounds,
    band_for,
    directives,
    distance_between,
    draw,
    hold_out_by_lecture,
    load_candidate_settings,
    load_controlled_settings,
    load_judging_settings,
    load_pair_set_settings,
    order_by_construction,
    order_by_judge,
    pairs_of_moment,
    read_answer,
    side_shown_first,
    summarise,
    winner_of,
)
from ad_for_edu.pairs.build import build_axes, build_candidate_mode, build_combination
from ad_for_edu.prompts.contracts import FOCUS_DIRECTIVES

MOMENT = {
    "type": "figure",
    "transcript_window": "so the energy of the cell is produced in this organelle",
    "slide_ocr": "Mitochondrion: cristae, matrix",
    "pause_after": 1.5,
    "reachable_rung": 3,
    "described_so_far": "(none yet)",
}
WITH_BAND = {**MOMENT, "directive": "REFERENT", "band": "M", "band_min": 16, "band_max": 25}


# ---------------------------------------------------------------- bands
def test_the_four_bands():
    assert set(BANDS) == {"S", "M", "L", "XL"}
    assert BANDS["M"] == (16, 25)


def test_the_widest_band_has_a_ceiling_the_moment_can_deliver():
    # The third rung delivers up to twenty seconds late: about 47 words at 140 a minute.
    assert band_bounds("XL", pause=0.0, rung=3) == (40, 46)
    # A long gap can deliver the whole band.
    assert band_bounds("XL", pause=20.0, rung=3) == (40, 60)


def test_a_moment_that_cannot_deliver_the_floor_does_not_get_the_widest_band():
    assert band_bounds("XL", pause=0.0, rung=4) is None


def test_the_other_bands_have_the_bounds_they_declare():
    assert band_bounds("M", pause=0.0, rung=3) == (16, 25)
    assert band_bounds("S", pause=99.0, rung=1) == (12, 15)


def test_a_moment_the_words_leave_unsaid_gets_the_widest_band_where_it_can():
    band, low, high = band_for("UNCOVERED", rung=3, pause=0.0)
    assert band == "XL" and (low, high) == (40, 46)


def test_a_moment_the_words_partly_cover_gets_the_band_of_its_rung():
    assert band_for("PARTLY_COVERED", rung=3, pause=0.0)[0] == "M"
    assert band_for("UNCOVERED", rung=2, pause=0.0)[0] == "S"
    assert band_for("FULLY_COVERED", rung=1, pause=20.0)[0] == "L"


# ---------------------------------------------------------------- candidates
def test_the_two_ways_of_asking():
    assert CANDIDATE_MODES.names() == ("per_call", "verbalized_sampling")
    assert CANDIDATE_MODES.create("verbalized_sampling").per_call == 5
    assert CANDIDATE_MODES.create("per_call").per_call == 1


@pytest.mark.parametrize("name", CANDIDATE_MODES.names())
def test_the_word_bounds_reach_the_model(name):
    rendered = CANDIDATE_MODES.create(name).render(WITH_BAND)
    assert "16" in rendered and "25" in rendered and "REFERENT" in rendered


def test_a_call_without_the_bounds_is_refused():
    values = {key: value for key, value in WITH_BAND.items() if key != "band_max"}
    with pytest.raises(ContractError):
        CANDIDATE_MODES.create("per_call").render(values)


def test_several_descriptions_are_read_from_one_reply():
    mode = CANDIDATE_MODES.create("verbalized_sampling")
    found = mode.parse(
        '[{"ad_text": "Folded cristae.", "rung": 3}, '
        '{"ad_text": "The matrix fills it.", "rung": 3}]'
    )
    assert [candidate.ad_text for candidate in found] == [
        "Folded cristae.",
        "The matrix fills it.",
    ]


def test_one_description_is_read_from_a_reply_that_is_one_object():
    assert len(CANDIDATE_MODES.create("per_call").parse('{"ad_text": "x", "rung": 1}')) == 1
    assert len(CANDIDATE_MODES.create("verbalized_sampling").parse('{"ad_text": "x"}')) == 1


def test_an_entry_without_a_description_is_not_a_candidate():
    found = CANDIDATE_MODES.create("verbalized_sampling").parse(
        '[{"ad_text": "  "}, {"rung": 3}, {"ad_text": "Real."}]'
    )
    assert [candidate.ad_text for candidate in found] == ["Real."]


def test_a_reply_with_no_object_gives_nothing():
    for name in CANDIDATE_MODES.names():
        assert CANDIDATE_MODES.create(name).parse("I cannot do this") == []
        assert CANDIDATE_MODES.create(name).parse(None) == []


def test_a_description_is_usable_inside_its_band_and_with_a_rung():
    inside = Candidate(" ".join(["word"] * 20), rung=3)
    assert inside.words == 20 and inside.usable(16, 25)
    assert not Candidate(" ".join(["word"] * 12), rung=3).usable(16, 25)
    assert not Candidate(" ".join(["word"] * 40), rung=3).usable(16, 25)
    assert not Candidate(" ".join(["word"] * 20), rung=None).usable(16, 25)
    assert not Candidate(" ".join(["word"] * 20), rung=5).usable(16, 25)
    assert not Candidate("   ", rung=3).usable(0, 100)


def test_the_five_focus_directives_are_declared_and_named():
    assert tuple(directives()) == FOCUS_DIRECTIVES
    assert len(FOCUS_DIRECTIVES) == 5
    assert set(directives(["REFERENT"])) == {"REFERENT"}
    with pytest.raises(ValueError):
        directives(["COLOUR"])


# ---------------------------------------------------------------- controlled pairs
def test_one_category_of_the_standard_each():
    assert sorted(CONTROLLED_AXES.names()) == sorted(CATEGORIES)


def test_the_category_about_pointing_needs_a_moment_with_pointing():
    deixis = CONTROLLED_AXES.create("deixis")
    assert deixis.applies_to("pointing") and not deixis.applies_to("figure")
    assert CONTROLLED_AXES.create("style").applies_to("figure")
    assert axes_for("figure") == [name for name in CONTROLLED_AXES.names() if name != "deixis"]
    assert "deixis" in axes_for("pointing")


def test_the_category_about_pointing_draws_two_pairs_from_one_moment():
    assert CONTROLLED_AXES.create("deixis").pairs_per_moment == 2
    assert CONTROLLED_AXES.create("style").pairs_per_moment == 1


def test_a_pair_that_keeps_its_claim_has_no_faults():
    style = CONTROLLED_AXES.create("style")
    pair = {
        "axis": "style",
        "a": "Two curves cross at the threshold value.",
        "b": "Two curves crossed at the threshold value.",
        "what_differs": "tense",
    }
    assert style.violations(pair) == []


def test_a_pair_whose_sides_differ_in_length_does_not_isolate_its_category():
    style = CONTROLLED_AXES.create("style")
    pair = {
        "axis": "style",
        "a": "Two curves cross.",
        "b": "Two curves crossed at the threshold value after a long and careful build up.",
        "what_differs": "tense",
    }
    faults = style.violations(pair)
    assert any("words" in fault for fault in faults)


def test_the_category_about_length_requires_the_sides_to_differ_in_length():
    length = CONTROLLED_AXES.create("length")
    same = {
        "axis": "length",
        "a": "Two curves cross.",
        "b": "Two curves meet.",
        "what_differs": "x",
    }
    faults = length.violations(same)
    assert any("audible" in fault for fault in faults)
    longer = {
        "axis": "length",
        "a": "Two curves cross.",
        "b": " ".join(["word"] * 40),
        "what_differs": "length",
    }
    assert longer["b"] and length.violations(longer) == []


def test_two_identical_sides_are_not_a_pair():
    faults = CONTROLLED_AXES.create("style").violations(
        {"axis": "style", "a": "Same text.", "b": "Same text.", "what_differs": "x"}
    )
    assert any("the same" in fault for fault in faults)


def test_a_pair_that_names_another_category_is_a_fault():
    faults = CONTROLLED_AXES.create("style").violations(
        {"axis": "deixis", "a": "One two three.", "b": "One two four.", "what_differs": "x"}
    )
    assert any("category" in fault for fault in faults)


def test_a_pair_missing_a_side_is_a_fault():
    faults = CONTROLLED_AXES.create("style").violations({"axis": "style", "a": "x", "b": ""})
    assert faults


def test_a_category_outside_the_standard_cannot_be_defined():
    from ad_for_edu.pairs.axes import ControlledAxis

    with pytest.raises(TypeError):

        class Wrong(ControlledAxis):
            category = "redundancy"


# ---------------------------------------------------------------- pairing
def offered(*texts_and_writers) -> list[Offered]:
    return [Offered(text=text, writer=writer) for text, writer in texts_and_writers]


def overlap(first: str, second: str) -> float:
    """A stand-in for character overlap: the share of words the two share."""
    one, other = set(first.split()), set(second.split())
    return len(one & other) / len(one | other) if (one | other) else 1.0


def test_two_descriptions_too_alike_are_not_a_pair():
    same = offered(("alpha beta gamma", "x"), ("alpha beta gamma", "y"))
    assert pairs_of_moment(same, overlap, random.Random(0)) == []


def test_pairs_across_writers_come_before_pairs_within_one():
    candidates = offered(
        ("alpha beta", "x"),
        ("gamma delta", "x"),
        ("epsilon zeta", "y"),
    )
    kept = pairs_of_moment(candidates, overlap, random.Random(0), cap=1)
    assert kept[0].across_writers


def test_no_description_is_paired_against_everything():
    candidates = offered(
        ("alpha", "x"), ("beta", "y"), ("gamma", "z"), ("delta", "w"), ("epsilon", "v")
    )
    kept = pairs_of_moment(candidates, overlap, random.Random(0), cap=4, per_description=2)
    appearances: dict[str, int] = {}
    for pairing in kept:
        for side in (pairing.first.text, pairing.second.text):
            appearances[side] = appearances.get(side, 0) + 1
    assert kept and max(appearances.values()) <= 2


def test_a_moment_supplies_no_more_pairs_than_the_cap():
    candidates = offered(*[(f"word{index}", f"writer-{index}") for index in range(6)])
    assert len(pairs_of_moment(candidates, overlap, random.Random(0), cap=3)) == 3


def test_the_pairing_is_reproducible_from_its_seed():
    candidates = offered(*[(f"word{index}", f"writer-{index % 2}") for index in range(6)])
    first = pairs_of_moment(candidates, overlap, random.Random(7), cap=4)
    again = pairs_of_moment(candidates, overlap, random.Random(7), cap=4)
    assert [(one.first.text, one.second.text) for one in first] == [
        (one.first.text, one.second.text) for one in again
    ]


def test_the_distance_does_not_depend_on_which_side_is_given_first():
    assert distance_between("alpha beta", "beta gamma", overlap) == distance_between(
        "beta gamma", "alpha beta", overlap
    )
    assert distance_between("alpha", "alpha", overlap) == 0.0
    assert distance_between("alpha", "beta", overlap) == 100.0


# ---------------------------------------------------------------- drawing a set
def rows(kind: str, count: int, *, axis=None, lecture="lec-a", first=0) -> list[dict]:
    return [
        {
            "pair_id": f"{kind}-{index}",
            "moment_id": f"{lecture}#{index:04d}",
            "lecture": lecture,
            "axis": axis,
            "kind": kind,
        }
        for index in range(first, first + count)
    ]


def test_a_draw_takes_the_number_of_each_kind_it_asks_for():
    available = {
        "controlled": rows("controlled", 30, axis="style")
        + rows("controlled", 30, axis="deixis", first=100),
        "natural_decided": rows("natural_decided", 40, lecture="lec-b"),
    }
    plan = DrawPlan(
        per_kind={"controlled": 20, "natural_decided": 10},
        per_category={"deixis": 12},
        category_default=8,
        seed=3,
    )
    drawn = draw(available, plan)
    assert len(drawn["natural_decided"]) == 10
    per_axis: dict[str, int] = {}
    for row in drawn["controlled"]:
        per_axis[row["axis"]] = per_axis.get(row["axis"], 0) + 1
    assert per_axis == {"deixis": 12, "style": 8}


def test_a_draw_spreads_over_the_lectures():
    available = {
        "natural_decided": (
            rows("natural_decided", 20, lecture="lec-a")
            + rows("natural_decided", 20, lecture="lec-b", first=100)
        )
    }
    drawn = draw(available, DrawPlan(per_kind={"natural_decided": 10}, seed=1))
    lectures = {row["lecture"] for row in drawn["natural_decided"]}
    assert lectures == {"lec-a", "lec-b"}


def test_no_moment_supplies_more_pairs_than_the_cap():
    shared = [
        {"pair_id": f"p{index}", "moment_id": "lec-a#0001", "lecture": "lec-a", "axis": None}
        for index in range(10)
    ]
    drawn = draw(
        {"natural_decided": shared},
        DrawPlan(per_kind={"natural_decided": 10}, maximum_per_moment=2, seed=1),
    )
    assert len(drawn["natural_decided"]) == 2


def test_a_draw_is_reproducible_from_its_seed():
    available = {"natural_decided": rows("natural_decided", 30)}
    plan = DrawPlan(per_kind={"natural_decided": 10}, maximum_per_moment=1, seed=11)
    first = [row["pair_id"] for row in draw(available, plan)["natural_decided"]]
    again = [row["pair_id"] for row in draw(available, plan)["natural_decided"]]
    other = [
        row["pair_id"]
        for row in draw(available, DrawPlan(per_kind={"natural_decided": 10}, seed=12))[
            "natural_decided"
        ]
    ]
    assert first == again and first != other


# ---------------------------------------------------------------- holding out
def test_the_development_set_shares_no_lecture_with_what_is_left():
    pairs = []
    for lecture in ("lec-a", "lec-b", "lec-c", "lec-d"):
        pairs += rows("natural", 25, lecture=lecture)
    kept, held = hold_out_by_lecture(pairs, 0.25, seed=17)
    assert {row["lecture"] for row in kept} & {row["lecture"] for row in held} == set()
    assert len(kept) + len(held) == len(pairs)
    assert 0 < len(held) <= 50


def test_holding_out_is_reproducible_from_its_seed():
    pairs = []
    for lecture in ("lec-a", "lec-b", "lec-c", "lec-d"):
        pairs += rows("natural", 10, lecture=lecture)
    first = hold_out_by_lecture(pairs, 0.25, seed=5)[1]
    again = hold_out_by_lecture(pairs, 0.25, seed=5)[1]
    assert [row["pair_id"] for row in first] == [row["pair_id"] for row in again]


@pytest.mark.parametrize("share", [0.0, 1.0, -0.1, 1.5])
def test_a_share_outside_the_interval_is_refused(share):
    with pytest.raises(ContractError):
        hold_out_by_lecture(rows("natural", 10), share, seed=1)


def test_a_split_by_lecture_needs_more_than_one_lecture():
    with pytest.raises(ContractError):
        hold_out_by_lecture(rows("natural", 10, lecture="lec-a"), 0.2, seed=1)


# ---------------------------------------------------------------- ordering
def test_a_controlled_pair_knows_its_own_direction():
    ordered = order_by_construction({"pair_id": "p1"}, compliant_side="a")
    assert ordered.chosen == "a" and ordered.source == "construction"
    assert not ordered.flipped
    with pytest.raises(ValueError):
        order_by_construction({"pair_id": "p1"}, compliant_side="first")


def test_which_side_is_shown_first_in_each_order():
    assert ORDERS == (0, 1)
    assert side_shown_first(0) == "a" and side_shown_first(1) == "b"
    with pytest.raises(ValueError):
        side_shown_first(2)


def test_an_answer_names_a_recorded_side_whatever_order_it_was_shown_in():
    # Shown as it is recorded: "the first" is the side recorded first.
    assert winner_of("1", 0) == "a" and winner_of("2", 0) == "b"
    # Shown the other way round: "the first" is the side recorded second.
    assert winner_of("1", 1) == "b" and winner_of("2", 1) == "a"
    assert winner_of("tie", 0) == "tie" and winner_of(None, 0) is None


def test_a_reply_is_read_into_the_side_it_names():
    assert read_answer("1", 0) == "a"
    assert read_answer("2", 1) == "a"
    assert read_answer("TIE", 0) == "tie"
    assert read_answer("Description 1 is better than Description 2.", 0) is None


def test_both_orders_must_agree_for_a_pair_to_be_ordered():
    combination = ORDER_COMBINATIONS.create("agree_or_undecided")
    assert combination.combine("a", "a") is Outcome.FIRST
    assert combination.combine("b", "b") is Outcome.SECOND
    assert combination.combine("tie", "tie") is Outcome.TIE
    assert combination.combine("a", "b") is Outcome.UNDECIDED
    assert combination.combine("a", None) is Outcome.UNDECIDED
    assert combination.combine(None, None) is Outcome.UNDECIDED


def test_the_other_rule_records_a_disagreement_as_a_tie():
    combination = ORDER_COMBINATIONS.create("agree_or_tie")
    assert combination.combine("a", "b") is Outcome.TIE
    assert combination.combine("a", "a") is Outcome.FIRST
    assert combination.combine("a", None) is Outcome.UNDECIDED


def test_a_pair_the_judge_answered_differently_each_way_is_marked():
    combination = ORDER_COMBINATIONS.create("agree_or_undecided")
    flipped = order_by_judge("p1", {0: "a", 1: "b"}, combination)
    assert flipped.flipped and flipped.chosen is None
    agreed = order_by_judge("p2", {0: "a", 1: "a"}, combination)
    assert not agreed.flipped and agreed.chosen == "a"


def test_a_pair_with_only_one_order_is_not_ordered():
    combination = ORDER_COMBINATIONS.create("agree_or_undecided")
    assert order_by_judge("p1", {0: "a"}, combination).chosen is None


def test_the_summary_counts_the_outcomes():
    combination = ORDER_COMBINATIONS.create("agree_or_undecided")
    ordered = [
        order_by_judge("p1", {0: "a", 1: "a"}, combination),
        order_by_judge("p2", {0: "a", 1: "b"}, combination),
        order_by_judge("p3", {0: "tie", 1: "tie"}, combination),
        order_by_construction({"pair_id": "p4"}),
    ]
    assert summarise(ordered) == {
        "pairs": 4,
        "ordered": 2,
        "tied": 1,
        "unordered": 1,
        "flipped": 1,
        "share_ordered": 0.5,
    }


# ---------------------------------------------------------------- the settings
def test_the_shipped_candidate_settings(shipped_configs):
    settings = load_candidate_settings(shipped_configs / "candidates.yaml")
    assert settings.mode.name == "verbalized_sampling"
    assert settings.directives == ("REFERENT",)
    assert settings.temperature == 0.7
    assert build_candidate_mode(settings).per_call == 5


def test_the_shipped_controlled_settings(shipped_configs):
    settings = load_controlled_settings(shipped_configs / "controlled_pairs.yaml")
    assert sorted(settings.axes) == sorted(CATEGORIES)
    assert settings.compliant_side == "a"
    assert [axis.category for axis in build_axes(settings)] == list(settings.axes)


def test_the_shipped_pair_set_settings(shipped_configs):
    settings = load_pair_set_settings(shipped_configs / "pair_sets.yaml")
    assert settings.maximum_per_moment == 4 and settings.minimum_distance == 12.0
    assert settings.rated_pool.per_kind == {
        "controlled": 250,
        "natural_decided": 100,
        "natural_flip": 50,
    }
    assert settings.rated_pool.total == 400
    assert settings.rated_pool.per_category == {"deixis": 50}
    assert settings.rated_pool.category_default == 40
    assert settings.rated_pool.maximum_per_moment == 2
    assert settings.head_to_head == {"moments": 150, "seed": 17}
    assert settings.development_share == 0.12


def test_the_numbers_of_the_rated_set_add_up(shipped_configs):
    settings = load_pair_set_settings(shipped_configs / "pair_sets.yaml")
    pool = settings.rated_pool
    other = [name for name in CATEGORIES if name not in pool.per_category]
    assert pool.per_category["deixis"] + len(other) * pool.category_default == pool.per_kind[
        "controlled"
    ]


def test_the_shipped_judging_settings(shipped_configs):
    settings = load_judging_settings(shipped_configs / "pair_judging.yaml")
    assert settings.clip_seconds == 16.0 and settings.clip_width == 640
    assert settings.combination.name == "agree_or_undecided"
    assert build_combination(settings).combine("a", "b") is Outcome.UNDECIDED


def test_the_judge_of_the_settings_may_order_pairs(shipped_configs):
    from ad_for_edu.llm import ModelCatalog

    settings = load_judging_settings(shipped_configs / "pair_judging.yaml")
    catalog = ModelCatalog.load(shipped_configs / "models.yaml")
    catalog.get(settings.model).require_role("pair_judge")


def test_an_unknown_directive_in_the_settings_is_refused(tmp_path):
    path = tmp_path / "candidates.yaml"
    path.write_text("model: m\nmode: per_call\ndirectives: [COLOUR]\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_candidate_settings(path)


def test_a_misspelled_pair_set_key_is_refused(tmp_path):
    path = tmp_path / "pair_sets.yaml"
    path.write_text(
        "pairing: {maximum_per_moments: 4}\nrated_pool: {per_kind: {controlled: 1}}\n",
        encoding="utf-8",
    )
    with pytest.raises(SettingsError):
        load_pair_set_settings(path)
