"""The preference head: the fit, the folds, the frozen shot and what each guard refuses."""

from __future__ import annotations

import random

import pytest

from ad_for_edu.agreement.head import (
    FeaturedPair,
    Head,
    features_from_scores,
    fit,
    freeze,
    logistic_weights,
    out_of_fold,
    score_frozen,
    sign_stable,
)
from ad_for_edu.core.errors import ContractError


def pairs_where_x_decides(lectures=("lec-a", "lec-b", "lec-c"), each=30, noise=0.1, seed=0):
    """Pairs whose better side is the one with the larger x, plus an unrelated y."""
    generator = random.Random(seed)
    pairs, labels = {}, {"r1": {}, "r2": {}, "r3": {}}
    for lecture in lectures:
        for index in range(each):
            pair_id = f"{lecture}#{index:04d}"
            first, second = generator.random(), generator.random()
            pairs[pair_id] = FeaturedPair(
                pair_id,
                lecture,
                {"x": first, "y": generator.random()},
                {"x": second, "y": generator.random()},
            )
            better = "a" if first > second else "b"
            for rater in labels:
                flipped = "b" if better == "a" else "a"
                labels[rater][pair_id] = flipped if generator.random() < noise else better
    return pairs, labels


# ------------------------------------------------------------------- the fit itself
def test_the_fit_recovers_the_feature_that_decides():
    pairs, labels = pairs_where_x_decides()
    head = fit(pairs, labels, ["x", "y"])
    weight_of = dict(zip(head.features, head.weights, strict=True))
    assert weight_of["x"] > 2.0
    assert abs(weight_of["y"]) < 1.0


def test_the_fit_agrees_with_a_known_logistic_answer():
    # A single feature, six observations, fitted with a ridge of one: the answer is
    # the root of the score equation, which is checked here against its own gradient.
    rows = [[1.0], [1.0], [1.0], [-1.0], [-1.0], [-1.0]]
    outcomes = [1, 1, 0, 0, 0, 1]
    (weight,) = logistic_weights(rows, outcomes, strength=1.0)
    gradient = sum(
        (outcome - 1.0 / (1.0 + pow(2.718281828459045, -weight * row[0]))) * row[0]
        for row, outcome in zip(rows, outcomes, strict=True)
    ) - weight
    assert abs(gradient) < 1e-6


def test_counting_a_side_twice_weighs_it_twice():
    rows = [[1.0], [-1.0]]
    once = logistic_weights(rows, [1, 0], [1.0, 1.0])
    heavier = logistic_weights(rows, [1, 0], [4.0, 1.0])
    assert heavier[0] > once[0]


# --------------------------------------------------------------- antisymmetry
def test_the_same_pair_swapped_gives_the_other_side():
    head = Head(("x", "y"), (2.0, -0.5))
    difference = [0.3, 0.2]
    assert head.prefers(difference) == "a"
    assert head.prefers([-value for value in difference]) == "b"
    assert head.margin(difference) == pytest.approx(-head.margin([-v for v in difference]))


def test_a_feature_can_be_held_equal_at_prediction_time():
    head = Head(("x", "words"), (1.0, 5.0))
    difference = [0.1, -0.5]
    assert head.prefers(difference) == "b"
    assert head.prefers(head.held(difference, "words")) == "a"


def test_holding_a_feature_that_is_not_there_is_refused():
    head = Head(("x",), (1.0,))
    with pytest.raises(ContractError, match="no feature 'words'"):
        head.held([0.1], "words")


# ------------------------------------------------------------------- missing values
def test_a_pair_missing_a_feature_on_one_side_has_no_difference():
    pair = FeaturedPair("p1", "lec-a", {"x": 1.0}, {"x": 0.5, "y": 2.0})
    assert pair.difference(["x"]) == [0.5]
    assert pair.difference(["x", "y"]) is None


def test_a_pair_missing_a_feature_is_counted_as_skipped_not_scored():
    pairs, labels = pairs_where_x_decides()
    lost = next(iter(pairs))
    pairs[lost] = FeaturedPair(lost, pairs[lost].lecture, {"x": 1.0}, {"x": 0.0})
    found = out_of_fold(pairs, labels, ["x", "y"])
    assert lost not in found.preferred
    assert found.skipped == 1


# ------------------------------------------------------------------- out of fold
def test_every_pair_is_predicted_by_a_fit_that_saw_no_pair_of_its_lecture():
    pairs, labels = pairs_where_x_decides()
    found = out_of_fold(pairs, labels, ["x", "y"])
    assert set(found.preferred) == set(pairs)
    assert found.folds == 3
    assert {found.fold_of[pair_id] for pair_id in found.preferred} == {
        "lec-a",
        "lec-b",
        "lec-c",
    }
    for pair_id, fold in found.fold_of.items():
        assert pairs[pair_id].lecture == fold


def test_out_of_fold_predictions_track_the_raters_above_chance():
    pairs, labels = pairs_where_x_decides()
    found = out_of_fold(pairs, labels, ["x", "y"])
    agreed = sum(1 for pair_id, side in labels["r1"].items() if found.preferred[pair_id] == side)
    assert agreed / len(labels["r1"]) > 0.8


def test_one_lecture_cannot_be_predicted_out_of_fold():
    pairs, labels = pairs_where_x_decides(lectures=("lec-a",))
    with pytest.raises(ContractError, match="at least two lectures"):
        out_of_fold(pairs, labels, ["x", "y"])


def test_the_fold_weights_are_reported_with_their_spread():
    pairs, labels = pairs_where_x_decides()
    found = out_of_fold(pairs, labels, ["x", "y"])
    as_dict = found.as_dict()
    assert as_dict["fold_disjoint_on"] == "lecture"
    assert set(as_dict["mean_weights"]) == {"x", "y"}
    assert all(value >= 0 for value in as_dict["weight_spread"].values())


# ------------------------------------------------------------------ sign stability
def test_a_feature_the_raters_pull_opposite_ways_is_dropped():
    pairs = {}
    labels = {"r1": {}, "r2": {}}
    generator = random.Random(5)
    for index in range(40):
        for lecture in ("lec-a", "lec-b"):
            pair_id = f"{lecture}#{index:04d}"
            first, second = generator.random(), generator.random()
            pairs[pair_id] = FeaturedPair(
                pair_id, lecture, {"x": first, "taste": 1.0}, {"x": second, "taste": 0.0}
            )
            better = "a" if first > second else "b"
            labels["r1"][pair_id] = "a" if index % 2 else better
            labels["r2"][pair_id] = "b" if index % 2 else better
    stable, dropped = sign_stable(pairs, labels, ["x", "taste"])
    assert "taste" in dropped and "x" in stable


def test_a_named_feature_is_kept_whatever_its_signs():
    pairs, labels = pairs_where_x_decides()
    stable, dropped = sign_stable(pairs, labels, ["x", "y"], keep=["y"])
    assert "y" in stable and "y" not in dropped


# ------------------------------------------------------------------- the frozen shot
def test_the_frozen_weights_never_saw_the_held_out_rater():
    pairs, labels = pairs_where_x_decides()
    # Give one rater the opposite preference: weights that had seen it would follow.
    labels["r3"] = {
        pair_id: ("b" if side == "a" else "a")
        for pair_id, side in labels["r3"].items()
    }
    frozen = freeze(pairs, labels, ["x", "y"], held_out_rater="r3")
    scored = score_frozen(frozen, pairs, labels["r3"])
    assert scored["refitted"] is False
    assert scored["agreement"] < 0.3
    assert scored["n"] == len(labels["r3"])
    assert scored["interval"]["method"] == "wilson"


def test_holding_out_a_rater_who_is_not_there_is_refused():
    pairs, labels = pairs_where_x_decides()
    with pytest.raises(ContractError, match="no rater 'r9'"):
        freeze(pairs, labels, ["x"], held_out_rater="r9")


def test_holding_out_the_only_rater_is_refused():
    pairs, labels = pairs_where_x_decides()
    with pytest.raises(ContractError, match="leaves no one"):
        freeze(pairs, {"r1": labels["r1"]}, ["x"], held_out_rater="r1")


def test_a_tie_is_neither_agreement_nor_disagreement():
    pairs, _ = pairs_where_x_decides(lectures=("lec-a", "lec-b"), each=2)
    head = Head(("x",), (1.0,))
    every = {pair_id: "tie" for pair_id in pairs}
    assert score_frozen(head, pairs, every)["n"] == 0


# --------------------------------------------------------------------- guards
def test_weights_and_features_must_be_the_same_length():
    with pytest.raises(ContractError, match="2 features and 1 weights"):
        Head(("x", "y"), (1.0,))


def test_a_pair_with_the_wrong_number_of_differences_is_refused():
    head = Head(("x", "y"), (1.0, 1.0))
    with pytest.raises(ContractError, match="1 differences against 2 weights"):
        head.margin([0.5])


def test_rows_of_different_widths_are_refused():
    with pytest.raises(ContractError, match="the same features"):
        logistic_weights([[1.0, 2.0], [1.0]], [1, 0])


def test_fitting_on_nothing_is_refused():
    with pytest.raises(ContractError, match="nothing to fit on"):
        logistic_weights([], [])


def test_a_count_per_row_is_required_if_counts_are_given():
    with pytest.raises(ContractError, match="2 rows and 1 counts"):
        logistic_weights([[1.0], [-1.0]], [1, 0], [1.0])


def test_collinear_features_are_refused_rather_than_solved_arbitrarily():
    rows = [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]
    with pytest.raises(ContractError, match="collinear"):
        logistic_weights(rows, [1, 1, 0], strength=0.0)


def test_a_fit_with_no_usable_pair_is_refused():
    pairs = {"p1": FeaturedPair("p1", "lec-a", {"x": 1.0}, {"x": 0.0})}
    with pytest.raises(ContractError, match="no pair carries every feature"):
        fit(pairs, {"r1": {"p1": "tie"}}, ["x"])


# ------------------------------------------------------------------ reading features
def test_features_are_read_from_a_table_of_per_side_numbers():
    rows = [{"pair_id": "p1", "lecture": "lec-a"}, {"pair_id": "p2", "moment_id": "lec-b#0003"}]
    scored = {
        "p1": {"a": {"x": 1.0}, "b": {"x": 0.5}},
        "p2": {"a": {"x": 0.2}, "b": {"x": 0.9}},
        "p3": {"a": {"x": 0.1}, "b": {"x": 0.1}},
    }
    found = features_from_scores(rows, scored)
    assert sorted(found) == ["p1", "p2"]
    assert found["p2"].lecture == "lec-b"
    assert found["p1"].difference(["x"]) == [0.5]


def test_a_pair_scored_on_one_side_only_is_left_out():
    rows = [{"pair_id": "p1", "lecture": "lec-a"}]
    assert features_from_scores(rows, {"p1": {"a": {"x": 1.0}}}) == {}
