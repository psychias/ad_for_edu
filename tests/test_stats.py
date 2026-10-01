"""Statistics: known values, stated conventions, reproducibility under a seed."""

from __future__ import annotations

import random
import statistics

import pytest

from ad_for_edu.stats import (
    AGREEMENT,
    CORRECTIONS,
    INTERVALS,
    PAIRED_TESTS,
    PAIRINGS,
    ClusterBootstrap,
    ExactMcNemar,
    Holm,
    KrippendorffNominal,
    Observation,
    UnitBootstrap,
)
from ad_for_edu.stats.agreement import complete_units, unanimous
from ad_for_edu.stats.correlation import rank_of, spearman, spearman_with_interval
from ad_for_edu.stats.intervals import paired_difference, percentile_bounds, wilson
from ad_for_edu.stats.paired import AllPairs, BothDecide, coverage_and_accuracy, exact_two_sided
from ad_for_edu.stats.power import (
    detectable_agreement,
    detectable_difference,
    two_sided_power,
)


def test_only_the_strategies_in_use_are_registered():
    assert INTERVALS.names() == ("bootstrap", "cluster_bootstrap")
    assert PAIRINGS.names() == ("all_pairs", "both_decide")
    assert PAIRED_TESTS.names() == ("exact_mcnemar",)
    assert CORRECTIONS.names() == ("holm",)
    assert AGREEMENT.names() == ("krippendorff_nominal",)


# ---------------------------------------------------------------- intervals
def observations(per_cluster: dict[str, list[float]]) -> list[Observation]:
    return [Observation(c, v) for c, values in per_cluster.items() for v in values]


@pytest.mark.parametrize(
    ("rule", "expected"),
    [("inclusive", (2, 96)), ("upper", (2, 97)), ("symmetric", (2, 96))],
)
def test_the_three_position_rules_on_a_hundred_values(rule, expected):
    values = list(range(100))
    assert percentile_bounds(values, 0.95, rule) == expected


def test_the_position_rules_differ_where_the_sizes_make_them():
    values = list(range(2000))
    assert percentile_bounds(values, 0.95, "inclusive") == (49, 1949)
    assert percentile_bounds(values, 0.95, "upper") == (50, 1950)
    assert percentile_bounds(values, 0.95, "symmetric") == (50, 1949)


def test_an_unknown_position_rule_is_refused():
    with pytest.raises(ValueError):
        percentile_bounds([1.0, 2.0], 0.95, "nearest")
    with pytest.raises(ValueError):
        ClusterBootstrap(quantile_rule="nearest")


def test_the_cluster_bootstrap_is_reproducible_under_its_seed():
    data = observations({"lec-a": [1, 2, 3], "lec-b": [7, 8], "lec-c": [4]})
    first = ClusterBootstrap(resamples=500, seed=3).interval(data)
    second = ClusterBootstrap(resamples=500, seed=3).interval(data)
    other = ClusterBootstrap(resamples=500, seed=4).interval(data)
    assert first == second
    assert (first.low, first.high) != (other.low, other.high)
    assert first.estimate == pytest.approx(statistics.fmean([1, 2, 3, 7, 8, 4]))
    assert first.n == 6 and first.n_clusters == 3
    assert first.low <= first.estimate <= first.high


def test_the_cluster_bootstrap_matches_a_reference_implementation():
    per_cluster = {"lec-a": [0.2, 0.4, 0.9], "lec-b": [0.5], "lec-c": [0.7, 0.1], "lec-d": [1.0]}
    clusters = list(per_cluster)
    generator = random.Random(17)
    means = []
    for _ in range(300):
        pool = []
        for _ in clusters:
            pool.extend(per_cluster[clusters[generator.randrange(len(clusters))]])
        means.append(statistics.fmean(pool))
    means.sort()
    expected = (means[int(0.025 * 299)], means[int(0.975 * 299)])
    got = ClusterBootstrap(resamples=300, seed=17, quantile_rule="inclusive").interval(
        observations(per_cluster)
    )
    assert (got.low, got.high) == expected


def test_clustering_widens_the_interval_when_clusters_differ():
    per_cluster = {f"lec-{i}": [float(i)] * 20 for i in range(6)}
    data = observations(per_cluster)
    clustered = ClusterBootstrap(resamples=800, seed=1).interval(data)
    independent = UnitBootstrap(resamples=800, seed=1).interval(data)
    assert (clustered.high - clustered.low) > 2 * (independent.high - independent.low)


def test_one_cluster_gives_an_estimate_without_an_interval():
    result = ClusterBootstrap().interval(observations({"lec-a": [1, 2, 3]}))
    assert result.estimate == 2 and not result.defined and result.n_clusters == 1


def test_no_observations_give_nothing():
    result = ClusterBootstrap().interval([])
    assert result.estimate is None and not result.defined and result.n == 0


def test_a_win_rate_is_the_mean_of_win_indicators_over_decided_moments():
    wins = observations({"lec-a": [1, 1, 0], "lec-b": [0, 1], "lec-c": [1]})
    result = ClusterBootstrap(resamples=400, seed=0).interval(wins)
    assert result.estimate == pytest.approx(4 / 6)
    assert result.covers(0.5) or result.low > 0.5


def test_paired_differences_use_only_shared_units():
    left = {"lec-a#1": 6.0, "lec-a#2": 7.0, "lec-b#1": 5.0}
    right = {"lec-a#1": 5.0, "lec-b#1": 5.5, "lec-c#1": 9.0}
    differences = paired_difference(left, right, lambda unit: unit.split("#")[0])
    assert [(o.cluster, o.value) for o in differences] == [("lec-a", 1.0), ("lec-b", -0.5)]


def test_the_interval_states_how_it_was_made():
    method = ClusterBootstrap(resamples=5000, seed=0).interval(
        observations({"a": [1], "b": [2]})
    ).method
    assert "5000" in method and "seed 0" in method and "inclusive" in method


# ---------------------------------------------------------------- paired tests
FIRST = {"p1": True, "p2": True, "p3": None, "p4": False, "p5": True, "p6": None}
SECOND = {"p1": False, "p2": True, "p3": True, "p4": None, "p5": False, "p6": None}


def test_both_decide_keeps_pairs_both_scorers_decided():
    assert BothDecide().select(FIRST, SECOND) == ["p1", "p2", "p5"]


def test_all_pairs_keeps_everything_and_counts_no_decision_as_not_right():
    result = ExactMcNemar().test(FIRST, SECOND, AllPairs())
    assert result.n == 6
    assert result.discordant == (2, 1)  # first alone right on p1, p5; second alone on p3
    assert result.accuracy_first == 0.5 and result.accuracy_second == pytest.approx(0.3333)


def test_the_pairing_rule_changes_the_result_and_is_recorded():
    both = ExactMcNemar().test(FIRST, SECOND, BothDecide())
    every = ExactMcNemar().test(FIRST, SECOND, AllPairs())
    assert both.n == 3 and both.discordant == (2, 0)
    assert both.pairing == "both_decide" and every.pairing == "all_pairs"
    assert both.discordant != every.discordant


def test_the_test_is_symmetric_in_its_two_scorers():
    forward = ExactMcNemar().test(FIRST, SECOND, AllPairs())
    backward = ExactMcNemar().test(SECOND, FIRST, AllPairs())
    assert forward.p_value == backward.p_value
    assert forward.discordant == backward.discordant[::-1]


@pytest.mark.parametrize(
    ("counts", "p_value"),
    [((0, 0), 1.0), ((5, 5), 1.0), ((10, 0), 2 * 0.5**10), ((58, 8), None), ((66, 59), None)],
)
def test_exact_p_values(counts, p_value):
    got = exact_two_sided(*counts)
    if p_value is not None:
        assert got == pytest.approx(p_value)
    elif counts == (58, 8):
        assert got < 1e-4
    else:
        assert got == pytest.approx(0.59, abs=0.01)


def test_coverage_and_accuracy_are_reported_together():
    summary = coverage_and_accuracy(FIRST)
    assert summary["n_pairs"] == 6 and summary["n_decided"] == 4 and summary["n_right"] == 3
    assert summary["coverage"] == pytest.approx(4 / 6)
    assert summary["accuracy"] == 0.75


def test_a_scorer_that_never_decides_has_no_accuracy():
    summary = coverage_and_accuracy({"p1": None, "p2": None})
    assert summary["coverage"] == 0 and summary["accuracy"] is None


# ---------------------------------------------------------------- correction
def test_holm_multiplies_by_the_number_of_tests_still_in_play():
    adjusted = Holm().adjust({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted["a"].adjusted == pytest.approx(0.03)
    assert adjusted["c"].adjusted == pytest.approx(0.06)
    assert adjusted["b"].adjusted == pytest.approx(0.06)  # never below the one before it
    assert adjusted["a"].reject and not adjusted["c"].reject and not adjusted["b"].reject


def test_holm_stops_rejecting_at_the_first_failure():
    adjusted = Holm().adjust({"a": 0.001, "b": 0.03, "c": 0.04})
    assert adjusted["a"].reject
    assert adjusted["b"].adjusted == pytest.approx(0.06) and not adjusted["b"].reject
    # 0.04 alone would pass, but the test before it failed, so this one is not rejected.
    assert adjusted["c"].adjusted == pytest.approx(0.06) and not adjusted["c"].reject


def test_holm_never_exceeds_one_and_handles_one_test():
    assert Holm().adjust({"a": 0.9, "b": 0.8})["b"].adjusted == 1.0
    assert Holm().adjust({"only": 0.03})["only"].adjusted == pytest.approx(0.03)


# ---------------------------------------------------------------- agreement
def test_perfect_agreement_is_one():
    assert KrippendorffNominal().value([["a", "a"], ["b", "b"], ["tie", "tie"]]) == 1.0


def test_a_textbook_example():
    # Two raters, ten units, labels 0/1; they differ on three units.
    first = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
    second = [0, 0, 0, 0, 1, 1, 1, 1, 1, 0]
    units = [list(pair) for pair in zip(first, second, strict=True)]
    # n = 20 values; coincidences: o_01 + o_10 = 6; totals n_0 = 11, n_1 = 9.
    expected = 1 - (6 / 20) / ((2 * 11 * 9) / (20 * 19))
    assert KrippendorffNominal().value(units) == pytest.approx(expected)


def test_systematic_disagreement_is_negative():
    units = [["a", "b"], ["b", "a"], ["a", "b"], ["b", "a"]]
    assert KrippendorffNominal().value(units) < 0


def test_missing_labels_are_skipped_not_counted_as_disagreement():
    full = [["a", "a", "a"], ["b", "b", "b"], ["a", "b", "a"]]
    with_gaps = full + [["a", None, None], [None, None, None]]
    coefficient = KrippendorffNominal()
    assert coefficient.value(with_gaps) == pytest.approx(coefficient.value(full))


def test_units_are_built_from_items_every_rater_labelled():
    labels = {"R1": {"p1": "a", "p2": "tie"}, "R2": {"p1": "a", "p2": "b", "p3": "a"}}
    assert complete_units(labels, ["p1", "p2", "p3"]) == [["a", "a"], ["tie", "b"]]


def test_unanimous_counts_full_agreement_on_a_side():
    units = [["a", "a", "a"], ["tie", "tie", "tie"], ["a", "b", "a"], ["b", "b", "b"]]
    assert unanimous(units) == 3
    assert unanimous(units, ignoring="tie") == 2


# ---------------------------------------------------------------- power
def test_more_labels_detect_a_smaller_effect():
    small, large = detectable_agreement(50), detectable_agreement(300)
    assert 0.5 < large < small < 1.0


def test_too_few_labels_detect_nothing():
    assert detectable_agreement(3) is None
    assert detectable_agreement(0) is None


def test_the_detected_agreement_reaches_the_power_it_promises():
    from scipy.stats import binom

    n = 180
    share = detectable_agreement(n)
    critical = next(k for k in range(n + 1) if binom.sf(k - 1, n, 0.5) <= 0.05)
    assert binom.sf(critical - 1, n, share) >= 0.80
    assert binom.sf(critical - 1, n, share - 0.01) < 0.80


def test_power_is_the_size_of_the_test_under_the_null():
    assert two_sided_power(40, 0.5) <= 0.05


def test_the_detectable_difference_scales_with_the_share_of_discordant_pairs():
    found = detectable_difference(82, 246)
    assert 0.5 < found.win_share < 1.0
    assert found.percentage_points == pytest.approx(
        100 * (2 * found.win_share - 1) * 82 / 246
    )
    assert two_sided_power(82, found.win_share) >= 0.80


def test_without_discordant_pairs_nothing_is_detectable():
    assert detectable_difference(0, 100).percentage_points is None


# ---------------------------------------------------------------- correlation
def test_rank_correlation_of_a_monotone_relation_is_one():
    assert spearman([1, 2, 3, 4], [10, 20, 40, 80]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)


def test_the_correlation_interval_is_reproducible_and_brackets_the_estimate():
    x = [0.35, 0.33, 0.32, 0.30, 0.29, 0.27, 0.31, 0.34]
    y = [0.75, 0.80, 0.84, 0.85, 0.86, 0.79, 0.78, 0.74]
    first = spearman_with_interval(x, y, random.Random(0), resamples=300)
    second = spearman_with_interval(x, y, random.Random(0), resamples=300)
    assert first == second
    assert first.low <= first.estimate <= first.high
    assert first.n == 8


def test_a_correlation_needs_paired_values():
    with pytest.raises(ValueError):
        spearman_with_interval([1, 2, 3], [1, 2], random.Random(0))


def test_rank_positions():
    scores = {"system-a": 0.85, "system-b": 0.80, "readout": 0.53}
    assert rank_of(scores, "readout") == 3
    assert rank_of(scores, "readout", highest_first=False) == 1


# --------------------------------------------------- the score interval for a share
def test_the_score_interval_matches_its_published_values():
    # Seven of ten, at 95%: the score interval is [0.3968, 0.8922].
    found = wilson(7, 10)
    assert found.estimate == pytest.approx(0.7)
    assert found.low == pytest.approx(0.3968, abs=5e-5)
    assert found.high == pytest.approx(0.8922, abs=5e-5)
    assert found.method == "wilson"


def test_the_score_interval_stays_inside_the_bounds_at_the_extremes():
    none_of_them = wilson(0, 12)
    assert none_of_them.low == 0.0 and 0 < none_of_them.high < 0.3
    all_of_them = wilson(12, 12)
    assert all_of_them.high == 1.0 and 0.7 < all_of_them.low < 1.0


def test_the_score_interval_narrows_as_the_count_grows():
    narrow = wilson(400, 800)
    wide = wilson(5, 10)
    assert (narrow.high - narrow.low) < (wide.high - wide.low)


def test_a_share_of_nothing_has_no_interval():
    empty = wilson(0, 0)
    assert empty.estimate is None and not empty.defined and empty.n == 0


def test_more_successes_than_trials_is_refused():
    with pytest.raises(ValueError, match="is not a share"):
        wilson(11, 10)
    with pytest.raises(ValueError, match="is not a share"):
        wilson(-1, 10)
