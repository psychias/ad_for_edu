"""The tables: pair decisions, the systems table, head to head, comparisons, rater agreement."""

from __future__ import annotations

import pytest

from ad_for_edu.agreement import Cell, build_report, cell_for, verdicts_for
from ad_for_edu.agreement import render as render_agreement
from ad_for_edu.core.errors import ContractError
from ad_for_edu.evaluation import (
    PAIR_DECISION_RULES,
    Cascade,
    PairScores,
    SingleScorer,
    SystemRow,
    agreement_between,
    compare,
    compare_all,
    correlations,
    draw_moments,
    measure,
    outcomes,
    over_seeds,
    rank_stability,
    ranks_of,
    render,
    render_correlations,
    score_system,
    verdicts_against,
    win_rate,
)
from ad_for_edu.inference.predictions import Prediction
from ad_for_edu.metrics.base import Metric
from ad_for_edu.pairs.ordering import ORDER_COMBINATIONS, Outcome
from ad_for_edu.stats import ExactMcNemar, Holm, KrippendorffNominal
from ad_for_edu.stats.paired import AllPairs, BothDecide
from ad_for_edu.systems import EvalMoment


# ---------------------------------------------------------------- pair decisions
def test_a_scorer_decides_a_pair_when_the_sides_differ():
    assert PairScores(0.8, 0.6).decided and PairScores(0.8, 0.6).preferred == "a"
    assert PairScores(0.6, 0.8).preferred == "b"
    assert PairScores(0.7, 0.7).tied and not PairScores(0.7, 0.7).decided
    assert not PairScores(None, 0.5).decided and not PairScores(None, 0.5).tied


def test_the_two_decision_rules():
    assert PAIR_DECISION_RULES.names() == ("cascade", "single_scorer")


def test_one_scorer_decides_or_the_pair_is_undecided():
    rule = SingleScorer("rubric")
    assert rule.decide("p1", {"rubric": PairScores(0.8, 0.6)}).preferred == "a"
    assert rule.decide("p1", {"rubric": PairScores(0.7, 0.7)}).preferred is None
    assert rule.decide("p1", {"chrf": PairScores(0.8, 0.6)}).preferred is None


def test_a_cascade_takes_the_first_scorer_that_decides():
    cascade = Cascade(["rubric", "subscores", "chrf"])
    decided = cascade.decide(
        "p1",
        {
            "rubric": PairScores(0.7, 0.7),
            "subscores": PairScores(0.9, 0.4),
            "chrf": PairScores(0.2, 0.8),
        },
    )
    assert decided.preferred == "a" and decided.by == "subscores"


def test_a_cascade_that_decides_nothing_says_so():
    cascade = Cascade(["rubric", "chrf"])
    assert cascade.decide("p1", {"rubric": PairScores(0.7, 0.7)}).preferred is None


def test_a_cascade_needs_a_scorer():
    with pytest.raises(ValueError):
        Cascade([])


def test_coverage_and_accuracy_are_reported_together():
    decisions = [
        SingleScorer("m").decide(f"p{index}", {"m": scores})
        for index, scores in enumerate(
            [PairScores(0.8, 0.6), PairScores(0.4, 0.9), PairScores(0.5, 0.5), PairScores(0.9, 0.1)]
        )
    ]
    correct = {"p0": "a", "p1": "a", "p2": "a", "p3": "a"}
    found = measure(decisions, correct)
    assert found.pairs == 4 and found.decided == 3 and found.right == 2 and found.ties == 1
    assert found.coverage == 0.75
    assert found.accuracy == pytest.approx(2 / 3)
    # Counting a tie against the scorer is a different number, and both are reported.
    assert found.accuracy_with_ties_as_failures == 0.5


def test_a_scorer_that_decides_few_pairs_does_not_look_better_for_it():
    picky = [
        SingleScorer("m").decide("p0", {"m": PairScores(0.9, 0.1)}),
        SingleScorer("m").decide("p1", {"m": PairScores(0.5, 0.5)}),
    ]
    thorough = [
        SingleScorer("m").decide("p0", {"m": PairScores(0.9, 0.1)}),
        SingleScorer("m").decide("p1", {"m": PairScores(0.1, 0.9)}),
    ]
    correct = {"p0": "a", "p1": "a"}
    assert measure(picky, correct).accuracy == 1.0
    assert measure(thorough, correct).accuracy == 0.5
    # Coverage is what tells them apart.
    assert measure(picky, correct).coverage == 0.5
    assert measure(thorough, correct).coverage == 1.0


def test_which_stage_of_a_cascade_decided_each_pair_is_recorded():
    cascade = Cascade(["rubric", "chrf"])
    decisions = [
        cascade.decide("p0", {"rubric": PairScores(0.9, 0.1), "chrf": PairScores(0.1, 0.9)}),
        cascade.decide("p1", {"rubric": PairScores(0.5, 0.5), "chrf": PairScores(0.9, 0.1)}),
    ]
    found = measure(decisions, {"p0": "a", "p1": "a"})
    assert found.by_stage == {"rubric": 1, "chrf": 1}


def test_a_pair_whose_right_side_is_unknown_is_left_out():
    decisions = [SingleScorer("m").decide("p0", {"m": PairScores(0.9, 0.1)})]
    assert measure(decisions, {}).pairs == 0


def test_the_verdicts_of_a_rule_against_the_right_sides():
    decisions = [
        SingleScorer("m").decide("p0", {"m": PairScores(0.9, 0.1)}),
        SingleScorer("m").decide("p1", {"m": PairScores(0.1, 0.9)}),
        SingleScorer("m").decide("p2", {"m": PairScores(0.5, 0.5)}),
    ]
    assert verdicts_against(decisions, {"p0": "a", "p1": "a", "p2": "a"}) == {
        "p0": True,
        "p1": False,
        "p2": None,
    }


# ---------------------------------------------------------------- the systems table
class Fixed(Metric):
    """A metric that gives one value, for testing the table."""

    def __init__(self, value, needs_references=False):
        self.value = value
        # Per instance: setting it on the class would make every instance share it.
        self.needs_references = needs_references

    def score(self, items):
        return [self.value if self.scorable(item) else None for item in items]


def moments_for(names) -> dict[str, EvalMoment]:
    return {name: EvalMoment(moment_id=name, type="figure", time=float(index))
            for index, name in enumerate(names)}


def predictions_for(names, text="A description."):
    return [
        Prediction(output_id=name, moment_id=name, emit=True, ad_text=text, forced=True)
        for name in names
    ]


def test_a_row_says_how_many_moments_each_metric_could_score():
    names = ["lec-a#0001", "lec-a#0002"]
    references = {"lec-a#0001": ["a reference"]}
    row = score_system(
        "system-x",
        predictions_for(names),
        {"chrf": Fixed(0.5, needs_references=True), "clip": Fixed(0.7)},
        moments_for(names),
        references,
    )
    assert row.values["chrf"] == 0.5 and row.counts["chrf"] == 1
    assert row.values["clip"] == 0.7 and row.counts["clip"] == 2
    assert row.moments == 2 and row.described == 2


def test_a_moment_nothing_is_known_about_stops_the_scoring():
    with pytest.raises(ContractError):
        score_system(
            "system-x",
            predictions_for(["lec-z#0001"]),
            {"chrf": Fixed(0.5)},
            moments_for(["lec-a#0001"]),
            {},
        )


def test_a_metric_that_returns_the_wrong_number_of_values_is_refused():
    class Short(Metric):
        def score(self, items):
            return [0.5]

    with pytest.raises(ContractError):
        score_system(
            "system-x",
            predictions_for(["lec-a#0001", "lec-a#0002"]),
            {"short": Short()},
            moments_for(["lec-a#0001", "lec-a#0002"]),
            {},
        )


def test_a_row_over_several_seeds_says_how_many_it_has():
    rows = [
        SystemRow("system-x", {"chrf": 0.30}, {"chrf": 10}, 10, 10, seeds=(0,)),
        SystemRow("system-x", {"chrf": 0.34}, {"chrf": 10}, 10, 10, seeds=(1,)),
    ]
    averaged = over_seeds(rows)
    assert averaged.values["chrf"] == pytest.approx(0.32)
    assert averaged.seeds == (0, 1) and averaged.seed_count == 2


def test_rows_carrying_different_metrics_cannot_be_averaged():
    rows = [
        SystemRow("s", {"chrf": 0.3}, {"chrf": 1}, 1, 1, seeds=(0,)),
        SystemRow("s", {"clip": 0.3}, {"clip": 1}, 1, 1, seeds=(1,)),
    ]
    with pytest.raises(ContractError):
        over_seeds(rows)


def test_the_table_prints_the_number_behind_every_value():
    rows = [SystemRow("system-x", {"chrf": 0.344}, {"chrf": 512}, 516, 516, seeds=(0, 1, 2))]
    table = render(rows, ["chrf"], title="Systems")
    assert "0.344 (512)" in table and "| 3 |" in table
    assert "seeds" in table


# ---------------------------------------------------------------- head to head
def test_a_moment_counts_only_when_both_orders_agree():
    combination = ORDER_COMBINATIONS.create("agree_or_undecided")
    settled = outcomes(
        {
            "lec-a#0001": {0: "a", 1: "a"},
            "lec-a#0002": {0: "a", 1: "b"},
            "lec-a#0003": {0: "b", 1: "b"},
            "lec-a#0004": {0: "a"},
        },
        combination,
    )
    assert settled["lec-a#0001"] is Outcome.FIRST
    assert settled["lec-a#0002"] is Outcome.UNDECIDED
    assert settled["lec-a#0003"] is Outcome.SECOND
    assert settled["lec-a#0004"] is Outcome.UNDECIDED


def test_the_win_rate_is_over_the_decided_moments():
    combination = ORDER_COMBINATIONS.create("agree_or_undecided")
    answers = {
        f"lec-{lecture}#{index:04d}": {0: side, 1: side}
        for lecture in "ab"
        for index, side in enumerate(["a", "a", "b"])
    }
    answers["lec-a#0009"] = {0: "a", 1: "b"}
    result = win_rate(
        outcomes(answers, combination),
        first="sft",
        second="dpo",
        judge="judge-x",
        resamples=200,
    )
    assert result.won == 4 and result.lost == 2 and result.undecided == 1
    assert result.win_rate == pytest.approx(4 / 6)
    assert result.interval.n_clusters == 2


def test_the_direction_of_a_win_rate_follows_the_names():
    combination = ORDER_COMBINATIONS.create("agree_or_undecided")
    settled = outcomes({"lec-a#0001": {0: "a", 1: "a"}}, combination)
    first = win_rate(settled, first="sft", second="dpo", judge="j", resamples=50)
    swapped = win_rate(
        settled, first="dpo", second="sft", judge="j", first_is_side="b", resamples=50
    )
    assert first.win_rate == 1.0 and swapped.win_rate == 0.0


def test_the_interval_of_a_win_rate_is_over_lectures():
    combination = ORDER_COMBINATIONS.create("agree_or_undecided")
    answers = {
        f"lec-{lecture}#{index:04d}": {
            0: "a" if lecture == "a" else "b",
            1: "a" if lecture == "a" else "b",
        }
        for lecture in "abcd"
        for index in range(10)
    }
    result = win_rate(
        outcomes(answers, combination), first="x", second="y", judge="j", resamples=500
    )
    assert result.interval.n == 40 and result.interval.n_clusters == 4
    # Whole lectures agree with themselves, so the interval is wide.
    assert result.interval.high - result.interval.low > 0.5


def test_a_draw_of_moments_is_reproducible():
    moments = [f"lec-a#{index:04d}" for index in range(50)]
    first = draw_moments(moments, 10, seed=17)
    assert first == draw_moments(moments, 10, seed=17)
    assert first != draw_moments(moments, 10, seed=18)
    assert len(first) == 10


def test_a_draw_larger_than_what_exists_is_refused():
    with pytest.raises(ContractError):
        draw_moments(["lec-a#0001"], 10, seed=1)


def test_how_far_two_judges_agree():
    combination = ORDER_COMBINATIONS.create("agree_or_undecided")
    first = outcomes({"m1": {0: "a", 1: "a"}, "m2": {0: "b", 1: "b"}}, combination)
    second = outcomes({"m1": {0: "a", 1: "a"}, "m2": {0: "a", 1: "a"}}, combination)
    assert agreement_between(first, second) == {
        "both_decided": 2,
        "agreed": 1,
        "share_agreed": 0.5,
    }


# ---------------------------------------------------------------- comparisons
def test_a_comparison_states_the_smallest_difference_it_could_have_found():
    first = {f"p{index}": index % 3 != 0 for index in range(120)}
    second = {f"p{index}": index % 2 == 0 for index in range(120)}
    found = compare("rubric", "chrf", {"rubric": first, "chrf": second}, ExactMcNemar(), AllPairs())
    assert found.result.n == 120
    assert found.smallest_detectable is not None and found.smallest_detectable > 0
    assert "smallest_detectable_difference_pp" in found.as_dict()


def test_the_rule_for_undecided_pairs_changes_the_comparison():
    first = {"p1": True, "p2": True, "p3": None}
    second = {"p1": False, "p2": None, "p3": None}
    every = compare("a", "b", {"a": first, "b": second}, ExactMcNemar(), AllPairs())
    both = compare("a", "b", {"a": first, "b": second}, ExactMcNemar(), BothDecide())
    assert every.result.n == 3 and both.result.n == 1
    assert every.result.discordant != both.result.discordant
    assert every.as_dict()["pairing"] == "all_pairs"


def test_a_family_of_comparisons_is_corrected_for_its_size():
    verdicts = {
        "chrf": {f"p{index}": index % 2 == 0 for index in range(100)},
        "rubric": {f"p{index}": index % 10 != 0 for index in range(100)},
        "clip": {f"p{index}": index % 3 == 0 for index in range(100)},
    }
    found = compare_all("chrf", verdicts, ExactMcNemar(), AllPairs(), Holm())
    assert set(found) == {"rubric", "clip"}
    for entry in found.values():
        assert entry["p_corrected"] >= entry["p"]
        assert isinstance(entry["distinct"], bool)


def test_two_metrics_that_order_the_systems_alike_correlate():
    values = {
        "chrf": {"s1": 0.35, "s2": 0.33, "s3": 0.30, "s4": 0.28},
        "bert": {"s1": 0.53, "s2": 0.52, "s3": 0.51, "s4": 0.50},
        "compliance": {"s1": 0.75, "s2": 0.80, "s3": 0.84, "s4": 0.86},
    }
    found = correlations(values, resamples=200, seed=0)
    assert found[("bert", "chrf")].estimate == pytest.approx(1.0)
    assert found[("chrf", "compliance")].estimate == pytest.approx(-1.0)
    assert "rho" in render_correlations(found)


def test_the_matrix_of_correlations_is_reproducible():
    values = {
        "a": {f"s{index}": index * 0.1 for index in range(8)},
        "b": {f"s{index}": (index % 4) * 0.1 for index in range(8)},
    }
    first = correlations(values, resamples=100, seed=5)
    again = correlations(values, resamples=100, seed=5)
    assert first[("a", "b")] == again[("a", "b")]


def test_where_each_system_lands_under_a_metric():
    assert ranks_of({"s1": 0.9, "s2": 0.5, "readout": 0.2}) == {"s1": 1, "s2": 2, "readout": 3}


def test_how_the_order_changes_as_a_setting_changes():
    by_setting = {
        0.2: {"s1": 0.90, "s2": 0.80, "readout": 0.60},
        0.4: {"s1": 0.88, "s2": 0.79, "readout": 0.53},
        0.6: {"s1": 0.85, "s2": 0.77, "readout": 0.50},
    }
    found = rank_stability(by_setting, resamples=100, seed=0)
    assert found["systems"] == 3
    assert all(value == pytest.approx(1.0) for value in found["against_first"].values())
    assert all(ranks["readout"] == 3 for ranks in found["ranks"].values())


def test_stability_needs_more_than_one_setting():
    with pytest.raises(ValueError):
        rank_stability({0.4: {"s1": 1.0}})


# ---------------------------------------------------------------- rater agreement
def pairs_of(kinds) -> list[dict]:
    return [
        {"pair_id": f"p{index}", "axis": axis, "stratum": stratum}
        for index, (axis, stratum) in enumerate(kinds)
    ]


def test_a_metric_agrees_with_a_rater_on_that_rater_s_own_choices():
    labels = {"p0": "a", "p1": "b", "p2": "tie"}
    scores = {"p0": PairScores(0.9, 0.1), "p1": PairScores(0.9, 0.1), "p2": PairScores(0.9, 0.1)}
    cell = cell_for(labels, scores)
    assert cell.agreed == 1 and cell.disagreed == 1 and cell.n == 2
    assert cell.agreement == 0.5


def test_a_metric_that_ties_is_counted_not_dropped_silently():
    labels = {"p0": "a", "p1": "a"}
    scores = {"p0": PairScores(0.9, 0.1), "p1": PairScores(0.5, 0.5)}
    cell = cell_for(labels, scores)
    assert cell.n == 1 and cell.metric_ties == 1 and cell.agreement == 1.0


def test_the_verdicts_of_a_metric_against_a_rater():
    labels = {"p0": "a", "p1": "b", "p2": "tie"}
    scores = {"p0": PairScores(0.9, 0.1), "p1": PairScores(0.5, 0.5)}
    assert verdicts_for(labels, scores) == {"p0": True, "p1": None}


def test_the_pairs_a_judge_could_not_order_are_their_own_row():
    pairs = pairs_of([("deixis", "controlled"), (None, "natural_decided"), (None, "natural_flip")])
    labels = {"R1": {"p0": "a", "p1": "a", "p2": "a"}}
    scores = {"metric": {name: PairScores(0.9, 0.1) for name in ("p0", "p1", "p2")}}
    report = build_report(pairs, labels, scores, coefficient=KrippendorffNominal())
    assert report.overall["R1"]["metric"].n == 2
    assert report.order_sensitive["R1"]["metric"].n == 1
    assert set(report.by_group) == {"deixis", "natural"}


def test_the_report_counts_the_ties_of_each_rater():
    pairs = pairs_of([("style", "controlled"), ("style", "controlled")])
    labels = {"R1": {"p0": "a", "p1": "tie"}, "R2": {"p0": "a", "p1": "a"}}
    scores = {"metric": {"p0": PairScores(0.9, 0.1), "p1": PairScores(0.9, 0.1)}}
    report = build_report(pairs, labels, scores, coefficient=KrippendorffNominal())
    assert report.agreement["ties_per_rater"] == {"R1": 1, "R2": 0}
    assert report.agreement["units"] == 2
    assert report.agreement["unanimous_on_a_side"] == 1


def test_the_report_names_the_raters_by_what_it_calls_them():
    pairs = pairs_of([("style", "controlled")])
    labels = {"R1": {"p0": "a"}, "R2": {"p0": "b"}}
    scores = {"metric": {"p0": PairScores(0.9, 0.1)}}
    report = build_report(pairs, labels, scores, coefficient=KrippendorffNominal())
    assert report.raters == ("R1", "R2")
    printed = render_agreement(report, title="Agreement")
    assert "R1" in printed and "R2" in printed


def test_the_printed_report_carries_the_number_behind_every_cell():
    pairs = pairs_of([("style", "controlled"), ("style", "controlled")])
    labels = {"R1": {"p0": "a", "p1": "a"}}
    scores = {"metric": {"p0": PairScores(0.9, 0.1), "p1": PairScores(0.1, 0.9)}}
    report = build_report(pairs, labels, scores, coefficient=KrippendorffNominal())
    printed = render_agreement(report, title="Agreement")
    assert "50.0 (2)" in printed
    assert "presentation orders" in printed


def test_a_cell_with_no_choices_has_no_agreement():
    assert Cell(0, 0, 3).agreement is None and Cell(0, 0, 3).n == 0
