"""The metrics: what each needs, the best value over references, the two axes."""

from __future__ import annotations

import pytest

from ad_for_edu.compliance import NoveltyKnee, build_scorer, load_compliance_settings
from ad_for_edu.compliance.contexts import InMemoryContexts
from ad_for_edu.core.errors import ContractError, MissingSourceError, StrategyConfigError
from ad_for_edu.core.io import write_jsonl
from ad_for_edu.metrics import (
    METRICS,
    SIMILARITY_FLOOR,
    CharacterOverlap,
    ComplianceMetric,
    Metric,
    ScoringItem,
    StoredRating,
    StoredRubric,
    best_over_references,
    mean_of,
    read_answers,
    rescale,
)
from tests.fixtures.factory import reference_rows


def item(moment: str, text: str | None, references=(), image=None, **fields) -> ScoringItem:
    return ScoringItem(moment, text, tuple(references), image, fields)


# ---------------------------------------------------------------- the family
def test_the_metrics_of_the_evaluation():
    assert METRICS.names() == (
        "bertscore",
        "chrf",
        "clipscore",
        "compliance",
        "reference_rating",
        "rubric",
    )


def test_each_metric_declares_what_it_needs():
    assert METRICS.get("chrf").needs_references and not METRICS.get("chrf").needs_image
    assert METRICS.get("clipscore").needs_image and not METRICS.get("clipscore").needs_references
    assert METRICS.get("rubric").reads_answers
    assert not METRICS.get("compliance").needs_references


def test_a_metric_says_which_items_it_can_score():
    metric = CharacterOverlap()
    assert metric.scorable(item("lec-a#0001", "A diagram.", ["A diagram."]))
    assert not metric.scorable(item("lec-a#0001", "A diagram."))
    assert not metric.scorable(item("lec-a#0001", "  ", ["A diagram."]))
    assert not metric.scorable(item("lec-a#0001", None, ["A diagram."]))


# ---------------------------------------------------------------- best over references
def test_a_description_is_scored_against_its_best_reference():
    scored = []

    def pairs(candidates, references):
        scored.extend(zip(candidates, references, strict=True))
        return [0.2, 0.9, 0.5]

    items = [item("lec-a#0001", "candidate", ["first", "second", "third"])]
    assert best_over_references(items, pairs) == [0.9]
    assert scored == [("candidate", "first"), ("candidate", "second"), ("candidate", "third")]


def test_every_pair_of_every_item_is_scored_in_one_call():
    calls = []

    def pairs(candidates, references):
        calls.append(len(candidates))
        return [0.5] * len(candidates)

    items = [
        item("lec-a#0001", "one", ["a", "b"]),
        item("lec-a#0002", "two", ["c"]),
    ]
    assert best_over_references(items, pairs) == [0.5, 0.5]
    assert calls == [3]


def test_an_item_without_a_reference_or_without_a_description_scores_nothing():
    def pairs(candidates, references):
        return [0.5] * len(candidates)

    items = [
        item("lec-a#0001", "one", []),
        item("lec-a#0002", None, ["a"]),
        item("lec-a#0003", "three", ["c"]),
    ]
    assert best_over_references(items, pairs) == [None, None, 0.5]


def test_nothing_to_score_calls_nothing():
    def pairs(candidates, references):
        raise AssertionError("must not be called")

    assert best_over_references([item("lec-a#0001", None)], pairs) == [None]


def test_the_mean_is_over_the_values_that_exist():
    assert mean_of([1.0, None, 0.5]) == 0.75
    assert mean_of([None, None]) is None
    assert mean_of([]) is None


# ---------------------------------------------------------------- overlap
def test_character_overlap_is_one_for_the_same_text_and_falls_for_another():
    metric = CharacterOverlap()
    same = metric.one("Two curves cross.", "Two curves cross.")
    other = metric.one("Two curves cross.", "A table of prices.")
    assert same == pytest.approx(1.0)
    assert 0.0 <= other < 0.5


def test_character_overlap_takes_the_best_reference():
    metric = CharacterOverlap()
    scored = metric.score(
        [item("lec-a#0001", "Two curves cross.", ["A table.", "Two curves cross."])]
    )
    assert scored == [pytest.approx(1.0)]


def test_the_rescaled_axis_puts_the_floor_at_zero():
    assert rescale(SIMILARITY_FLOOR) == pytest.approx(0.0)
    assert rescale(1.0) == pytest.approx(1.0)
    assert rescale(0.885) == pytest.approx((0.885 - SIMILARITY_FLOOR) / (1 - SIMILARITY_FLOOR))


def test_a_value_below_the_floor_stays_negative():
    assert rescale(0.70) < 0


def test_the_rescaling_cannot_change_an_order():
    raw = [0.885, 0.874, 0.879, 0.762]
    assert sorted(range(4), key=lambda i: raw[i]) == sorted(
        range(4), key=lambda i: rescale(raw[i])
    )


# ---------------------------------------------------------------- stored answers
def test_stored_answers_are_read_by_moment(tmp_path):
    path = tmp_path / "answers.jsonl"
    write_jsonl(
        path,
        [
            {"moment_id": "lec-a#0001", "score": 4},
            {"moment_id": "lec-a#0002", "error": "timeout"},
            {"moment_id": "lec-a#0003", "score": None},
            {"moment_id": "lec-a#0004", "score": 2},
        ],
    )
    assert read_answers(path) == {"lec-a#0001": 4.0, "lec-a#0004": 2.0}


def test_a_moment_answered_twice_is_an_error(tmp_path):
    path = tmp_path / "answers.jsonl"
    write_jsonl(path, [{"moment_id": "lec-a#0001", "score": 4}] * 2)
    with pytest.raises(ContractError):
        read_answers(path)


def test_answers_can_be_selected_by_system(tmp_path):
    path = tmp_path / "answers.jsonl"
    write_jsonl(
        path,
        [
            {"moment_id": "lec-a#0001", "score": 4, "system": "one"},
            {"moment_id": "lec-a#0001", "score": 2, "system": "two"},
        ],
    )
    assert read_answers(path, system="two") == {"lec-a#0001": 2.0}


def test_a_store_that_is_not_there_is_an_error(tmp_path):
    with pytest.raises(MissingSourceError):
        read_answers(tmp_path / "absent.jsonl")


def test_a_rating_is_put_on_the_unit_interval():
    metric = StoredRating({"lec-a#0001": 5.0, "lec-a#0002": 1.0, "lec-a#0003": 3.0})
    scored = metric.score(
        [item(f"lec-a#{index:04d}", "x", ["y"]) for index in (1, 2, 3, 4)]
    )
    assert scored == [1.0, 0.0, 0.5, None]


def test_a_rubric_grade_is_reported_on_its_own_scale():
    metric = StoredRubric({"lec-a#0001": 6.32})
    assert metric.score([item("lec-a#0001", "x")]) == [6.32]


def test_a_scale_that_does_not_rise_is_refused():
    with pytest.raises(ValueError):
        StoredRating({}, scale=(5.0, 5.0))


def test_a_metric_that_reads_answers_never_calls_anything():
    metric = StoredRubric({"lec-a#0001": 6.0})
    assert metric.reads_answers
    assert metric.score([item("lec-a#0002", "x")]) == [None]


# ---------------------------------------------------------------- compliance as a metric
@pytest.fixture
def contexts(shipped_configs):
    source = InMemoryContexts(reference_rows())
    return source.contexts(), source.times()


def test_compliance_is_scored_through_the_shipped_scorer(shipped_configs, contexts):
    moments, times = contexts
    settings = load_compliance_settings(shipped_configs / "compliance.yaml")
    metric = ComplianceMetric(build_scorer("mechanical", settings), moments)
    assert metric.mode == "mechanical"
    scored = metric.score([item("lec-a#0001", "An outline lists four topics.", time=12.0)])
    assert scored[0] is not None and 0.0 <= scored[0] <= 1.0


def test_a_system_that_repeats_itself_is_scaled_down(shipped_configs, contexts):
    moments, times = contexts
    settings = load_compliance_settings(shipped_configs / "compliance.yaml")
    scorer = build_scorer("mechanical", settings)
    repeated = [
        item(moment, "Cell structure", time=times[moment])
        for moment in ("lec-a#0001", "lec-a#0002", "lec-a#0003")
    ]
    plain = ComplianceMetric(scorer, moments).score(repeated)
    scaled = ComplianceMetric(scorer, moments, NoveltyKnee(0.4)).score(repeated)
    assert scaled[0] == plain[0]
    assert scaled[1] == 0.0 and scaled[2] == 0.0
    assert mean_of(scaled) < mean_of(plain)


def test_an_empty_description_scores_nothing(shipped_configs, contexts):
    moments, _times = contexts
    settings = load_compliance_settings(shipped_configs / "compliance.yaml")
    metric = ComplianceMetric(build_scorer("mechanical", settings), moments)
    assert metric.score([item("lec-a#0001", None, time=1.0)]) == [None]


def test_a_moment_without_a_context_stops_the_scoring(shipped_configs, contexts):
    moments, _times = contexts
    settings = load_compliance_settings(shipped_configs / "compliance.yaml")
    metric = ComplianceMetric(build_scorer("mechanical", settings), moments)
    with pytest.raises(ContractError):
        metric.score([item("lec-z#0001", "Anything.", time=1.0)])


# ---------------------------------------------------------------- the model-backed metrics
def test_the_embedded_similarity_reports_both_axes_from_one_call():
    from ad_for_edu.metrics.overlap import EmbeddedSimilarity

    raw = EmbeddedSimilarity()
    scaled = EmbeddedSimilarity(rescaled=True)
    raw._scorer = scaled._scorer = _FakeEmbedder([0.885, 0.762])
    items = [item("lec-a#0001", "one", ["a"]), item("lec-a#0002", "two", ["b"])]
    assert raw.score(items) == [pytest.approx(0.885), pytest.approx(0.762)]
    raw._scorer.reset()
    assert scaled.score(items) == [
        pytest.approx(rescale(0.885)),
        pytest.approx(rescale(0.762)),
    ]


class _FakeEmbedder:
    def __init__(self, values):
        self.values = list(values)
        self.given = list(values)
        self.calls = 0

    def reset(self):
        self.values = list(self.given)

    def score(self, candidates, references, batch_size=32):
        self.calls += 1
        taken = [self.values.pop(0) for _ in candidates]
        return taken, taken, taken


def test_alignment_reports_nothing_when_the_image_is_not_there(tmp_path):
    from ad_for_edu.metrics.alignment import ImageTextAlignment

    metric = ImageTextAlignment()
    assert metric.score([item("lec-a#0001", "x", image=tmp_path / "absent.jpg")]) == [None]
    assert metric.score([item("lec-a#0001", "x")]) == [None]


def test_alignment_scales_the_similarity_and_treats_a_negative_one_as_none(tmp_path):
    from ad_for_edu.metrics.alignment import SCALE, ImageTextAlignment

    picture = tmp_path / "000010.jpg"
    picture.write_bytes(b"not really an image")
    metric = ImageTextAlignment()
    metric.one = lambda text, image: SCALE * max(-0.1, 0.0)
    assert metric.score([item("lec-a#0001", "x", image=picture)]) == [0.0]


def test_a_metric_outside_the_family_cannot_be_registered():
    with pytest.raises(StrategyConfigError, match="does not derive from"):

        @METRICS.register("not_a_metric")
        class Stranger:
            pass


def test_every_registered_metric_implements_the_interface():
    for name in METRICS.names():
        assert issubclass(METRICS.get(name), Metric)
