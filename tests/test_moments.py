"""Detection, stills, windows, rungs, context and classification."""

from __future__ import annotations

import pytest

from ad_for_edu.core.errors import ContractError, SettingsError
from ad_for_edu.core.text import NO_SLIDE_TEXT, NO_SPEECH, NOTHING_DESCRIBED_YET
from ad_for_edu.data.media import Gap, SlideTextFrame
from ad_for_edu.data.schema import REJECT, CandidateEvent
from ad_for_edu.moments import (
    RUNG_POLICIES,
    RUNG_WORDS,
    STILL_POLICIES,
    TRANSCRIPT_WINDOWS,
    ContextBuilder,
    DetectionSettings,
    MomentDetector,
    batch_by_lecture,
    batch_fault,
    batch_request_text,
    build_detector,
    deliverable_words,
    fits,
    load_classification_settings,
    load_detection_settings,
    offered_types,
    parse_objects,
    prompt_for,
    read_answer,
    to_moment,
    vocabulary_fault,
    words_in,
)
from ad_for_edu.moments.classification import Classification, types_block
from ad_for_edu.moments.context import pause_after, slide_text_at
from ad_for_edu.moments.detection import (
    FRAME_CHANNELS,
    GAP_CHANNELS,
    FramePair,
    LectureSignals,
    next_gap,
    usable_gaps,
)
from ad_for_edu.moments.detection.channels import content_words
from ad_for_edu.moments.rungs import RungBudgets, UniformBudget
from ad_for_edu.moments.stills import EventWindow, KeyframeWindow
from ad_for_edu.moments.windows import Fixed, Span, Symmetric, UntilNextEvent


def candidate(**fields) -> CandidateEvent:
    values = {
        "id": "lec-a#0001",
        "lecture": "lec-a",
        "t": 30.0,
        "channel": "slide_turnover",
        "visual": True,
        "gap_after": 1.5,
        "detail": {},
    }
    values.update(fields)
    return CandidateEvent(**values)


def pair(before: str, after: str, apart: float = 2.0) -> FramePair:
    return FramePair(
        earlier=SlideTextFrame(10.0, before), later=SlideTextFrame(10.0 + apart, after)
    )


# ---------------------------------------------------------------- the channels
def test_the_channels_of_the_detector():
    assert FRAME_CHANNELS.names() == ("added_words", "pixel_motion", "slide_turnover")
    assert GAP_CHANNELS.names() == ("speech_gap",)


def test_content_words_ignore_case_short_tokens_and_common_words():
    assert content_words("The Action Potential of a cell") == {"action", "potential", "cell"}
    assert content_words(None) == set()


def test_a_turnover_of_the_slide_text_is_a_new_slide():
    channel = FRAME_CHANNELS.create("slide_turnover", overlap_below=0.55)
    hit = channel.evaluate(
        pair("Action potential phases", "Sodium channels open"), LectureSignals()
    )
    assert hit is not None and hit.channel == "slide_turnover"
    assert hit.detail["text_overlap"] < 0.55


def test_the_same_slide_is_not_a_turnover():
    channel = FRAME_CHANNELS.create("slide_turnover")
    assert channel.evaluate(pair("Action potential", "Action potential"), LectureSignals()) is None


def test_two_empty_readings_are_not_a_turnover():
    channel = FRAME_CHANNELS.create("slide_turnover")
    assert channel.evaluate(pair("", ""), LectureSignals()) is None


def test_words_appearing_on_the_slide_are_an_addition():
    channel = FRAME_CHANNELS.create("added_words", at_least=3)
    added = channel.evaluate(pair("Phases", "Phases sodium potassium channel"), LectureSignals())
    assert added is not None and added.detail["added_words"] == 3


def test_one_or_two_added_words_are_not_an_addition():
    channel = FRAME_CHANNELS.create("added_words", at_least=3)
    assert channel.evaluate(pair("Phases", "Phases sodium"), LectureSignals()) is None


def test_movement_is_measured_between_the_two_keyframes(tmp_path):
    from PIL import Image

    channel = FRAME_CHANNELS.create("pixel_motion", difference_at_least=0.06)
    quiet, loud = tmp_path / "a.jpg", tmp_path / "b.jpg"
    Image.new("L", (64, 64), 40).save(quiet)
    Image.new("L", (64, 64), 200).save(loud)
    still = FramePair(SlideTextFrame(1.0, "x"), SlideTextFrame(3.0, "x"), quiet, quiet)
    moved = FramePair(SlideTextFrame(1.0, "x"), SlideTextFrame(3.0, "x"), quiet, loud)
    assert channel.evaluate(still, LectureSignals()) is None
    hit = channel.evaluate(moved, LectureSignals())
    assert hit is not None and hit.detail["picture_difference"] > 0.06


def test_keyframes_a_whole_interval_apart_are_left_alone(tmp_path):
    from PIL import Image

    channel = FRAME_CHANNELS.create("pixel_motion")
    quiet, loud = tmp_path / "a.jpg", tmp_path / "b.jpg"
    Image.new("L", (64, 64), 40).save(quiet)
    Image.new("L", (64, 64), 200).save(loud)
    far = FramePair(SlideTextFrame(0.0, "x"), SlideTextFrame(30.0, "x"), quiet, loud)
    assert channel.evaluate(far, LectureSignals(heartbeat_seconds=30.0)) is None


def test_movement_compares_against_the_keyframe_this_channel_looked_at_last(tmp_path):
    from PIL import Image

    channel = FRAME_CHANNELS.create("pixel_motion", difference_at_least=0.06)
    first, second = tmp_path / "a.jpg", tmp_path / "b.jpg"
    Image.new("L", (64, 64), 40).save(first)
    Image.new("L", (64, 64), 200).save(second)
    signals = LectureSignals()
    # The first look records the later keyframe as what the next comparison starts from.
    channel.evaluate(
        FramePair(SlideTextFrame(1.0, "x"), SlideTextFrame(3.0, "x"), first, second),
        signals,
    )
    assert signals.last_signature is not None
    # The next pair is between two copies of the same picture, but the recorded one differs.
    hit = channel.evaluate(
        FramePair(SlideTextFrame(3.0, "x"), SlideTextFrame(5.0, "x"), first, first), signals
    )
    assert hit is not None


def test_the_channel_forgets_its_keyframe_when_it_declines_to_look(tmp_path):
    from PIL import Image

    channel = FRAME_CHANNELS.create("pixel_motion", difference_at_least=0.06)
    first, second = tmp_path / "a.jpg", tmp_path / "b.jpg"
    Image.new("L", (64, 64), 40).save(first)
    Image.new("L", (64, 64), 200).save(second)
    signals = LectureSignals(heartbeat_seconds=30.0)
    channel.evaluate(
        FramePair(SlideTextFrame(1.0, "x"), SlideTextFrame(3.0, "x"), first, second),
        signals,
    )
    assert signals.last_signature is not None
    # A pair a whole interval apart is not looked at, and what was recorded is dropped.
    channel.evaluate(
        FramePair(SlideTextFrame(3.0, "x"), SlideTextFrame(40.0, "x"), first, second), signals
    )
    assert signals.last_signature is None
    # A pair with no keyframe on disk drops it too.
    channel.evaluate(
        FramePair(SlideTextFrame(1.0, "x"), SlideTextFrame(3.0, "x"), first, second),
        signals,
    )
    channel.evaluate(
        FramePair(SlideTextFrame(3.0, "x"), SlideTextFrame(5.0, "x"), None, None),
        signals,
    )
    assert signals.last_signature is None


def test_the_gap_channel_proposes_the_stretches_nobody_claimed():
    channel = GAP_CHANNELS.create("speech_gap")
    gaps = [Gap(2.0, 4.0), Gap(10.0, 12.0), Gap(20.0, 23.0)]
    proposed = channel.propose(gaps, claimed={10.0})
    assert [hit.detail["t"] for hit in proposed] == [2.0, 20.0]
    assert proposed[1].detail["gap_length"] == 3.0


# ---------------------------------------------------------------- the detector
class FakeLecture:
    def __init__(self, readings, gaps, keyframes=()):
        self.lecture_id = "lec-a"
        self.slide_text = tuple(readings)
        self.gaps = tuple(gaps)
        self.keyframes = tuple(keyframes)


def readings(*entries) -> list[SlideTextFrame]:
    return [SlideTextFrame(time, text) for time, text in entries]


#: Eight words on the slide, then the same eight with three more.
EIGHT = "alpha beta gamma delta epsilon zeta eta theta"
ELEVEN = EIGHT + " iota kappa lambda"


def three_words_added() -> list[SlideTextFrame]:
    return readings((10.0, EIGHT), (12.0, ELEVEN))


def test_only_the_shortest_usable_stretches_count():
    gaps = [Gap(1.0, 1.4), Gap(5.0, 8.0), Gap(2.0, 4.0)]
    assert [(gap.start, gap.end) for gap in usable_gaps(gaps, 1.5)] == [(2.0, 4.0), (5.0, 8.0)]


def test_the_next_stretch_after_an_event_is_taken():
    gaps = [Gap(2.0, 4.0), Gap(40.0, 42.0)]
    assert next_gap(gaps, 1.9, 12.0).start == 2.0
    assert next_gap(gaps, 20.0, 12.0) is None
    assert next_gap(gaps, 30.0, 12.0).start == 40.0


def test_the_first_channel_that_matches_names_the_candidate():
    detector = MomentDetector(
        [FRAME_CHANNELS.create("slide_turnover"), FRAME_CHANNELS.create("added_words")],
        GAP_CHANNELS.create("speech_gap"),
    )
    lecture = FakeLecture(
        readings((10.0, "Phases of the action potential"), (12.0, "Sodium channels open wide")),
        [Gap(12.5, 15.0)],
    )
    found = detector.detect(lecture)
    visual = [event for event in found if event.visual]
    assert len(visual) == 1 and visual[0].channel == "slide_turnover"


def test_a_visual_candidate_carries_the_stretch_it_can_be_spoken_into():
    detector = build_detector_without_motion()
    lecture = FakeLecture(
        three_words_added(),
        [Gap(13.0, 16.0)],
    )
    visual = [event for event in detector.detect(lecture) if event.visual][0]
    assert visual.gap_after == 3.0 and visual.detail["gap_start"] == 13.0


def test_a_visual_candidate_the_lecturer_talks_through_is_kept_and_marked():
    detector = build_detector_without_motion()
    lecture = FakeLecture(
        three_words_added(), [Gap(90.0, 95.0)]
    )
    visual = [event for event in detector.detect(lecture) if event.visual][0]
    assert visual.detail["gap_start"] is None and visual.gap_after == 0.0


def test_a_stretch_a_visual_event_claimed_is_not_proposed_again():
    detector = build_detector_without_motion()
    lecture = FakeLecture(
        three_words_added(),
        [Gap(13.0, 16.0), Gap(40.0, 42.0)],
    )
    found = detector.detect(lecture)
    assert [event.channel for event in found] == ["added_words", "speech_gap"]
    assert [event.t for event in found] == [12.0, 40.0]


def test_the_two_kinds_of_candidate_have_different_identifiers():
    detector = build_detector_without_motion()
    lecture = FakeLecture(
        three_words_added(),
        [Gap(13.0, 16.0), Gap(40.0, 42.0)],
    )
    found = detector.detect(lecture)
    assert found[0].id == "lec-a#0001" and found[1].id == "lec-a#g002"


def test_the_summary_counts_by_channel():
    detector = build_detector_without_motion()
    lecture = FakeLecture(
        three_words_added(),
        [Gap(13.0, 16.0), Gap(40.0, 42.0)],
    )
    summary = detector.summarise(detector.detect(lecture))
    assert summary == {
        "candidates": 2,
        "visual": 1,
        "speech_only": 1,
        "by_channel": {"slide_turnover": 0, "added_words": 1, "speech_gap": 1},
        "visual_without_a_gap": 0,
    }


def build_detector_without_motion() -> MomentDetector:
    return MomentDetector(
        [FRAME_CHANNELS.create("slide_turnover"), FRAME_CHANNELS.create("added_words")],
        GAP_CHANNELS.create("speech_gap"),
        DetectionSettings(),
    )


def test_a_detector_needs_a_frame_channel():
    with pytest.raises(ValueError):
        MomentDetector([], GAP_CHANNELS.create("speech_gap"))


def test_the_shipped_detection_settings(shipped_configs):
    settings = load_detection_settings(shipped_configs / "detection.yaml")
    assert [spec.name for spec in settings.frame_channels] == [
        "slide_turnover",
        "added_words",
        "pixel_motion",
    ]
    assert settings.gap_channel.name == "speech_gap"
    assert settings.settings.minimum_gap == 1.5
    assert settings.settings.heartbeat_seconds == 30.0
    detector = build_detector(settings)
    assert detector.channel_names == (
        "slide_turnover",
        "added_words",
        "pixel_motion",
        "speech_gap",
    )


# ---------------------------------------------------------------- stills
def test_the_window_reaches_past_the_event_at_every_distance():
    policy = EventWindow()
    for apart in [value / 4 for value in range(0, 121)]:
        planned = policy.plan(candidate(detail={"keyframes_apart": apart}))
        assert planned.reaches_past_the_event, apart
        assert planned.after_event > 0


def test_a_wider_pair_of_keyframes_gets_more_stills():
    policy = EventWindow(wide_above_seconds=3.0)
    assert policy.plan(candidate(detail={"keyframes_apart": 2.0})).count == 3
    assert policy.plan(candidate(detail={"keyframes_apart": 6.0})).count == 5


def test_a_pair_one_second_apart_still_gets_a_usable_window():
    planned = EventWindow().plan(candidate(detail={"keyframes_apart": 0.0}))
    assert planned.span >= 6.0 and planned.back >= 2.0
    assert planned.count == 3


def test_how_far_back_the_window_opens_is_bounded():
    policy = EventWindow(minimum_back=2.0, maximum_back=20.0)
    assert policy.plan(candidate(detail={"keyframes_apart": 900.0})).back == 20.0
    assert policy.plan(candidate(detail={"keyframes_apart": 0.5})).back == 2.0


def test_the_window_never_opens_before_the_start_of_the_recording():
    planned = EventWindow().plan(candidate(t=1.0, detail={"keyframes_apart": 10.0}))
    assert planned.start == 0.0


def test_a_window_of_two_stills_has_no_interval_after_the_event():
    with pytest.raises(ValueError):
        EventWindow(narrow_count=2)


def test_the_keyframes_of_a_moment_are_those_in_its_window(tmp_path):
    policy = KeyframeWindow(before=2.0, after=5.0, count=3)
    available = [(20, tmp_path / "a.jpg"), (31, tmp_path / "b.jpg"), (33, tmp_path / "c.jpg")]
    chosen = policy.keyframes(candidate(t=30.0), available)
    assert [path.name for path in chosen] == ["b.jpg", "c.jpg"]


def test_where_the_window_holds_no_keyframe_the_nearest_one_stands_in(tmp_path):
    policy = KeyframeWindow()
    available = [(200, tmp_path / "far.jpg"), (45, tmp_path / "near.jpg")]
    assert policy.keyframes(candidate(t=30.0), available)[0].name == "near.jpg"
    assert policy.keyframes(candidate(t=30.0), []) == []


def test_the_still_policies_in_use():
    assert STILL_POLICIES.names() == ("event_window", "keyframe_window")


# ---------------------------------------------------------------- windows
def test_the_windows_of_the_stages():
    assert TRANSCRIPT_WINDOWS.names() == ("fixed", "next_event", "symmetric")


def test_the_classifier_reads_as_much_before_as_after():
    assert Symmetric(8.0).span(100.0) == Span(92.0, 108.0)


def test_a_writer_reads_more_after_the_moment_than_before():
    span = Fixed(before=5.0, after=8.0).span(100.0)
    assert span == Span(95.0, 108.0)
    assert span.end - 100.0 > 100.0 - span.start


def test_the_window_may_run_to_the_next_event():
    window = UntilNextEvent(before=5.0, cap=45.0)
    assert window.span(100.0, next_event=120.0) == Span(95.0, 120.0)
    assert window.span(100.0, next_event=None) == Span(95.0, 145.0)
    assert window.span(100.0, next_event=900.0) == Span(95.0, 145.0)
    assert window.span(100.0, next_event=90.0) == Span(95.0, 145.0)


def test_the_words_of_a_window_are_those_beginning_inside_it():
    from ad_for_edu.data.media import Word

    words = [Word(94.0, "before"), Word(96.0, "inside"), Word(107.9, "also"), Word(108.0, "after")]
    assert words_in(words, Span(95.0, 108.0)) == "inside also"


def test_a_silent_window_says_so_rather_than_coming_back_empty():
    from ad_for_edu.data.media import Word

    assert words_in([Word(1.0, "far")], Span(95.0, 108.0)) == NO_SPEECH
    assert words_in([], Span(0.0, 10.0)) == NO_SPEECH


# ---------------------------------------------------------------- rungs
def test_the_rung_policies_in_use():
    assert RUNG_POLICIES.names() == ("rung_budgets", "uniform_budget")


def test_a_rung_is_tested_against_its_own_budget():
    policy = RungBudgets()
    # The first rung needs the whole description inside the gap: 25 words is 10.7 s.
    assert policy.reachable(11.0) == 1
    # The second compresses to 15 words, which is 6.4 s.
    assert policy.reachable(7.0) == 2
    # Below that the third delivers late.
    assert policy.reachable(0.0) == 3


def test_one_budget_for_every_rung_makes_the_compressing_rung_unreachable():
    policy = UniformBudget(words=25)
    assert {policy.reachable(pause) for pause in (0.0, 1.0, 3.0, 6.5, 8.0)} == {3}
    assert policy.reachable(11.0) == 1


def test_a_description_that_fits_no_rung_falls_to_the_last():
    assert RungBudgets().reachable(-1.0) == 3
    assert fits(3, 25, 0.0) and not fits(1, 25, 0.0)
    assert fits(4, 5, 0.0) and not fits(4, 6, 0.0)
    assert fits(5, 10_000, 0.0)


def test_the_words_a_rung_can_deliver_follow_the_gap():
    assert deliverable_words(1, 0.0) == 0.0
    assert deliverable_words(1, 60.0) == pytest.approx(140.0)
    assert deliverable_words(3, 0.0) == pytest.approx(20.0 / 60.0 * 140.0)
    assert deliverable_words(4, 100.0) == RUNG_WORDS[4]


def test_the_budgets_of_the_ladder():
    assert RUNG_WORDS == {1: 25, 2: 15, 3: 25, 4: 5}


# ---------------------------------------------------------------- context
class ContextLecture:
    def __init__(self):
        from ad_for_edu.data.media import Word

        self.lecture_id = "lec-a"
        self.words = (Word(96.0, "so"), Word(97.0, "here"), Word(300.0, "later"))
        self.gaps = (Gap(101.0, 103.0), Gap(400.0, 410.0))
        self.slide_text = (
            SlideTextFrame(98.0, "Action potential phases"),
            SlideTextFrame(300.0, "Another slide"),
        )
        self.keyframes = ()


def test_the_six_fields_come_from_six_readers():
    builder = ContextBuilder(Fixed(5.0, 8.0), RungBudgets())
    built = builder.build("lec-a#0007", "figure", ContextLecture(), 100.0)
    assert built.type == "figure"
    assert built.transcript_window == "so here"
    assert built.slide_ocr == "Action potential phases"
    assert built.pause_after == 2.0
    assert built.reachable_rung == 3
    assert built.described_so_far == NOTHING_DESCRIBED_YET


def test_the_slide_of_another_moment_is_not_handed_over():
    assert slide_text_at(ContextLecture(), 100.0) == "Action potential phases"
    assert slide_text_at(ContextLecture(), 200.0) == NO_SLIDE_TEXT


def test_a_moment_with_no_usable_gap_reports_none_rather_than_guessing():
    assert pause_after(ContextLecture(), 100.0) == 2.0
    assert pause_after(ContextLecture(), 250.0) == 0.0


def test_the_longest_unbroken_stretch_counts_not_the_total():
    class Broken(ContextLecture):
        def __init__(self):
            super().__init__()
            self.gaps = (Gap(101.0, 101.5), Gap(102.0, 102.5), Gap(103.0, 103.5))

    assert pause_after(Broken(), 100.0) == 0.5


def test_what_has_been_described_is_carried_into_the_context():
    builder = ContextBuilder(Fixed(), RungBudgets())
    built = builder.build(
        "lec-a#0007", "figure", ContextLecture(), 100.0, described_so_far=["the axes", "the peak"]
    )
    assert built.described_so_far == "the axes; the peak"


def test_the_context_fills_exactly_the_placeholders_of_a_writer_prompt():
    from ad_for_edu.prompts import template

    builder = ContextBuilder(Fixed(), RungBudgets())
    built = builder.build("lec-a#0007", "figure", ContextLecture(), 100.0)
    bound = template("reference_fixed_window").bind(built.as_prompt_values())
    assert "figure" in bound and "so here" in bound


# ---------------------------------------------------------------- classification
def test_the_reply_is_read_one_object_per_line():
    reply = (
        '```json\n'
        '{"id": "lec-a#0001", "t": "00:30", "type": "figure", "what_on_screen": "a diagram"}\n'
        '{"id": "lec-a#0002", "t": "00:41", "type": "reject", "what_on_screen": "nothing"}\n'
        '```'
    )
    answers = parse_objects(reply)
    assert [answer["id"] for answer in answers] == ["lec-a#0001", "lec-a#0002"]


def test_a_reply_of_one_structure_does_not_collapse_a_batch():
    # A single object spanning the whole reply is one answer, not the whole batch.
    assert len(parse_objects('{"id": "lec-a#0001", "type": "figure"}')) == 1


def test_a_preamble_is_not_read_as_an_answer():
    reply = 'Let me look at these.\n{"id": "lec-a#0001", "type": "figure"}'
    assert len(parse_objects(reply)) == 1


def test_a_malformed_line_is_dropped_and_shows_up_as_a_short_batch():
    reply = '{"id": "lec-a#0001", "type": "figure"}\n{"id": "lec-a#0002", "type":\n'
    answers = parse_objects(reply)
    assert len(answers) == 1
    assert batch_fault(["lec-a#0001", "lec-a#0002"], answers) is not None


def test_a_complete_and_correctly_identified_batch_passes():
    wanted = ["lec-a#0001", "lec-a#0002"]
    got = [{"id": "lec-a#0002"}, {"id": "lec-a#0001"}]
    assert batch_fault(wanted, got) is None


def test_a_reply_that_stops_early_is_rejected():
    fault = batch_fault(["lec-a#0001", "lec-a#0002", "lec-a#0003"], [{"id": "lec-a#0001"}])
    assert fault is not None and "3" in fault and "1" in fault


def test_a_repeated_identifier_at_the_right_count_is_rejected():
    wanted = ["lec-a#0001", "lec-a#0002"]
    got = [{"id": "lec-a#0001"}, {"id": "lec-a#0001"}]
    fault = batch_fault(wanted, got)
    assert fault is not None and "more than once" in fault


def test_an_identifier_from_another_batch_is_rejected():
    fault = batch_fault(["lec-a#0001"], [{"id": "lec-b#0009"}])
    assert fault is not None and "not answered" in fault


def test_the_offered_labels_are_read_back_out_of_the_prompt():
    prompt = prompt_for(draws_pointer=True)
    offered = offered_types(prompt)
    assert "pointing" in offered and "reject" in offered
    assert len(offered) == 13


def test_a_recording_without_a_pointer_is_not_offered_the_label():
    without = prompt_for(draws_pointer=False)
    assert "pointing" not in offered_types(without)
    assert "cursor" not in without.lower()
    assert len(offered_types(without)) == 12


def test_a_label_outside_the_offered_set_is_rejected():
    prompt = prompt_for(draws_pointer=True)
    assert vocabulary_fault(prompt, [{"type": "figure"}]) is None
    fault = vocabulary_fault(prompt, [{"type": "slide_transition"}, {"type": "figure"}])
    assert fault is not None and "slide_transition" in fault


def test_a_prompt_whose_label_list_did_not_compose_fails_even_on_good_labels():
    # The guard that checks only the labels would pass here: the reply agrees with
    # itself, and the label the prompt should have offered is absent from both sides.
    broken = "TYPES - use exactly one:\n  figure      a diagram\n\nreject anything else"
    fault = vocabulary_fault(broken, [{"type": "figure"}])
    assert fault is not None and "did not compose" in fault


def test_a_prompt_with_an_unfilled_placeholder_is_not_sent():
    from ad_for_edu.moments.classification import assert_ready_to_send

    with pytest.raises(ContractError) as caught:
        assert_ready_to_send("Classify these. {types}")
    assert "{types}" in str(caught.value)


def test_a_prompt_without_a_label_list_is_not_sent():
    from ad_for_edu.moments.classification import assert_ready_to_send

    with pytest.raises(ContractError):
        assert_ready_to_send("Classify these candidates carefully.")


def test_the_label_block_offers_one_line_per_label():
    block = types_block(["slide", "figure", "reject"])
    assert offered_types(block) == {"slide", "figure", "reject"}


def test_a_batch_never_spans_two_lectures():
    candidates = [
        candidate(id="lec-a#0001", lecture="lec-a"),
        candidate(id="lec-b#0001", lecture="lec-b"),
        candidate(id="lec-a#0002", lecture="lec-a"),
        candidate(id="lec-a#0003", lecture="lec-a"),
    ]
    batches = batch_by_lecture(candidates, 2)
    assert [[event.id for event in batch] for batch in batches] == [
        ["lec-a#0001", "lec-a#0002"],
        ["lec-a#0003"],
        ["lec-b#0001"],
    ]
    assert all(len({event.lecture for event in batch}) == 1 for batch in batches)


def test_a_batch_of_no_candidates_is_refused():
    with pytest.raises(ValueError):
        batch_by_lecture([candidate()], 0)


def test_the_request_names_every_candidate_and_the_count_asked_for():
    candidates = [candidate(id="lec-a#0001"), candidate(id="lec-a#0002")]
    text = batch_request_text(candidates, {"lec-a#0001": "so here", "lec-a#0002": NO_SPEECH})
    assert "lec-a#0001" in text and "lec-a#0002" in text
    assert "exactly 2" in text and NO_SPEECH in text


def test_the_request_is_exactly_the_lines_it_is_meant_to_be():
    """The text itself, because this is what a provider receives.

    A candidate per line in the order asked, then a blank line, then the count. A
    reordering or a lost blank line would leave every other check here passing.
    """
    candidates = [candidate(id="lec-a#0001", channel="added_words")]
    text = batch_request_text(candidates, {"lec-a#0001": "so here we are"})
    assert text.split(chr(10)) == [
        "CANDIDATES IN THIS BATCH:",
        "- id: lec-a#0001 | detector channel: added_words | transcript: so here we are",
        "",
        "Return exactly 1 JSON objects, one per candidate, in this order.",
    ]


def test_a_candidate_without_a_transcript_cannot_be_asked_about():
    # Reading the window is not optional: a missing one would be sent as an empty
    # line and read as a lecture that said nothing there.
    with pytest.raises(KeyError):
        batch_request_text([candidate(id="lec-a#0001")], {})


def test_an_answer_is_read_with_the_time_it_states():
    # The time comes in the form the stills carry it: minutes and seconds.
    answer = {"id": "lec-a#0001", "t": "01:30", "type": "figure", "what_on_screen": " a diagram "}
    read = read_answer(answer, candidate(t=95.0))
    assert read.time == 90.0 and read.type == "figure"
    assert read.what_on_screen == "a diagram"


def test_a_time_past_the_hour_is_read_as_minutes():
    read = read_answer({"id": "lec-a#0001", "t": "71:03.50", "type": "slide"}, candidate(t=1.0))
    assert read.time == pytest.approx(4263.5)


def test_an_answer_without_a_readable_time_keeps_the_time_of_the_candidate():
    read = read_answer({"id": "lec-a#0001", "type": "figure"}, candidate(t=95.0))
    assert read.time == 95.0
    read = read_answer({"id": "lec-a#0001", "t": "later", "type": "slide"}, candidate(t=95.0))
    assert read.time == 95.0


def test_an_answer_outside_the_closed_set_is_refused():
    with pytest.raises(ContractError):
        read_answer({"id": "lec-a#0001", "type": "pointer"}, candidate())


def test_a_rejected_candidate_is_not_a_moment():
    rejected = read_answer({"id": "lec-a#0001", "type": REJECT}, candidate())
    assert rejected.rejected
    with pytest.raises(ContractError):
        to_moment(rejected, candidate(), "model-x")


def test_an_accepted_candidate_becomes_a_moment():
    accepted = Classification("lec-a#0001", "pointing", 31.0, "a cell", revisit=True)
    moment = to_moment(accepted, candidate(channel="pixel_motion"), "model-x")
    assert moment.id == "lec-a#0001" and moment.type == "pointing"
    assert moment.t == 31.0 and moment.channel == "pixel_motion"
    assert moment.revisit and moment.model == "model-x"


def test_the_shipped_classification_settings(shipped_configs):
    settings = load_classification_settings(shipped_configs / "classification.yaml")
    assert settings.model == "gemini-3.1-pro"
    assert settings.batch_size == 10 and settings.temperature == 0.0
    assert settings.reasoning_effort == "low" and settings.max_tokens == 8000
    assert settings.still_policy.name == "event_window"
    assert settings.transcript_window.name == "symmetric"
    assert settings.transcript_window.params == {"seconds": 8.0}


def test_a_misspelled_classification_key_is_refused(tmp_path):
    path = tmp_path / "classification.yaml"
    path.write_text(
        "model: m\nbatch_size: 10\nstill_policy: event_window\n"
        "transcript_window: symmetric\nbatch_sizes: 4\n",
        encoding="utf-8",
    )
    with pytest.raises(SettingsError):
        load_classification_settings(path)
