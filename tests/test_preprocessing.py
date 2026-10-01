"""Preparing a lecture: media operations, transcription, gaps, slide text, pointer probe."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from ad_for_edu.core.errors import MissingSourceError, ProcessError, SettingsError
from ad_for_edu.core.timecodes import packed_to_seconds
from ad_for_edu.preprocessing import (
    GAP_DETECTORS,
    SLIDE_TEXT_READERS,
    STEPS,
    TRANSCRIBERS,
    LecturePreprocessor,
    Transcript,
    allowed_types,
    gaps_between,
    load_preprocessing_settings,
    media,
    probe_frames,
    renders_cursor,
)
from ad_for_edu.preprocessing.cursor import (
    CONTENT,
    POINTER,
    STILL,
    cell_deltas,
    layout_mask,
    timestamp_mask,
)
from ad_for_edu.preprocessing.slide_text import EasyOcrReader, SlideReading
from ad_for_edu.preprocessing.speech_gaps import SileroGaps
from ad_for_edu.preprocessing.transcription import HostedWhisper

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
needs_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg is not installed")

#: A font to draw the burned-in time with. ffmpeg crashes rather than falling back
#: when it can find no font configuration, so the tests name one where they can.
FONTS = ("C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT = next((name for name in FONTS if Path(name).is_file()), None)


def make_video(target: Path, seconds: int = 8, rate: int = 5) -> Path:
    media._run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=duration={seconds}:size=320x240:rate={rate}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={seconds}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-shortest",
            str(target),
        ],
        what="build a test recording",
    )
    return target


@pytest.fixture(scope="module")
def recording(tmp_path_factory) -> Path:
    if not HAS_FFMPEG:
        pytest.skip("ffmpeg is not installed")
    return make_video(tmp_path_factory.mktemp("media") / "lec-a.mp4")


# ---------------------------------------------------------------- the strategies in use
def test_only_the_strategies_in_use_are_registered():
    assert TRANSCRIBERS.names() == ("hosted_whisper",)
    assert GAP_DETECTORS.names() == ("silero",)
    assert SLIDE_TEXT_READERS.names() == ("easyocr",)


def test_the_shipped_settings(shipped_configs):
    settings = load_preprocessing_settings(shipped_configs / "preprocessing.yaml")
    assert settings.transcriber.name == "hosted_whisper"
    assert settings.transcriber.params["chunk_seconds"] == 600
    assert settings.gap_detector.params == {"minimum_gap": 0.5}
    assert settings.keyframes.grid_seconds == 30
    assert settings.keyframes.scene_threshold == 0.3


def test_a_misspelled_settings_key_is_refused(tmp_path):
    path = tmp_path / "preprocessing.yaml"
    path.write_text(
        "transcriber: hosted_whisper\ngap_detector: silero\n"
        "slide_text_reader: easyocr\nkeyframes: {grid_second: 30}\n",
        encoding="utf-8",
    )
    with pytest.raises(SettingsError) as caught:
        load_preprocessing_settings(path)
    assert "grid_second" in str(caught.value) and "grid_seconds" in str(caught.value)


# ---------------------------------------------------------------- the burned-in time
def test_the_burned_time_is_minutes_seconds_and_hundredths_with_no_hour_field():
    overlay = media.timestamp_overlay(0.0)
    assert overlay.count("%{eif") == 3
    assert overlay.count("}" + chr(92) + ":%{") == 1
    assert overlay.count("}.%{") == 1
    assert "/3600" not in overlay
    assert "t = " in overlay and "drawtext" in overlay


def test_a_font_is_named_when_one_is_given():
    assert "fontfile=" not in media.timestamp_overlay(0.0)
    named = media.timestamp_overlay(0.0, "C:/fonts/a.ttf")
    assert "fontfile='C" + chr(92) + ":/fonts/a.ttf'" in named


def test_a_windows_path_is_written_as_a_filter_value_accepts_it():
    escaped = media.escape_for_filter("C:" + chr(92) + "fonts" + chr(92) + "a.ttf")
    assert escaped == "C" + chr(92) + ":/fonts/a.ttf"
    assert media.escape_for_filter("/usr/share/a.ttf") == "/usr/share/a.ttf"


def test_a_missing_font_is_reported_as_such(monkeypatch):
    failure = media.subprocess.CalledProcessError(
        1, ["ffmpeg"], stderr="Fontconfig error: Cannot load default config file: No such file"
    )
    monkeypatch.setattr(media.subprocess, "run", _raise(failure))
    with pytest.raises(ProcessError) as caught:
        media.duration(Path("any.mp4"))
    assert "no font" in str(caught.value)


def test_the_burned_time_counts_from_the_start_of_the_stretch():
    assert "(t+0.000)" in media.timestamp_overlay(0.0)
    assert "(t+125.500)" in media.timestamp_overlay(125.5)


@needs_ffmpeg
def test_the_length_of_a_recording_is_read(recording):
    assert media.duration(recording) == pytest.approx(8.0, abs=0.5)


@needs_ffmpeg
def test_a_recording_that_is_not_there_is_an_error(tmp_path):
    with pytest.raises(ProcessError):
        media.duration(tmp_path / "absent.mp4")


def test_a_missing_program_is_named(monkeypatch):
    monkeypatch.setattr(media.subprocess, "run", _raise(FileNotFoundError()))
    with pytest.raises(MissingSourceError) as caught:
        media.duration(Path("any.mp4"))
    assert "ffprobe" in str(caught.value)


def _raise(error):
    def fail(*_args, **_kwargs):
        raise error

    return fail


# ---------------------------------------------------------------- stills
@needs_ffmpeg
def test_stills_are_numbered_from_zero_and_span_the_stretch(recording, tmp_path):
    request = media.StillRequest(
        recording, start=2.0, span=3.0, count=5, target=tmp_path / "s", font_file=FONT
    )
    stills = media.extract_stills(request)
    assert [path.name for path in stills] == [f"{index:02d}.jpg" for index in range(5)]
    assert all(path.stat().st_size > media.MIN_STILL_BYTES for path in stills)


@needs_ffmpeg
def test_stills_already_extracted_are_not_extracted_again(recording, tmp_path):
    request = media.StillRequest(
        recording, start=1.0, span=2.0, count=3, target=tmp_path / "s", font_file=FONT
    )
    first = media.extract_stills(request)
    stamps = [path.stat().st_mtime_ns for path in first]
    again = media.extract_stills(request)
    assert again == first
    assert [path.stat().st_mtime_ns for path in again] == stamps


@needs_ffmpeg
def test_a_stretch_before_the_start_of_the_recording_is_clamped(recording, tmp_path):
    request = media.StillRequest(
        recording, start=-5.0, span=2.0, count=2, target=tmp_path / "s", font_file=FONT
    )
    assert len(media.extract_stills(request)) == 2


@pytest.mark.parametrize(("count", "span"), [(0, 3.0), (-1, 3.0), (3, 0.0), (3, -1.0)])
def test_a_request_for_no_stills_or_no_time_is_refused(count, span, tmp_path):
    with pytest.raises(ValueError):
        media.StillRequest(Path("v.mp4"), start=0.0, span=span, count=count, target=tmp_path)


@needs_ffmpeg
def test_a_clip_is_small_silent_and_reused(recording, tmp_path):
    target = tmp_path / "clip.mp4"
    media.extract_clip(recording, 1.0, 4.0, target)
    assert target.stat().st_size >= media.MIN_CLIP_BYTES
    stamp = target.stat().st_mtime_ns
    media.extract_clip(recording, 1.0, 4.0, target)
    assert target.stat().st_mtime_ns == stamp


# ---------------------------------------------------------------- keyframes
def test_keyframe_times_join_the_grid_and_the_scene_changes():
    times = media.keyframe_times(95.0, [12.4, 47.9], grid_seconds=30.0)
    assert times == [0.0, 12.4, 30.0, 47.9, 60.0, 90.0]


def test_times_too_close_together_collapse_to_the_first_of_them():
    times = media.keyframe_times(60.0, [30.2, 30.6], grid_seconds=30.0, minimum_gap=1.0)
    assert times == [0.0, 30.0]
    assert media.keyframe_times(60.0, [31.5], grid_seconds=30.0) == [0.0, 30.0, 31.5]


@needs_ffmpeg
def test_a_keyframe_is_named_by_its_time_and_kept(recording, tmp_path):
    made = media.extract_keyframes(recording, [0.0, 65.0 % 8, 6.0], tmp_path / "lec-a")
    assert [path.name for _time, path in made] == ["000000.jpg", "000001.jpg", "000006.jpg"]
    for time, path in made:
        assert packed_to_seconds(path.name) == time
    stamps = {path: path.stat().st_mtime_ns for _t, path in made}
    media.extract_keyframes(recording, [0.0, 6.0], tmp_path / "lec-a")
    assert all(path.stat().st_mtime_ns == stamp for path, stamp in stamps.items())


# ---------------------------------------------------------------- transcription
class FakeTranscriptions:
    def __init__(self, replies):
        self.replies = list(replies)
        self.asked = 0

    def create(self, **_arguments):
        self.asked += 1
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(model_dump=lambda: reply)


def whisper(replies, **params):
    transcriptions = FakeTranscriptions(replies)
    fake = SimpleNamespace(audio=SimpleNamespace(transcriptions=transcriptions))
    service = HostedWhisper(client_factory=lambda: fake, **params)
    service._sleep = lambda seconds: None
    return service, transcriptions


def reply(words, language="en"):
    return {
        "language": language,
        "segments": [{"start": words[0]["start"], "end": words[-1]["end"], "text": " spoken "}],
        "words": words,
    }


def test_a_transcript_carries_the_second_each_word_begins_at(tmp_path):
    audio = tmp_path / "lec-a.mp3"
    audio.write_bytes(b"audio")
    service, calls = whisper([reply([{"start": 1.0, "end": 1.4, "word": "so"}])])
    transcript = service.transcribe("lec-a", audio, 60.0)
    assert calls.asked == 1
    assert transcript.words == ({"start": 1.0, "end": 1.4, "word": "so"},)
    assert transcript.language == "en" and transcript.duration == 60.0


@needs_ffmpeg
def test_long_audio_is_sent_in_stretches_and_the_times_put_back(tmp_path, recording):
    audio = media.extract_audio(recording, tmp_path / "lec-a.mp3")
    service, calls = whisper(
        [
            reply([{"start": 1.0, "end": 1.5, "word": "first"}]),
            reply([{"start": 2.0, "end": 2.5, "word": "second"}]),
        ],
        chunk_seconds=4.0,
    )
    transcript = service.transcribe("lec-a", audio, 8.0)
    assert calls.asked == 2
    assert [word["word"] for word in transcript.words] == ["first", "second"]
    assert [word["start"] for word in transcript.words] == [1.0, 6.0]
    assert transcript.segments[1]["start"] == 6.0
    assert not list(tmp_path.glob("lec-a_*.mp3"))


def test_a_transient_failure_is_retried_and_a_lasting_one_reported(tmp_path):
    audio = tmp_path / "lec-a.mp3"
    audio.write_bytes(b"audio")
    words = [{"start": 0.5, "end": 0.9, "word": "yes"}]
    service, calls = whisper([RuntimeError("503 unavailable"), reply(words)])
    assert service.transcribe("lec-a", audio, 10.0).words[0]["word"] == "yes"
    assert calls.asked == 2
    service, _ = whisper([RuntimeError("timeout")] * 5, attempts=3)
    with pytest.raises(ProcessError):
        service.transcribe("lec-a", audio, 10.0)


def test_another_failure_is_raised_as_it_is(tmp_path):
    audio = tmp_path / "lec-a.mp3"
    audio.write_bytes(b"audio")
    service, _ = whisper([KeyError("unexpected")])
    with pytest.raises(KeyError):
        service.transcribe("lec-a", audio, 10.0)


def test_a_transcript_writes_the_shape_the_media_reader_expects(tmp_path):
    written = Transcript("lec-a", ({"start": 1.0, "end": 1.2, "word": "x"},), duration=5.0)
    assert set(written.as_dict()) == {
        "lecture_id",
        "duration",
        "language",
        "model",
        "segments",
        "words",
    }


# ---------------------------------------------------------------- speech gaps
def test_the_gaps_are_what_lies_between_the_speech():
    speech = [(2.0, 4.0), (6.0, 7.0)]
    assert [(gap.start, gap.end) for gap in gaps_between(speech, 10.0, 0.5)] == [
        (0.0, 2.0),
        (4.0, 6.0),
        (7.0, 10.0),
    ]


def test_a_gap_shorter_than_the_minimum_is_not_a_gap():
    found = gaps_between([(0.4, 4.0), (4.3, 9.9)], 10.0, 0.5)
    assert [(gap.start, gap.end) for gap in found] == []


def test_overlapping_speech_does_not_create_a_gap():
    found = gaps_between([(1.0, 5.0), (2.0, 3.0), (6.0, 8.0)], 9.0, 0.5)
    assert [(gap.start, gap.end) for gap in found] == [(0.0, 1.0), (5.0, 6.0), (8.0, 9.0)]


def test_a_lecture_without_speech_is_one_long_gap():
    found = gaps_between([], 30.0, 0.5)
    assert len(found) == 1 and found[0].length == 30.0


def test_the_detector_is_asked_for_speech_and_the_gaps_derived(tmp_path, monkeypatch):
    audio = tmp_path / "lec-a.mp3"
    audio.write_bytes(b"audio")
    seen = {}

    def detect(samples, model, sampling_rate, return_seconds):
        seen["rate"] = sampling_rate
        seen["samples"] = len(samples)
        return [{"start": 1.0, "end": 3.0}]

    import numpy

    monkeypatch.setattr(
        "ad_for_edu.preprocessing.speech_gaps.read_audio_samples",
        lambda path: numpy.zeros(160, dtype=numpy.float32),
    )
    detector = SileroGaps(minimum_gap=0.5, model_factory=lambda: object(), detector=detect)
    found = detector.gaps(audio, 6.0)
    assert seen["rate"] == 16000 and seen["samples"] == 160
    assert [(gap.start, gap.end) for gap in found] == [(0.0, 1.0), (3.0, 6.0)]


# ---------------------------------------------------------------- slide text
def test_the_text_of_a_keyframe_is_read_with_its_boxes(tmp_path):
    found = [
        ([[0, 0], [10, 0], [10, 8], [0, 8]], "Action", 0.9),
        ([[0, 10], [20, 10], [20, 18], [0, 18]], "Potential", 0.7),
    ]
    reader = EasyOcrReader(reader_factory=lambda: SimpleNamespace(readtext=lambda path: found))
    readings = reader.read([(12.0, tmp_path / "000012.jpg")])
    assert readings[0].text == "Action Potential"
    assert readings[0].confidence == pytest.approx(0.8)
    assert readings[0].boxes[0]["box"] == [[0, 0], [10, 0], [10, 8], [0, 8]]


def test_a_frame_that_cannot_be_read_yields_no_text_not_a_failed_lecture(tmp_path):
    def readtext(path):
        raise OSError("cannot open")

    reader = EasyOcrReader(reader_factory=lambda: SimpleNamespace(readtext=readtext))
    readings = reader.read([(1.0, tmp_path / "a.jpg"), (2.0, tmp_path / "b.jpg")])
    assert len(readings) == 2 and all(reading.text == "" for reading in readings)


def test_a_reading_is_written_with_a_clock_time():
    assert SlideReading(3723.0, "x").as_dict()["timestamp"] == "01:02:03"


# ---------------------------------------------------------------- the pointer probe
def grid(value: int, shape=(64, 96)):
    import numpy

    return numpy.full(shape, value, dtype=numpy.int16)


def with_patch(base, box, value):
    import numpy

    changed = numpy.array(base, copy=True)
    top, left, height, width = box
    changed[top : top + height, left : left + width] = value
    return changed


def test_nothing_moving_is_still():
    frames = [grid(100), grid(100), grid(100)]
    probe = probe_frames(frames)
    assert probe.hot_cells == 0 and probe.verdict() == STILL
    assert probe.frames == 3 and probe.per_pair == (0, 0)


def test_a_small_patch_moving_is_a_pointer():
    frames = [grid(100), with_patch(grid(100), (32, 48, 12, 12), 255)]
    probe = probe_frames(frames)
    assert 0 < probe.hot_cells <= 6 and probe.verdict() == POINTER
    assert probe.at is not None


def test_a_change_across_the_picture_is_content():
    frames = [grid(100), grid(220)]
    probe = probe_frames(frames)
    assert probe.hot_cells > 6 and probe.verdict() == CONTENT


def test_a_change_between_the_last_two_frames_counts_as_much_as_an_early_one():
    still, moved = grid(100), with_patch(grid(100), (16, 16, 12, 12), 255)
    late = probe_frames([still, still, still, moved])
    early = probe_frames([still, moved, still, still])
    assert late.hot_cells == early.hot_cells > 0


def test_one_frame_cannot_be_probed():
    assert probe_frames([grid(100)]).verdict() == STILL
    assert probe_frames([]).frames == 0


def test_what_moves_in_every_frame_pair_is_part_of_the_layout():
    import numpy

    base = grid(100)
    pairs = []
    for _ in range(5):
        # A patch on the left changes in every pair; one on the right changes once.
        pairs.append(cell_deltas(base, with_patch(base, (0, 0, 16, 16), 255)))
    pairs.append(cell_deltas(base, with_patch(base, (32, 64, 16, 16), 255)))
    mask = layout_mask(pairs, share=0.6)
    assert mask[0][0] and not mask[2][4]
    moved = with_patch(with_patch(base, (0, 0, 16, 16), 255), (32, 64, 12, 12), 255)
    assert probe_frames([base, moved]).hot_cells > probe_frames([base, moved], mask).hot_cells
    assert numpy.count_nonzero(mask) == 1


def test_the_area_of_the_burned_time_is_left_out():
    base = grid(100, shape=(240, 320))
    mask = timestamp_mask(base.shape, media.TIMESTAMP_BOX)
    assert mask[0][0] and not mask[8][8]
    ticking = with_patch(base, (0, 0, 64, 210), 255)
    assert probe_frames([base, ticking], mask).verdict() == STILL
    assert probe_frames([base, ticking]).verdict() == CONTENT


def test_a_recording_that_shows_a_pointer_often_enough_keeps_the_label():
    many = [POINTER] * 12 + [STILL] * 18 + [CONTENT] * 10
    assert renders_cursor(many)
    assert allowed_types(("slide", "pointing", "figure"), True) == ["slide", "pointing", "figure"]


def test_a_recording_that_never_shows_one_does_not_get_the_label():
    never = [STILL] * 35 + [CONTENT] * 15
    assert not renders_cursor(never)
    assert allowed_types(("slide", "pointing", "figure"), False) == ["slide", "figure"]


def test_too_few_sampled_points_leave_the_question_open():
    assert renders_cursor([STILL] * 19)
    assert renders_cursor([])


# ---------------------------------------------------------------- the steps
class ScriptedTranscriber:
    paid = True

    def transcribe(self, lecture_id, audio, duration):
        return Transcript(
            lecture_id, ({"start": 1.0, "end": 1.2, "word": "so"},), duration=duration
        )


class ScriptedGaps:
    paid = False

    def gaps(self, audio, duration):
        from ad_for_edu.data.media import Gap

        return [Gap(2.0, 3.5)]


class ScriptedSlideText:
    paid = False

    def __init__(self):
        self.seen: list = []

    def read(self, frames):
        self.seen = list(frames)
        return [SlideReading(time, f"slide at {int(time)}") for time, _path in frames]


@pytest.fixture
def preprocessor(tmp_path, recording) -> LecturePreprocessor:
    from ad_for_edu.data.media import MediaLayout

    for name in ("transcripts", "vad", "ocr", "keyframes", "audio", "videos"):
        (tmp_path / name).mkdir()
    shutil.copy(recording, tmp_path / "videos" / "lec-a.mp4")
    layout = MediaLayout(
        transcripts=tmp_path / "transcripts",
        speech_gaps=tmp_path / "vad",
        slide_text=tmp_path / "ocr",
        keyframes=tmp_path / "keyframes",
        videos=tmp_path / "videos",
    )
    return LecturePreprocessor(
        layout=layout,
        audio_dir=tmp_path / "audio",
        transcriber=ScriptedTranscriber(),
        gap_detector=ScriptedGaps(),
        slide_text_reader=ScriptedSlideText(),
    )


@needs_ffmpeg
def test_the_five_steps_write_what_the_later_stages_read(preprocessor):
    video = preprocessor.layout.videos / "lec-a.mp4"
    results = preprocessor.run("lec-a", video)
    assert [result.step for result in results] == list(STEPS)
    assert all(result.done for result in results)
    lecture = preprocessor.layout.lecture("lec-a")
    assert [word.text for word in lecture.words] == ["so"]
    assert [(gap.start, gap.end) for gap in lecture.gaps] == [(2.0, 3.5)]
    assert lecture.keyframes
    assert lecture.slide_text[0].text.startswith("slide at")


@needs_ffmpeg
def test_a_step_already_done_is_not_done_again(preprocessor):
    video = preprocessor.layout.videos / "lec-a.mp4"
    preprocessor.run("lec-a", video)
    again = preprocessor.run("lec-a", video)
    assert [result.done for result in again] == [False] * len(STEPS)


@needs_ffmpeg
def test_the_slide_text_is_read_off_the_keyframes_that_were_extracted(preprocessor):
    video = preprocessor.layout.videos / "lec-a.mp4"
    preprocessor.run("lec-a", video, steps=("audio", "keyframes", "slide_text"))
    frames = preprocessor.layout.lecture("lec-a").keyframes
    assert len(preprocessor.slide_text_reader.seen) == len(frames)
    written = json.loads((preprocessor.layout.slide_text / "lec-a.json").read_text("utf-8"))
    assert len(written["frames"]) == len(frames)


@needs_ffmpeg
def test_only_the_named_steps_run(preprocessor):
    video = preprocessor.layout.videos / "lec-a.mp4"
    results = preprocessor.run("lec-a", video, steps=("audio", "gaps"))
    assert [result.step for result in results] == ["audio", "gaps"]
    assert not (preprocessor.layout.transcripts / "lec-a.json").exists()


def test_an_unknown_step_is_refused(preprocessor):
    with pytest.raises(ValueError):
        preprocessor.run("lec-a", Path("v.mp4"), steps=("audio", "subtitles"))


def test_the_transcript_is_the_one_step_that_costs_money(preprocessor):
    assert preprocessor.paid_steps() == ("transcript",)
    assert preprocessor.paid_steps(("audio", "gaps")) == ()
