"""Training examples, the cells, decoding, predictions and the systems."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ad_for_edu.core.errors import ContractError, SettingsError
from ad_for_edu.core.runs import RunKey
from ad_for_edu.data.schema import ReferenceRow
from ad_for_edu.data.splits import ManifestSplit
from ad_for_edu.inference import (
    DECODING_ATTEMPTS,
    PREFILL,
    SALVAGE_STEPS,
    Prediction,
    assert_covers,
    merge_repairs,
    read_completion,
    read_predictions,
    says_something,
    summarise,
    write_predictions,
)
from ad_for_edu.systems import (
    DESCRIPTION_SYSTEMS,
    EvalMoment,
    ReferenceWriter,
    SlideTitleReadout,
    TrainedDescriber,
    conversation,
    slide_title,
)
from ad_for_edu.training import (
    INPUT_ARMS,
    TRAINING_METHODS,
    Grid,
    LoraSettings,
    TrainingData,
    TrainingSettings,
    answer_for,
    as_preference_rows,
    as_typed,
    build_arm,
    checked_settings,
    example_for,
    keyframe_references,
    load_backbones,
    load_grid,
    prepare,
    settings_of,
    split_by_moment,
    system_prompt,
)

ROW = ReferenceRow(
    output_id="lec-a#0017::writer-x",
    type="figure",
    emit=True,
    ad_text="Two curves cross at the threshold.",
    rung=3,
    rationale="the crossing is unspoken",
    family="writer-x",
    transcript_window="so the two of them meet here",
    slide_ocr="Threshold crossing",
    what_on_screen="A plot with two curves",
    pause_after=1.5,
)


# ---------------------------------------------------------------- examples
def test_the_two_input_arms():
    assert INPUT_ARMS.names() == ("multimodal", "text_only")
    assert INPUT_ARMS.create("multimodal").carries_images
    assert not INPUT_ARMS.create("text_only").carries_images


def test_an_example_carries_the_standard_the_moment_and_the_answer():
    example = example_for(ROW, build_arm("text_only"))
    roles = [message["role"] for message in example["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert example["messages"][0]["content"] == system_prompt()
    assert example["output_id"] == ROW.output_id and example["moment_id"] == "lec-a#0017"
    answer = json.loads(example["messages"][2]["content"])
    assert answer == {
        "emit": True,
        "ad_text": ROW.ad_text,
        "rung": 3,
        "rationale": ROW.rationale,
    }


def test_what_the_writer_concluded_is_not_given_to_the_describer():
    text = example_for(ROW, build_arm("multimodal"))["messages"][1]["content"][-1]["text"]
    assert ROW.transcript_window in text and ROW.slide_ocr in text
    assert ROW.what_on_screen not in text
    assert "coverage" not in text.lower() and "redundant" not in text.lower()


def test_without_pictures_the_summary_of_the_screen_stands_in():
    text = example_for(ROW, build_arm("text_only"))["messages"][1]["content"][0]["text"]
    assert ROW.what_on_screen in text


def test_the_keyframes_are_named_not_carried(tmp_path):
    example = example_for(ROW, build_arm("multimodal"), ["keyframes/lec-a/000017.jpg"])
    parts = example["messages"][1]["content"]
    assert parts[0] == {"type": "image", "image": "keyframes/lec-a/000017.jpg"}
    assert parts[-1]["type"] == "text"


def test_the_text_only_arm_carries_no_picture():
    parts = example_for(ROW, build_arm("text_only"), ["keyframes/lec-a/000017.jpg"])["messages"][1][
        "content"
    ]
    assert all(part["type"] == "text" for part in parts)


def test_a_moment_passed_over_in_silence_is_still_an_example():
    silent = ReferenceRow(output_id="lec-a#0001::w", type="slide", emit=False)
    answer = json.loads(answer_for(silent))
    assert answer == {"emit": False, "ad_text": None, "rung": None, "rationale": None}


def test_the_keyframes_of_a_moment_are_those_around_it(tmp_path):
    available = [
        (10, tmp_path / "000010.jpg"),
        (18, tmp_path / "000018.jpg"),
        (99, tmp_path / "x.jpg"),
    ]
    named = keyframe_references("lec-a", available, 17.0)
    assert named == ["keyframes/lec-a/000018.jpg"]


def test_where_no_keyframe_is_near_the_nearest_one_stands_in(tmp_path):
    available = [(200, tmp_path / "000200.jpg")]
    assert keyframe_references("lec-a", available, 17.0) == ["keyframes/lec-a/000200.jpg"]
    assert keyframe_references("lec-a", [], 17.0) == []


def test_every_message_can_be_written_as_parts():
    typed = as_typed([{"role": "system", "content": "text"}])
    assert typed == [{"role": "system", "content": [{"type": "text", "text": "text"}]}]


# ---------------------------------------------------------------- the material
def rows_for(moments, writers=("a", "b")) -> list[dict]:
    return [
        {"output_id": f"{moment}::{writer}", "lecture": moment.split("#")[0]}
        for moment in moments
        for writer in writers
    ]


def test_a_split_by_moment_keeps_every_row_of_a_moment_together():
    rows = rows_for([f"lec-a#{index:04d}" for index in range(10)])
    data = split_by_moment(rows, 0.2, seed=17)
    from ad_for_edu.core.ids import moment_of

    train = {moment_of(row) for row in data.train}
    development = {moment_of(row) for row in data.development}
    assert train & development == set()
    assert len(data.train) + len(data.development) == 20
    assert len(data.development) % 2 == 0


def test_a_split_is_reproducible_from_its_seed():
    rows = rows_for([f"lec-a#{index:04d}" for index in range(10)])
    first = split_by_moment(rows, 0.2, seed=3).development
    again = split_by_moment(rows, 0.2, seed=3).development
    assert [row["output_id"] for row in first] == [row["output_id"] for row in again]


def test_material_from_a_test_lecture_stops_the_run(tmp_path):
    manifest = tmp_path / "lectures.csv"
    manifest.write_text(
        "lecture_id,video,course,split,renders_cursor\n"
        "lec-a,a.mp4,biology,train,\n"
        "lec-b,b.mp4,algorithms,test,\n",
        encoding="utf-8",
    )
    from ad_for_edu.data import LectureManifest

    policy = ManifestSplit(LectureManifest.load(manifest))
    good = rows_for([f"lec-a#{index:04d}" for index in range(6)])
    prepare(good, policy, share=0.34, seed=1, by="moment")
    with pytest.raises(ContractError) as caught:
        prepare(good + rows_for(["lec-b#0001"]), policy, share=0.2, seed=1, by="moment")
    assert "lec-b" in str(caught.value)


def test_the_preference_arm_holds_out_whole_lectures(tmp_path):
    manifest = tmp_path / "lectures.csv"
    manifest.write_text(
        "lecture_id,video,course,split,renders_cursor\n"
        + "".join(f"lec-{name},x.mp4,biology,train,\n" for name in "abcd"),
        encoding="utf-8",
    )
    from ad_for_edu.data import LectureManifest

    policy = ManifestSplit(LectureManifest.load(manifest))
    rows = rows_for([f"lec-{name}#{index:04d}" for name in "abcd" for index in range(5)])
    data = prepare(rows, policy, share=0.25, seed=17, by="lecture")
    from ad_for_edu.core.ids import lecture_of, moment_of

    train = {lecture_of(moment_of(row)) for row in data.train}
    development = {lecture_of(moment_of(row)) for row in data.development}
    assert train & development == set()


def test_nothing_to_train_on_is_refused():
    with pytest.raises(ContractError):
        TrainingData([], [])


def test_an_unknown_way_of_holding_out_is_refused(tmp_path):
    from ad_for_edu.data import LectureManifest

    manifest = tmp_path / "lectures.csv"
    manifest.write_text(
        "lecture_id,video,course,split,renders_cursor" + chr(10)
        + "lec-a,a.mp4,biology,train," + chr(10),
        encoding="utf-8",
    )
    policy = ManifestSplit(LectureManifest.load(manifest))
    with pytest.raises(ContractError):
        prepare(rows_for(["lec-a#0001"]), policy, share=0.2, seed=1, by="writer")


def test_a_pair_with_no_direction_is_left_out_rather_than_given_one():
    pairs = [
        {"pair_id": "p1", "moment_id": "lec-a#0001", "a": "first", "b": "second", "winner": "a"},
        {"pair_id": "p2", "moment_id": "lec-a#0002", "a": "first", "b": "second", "winner": None},
    ]
    rows = as_preference_rows(
        pairs, prompt_of=lambda pair: pair["moment_id"], chosen_of=lambda pair: pair["winner"] or ""
    )
    assert len(rows) == 1
    assert rows[0]["chosen"] == "first" and rows[0]["rejected"] == "second"


# ---------------------------------------------------------------- the cells
@pytest.fixture
def grid(shipped_configs) -> Grid:
    return load_grid(shipped_configs / "training" / "grid.yaml")


def test_the_grid_names_the_cells(grid):
    assert len(grid.backbones) == 7
    assert grid.arms == ("multimodal", "text_only")
    assert grid.methods == ("sft", "dpo")
    assert grid.seeds == (0, 1, 2)
    assert len(grid.cells()) == 7 * 2 * 2 * 3


def test_the_grid_says_what_each_arm_holds_out(grid):
    assert grid.development_for("sft") == {"share": 0.10, "seed": 17, "by": "moment"}
    assert grid.development_for("dpo") == {"share": 0.12, "seed": 17, "by": "lecture"}
    with pytest.raises(SettingsError):
        grid.development_for("no-such-method")


def test_every_backbone_pins_a_revision(shipped_configs):
    backbones = load_backbones(shipped_configs / "training" / "backbones")
    assert len(backbones) == 7
    for backbone in backbones.values():
        assert len(backbone.revision) >= 32
        assert backbone.model_id.count("/") == 1


def test_a_backbone_without_a_revision_is_refused(tmp_path):
    path = tmp_path / "b.yaml"
    path.write_text("model_id: vendor/model\nrevision: ''\n", encoding="utf-8")
    from ad_for_edu.training import Backbone

    with pytest.raises(SettingsError) as caught:
        Backbone.load(path)
    assert "revision" in str(caught.value)


def test_the_settings_of_a_cell_are_layered(shipped_configs):
    base = shipped_configs
    supervised, lora, backbone = settings_of(
        RunKey("qwen3-vl-8b", "text_only", "sft", 0), base=base
    )
    preference, _lora, _backbone = settings_of(
        RunKey("qwen3-vl-8b", "text_only", "dpo", 1), base=base
    )
    # The method overrides the defaults.
    assert supervised.epochs == 3 and supervised.learning_rate == 1.0e-4
    assert preference.epochs == 2 and preference.learning_rate == 1.5e-5
    # The seed of the cell wins over any layer.
    assert preference.seed == 1
    assert backbone.model_id == "Qwen/Qwen3-VL-8B-Instruct"
    assert lora == LoraSettings(16, 32, 0.05, ("q_proj", "k_proj", "v_proj", "o_proj"))


def test_a_backbone_overrides_what_it_needs_to(shipped_configs):
    small, _lora, _backbone = settings_of(
        RunKey("qwen3-vl-4b", "multimodal", "sft", 0), base=shipped_configs
    )
    large, _lora2, _backbone2 = settings_of(
        RunKey("qwen3-vl-8b", "multimodal", "sft", 0), base=shipped_configs
    )
    assert (small.batch_size, small.gradient_accumulation) == (2, 4)
    assert (large.batch_size, large.gradient_accumulation) == (1, 8)
    assert small.effective_batch == large.effective_batch == 8


def test_a_backbone_may_need_something_told_about_its_pictures(shipped_configs):
    from ad_for_edu.training.build import processor_kwargs

    key = RunKey("internvl3.5-4b", "multimodal", "sft", 0)
    _settings, _lora, backbone = settings_of(key, base=shipped_configs)
    assert processor_kwargs(key, backbone) == {"crop_to_patches": False}
    text_only = RunKey("internvl3.5-4b", "text_only", "sft", 0)
    assert processor_kwargs(text_only, backbone) == {}


def test_the_preference_method_carries_its_own_settings(shipped_configs):
    preference, _lora, _backbone = settings_of(
        RunKey("qwen3-vl-8b", "text_only", "dpo", 0), base=shipped_configs
    )
    assert preference.method == {"beta": 0.1, "rpo_alpha": 0.1}


@pytest.mark.parametrize(
    "key",
    [
        ("no-such-backbone", "text_only", "sft", 0),
        ("qwen3-vl-8b", "audio_only", "sft", 0),
        ("qwen3-vl-8b", "text_only", "no-such-method", 0),
    ],
)
def test_an_unknown_part_of_a_cell_is_refused(shipped_configs, key):
    with pytest.raises(SettingsError):
        settings_of(RunKey(*key), base=shipped_configs)


def test_a_misspelled_settings_key_is_refused(tmp_path, shipped_configs):
    import shutil

    copied = tmp_path / "training"
    shutil.copytree(shipped_configs / "training", copied)
    (copied / "defaults.yaml").write_text("epoch: 3\n", encoding="utf-8")
    with pytest.raises(SettingsError) as caught:
        settings_of(RunKey("qwen3-vl-8b", "text_only", "sft", 0), base=tmp_path)
    assert "epoch" in str(caught.value)


# ---------------------------------------------------------------- the methods
def test_the_two_methods():
    assert TRAINING_METHODS.names() == ("dpo", "sft")
    assert TRAINING_METHODS.create("dpo").starts_from_trained
    assert not TRAINING_METHODS.create("sft").starts_from_trained


def test_a_setting_the_library_does_not_have_stops_the_run():
    with pytest.raises(ContractError) as caught:
        checked_settings({"rpo_alpha": 0.1}, {"beta", "loss_type"}, what="the preference trainer")
    assert "rpo_alpha" in str(caught.value)


def test_a_setting_left_unset_is_not_a_request():
    passed = checked_settings(
        {"beta": None, "loss_type": "sigmoid"}, {"loss_type"}, what="a trainer"
    )
    assert passed == {"loss_type": "sigmoid"}


def test_without_the_library_the_settings_pass_through():
    assert checked_settings({"anything": 1}, None, what="a trainer") == {"anything": 1}


def test_the_settings_a_method_passes_on(tmp_path):
    settings = TrainingSettings(
        epochs=2, learning_rate=1.5e-5, batch_size=1, gradient_accumulation=8, seed=2,
        method={"beta": 0.1},
    )
    method = TRAINING_METHODS.create("dpo")
    passed = method.library_settings(
        settings, tmp_path, available={
            "output_dir", "num_train_epochs", "learning_rate",
            "per_device_train_batch_size", "gradient_accumulation_steps", "warmup_steps",
            "max_grad_norm", "weight_decay", "logging_steps", "save_strategy", "save_steps",
            "save_total_limit", "eval_strategy", "eval_steps", "bf16", "fp16", "seed",
            "max_length", "beta",
        },
    )
    assert passed["num_train_epochs"] == 2 and passed["learning_rate"] == 1.5e-5
    assert passed["beta"] == 0.1 and passed["seed"] == 2
    assert passed["bf16"] is True and passed["fp16"] is False


# ---------------------------------------------------------------- decoding
def steps():
    return [
        SALVAGE_STEPS.create(name)
        for name in ("strict_object", "balanced_object", "open_string")
    ]


def test_a_complete_object_is_read_as_it_is():
    decoded = read_completion('Two curves cross.", "rung": 3, "rationale": "r"}', steps())
    assert decoded.ad_text == "Two curves cross." and decoded.rung == 3
    assert decoded.how == "strict_object" and decoded.filled


def test_an_object_wrapped_in_remarks_is_still_read():
    completion = 'Two curves.", "rung": 3}\nThat is my answer.'
    decoded = read_completion(completion, steps())
    assert decoded.ad_text == "Two curves." and decoded.filled


def test_a_reply_cut_off_in_the_middle_keeps_what_was_written():
    decoded = read_completion("Two curves cross at the thresh", steps())
    assert decoded.ad_text == "Two curves cross at the thresh"
    assert decoded.how == "open_string"


def test_a_description_closed_immediately_is_not_a_description():
    decoded = read_completion('", "rung": 3}', steps())
    assert not decoded.filled and decoded.how == "unrecovered"


def test_a_reply_of_punctuation_is_not_a_description():
    for completion in ('"}', '   ", "rung": 1}', '\\"\\""}'):
        assert not read_completion(completion, steps()).filled


def test_what_counts_as_saying_something():
    assert says_something("a") and says_something("Two curves.")
    assert not says_something("") and not says_something(None)
    assert not says_something('""') and not says_something("None")


def test_the_ways_of_asking_differ_in_how_they_draw():
    greedy = DECODING_ATTEMPTS.create("greedy", maximum_new_tokens=160)
    sampled = DECODING_ATTEMPTS.create("sampled", attempt=2)
    assert greedy.generation_settings(0, 0)["do_sample"] is False
    settings = sampled.generation_settings(0, 3)
    assert settings["do_sample"] is True and settings["temperature"] == 0.7
    assert settings["seed"] == 3002


def test_two_attempts_of_one_seed_draw_differently():
    first = DECODING_ATTEMPTS.create("sampled", attempt=1).generation_settings(0, 5)
    second = DECODING_ATTEMPTS.create("sampled", attempt=2).generation_settings(0, 5)
    assert first["seed"] != second["seed"]


def test_what_is_written_for_the_describer_opens_the_description():
    assert PREFILL.endswith('"') and '"ad_text"' in PREFILL and '"emit": true' in PREFILL


# ---------------------------------------------------------------- predictions
def predictions_for(moments, text="A description."):
    return [
        Prediction(output_id=moment, moment_id=moment, emit=True, ad_text=text, forced=True)
        for moment in moments
    ]


def test_predictions_are_written_and_read_back(tmp_path):
    path = tmp_path / "predictions.jsonl"
    write_predictions(path, predictions_for(["lec-a#0001", "lec-a#0002"]))
    read = read_predictions(path)
    assert [prediction.moment_id for prediction in read] == ["lec-a#0001", "lec-a#0002"]
    assert all(prediction.filled for prediction in read)


def test_a_moment_predicted_twice_is_refused(tmp_path):
    from ad_for_edu.core.io import write_jsonl

    path = tmp_path / "predictions.jsonl"
    write_jsonl(
        path,
        [
            {"moment_id": "lec-a#0001", "ad_text": "one"},
            {"moment_id": "lec-a#0001", "ad_text": "two"},
        ],
    )
    with pytest.raises(ContractError) as caught:
        read_predictions(path)
    assert "twice" in str(caught.value)


def test_the_predictions_must_be_for_the_moments_asked_about():
    wanted = ["lec-a#0001", "lec-a#0002"]
    assert_covers(predictions_for(wanted), wanted)
    with pytest.raises(ContractError) as caught:
        assert_covers(predictions_for(["lec-a#0001"]), wanted)
    assert "missing" in str(caught.value)
    with pytest.raises(ContractError):
        assert_covers(predictions_for([*wanted, "lec-b#0001"]), wanted)


def test_a_second_pass_fills_only_what_the_first_left_empty():
    original = [
        Prediction("lec-a#0001", "lec-a#0001", emit=False, ad_text=None, forced=True),
        Prediction("lec-a#0002", "lec-a#0002", emit=True, ad_text="written", forced=True),
    ]
    repairs = {
        "lec-a#0001": Prediction(
            "lec-a#0001", "lec-a#0001", emit=True, ad_text="recovered", forced=True, how="sampled"
        )
    }
    merged, counts = merge_repairs(original, repairs)
    assert [prediction.ad_text for prediction in merged] == ["recovered", "written"]
    # The decision of the first pass is kept: a repair asks again only for the words.
    assert merged[0].emit is False
    assert counts == {"kept": 1, "repaired": 1, "still_empty": 0}


def test_a_second_pass_may_not_touch_a_moment_that_was_answered():
    original = predictions_for(["lec-a#0001"])
    with pytest.raises(ContractError):
        merge_repairs(original, {"lec-a#0001": predictions_for(["lec-a#0001"])[0]})


def test_a_second_pass_for_an_unknown_moment_is_refused():
    original = [Prediction("lec-a#0001", "lec-a#0001", emit=False, ad_text=None)]
    with pytest.raises(ContractError):
        merge_repairs(original, {"lec-z#0001": original[0]})


def test_the_summary_says_how_the_descriptions_were_obtained():
    predictions = [
        Prediction("m1", "m1", True, "one", how="greedy/strict_object"),
        Prediction("m2", "m2", True, None, how="unrecovered"),
        Prediction("m3", "m3", False, "three", how="greedy/strict_object"),
    ]
    summary = summarise(predictions)
    assert summary["moments"] == 3 and summary["described"] == 2 and summary["empty"] == 1
    assert summary["chose_to_describe"] == 2
    assert summary["read_by"]["greedy/strict_object"] == 2


# ---------------------------------------------------------------- the systems
def moments_for(*entries) -> list[EvalMoment]:
    return [
        EvalMoment(moment_id=moment, slide_ocr=slide, type="figure") for moment, slide in entries
    ]


def test_the_four_kinds_of_system():
    assert DESCRIPTION_SYSTEMS.names() == (
        "adapter",
        "reference_writer",
        "slide_title",
        "zero_shot",
    )


def test_the_readout_answers_every_moment_with_the_title_of_its_slide():
    readout = SlideTitleReadout()
    answers = readout.describe(
        moments_for(
            ("lec-a#0001", "Action Potential: phases\nSodium influx"),
            ("lec-a#0002", "Threshold crossing"),
        )
    )
    assert [prediction.ad_text for prediction in answers] == [
        "Action Potential",
        "Threshold crossing",
    ]
    assert all(prediction.forced for prediction in answers)


def test_the_readout_of_a_slide_with_no_text_says_nothing():
    answers = SlideTitleReadout().describe(moments_for(("lec-a#0001", "(no slide text detected)")))
    assert answers[0].ad_text is None and not answers[0].filled


def test_the_title_is_cut_to_a_clause_and_a_length():
    assert slide_title("A very long title that runs on and on and on and on", most_words=5) == (
        "A very long title that"
    )
    assert slide_title("Title - subtitle") == "Title"


def test_a_reference_writer_answers_the_moments_it_chose():
    writer = ReferenceWriter("writer-x", {"lec-a#0001": "Two curves cross."})
    moments = moments_for(("lec-a#0001", "x"), ("lec-a#0002", "y"))
    answers = writer.describe(moments)
    assert answers[0].filled and not answers[1].filled
    assert not writer.forced and not writer.on_shared_moments
    assert [moment.moment_id for moment in writer.own_moments(moments)] == ["lec-a#0001"]


def test_a_trained_describer_is_asked_the_way_it_was_trained():
    asked = []

    def generate(messages, keyframes, settings):
        asked.append((messages, settings))
        return 'Two curves cross.", "rung": 3}'

    describer = TrainedDescriber(
        generate,
        build_arm("text_only"),
        [DECODING_ATTEMPTS.create("greedy")],
        steps(),
        name="system-x",
    )
    answers = describer.describe(moments_for(("lec-a#0001", "Threshold")))
    assert answers[0].ad_text == "Two curves cross." and answers[0].forced
    assert answers[0].how == "greedy/strict_object"
    messages, _settings = asked[0]
    assert [message["role"] for message in messages] == ["system", "user"]
    assert messages[0]["content"] == system_prompt()


def test_a_moment_the_first_way_does_not_answer_is_asked_again():
    replies = ['", "rung": 3}', 'Recovered on the second try.", "rung": 3}']

    def generate(messages, keyframes, settings):
        return replies.pop(0)

    describer = TrainedDescriber(
        generate,
        build_arm("text_only"),
        [DECODING_ATTEMPTS.create("greedy"), DECODING_ATTEMPTS.create("sampled", attempt=1)],
        steps(),
    )
    answers = describer.describe(moments_for(("lec-a#0001", "Threshold")))
    assert answers[0].ad_text == "Recovered on the second try."
    assert answers[0].how.startswith("sampled/")


def test_a_moment_no_way_answers_is_recorded_as_unanswered():
    describer = TrainedDescriber(
        lambda messages, keyframes, settings: '", "rung": 3}',
        build_arm("text_only"),
        [DECODING_ATTEMPTS.create("greedy")],
        steps(),
    )
    answers = describer.describe(moments_for(("lec-a#0001", "Threshold")))
    assert not answers[0].filled and answers[0].how == "unrecovered"


def test_a_describer_needs_a_way_of_being_asked():
    with pytest.raises(ValueError):
        TrainedDescriber(lambda *_: "", build_arm("text_only"), [], steps())


def test_the_conversation_is_the_training_one_without_the_answer():
    moment = EvalMoment(
        moment_id="lec-a#0001",
        type="figure",
        slide_ocr="Threshold",
        what_on_screen="A plot",
        transcript_window="here they meet",
        pause_after=1.5,
        keyframes=(Path("keyframes/lec-a/000017.jpg"),),
    )
    with_pictures = conversation(moment, build_arm("multimodal"))
    without = conversation(moment, build_arm("text_only"))
    assert with_pictures[1]["content"][0]["type"] == "image"
    assert all(part["type"] == "text" for part in without[1]["content"])
    assert "A plot" in without[1]["content"][0]["text"]
    assert "A plot" not in with_pictures[1]["content"][-1]["text"]
