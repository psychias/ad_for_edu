"""Each component: its end points, its grading, and when it does not apply."""

from __future__ import annotations

import pytest

from ad_for_edu.compliance.components.base import COMPONENTS, Component
from ad_for_edu.compliance.components.deixis import DeixisResolution
from ad_for_edu.compliance.components.faithfulness import FaithfulnessNLI
from ad_for_edu.compliance.components.length import LengthBudgetRatio
from ad_for_edu.compliance.components.non_redundancy import (
    NonRedundancyEmbedding,
    credit_for_similarity,
)
from ad_for_edu.compliance.components.style import (
    PROPERTIES,
    StyleProperties,
    style_violations,
)
from ad_for_edu.compliance.components.terminology import TerminologyGrounding
from ad_for_edu.data.schema import CATEGORIES
from tests.fixtures.factory import FakeModels, context, empty_context


def test_every_category_has_exactly_one_component():
    categories = sorted(COMPONENTS.get(name).category for name in COMPONENTS.names())
    assert categories == sorted(CATEGORIES)


def test_a_component_outside_the_categories_cannot_be_defined():
    with pytest.raises(TypeError):

        class Wrong(Component):
            category = "redundancy"

            def score(self, text, moment):
                return 1.0


# ---------------------------------------------------------------- style
def test_a_plain_present_tense_description_is_fully_compliant():
    style = StyleProperties()
    assert style.score("A labelled diagram appears beside two curves.", context()) == 1.0


@pytest.mark.parametrize(
    ("text", "violated"),
    [
        ("You see two curves.", "second_person"),
        ("Two curves appeared on the axes.", "past_tense"),
        ("A circle is drawn around the peak.", "passive_voice"),
        ("An elegant diagram of the cell.", "evaluative"),
        ("The slide shows two curves.", "meta_reference"),
    ],
)
def test_each_property_is_detected_and_costs_half(text, violated):
    counts = style_violations(text)
    assert counts[violated] > 0
    assert [name for name in PROPERTIES if counts[name] > 0] == [violated]
    assert StyleProperties().score(text, context()) == 0.5


def test_two_violated_properties_score_zero_and_the_score_never_goes_below():
    style = StyleProperties()
    assert style.score("The slide shows what you drew.", context()) == 0.0
    assert style.score("You were shown an elegant slide; the slide shows it.", context()) == 0.0


def test_one_property_matched_many_times_counts_once():
    assert StyleProperties().score("You and you and yourself.", context()) == 0.5


def test_a_state_described_with_a_participle_is_not_a_passive():
    counts = style_violations("The axis is labeled time and the bars are colored blue.")
    assert counts["passive_voice"] == 0


def test_an_indefinite_noun_with_a_display_verb_is_not_meta_language():
    assert style_violations("A diagram shows the membrane.")["meta_reference"] == 0
    assert style_violations("The diagram shows the membrane.")["meta_reference"] == 1


def test_text_read_from_the_slide_in_quotes_is_not_a_violation():
    text = 'The first bullet reads "Do you feel tired?" in red.'
    assert all(count == 0 for count in style_violations(text).values())


def test_the_cost_per_property_is_validated():
    with pytest.raises(ValueError):
        StyleProperties(cost_per_property=0)


# ---------------------------------------------------------------- terminology
def test_terminology_is_the_share_of_terms_found_on_the_slide():
    moment = context(slide_ocr="Mitochondrion cristae matrix", what_on_screen="")
    terminology = TerminologyGrounding()
    assert terminology.score("Mitochondrion cristae", moment) == 1.0
    assert terminology.score("Mitochondrion chloroplast", moment) == 0.5
    assert terminology.score("Chloroplast thylakoid", moment) == 0.0


def test_terminology_ignores_case_and_the_delivery_prefix():
    moment = context(slide_ocr="MITOCHONDRION", what_on_screen="")
    assert TerminologyGrounding().score("[after] mitochondrion", moment) == 1.0


def test_a_no_content_marker_grounds_nothing():
    # The marker contains the words "slide" and "text"; neither may earn credit.
    assert TerminologyGrounding().score("slide text detected", empty_context()) == 0.0


def test_a_description_without_content_terms_is_not_penalised():
    assert TerminologyGrounding().score("It is an ox.", context()) == 1.0


# ---------------------------------------------------------------- length
def test_length_is_full_while_the_description_fits_the_budget():
    length = LengthBudgetRatio(words_per_minute=120, lag_cap_seconds=20)
    moment = context(pause_after=0.0)
    assert length.score(" ".join(["word"] * 40), moment) == 1.0  # 20 s of speech, 20 s budget
    assert length.score(" ".join(["word"] * 80), moment) == pytest.approx(0.5)


def test_a_longer_pause_raises_the_budget():
    length = LengthBudgetRatio(words_per_minute=120, lag_cap_seconds=20)
    text = " ".join(["word"] * 80)  # 40 s of speech
    assert length.score(text, context(pause_after=0.0)) == pytest.approx(0.5)
    assert length.score(text, context(pause_after=10.0)) == pytest.approx(0.75)


def test_a_missing_pause_counts_as_no_pause():
    length = LengthBudgetRatio(words_per_minute=120, lag_cap_seconds=20)
    assert length.score(" ".join(["word"] * 80), context(pause_after=None)) == pytest.approx(0.5)


def test_the_delivery_prefix_is_not_counted_as_a_word():
    length = LengthBudgetRatio(words_per_minute=60, lag_cap_seconds=2)
    assert length.score("[after] one two", context(pause_after=0.0)) == 1.0


# ---------------------------------------------------------------- deixis
def test_a_pointing_word_followed_by_its_target_is_resolved():
    moment = context(slide_ocr="Mitochondrion cristae", what_on_screen="")
    assert DeixisResolution().score("This mitochondrion has folded cristae.", moment) == 1.0


def test_an_unresolved_pointing_word_scores_zero():
    moment = context(slide_ocr="Mitochondrion cristae", what_on_screen="")
    assert DeixisResolution().score("Look at this one over here.", moment) == 0.0


def test_the_score_is_the_share_of_resolved_pointing_words():
    moment = context(slide_ocr="Mitochondrion cristae", what_on_screen="")
    text = "This mitochondrion is large, and that thing is small."
    assert DeixisResolution().score(text, moment) == 0.5


def test_the_target_must_follow_within_the_window():
    moment = context(slide_ocr="cristae", what_on_screen="")
    text = "This very large folded inner membrane structure called cristae"
    assert DeixisResolution(window=3).score(text, moment) == 0.0
    assert DeixisResolution(window=8).score(text, moment) == 1.0


def test_narrating_the_gesture_scores_zero():
    moment = context(slide_ocr="Mitochondrion", what_on_screen="")
    assert DeixisResolution().score("The cursor moves to the mitochondrion.", moment) == 0.0


def test_no_pointing_word_means_nothing_is_left_open():
    assert DeixisResolution().score("A mitochondrion with folded cristae.", context()) == 1.0


def test_with_nothing_on_the_slide_nothing_can_be_checked():
    assert DeixisResolution().score("Look at this one here.", empty_context()) == 1.0


@pytest.mark.parametrize(
    ("fact", "applies"), [(None, True), (True, True), (False, False)]
)
def test_deixis_applies_unless_the_lecture_is_stated_to_show_no_pointer(fact, applies):
    assert DeixisResolution().applies(context(renders_cursor=fact)) is applies


# ---------------------------------------------------------------- faithfulness
def test_faithfulness_is_one_minus_the_contradiction_probability():
    models = FakeModels()
    faithful = FaithfulnessNLI(models)
    assert faithful.score("The membrane is folded.", context()) == 1.0
    assert faithful.score("The membrane is not folded.", context()) == 0.0
    assert len(models.contradiction_calls) == 2


def test_without_a_premise_the_model_is_not_asked():
    models = FakeModels()
    assert FaithfulnessNLI(models).score("The membrane is not folded.", empty_context()) == 1.0
    assert models.contradiction_calls == []


def test_the_premise_carries_no_marker():
    models = FakeModels()
    moment = context(slide_ocr="(no slide text detected)", what_on_screen="A folded membrane")
    FaithfulnessNLI(models).score("A membrane.", moment)
    assert models.contradiction_calls == [("A folded membrane", "A membrane.")]


# ---------------------------------------------------------------- non-redundancy
@pytest.mark.parametrize(
    ("similarity", "expected"),
    [(0.0, 1.0), (0.17, 1.0), (0.585, 0.5), (1.0, 0.0), (1.2, 0.0)],
)
def test_credit_falls_linearly_above_the_knee(similarity, expected):
    assert credit_for_similarity(similarity, 0.17) == pytest.approx(expected)


def test_repeating_the_lecturer_scores_low():
    models = FakeModels()
    moment = context(transcript_window="the energy of the cell is produced here")
    component = NonRedundancyEmbedding(models, knee=0.17)
    assert component.score("the energy of the cell is produced here", moment) == 0.0
    assert component.score("Folded cristae fill the matrix.", moment) == 1.0


def test_without_a_transcript_window_the_model_is_not_asked():
    models = FakeModels()
    moment = context(transcript_window="")
    assert NonRedundancyEmbedding(models).score("Anything.", moment) == 1.0
    assert models.similarity_calls == []


def test_the_knee_is_validated():
    with pytest.raises(ValueError):
        NonRedundancyEmbedding(FakeModels(), knee=1.0)


@pytest.mark.parametrize("name", ["faithfulness_nli", "non_redundancy_embedding"])
def test_model_backed_components_say_so(name):
    assert COMPONENTS.get(name).requires_models


@pytest.mark.parametrize(
    "name",
    ["style_properties", "terminology_grounding", "length_budget_ratio", "deixis_resolution"],
)
def test_rule_based_components_need_no_model(name):
    assert not COMPONENTS.get(name).requires_models
