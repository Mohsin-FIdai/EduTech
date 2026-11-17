"""
test_pipeline.py — Unit tests for roman_urdu_nlp.pipeline.Pipeline

Run with: pytest tests/test_pipeline.py -v
"""

import pytest
from unittest.mock import patch
from roman_urdu_nlp.pipeline import Pipeline, PipelineResult


@pytest.fixture
def nlp():
    return Pipeline()


@pytest.fixture
def mock_translate():
    """Patch the Google Translate API for all pipeline tests."""
    with patch("roman_urdu_nlp.translator.GoogleTranslator") as mock_gt:
        mock_gt.return_value.translate.return_value = "I am very hungry."
        yield mock_gt


class TestPipelineResult:
    def test_result_has_normalized(self, nlp, mock_translate):
        result = nlp("Muje boht bhok lgi h")
        assert hasattr(result, "normalized")

    def test_result_has_urdu(self, nlp, mock_translate):
        result = nlp("Muje boht bhok lgi h")
        assert hasattr(result, "urdu")

    def test_result_has_english(self, nlp, mock_translate):
        result = nlp("Muje boht bhok lgi h")
        assert hasattr(result, "english")

    def test_result_has_stages(self, nlp, mock_translate):
        result = nlp("Muje boht bhok lgi h")
        assert hasattr(result, "stages")
        assert isinstance(result.stages, dict)

    def test_result_type(self, nlp, mock_translate):
        result = nlp("Muje boht bhok lgi h")
        assert isinstance(result, PipelineResult)


class TestStagesPresent:
    def test_stages_keys(self, nlp, mock_translate):
        result = nlp("mujhe bohat bhook lagi hai")
        expected_keys = {"cleaned", "segments", "normalized", "corrected", "urdu", "english"}
        assert expected_keys.issubset(result.stages.keys())

    def test_cleaned_stage_has_no_emoji(self, nlp, mock_translate):
        result = nlp("bohat acha hai 😂😂")
        assert "😂" not in result.stages["cleaned"]

    def test_segments_is_list(self, nlp, mock_translate):
        result = nlp("mujhe bohat bhook lagi hai")
        assert isinstance(result.stages["segments"], list)


class TestNormalization:
    def test_hunger_normalized(self, nlp, mock_translate):
        result = nlp("Muje boht bhok lgi h 😂😂")
        assert "Mujhe" in result.normalized or "mujhe" in result.normalized
        assert "bohat" in result.normalized
        assert "lagi" in result.normalized

    def test_sleepy_normalized(self, nlp, mock_translate):
        result = nlp("mjy neend aa rhi h yaar 😴😴")
        assert "mujhe" in result.normalized
        assert "rahi" in result.normalized
        assert "hai" in result.normalized


class TestUrduOutput:
    def test_urdu_contains_arabic_chars(self, nlp, mock_translate):
        result = nlp("mujhe bohat bhook lagi hai")
        assert any("\u0600" <= ch <= "\u06FF" for ch in result.urdu)

    def test_urdu_not_empty_for_valid_input(self, nlp, mock_translate):
        result = nlp("kal exam hai")
        assert result.urdu != ""


class TestEdgeCases:
    def test_empty_string(self, nlp):
        result = nlp("")
        assert result.normalized == ""
        assert result.urdu == ""
        assert result.english == ""

    def test_pure_emoji(self, nlp):
        result = nlp("😂😂😂")
        assert result.normalized == ""
        assert result.urdu == ""

    def test_none_raises_typeerror(self, nlp):
        with pytest.raises(TypeError):
            nlp(None)  # type: ignore[arg-type]

    def test_pure_english(self, nlp, mock_translate):
        result = nlp("hello how are you")
        assert result.normalized  # something returned
        assert isinstance(result.english, str)


class TestCallable:
    def test_pipeline_is_callable(self, nlp):
        assert callable(nlp)

    def test_pipeline_repr(self, nlp, mock_translate):
        result = nlp("hai")
        assert "PipelineResult" in repr(result)
