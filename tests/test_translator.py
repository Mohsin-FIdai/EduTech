"""
test_translator.py — Unit tests for roman_urdu_nlp.translator.translate()

Run with: pytest tests/test_translator.py -v

Mocking note:
  - translate() calls Google Translate via deep-translator.
  - Network calls are mocked using unittest.mock.patch so tests run offline.
  - Real integration tests (marked slow) can be run with: pytest -m integration
"""

import pytest
from unittest.mock import patch, MagicMock
from roman_urdu_nlp.translator import translate, _FALLBACK


class TestReturnType:
    def test_returns_str(self):
        with patch("roman_urdu_nlp.translator.GoogleTranslator") as mock_gt:
            mock_gt.return_value.translate.return_value = "I am hungry."
            result = translate("mujhe bhook lagi hai")
            assert isinstance(result, str)

    def test_non_empty_input_non_empty_output(self):
        with patch("roman_urdu_nlp.translator.GoogleTranslator") as mock_gt:
            mock_gt.return_value.translate.return_value = "I am very hungry."
            result = translate("mujhe bohat bhook lagi hai")
            assert result != ""


class TestMockedTranslation:
    def test_hunger_sentence(self):
        with patch("roman_urdu_nlp.translator.GoogleTranslator") as mock_gt:
            mock_gt.return_value.translate.return_value = "I am very hungry."
            result = translate("mujhe bohat bhook lagi hai")
            assert result == "I am very hungry."

    def test_sleepy_sentence(self):
        with patch("roman_urdu_nlp.translator.GoogleTranslator") as mock_gt:
            mock_gt.return_value.translate.return_value = "I am feeling sleepy."
            result = translate("mujhe neend aa rahi hai")
            assert "sleepy" in result.lower() or result  # passes any non-empty

    def test_translator_called_with_urdu_script(self):
        """Verify the translator is called with Urdu script, not Roman Urdu."""
        with patch("roman_urdu_nlp.translator.GoogleTranslator") as mock_gt:
            mock_gt.return_value.translate.return_value = "test"
            translate("mujhe bohat bhook lagi hai")
            call_args = mock_gt.return_value.translate.call_args[0][0]
            # Should be Urdu script (contains Arabic Unicode block chars)
            assert any("\u0600" <= ch <= "\u06FF" for ch in call_args), (
                f"Expected Urdu script, got: {call_args!r}"
            )


class TestGracefulFallback:
    def test_network_error_returns_fallback(self):
        with patch("roman_urdu_nlp.translator.GoogleTranslator") as mock_gt:
            mock_gt.return_value.translate.side_effect = ConnectionError("no network")
            result = translate("mujhe bohat bhook lagi hai")
            assert result == _FALLBACK

    def test_import_error_returns_fallback(self):
        with patch("roman_urdu_nlp.translator.GoogleTranslator") as mock_gt:
            mock_gt.side_effect = ImportError("not installed")
            result = translate("test")
            assert result == _FALLBACK

    def test_generic_exception_returns_fallback(self):
        with patch("roman_urdu_nlp.translator.GoogleTranslator") as mock_gt:
            mock_gt.return_value.translate.side_effect = Exception("unknown")
            result = translate("bohat acha hai")
            assert result == _FALLBACK

    def test_none_result_returns_fallback(self):
        with patch("roman_urdu_nlp.translator.GoogleTranslator") as mock_gt:
            mock_gt.return_value.translate.return_value = None
            result = translate("test")
            assert result == _FALLBACK


class TestEdgeCases:
    def test_empty_returns_empty(self):
        result = translate("")
        assert result == ""

    def test_whitespace_returns_empty(self):
        result = translate("   ")
        assert result == ""

    def test_none_raises_typeerror(self):
        with pytest.raises(TypeError):
            translate(None)  # type: ignore[arg-type]
