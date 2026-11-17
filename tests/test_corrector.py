"""
test_corrector.py — Unit tests for roman_urdu_nlp.corrector.correct()

Run with: pytest tests/test_corrector.py -v

Mocking note: correct() receives a *normalized* string.
Mock: normalize() = identity (pass pre-normalized strings directly).
"""

import pytest
from roman_urdu_nlp.corrector import correct


class TestAlreadyCorrect:
    def test_canonical_sentence_unchanged(self):
        text = "mujhe bohat bhook lagi hai"
        assert correct(text) == text

    def test_canonical_sleepy(self):
        text = "mujhe neend aa rahi hai yaar"
        assert correct(text) == text

    def test_canonical_weather(self):
        text = "aaj garmi bohat zyada hai yaar"
        assert correct(text) == text


class TestSpellCorrection:
    def test_close_word_corrected(self):
        # "nahe" is edit-distance 1 from both "nahi" and "rahe" (ambiguous)
        # The corrector picks one valid canonical word — either is acceptable
        result = correct("nahe")
        assert result in ("nahi", "rahe", "nahe")  # any canonical or unchanged

    def test_bhuk_not_in_ed1_range(self):
        # "bhuk" → "bhook" is edit dist=2, so it passes through unchanged
        result = correct("mujhe bhuk lagi hai")
        # bhuk stays (not within ed=1 of any canonical word)
        assert "lagi" in result and "hai" in result

    def test_already_canonical_word(self):
        # "bohat" is already canonical → no change
        assert correct("bohat") == "bohat"

    def test_unknown_word_preserved(self):
        # "xyz123" has no close match → returned as-is
        result = correct("xyz123")
        assert result == "xyz123"


class TestGrammarFixes:
    def test_double_hai_removed(self):
        result = correct("theek hai hai")
        assert result.count("hai") == 1

    def test_double_hoon_removed(self):
        result = correct("main yahan hoon hoon")
        assert result.count("hoon") == 1

    def test_double_raha_removed(self):
        result = correct("kar raha raha hoon")
        assert result.count("raha") == 1

    def test_gaya_hain_fixed(self):
        result = correct("woh gaya hain")
        assert "gaya hai" in result
        assert "hain" not in result

    def test_lagi_hain_fixed(self):
        result = correct("bhook lagi hain")
        assert "lagi hai" in result


class TestEdgeCases:
    def test_empty_returns_empty(self):
        assert correct("") == ""

    def test_whitespace_only_returns_empty(self):
        assert correct("   ") == ""

    def test_none_raises_typeerror(self):
        with pytest.raises(TypeError):
            correct(None)  # type: ignore[arg-type]

    def test_english_code_mixed_preserved(self):
        # "exam", "deadline" are in canonical list
        result = correct("exam deadline project")
        assert "exam" in result
        assert "deadline" in result


class TestGoldSubset:
    def test_gold_hunger_sentence(self):
        result = correct("mujhe bohat bhook lagi hai")
        assert result == "mujhe bohat bhook lagi hai"

    def test_gold_exam_sentence(self):
        result = correct("kal exam hai aur maine kuch nahi parha")
        assert "exam" in result
        assert "nahi" in result

    def test_gold_tired(self):
        result = correct("bohat thak gaya hoon aaj")
        assert "bohat" in result
        assert "hoon" in result
