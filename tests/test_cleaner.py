"""
test_cleaner.py — Unit tests for roman_urdu_nlp.cleaner.clean()

Run with: pytest tests/test_cleaner.py -v
"""

import pytest
from roman_urdu_nlp.cleaner import clean


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------

class TestEmojiRemoval:
    def test_removes_face_emoji(self):
        assert clean("Muje boht bhok lgi h 😂😂") == "Muje boht bhok lgi h"

    def test_removes_sweat_emoji(self):
        # "yarrr" → triple-collapse → "yarr" (stops at 2, not 1)
        assert clean("aaj garmi boht zyada h yarrr 🥵🥵🥵") == "aaj garmi boht zyada h yarr"

    def test_removes_sleep_emoji(self):
        assert clean("mjy neend aa rhi h yaar 😴😴") == "mjy neend aa rhi h yaar"

    def test_pure_emoji_returns_empty(self):
        assert clean("😂😂😂") == ""

    def test_removes_anxiety_emoji(self):
        result = clean("kal exam hai aur maine kuch nhi parha 😰")
        assert "😰" not in result


class TestRepeatedCharCollapse:
    def test_triple_r_collapsed(self):
        # "yarrr" → "yarr" (triple collapsed to double)
        result = clean("yarrr")
        assert result == "yarr"

    def test_quad_r_collapsed_to_double(self):
        result = clean("yarrrrr")
        assert result == "yarr"

    def test_double_preserved(self):
        # meaningful doubles like "aa" must survive
        assert clean("baahar") == "baahar"

    def test_triple_exclaim_noise(self):
        # "bohhht" → "bohht"
        result = clean("bohhht")
        assert result == "bohht"


class TestLaughCollapse:
    def test_long_laugh_collapsed(self):
        result = clean("hahahaha")
        assert result == "haha"

    def test_triple_ha_collapsed(self):
        result = clean("hahaha")
        assert result == "haha"

    def test_double_ha_preserved(self):
        result = clean("haha")
        assert result == "haha"

    def test_laugh_in_sentence(self):
        result = clean("hahahaha ye boht funny tha yaar 😂😂😂")
        assert result == "haha ye boht funny tha yaar"


class TestPunctuationNoise:
    def test_triple_question_marks(self):
        # after char-collapse ??? → ?? → ? then strip removes all trailing punct
        result = clean("kya haal hai bhai??? sab theek thaak??")
        assert "?" not in result

    def test_triple_exclamation(self):
        result = clean("aaj khana boht acha tha yar seriously!!")
        assert "!" not in result

    def test_inline_punctuation_stripped(self):
        # standalone punctuation cluster removed
        result = clean("hello... world")
        assert "..." not in result


class TestWhitespaceNormalisation:
    def test_multiple_spaces_collapsed(self):
        result = clean("boht   thak   gaya")
        assert result == "boht thak gaya"

    def test_leading_trailing_stripped(self):
        result = clean("  hello world  ")
        assert result == "hello world"

    def test_newline_collapsed(self):
        result = clean("line one\nline two")
        assert result == "line one line two"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_string_returns_empty(self):
        assert clean("") == ""

    def test_none_raises_typeerror(self):
        with pytest.raises(TypeError):
            clean(None)  # type: ignore[arg-type]

    def test_integer_raises_typeerror(self):
        with pytest.raises(TypeError):
            clean(42)  # type: ignore[arg-type]

    def test_already_clean_unchanged(self):
        text = "mujhe bohat bhook lagi hai"
        assert clean(text) == text

    def test_pure_english_unchanged(self):
        text = "hello world"
        assert clean(text) == text


# ---------------------------------------------------------------------------
# Gold-set subset — cleaner stage only
# ---------------------------------------------------------------------------

class TestGoldSetSubset:
    """Verify that cleaning doesn't destroy meaningful content."""

    def test_gold_hunger(self):
        result = clean("Muje boht bhok lgi h 😂😂")
        # core words must survive
        assert "Muje" in result
        assert "boht" in result
        assert "bhok" in result

    def test_gold_exam(self):
        result = clean("kal exam hai aur maine kuch nhi parha 😰")
        assert "exam" in result
        assert "parha" in result
        assert "😰" not in result

    def test_gold_traffic(self):
        result = clean("karachi ka traffic boht bura hai yaar seriously")
        assert "karachi" in result
        assert "traffic" in result
