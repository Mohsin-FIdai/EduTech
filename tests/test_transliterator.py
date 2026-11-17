"""
test_transliterator.py — Unit tests for roman_urdu_nlp.transliterator.transliterate()

Run with: pytest tests/test_transliterator.py -v

Mocking note: transliterate() receives a *corrected* string.
Mock: correct() = identity (pass pre-corrected strings directly).
"""

import pytest
from roman_urdu_nlp.transliterator import transliterate


class TestReturnType:
    def test_returns_str(self):
        assert isinstance(transliterate("hai"), str)

    def test_returns_urdu_script(self):
        result = transliterate("hai")
        # must contain at least one Urdu/Arabic Unicode character
        assert any("\u0600" <= ch <= "\u06FF" for ch in result)


class TestWordMapLookup:
    def test_mujhe(self):
        assert transliterate("mujhe") == "مجھے"

    def test_bohat(self):
        assert transliterate("bohat") == "بہت"

    def test_hai(self):
        assert transliterate("hai") == "ہے"

    def test_nahi(self):
        assert transliterate("nahi") == "نہیں"

    def test_aaj(self):
        assert transliterate("aaj") == "آج"

    def test_yaar(self):
        assert transliterate("yaar") == "یار"

    def test_exam(self):
        assert transliterate("exam") == "امتحان"

    def test_traffic(self):
        assert transliterate("traffic") == "ٹریفک"


class TestFullSentences:
    def test_hunger_sentence(self):
        result = transliterate("mujhe bohat bhook lagi hai")
        assert "مجھے" in result
        assert "بہت" in result
        assert "بھوک" in result
        assert "ہے" in result

    def test_tired_sentence(self):
        result = transliterate("bohat thak gaya hoon aaj")
        assert "بہت" in result
        assert "آج" in result

    def test_question_sentence(self):
        result = transliterate("kya haal hai bhai")
        assert "کیا" in result
        assert "بھائی" in result

    def test_gold_sleepy(self):
        result = transliterate("mujhe neend aa rahi hai yaar")
        assert "مجھے" in result
        assert "نیند" in result
        assert "یار" in result


class TestPhoneticFallback:
    def test_unknown_word_not_empty(self):
        result = transliterate("zubair")
        assert len(result) > 0

    def test_phonetic_has_urdu_chars(self):
        result = transliterate("khwaab")
        assert any("\u0600" <= ch <= "\u06FF" for ch in result)


class TestEdgeCases:
    def test_empty_returns_empty(self):
        assert transliterate("") == ""

    def test_whitespace_returns_empty(self):
        assert transliterate("   ") == ""

    def test_none_raises_typeerror(self):
        with pytest.raises(TypeError):
            transliterate(None)  # type: ignore[arg-type]

    def test_single_known_word(self):
        assert transliterate("aur") == "اور"

    def test_multiple_words_joined_with_space(self):
        result = transliterate("aur bhi")
        parts = result.split()
        assert len(parts) == 2
