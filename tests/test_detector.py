"""
test_detector.py — Unit tests for roman_urdu_nlp.detector.detect_segments()

Run with: pytest tests/test_detector.py -v

Mocking note: The detector only requires a *cleaned* string as input.
We do NOT import cleaner here — the mock is simply passing a pre-cleaned
string literal, which is equivalent to clean()'s contract output.
"""

import pytest
from roman_urdu_nlp.detector import detect_segments


class TestReturnType:
    def test_returns_list(self):
        result = detect_segments("hello world")
        assert isinstance(result, list)

    def test_each_element_is_tuple(self):
        result = detect_segments("mujhe bhook lagi hai")
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2

    def test_lang_values_valid(self):
        result = detect_segments("meeting kal hai")
        valid = {"roman_urdu", "english"}
        for _, lang in result:
            assert lang in valid


class TestPureRomanUrdu:
    def test_pure_roman_urdu(self):
        result = detect_segments("mujhe bohat bhook lagi hai")
        assert all(lang == "roman_urdu" for _, lang in result)

    def test_reconstructed_text_matches_input(self):
        text = "yaar kahan ho tum"
        result = detect_segments(text)
        reconstructed = " ".join(seg for seg, _ in result)
        assert reconstructed == text

    def test_single_roman_word(self):
        result = detect_segments("yaar")
        assert result == [("yaar", "roman_urdu")]


class TestPureEnglish:
    def test_pure_english_greeting(self):
        result = detect_segments("hello how are you")
        assert all(lang == "english" for _, lang in result)

    def test_tech_word_english(self):
        result = detect_segments("download")
        assert result == [("download", "english")]


class TestCodeMixed:
    def test_meeting_word_flagged_english(self):
        result = detect_segments("meeting kal hai")
        langs = {seg: lang for seg, lang in result}
        # "meeting" should be english
        assert any("meeting" in seg and lang == "english" for seg, lang in result)
        # "kal hai" or "kal" should be roman_urdu
        assert any(lang == "roman_urdu" for _, lang in result)

    def test_mixed_tech(self):
        result = detect_segments("phone hang ho gaya hai")
        # "phone" → english, rest → roman_urdu
        assert result[0] == ("phone", "english")
        assert result[-1][1] == "roman_urdu"

    def test_segments_cover_all_words(self):
        text = "project submit karna hai kal tak deadline hai"
        result = detect_segments(text)
        all_words = " ".join(seg for seg, _ in result).split()
        assert all_words == text.split()


class TestEdgeCases:
    def test_empty_returns_empty_list(self):
        assert detect_segments("") == []

    def test_whitespace_only_returns_empty(self):
        assert detect_segments("   ") == []

    def test_none_raises_typeerror(self):
        with pytest.raises(TypeError):
            detect_segments(None)  # type: ignore[arg-type]

    def test_single_word_roman(self):
        result = detect_segments("aaj")
        assert len(result) == 1
        assert result[0][1] == "roman_urdu"

    def test_roman_urdu_markers_not_english(self):
        # "the" is in English list but also in Roman Urdu as copula — low priority
        # we test real Roman Urdu markers are tagged correctly
        for word in ["hai", "kya", "mujhe", "bohat", "nahi"]:
            result = detect_segments(word)
            assert result[0][1] == "roman_urdu", f"{word!r} should be roman_urdu"


class TestGoldSetSubset:
    def test_hunger_sentence_mostly_roman(self):
        result = detect_segments("Muje boht bhok lgi h")
        roman_count = sum(1 for _, l in result if l == "roman_urdu")
        assert roman_count >= 1

    def test_mixed_office_sentence(self):
        result = detect_segments("meeting kal hai office mein prepare karo")
        langs = [lang for _, lang in result]
        # should have both languages
        assert "english" in langs
        assert "roman_urdu" in langs
