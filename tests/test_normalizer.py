"""
test_normalizer.py — Unit tests for roman_urdu_nlp.normalizer.normalize()

Run with: pytest tests/test_normalizer.py -v

Mocking note: normalize() receives a *cleaned* string.
Mock: clean() = identity (we pass pre-cleaned strings directly).
"""

import pytest
from roman_urdu_nlp.normalizer import normalize


class TestPronounExpansion:
    def test_mjy_to_mujhe(self):
        assert normalize("mjy") == "mujhe"

    def test_mje_to_mujhe(self):
        assert normalize("mje") == "mujhe"

    def test_muje_to_mujhe(self):
        assert normalize("muje bohat bhook lagi hai") == "mujhe bohat bhook lagi hai"

    def test_capitalised_mjy(self):
        # First token capitalised → replacement capitalised
        assert normalize("Muje") == "Mujhe"


class TestAuxiliaryNormalization:
    def test_h_to_hai(self):
        assert normalize("bohat thak gaya h") == "bohat thak gaya hai"

    def test_rhi_to_rahi(self):
        assert normalize("aa rhi h") == "aa rahi hai"

    def test_rha_to_raha(self):
        assert normalize("kar rha hun") == "kar raha hoon"

    def test_hun_to_hoon(self):
        assert normalize("main yahan hun") == "main yahan hoon"


class TestNegation:
    def test_nhi_to_nahi(self):
        assert normalize("nhi") == "nahi"

    def test_ni_to_nahi(self):
        assert normalize("ni") == "nahi"

    def test_nhi_in_sentence(self):
        result = normalize("mujhe samajh nhi aata")
        assert "nahi" in result


class TestIntensifiers:
    def test_boht_to_bohat(self):
        assert normalize("boht") == "bohat"

    def test_bhut_to_bohat(self):
        assert normalize("bhut") == "bohat"

    def test_bahut_to_bohat(self):
        assert normalize("bahut") == "bohat"

    def test_zyda_to_zyada(self):
        assert normalize("zyda") == "zyada"


class TestAdjectives:
    def test_achha_to_acha(self):
        assert normalize("achha") == "acha"

    def test_accha_to_acha(self):
        assert normalize("acca") == "acca"  # unknown → unchanged

    def test_thik_to_theek(self):
        assert normalize("thik") == "theek"


class TestNounNormalization:
    def test_bhok_to_bhook(self):
        assert normalize("bhok") == "bhook"

    def test_yar_to_yaar(self):
        assert normalize("yar") == "yaar"


class TestAbbreviations:
    def test_plz_to_please(self):
        assert normalize("plz") == "please"

    def test_pls_to_please(self):
        assert normalize("pls") == "please"

    def test_fir_to_phir(self):
        assert normalize("fir") == "phir"

    def test_pe_to_par(self):
        assert normalize("pe") == "par"

    def test_lol_to_haha(self):
        assert normalize("lol") == "haha"


class TestWholesentence:
    def test_gold_hunger(self):
        result = normalize("Muje boht bhok lgi h")
        assert "Mujhe" in result
        assert "bohat" in result
        assert "bhook" in result
        assert "lagi" in result
        assert "hai" in result

    def test_gold_sleepy(self):
        result = normalize("mjy neend aa rhi h yaar")
        assert "mujhe" in result
        assert "rahi" in result
        assert "hai" in result

    def test_gold_exam(self):
        result = normalize("kal exam hai aur maine kuch nhi parha")
        assert "nahi" in result

    def test_already_normalized_unchanged(self):
        text = "mujhe bohat bhook lagi hai"
        assert normalize(text) == text


class TestEdgeCases:
    def test_empty_returns_empty(self):
        assert normalize("") == ""

    def test_whitespace_only_returns_empty(self):
        assert normalize("   ") == ""

    def test_none_raises_typeerror(self):
        with pytest.raises(TypeError):
            normalize(None)  # type: ignore[arg-type]

    def test_unknown_words_preserved(self):
        assert normalize("biryani") == "biryani"

    def test_pure_english_preserved(self):
        assert normalize("exam deadline project") == "exam deadline project"
