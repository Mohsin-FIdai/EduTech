"""
urdu_transliteration.py — Perso-Arabic Urdu Script to Roman Urdu Transliterator.

Resolved Component 0 (Path A):
Whisper native ASR outputs Urdu speech in Perso-Arabic script.
This module transliterates Perso-Arabic Urdu into standard Roman Urdu (Latin script)
so that it can be consumed by roman_urdu_nlp and downstream intelligence pipelines.
"""

from __future__ import annotations
import regex
from roman_urdu_nlp.transliterator import _WORD_MAP

# Inverted canonical word dictionary: Urdu Script -> Roman Urdu
_URDU_TO_ROMAN_WORD_MAP: dict[str, str] = {
    v.strip(): k.strip()
    for k, v in _WORD_MAP.items()
}

# Common phrases / compound mappings
_COMPOUND_MAP: dict[str, str] = {
    "آپ کا": "aapka",
    "آپ کی": "aapki",
    "آپ کے": "aapke",
    "ان کا": "unka",
    "ان کی": "unki",
    "ان کے": "unke",
    "اس کا": "uska",
    "اس کی": "uski",
    "اس کے": "uske",
    "جس کا": "jiska",
    "جس کی": "jiski",
    "جس کے": "jiske",
    "کس کا": "kiska",
    "کس کی": "kiski",
    "کس کے": "kiske",
    "ان میں": "un mein",
    "اس میں": "is mein",
    "کیوں کہ": "kyunke",
    "کیونکہ": "kyunke",
    "ہوں گے": "honge",
    "ہوں گی": "hongi",
}

# Arabic/Urdu Punctuation mapping
_PUNCTUATION_MAP: dict[str, str] = {
    "۔": ". ",
    "،": ", ",
    "؟": "? ",
    "؛": "; ",
    "٪": "%",
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "–": "-",
    "—": " - ",
}

# Multi-character digraphs & aspirated consonants
_DIGRAPHS: list[tuple[str, str]] = [
    ("بھ", "bh"), ("پھ", "ph"), ("تھ", "th"), ("ٹھ", "th"),
    ("جھ", "jh"), ("چھ", "chh"), ("دھ", "dh"), ("ڈھ", "dh"),
    ("رھ", "rh"), ("ڑھ", "rh"), ("کھ", "kh"), ("گھ", "gh"),
    ("لھ", "lh"), ("مھ", "mh"), ("نھ", "nh"), ("وں", "on"),
    ("یں", "ain"), ("ئے", "aye"), ("ائی", "aai"), ("او", "ao"),
]

# Single character phonetic mapping
_CHAR_MAP: dict[str, str] = {
    "ا": "a", "آ": "aa", "ب": "b", "پ": "p", "ت": "t", "ٹ": "t",
    "ث": "s", "ج": "j", "چ": "ch", "ح": "h", "خ": "kh", "د": "d",
    "ڈ": "d", "ذ": "z", "ر": "r", "ڑ": "r", "ز": "z", "ژ": "zh",
    "س": "s", "ش": "sh", "ص": "s", "ض": "z", "ط": "t", "ظ": "z",
    "ع": "a", "غ": "gh", "ف": "f", "ق": "q", "ک": "k", "گ": "g",
    "ل": "l", "م": "m", "ن": "n", "ں": "n", "و": "o", "ہ": "h",
    "ھ": "h", "ۂ": "h", "ی": "i", "ے": "e", "ء": "'", "ئ": "e",
    "ؤ": "o", "ۃ": "t",
    "۔": ".", "،": ",", "؟": "?", "؛": ";", "٪": "%",
}

_URDU_SCRIPT_REGEX = regex.compile(r"[\p{Arabic}\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]+")


class UrduToRomanTransliterator:
    """Transliterator converting Perso-Arabic Urdu script to Roman Urdu."""

    @classmethod
    def is_urdu_script(cls, text: str) -> bool:
        """Check if text contains Perso-Arabic Urdu characters."""
        return bool(_URDU_SCRIPT_REGEX.search(text))

    @classmethod
    def transliterate_word(cls, word: str) -> str:
        """Transliterate a single Urdu word into Roman Urdu."""
        # 1. Direct dictionary lookup
        if word in _URDU_TO_ROMAN_WORD_MAP:
            return _URDU_TO_ROMAN_WORD_MAP[word]

        # 2. Phoneme-by-phoneme fallback for OOV words
        result = word
        for urdu_digraph, roman_digraph in _DIGRAPHS:
            result = result.replace(urdu_digraph, roman_digraph)

        chars = []
        for ch in result:
            chars.append(_CHAR_MAP.get(ch, ch))

        return "".join(chars)

    @classmethod
    def transliterate(cls, text: str) -> str:
        """
        Transliterate text containing Urdu script into Roman Urdu while
        preserving English words, punctuation, numbers, and spacing.
        """
        if not text or not text.strip():
            return text

        # Check for compounds first
        processed = text
        for comp_urdu, comp_roman in _COMPOUND_MAP.items():
            processed = processed.replace(comp_urdu, comp_roman)

        # Replace Arabic/Urdu punctuation marks with standard Latin equivalents
        for p_urdu, p_lat in _PUNCTUATION_MAP.items():
            processed = processed.replace(p_urdu, p_lat)

        tokens = regex.split(r"(\s+|[^\p{L}\p{N}]+)", processed)
        result_tokens: list[str] = []

        for token in tokens:
            if not token:
                continue
            if cls.is_urdu_script(token):
                result_tokens.append(cls.transliterate_word(token))
            else:
                result_tokens.append(token)

        final_text = "".join(result_tokens)
        # Clean up any double spaces introduced by punctuation replacement
        import re
        final_text = re.sub(r' +', ' ', final_text)
        return final_text.strip()
