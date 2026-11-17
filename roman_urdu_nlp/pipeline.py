"""
pipeline.py — Pipeline Integrator for roman_urdu_nlp.

Contract:
    Class    : Pipeline
    Method   : __call__(text: str) -> PipelineResult
    Input    : Raw Roman Urdu string (as user types it).
    Output   : PipelineResult dataclass with:
               - normalized : str  — canonical Roman Urdu
               - urdu       : str  — Urdu script
               - english    : str  — English translation
               - stages     : dict — intermediate outputs per stage
                              {"cleaned", "normalized", "corrected",
                               "segments", "urdu", "english"}

    Graceful degradation: no stage swallows exceptions silently.
    Empty/emoji-only input → all output fields are empty strings, stages
    contain the intermediate empty values.

Usage:
    from roman_urdu_nlp import Pipeline
    nlp = Pipeline()
    result = nlp("Muje boht bhok lgi h 😂😂")
    result.normalized  # "Mujhe bohat bhook lagi hai"
    result.urdu        # "مجھے بہت بھوک لگی ہے"
    result.english     # "I am very hungry."
"""

from __future__ import annotations
from dataclasses import dataclass, field

from roman_urdu_nlp.cleaner import clean
from roman_urdu_nlp.detector import detect_segments, tag_tokens
from roman_urdu_nlp.normalizer import normalize
from roman_urdu_nlp.corrector import correct
from roman_urdu_nlp.transliterator import transliterate
from roman_urdu_nlp.translator import translate


@dataclass
class PipelineResult:
    """
    Container for the output of the roman_urdu_nlp Pipeline.

    Attributes
    ----------
    normalized : str
        Canonical Roman Urdu after cleaning, normalization, and correction.
    urdu : str
        Urdu script transliteration.
    english : str
        English translation.
    stages : dict
        Intermediate values at each pipeline stage for debugging/evaluation.
    token_tags : list[dict]
        Fine-grained token-level language and technical tags.
    language_distribution : dict[str, float]
        Percentage breakdown of languages in text.
    """
    normalized: str
    urdu: str
    english: str
    stages: dict = field(default_factory=dict)
    token_tags: list[dict] = field(default_factory=list)
    language_distribution: dict[str, float] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"PipelineResult(\n"
            f"  normalized={self.normalized!r},\n"
            f"  urdu={self.urdu!r},\n"
            f"  english={self.english!r}\n"
            f")"
        )


class Pipeline:
    """
    The roman_urdu_nlp processing pipeline.

    Chains: clean → detect/tag → normalize → correct → transliterate / translate
    """

    def __call__(self, text: str, skip_translation: bool = False) -> PipelineResult:
        """
        Process a raw Roman Urdu string through the full pipeline.

        Parameters
        ----------
        text : str
            Raw input (may contain emojis, noise, abbreviations).
        skip_translation : bool, optional
            Whether to skip external English translation (defaults to False).

        Returns
        -------
        PipelineResult
            Structured result with .normalized, .urdu, .english, .stages, .token_tags, .language_distribution.

        Raises
        ------
        TypeError
            If ``text`` is not a ``str``.
        """
        if not isinstance(text, str):
            raise TypeError(f"Pipeline expects str, got {type(text).__name__!r}")

        stages: dict = {}

        # Stage 1: Clean
        cleaned = clean(text)
        stages["cleaned"] = cleaned

        # Stage 2: Detect language segments and token tags
        segments = detect_segments(cleaned)
        stages["segments"] = segments

        tokens_tagged = tag_tokens(cleaned)
        stages["token_tags"] = tokens_tagged

        # Compute language distribution
        total_valid = sum(1 for t in tokens_tagged if t["lang"] in ("ur", "en"))
        if total_valid > 0:
            ur_count = sum(1 for t in tokens_tagged if t["lang"] == "ur")
            en_count = sum(1 for t in tokens_tagged if t["lang"] == "en")
            lang_dist = {
                "ur": round(ur_count / total_valid, 2),
                "en": round(en_count / total_valid, 2)
            }
        else:
            lang_dist = {"ur": 1.0}
        stages["language_distribution"] = lang_dist

        # Stage 3: Normalize
        normalized = normalize(cleaned)
        stages["normalized"] = normalized

        # Stage 4: Correct
        corrected = correct(normalized)
        stages["corrected"] = corrected

        # Stage 5a: Transliterate → Urdu script
        urdu = transliterate(corrected)
        stages["urdu"] = urdu

        # Stage 5b: Translate → English (skipped if skip_translation=True)
        if skip_translation:
            english = ""
        else:
            english = translate(corrected)
        stages["english"] = english

        return PipelineResult(
            normalized=corrected,
            urdu=urdu,
            english=english,
            stages=stages,
            token_tags=tokens_tagged,
            language_distribution=lang_dist,
        )

