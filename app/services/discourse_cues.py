"""
app/services/discourse_cues.py — Polarity-Aware Discourse Cue Classifier.

Classifies instructor hints into structured exam signals:
ExamHint:
  polarity: "affirmed" | "negated" | "hedged"
  cue_type: "exam" | "importance" | "memorize" | "marks" | "question"
  raw_quote: str (from RAW segment, never cleaned)
  segment_id: Any
  start_time: float
  end_time: float
  confidence: float
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal, Sequence
from app.config import EXAM_CUE_MIN_CONFIDENCE


# Spelling normalization map for cue matching (quote stays raw)
_SPELLING_VARIANTS = {
    "ayega": "aayega",
    "aaega": "aayega",
    "ayegi": "aayega",
    "aaegi": "aayega",
    "aaye": "aayega",
    "pepar": "paper",
    "zaruri": "zaroori",
    "zarori": "zaroori",
    "zroori": "zaroori",
    "me": "mein",
    "main": "mein",
    "bht": "bohat",
    "boht": "bohat",
}

# Negation tokens triggering polarity flip within +/- 4 words
_NEGATION_TOKENS = frozenset({"nahi", "nahin", "not", "never", "mat", "na", "no", "without", "baghair", "bina"})

# Hedge tokens triggering hedged polarity within +/- 4 words
_HEDGE_TOKENS = frozenset({"shayad", "maybe", "probably", "possibly", "ho_sakta", "lagta", "might", "could"})

# Cue category regex patterns (applied on normalized text)
_CUE_PATTERNS: list[tuple[Literal["exam", "importance", "memorize", "marks", "question"], re.Pattern, float]] = [
    # Explicit exam warning (tolerant of words like 'nahi' between 'mein' and 'aayega')
    ("exam", re.compile(r"\b(?:exam|paper|midterm|final|quiz)\s+(?:mein|in)\s+(?:[a-zA-Z0-9']+\s+){0,3}?(?:aayega|included|pucha|poocha)\b", re.IGNORECASE), 0.95),
    ("exam", re.compile(r"\b(?:important|zaroori)\s+for\s+(?:the\s+)?(?:exam|paper|test|quiz|midterm|finals)\b", re.IGNORECASE), 0.92),
    ("exam", re.compile(r"\b(?:yeh|ye|this)\s+(?:exam|paper)\s+ka\s+(?:part|hissa|question)\s+hai\b", re.IGNORECASE), 0.90),
    ("exam", re.compile(r"\b(?:exam|paper)\s+(?:point\s+of\s+view|ke\s+lihaaz\s+se)\b", re.IGNORECASE), 0.88),

    # Memorization command
    ("memorize", re.compile(r"\b(?:memorize|ratta|yaad\s+rakhein|yaad\s+rakhna|must\s+remember|keep\s+in\s+mind)\b", re.IGNORECASE), 0.85),
    ("memorize", re.compile(r"\b(?:formula|definition)\s+(?:yaad|memorize)\s+(?:karein|karna|honi\s+chahiye)\b", re.IGNORECASE), 0.88),

    # Importance declaration
    ("importance", re.compile(r"\b(?:very\s+important|extremely\s+important|bohat\s+zaroori|buhat\s+important|crucial\s+topic|core\s+concept)\b", re.IGNORECASE), 0.85),
    ("importance", re.compile(r"\b(?:most\s+frequently\s+asked|high\s+yield|key\s+takeaway)\b", re.IGNORECASE), 0.82),

    # Marks announcement (contextual: requires numbers or ke/ka/ki)
    ("marks", re.compile(r"\b(?:(\d{1,2})\s*(?:marks?|number)\s*(?:ka|ke|ki|for)?)\b", re.IGNORECASE), 0.86),
    ("marks", re.compile(r"\b(?:marks?|weightage)\s+(?:ke\s+lihaaz\s+se|allocation)\b", re.IGNORECASE), 0.80),

    # Typical question announcement
    ("question", re.compile(r"\b(?:standard|typical|expected|classic)\s+(?:exam\s+)?question\b", re.IGNORECASE), 0.82),
    ("question", re.compile(r"\b(?:long\s+question|short\s+question|frequently\s+asked\s+question)\b", re.IGNORECASE), 0.85),
]


@dataclass(frozen=True)
class ExamHint:
    polarity: Literal["affirmed", "negated", "hedged"]
    cue_type: Literal["exam", "importance", "memorize", "marks", "question"]
    raw_quote: str
    segment_id: Any
    start_time: float
    end_time: float
    confidence: float


class DiscourseCueClassifier:
    """Classifies discourse cues with strict polarity, context, and window gating."""

    @classmethod
    def _normalize_for_matching(cls, text: str) -> list[str]:
        words = re.findall(r"\b[a-zA-Z0-9']+\b", text.lower())
        return [_SPELLING_VARIANTS.get(w, w) for w in words]

    @classmethod
    def classify_segment(cls, segment: Any) -> list[ExamHint]:
        """
        Scans a raw segment for discourse cues, evaluating polarity within +/- 4 words.
        Returns a list of verified ExamHint records.
        """
        raw_text = getattr(segment, "english_text", None) or getattr(segment, "normalized_text", None) or getattr(segment, "text", "") or ""
        if not raw_text or len(raw_text.strip()) < 8:
            return []

        seg_id = getattr(segment, "id", None) or getattr(segment, "segment_index", 0)
        start_t = float(getattr(segment, "start_time", 0.0) or 0.0)
        end_t = float(getattr(segment, "end_time", start_t + 1.0) or (start_t + 1.0))

        norm_words = cls._normalize_for_matching(raw_text)
        norm_text = " ".join(norm_words)

        # Contextual suppression:
        # Bare 'test' preceded by 'unit' or 'a/b' is software/engineering, NOT exam
        if re.search(r"\b(?:unit|a\/b|hypothesis|chi|t)\s+test\b", norm_text):
            if not re.search(r"\b(?:exam|paper|midterm|final|quiz)\b", norm_text):
                return []

        # 'question marks' is punctuation, NOT exam marks
        if re.search(r"\bquestion\s+marks?\b", norm_text):
            return []

        hints: list[ExamHint] = []

        for cue_type, pat, base_conf in _CUE_PATTERNS:
            match = pat.search(norm_text)
            if not match:
                continue

            match_start, match_end = match.start(), match.end()
            words_before = norm_text[:match_start].split()
            words_after = norm_text[match_end:].split()
            matched_words = match.group(0).split()

            window_pre = set(words_before[-4:]) if words_before else set()
            window_post = set(words_after[:4]) if words_after else set()
            window_inside = set(matched_words)
            window_words = window_pre | window_post | window_inside

            # Polarity evaluation
            is_negated = bool(window_words & _NEGATION_TOKENS)
            is_hedged = bool(window_words & _HEDGE_TOKENS)

            if is_negated:
                polarity: Literal["affirmed", "negated", "hedged"] = "negated"
                conf = round(base_conf * 0.90, 2)
            elif is_hedged:
                polarity = "hedged"
                # Hedged retains confidence above 0.45 but reduced by half
                conf = round(base_conf * 0.50, 2)
            else:
                polarity = "affirmed"
                conf = round(base_conf, 2)

            # Accept if conf clears threshold OR if hedged (which is halved by definition)
            min_thresh = (EXAM_CUE_MIN_CONFIDENCE * 0.50) if polarity == "hedged" else EXAM_CUE_MIN_CONFIDENCE
            if conf >= min_thresh:
                hints.append(ExamHint(
                    polarity=polarity,
                    cue_type=cue_type,
                    raw_quote=raw_text.strip(),
                    segment_id=seg_id,
                    start_time=round(start_t, 2),
                    end_time=round(end_t, 2),
                    confidence=conf,
                ))

        return hints

    @classmethod
    def extract_cues(cls, segments: Sequence[Any]) -> list[ExamHint]:
        """Scans all segments and dedupes cues across adjacent utterances."""
        all_hints: list[ExamHint] = []
        seen_quotes: set[str] = set()

        for seg in segments:
            seg_hints = cls.classify_segment(seg)
            for h in seg_hints:
                quote_key = (h.raw_quote.lower(), h.cue_type, h.polarity)
                if quote_key not in seen_quotes:
                    seen_quotes.add(quote_key)
                    all_hints.append(h)

        return all_hints
