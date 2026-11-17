"""
app/services/intelligence_preprocessor.py — Hidden Intelligence Preprocessing Layer.

Generates a separate, persisted IntelligenceView artifact ("llm_text" + "match_text")
between the read-only display transcript tables and downstream intelligence modules.

Rules enforced:
- R1: User-visible transcript (text, normalized_text, english_text) is NEVER modified or overwritten.
- R2: Separate derived artifact consumed only by intelligence modules.
- R3: Five existing transcription tables remain read-only; stored in `intelligence_views`.
- R4: Fully dynamic (runtime topic_hint only; zero hardcoded domains).
- R5: Deterministic, rule-based (S0-S9), reusing roman_urdu_nlp primitives without LLM calls.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import logging
import re
import time
import unicodedata
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

from sqlalchemy import Column, DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Session

from app.config import BASE_DIR
from app.database import Base

logger = logging.getLogger(__name__)

PREPROCESSING_VERSION = "1.1.0"
ENGLISH_DOMINANT_THRESHOLD = 0.60
RETENTION_MIN_RATIO = 0.80
CHUNK_CONFIDENCE_TRUST_THRESHOLD = 0.3679  # exp(-1.0) Whisper avg_logprob < -1.0 threshold

# ---------------------------------------------------------------------------
# Lazy-loaded English Vocabulary, Roman Urdu Lexicon & Stopword Sets
# ---------------------------------------------------------------------------
_EN_VOCAB_CACHE: Optional[frozenset[str]] = None
_STOPWORDS_CACHE: Optional[frozenset[str]] = None
_RU_LEXICON_CACHE: Optional[frozenset[str]] = None

_MATCH_TEXT_KEEP_WORDS = frozenset({
    # English negations & quantifiers preserved in match_text (F3)
    "not", "no", "nor", "neither", "never", "none", "nothing", "nowhere", "nobody",
    "hardly", "scarcely", "barely", "without", "only", "all", "most", "every",
    "any", "few", "many", "much", "more", "less", "least", "both", "each",
    # Common contractions containing negation
    "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't",
    "won't", "wouldn't", "couldn't", "shouldn't", "can't", "cannot",
    "haven't", "hasn't", "hadn't", "mustn't", "needn't", "n't",
    # Roman Urdu negations (F3)
    "nahi", "nahin", "na", "naa", "mat", "bina", "baghair", "bagair",
})

_ROMAN_URDU_STOPWORDS = frozenset({
    "hai", "hain", "tha", "thi", "the", "hoga", "hogi", "honge", "hoon", "ho",
    "ka", "ke", "ki", "ko", "se", "mein", "me", "par", "pe", "ne", "tk", "tak",
    "aur", "ya", "lekin", "magar", "agar", "toh", "to", "phir", "bhi", "hi",
    "ye", "yeh", "wo", "woh", "is", "us", "in", "un", "jo", "jis", "jin",
    "kya", "kyun", "kyu", "kab", "kahan", "kaise", "koi", "kuch", "sab",
    "hum", "aap", "tum", "main", "mein", "mera", "meri", "mere", "apna",
    "wala", "wali", "wale", "jab", "tab", "ab", "abhi", "kar", "karna",
    "karte", "karti", "karta", "raha", "rahi", "rahe", "gaya", "gayi", "gaye",
    "acha", "theek", "han", "haan", "nahi", "na", "mat", "bht", "bohat",
})

_FUNCTION_WORD_STUTTER_COLLAPSE = frozenset({
    "ye", "yeh", "wo", "woh", "ka", "ki", "ke", "ko", "se", "mein", "hai", "hain",
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "in", "on", "for",
    "and", "or", "but", "that", "this", "it", "at", "by", "from", "with", "as",
    "so", "then", "if", "we", "you", "they", "i",
})

_LEGITIMATE_REDUPLICATIONS = frozenset({
    "bohat", "bahut", "alag", "dheere", "saath", "chote", "bare", "bade",
    "naya", "naye", "tez", "ahista", "jaldi", "ghar", "baat", "kabhi", "kuch",
    "thoda", "thora", "zyada", "kam", "seedha", "gol", "ek", "do",
})

_DISCOURSE_WORDS_KEEP = frozenset({
    "matlab", "yani", "basically", "actually", "so", "acha", "accha", "theek",
    "therefore", "however", "moreover", "furthermore", "because", "then",
})

_PURE_FILLERS_RE = re.compile(
    r"(?<!\w)(?:um+|uh+|uhh+|hmm+|erm+|ah+|aa+|ummm+)(?!\w)",
    re.IGNORECASE,
)

_SPELLED_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000, "lakh": 100000,
    "million": 1000000, "crore": 10000000,
}

_WHISPER_HALLUCINATION_EXACT_RE = re.compile(
    r"^\s*(?:"
    r"thanks?\s+(?:you\s+)?for\s+watching[.!]?|"
    r"please\s+(?:like\s+and\s+)?subscribe(?:\s+to\s+(?:my|our)\s+channel)?[.!]?|"
    r"\[(?:music|applause|laughter|silence|inaudible|noise)\]|"
    r"\((?:music|applause|laughter|silence|inaudible|noise)\)|"
    r"(?:https?://|www\.)\S+|"
    r"subtitles?\s+by\s+.*|"
    r"amara\.org\s+community"
    r")\s*$",
    re.IGNORECASE,
)

_LLM_REFUSAL_OR_GARBAGE_RE = re.compile(
    r"(?:"
    r"cannot\s+translate\s+(?:that|it|this)|"
    r"does\s+not\s+contain\s+any\s+text\s+to\s+translate|"
    r"please\s+provide\s+a\s+(?:complete|full|specific)\s+(?:roman\s+urdu\s+)?(?:lecture\s+)?sentence|"
    r"clarify\s+what\s+aspect\s+of\s+the\s+translation|"
    r"inappropriate\s+content\s+for\s+this\s+platform|"
    r"appears\s+to\s+be\s+in\s+a\s+language\s+other\s+than\s+urdu|"
    r"profits\s+on\s+the\s+fijian|"
    r"role\s+of\s+education\s+in\s+economic\s+development"
    r")",
    re.IGNORECASE,
)

_STITCH_CONNECTORS = frozenset({
    "and", "or", "but", "aur", "lekin", "magar", "kyunke", "because",
    "so", "to", "that", "which", "who", "where", "when", "while", "whereas",
    "for", "with", "into", "from", "by", "in", "on", "at", "then",
})

_RUNON_SPLIT_CONNECTORS = frozenset({
    "aur", "lekin", "phir", "because", "so", "but", "and", "however", "therefore",
})


def _get_english_vocab() -> frozenset[str]:
    """Load English dictionary from installed spacy WordNet lookups + detector + sklearn."""
    global _EN_VOCAB_CACHE
    if _EN_VOCAB_CACHE is not None:
        return _EN_VOCAB_CACHE

    from roman_urdu_nlp.detector import _ENGLISH_WORDS
    vocab: set[str] = set(_ENGLISH_WORDS)

    try:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        vocab.update(w.lower() for w in ENGLISH_STOP_WORDS)
    except Exception:
        pass

    try:
        import importlib.util
        import os
        import srsly

        spec = importlib.util.find_spec("en_core_web_sm")
        if spec and spec.origin:
            base = os.path.dirname(spec.origin)
            lbin = None
            for root, _dirs, files in os.walk(base):
                if "lookups.bin" in files and "lemmatizer" in root:
                    lbin = os.path.join(root, "lookups.bin")
                    break
            if lbin and os.path.exists(lbin):
                data = srsly.read_msgpack(lbin)
                for tbl_name in ("lemma_index", "lemma_exc"):
                    for _pos, words in data.get(tbl_name, {}).items():
                        vocab.update(w.lower() for w in words if isinstance(w, str) and w.isalpha() and len(w) >= 2)
    except Exception as e:
        logger.debug(f"en_core_web_sm lookups.bin vocab load fallback: {e}")

    _EN_VOCAB_CACHE = frozenset(vocab)
    return _EN_VOCAB_CACHE


def _get_roman_urdu_lexicon() -> frozenset[str]:
    """Union of roman_urdu_nlp vocabulary sets for conservative A1 revert safety (F2)."""
    global _RU_LEXICON_CACHE
    if _RU_LEXICON_CACHE is not None:
        return _RU_LEXICON_CACHE

    from roman_urdu_nlp.detector import _ROMAN_URDU_MARKERS
    from roman_urdu_nlp.normalizer import _NORM_MAP
    from roman_urdu_nlp.corrector import _CANONICAL_WORDS

    lex: set[str] = set(_ROMAN_URDU_MARKERS)
    lex.update(k.lower() for k in _NORM_MAP.keys())
    lex.update(v.lower() for v in _NORM_MAP.values())
    lex.update(w.lower() for w in _CANONICAL_WORDS)
    _RU_LEXICON_CACHE = frozenset(lex)
    return _RU_LEXICON_CACHE


def _get_all_stopwords() -> frozenset[str]:
    global _STOPWORDS_CACHE
    if _STOPWORDS_CACHE is not None:
        return _STOPWORDS_CACHE

    sw: set[str] = set(_ROMAN_URDU_STOPWORDS)
    try:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        sw.update(w.lower() for w in ENGLISH_STOP_WORDS)
    except Exception:
        sw.update({
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
            "of", "in", "on", "at", "to", "for", "with", "by", "from", "and",
            "or", "but", "if", "then", "so", "this", "that", "these", "those",
            "it", "its", "we", "you", "i", "they", "he", "she", "will", "can",
            "do", "does", "did", "have", "has", "had", "as", "into", "about",
        })
    # Preserve negations and quantifiers in match_text (F3)
    sw -= _MATCH_TEXT_KEEP_WORDS
    _STOPWORDS_CACHE = frozenset(sw)
    return _STOPWORDS_CACHE


def _is_in_english_vocab(word: str, en_vocab: frozenset[str]) -> bool:
    w = re.sub(r"[^a-zA-Z]", "", word).lower()
    if not w:
        return False
    if w in en_vocab:
        return True
    # Check standard English inflections not stored as lemmas in lemma_index
    if len(w) > 3 and w.endswith("s") and w[:-1] in en_vocab:
        return True
    if len(w) > 4 and w.endswith("es") and w[:-2] in en_vocab:
        return True
    if len(w) > 4 and w.endswith("ed") and (w[:-2] in en_vocab or w[:-1] in en_vocab):
        return True
    if len(w) > 5 and w.endswith("ing") and (w[:-3] in en_vocab or (w[:-3] + "e") in en_vocab):
        return True
    if len(w) > 4 and w.endswith("ly") and w[:-2] in en_vocab:
        return True
    return False


def _has_english_neighbor_confirmation(
    tokens: list[str],
    idx: int,
    en_vocab: frozenset[str],
) -> bool:
    """
    F2: Confirm whether up to 2 tokens on each side of `idx` form valid English (or numbers).
    Requires at least one neighbor token, and every available neighbor within window [idx-2..idx+2]
    to be valid English vocabulary or numeric.
    """
    num_re = re.compile(r"^\d+(?:[.,\-]\d+)*%?$")
    neighbors: list[str] = []
    for offset in (-2, -1, 1, 2):
        j = idx + offset
        if 0 <= j < len(tokens):
            tok_clean = tokens[j].strip(".,!?;:\"'()[]{}")
            if tok_clean:
                neighbors.append(tok_clean)
    if not neighbors:
        return False
    for nb in neighbors:
        if num_re.match(nb):
            continue
        base = nb.split("'")[0] if "'" in nb else nb
        if not _is_in_english_vocab(base, en_vocab):
            return False
    return True


# ---------------------------------------------------------------------------
# Database Model (R3: Separate table, zero changes to protected app/models.py)
# ---------------------------------------------------------------------------

class IntelligenceViewRecord(Base):
    __tablename__ = "intelligence_views"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    recording_id = Column(String(36), ForeignKey("recordings.id"), nullable=False, index=True)
    transcript_id = Column(String(36), ForeignKey("transcripts.id"), nullable=False, index=True)
    transcript_hash = Column(String(64), nullable=False, index=True)
    preprocessing_version = Column(String(16), nullable=False, default=PREPROCESSING_VERSION)
    topic_hint = Column(Text, nullable=True)

    segments_json = Column(Text, nullable=False)
    full_llm_text = Column(Text, nullable=False)
    full_match_text = Column(Text, nullable=False)
    rule_stats_json = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    __table_args__ = (
        Index("ix_intel_view_lookup", "recording_id", "transcript_hash", "preprocessing_version"),
    )


# ---------------------------------------------------------------------------
# Data Structures & Adapter
# ---------------------------------------------------------------------------

@dataclass
class IntelligenceSegment:
    intelligence_segment_id: int
    source_segment_ids: list[Any]
    start_time: float
    end_time: float
    llm_text: str
    match_text: str
    confidence: Optional[float]
    no_speech_prob: Optional[float]
    language: Optional[str] = None
    translation_source: Optional[str] = None
    source_choice: str = "repaired_source"
    retention_ratio: float = 1.0
    is_fallback: bool = False
    trust: str = "high"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class IntelligenceSegmentAdapter:
    """
    Duck-typed ORM-compatible adapter exposing IntelligenceSegment cleanly to
    existing downstream intelligence modules without altering display models.
    """

    def __init__(self, seg: IntelligenceSegment):
        self._seg = seg
        self.id = seg.source_segment_ids[0] if seg.source_segment_ids else seg.intelligence_segment_id
        self.segment_index = seg.intelligence_segment_id
        self.source_segment_ids = list(seg.source_segment_ids)
        self.start_time = seg.start_time
        self.end_time = seg.end_time
        self.start = seg.start_time
        self.end = seg.end_time
        self.text = seg.llm_text
        self.normalized_text = seg.llm_text
        self.english_text = seg.llm_text
        self.llm_text = seg.llm_text
        self.match_text = seg.match_text
        self.confidence = seg.confidence
        self.no_speech_prob = seg.no_speech_prob
        self.language = seg.language
        self.translation_source = seg.translation_source
        self.trust = seg.trust
        self.source_choice = seg.source_choice
        self.is_intel_view = True


@dataclass
class IntelligenceView:
    recording_id: str
    transcript_id: Optional[str]
    transcript_hash: str
    preprocessing_version: str
    segments: list[IntelligenceSegment]
    full_llm_text: str
    full_match_text: str
    rule_stats: dict[str, Any] = field(default_factory=dict)

    def to_adapters(self) -> list[IntelligenceSegmentAdapter]:
        return [IntelligenceSegmentAdapter(s) for s in self.segments]


# ---------------------------------------------------------------------------
# Core Preprocessing Pipeline (Steps S0 - S9)
# ---------------------------------------------------------------------------

class IntelligencePreprocessor:
    """
    Deterministic, rule-based Intelligence Preprocessing Layer (S0-S9).
    Strictly preserves user-visible display tables untouched (R1).
    """

    @classmethod
    def compute_display_transcript_hash(cls, segments: Sequence[Any]) -> str:
        """
        Hashes all display fields (id, text, normalized_text, english_text) across
        ordered segments so manual corrections to text or normalized_text invalidate cache.
        """
        h = hashlib.sha256()
        for s in segments:
            sid = getattr(s, "id", None) or getattr(s, "segment_index", 0)
            txt = getattr(s, "text", "") or ""
            norm = getattr(s, "normalized_text", "") or ""
            eng = getattr(s, "english_text", "") or ""
            h.update(f"{sid}\x1f{txt}\x1f{norm}\x1f{eng}\x1e".encode("utf-8"))
        return h.hexdigest()

    @classmethod
    def compute_english_dominance(
        cls,
        text: str,
        topic_hint: Optional[str] = None,
    ) -> tuple[bool, float]:
        """
        Uses roman_urdu_nlp.detector.tag_tokens() augmented with the project's
        installed English WordNet vocabulary (en_core_web_sm) & runtime topic terms.
        Threshold: ratio >= 0.60 -> English-dominant.
        """
        if not text or not text.strip():
            return False, 0.0

        from roman_urdu_nlp.detector import tag_tokens, _ROMAN_URDU_MARKERS

        en_vocab = _get_english_vocab()
        # Exclude dual-language homographs ("the", "is", "to", "main", "par", "no") from overriding English
        strict_ur = _ROMAN_URDU_MARKERS - {"the", "is", "to", "main", "par", "no", "in", "on", "me", "he", "be"}
        topic_tokens = {w.lower() for w in re.findall(r"[a-zA-Z]{2,}", topic_hint or "")}

        tags = tag_tokens(text)
        word_tags = [t for t in tags if t["lang"] in ("en", "ur")]
        if not word_tags:
            return False, 0.0

        en_count = 0
        for t in word_tags:
            raw_tok = t["token"]
            w = re.sub(r"[^a-zA-Z']", "", raw_tok).strip("'").lower()
            base_w = w.split("'")[0] if "'" in w else w
            if t["lang"] == "en":
                en_count += 1
            elif base_w not in strict_ur and (
                base_w in topic_tokens or _is_in_english_vocab(base_w, en_vocab)
            ):
                en_count += 1

        ratio = en_count / len(word_tags)
        return (ratio >= ENGLISH_DOMINANT_THRESHOLD), round(ratio, 4)

    @classmethod
    def _extract_numeric_tokens(cls, text: str) -> list[str]:
        """Extract formatted numeric tokens (e.g., 20,000, 1,000-20,000, 50%, 3.14)."""
        return re.findall(r"\b\d+(?:[.,\-]\d+)*%?", text or "")

    @classmethod
    def _all_numbers_present(cls, source_text: str, target_text: str) -> bool:
        """
        Verify every numeric value in source_text appears in target_text
        either verbatim/digit-normalized or as spelled-out number equivalent.
        """
        src_nums = cls._extract_numeric_tokens(source_text)
        if not src_nums:
            return True
        tgt_low = (target_text or "").lower()
        tgt_digits = [re.sub(r"[,]", "", n) for n in cls._extract_numeric_tokens(target_text)]

        # Also recognize spelled-out integers in target
        tgt_words = re.findall(r"\b[a-z]+\b", tgt_low)
        has_spelled_num = any(w in _SPELLED_NUMBERS for w in tgt_words)

        for num in src_nums:
            num_clean = re.sub(r"[,]", "", num)
            if num in target_text or num_clean in tgt_digits:
                continue
            # Handle simple spelled-out integers 0..20 or lakh/thousand when digits were spelled out
            if num_clean.isdigit() and int(num_clean) in _SPELLED_NUMBERS.values() and has_spelled_num:
                inv = [k for k, v in _SPELLED_NUMBERS.items() if v == int(num_clean)]
                if any(k in tgt_words for k in inv):
                    continue
            return False
        return True

    @classmethod
    def _has_content_word_overlap(cls, source_text: str, candidate_text: str) -> bool:
        """F1: True iff candidate_text shares at least one non-stopword content token with source_text."""
        sw = _get_all_stopwords()
        src_toks = {
            w for w in re.findall(r"[a-zA-Z0-9\u0600-\u06FF]+", (source_text or "").lower())
            if len(w) >= 2 and w not in sw
        }
        cand_toks = {
            w for w in re.findall(r"[a-zA-Z0-9\u0600-\u06FF]+", (candidate_text or "").lower())
            if len(w) >= 2 and w not in sw
        }
        return bool(src_toks & cand_toks)

    @classmethod
    def s0_source_repair(
        cls,
        raw_text: str,
        normalized_text: Optional[str],
        english_text: Optional[str],
        translation_source: Optional[str],
        topic_hint: Optional[str],
        previous_baseline: Optional[dict[str, str]],
        seg_id: Any,
        stats: dict[str, Any],
        chunk_confidence: Optional[float] = None,
        seg_confidence: Optional[float] = None,
    ) -> tuple[str, str, bool, str]:
        """
        S0: Align `text` vs `normalized_text` with difflib, repair numeric corruptions
        (e.g. 20,000 -> 20,00) and conservatively revert roman_urdu_nlp English word corruptions (F2).
        Selects source safely (F1) and returns:
          (selected_source_text, source_choice_label, is_english_dominant, trust_label)
        """
        stats.setdefault("s0_numeric_reverts", 0)
        stats.setdefault("s0_english_reverts", 0)
        stats.setdefault("s0_revert_samples", [])
        stats.setdefault("s0_unsure_reverts", [])
        stats.setdefault("source_choices", {})

        raw = (raw_text or "").strip()
        norm = (normalized_text or raw).strip()
        eng = (english_text or "").strip()

        en_vocab = _get_english_vocab()
        ru_lexicon = _get_roman_urdu_lexicon()
        is_en_dom, en_ratio = cls.compute_english_dominance(raw, topic_hint=topic_hint)

        raw_tokens = raw.split()
        norm_tokens = norm.split()

        repaired_tokens: list[str] = []
        sm = difflib.SequenceMatcher(None, raw_tokens, norm_tokens)
        num_re = re.compile(r"^\d+(?:[.,\-]\d+)*%?[.,!?;:]*$")

        def _ctx_window(pos: int) -> tuple[str, str]:
            before = " ".join(raw_tokens[max(0, pos - 5):pos])
            after = " ".join(raw_tokens[pos + 1:pos + 6])
            return before, after

        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op == "equal":
                repaired_tokens.extend(norm_tokens[j1:j2])
                continue

            orig_slice = raw_tokens[i1:i2]
            norm_slice = norm_tokens[j1:j2]

            if op == "replace" and len(orig_slice) == len(norm_slice):
                for rel_idx, (ot, nt) in enumerate(zip(orig_slice, norm_slice)):
                    tok_pos = i1 + rel_idx
                    ot_clean = ot.strip(".,!?;:\"'()[]{}")
                    nt_clean = nt.strip(".,!?;:\"'()[]{}")

                    # 1. Numeric preservation (mechanical, applies to ALL segments)
                    if num_re.match(ot) and ot_clean != nt_clean:
                        repaired_tokens.append(ot)
                        stats["s0_numeric_reverts"] += 1
                        cb, ca = _ctx_window(tok_pos)
                        stats["s0_revert_samples"].append({
                            "seg": seg_id, "orig": ot, "corrupt": nt, "type": "numeric",
                            "context_before": cb, "context_after": ca,
                        })
                        continue

                    # 2. English word corruption check (F2 conservative guard against Roman Urdu lexicon)
                    if ot_clean.lower() != nt_clean.lower() and _is_in_english_vocab(ot_clean, en_vocab):
                        in_ru_lex = nt_clean.lower() in ru_lexicon
                        if in_ru_lex:
                            # Do NOT auto-revert unless English-dominant AND neighbor 2 tokens each side form English
                            if is_en_dom and _has_english_neighbor_confirmation(raw_tokens, tok_pos, en_vocab):
                                repaired_tokens.append(ot)
                                stats["s0_english_reverts"] += 1
                                cb, ca = _ctx_window(tok_pos)
                                stats["s0_revert_samples"].append({
                                    "seg": seg_id, "orig": ot, "corrupt": nt, "type": "english",
                                    "context_before": cb, "context_after": ca,
                                })
                                continue
                            else:
                                cb, ca = _ctx_window(tok_pos)
                                stats["s0_unsure_reverts"].append({
                                    "seg": seg_id,
                                    "orig": ot,
                                    "corrupt": nt,
                                    "is_en_dom": is_en_dom,
                                    "context_before": cb,
                                    "context_after": ca,
                                    "reason": "corrupt_in_ru_lexicon_unconfirmed",
                                })
                                repaired_tokens.append(nt)
                                continue
                        elif is_en_dom:
                            repaired_tokens.append(ot)
                            stats["s0_english_reverts"] += 1
                            cb, ca = _ctx_window(tok_pos)
                            stats["s0_revert_samples"].append({
                                "seg": seg_id, "orig": ot, "corrupt": nt, "type": "english",
                                "context_before": cb, "context_after": ca,
                            })
                            continue

                    # 3. Restore stripped terminal punctuation if words otherwise identical
                    if is_en_dom and ot_clean.lower() == nt_clean.lower() and ot != nt and ot.endswith((".", "?", "!")):
                        repaired_tokens.append(ot)
                        continue

                    repaired_tokens.append(nt)
            else:
                # Non-1:1 diff (insert/delete/length mismatch): apply F2 lexicon + numeric rules carefully
                norm_cleans = {nt.strip(".,!?;:\"'()[]{}") for nt in norm_slice}
                has_missing_num = any(num_re.match(ot) and ot.strip(".,!?;:\"'()[]{}") not in norm_cleans for ot in orig_slice)
                norm_has_ru_lex = any(nc.lower() in ru_lexicon for nc in norm_cleans if nc)
                span_confirmed_en = is_en_dom and (
                    not norm_has_ru_lex
                    or all(_has_english_neighbor_confirmation(raw_tokens, i1 + k, en_vocab) for k in range(len(orig_slice)))
                )
                has_en_word = any(_is_in_english_vocab(ot, en_vocab) for ot in orig_slice)

                if has_missing_num or (span_confirmed_en and has_en_word):
                    repaired_tokens.extend(orig_slice)
                    for rel_idx, ot in enumerate(orig_slice):
                        tok_pos = i1 + rel_idx
                        cb, ca = _ctx_window(tok_pos)
                        ot_clean = ot.strip(".,!?;:\"'()[]{}")
                        if num_re.match(ot) and ot_clean not in norm_cleans:
                            stats["s0_numeric_reverts"] += 1
                            stats["s0_revert_samples"].append({
                                "seg": seg_id, "orig": ot, "corrupt": " ".join(norm_slice), "type": "numeric",
                                "context_before": cb, "context_after": ca,
                            })
                        elif _is_in_english_vocab(ot_clean, en_vocab):
                            stats["s0_english_reverts"] += 1
                            stats["s0_revert_samples"].append({
                                "seg": seg_id, "orig": ot, "corrupt": " ".join(norm_slice), "type": "english",
                                "context_before": cb, "context_after": ca,
                            })
                else:
                    if is_en_dom and has_en_word and norm_has_ru_lex:
                        cb = " ".join(raw_tokens[max(0, i1 - 5):i1])
                        ca = " ".join(raw_tokens[i2:i2 + 5])
                        stats["s0_unsure_reverts"].append({
                            "seg": seg_id,
                            "orig": " ".join(orig_slice),
                            "corrupt": " ".join(norm_slice),
                            "is_en_dom": is_en_dom,
                            "context_before": cb,
                            "context_after": ca,
                            "reason": "span_corrupt_in_ru_lexicon_unconfirmed",
                        })
                    repaired_tokens.extend(norm_slice if norm_slice else orig_slice)

        repaired_source = " ".join(repaired_tokens).strip() or raw

        # Manual Correction Detection (A2): check stored baseline for edits to text/normalized_text
        raw_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        norm_hash = hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]
        eng_hash = hashlib.sha256(eng.encode("utf-8")).hexdigest()[:16]
        manually_edited_since_baseline = False
        if previous_baseline:
            if (previous_baseline.get("raw_hash") != raw_hash or previous_baseline.get("norm_hash") != norm_hash) and (
                previous_baseline.get("eng_hash") == eng_hash
            ):
                manually_edited_since_baseline = True

        # Determine effective chunk/segment acoustic confidence for trust (F1)
        eff_conf = chunk_confidence if chunk_confidence is not None else seg_confidence
        low_chunk_conf = (eff_conf is not None and eff_conf < CHUNK_CONFIDENCE_TRUST_THRESHOLD)

        def _record_choice(choice: str, is_fallback: bool) -> tuple[str, str, bool, str]:
            stats["source_choices"][choice] = stats["source_choices"].get(choice, 0) + 1
            trust = "low" if (low_chunk_conf or is_fallback) else "high"
            return repaired_source if choice != "english_text_verified" else eng, choice, is_en_dom, trust

        # Source Choice Rule (A1 + F1)
        if is_en_dom:
            return _record_choice("repaired_english_dominant", is_fallback=False)

        if manually_edited_since_baseline:
            return _record_choice("repaired_manual_correction", is_fallback=False)

        if not eng or _LLM_REFUSAL_OR_GARBAGE_RE.search(eng):
            return _record_choice("repaired_fallback_no_valid_english", is_fallback=True)

        # F1 safety gates before allowing english_text substitution
        if low_chunk_conf:
            return _record_choice("repaired_fallback_low_chunk_conf", is_fallback=True)

        src_word_count = len(repaired_source.split())
        if src_word_count < 6:
            return _record_choice("repaired_fallback_short_segment", is_fallback=True)

        eng_word_count = len(eng.split())
        if abs(eng_word_count - src_word_count) / max(1, src_word_count) > 0.40:
            return _record_choice("repaired_fallback_length_mismatch", is_fallback=True)

        if not cls._all_numbers_present(repaired_source, eng):
            return _record_choice("repaired_fallback_number_mismatch", is_fallback=True)

        if not cls._has_content_word_overlap(repaired_source, eng):
            return _record_choice("repaired_fallback_no_content_overlap", is_fallback=True)

        return _record_choice("english_text_verified", is_fallback=False)

    @classmethod
    def s1_unicode_and_whitespace(cls, text: str, stats: dict[str, Any]) -> str:
        """S1: NFKC normalize, strip zero-width/control chars, unify quotes/dashes, collapse spaces."""
        if not text:
            return ""
        orig = text
        t = unicodedata.normalize("NFKC", text)
        # Strip zero-width & control chars (including replacement char \ufffd)
        t = re.sub(r"[\u200b\u200c\u200d\ufeff\u00ad\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ufffd]", "", t)
        # Unify quotes, dashes, ellipsis
        t = t.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
        t = t.replace("—", "-").replace("–", "-").replace("…", "...")
        t = re.sub(r"\s+", " ", t).strip()
        if t != orig:
            stats["rule_counts"]["S1_unicode_whitespace"] += 1
        return t

    @classmethod
    def s2_mask_protected_terms(
        cls,
        text: str,
        raw_text: str,
        topic_hint: Optional[str],
        stats: dict[str, Any],
        seg_idx: int = 0,
    ) -> tuple[str, dict[str, str]]:
        """
        S2: Extract & mask protected terms before edits:
          (a) runtime topic_hint terms,
          (b) ALL-CAPS acronyms,
          (c) CamelCase / snake_case code tokens,
          (d) tokens with digits / units,
          (e) mid-sentence capitalized multi-word English phrases (NOT sentence-initial).
        Uses globally unique placeholders (__PROT_TERM_{seg_idx}_{counter}__) so S6 merges never collide.
        """
        if not text:
            return "", {}

        cat_counts = stats.setdefault(
            "s2_category_counts",
            {
                "a_topic_hint": 0,
                "b_acronyms": 0,
                "c_code_tokens": 0,
                "d_digits_units": 0,
                "e_mid_sentence_caps": 0,
                "e_raw_phrase_recap": 0,
            },
        )
        masked = text
        term_map: dict[str, str] = {}
        counter = 0

        def _make_replacer(category: str):
            def _mask_match(m: re.Match) -> str:
                nonlocal counter
                val = m.group(0)
                if val.startswith("__PROT_TERM_") and val.endswith("__"):
                    return val
                key = f"__PROT_TERM_{seg_idx}_{counter}__"
                counter += 1
                term_map[key] = val
                cat_counts[category] = cat_counts.get(category, 0) + 1
                return key
            return _mask_match

        # (a) User runtime topic_hint multi-word or single >=3 char terms
        if topic_hint and topic_hint.strip():
            th = topic_hint.strip()
            topic_phrases = [th] + [w for w in re.findall(r"[A-Za-z0-9]+", th) if len(w) >= 3]
            for tp in sorted(set(topic_phrases), key=len, reverse=True):
                pat = re.compile(r"\b" + re.escape(tp) + r"\b", re.IGNORECASE)
                masked = pat.sub(_make_replacer("a_topic_hint"), masked)

        # (b) ALL-CAPS acronyms (2..6 uppercase chars, e.g. CPU, QR, AI, MFA, CNS, PNS)
        masked = re.sub(r"\b[A-Z]{2,6}\b", _make_replacer("b_acronyms"), masked)

        # (c) CamelCase / snake_case / code-like tokens (e.g. FinTech, VScodes, get_db, pageSize)
        masked = re.sub(r"\b[a-zA-Z]*[a-z][A-Z][a-zA-Z]*\b", _make_replacer("c_code_tokens"), masked)
        masked = re.sub(r"\b[a-zA-Z]+_[a-zA-Z0-9_]+\b", _make_replacer("c_code_tokens"), masked)

        # (d) Tokens containing digits or units (e.g. 20,000, 50ms, 100%, 4KB, 10th)
        masked = re.sub(
            r"\b\d+(?:[.,\-]\d+)*(?:%|ms|sec|s|kb|mb|gb|tb|hz|khz|mhz|ghz|v|a|w|ohm|th|st|nd|rd)?\b",
            _make_replacer("d_digits_units"),
            masked,
            flags=re.IGNORECASE,
        )

        # (e) Mid-sentence capitalized multi-word English phrases (NEVER sentence-initial!)
        # Preceded explicitly by lowercase letter, digit, comma, or semicolon + space
        mid_cap_re = re.compile(
            r"(?<=[a-z0-9,;]\s)([A-Z][a-z]+(?:\s+(?:and|of|the|in|for|on|to)\s+)?(?:\s+[A-Z][a-z]+)+)"
        )
        masked = mid_cap_re.sub(_make_replacer("e_mid_sentence_caps"), masked)

        # Also check mid-sentence capitalized phrases present in raw_text if read text was lowercased
        for m in mid_cap_re.finditer(cls.s1_unicode_and_whitespace(raw_text or "", {"rule_counts": {"S1_unicode_whitespace": 0}})):
            orig_phrase = m.group(1)
            if orig_phrase not in term_map.values():
                pat = re.compile(r"\b" + re.escape(orig_phrase) + r"\b", re.IGNORECASE)
                if pat.search(masked):
                    def _mask_with_orig_case(_m: re.Match, ph: str = orig_phrase) -> str:
                        nonlocal counter
                        key = f"__PROT_TERM_{seg_idx}_{counter}__"
                        counter += 1
                        term_map[key] = ph
                        cat_counts["e_raw_phrase_recap"] = cat_counts.get("e_raw_phrase_recap", 0) + 1
                        return key
                    masked = pat.sub(_mask_with_orig_case, masked)

        if term_map:
            stats["rule_counts"]["S2_protected_terms_masked"] += len(term_map)
        return masked, term_map

    @classmethod
    def s3_whisper_artifact_removal(
        cls,
        text: str,
        no_speech_prob: Optional[float],
        confidence: Optional[float],
        stats: dict[str, Any],
    ) -> tuple[Optional[str], Optional[str]]:
        """
        S3: Drop Whisper silence/hallucination segments and collapse consecutive n-gram loops.
        Uses only real DB columns: no_speech_prob and confidence (<0.3679 == avg_logprob<-1.0).
        Returns (cleaned_text_or_None, drop_reason_or_None).
        """
        if not text or not text.strip():
            stats["rule_counts"]["S3_dropped_empty"] += 1
            return None, "empty"

        # 1. Acoustic silence + low confidence drop
        if (no_speech_prob is not None and no_speech_prob > 0.6) and (confidence is not None and confidence < 0.3679):
            stats["rule_counts"]["S3_dropped_acoustic"] += 1
            return None, f"acoustic(nsp={no_speech_prob},conf={confidence})"

        # 2. Full-segment Whisper hallucination / translation refusal boilerplate
        unmasked_check = text
        if _WHISPER_HALLUCINATION_EXACT_RE.match(unmasked_check) or _LLM_REFUSAL_OR_GARBAGE_RE.search(unmasked_check):
            stats["rule_counts"]["S3_dropped_hallucination"] += 1
            return None, "hallucination_or_refusal"

        # 3. Strip empty non-alphanumeric noise segments
        if not re.search(r"[A-Za-z0-9\u0600-\u06FF]", text):
            stats["rule_counts"]["S3_dropped_empty"] += 1
            return None, "no_alnum"

        # 4. Collapse consecutive n-gram repetition loops (n = 1..6 repeated 3+ times -> keep 1)
        words = text.split()
        changed = False
        for n in range(6, 0, -1):
            if len(words) < n * 3:
                continue
            out_words: list[str] = []
            i = 0
            while i < len(words):
                gram = [w.lower().strip(".,!?;:") for w in words[i:i + n]]
                if len(gram) == n and not any(w.startswith("__prot_term_") for w in gram):
                    repeats = 1
                    j = i + n
                    while j + n <= len(words):
                        next_gram = [w.lower().strip(".,!?;:") for w in words[j:j + n]]
                        if next_gram == gram:
                            repeats += 1
                            j += n
                        else:
                            break
                    if repeats >= 3:
                        out_words.extend(words[i:i + n])
                        i = j
                        changed = True
                        continue
                out_words.append(words[i])
                i += 1
            words = out_words

        if changed:
            stats["rule_counts"]["S3_loop_collapsed"] += 1

        return " ".join(words).strip(), None

    @classmethod
    def s4_disfluency_cleanup(cls, text: str, stats: dict[str, Any]) -> str:
        """
        S4: Remove pure fillers (um, uh, hmm, erm, ah, aa).
        Keep meaningful discourse markers (matlab, yani, basically, so, acha) unless consecutive dup.
        Stutter collapse: 3+ identical -> 1; 2 identical -> 1 ONLY if function word.
        Preserve legitimate reduplication (bohat bohat, alag alag, dheere dheere).
        """
        if not text:
            return ""
        orig = text
        # Remove pure vocal fillers cleanly while fixing orphaned punctuation commas
        t = _PURE_FILLERS_RE.sub("", text)
        t = re.sub(r"\s*,\s*,+", ",", t)
        t = re.sub(r"^\s*[,;:]\s*", "", t)
        t = re.sub(r"\s+", " ", t).strip()
        if t != orig:
            stats["rule_counts"]["S4_pure_fillers_removed"] += 1

        tokens = t.split()
        if len(tokens) < 2:
            return t

        cleaned: list[str] = []
        i = 0
        stutter_collapsed = False
        while i < len(tokens):
            cur = tokens[i]
            cur_low = cur.strip(".,!?;:\"'()").lower()
            if cur.startswith("__PROT_TERM_"):
                cleaned.append(cur)
                i += 1
                continue

            # Count consecutive identical word tokens
            j = i + 1
            while j < len(tokens):
                nxt_low = tokens[j].strip(".,!?;:\"'()").lower()
                if nxt_low == cur_low and nxt_low != "":
                    j += 1
                else:
                    break

            run_len = j - i
            if run_len >= 3:
                cleaned.append(tokens[j - 1])  # keep final token (carries any trailing punctuation)
                stutter_collapsed = True
                i = j
            elif run_len == 2:
                if cur_low in _LEGITIMATE_REDUPLICATIONS:
                    cleaned.extend(tokens[i:j])
                elif cur_low in _FUNCTION_WORD_STUTTER_COLLAPSE or cur_low in _DISCOURSE_WORDS_KEEP:
                    cleaned.append(tokens[j - 1])
                    stutter_collapsed = True
                else:
                    cleaned.extend(tokens[i:j])
                i = j
            else:
                cleaned.append(cur)
                i += 1

        if stutter_collapsed:
            stats["rule_counts"]["S4_stutters_collapsed"] += 1

        res = " ".join(cleaned).strip()
        res = re.sub(r"\s+([.,!?;:])", r"\1", res)
        return res

    @classmethod
    def s5_roman_urdu_canonicalize(
        cls,
        text: str,
        is_english_dominant: bool,
        translation_source: Optional[str],
        stats: dict[str, Any],
        source_choice: str = "repaired_source",
    ) -> str:
        """
        S5: Reuse roman_urdu_nlp.normalizer._NORM_MAP selectively for Roman Urdu tokens ONLY.
        Skipped on English-dominant segments and verified English translations
        so English homographs (per, bat, para, ha) are never corrupted, while still
        firing on Roman-Urdu fallback segments.
        """
        if not text:
            return ""
        if is_english_dominant or source_choice == "english_text_verified":
            return text

        from roman_urdu_nlp.normalizer import _NORM_MAP
        from roman_urdu_nlp.detector import tag_tokens

        en_vocab = _get_english_vocab()
        tags = tag_tokens(text)
        tokens = text.split()
        if len(tags) != len(tokens):
            return text

        out_tokens: list[str] = []
        changed = False
        for idx, (tok, tag) in enumerate(zip(tokens, tags)):
            if tok.startswith("__PROT_TERM_"):
                out_tokens.append(tok)
                continue
            m = re.match(r"^([^\w]*)([\w']+)([^\w]*)$", tok)
            if not m:
                out_tokens.append(tok)
                continue
            pre, core, post = m.groups()
            low = core.lower()

            # Only canonicalize if token is in _NORM_MAP AND either tagged 'ur' or not a known English word
            if low in _NORM_MAP and (tag["lang"] == "ur" or low not in en_vocab):
                # Do not replace English prepositions/words ("per", "to", "in") if adjacent English
                if low in {"per", "bat", "para", "ha", "hey"} and _is_in_english_vocab(low, en_vocab):
                    out_tokens.append(tok)
                    continue
                canon = _NORM_MAP[low]
                if core[0].isupper() and idx == 0:
                    canon = canon[0].upper() + canon[1:]
                if canon.lower() != low:
                    changed = True
                out_tokens.append(f"{pre}{canon}{post}")
            else:
                out_tokens.append(tok)

        if changed:
            stats["rule_counts"]["S5_roman_urdu_canonicalized"] += 1
        return " ".join(out_tokens)

    @classmethod
    def _starts_lowercase_or_connector(cls, text: str) -> bool:
        t = (text or "").lstrip()
        if not t:
            return False
        first_tok = re.sub(r"^[^a-zA-Z0-9_]+", "", t.split()[0])
        if first_tok and first_tok[0].islower():
            return True
        return first_tok.lower() in _STITCH_CONNECTORS

    @classmethod
    def s6_segment_stitching(
        cls,
        work_segs: list[dict[str, Any]],
        stats: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        S6: Stitch segment into previous if previous has NO terminal punctuation (.!?)
        AND (current starts lowercase OR connector OR gap < 0.8s).
        Never merge across gap > 3.0s or speaker change.
        Adapter carries min(confidence) and min(no_speech_prob) across merged units (A4).
        """
        if not work_segs:
            return []

        stitched: list[dict[str, Any]] = [dict(work_segs[0])]
        for cur in work_segs[1:]:
            prev = stitched[-1]
            prev_text = prev["masked_text"].rstrip()
            gap = float(cur["start_time"]) - float(prev["end_time"])
            speaker_changed = (
                prev.get("speaker") is not None
                and cur.get("speaker") is not None
                and prev.get("speaker") != cur.get("speaker")
            )
            has_terminal_punct = bool(prev_text and prev_text[-1] in ".!?")
            trust_mismatch = prev.get("trust", "high") != cur.get("trust", "high")

            should_merge = (
                not has_terminal_punct
                and not speaker_changed
                and not trust_mismatch
                and gap <= 3.0
                and (cls._starts_lowercase_or_connector(cur["masked_text"]) or gap < 0.8)
            )

            if should_merge:
                prev["masked_text"] = (prev_text + " " + cur["masked_text"].lstrip()).strip()
                prev["text_after_s3"] = (prev["text_after_s3"] + " " + cur["text_after_s3"]).strip()
                prev["text_after_s4"] = (prev["text_after_s4"] + " " + cur["text_after_s4"]).strip()
                prev["term_map"].update(cur["term_map"])
                prev["source_segment_ids"].extend(cur["source_segment_ids"])
                prev["end_time"] = max(float(prev["end_time"]), float(cur["end_time"]))
                # Real min confidence & min no_speech_prob across merged segments (A4)
                confs = [c for c in (prev["confidence"], cur["confidence"]) if c is not None]
                nsps = [n for n in (prev["no_speech_prob"], cur["no_speech_prob"]) if n is not None]
                prev["confidence"] = min(confs) if confs else None
                prev["no_speech_prob"] = min(nsps) if nsps else None
                if cur.get("trust") == "low":
                    prev["trust"] = "low"
                stats["rule_counts"]["S6_segments_merged"] += 1
            else:
                stitched.append(dict(cur))

        return stitched

    @classmethod
    def s7_sentence_shaping(
        cls,
        stitched_segs: list[dict[str, Any]],
        stats: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        S7: Ensure terminal punctuation; split run-ons > 60 words at sentence-level
        connectors (aur, lekin, phir, because, so, but, and) preserving source_segment_ids
        and proportionally interpolating start/end timestamps.
        """
        shaped: list[dict[str, Any]] = []

        for seg in stitched_segs:
            txt = seg["masked_text"].strip()
            if not txt:
                continue

            words = txt.split()
            # Drop ultra-short unpunctuated dangling filler fragments < 4 words unless protected term
            if len(words) < 4 and "__PROT_TERM_" not in txt and len(shaped) > 0:
                if (seg["start_time"] - shaped[-1]["end_time"]) <= 3.0 and not any(
                    w.lower() in {"yes", "no", "true", "false"} for w in words
                ):
                    # Check if really just trailing fragment; preserve if valid content
                    pass

            # Ensure terminal punctuation
            if txt[-1] not in ".!?":
                txt += "."
                stats["rule_counts"]["S7_terminal_punct_restored"] += 1

            words = txt.split()
            if len(words) <= 60:
                new_seg = dict(seg)
                new_seg["masked_text"] = txt
                shaped.append(new_seg)
                continue

            # Split run-on > 60 words at connectors near target chunk midpoint (~45..55 words)
            chunks: list[str] = []
            start_idx = 0
            while start_idx < len(words):
                if len(words) - start_idx <= 60:
                    chunks.append(" ".join(words[start_idx:]))
                    break
                # Look for connector between word offsets 30 and 58
                split_idx = None
                for k in range(start_idx + 55, start_idx + 25, -1):
                    if k < len(words):
                        w_clean = words[k].strip(".,!?;:").lower()
                        if w_clean in _RUNON_SPLIT_CONNECTORS:
                            split_idx = k
                            break
                if split_idx is None:
                    split_idx = start_idx + 55
                # Keep sign/polarity modifier attached to a following number (e.g. "negative 50,000")
                if split_idx > start_idx + 1 and words[split_idx - 1].strip(".,!?;:").lower() in {
                    "negative", "minus", "positive", "plus"
                }:
                    split_idx -= 1
                part = " ".join(words[start_idx:split_idx]).rstrip(",;:- ")
                if part and part[-1] not in ".!?":
                    part += "."
                chunks.append(part)
                start_idx = split_idx

            stats["rule_counts"]["S7_runons_split"] += 1
            # Validate parent stitched block retention BEFORE splitting so splitting does not halve retention
            from roman_urdu_nlp.normalizer import _NORM_MAP as _NM_S7
            sw_s7 = _get_all_stopwords()
            en_v_s7 = _get_english_vocab()

            def _unmask_s7(s_in: str) -> str:
                u = s_in
                for k_t, v_t in seg["term_map"].items():
                    u = u.replace(k_t, v_t)
                return u

            def _ct_s7(s_in: str) -> list[str]:
                u = _unmask_s7(s_in)
                toks = re.findall(r"\b[a-zA-Z0-9\u0600-\u06FF]+\b", u.lower())
                return [w for w in (_NM_S7.get(x, x) if x not in en_v_s7 else x for x in toks) if w not in sw_s7]

            p_s4 = _ct_s7(seg["text_after_s4"])
            txt_unmasked_low = _unmask_s7(txt).lower()
            p_cand = set(_ct_s7(txt)) | set(re.findall(r"\b[a-zA-Z0-9\u0600-\u06FF]+\b", txt_unmasked_low))
            parent_ratio = min(1.0, sum(1 for tok in p_s4 if tok in p_cand) / len(p_s4)) if p_s4 else 1.0
            pre_fb = False
            if parent_ratio < RETENTION_MIN_RATIO and seg["text_after_s3"]:
                pre_fb = True
                stats["rule_counts"]["S8_retention_fallbacks"] += 1
                # Fallback parent text to S1+S3 before splitting so chunks remain distinct and loop-free
                fb_words = seg["text_after_s3"].split()
                chunks = []
                s_idx = 0
                while s_idx < len(fb_words):
                    e_idx = min(len(fb_words), s_idx + 55)
                    part_fb = " ".join(fb_words[s_idx:e_idx])
                    chunks.append(part_fb if part_fb[-1] in ".!?" else part_fb + ".")
                    s_idx = e_idx
                words = fb_words

            total_w = max(1, len(words))
            st_base = float(seg["start_time"])
            dur_base = max(0.0, float(seg["end_time"]) - st_base)
            cum_w = 0
            for ch in chunks:
                ch_w = len(ch.split())
                ch_st = st_base + dur_base * (cum_w / total_w)
                cum_w += ch_w
                ch_et = st_base + dur_base * (cum_w / total_w)
                sub = dict(seg)
                sub_txt = ch if ch[-1] in ".!?" else ch + "."
                sub["masked_text"] = sub_txt
                sub["text_after_s4"] = sub_txt
                sub["text_after_s3"] = sub_txt
                sub["pre_split_fallback"] = pre_fb
                sub["parent_ratio"] = round(parent_ratio, 3)
                sub["start_time"] = round(ch_st, 2)
                sub["end_time"] = round(ch_et, 2)
                shaped.append(sub)

        # F5: Enforce strictly monotonic, non-overlapping timestamps across consecutive output segments
        prev_end = 0.0
        for s_out in shaped:
            st_v = round(float(s_out["start_time"]), 2)
            et_v = round(float(s_out["end_time"]), 2)
            if st_v < prev_end:
                st_v = prev_end
            if et_v < st_v:
                et_v = st_v
            s_out["start_time"] = st_v
            s_out["end_time"] = et_v
            prev_end = et_v

        return shaped

    @classmethod
    def s8_unmask_and_retention_gate(
        cls,
        shaped_segs: list[dict[str, Any]],
        stats: dict[str, Any],
    ) -> list[IntelligenceSegment]:
        """
        S8: Unmask protected terms into `llm_text` and enforce post-S4 retention gate (>= 0.80).
        If retention < 0.80, falls back to S1 + S3 output (so artifacts do not return).
        """
        from roman_urdu_nlp.normalizer import _NORM_MAP

        stopwords = _get_all_stopwords()
        final_segs: list[IntelligenceSegment] = []

        def _unmask(txt: str, tmap: dict[str, str]) -> str:
            res = txt
            for key, orig_val in tmap.items():
                res = res.replace(key, orig_val)
            # Clean any accidental leftover placeholder tokens defensively
            return re.sub(r"\s+", " ", res).strip()

        def _content_tokens(txt: str) -> list[str]:
            toks = re.findall(r"\b[a-zA-Z0-9\u0600-\u06FF]+\b", txt.lower())
            canon = [_NORM_MAP.get(w, w) if w not in _get_english_vocab() else w for w in toks]
            return [w for w in canon if w not in stopwords]

        for idx, seg in enumerate(shaped_segs):
            tmap = seg["term_map"]
            llm_candidate = _unmask(seg["masked_text"], tmap)
            post_s4_unmasked = _unmask(seg["text_after_s4"], tmap)
            post_s3_unmasked = _unmask(seg["text_after_s3"], tmap)

            # Ensure leading capitalization
            if llm_candidate and llm_candidate[0].islower():
                llm_candidate = llm_candidate[0].upper() + llm_candidate[1:]

            s4_ct = _content_tokens(post_s4_unmasked)
            llm_ct = _content_tokens(llm_candidate)
            all_llm_toks = set(re.findall(r"\b[a-zA-Z0-9\u0600-\u06FF]+\b", llm_candidate.lower()))

            if not s4_ct:
                ratio = 1.0
            else:
                # Compute token-presence retention against post-S4 content tokens (allowing canonical variant equivalents)
                llm_set = set(llm_ct) | all_llm_toks
                kept_count = sum(1 for tok in s4_ct if tok in llm_set)
                ratio = min(1.0, kept_count / len(s4_ct))

            if seg.get("pre_split_fallback"):
                ratio = seg.get("parent_ratio", 0.0)
                is_fallback = True
                llm_final = llm_candidate
                stats["fallback_segments"].append({
                    "intelligence_segment_id": idx,
                    "source_segment_ids": seg["source_segment_ids"],
                    "ratio": round(ratio, 3),
                    "dropped_candidate": llm_candidate,
                    "fallback_s1_s3": llm_candidate,
                })
            else:
                is_fallback = False
                if ratio < RETENTION_MIN_RATIO and post_s3_unmasked:
                    is_fallback = True
                    stats["rule_counts"]["S8_retention_fallbacks"] += 1
                    stats["fallback_segments"].append({
                        "intelligence_segment_id": idx,
                        "source_segment_ids": seg["source_segment_ids"],
                        "ratio": round(ratio, 3),
                        "dropped_candidate": llm_candidate,
                        "fallback_s1_s3": post_s3_unmasked,
                    })
                    llm_final = post_s3_unmasked
                    if llm_final and llm_final[-1] not in ".!?":
                        llm_final += "."
                else:
                    llm_final = llm_candidate
                    if "parent_ratio" in seg:
                        ratio = seg["parent_ratio"]

            match_text = cls.s9_build_match_text(llm_final)
            final_segs.append(IntelligenceSegment(
                intelligence_segment_id=idx,
                source_segment_ids=list(dict.fromkeys(seg["source_segment_ids"])),
                start_time=round(float(seg["start_time"]), 2),
                end_time=round(float(seg["end_time"]), 2),
                llm_text=llm_final,
                match_text=match_text,
                confidence=seg["confidence"],
                no_speech_prob=seg["no_speech_prob"],
                language=seg.get("language"),
                translation_source=seg.get("translation_source"),
                source_choice=seg.get("source_choice", "repaired_source"),
                retention_ratio=round(ratio, 3),
                is_fallback=is_fallback,
                trust=seg.get("trust", "high"),
            ))

        return final_segs

    @classmethod
    def s9_build_match_text(cls, llm_text: str) -> str:
        """
        S9: Derive `match_text` from `llm_text`:
        lowercase, strip English & Roman Urdu stopwords, collapse spaces.
        """
        if not llm_text:
            return ""
        stopwords = _get_all_stopwords()
        tokens = re.findall(r"[a-zA-Z0-9\u0600-\u06FF]+(?:['\-][a-zA-Z0-9]+)*%?", llm_text.lower())
        kept = [t for t in tokens if t not in stopwords and len(t) >= 1]
        return " ".join(kept)

    # -----------------------------------------------------------------------
    # End-to-End Orchestration & Persistence
    # -----------------------------------------------------------------------

    @classmethod
    def build_view_in_memory(
        cls,
        recording_id: str,
        segments_orm: Sequence[Any],
        topic_hint: Optional[str] = None,
        transcript_id: Optional[str] = None,
        previous_segment_hashes: Optional[dict[str, dict[str, str]]] = None,
        write_diff_log: bool = True,
    ) -> IntelligenceView:
        """Executes deterministic S0-S9 pipeline in memory without mutating any ORM row."""
        t_start = time.monotonic()
        t_hash = cls.compute_display_transcript_hash(segments_orm)
        prev_map = previous_segment_hashes or {}

        stats: dict[str, Any] = {
            "total_input_segments": len(segments_orm),
            "s0_numeric_reverts": 0,
            "s0_english_reverts": 0,
            "s0_revert_samples": [],
            "s0_unsure_reverts": [],
            "source_choices": {},
            "trust_counts": {"high": 0, "low": 0},
            "s2_category_counts": {
                "a_topic_hint": 0,
                "b_acronyms": 0,
                "c_code_tokens": 0,
                "d_digits_units": 0,
                "e_mid_sentence_caps": 0,
                "e_raw_phrase_recap": 0,
            },
            "rule_counts": {
                "S1_unicode_whitespace": 0,
                "S2_protected_terms_masked": 0,
                "S3_dropped_acoustic": 0,
                "S3_dropped_hallucination": 0,
                "S3_dropped_empty": 0,
                "S3_loop_collapsed": 0,
                "S4_pure_fillers_removed": 0,
                "S4_stutters_collapsed": 0,
                "S5_roman_urdu_canonicalized": 0,
                "S6_segments_merged": 0,
                "S7_terminal_punct_restored": 0,
                "S7_runons_split": 0,
                "S8_retention_fallbacks": 0,
            },
            "dropped_segments": [],
            "fallback_segments": [],
            "segment_baselines": {},
            "samples": [],
            "elapsed_ms": 0.0,
        }

        # Precompute mean confidence per ASR chunk_index (F1)
        chunk_confs: dict[Any, list[float]] = {}
        for s in segments_orm:
            c_idx = getattr(s, "chunk_index", None)
            c_val = getattr(s, "confidence", None)
            if c_idx is not None and c_val is not None:
                chunk_confs.setdefault(c_idx, []).append(float(c_val))
        chunk_mean_conf: dict[Any, float] = {
            k: (sum(v) / len(v)) for k, v in chunk_confs.items() if v
        }

        work_segs: list[dict[str, Any]] = []
        for idx, s in enumerate(segments_orm):
            sid = getattr(s, "id", None) or getattr(s, "segment_index", idx)
            sid_str = str(sid)
            raw_txt = getattr(s, "text", "") or ""
            norm_txt = getattr(s, "normalized_text", None)
            eng_txt = getattr(s, "english_text", None)
            trans_src = getattr(s, "translation_source", None)
            lang = getattr(s, "language", None)
            nsp = getattr(s, "no_speech_prob", None)
            conf = getattr(s, "confidence", None)
            c_idx = getattr(s, "chunk_index", None)
            c_mean = chunk_mean_conf.get(c_idx) if c_idx is not None else None
            st = float(getattr(s, "start_time", None) if getattr(s, "start_time", None) is not None else getattr(s, "start", 0.0) or 0.0)
            et = float(getattr(s, "end_time", None) if getattr(s, "end_time", None) is not None else getattr(s, "end", st) or st)
            speaker = getattr(s, "speaker", None)

            # Save display baseline hashes for A2 manual correction tracking
            stats["segment_baselines"][sid_str] = {
                "raw_hash": hashlib.sha256(raw_txt.strip().encode("utf-8")).hexdigest()[:16],
                "norm_hash": hashlib.sha256((norm_txt or raw_txt).strip().encode("utf-8")).hexdigest()[:16],
                "eng_hash": hashlib.sha256((eng_txt or "").strip().encode("utf-8")).hexdigest()[:16],
            }

            # S0: Source repair & selection (F1 + F2)
            selected_src, src_choice, is_en_dom, seg_trust = cls.s0_source_repair(
                raw_text=raw_txt,
                normalized_text=norm_txt,
                english_text=eng_txt,
                translation_source=trans_src,
                topic_hint=topic_hint,
                previous_baseline=prev_map.get(sid_str),
                seg_id=sid,
                stats=stats,
                chunk_confidence=c_mean,
                seg_confidence=conf,
            )

            # S1: Unicode & whitespace
            s1_txt = cls.s1_unicode_and_whitespace(selected_src, stats)

            # S2: Protected-term masking (globally unique per seg_idx)
            s2_masked, tmap = cls.s2_mask_protected_terms(s1_txt, raw_txt, topic_hint, stats, seg_idx=idx)

            # S3: Whisper artifact & repetition removal
            s3_txt, drop_reason = cls.s3_whisper_artifact_removal(s2_masked, nsp, conf, stats)
            if s3_txt is None:
                stats["dropped_segments"].append({
                    "id": sid,
                    "segment_index": getattr(s, "segment_index", idx),
                    "reason": drop_reason,
                    "no_speech_prob": nsp,
                    "confidence": conf,
                    "text": raw_txt,
                })
                continue

            # S4: Disfluency cleanup
            s4_txt = cls.s4_disfluency_cleanup(s3_txt, stats)

            # S5: Conditional Roman Urdu canonicalization
            s5_txt = cls.s5_roman_urdu_canonicalize(
                s4_txt, is_en_dom, trans_src, stats, source_choice=src_choice
            )

            work_segs.append({
                "source_segment_ids": [sid],
                "start_time": st,
                "end_time": et,
                "speaker": speaker,
                "language": lang,
                "translation_source": trans_src,
                "source_choice": src_choice,
                "trust": seg_trust,
                "confidence": conf,
                "no_speech_prob": nsp,
                "term_map": tmap,
                "text_after_s3": s3_txt,
                "text_after_s4": s4_txt,
                "masked_text": s5_txt,
                "display_sample": raw_txt,
            })

        # S6: Boundary segment stitching
        stitched = cls.s6_segment_stitching(work_segs, stats)

        # S7: Sentence shaping & run-on splitting
        shaped = cls.s7_sentence_shaping(stitched, stats)

        # S8 + S9: Unmask, retention gate, build llm_text + match_text
        intel_segments = cls.s8_unmask_and_retention_gate(shaped, stats)
        for iseg in intel_segments:
            stats["trust_counts"][iseg.trust] = stats["trust_counts"].get(iseg.trust, 0) + 1

        full_llm_text = " ".join(s.llm_text for s in intel_segments if s.llm_text)
        full_match_text = " ".join(s.match_text for s in intel_segments if s.match_text)

        stats["total_output_segments"] = len(intel_segments)
        stats["elapsed_ms"] = round((time.monotonic() - t_start) * 1000, 2)

        # Capture up to 10 representative before/after samples for dev diff log
        orig_by_id = {
            (getattr(s, "id", None) or getattr(s, "segment_index", i)): (getattr(s, "text", "") or "")
            for i, s in enumerate(segments_orm)
        }
        for iseg in intel_segments[:10]:
            src_disp = " ".join(orig_by_id.get(sid, "") for sid in iseg.source_segment_ids)
            stats["samples"].append({
                "intel_seg_id": iseg.intelligence_segment_id,
                "source_ids": iseg.source_segment_ids,
                "display_text": src_disp,
                "llm_text": iseg.llm_text,
                "match_text": iseg.match_text,
            })

        if write_diff_log:
            cls._write_dev_diff_report(recording_id, t_hash, stats)

        return IntelligenceView(
            recording_id=recording_id,
            transcript_id=transcript_id,
            transcript_hash=t_hash,
            preprocessing_version=PREPROCESSING_VERSION,
            segments=intel_segments,
            full_llm_text=full_llm_text,
            full_match_text=full_match_text,
            rule_stats=stats,
        )

    @classmethod
    def filter_high_trust_adapters(cls, adapters: Sequence[Any]) -> list[Any]:
        """
        Phase 2 (F1): Exclude `trust == "low"` segments from quiz/notes grounding when
        high-trust segments exist. Falls back to full list only if 100% are low-trust.
        """
        high = [a for a in adapters if getattr(a, "trust", "high") != "low"]
        return high if high else list(adapters)

    @classmethod
    def build_per_source_embedding_texts(
        cls,
        segments_orm: Sequence[Any],
        topic_hint: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Phase 2 (F6): Produce ONE repaired `llm_text` per source `TranscriptSegment`
        (S0-S5 + unmask, NO S6 merge / NO S7 split, NEVER `match_text`).
        Includes `versioned_hash` = sha256(f"{PREPROCESSING_VERSION}:{llm_text}") so
        SegmentEmbedding automatically invalidates on preprocessing version or edit changes.
        """
        chunk_confs: dict[Any, list[float]] = {}
        for s in segments_orm:
            c_idx = getattr(s, "chunk_index", None)
            c_val = getattr(s, "confidence", None)
            if c_idx is not None and c_val is not None:
                chunk_confs.setdefault(c_idx, []).append(float(c_val))
        chunk_mean_conf: dict[Any, float] = {k: (sum(v) / len(v)) for k, v in chunk_confs.items() if v}

        dummy_stats: dict[str, Any] = {
            "rule_counts": {
                "S1_unicode_whitespace": 0, "S2_protected_terms_masked": 0,
                "S3_dropped_acoustic": 0, "S3_dropped_hallucination": 0, "S3_dropped_empty": 0,
                "S3_loop_collapsed": 0, "S4_pure_fillers_removed": 0, "S4_stutters_collapsed": 0,
                "S5_roman_urdu_canonicalized": 0,
            }
        }
        out: list[dict[str, Any]] = []
        for idx, s in enumerate(segments_orm):
            sid = getattr(s, "id", None) or getattr(s, "segment_index", idx)
            raw_txt = getattr(s, "text", "") or ""
            norm_txt = getattr(s, "normalized_text", None)
            eng_txt = getattr(s, "english_text", None)
            trans_src = getattr(s, "translation_source", None)
            nsp = getattr(s, "no_speech_prob", None)
            conf = getattr(s, "confidence", None)
            c_idx = getattr(s, "chunk_index", None)
            c_mean = chunk_mean_conf.get(c_idx) if c_idx is not None else None

            selected_src, src_choice, is_en_dom, seg_trust = cls.s0_source_repair(
                raw_text=raw_txt, normalized_text=norm_txt, english_text=eng_txt,
                translation_source=trans_src, topic_hint=topic_hint, previous_baseline=None,
                seg_id=sid, stats=dummy_stats, chunk_confidence=c_mean, seg_confidence=conf,
            )
            s1_txt = cls.s1_unicode_and_whitespace(selected_src, dummy_stats)
            s2_masked, tmap = cls.s2_mask_protected_terms(s1_txt, raw_txt, topic_hint, dummy_stats, seg_idx=idx)
            s3_txt, _ = cls.s3_whisper_artifact_removal(s2_masked, nsp, conf, dummy_stats)
            if s3_txt is None:
                # Preserve 1:1 row cardinality with repaired S1 text (empty/noise if nothing valid)
                llm_txt = s1_txt
            else:
                s4_txt = cls.s4_disfluency_cleanup(s3_txt, dummy_stats)
                s5_txt = cls.s5_roman_urdu_canonicalize(
                    s4_txt, is_en_dom, trans_src, dummy_stats, source_choice=src_choice
                )
                llm_txt = s5_txt
                for k_t, v_t in tmap.items():
                    llm_txt = llm_txt.replace(k_t, v_t)
                llm_txt = re.sub(r"\s+", " ", llm_txt).strip()
            versioned_hash = hashlib.sha256(
                f"{PREPROCESSING_VERSION}:{llm_txt}".encode("utf-8")
            ).hexdigest()
            out.append({
                "segment_id": sid,
                "llm_text": llm_txt,
                "versioned_hash": versioned_hash,
                "trust": seg_trust,
                "source_choice": src_choice,
            })
        return out

    @classmethod
    def get_or_create_view(
        cls,
        recording_id: str,
        db: Session,
        topic_hint: Optional[str] = None,
        force_refresh: bool = False,
    ) -> IntelligenceView:
        """
        Retrieves persisted IntelligenceView matching (recording_id, transcript_hash, PREPROCESSING_VERSION),
        or computes and persists a new view using strictly read-only queries on display tables.
        """
        from app.models import Recording, Transcript, TranscriptSegment

        recording = db.query(Recording).filter(Recording.id == recording_id).first()
        if topic_hint is None and recording is not None:
            topic_hint = getattr(recording, "topic", None)

        transcript = db.query(Transcript).filter(Transcript.recording_id == recording_id).first()
        segments_orm = (
            db.query(TranscriptSegment)
            .filter(TranscriptSegment.transcript_id == transcript.id)
            .order_by(TranscriptSegment.segment_index.asc())
            .all()
            if transcript
            else []
        )

        t_hash = cls.compute_display_transcript_hash(segments_orm)

        if not force_refresh:
            cached = (
                db.query(IntelligenceViewRecord)
                .filter(
                    IntelligenceViewRecord.recording_id == recording_id,
                    IntelligenceViewRecord.transcript_hash == t_hash,
                    IntelligenceViewRecord.preprocessing_version == PREPROCESSING_VERSION,
                )
                .first()
            )
            if cached:
                segs_data = json.loads(cached.segments_json or "[]")
                return IntelligenceView(
                    recording_id=recording_id,
                    transcript_id=cached.transcript_id,
                    transcript_hash=cached.transcript_hash,
                    preprocessing_version=cached.preprocessing_version,
                    segments=[IntelligenceSegment(**d) for d in segs_data],
                    full_llm_text=cached.full_llm_text,
                    full_match_text=cached.full_match_text,
                    rule_stats=json.loads(cached.rule_stats_json or "{}"),
                )

        # Look up any prior baseline for manual correction diff detection (A2)
        prev_record = (
            db.query(IntelligenceViewRecord)
            .filter(IntelligenceViewRecord.recording_id == recording_id)
            .order_by(IntelligenceViewRecord.created_at.desc())
            .first()
        )
        prev_baselines = None
        if prev_record and prev_record.rule_stats_json:
            try:
                prev_baselines = json.loads(prev_record.rule_stats_json).get("segment_baselines")
            except Exception:
                prev_baselines = None

        view = cls.build_view_in_memory(
            recording_id=recording_id,
            segments_orm=segments_orm,
            topic_hint=topic_hint,
            transcript_id=transcript.id if transcript else None,
            previous_segment_hashes=prev_baselines,
        )

        # Persist in intelligence_views (creating table lazily if needed on test DBs)
        try:
            record = IntelligenceViewRecord(
                id=str(uuid.uuid4()),
                recording_id=recording_id,
                transcript_id=transcript.id if transcript else "",
                transcript_hash=view.transcript_hash,
                preprocessing_version=PREPROCESSING_VERSION,
                topic_hint=topic_hint,
                segments_json=json.dumps([s.to_dict() for s in view.segments], ensure_ascii=False),
                full_llm_text=view.full_llm_text,
                full_match_text=view.full_match_text,
                rule_stats_json=json.dumps(view.rule_stats, ensure_ascii=False),
            )
            db.add(record)
            db.commit()
        except Exception as e:
            logger.warning(f"Failed to persist intelligence view (rolling back): {e}")
            db.rollback()

        return view

    @classmethod
    def _write_dev_diff_report(cls, recording_id: str, t_hash: str, stats: dict[str, Any]) -> None:
        """Dev-only diagnostics written to logs/intel_preprocess_diff.log (never surfaced in UI)."""
        try:
            log_dir = Path(BASE_DIR) / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / "intel_preprocess_diff.log"
            payload = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "recording_id": recording_id,
                "transcript_hash": t_hash[:16],
                "preprocessing_version": PREPROCESSING_VERSION,
                "elapsed_ms": stats.get("elapsed_ms"),
                "input_segments": stats.get("total_input_segments"),
                "output_segments": stats.get("total_output_segments"),
                "rule_counts": stats.get("rule_counts"),
                "source_choices": stats.get("source_choices"),
                "s0_reverts": {
                    "numeric": stats.get("s0_numeric_reverts"),
                    "english": stats.get("s0_english_reverts"),
                    "samples": stats.get("s0_revert_samples", [])[:10],
                },
                "dropped_count": len(stats.get("dropped_segments", [])),
                "fallback_count": len(stats.get("fallback_segments", [])),
                "samples": stats.get("samples", [])[:5],
            }
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            pass
