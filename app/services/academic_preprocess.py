"""
app/services/academic_preprocess.py — Independent Academic Preprocessor.

Generates an academic analysis copy of lecture transcript segments:
- Cleans each segment independently; never merges across segment boundaries.
- Emits AnalysisSpan(segment_id, raw_start, raw_end, clean_text, token_map).
- token_map maps each cleaned token index to the exact (start_char, end_char) in the raw segment.
- Preserves: negations (not, no, nahi, na, mat), hedges (shayad, maybe, probably),
  acronyms, numerals, formulas, and code-switched technical terms.
- Removes only conversational disfluencies (bhaiya, acha ji, dekhein, etc.).
- Compiles a recording-scoped vocabulary of stemmed content tokens and key terms.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence


# Disfluency regex patterns that do not carry academic meaning
_DISFLUENCY_PATTERNS = [
    re.compile(r"\b(?:assalamu\s+alaikum(?:\s+everyone)?|walaikum\s+assalam)\b", re.IGNORECASE),
    re.compile(r"\b(?:good\s+morning(?:\s*,\s*|\s+)?(?:everyone|all)?|good\s+afternoon|good\s+evening)\b", re.IGNORECASE),
    re.compile(r"\b(?:bhaiya|bhiya|dekhein|theek\s+hai|acha\s+ji|acha\s+jee|jee\s+haan|haan\s+ji)\b", re.IGNORECASE),
    re.compile(r"\b(?:1\s+more\s+cheese|ek\s+ahem\s+cheez|sahi\s+hai|samajh\s+aaya)\b", re.IGNORECASE),
    re.compile(r"\b(?:you\s+know|like\s+i\s+said|as\s+you\s+can\s+see|let's\s+move\s+on)\b", re.IGNORECASE),
    re.compile(r"here(?:'s|\s+is)\s+the\s+(?:english\s+)?translation[^:\n]*:", re.IGNORECASE),
    re.compile(r"[^.!?]*\b(?:cannot\s+translate|text\s+to\s+translate|please\s+provide\s+a\s+complete|profits\s+on\s+the\s+fijian|role\s+of\s+education\s+in\s+economic\s+development|kapriya\s+chhuri|visa\s+vaidhya)\b[^.!?]*[.!?]?", re.IGNORECASE),
    re.compile(
        r"(?:"
        r"\bcan\s+(?:you|everyone)\s+(?:hear|see)\s+me\b|"
        r"\bam\s+i\s+audible\b|"
        r"\bis\s+my\s+screen\s+visible\b|"
        r"\bplease\s+(?:mute|unmute)\s+your\b|"
        r"\blet\s+me\s+share\s+my\s+screen\b|"
        r"\bwhile\s+i\s+share\s+my\s+screen\b|"
        r"\bopen\s+your\s+(?:books?|notebooks?|laptops?)\s+to\s+page\b|"
        r"\bturn\s+to\s+page\s+\d+\b|"
        r"\bany\s+questions\s*(?:so\s+far|from\s+anyone)?\??|"
        r"\braise\s+your\s+hand\b|"
        r"\bas\s+an\s+ai\s+language\s+model\b|"
        r"\bsure,?\s+i\s+can\s+help\s+(?:you|with\s+that)\b|"
        r"\bsummarize\s+this\s+transcript\b|"
        r"\bwill\s+this\s+(?:topic\s+)?be\s+in\s+the\s+(?:midterm|final)\s+exam\b"
        r")",
        re.IGNORECASE,
    ),
]

# Critical tokens that MUST be preserved
_NEGATIONS = frozenset({"not", "no", "never", "none", "neither", "nor", "nahi", "nahin", "na", "mat", "bina", "baghair"})


@dataclass(frozen=True)
class AnalysisSpan:
    segment_id: Any
    raw_start: float
    raw_end: float
    raw_text: str
    clean_text: str
    token_map: list[tuple[int, int]]  # token_idx -> (start_char, end_char) in raw_text
    tokens: list[str]


@dataclass
class PreprocessResult:
    spans: list[AnalysisSpan]
    full_clean_text: str
    recording_vocab: set[str]
    total_tokens: int


def _stem_simple(token: str) -> str:
    """Lightweight suffix-stripping without external dependencies."""
    w = token.lower()
    if len(w) <= 4:
        return w
    for suffix in ("ing", "tion", "tions", "ies", "ied", "ed", "ly", "es", "s"):
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            return w[:-len(suffix)]
    return w


class AcademicPreprocessor:
    """Preprocesses raw transcript segments into clean academic analysis spans."""

    @classmethod
    def clean_segment_text(cls, raw_text: str) -> tuple[str, list[tuple[int, int]], list[str]]:
        """
        Cleans disfluencies while preserving negations, formulas, and token character offsets.
        Returns: (clean_text, token_map, clean_tokens)
        """
        if not raw_text or not raw_text.strip():
            return "", [], []

        # Find all raw word tokens with their exact character offsets
        raw_token_spans: list[tuple[str, int, int]] = []
        for m in re.finditer(r"\S+", raw_text):
            raw_token_spans.append((m.group(0), m.start(), m.end()))

        # Disfluency masking mask
        mask = [True] * len(raw_text)
        for pat in _DISFLUENCY_PATTERNS:
            for match in pat.finditer(raw_text):
                # Ensure we don't mask negations or hedges even if a pattern touches them
                start, end = match.start(), match.end()
                matched_sub = raw_text[start:end].lower()
                if not any(neg in matched_sub.split() for neg in _NEGATIONS):
                    for idx in range(start, end):
                        mask[idx] = False

        # Filter tokens that fall predominantly inside unmasked spans
        clean_tokens: list[str] = []
        token_map: list[tuple[int, int]] = []

        for token_str, s_pos, e_pos in raw_token_spans:
            # Check mask coverage
            active_chars = sum(1 for c_idx in range(s_pos, e_pos) if mask[c_idx])
            if active_chars >= (e_pos - s_pos) * 0.5:
                # Token is valid; preserve
                clean_tok = re.sub(r"^[^\w\(\)\+\-\*\/\=]+|[^\w\(\)\+\-\*\/\=]+$", "", token_str)
                if clean_tok:
                    clean_tokens.append(clean_tok)
                    token_map.append((s_pos, e_pos))

        clean_text = " ".join(clean_tokens)
        return clean_text, token_map, clean_tokens

    @classmethod
    def process_segments(cls, segments: Sequence[Any]) -> PreprocessResult:
        """
        Processes a sequence of TranscriptSegment rows (or IntelligenceSegmentAdapters)
        into AnalysisSpans and compiles a recording-specific vocabulary.
        Bypasses destructive legacy disfluency regexes on pre-cleaned intel-view segments
        and excludes `trust == "low"` segments when high-trust segments exist.
        """
        spans: list[AnalysisSpan] = []
        vocab: set[str] = set()
        clean_texts: list[str] = []
        total_tokens = 0

        has_high_trust = any(getattr(seg, "trust", "high") != "low" for seg in segments)

        for seg in segments:
            if has_high_trust and getattr(seg, "trust", "high") == "low":
                continue

            seg_id = getattr(seg, "id", None) or getattr(seg, "segment_index", 0)
            raw_start = float(getattr(seg, "start_time", 0.0) or 0.0)
            raw_end = float(getattr(seg, "end_time", raw_start + 1.0) or (raw_start + 1.0))

            if getattr(seg, "is_intel_view", False):
                # Bypass legacy disfluency stripping / punctuation deletion; preserve grammatical llm_text
                raw_text = getattr(seg, "llm_text", None) or getattr(seg, "text", "") or ""
                token_map: list[tuple[int, int]] = []
                tokens: list[str] = []
                for m in re.finditer(r"\S+", raw_text):
                    tok_clean = re.sub(r"^[^\w\(\)\+\-\*\/\=]+|[^\w\(\)\+\-\*\/\=]+$", "", m.group(0))
                    if tok_clean:
                        tokens.append(tok_clean)
                        token_map.append((m.start(), m.end()))
                clean_text = raw_text.strip()
                # F3: Incorporate match_text tokens into grounding/contamination vocab (never prompts/embeddings)
                for mt in (getattr(seg, "match_text", "") or "").split():
                    mt_clean = re.sub(r"[^a-zA-Z0-9]", "", mt.lower())
                    if len(mt_clean) >= 2:
                        vocab.add(mt_clean)
                        vocab.add(_stem_simple(mt_clean))
            else:
                raw_text = getattr(seg, "english_text", None) or getattr(seg, "normalized_text", None) or getattr(seg, "text", "") or ""
                clean_text, token_map, tokens = cls.clean_segment_text(raw_text)

            span = AnalysisSpan(
                segment_id=seg_id,
                raw_start=round(raw_start, 2),
                raw_end=round(raw_end, 2),
                raw_text=raw_text,
                clean_text=clean_text,
                token_map=token_map,
                tokens=tokens,
            )
            spans.append(span)
            if clean_text:
                clean_texts.append(clean_text)

            total_tokens += len(tokens)
            for tok in tokens:
                tok_clean = re.sub(r"[^a-zA-Z]", "", tok.lower())
                if len(tok_clean) >= 3:
                    vocab.add(tok_clean)
                    vocab.add(_stem_simple(tok_clean))

        full_clean = " ".join(clean_texts)
        return PreprocessResult(
            spans=spans,
            full_clean_text=full_clean,
            recording_vocab=vocab,
            total_tokens=total_tokens,
        )


# ---------------------------------------------------------------------------
# Keyphrase extraction (C-value) — single-document, no background corpus needed
# ---------------------------------------------------------------------------
import math
from collections import Counter, defaultdict

_KP_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "of", "in", "on", "at", "to",
    "for", "with", "by", "from", "and", "or", "but", "if", "then", "so", "this", "that",
    "these", "those", "it", "its", "we", "you", "i", "they", "he", "she", "will", "can",
    "do", "does", "did", "have", "has", "had", "as", "into", "about", "also", "just",
    "very", "more", "most", "some", "any", "such", "than", "which", "what", "when", "where",
    "how", "why", "who", "our", "your", "their", "my", "me", "us", "them",
    "hai", "hain", "ka", "ke", "ki", "ko", "se", "mein", "par", "aur", "ye", "yeh", "wo",
    "woh", "to", "bhi", "kya", "kyun", "jab", "tab", "toh",
    # generic verbs / adjectives / adverbs / number words that make poor phrase boundaries
    "because", "therefore", "however", "needs", "need", "gives", "give", "given", "gets", "get",
    "makes", "make", "made", "uses", "use", "used", "using", "takes", "take", "taken", "keeps", "keep",
    "higher", "lower", "larger", "smaller", "large", "small", "long", "short", "too", "almost", "always",
    "never", "every", "each", "eventually", "completely", "fully", "slowly", "while", "once", "over",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "next", "same", "other", "another", "many", "much", "several", "both", "either", "not", "no",
    "behaves", "happens", "occurs", "means", "mean", "called", "known", "being",
    "like", "when", "after", "before", "until", "between", "through", "during", "within", "without",
})


def _is_candidate_edge(tok: str) -> bool:
    t = tok.lower()
    return len(t) >= 2 and t.isalpha() and t not in _KP_STOPWORDS


def extract_keyphrases(
    spans: Sequence[AnalysisSpan],
    top_k: int = 15,
    min_freq: int = 2,
    max_len: int = 4,
) -> list[tuple[str, float, int]]:
    """
    C-value keyphrase extraction over clean spans. Candidates never cross segment boundaries.
    Returns [(phrase, c_value, frequency)] sorted by c_value desc.
    Nested-term penalty keeps 'descent' from outranking 'gradient descent'.
    """
    freq: Counter = Counter()
    for sp in spans:
        words = [re.sub(r"[^\w]", "", t).lower() for t in sp.tokens]
        words = [w for w in words if w]
        n = len(words)
        for i in range(n):
            for L in range(1, max_len + 1):
                j = i + L
                if j > n:
                    break
                gram = words[i:j]
                if not (_is_candidate_edge(gram[0]) and _is_candidate_edge(gram[-1])):
                    continue
                if any(re.fullmatch(r"\d+", g) for g in gram):
                    continue
                freq[" ".join(gram)] += 1

    cands = {g: f for g, f in freq.items() if f >= min_freq}
    if not cands:
        return []

    # Super-term index: for each candidate, the longer candidates that contain it as a contiguous sub-phrase
    supers: dict[str, list[str]] = defaultdict(list)
    by_len = sorted(cands, key=lambda g: len(g.split()))
    for longer in by_len:
        lw = longer.split()
        if len(lw) < 2:
            continue
        for L in range(1, len(lw)):
            for i in range(len(lw) - L + 1):
                sub = " ".join(lw[i:i + L])
                if sub in cands and sub != longer:
                    supers[sub].append(longer)

    scored: list[tuple[str, float, int]] = []
    for g, f in cands.items():
        length = len(g.split())
        weight = math.log2(length + 1)
        sup = supers.get(g, [])
        if sup:
            penalty = sum(cands[s] for s in sup) / len(sup)
            c_val = weight * (f - penalty)
        else:
            c_val = weight * f
        if c_val > 0:
            scored.append((g, round(c_val, 3), f))
    scored.sort(key=lambda x: (-x[1], x[0]))
    return scored[:top_k]


# ---------------------------------------------------------------------------
# Deterministic importance scoring (1-5). Weights/cut points are fixed constants
# (documented in eval_results/thresholds.md). Qwen is never used for scoring.
# ---------------------------------------------------------------------------
IMPORTANCE_WEIGHTS = {"time_share": 0.30, "repetition": 0.20, "term_density": 0.25, "structure": 0.25}
IMPORTANCE_CUTS = (0.20, 0.40, 0.60, 0.80)  # <0.20→1, <0.40→2, <0.60→3, <0.80→4, else 5

_STRUCTURE_PATTERNS = {
    "definition": re.compile(r"\b(?:is defined as|is called|refers to|means|is known as|is a|is an)\b", re.I),
    "example": re.compile(r"\b(?:for example|for instance|e\.g\.|such as|jaise|misal)\b", re.I),
    "steps": re.compile(r"\b(?:first|second|third|step|then|finally|next)\b", re.I),
    "formula": re.compile(r"[A-Za-z]\s*=\s*[A-Za-z0-9(]"),
}


def importance_label(score: int) -> str:
    """Derived at serialization only (D3): 1-2 low, 3 medium, 4-5 high."""
    return "low" if score <= 2 else ("medium" if score == 3 else "high")


def score_importance(
    topic_spans: Sequence[AnalysisSpan],
    all_spans: Sequence[AnalysisSpan],
    keyphrases: Sequence[tuple[str, float, int]],
    similarity_fn=None,
) -> dict:
    """
    Returns {"score": int 1-5, "components": {...}}.
    similarity_fn(textA, textB)->float is optional (MiniLM cosine). Without it a lexical Jaccard is used.
    """
    total_dur = sum(max(0.0, s.raw_end - s.raw_start) for s in all_spans) or 1.0
    topic_dur = sum(max(0.0, s.raw_end - s.raw_start) for s in topic_spans)
    time_share = min(1.0, (topic_dur / total_dur) * 3.0)  # 33% of lecture saturates

    topic_text = " ".join(s.clean_text for s in topic_spans)
    topic_words = [w.lower() for w in re.findall(r"[A-Za-z]+", topic_text)]
    kp_set = {k for k, _, _ in keyphrases}
    kp_hits = sum(topic_text.lower().count(k) for k in kp_set) if kp_set else 0
    term_density = min(1.0, (kp_hits / max(1, len(topic_words))) * 6.0)

    # Repetition: re-explanation in non-adjacent segments outside the topic
    topic_ids = {s.segment_id for s in topic_spans}
    others = [s for s in all_spans if s.segment_id not in topic_ids and s.clean_text]
    rep_hits = 0
    for t in topic_spans[:6]:
        for o in others[:60]:
            if similarity_fn is not None:
                sim = similarity_fn(t.clean_text, o.clean_text)
            else:
                a, b = set(t.tokens), set(o.tokens)
                sim = len(a & b) / max(1, len(a | b))
            if sim >= (0.6 if similarity_fn is not None else 0.5):
                rep_hits += 1
                break
    repetition = min(1.0, rep_hits / max(1, min(len(topic_spans), 6)))

    present = sum(1 for p in _STRUCTURE_PATTERNS.values() if any(p.search(s.raw_text) for s in topic_spans))
    structure = present / len(_STRUCTURE_PATTERNS)

    comp = {"time_share": time_share, "repetition": repetition, "term_density": term_density, "structure": structure}
    total = sum(IMPORTANCE_WEIGHTS[k] * v for k, v in comp.items())
    score = 1 + sum(1 for c in IMPORTANCE_CUTS if total >= c)
    return {"score": int(score), "total": round(total, 3), "components": {k: round(v, 3) for k, v in comp.items()}}
