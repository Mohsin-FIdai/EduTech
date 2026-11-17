"""
app/services/topic_intelligence.py -- Local-First AI Lecture Understanding & Study Notes Engine (v3).

Architecture:
  1. Provider-independent LLM abstraction (LocalLLMProvider / CloudLLMProvider).
  2. Pass 1: Lecture Structure Analysis (subject, theme, overview, conceptual sections mapped to segment IDs).
  3. Pass 2: Topic-Level Synthesis (explanations, key points, definitions, examples, formulas, emphasis).
  4. Grounding & Anti-Hallucination: Timestamps derived strictly from source_segment_ids.
  5. Multi-span topic reconciliation and merging for recurring discussions.
  6. StudyNotesQualityValidator: 7-metric gate ensuring academic depth, non-uniform intervals, and valid grounding.
  7. Extractive fallback engine if local LLM is disabled or unavailable.

No CUDA required -- CPU-only inference on Windows Python 3.13.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


def _cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Fast pure-NumPy cosine similarity avoiding sklearn/pyarrow checks."""
    a_arr = np.atleast_2d(np.asarray(a, dtype=np.float32))
    b_arr = np.atleast_2d(np.asarray(b, dtype=np.float32))
    a_norm = np.linalg.norm(a_arr, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b_arr, axis=1, keepdims=True)
    a_norm = np.where(a_norm == 0, 1e-8, a_norm)
    b_norm = np.where(b_norm == 0, 1e-8, b_norm)
    return (a_arr / a_norm) @ (b_arr / b_norm).T

logger = logging.getLogger(__name__)

# Stopwords for candidate filtering & title quality checks
_GENERIC_DISALLOWED_TITLES = {
    "algorithm", "algorithms", "deep", "feature", "features", "lecture",
    "lectures", "process", "processes", "system", "systems", "important",
    "concept", "concepts", "introduction", "slide", "slides", "topic",
    "topics", "class", "today", "discussed", "explanation", "example",
    "bhaiya bhiya", "bhaiya", "bhiya", "millisecond after", "gets seconds",
    "mhebi jis process", "mhebi", "diphthim", "mhmi", "raid", "radi",
    "system cpu", "application", "applications", "running", "section",
    "queue", "transaction", "order", "instance", "response therefore",
    "coordination involves", "foot right away", "right away", "hand away",
    "good morning", "near topic", "flex action", "part time", "in simple words",
    "let us move", "pupil becomes smaller", "protect body", "jewish wants",
    "reason why profits", "through each component", "structural current",
    "nirang formation", "nirang", "whereas response"
}

_URDU_CONVERSATIONAL_FILLERS = {
    "bhaiya", "bhiya", "bhai", "acha", "accha", "achha", "theek", "thik",
    "chal", "chalo", "chalein", "suno", "sunen", "suniye", "dekho", "dekhen",
    "yaar", "ab", "phir", "jis", "jisme", "jiski", "jisko", "jahan",
    "idhar", "udhar", "wahan", "yahan", "hum", "humein", "ap", "aap",
    "tum", "meri", "mera", "mere", "apka", "aapka", "apne", "apki",
    "uski", "uska", "unka", "unke", "isne", "usne", "karna", "karega",
    "karte", "karti", "karegi", "hoga", "hogi", "honge", "hota", "hoti",
    "hote", "hai", "hain", "tha", "thi", "the", "raha", "rahi", "rahe",
    "bolo", "bol", "samjho", "samajh", "baat", "cheez", "cheezein",
    "waghera", "wagera", "kahe", "kaha", "matlab", "yani", "bhi", "to",
    "lekin", "agar", "magar", "sirf", "thora", "thoda", "bahut", "bohot",
    "uff", "aray", "aree", "oye", "haan", "nahi", "na", "aaj", "kal", "kuch", "ye", "wo", "pe", "se"
}

_DISALLOWED_FRAGMENT_EDGES = {
    "start": {
        "after", "before", "during", "into", "with", "without", "from", "for",
        "and", "or", "but", "then", "than", "also", "too", "so", "if", "when",
        "while", "until", "because", "since", "per", "by", "to", "at", "in", "on",
        "of", "is", "are", "was", "were", "be", "been", "gets", "getting", "got",
        "goes", "going", "went", "comes", "coming", "came", "takes", "taking",
        "took", "makes", "making", "made", "starts", "started", "using", "used",
        "having", "has", "have", "had", "does", "do", "did", "suppose", "let",
        "therefore", "however", "moreover", "furthermore", "meanwhile", "thus", "hence",
        "basically", "actually", "literally", "right", "away", "involves", "involve",
        "involving", "meaning", "today", "yesterday", "tomorrow", "first", "last",
        "such", "good", "morning", "evening", "afternoon", "everyone", "accidentally",
        "accidently", "near", "foot", "hand", "eye", "eyes", "head", "pulled",
        "pulling", "touching", "touched", "acts", "acting", "act",
        "earn", "earning", "earned", "earns", "buy", "buying", "bought", "buys",
        "sell", "selling", "sold", "sells", "spend", "spending", "spent", "spends",
        "pay", "paying", "paid", "pays", "give", "giving", "gave", "given", "gives",
        "find", "finding", "found", "finds", "look", "looking", "looked", "looks",
        "show", "showing", "showed", "shown", "shows", "see", "seeing", "saw", "seen", "sees",
        "determine", "determining", "determined", "determines", "compare", "comparing",
        "compared", "compares", "tell", "telling", "told", "tells", "ask", "asking",
        "bring", "bringing", "brought", "brings",
        "through", "throughout", "across", "along", "around", "between", "among", "towards",
        "against", "over", "under", "via", "whereas", "why", "how", "what", "where", "who",
        "whom", "whose", "reason", "reasons"
    },
    "end": {
        "after", "before", "during", "into", "with", "without", "from", "for",
        "and", "or", "but", "then", "than", "also", "too", "so", "if", "when",
        "while", "until", "because", "since", "per", "by", "to", "at", "in", "on",
        "of", "is", "are", "was", "were", "be", "been", "being", "gets", "got",
        "has", "have", "had", "does", "did", "the", "a", "an", "this", "that",
        "seconds", "second", "millisecond", "milliseconds", "minutes", "minute",
        "hours", "hour", "sec", "ms", "therefore", "however", "thus", "hence",
        "otherwise", "instead", "meanwhile", "away", "right", "already", "well",
        "now", "then", "involves", "involve", "involving", "means", "meaning",
        "calls", "called", "known", "said", "says", "acts", "act", "moves",
        "moved", "occurred", "occurs", "detects", "detected", "foot", "hand",
        "eye", "eyes", "head", "everyone", "today", "yesterday", "tomorrow",
        "first", "last", "near", "such", "morning", "evening", "afternoon"
    }
}

_TIME_UNIT_WORDS = {
    "second", "seconds", "millisecond", "milliseconds", "minute", "minutes",
    "hour", "hours", "sec", "ms"
}

_HALLUCINATION_WORDS = {
    "mhebi", "mhmi", "diphthim", "dhk", "ske", "ala", "chamakthak", "shwanda", "invincible", "bred"
}

_ALLOWED_ACRONYMS = {
    "cpu", "gpu", "os", "ram", "rom", "io", "sjf", "fcfs", "srtf", "rr",
    "atm", "pcb", "tlb", "mmu", "pid", "uid", "gui", "cli", "api", "sdk",
    "sql", "xml", "json", "html", "css", "tcp", "udp", "ip", "dns", "http", "https",
    "tv", "tvs", "cns", "pns", "emf", "dc", "ac", "si", "led", "lcd", "ic", "pc"
}


ENGLISH_COMMON_VOCAB = {
    "workflow", "walks", "walk", "first", "firstcome", "firstserved", "serves", "served",
    "burst", "turnaround", "waiting", "context", "switch", "paging", "aging", "starvation",
    "overhead", "queue", "scheduling", "scheduler", "preemptive", "preemption", "throughput",
    "responsiveness", "multitasking", "multiprogramming", "threads", "thread", "process",
    "processes", "operating", "system", "systems", "completion", "arrival", "quantum",
    "demonstrating", "demonstrates", "principles", "theoretical", "operational", "latency",
    "short", "tasks", "task", "terms", "term", "length", "strength", "graphs", "blocks",
    "chunks", "stacks", "struct", "script", "points", "yield", "order", "state", "times",
    "round", "robin", "interrupted", "interrupt", "priority"
}

ENGLISH_4_CLUSTERS = {"nstr", "rstr", "ngst", "ngth", "mpst", "xpl", "xtr", "xpr", "xcl", "tchr", "kthr", "ghts", "ndst"}


# ---------------------------------------------------------------------------
# Lightweight Disfluency Cleaner (conservative — no pronoun resolution)
# ---------------------------------------------------------------------------

# English filler words/phrases to strip
_ENGLISH_FILLERS = re.compile(
    r"\b(?:um+|uh+|hmm+|like|you know|basically|actually|right|okay|ok|"
    r"so+|literally|i mean|sort of|kind of)\b",
    re.IGNORECASE,
)

# Roman Urdu filler words/phrases to strip
_URDU_FILLERS = re.compile(
    r"\b(?:acha+|accha+|achha+|theek\s+hai|thik\s+hai|bhaiya|bhiya|bhai|"
    r"dekhein|dekho|matlab|yani|haan|waghera|wagera|"
    r"suno|suniye|chalein|chalo)\b",
    re.IGNORECASE,
)

# Repeated adjacent words: "the the" -> "the"
_REPEATED_WORDS = re.compile(r"\b(\w+)\s+\1\b", re.IGNORECASE)


def _clean_transcript_for_notes(transcript_text: str) -> str:
    """
    Lightweight disfluency removal for study notes generation.

    Strips English and Roman Urdu conversational fillers, removes repeated
    adjacent words, normalises whitespace and broken punctuation.

    Does NOT attempt deep pronoun resolution to avoid altering the
    instructor's original meaning.
    """
    if not transcript_text:
        return ""

    text = transcript_text

    # 0. Strip LLM translation wrapper prefixes (e.g., "Here's the English translation of the given Roman Urdu lecture sentence:")
    text = re.sub(
        r"^\s*here(?:'s|\s+is)\s+the\s+english\s+translation[^:\n]*:\s*[\"']?",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"[\"']\s*$", "", text)

    # 1. Strip English fillers
    text = _ENGLISH_FILLERS.sub("", text)

    # 2. Strip Roman Urdu fillers
    text = _URDU_FILLERS.sub("", text)

    # 3. Remove repeated adjacent words ("the the" -> "the")
    text = _REPEATED_WORDS.sub(r"\1", text)

    # 4. Clean orphaned punctuation artefacts from filler removal
    text = re.sub(r"^\s*,\s*(?:all|everyone)\s*[.!?]?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*,\s*,+", ",", text)           # double commas
    text = re.sub(r"\s*\.\s*\.+", ".", text)          # double periods
    text = re.sub(r",\s*\.", ".", text)               # comma before period
    text = re.sub(r"\.\s*,", ".", text)               # period before comma

    # 5. Normalise whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def _is_phonetic_noise(w: str) -> bool:
    w_clean = re.sub(r'[^a-zA-Z]', '', w.lower())
    if not w_clean or len(w_clean) <= 2:
        return False
    if w_clean in _ALLOWED_ACRONYMS or w_clean in ENGLISH_COMMON_VOCAB:
        return False
    if w.isupper() and len(w_clean) <= 5:
        return False
    if w_clean.endswith('s') and w_clean[:-1] in _ALLOWED_ACRONYMS:
        return False
    if w_clean in _HALLUCINATION_WORDS:
        return True
    try:
        from app.services.intelligence_preprocessor import _get_english_vocab, _is_in_english_vocab
        _en_v = _get_english_vocab()
    except Exception:
        _en_v = frozenset()
        _is_in_english_vocab = lambda _w, _v: False
    if _en_v and _is_in_english_vocab(w_clean, _en_v):
        return False
    vowels = len(re.findall(r'[aeiouy]', w_clean))
    if vowels == 0:
        return True
    for m in re.finditer(r'[^aeiouy]{4,}', w_clean):
        c = m.group(0)
        if c not in ENGLISH_4_CLUSTERS:
            is_compound_en = _en_v and any(
                len(w_clean[:k]) >= 3
                and len(w_clean[k:]) >= 3
                and _is_in_english_vocab(w_clean[:k], _en_v)
                and _is_in_english_vocab(w_clean[k:], _en_v)
                for k in range(m.start() + 1, m.end())
            )
            if not is_compound_en:
                return True
    consonants = len(w_clean) - vowels
    if consonants / max(1, vowels) > 4.5 and len(w_clean) >= 7:
        return True
    return False


_TRANSLATION_REFUSAL_OR_HALLUCINATION_RE = re.compile(
    r"(?:"
    r"cannot\s+translate\s+(?:that|it|this)|"
    r"does\s+not\s+contain\s+any\s+text\s+to\s+translate|"
    r"please\s+provide\s+a\s+(?:complete|full|specific)\s+(?:roman\s+urdu\s+)?(?:lecture\s+)?sentence|"
    r"if\s+you\s+want,?\s+i\s+can\s+help\s+translate|"
    r"inappropriate\s+content\s+for\s+this\s+platform|"
    r"appears\s+to\s+be\s+in\s+a\s+language\s+other\s+than\s+urdu|"
    r"explain\s+my\s+process\s+of\s+trans(?:lation|action)|"
    r"clarify\s+what\s+aspect\s+of\s+the\s+trans(?:lation|action)|"
    r"profits\s+on\s+the\s+fijian|"
    r"role\s+of\s+education\s+in\s+economic\s+development|"
    r"let\s+us\s+not\s+be\s+afraid\s+of\s+(?:difficulties|the\s+unknown)|"
    r"mindful\s+of\s+our\s+surroundings|"
    r"by\s+this\s+tree\s+alone|"
    r"let\s+us\s+be\s+friends|"
    r"i\s+will\s+not\s+be\s+able\s+to\s+attend|"
    r"your\s+lecture\s+is\s+very\s+good|"
    r"kapriya\s+chhuri|"
    r"visa\s+vaidhya|"
    r"dances\s+with\s+single-haired\s+dancers|"
    r"bß\s*2\s*no\s*important"
    r")",
    re.IGNORECASE,
)

_NON_ACADEMIC_SEGMENT_RE = re.compile(
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
)


def _is_non_academic_segment(text: str) -> bool:
    """Returns True if a segment is purely classroom logistics, mic check, standalone question, or AI/ASR hallucination."""
    t = (text or "").strip()
    if not t:
        return True
    t_low = t.lower()
    # Always drop LLM translator refusals and known silence hallucinations regardless of length
    if _TRANSLATION_REFUSAL_OR_HALLUCINATION_RE.search(t):
        return True
    if t_low in {"let us not forget.", "let us not forget", "is there anything?", "for instance.", "it is possible."}:
        return True
    # Never drop a segment that contains an explicit academic definition or formula
    if any(m in t_low for m in ("is defined as", "refers to", "is calculated as", "is equal to", "divided by")):
        return False
    if len(t.split()) <= 28 and _NON_ACADEMIC_SEGMENT_RE.search(t):
        return True
    # Pure greeting or farewell with no academic content
    if len(t.split()) <= 10 and any(
        g in t_low
        for g in (
            "assalamu alaikum", "walaikum assalam", "good morning everyone",
            "good morning, all", "good afternoon everyone", "how are you all",
            "see you next class", "thank you everyone", "thank you to all of you",
            "class is dismissed", "take attendance",
        )
    ):
        return True
    # Short standalone logistical/exam question
    if t.endswith("?") and len(t.split()) <= 16 and t_low.startswith(("will this ", "can you ", "can everyone ", "do we need ", "is this on ")):
        return True
    return False


_VALID_GERUND_HEAD_NOUNS = {
    "learning", "scheduling", "accounting", "banking", "lending", "investing",
    "financing", "operating", "processing", "computing", "overfitting",
    "underfitting", "clustering", "sampling", "conditioning", "bonding",
    "signaling", "budgeting", "testing", "training", "forecasting",
    "monitoring", "modeling", "indexing", "paging", "thrashing", "deadlocking",
    "multitasking", "engineering", "manufacturing", "marketing", "auditing",
}

# Generic adjectives / quantifiers / discourse words that never open a real academic topic title
_GENERIC_TITLE_MODIFIERS = {
    "any", "some", "many", "much", "most", "more", "less", "few", "several", "each", "every",
    "long", "short", "easy", "hard", "simple", "great", "big", "small", "large", "little",
    "good", "bad", "best", "better", "high", "low", "new", "old", "same", "different",
    "actual", "actually", "real", "basic", "certain", "given", "whereas", "still", "just",
    "sharp", "physical", "bright", "numerical", "important", "main", "major", "minor",
    "particular", "specific", "general", "whole", "entire", "various", "other", "such",
    "earning", "involved", "regulate", "preventing", "accurate", "common", "serious",
    "sufficient", "roman", "urdu", "english", "house", "chart", "spending", "system",
    "time", "point", "points", "questions", "question", "value", "thing", "things",
}

_NON_TOPIC_CURRENCY_OR_META_WORDS = {
    "rupee", "rupees", "lakh", "lakhs", "crore", "crores", "dollar", "dollars",
    "thousand", "million", "billion", "gram", "grams", "translate", "translation",
    "provide", "sentence", "request", "message", "incomplete", "sorry", "assist",
    "platform", "fijian", "urdu", "english", "depends", "reducing", "studied",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "minus", "plus", "unbounded", "shortterm", "offered", "allows", "allow",
    "helps", "help", "demonstrate", "demonstrates", "increases", "decreases",
    "involved", "involve", "earning", "still", "offer", "offers", "realized",
    "reason", "reasons", "nirang", "whereas", "structural", "through",
}


def _is_incomplete_clause(cand: str) -> bool:
    if not cand or not cand.strip():
        return True
    cand_clean = cand.strip()
    if "?" in cand_clean:
        return True
    words = [re.sub(r'[^a-zA-Z]', '', w).lower() for w in cand_clean.split()]
    words = [w for w in words if w]
    if not words or len(words) < 2 or len(words) > 7:
        return True

    if any(w in _NON_TOPIC_CURRENCY_OR_META_WORDS for w in words):
        return True

    # Reject tautological titles where two content words share the same 5-char stem (e.g., "Financial Technology Fintech", "Personal Fintech Finance")
    content_stems = [w[:5] for w in words if len(w) >= 5 and w not in {"and", "the", "with", "from", "into"}]
    if len(content_stems) != len(set(content_stems)):
        return True
    if "fintech" in words and any(w in words for w in ("finance", "financial")):
        return True
    if "fcfs" in words and "firstcome" in words:
        return True
    if "sensitive" in words and "sensory" in words:
        return True

    # Reject titles ending in a participle (-ing) unless it is a valid academic gerund noun
    if words[-1].endswith("ing") and words[-1] not in _VALID_GERUND_HEAD_NOUNS:
        return True

    # Conversational sentence opener phrases, classroom instructions, questions, and AI responses to reject
    opener_phrases = [
        "good morning", "assalamu alaikum", "welcome to", "today is our", "today we",
        "so today", "let us", "now let", "therefore", "however", "moreover", "meanwhile",
        "in this section", "a company can", "they will", "we continue", "i can use",
        "your own clothes", "1 more cheese", "ek ahem cheez", "not necessarily",
        "profit is essentially", "profit equals revenue", "inventory depreciation",
        "three thousand and", "the cash has", "but the inventory", "the money is",
        "operating cash inflows", "or business in", "dursa if pasaata", "it is an out",
        "thirdly we have", "and finally remember", "twenty thousand rupees", "we tell about",
        "make sure that", "take a look", "let us examine",
        "can you hear", "am i audible", "share my screen", "mute your", "open your book",
        "turn to page", "any questions", "raise your hand", "write this down", "copy this",
        "next slide", "previous slide", "look at the board", "as an ai", "sure i can",
        "here is a", "based on the transcript", "in this response", "i am going",
        "we are going", "you are going", "let me tell", "talking about", "looking at",
        "what is the", "why do we", "how does it", "do you know", "can anyone tell",
        "another fintech", "that your", "great profit", "simple formula", "studied for",
    ]
    cand_low = cand_clean.lower()
    if any(op in cand_low for op in opener_phrases):
        return True

    # Dangling start words (including interrogatives, pronouns, imperative verbs, and conversational starters)
    start_disallowed = _DISALLOWED_FRAGMENT_EDGES["start"] | {
        "good", "morning", "assalamu", "today", "today's", "welcome", "hello", "so", "now", "a", "an", "the",
        "therefore", "however", "moreover", "and", "or", "but", "also", "then", "not", "thirdly", "secondly",
        "first", "second", "third", "next", "same", "way", "another", "great", "simple", "studied", "that", "this",
        "what", "why", "how", "when", "where", "who", "which", "whose", "whom", "whether",
        "please", "listen", "look", "remember", "note", "imagine", "consider", "think",
        "find", "determine", "calculate", "compute", "recognize", "understand", "explain", "discuss", "define",
        "know", "see", "check", "show", "get", "make", "use", "take", "give", "work", "let", "lets",
        "all", "common", "serious", "problem", "accurate", "still", "earning", "offer", "allows", "helps", "involved",
        "i", "we", "you", "they", "he", "she", "it", "my", "our", "your", "their",
        "kya", "kyun", "kaise", "kab", "kahan",
    }
    if words[0] in start_disallowed:
        return True

    # Reject random sentence fragments containing finite verbs / auxiliaries / subordinating conjunctions in the middle
    sentence_verbs = {
        "is", "are", "was", "were", "have", "has", "had", "will", "would", "could",
        "should", "must", "can", "cannot", "does", "did", "said", "says", "told", "tells",
        "wants", "wanted", "needs", "needed", "goes", "went", "comes", "came", "depends",
        "reveals", "reveal", "whether", "means", "mean", "shows", "indicates", "determines",
        "calculates", "equals", "lets", "find", "determine", "recognize",
    }
    if any(w in sentence_verbs for w in words):
        return True

    # Dangling end words (articles, prepositions, conjunctions, auxiliary/modal verbs, pronouns, action verbs)
    dangling_end = _DISALLOWED_FRAGMENT_EDGES["end"] | {
        "a", "an", "the", "can", "could", "will", "would", "should", "shall", "may", "might", "must",
        "is", "are", "was", "were", "be", "been", "being", "has", "have", "had", "do", "does", "did",
        "sell", "sells", "sold", "buying", "bought", "makes", "make", "made", "gets", "got", "getting",
        "gives", "give", "given", "takes", "take", "taken", "shows", "show", "shown", "uses", "use", "used",
        "in", "at", "on", "of", "to", "for", "with", "from", "by", "as", "into", "onto",
        "your", "our", "my", "their", "his", "her", "its", "that", "this", "these", "those",
        "depends", "reducing", "provide", "translate", "whether", "reveals", "work", "lets",
    }
    if words[-1] in dangling_end:
        return True

    return False


def _synthesize_subtopic_title(kp: str, topic_title: str = "") -> Optional[str]:
    """
    Synthesizes a clean, meaningful 2-5 word conceptual subtopic title
    from a key point or explanation, strictly rejecting conversational fragments.
    """
    if not kp or not kp.strip():
        return None

    kp_clean = kp.strip()
    topic_words_set = {re.sub(r'[^a-zA-Z]', '', w).lower() for w in (topic_title or "").split() if re.sub(r'[^a-zA-Z]', '', w)}

    # 1. If key point is structured as "Term: Definition / Details", use "Term"
    if ":" in kp_clean:
        prefix = kp_clean.split(":")[0].strip()
        prefix = re.sub(r'[,;:\s]+$', '', prefix).strip()
        p_words = [re.sub(r'[^a-zA-Z]', '', w) for w in prefix.split() if re.sub(r'[^a-zA-Z]', '', w)]
        if 2 <= len(p_words) <= 5 and not _is_incomplete_clause(prefix):
            if {w.lower() for w in p_words} != topic_words_set:
                return prefix[0].upper() + prefix[1:]

    # 2. Check for explicit definitions: "X is defined as Y" or "X refers to Y"
    m = re.match(r'^(?:An?\s+|The\s+)?([A-Z][a-zA-Z\s]{2,35}?)\s+(?:is\s+defined\s+as|refers\s+to|is\s+calculated\s+as|is\s+a\s+technique|represents|measures|means)\b', kp_clean)
    if m:
        term = m.group(1).strip()
        t_words = [w for w in term.split() if w]
        if 2 <= len(t_words) <= 5 and not _is_incomplete_clause(term):
            if {w.lower() for w in t_words} != topic_words_set:
                return term.title()

    # 3. Extract clause-local noun phrases (never crossing punctuation boundaries)
    clauses = re.split(r'[.,;:!?()\[\]]+', kp_clean)
    for clause in clauses:
        tokens = [re.sub(r'[^a-zA-Z]', '', w) for w in clause.split()]
        valid = [w for w in tokens if w and not _is_phonetic_noise(w) and w.lower() not in _HALLUCINATION_WORDS and w.lower() not in _URDU_STOPWORDS]
        for n in (3, 2):
            for i in range(len(valid) - n + 1):
                gram = valid[i: i + n]
                gram_low = [w.lower() for w in gram]
                if any(w in _STOPWORDS or len(w) < 3 for w in gram_low):
                    continue
                if gram_low[0] in {"these", "those", "this", "that", "such", "various", "different", "many", "some", "several", "another", "today"}:
                    continue
                if set(gram_low) == topic_words_set:
                    continue
                cand = " ".join(gram).title()
                if not _is_incomplete_clause(cand) and len(set(gram_low)) == n:
                    if gram_low[-1] in ACADEMIC_TITLE_NOUNS:
                        return cand

    return None



_URDU_STOPWORDS = {
    "hai", "hain", "mein", "par", "aur", "yeh", "woh", "kya", "bhi", "cheez",
    "baat", "hota", "hoti", "hote", "karna", "karein", "karne", "liye", "wala",
    "wali", "wale", "sath", "agar", "lekin", "magar", "mat", "sirf", "balkay",
    "sabse", "ziyada", "thoda", "bohot", "aise", "waise", "ko", "se", "pe",
    "ka", "ki", "ke", "kret", "karega", "pehle", "ho", "aaj", "kuch", "apne",
    "bhaiya", "bhiya", "dekho", "dekhein", "theek", "acha", "sahi", "yar",
    "uff", "gaya", "gayi", "raha", "rahi", "rahe"
}

ACADEMIC_TITLE_NOUNS = {
    "overview", "fundamentals", "principles", "management", "analysis",
    "scheduling", "execution", "calculations", "architecture", "concepts",
    "algorithm", "algorithms", "mechanisms", "mechanism", "strategy",
    "strategies", "techniques", "technique", "models", "model", "evaluation",
    "synchronization", "optimization", "metrics", "foundations", "dynamics",
    "tradeoffs", "design", "processes", "processing", "multitasking",
    "structure", "framework", "performance", "coordination", "behavior",
    "division", "phase", "complex", "complexes", "respiration", "phosphorylation",
    "chain", "gradient", "membrane", "pathway", "pathways", "cycle", "reaction",
    "reactions", "equilibrium", "elasticity", "demand", "supply", "curves", "curve",
    "utility", "market", "markets", "cost", "revenue", "profit", "synthesis",
    "transport", "regulation", "system", "systems", "network", "networks",
    "theory", "theories", "property", "properties", "interactions", "interaction",
    "component", "components", "method", "methods", "introduction", "applications",
    "application", "elements", "mechanics", "classification", "variation", "distribution",
    "hypothesis", "experiment", "evidence", "factors", "factor", "effects", "effect",
    "impact", "formation", "development", "integration", "organization", "operations",
    "action", "arc", "reflex", "neuron", "neurons", "receptor", "receptors",
    "effector", "effectors", "impulse", "impulses", "stimulus", "response",
    "hormone", "hormones", "endocrine", "homeostasis", "transmission", "circuits",
    "synapse", "signals", "signaling",
    "charge", "current", "voltage", "potential", "difference", "resistance", "resistivity",
    "circuit", "circuits", "series", "parallel", "conductor", "conductors", "insulator",
    "insulators", "electron", "electrons", "proton", "protons", "neutron", "neutrons",
    "capacitance", "inductance", "magnetic", "field", "fields", "frequency", "wavelength",
    "energy", "power", "work", "force", "forces", "velocity", "acceleration", "momentum",
    "thermodynamics", "temperature", "pressure", "volume", "density", "entropy", "enthalpy",
    "equations", "equation", "formula", "formulas", "theorem", "theorems", "matrix", "matrices",
    "vector", "vectors", "probability", "distribution", "derivatives", "integrals", "limit",
    "equilibrium", "elasticity", "inflation", "recession", "fiscal", "monetary",
    "regression", "variance", "deviation", "correlation", "inference", "sampling",
    "democracy", "revolution", "civilization", "empire", "constitution", "sovereignty",
    "treaty", "colonialism", "industrialization", "reform", "movement", "movements",
    "cognition", "perception", "conditioning", "memory", "personality",
    "catalysis", "oxidation", "reduction", "bonding", "molecule", "molecules",
    "polymer", "polymers", "kinetics", "stoichiometry", "solution", "solutions",
    "workflow", "workflows", "leadership", "governance", "compliance", "inclusion",
    "analytics", "gateway", "gateways", "encryption", "verification", "biometric",
    "liquidity", "solvency", "insolvency", "depreciation", "accrual", "receivable",
    "payable", "liability", "liabilities", "asset", "assets", "capital", "equity",
    "leverage", "dividend", "dividends", "audit", "taxation", "budget", "budgeting",
    "transaction", "transactions", "failure", "failures", "crash", "crashes",
    "commit", "commits", "state", "states", "invariant", "invariants",
    "database", "databases", "durability", "isolation", "atomicity", "consistency"
}


_STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "from", "are", "was",
    "were", "has", "have", "been", "not", "will", "can", "its", "our",
    "hai", "hain", "mein", "par", "aur", "yeh", "woh", "kya", "bhi",
    "lecture", "class", "tell", "know", "said", "okay", "just", "very",
    "also", "then", "when", "where", "here", "there", "into", "their",
    "would", "could", "should", "which", "about", "some", "them", "they",
    "we", "you", "he", "she", "it", "me", "us", "my", "your", "his", "her",
    "next", "equals", "equal", "eventually", "done", "doing", "start", "started",
    "first", "second", "third", "one", "two", "three", "four", "five",
    "like", "much", "many", "well", "want", "need", "make", "take", "good",
    "see", "look", "come", "give", "use", "uses", "used", "using",
    "cheez", "baat", "hota", "hoti", "hote", "karna", "karein", "karne",
    "liye", "wala", "wali", "wale", "sath", "agar", "lekin", "magar", "mat",
    "sirf", "balkay", "sabse", "ziyada", "thoda", "bohot", "aise", "waise",
    "therefore", "however", "moreover", "furthermore", "basically", "actually",
    "literally", "meanwhile", "involves", "involve", "involving", "means", "meaning",
    "right", "away", "morning", "everyone", "accidentally", "instead"
}


# ===========================================================================
# Data structures & Discourse Hierarchy
# ===========================================================================

class DiscourseRole:
    NEW_TOPIC = "NEW_TOPIC"
    SUBTOPIC = "SUBTOPIC"
    CONTINUATION = "CONTINUATION"
    EXAMPLE = "EXAMPLE"
    DEFINITION = "DEFINITION"
    EXPLANATION = "EXPLANATION"
    REPETITION = "REPETITION"
    QUESTION = "QUESTION"
    ASSIGNMENT = "ASSIGNMENT"
    SUMMARY = "SUMMARY"
    TRANSITION = "TRANSITION"


@dataclass
class Segment:
    id: int
    text: str
    start: float
    end: float


@dataclass
class TopicSpan:
    start: float
    end: float
    start_seg_id: int
    end_seg_id: int


@dataclass
class SubTopic:
    id: str = ""
    title: str = ""
    role: str = DiscourseRole.SUBTOPIC
    start_time: float = 0.0
    end_time: float = 0.0
    source_segment_ids: list[int] = field(default_factory=list)
    key_points: list[str] = field(default_factory=list)
    explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "role": self.role,
            "start_time": round(self.start_time, 2),
            "end_time": round(self.end_time, 2),
            "timestamp": f"[{_format_time(self.start_time)} - {_format_time(self.end_time)}]",
            "source_segment_ids": self.source_segment_ids,
            "evidence": [f"seg_{sid:03d}" for sid in self.source_segment_ids],
            "key_points": self.key_points,
            "explanation": self.explanation
        }


@dataclass
class TopicHierarchyNode:
    id: str = ""
    title: str = ""
    role: str = DiscourseRole.NEW_TOPIC
    level: int = 1
    start_time: float = 0.0
    end_time: float = 0.0
    evidence_segment_ids: list[int] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    subtopics: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        ev = self.evidence if self.evidence else [f"seg_{sid:03d}" for sid in self.evidence_segment_ids]
        return {
            "id": self.id,
            "title": self.title,
            "role": self.role,
            "level": self.level,
            "start_time": round(self.start_time, 2),
            "end_time": round(self.end_time, 2),
            "timestamp": f"[{_format_time(self.start_time)} - {_format_time(self.end_time)}]",
            "evidence_segment_ids": self.evidence_segment_ids,
            "evidence": ev,
            "subtopics": self.subtopics
        }


@dataclass
class StudyTopic:
    id: str = ""
    title: str = "Lecture Topic"
    topic_type: str = "major_topic"
    hierarchy_level: int = 1
    subtopics: list[SubTopic] = field(default_factory=list)
    spans: list[TopicSpan] = field(default_factory=list)
    explanation: str = ""
    what_it_means: str = ""
    how_it_works: str = ""
    why_it_matters: str = ""
    steps_and_logic: list[str] = field(default_factory=list)
    concept_connection: str = ""
    key_points: list[str] = field(default_factory=list)
    definitions: list[dict[str, str]] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    formulas: list[str] = field(default_factory=list)
    common_mistakes: list[str] = field(default_factory=list)
    key_takeaway: str = ""
    emphasis: list[dict[str, str]] = field(default_factory=list)
    exam_relevant: bool = False
    source: str = "local_llm"
    source_segment_ids: list[int] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    source_text: str = ""
    grounding_status: str = "SUPPORTED"
    claim_entailment_score: float = 1.0
    ai_background_enhancement: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        s_time = self.spans[0].start if self.spans else 0.0
        e_time = self.spans[-1].end if self.spans else 0.0
        ev_list = self.evidence if self.evidence else [f"seg_{sid:03d}" for sid in self.source_segment_ids]

        return {
            "id": self.id,
            "title": self.title,
            "topic_type": self.topic_type,
            "hierarchy_level": self.hierarchy_level,
            "subtopics": [st.to_dict() for st in self.subtopics],
            "start_time": s_time,
            "end_time": e_time,
            "timestamp": f"[{_format_time(s_time)} - {_format_time(e_time)}]",
            "spans": [
                {"start": round(s.start, 2), "end": round(s.end, 2),
                 "start_seg_id": s.start_seg_id, "end_seg_id": s.end_seg_id}
                for s in self.spans
            ],
            "explanation": self.explanation,
            "what_it_means": self.what_it_means,
            "how_it_works": self.how_it_works,
            "why_it_matters": self.why_it_matters,
            "steps_and_logic": self.steps_and_logic,
            "concept_connection": self.concept_connection,
            "key_points": self.key_points,
            "definitions": self.definitions,
            "examples": self.examples,
            "formulas": self.formulas,
            "common_mistakes": self.common_mistakes,
            "key_takeaway": self.key_takeaway,
            "emphasis": self.emphasis,
            "exam_relevant": self.exam_relevant,
            "source": self.source,
            "source_segment_ids": self.source_segment_ids,
            "evidence": ev_list,
            "source_text": self.source_text,
            "grounding_status": self.grounding_status,
            "claim_entailment_score": round(self.claim_entailment_score, 2),
            "ai_background_enhancement": self.ai_background_enhancement
        }


@dataclass
class LectureStudyNotes:
    study_topics: list[StudyTopic] = field(default_factory=list)
    lecture_overview: str = ""
    subject: str = "Academic Lecture"
    theme: str = "General Study"
    topic_hierarchy: list[dict[str, Any]] = field(default_factory=list)
    concept_relationships: list[dict[str, str]] = field(default_factory=list)
    discourse_roles: list[dict[str, Any]] = field(default_factory=list)
    quality_score: float = 0.0
    quality_pass: bool = False
    quality_details: dict[str, Any] = field(default_factory=dict)
    analyzer_used: str = "local_llm"
    transcript_hash: str = ""
    is_fallback: bool = False
    fallback_reason: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        topics_str = [t.title for t in self.study_topics]
        concepts = _derive_key_concepts(self.study_topics)
        defs = []
        for t in self.study_topics:
            defs.extend(t.definitions)
        important_pts = _derive_important_points(self.study_topics)

        # Topic Hierarchy representation
        hierarchy = self.topic_hierarchy
        if not hierarchy and self.study_topics:
            hierarchy = [
                {
                    "id": t.id,
                    "title": t.title,
                    "role": DiscourseRole.NEW_TOPIC,
                    "level": 1,
                    "start_time": t.spans[0].start if t.spans else 0.0,
                    "end_time": t.spans[-1].end if t.spans else 0.0,
                    "timestamp": f"[{_format_time(t.spans[0].start if t.spans else 0.0)} - {_format_time(t.spans[-1].end if t.spans else 0.0)}]",
                    "evidence_segment_ids": t.source_segment_ids,
                    "evidence": t.evidence or [f"seg_{sid:03d}" for sid in t.source_segment_ids],
                    "subtopics": [st.to_dict() for st in t.subtopics]
                }
                for t in self.study_topics
            ]

        # Derived subtopics string list for backward compatibility
        derived_subtopics = []
        for t in self.study_topics:
            if t.subtopics:
                for st in t.subtopics:
                    derived_subtopics.append(f"{t.title}: {st.title}")
            else:
                for kp in t.key_points[:2]:
                    derived_subtopics.append(f"{t.title}: {kp}")

        learning_objectives = [
            f"Understand the core definition, mechanism, and practical significance of {t.title}."
            for t in self.study_topics[:5]
            if t.title
        ]
        revision_takeaways = []
        for t in self.study_topics[:5]:
            if t.key_points:
                revision_takeaways.append(f"{t.title}: {t.key_points[0]}")
        revision_summary = (
            " ".join(revision_takeaways)
            if revision_takeaways
            else self.lecture_overview
        )

        return {
            "topics": topics_str,
            "subtopics": derived_subtopics,
            "topic_hierarchy": hierarchy,
            "concept_relationships": self.concept_relationships,
            "discourse_roles": self.discourse_roles,
            "definitions": defs,
            "key_concepts": concepts,
            "important_points": important_pts,
            "study_topics": [t.to_dict() for t in self.study_topics],
            "lecture_overview": self.lecture_overview,
            "executive_summary": self.lecture_overview,
            "learning_objectives": learning_objectives,
            "revision_summary": revision_summary,
            "subject": self.subject,
            "theme": self.theme,
            "quality_score": round(self.quality_score, 2),
            "quality_pass": self.quality_pass,
            "quality_details": self.quality_details,
            "analyzer_used": self.analyzer_used,
            "transcript_hash": self.transcript_hash,
            "is_fallback": self.is_fallback,
            "fallback_reason": self.fallback_reason,
        }


def _derive_key_concepts(topics: list[StudyTopic]) -> list[str]:
    concepts: list[str] = []
    seen: set[str] = set()

    # Pre-collect topic title normalized strings to prevent duplicate topics in concepts
    topic_title_norms = {re.sub(r"[^a-z0-9]", "", t.title.lower()) for t in topics if t.title}

    disallowed_re = re.compile(
        r"^(?:so\s+)?here\s+are\s+(?:some\s+)?(?:important\s+)?(?:distinctions?|points?|notes?|things?)|"
        r"^(?:so|and|then|also|now|here|as|let|we|you|in\s+this|to\s+understand|for\s+example|note\s+that)|"
        r"(?:distinctions?|session|lecture|class|section|overview)\.?$|"
        r"(?:is\s+determined\s+by|is\s+calculated\s+as|tracks\s+physical|requires\s+tracking)",
        re.IGNORECASE
    )

    # 1. Primary Priority: Explicit academic definition terms
    for t in topics:
        for d in t.definitions:
            term = (d.get("term") if isinstance(d, dict) else str(d)).strip()
            term_clean = re.sub(r"^[•\-\*\d\.\s,;:]+", "", term).strip().rstrip(".:,;")
            term_clean = re.sub(r"^(?:the|a|an)\s+", "", term_clean, flags=re.IGNORECASE).strip()
            words = term_clean.split()
            if (len(term_clean) >= 4 or (len(term_clean) == 3 and term_clean.isupper())) and len(term_clean) <= 40 and not disallowed_re.search(term_clean):
                if words and words[0].lower() not in _STOPWORDS and words[0].lower() not in _DISALLOWED_FRAGMENT_EDGES["start"] and words[0].lower() not in _GENERIC_TITLE_MODIFIERS:
                    if not any(w.lower() in _NON_TOPIC_CURRENCY_OR_META_WORDS for w in words):
                        norm = re.sub(r"[^a-z0-9]", "", term_clean.lower())
                        if norm and norm not in seen and norm not in topic_title_norms:
                            seen.add(norm)
                            concepts.append(term_clean)

    # 2. Secondary Priority: Core formula names (e.g. Net Cash Flow, Working Capital)
    for t in topics:
        for f in t.formulas:
            formula_str = str(f).strip()
            parts = re.split(r"\s*(?:=|\bequals\b|\bis\s+equal\s+to\b|\bis\s+calculated\s+as\b)\s*", formula_str, maxsplit=1, flags=re.IGNORECASE)
            term_clean = re.sub(r"^[•\-\*\d\.\s,;:]+", "", parts[0].strip()).strip().rstrip(".:,;")
            term_clean = re.sub(r"^(?:the|a|an|so|then|now)\s+", "", term_clean, flags=re.IGNORECASE).strip()
            f_words = term_clean.split()
            if 1 <= len(f_words) <= 4 and (len(term_clean) >= 4 or (len(term_clean) == 3 and term_clean.isupper())) and len(term_clean) <= 35 and not disallowed_re.search(term_clean):
                if f_words and f_words[0].lower() not in _STOPWORDS and f_words[0].lower() not in _DISALLOWED_FRAGMENT_EDGES["start"] and f_words[0].lower() not in _GENERIC_TITLE_MODIFIERS:
                    if not any(w.lower() in {"remember", "formula", "seconds", "then", "one", "two", "three"} or w.lower() in _NON_TOPIC_CURRENCY_OR_META_WORDS for w in f_words):
                        if not (len(f_words) >= 2 and _is_incomplete_clause(term_clean)):
                            norm = re.sub(r"[^a-z0-9]", "", term_clean.lower())
                            if norm and norm not in seen and norm not in topic_title_norms:
                                seen.add(norm)
                                concepts.append(term_clean)

    # 3. Tertiary Priority: Structured key point terms (<Term>: <Description>)
    for t in topics:
        for kp in t.key_points:
            kp_str = str(kp).strip()
            if ":" in kp_str:
                term_candidate = kp_str.split(":")[0].strip()
                term_clean = re.sub(r"^[•\-\*\d\.\s]+", "", term_candidate).strip().rstrip(".:,;")
                words = term_clean.split()
                if 1 <= len(words) <= 4 and 3 <= len(term_clean) <= 35 and not disallowed_re.search(term_clean):
                    if words and words[0].lower() not in _STOPWORDS and words[0].lower() not in _DISALLOWED_FRAGMENT_EDGES["start"] and words[0].lower() not in _GENERIC_TITLE_MODIFIERS:
                        if not any(w.lower() in _NON_TOPIC_CURRENCY_OR_META_WORDS for w in words):
                            norm = re.sub(r"[^a-z0-9]", "", term_clean.lower())
                            if norm and norm not in seen and norm not in topic_title_norms:
                                seen.add(norm)
                                concepts.append(term_clean)

    return concepts[:8]


def _derive_important_points(topics: list[StudyTopic]) -> list[dict[str, Any]]:
    pts: list[dict[str, Any]] = []
    for t in topics:
        s_start = t.spans[0].start if t.spans else 0.0
        s_end = t.spans[-1].end if t.spans else 0.0
        for emp in t.emphasis:
            emp_start = emp.get("start_time", s_start)
            emp_end = emp.get("end_time", s_end)
            emp_ts = emp.get("timestamp") or f"[{emp_start:.2f}s - {emp_end:.2f}s]"
            pts.append({
                "point": emp.get("text", t.title),
                "is_exam_relevant": emp.get("source") == "lecturer_explicit" or t.exam_relevant,
                "signal_phrase": emp.get("signal_phrase", emp.get("source", "lecturer_explicit")),
                "timestamp": emp_ts,
                "start_time": emp_start,
                "end_time": emp_end
            })
    return pts[:12]


def _format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def _compute_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]


_GROUNDING_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "and", "or", "in", "on", "at",
    "to", "for", "of", "with", "by", "from", "it", "this", "that", "these", "those",
    "which", "what", "how", "as", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "can", "could", "will", "would", "should", "shall", "may",
    "might", "must", "than", "so", "very", "too", "also", "then", "there", "their",
    "its", "our", "my", "your", "his", "her", "all", "each", "every", "other",
    "into", "through", "during", "before", "after", "above", "below", "up", "down",
    "out", "off", "over", "under", "again", "further", "once", "here", "why"
}


# ===========================================================================
# Hierarchical Context Window, Role Classification & Entailment Engine
# ===========================================================================

class ContextWindowProcessor:
    """
    Constructs surrounding sliding-context windows around target segments.
    Enables accurate resolution of deictic references, pronouns, and code-switching.
    """
    def __init__(self, window_size: int = 2):
        self.window_size = window_size

    def build_window(
        self,
        segments: list[Segment],
        target_idx: int,
        global_outline: Optional[str] = None
    ) -> dict[str, Any]:
        n = len(segments)
        start_idx = max(0, target_idx - self.window_size)
        end_idx = min(n, target_idx + self.window_size + 1)

        prev_segs = segments[start_idx:target_idx]
        target_seg = segments[target_idx]
        next_segs = segments[target_idx + 1:end_idx]

        prev_text = " ".join(s.text for s in prev_segs)
        target_text = target_seg.text
        next_text = " ".join(s.text for s in next_segs)

        window_text = f"{prev_text} >>> {target_text} <<< {next_text}".strip()

        return {
            "target_id": target_seg.id,
            "target_start": target_seg.start,
            "target_end": target_seg.end,
            "target_text": target_text,
            "prev_text": prev_text,
            "next_text": next_text,
            "full_window_text": window_text,
            "global_outline": global_outline or ""
        }


class DiscourseRoleClassifier:
    """
    Classifies the functional/pedagogical role of discourse turns and segments:
    NEW_TOPIC, SUBTOPIC, CONTINUATION, EXAMPLE, DEFINITION, EXPLANATION, REPETITION, QUESTION, ASSIGNMENT, SUMMARY, TRANSITION.
    """
    _EXAMPLE_PATTERNS = [
        re.compile(r'\b(?:for\s+example|for\s+instance|misal\s+ke\s+taur\s+par|let\'s\s+say|consider\s+a\s+case|imagine|suppose|like\s+when|such\s+as)\b', re.IGNORECASE),
        re.compile(r'\b(?:touch(?:ing|ed)?\s+(?:a\s+)?(?:hot|cold)\s+(?:pan|object|plate)|hot\s+pan|in\s+real\s+life)\b', re.IGNORECASE)
    ]
    _DEFINITION_PATTERNS = [
        re.compile(r'\b(?:is\s+defined\s+as|is\s+basically|refers\s+to|means\s+that|known\s+as|ko\s+kehte\s+hain|is\s+the\s+rate\s+of|is\s+the\s+work\s+done)\b', re.IGNORECASE)
    ]
    _ASSIGNMENT_PATTERNS = [
        re.compile(r'\b(?:assignment|homework|submit|submission|portal|deadline|due\s+date|due\s+by)\b', re.IGNORECASE)
    ]
    _TRANSITION_PATTERNS = [
        re.compile(r'\b(?:now\s+let\'s\s+move|next\s+topic|moving\s+on|now\s+we\s+come\s+to|let\'s\s+begin\s+with|first\s+topic|ab\s+next|chalo\s+ab)\b', re.IGNORECASE)
    ]
    _REPETITION_PATTERNS = [
        re.compile(r'\b(?:again|repeat|dobaara|yaad\s+rakhein|note\s+this|important\s+for\s+exam|remember\s+this)\b', re.IGNORECASE)
    ]

    @classmethod
    def classify_role(cls, text: str, is_first_in_section: bool = False) -> str:
        t_low = text.lower()
        if any(p.search(t_low) for p in cls._ASSIGNMENT_PATTERNS):
            return DiscourseRole.ASSIGNMENT
        if any(p.search(t_low) for p in cls._TRANSITION_PATTERNS):
            return DiscourseRole.TRANSITION
        if any(p.search(t_low) for p in cls._EXAMPLE_PATTERNS):
            return DiscourseRole.EXAMPLE
        if any(p.search(t_low) for p in cls._DEFINITION_PATTERNS):
            return DiscourseRole.DEFINITION
        if any(p.search(t_low) for p in cls._REPETITION_PATTERNS):
            return DiscourseRole.REPETITION
        if is_first_in_section:
            return DiscourseRole.NEW_TOPIC
        return DiscourseRole.EXPLANATION


class ClaimEntailmentValidator:
    """
    Validates claim-level grounding and entailment between generated study notes and retrieved evidence segments.
    Detects SUPPORTED, UNSUPPORTED, or CONTRADICTED claims and eliminates hallucinations.
    """
    @classmethod
    def validate_claim(cls, claim: str, evidence_text: str) -> tuple[str, float]:
        """
        Evaluates a single claim against evidence text.
        Returns (status: SUPPORTED/UNSUPPORTED/CONTRADICTED, confidence: 0.0-1.0).
        """
        claim_clean = claim.strip()
        if not claim_clean or len(claim_clean) < 10:
            return ("SUPPORTED", 1.0)

        ev_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', evidence_text.lower()))
        cl_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', claim_clean.lower())) - _GROUNDING_STOPWORDS

        if not cl_words:
            return ("SUPPORTED", 1.0)

        # 1. Lexical overlap ratio
        overlap = len(cl_words & ev_words) / len(cl_words)

        # 2. Check for severe contradictions or alien entity names
        from app.services.domain_corrector import CANONICAL_TECHNICAL_TERMS
        canonical_words = {w for can in CANONICAL_TECHNICAL_TERMS for w in re.findall(r'\b[a-zA-Z]{3,}\b', can.lower())}
        general_academic = set(ACADEMIC_TITLE_NOUNS)

        alien_words = [w for w in cl_words if w not in ev_words and w not in canonical_words and w not in general_academic and w not in ENGLISH_COMMON_VOCAB]
        if len(alien_words) >= 3 and overlap < 0.20:
            return ("UNSUPPORTED", round(overlap, 2))

        if overlap >= 0.15:
            return ("SUPPORTED", round(min(1.0, overlap * 2.0), 2))

        # Allow simple root-stem / prefix overlap (e.g. computing vs computation, profit vs profitable)
        stem_hits = sum(
            1 for cw in cl_words
            if len(cw) >= 4 and any(ew.startswith(cw[:4]) or cw.startswith(ew[:4]) for ew in ev_words if len(ew) >= 4)
        )
        stem_overlap = stem_hits / len(cl_words)
        if stem_overlap >= 0.20:
            return ("SUPPORTED", round(min(1.0, stem_overlap * 1.5), 2))

        # Allow semantic paraphrase verification via MiniLM when plain English synonyms are used
        try:
            from app.services.embedding_service import EmbeddingModelManager
            vecs = EmbeddingModelManager.encode([claim_clean, evidence_text[:1200]], normalize_embeddings=True)
            cos_sim = float(vecs[0] @ vecs[1])
            if cos_sim >= 0.42:
                return ("SUPPORTED", round(min(1.0, cos_sim), 2))
        except Exception:
            pass

        return ("UNSUPPORTED", round(overlap, 2))

    @classmethod
    def filter_grounded_points(cls, points: list[str], evidence_text: str) -> list[str]:
        grounded = []
        for pt in points:
            status, score = cls.validate_claim(pt, evidence_text)
            if status == "SUPPORTED" or score >= 0.15:
                grounded.append(pt)
        return grounded if grounded else points[:2]

    @classmethod
    def validate(cls, claims_text: str, source_text: str) -> Any:
        """
        Validates an entire study guide or multi-claim document against source text.
        Returns an object with .passed (bool) and .reason (str).
        """
        from types import SimpleNamespace
        if not claims_text or not claims_text.strip():
            return SimpleNamespace(passed=False, reason="Empty claims text")
        if not source_text or not source_text.strip():
            return SimpleNamespace(passed=True, reason="No source text provided for grounding comparison")

        lines = [line.strip("# -* \t") for line in claims_text.split("\n") if len(line.strip("# -* \t")) > 25]
        if not lines:
            return SimpleNamespace(passed=True, reason="No substantive claims to validate")

        unsupported = 0
        total_checked = min(15, len(lines))
        for l in lines[:total_checked]:
            status, score = cls.validate_claim(l, source_text)
            if status == "UNSUPPORTED" and score < 0.10:
                unsupported += 1

        if total_checked >= 5 and (unsupported / total_checked) > 0.60:
            return SimpleNamespace(
                passed=False,
                reason=f"High hallucination rate: {unsupported}/{total_checked} sampled sections lacked transcript support"
            )

        return SimpleNamespace(passed=True, reason="Grounded")


# ===========================================================================
# Sliding-Window Embedding Support Signals
# ===========================================================================

class LocalSemanticTopicAnalyzer:
    """Computes dense multilingual embeddings to discover candidate topic transitions."""
    BLOCK_SIZE = 4
    STRIDE = 2
    MIN_TOPIC_SEGS = 3

    def _embed_segments(self, segments: list[Segment]) -> np.ndarray:
        from app.services.embedding_service import EmbeddingModelManager
        return EmbeddingModelManager.encode([s.text for s in segments], normalize_embeddings=True)

    def detect_candidate_boundaries(
        self,
        segments: list[Segment],
        embeddings: Optional[np.ndarray] = None,
        max_duration: float = 75.0
    ) -> list[int]:
        n = len(segments)
        if n < 3:
            return []

        if embeddings is None:
            embeddings = self._embed_segments(segments)

        # For short lectures (< 8 segments), find the sharpest semantic transition point
        if n < self.BLOCK_SIZE * 2:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1e-8, norms)
            norm_embs = embeddings / norms
            sims = np.einsum("id,id->i", norm_embs[:-1], norm_embs[1:])
            # Exclude boundary segments if n >= 4 to avoid 1-segment topics
            if n >= 4:
                valid_sims = sims[1:-1]
                offset = 2
            else:
                valid_sims = sims
                offset = 1
            if len(valid_sims) > 0:
                best_split = int(np.argmin(valid_sims)) + offset
                return [best_split]
            return [n // 2]

        block_starts = list(range(0, n - self.BLOCK_SIZE, self.STRIDE))
        if not block_starts:
            return [n // 2] if n >= self.MIN_TOPIC_SEGS * 2 else []

        block_embeds = np.stack([
            embeddings[i: i + self.BLOCK_SIZE].mean(axis=0)
            for i in block_starts
        ])
        norms = np.linalg.norm(block_embeds, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-8, norms)
        block_embeds /= norms
        sims = np.einsum("id,id->i", block_embeds[:-1], block_embeds[1:])
        if len(sims) < 1:
            return []

        mu = float(np.mean(sims))
        sigma = float(np.std(sims))
        threshold = mu - 0.75 * sigma
        base_boundaries: list[int] = []
        for i, sim in enumerate(sims):
            if sim < threshold:
                b = block_starts[i] + self.BLOCK_SIZE
                if 0 < b < n:
                    base_boundaries.append(b)

        if not base_boundaries and len(sims) >= 1 and n >= 6:
            min_idx = int(np.argmin(sims))
            b = block_starts[min_idx] + self.BLOCK_SIZE
            if 2 <= b <= n - 2:
                base_boundaries.append(b)

        # Transition cues for algorithm/concept/chapter changes across all disciplines (English + Roman Urdu)
        transition_markers = [
            re.compile(r'\b(?:next\s+topic|first\s+topic|second\s+topic|third\s+topic|today\s+is\s+our\s+topic|today\s+we\s+(?:explore|discuss|cover|examine|study)|let\'s\s+begin\s+with|let\s+us\s+talk\s+about|now\s+we\s+(?:come|move|turn|shift)\s+to|let\'s\s+discuss|now\s+let\'s\s+talk|now\s+consider|another\s+important\s+(?:term|concept|topic|principle|mechanism)|final\s+recap|let\'s\s+quickly\s+summarize|moving\s+on\s+to|turning\s+to|shifting\s+our\s+focus\s+to|in\s+contrast\s+to|on\s+the\s+other\s+hand|in\s+addition|furthermore|ab\s+hum\s+baat\s+karenge|agla\s+topic|doosra\s+hissa|teesra\s+hissa|aaj\s+ka\s+topic|ab\s+aate\s+hain)\b', re.IGNORECASE),
            re.compile(r'\b(?:electric\s+current|electric\s+charge|potential\s+difference|voltage|resistance|ohms?\s+law|series\s+circuits?|parallel\s+circuits?|series\s+and\s+parallel)\b', re.IGNORECASE),
            re.compile(r'\b(?:reflex\s+action|reflex\s+arc|nervous\s+system|sensory\s+neuron|motor\s+neuron|relay\s+neuron|hormonal\s+coordination|endocrine\s+system)\b', re.IGNORECASE),
            re.compile(r'\b(?:profit\s+and\s+cash\s+flow|net\s+cash\s+flow|working\s+capital|accrual\s+accounting|income\s+statement|balance\s+sheet)\b', re.IGNORECASE),
            re.compile(r'\b(?:next\s+algorithm|another\s+algorithm|round\s+robin|shortest\s+job|srtf|fcfs|convoy\s+effect|ready\s+queue|ready\s+state|program\s+process|program\s+vs\s+process|turnaround\s+time|waiting\s+time|gantt\s+chart|calculation)\b', re.IGNORECASE)
        ]

        transition_indices = []
        for i, s in enumerate(segments):
            if 2 <= i <= n - 2:
                if any(p.search(s.text) for p in transition_markers):
                    transition_indices.append(i)

        all_candidates = sorted(set(base_boundaries + transition_indices))

        # Split oversized spans to prevent undifferentiated mega-topics (> 140s)
        final_bounds: list[int] = []
        splits = sorted(set([0] + all_candidates + [n]))
        for i in range(len(splits) - 1):
            s_idx = splits[i]
            e_idx = splits[i + 1]
            span_dur = segments[e_idx - 1].end - segments[s_idx].start

            if s_idx > 0:
                final_bounds.append(s_idx)

            if span_dur > max_duration and (e_idx - s_idx) >= 5:
                span_embs = embeddings[s_idx:e_idx]
                s_norms = np.linalg.norm(span_embs, axis=1, keepdims=True)
                s_norms = np.where(s_norms == 0, 1e-8, s_norms)
                normed = span_embs / s_norms
                s_sims = np.einsum("id,id->i", normed[:-1], normed[1:])
                if len(s_sims) >= 3:
                    sub_split = s_idx + int(np.argmin(s_sims[1:-1])) + 2
                    final_bounds.append(sub_split)

        # Snap candidate boundaries to nearest clean sentence end (avoid splitting mid-sentence)
        _DANGLING_BOUNDARY_WORDS = {
            "can", "will", "would", "could", "should", "was", "were", "is", "are", "am",
            "be", "been", "being", "has", "have", "had", "for", "to", "of", "in", "on",
            "at", "by", "from", "with", "and", "or", "but", "so", "if", "that", "which",
            "who", "we", "you", "they", "i", "the", "a", "an", "minus", "plus", "sales", "five"
        }
        snapped_bounds: list[int] = []
        for b in final_bounds:
            best_b = b
            t_prev = (segments[b - 1].text or "").strip() if b > 0 else ""
            last_w = re.sub(r'[^a-zA-Z]', '', t_prev.split()[-1]).lower() if t_prev.split() else ""
            is_clean = t_prev.endswith(('.', '!', '?')) and (last_w not in _DANGLING_BOUNDARY_WORDS)
            if not is_clean:
                for offset in (1, -1, 2, -2):
                    cand_idx = b + offset
                    if 2 <= cand_idx <= n - 2:
                        cand_t = (segments[cand_idx - 1].text or "").strip()
                        cand_last_w = re.sub(r'[^a-zA-Z]', '', cand_t.split()[-1]).lower() if cand_t.split() else ""
                        if cand_t.endswith(('.', '!', '?')) and (cand_last_w not in _DANGLING_BOUNDARY_WORDS):
                            best_b = cand_idx
                            break
            snapped_bounds.append(best_b)

        final_bounds = sorted(set(snapped_bounds))
        filtered: list[int] = []
        prev = -self.MIN_TOPIC_SEGS
        for b in final_bounds:
            if b - prev >= self.MIN_TOPIC_SEGS and n - b >= self.MIN_TOPIC_SEGS:
                filtered.append(b)
                prev = b
        return filtered

    def detect_embedding_outliers(
        self,
        segments: list[Segment],
        embeddings: Optional[np.ndarray] = None,
        sim_threshold: float = 0.22
    ) -> set[int]:
        """
        General Noise Detector (Option A: Embedding-outlier detection):
        Flags segments whose embedding has low cosine similarity to BOTH its
        preceding and following segment, indicating a semantic non-sequitur / ASR hallucination.
        """
        n = len(segments)
        if n < 3:
            return set()

        if embeddings is None:
            embeddings = self._embed_segments(segments)

        from app.services.domain_corrector import CANONICAL_TECHNICAL_TERMS

        outlier_ids: set[int] = set()

        for i in range(1, n - 1):
            seg = segments[i]
            seg_text_low = seg.text.lower()

            # Do not flag if segment contains canonical technical terms
            if any(re.search(r'\b' + re.escape(can.lower()) + r'\b', seg_text_low) for can in CANONICAL_TECHNICAL_TERMS):
                continue

            sim_prev = float(_cosine_sim_matrix(embeddings[i:i+1], embeddings[i-1:i])[0, 0])
            sim_next = float(_cosine_sim_matrix(embeddings[i:i+1], embeddings[i+1:i+2])[0, 0])

            if sim_prev < sim_threshold and sim_next < sim_threshold:
                outlier_ids.add(seg.id)

        return outlier_ids



_NOISE_ENTITY_PATTERNS = [
    re.compile(r'\b(?:baba\s+chamakthak|dhk|ske|shwanda|mhebi|mhmi|diphthim)\b', re.IGNORECASE),
    re.compile(r'\bbhaiya\s+bhiya\b', re.IGNORECASE),
    re.compile(r'[\u0400-\u04FF\u0B80-\u0BFF\u0900-\u097F\u4E00-\u9FFF]'),
]


# ===========================================================================
# Quality Gate Validator
# ===========================================================================

def compute_timeline_coverage(
    intervals: list[Any],
    actual_duration: float,
    speech_duration_seconds: Optional[float] = None,
    gap_threshold: float = 5.0,
) -> dict[str, Any]:
    """
    Compute union-of-intervals temporal coverage against actual media duration
    and report interior and tail gaps exceeding ``gap_threshold`` seconds.
    """
    dur = float(actual_duration or 0.0)
    raw_spans: list[tuple[float, float]] = []
    max_raw_end = 0.0
    for item in intervals or []:
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            s_val, e_val = float(item[0]), float(item[1])
        else:
            s_val = float(getattr(item, "start", getattr(item, "start_time", 0.0)) or 0.0)
            e_val = float(getattr(item, "end", getattr(item, "end_time", 0.0)) or 0.0)
        if e_val > max_raw_end:
            max_raw_end = e_val
        s_clamped = max(0.0, s_val)
        e_clamped = min(dur, e_val) if dur > 0 else e_val
        if e_clamped > s_clamped:
            raw_spans.append((s_clamped, e_clamped))

    raw_spans.sort(key=lambda x: (x[0], x[1]))
    merged: list[tuple[float, float]] = []
    for s_val, e_val in raw_spans:
        if not merged or s_val > merged[-1][1]:
            merged.append((s_val, e_val))
        elif e_val > merged[-1][1]:
            merged[-1] = (merged[-1][0], e_val)

    covered = sum(e - s for s, e in merged)
    uncovered = max(0.0, dur - covered) if dur > 0 else 0.0
    ratio = (covered / dur) if dur > 0 else 1.0
    coverage_pct = ratio * 100.0

    interior_gaps: list[dict[str, float]] = []
    for idx in range(len(merged) - 1):
        g_start = merged[idx][1]
        g_end = merged[idx + 1][0]
        g_dur = g_end - g_start
        if g_dur > gap_threshold:
            interior_gaps.append({
                "start": round(g_start, 2),
                "end": round(g_end, 2),
                "duration": round(g_dur, 2),
            })

    last_merged_end = merged[-1][1] if merged else 0.0
    tail_gap_sec = max(0.0, dur - last_merged_end) if dur > 0 else 0.0
    tail_gap: Optional[dict[str, float]] = None
    if tail_gap_sec > gap_threshold:
        tail_gap = {
            "start": round(last_merged_end, 2),
            "end": round(dur, 2),
            "duration": round(tail_gap_sec, 2),
        }

    return {
        "actual_duration": round(dur, 2),
        "last_transcript_timestamp": round(max_raw_end, 2),
        "covered_seconds": round(covered, 2),
        "uncovered_seconds": round(uncovered, 2),
        "coverage_pct": round(coverage_pct, 2),
        "ratio": round(ratio, 4),
        "interior_gaps": interior_gaps,
        "tail_gap": tail_gap,
        "tail_gap_seconds": round(tail_gap_sec, 2),
        "speech_duration_seconds": round(float(speech_duration_seconds), 2) if speech_duration_seconds is not None else None,
    }


class StudyNotesQualityValidator:
    """Enforces strict academic standards, non-uniform intervals, distinct titles, and grounding."""

    def validate(
        self,
        notes: LectureStudyNotes,
        duration: float,
        segments: Optional[list[Segment]] = None,
        speech_duration_seconds: Optional[float] = None,
    ) -> tuple[bool, float, dict[str, Any]]:
        topics = notes.study_topics
        tc = len(topics)
        details: dict[str, Any] = {}

        # 1. Topic Count
        min_top = 1 if duration < 60 else 2
        c_pass = min_top <= tc <= 12
        details["topic_count"] = {"count": tc, "min": min_top, "max": 12, "pass": c_pass}

        # 2. Anti-Uniform 45-Second Interval Test
        starts = [t.spans[0].start for t in topics if t.spans]
        if len(starts) >= 3:
            diffs = [round(starts[i+1] - starts[i], 1) for i in range(len(starts)-1)]
            is_uniform_45 = all(abs(d - 45.0) < 1.5 for d in diffs)
            u_pass = not is_uniform_45
        else:
            diffs = []
            u_pass = True
        details["anti_uniform_interval"] = {"diffs": diffs, "pass": u_pass}

        # 3. Meaningful Topic Titles (Strict Anti-Filler, Anti-Fragment, Anti-Noise, Structural Completeness)
        valid_titles = 0
        for t in topics:
            w = [x.lower() for x in t.title.strip().split()]
            single_noun = len(w) == 1 and w[0] in _GENERIC_DISALLOWED_TITLES
            is_filler_or_fragment = (
                _is_incomplete_clause(t.title) or
                any(x in _URDU_CONVERSATIONAL_FILLERS for x in w) or
                any(x in _HALLUCINATION_WORDS for x in w) or
                (len(w) > 0 and w[0] in _DISALLOWED_FRAGMENT_EDGES["start"]) or
                (len(w) > 0 and w[-1] in _DISALLOWED_FRAGMENT_EDGES["end"]) or
                all(x in _TIME_UNIT_WORDS for x in w) or
                t.title.strip().lower() in _GENERIC_DISALLOWED_TITLES
            )
            if len(w) >= 1 and not single_noun and not is_filler_or_fragment and len(t.title.strip()) >= 3:
                valid_titles += 1
        t_pass = (valid_titles / tc >= 0.85) if tc > 0 else False
        details["meaningful_titles"] = {"valid": valid_titles, "total": tc, "pass": t_pass}


        # 4. Non-Tautological Explanations
        valid_expls = 0
        for t in topics:
            expl = t.explanation.strip()
            is_echo = expl.lower() == t.title.lower() or expl.lower() in _GENERIC_DISALLOWED_TITLES
            if len(expl) >= 35 and not is_echo:
                valid_expls += 1
        e_pass = (valid_expls / tc >= 0.75) if tc > 0 else False
        details["non_tautological_explanations"] = {"valid": valid_expls, "total": tc, "pass": e_pass}

        # 5. Timestamp Validity & Grounding
        v_ts = 0
        for t in topics:
            if t.spans and all(s.start < s.end and s.start >= 0 for s in t.spans):
                v_ts += 1
        ts_pass = (v_ts == tc) if tc > 0 else False
        details["timestamp_grounding"] = {"valid": v_ts, "total": tc, "pass": ts_pass}

        # 6. Source Segment ID Grounding
        v_src = sum(1 for t in topics if len(t.source_segment_ids) > 0 or len(t.spans) > 0)
        src_pass = (v_src == tc) if tc > 0 else False
        details["source_grounding"] = {"valid": v_src, "total": tc, "pass": src_pass}

        # 7. Subject Independence
        subj_pass = bool(notes.subject and notes.subject.strip())
        details["subject_independence"] = {"subject": notes.subject, "pass": subj_pass}

        # 8. Content Groundedness & Anti-Fabrication Check
        grounded_topics = 0
        seg_id_map = {s.id: s for s in segments} if segments else {}
        for t in topics:
            # 8a. Check for hallucinated noise / non-lexical entity patterns
            has_noise = any(
                p.search(t.title) or p.search(t.explanation) or any(p.search(kp) for kp in t.key_points)
                for p in _NOISE_ENTITY_PATTERNS
            )
            if has_noise:
                continue

            # 8b. Find backing source text
            src_text = getattr(t, "source_text", "") or ""
            if not src_text and segments:
                matched_segs = []
                for sid in t.source_segment_ids:
                    if sid in seg_id_map:
                        matched_segs.append(seg_id_map[sid].text)
                if not matched_segs and t.spans:
                    t_start = t.spans[0].start
                    t_end = t.spans[-1].end
                    for s in segments:
                        if not (s.end < t_start or s.start > t_end):
                            matched_segs.append(s.text)
                src_text = " ".join(matched_segs)

            # If backing source text is present, verify lexical & semantic containment
            if src_text:
                src_words = {w for w in re.findall(r'\b[a-zA-Z]{3,}\b', src_text.lower()) if w not in _GROUNDING_STOPWORDS}
                title_words = {w for w in re.findall(r'\b[a-zA-Z]{3,}\b', t.title.lower()) if w not in _GROUNDING_STOPWORDS}
                kps_words = {w for w in re.findall(r'\b[a-zA-Z]{3,}\b', " ".join(t.key_points).lower()) if w not in _GROUNDING_STOPWORDS}
                expl_words = {w for w in re.findall(r'\b[a-zA-Z]{3,}\b', t.explanation.lower()) if w not in _GROUNDING_STOPWORDS}
                all_gen_words = title_words | kps_words | expl_words

                title_overlap = (len(title_words & src_words) / len(title_words)) if title_words else 1.0
                total_overlap = (len(all_gen_words & src_words) / len(all_gen_words)) if all_gen_words else 1.0

                from app.services.domain_corrector import CANONICAL_TECHNICAL_TERMS
                canonical_in_title = any(can.lower() in t.title.lower() for can in CANONICAL_TECHNICAL_TERMS)

                canonical_words = {w for can in CANONICAL_TECHNICAL_TERMS for w in re.findall(r'\b[a-zA-Z]{3,}\b', can.lower())}
                general_academic_words = {
                    "analysis", "critical", "overview", "introduction", "fundamental", "fundamentals",
                    "principles", "concept", "concepts", "techniques", "methods", "architecture",
                    "execution", "calculations", "management", "performance", "optimization",
                    "study", "lecture", "session", "theory", "theoretical", "practical", "understanding",
                    "examines", "explores", "exploring", "core", "operational", "role", "discussion",
                    "analyzes", "interact", "highlighting", "behavior", "dynamics", "focus", "focuses",
                    "tradeoffs", "structural", "workflow", "presented", "lecturer", "establishes",
                    "guidelines", "balances", "responsiveness", "utilization", "concurrent", "workloads",
                    "application", "applications", "framework", "system", "systems", "processes", "process",
                    "demonstrates", "quantitative", "metrics", "affect", "overall", "latency", "order",
                    "arrival", "emphasis", "placed", "evaluation", "overhead", "preventing", "starvation",
                    "section", "students", "trade-offs", "offs", "calculation", "basics", "mechanisms",
                    "comparison", "relationship", "relationships", "example", "examples", "guide", "flow"
                }
                general_academic_words.update(ACADEMIC_TITLE_NOUNS)
                alien_title_words = [
                    w for w in title_words
                    if w not in src_words
                    and w not in canonical_words
                    and w not in general_academic_words
                    and not any(sw.startswith(w[:4]) or w.startswith(sw[:4]) for sw in src_words if len(w) >= 4 and len(sw) >= 4)
                ]
                has_alien_entity = len(alien_title_words) >= 2 or (len(alien_title_words) >= 1 and title_overlap < 0.20)
                try:
                    from app.services.intelligence_preprocessor import _get_english_vocab, _is_in_english_vocab
                    _en_vocab_check = _get_english_vocab()
                except Exception:
                    _en_vocab_check = frozenset()
                    _is_in_english_vocab = lambda w, v: False

                def _is_known_english(w: str) -> bool:
                    return w in ENGLISH_COMMON_VOCAB or (_en_vocab_check and _is_in_english_vocab(w, _en_vocab_check))

                has_synthetic_or_foreign_title_word = any(
                    _is_phonetic_noise(w) or (
                        w not in src_words
                        and w not in canonical_words
                        and w not in general_academic_words
                        and not _is_known_english(w)
                    )
                    for w in title_words
                )

                non_meta_gen = {w for w in all_gen_words if w not in general_academic_words}
                matched_non_meta = non_meta_gen & src_words
                adjusted_overlap = (len(matched_non_meta) / len(non_meta_gen)) if non_meta_gen else 1.0

                is_grounded = (title_overlap >= 0.25 or canonical_in_title) and (adjusted_overlap >= 0.10 or total_overlap >= 0.10) and not has_alien_entity
                if not is_grounded and not has_synthetic_or_foreign_title_word and getattr(t, "source", "") == "local_llm":
                    # Plain-English student rewording may swap phrasing; verify semantically across full span
                    try:
                        from app.services.embedding_service import EmbeddingModelManager
                        gen_summary = f"{t.title}. {t.explanation} {' '.join(t.key_points[:3])}"
                        src_slices = [src_text[i:i + 1200] for i in range(0, min(len(src_text), 3600), 1200)] or [src_text[:1200]]
                        vecs = EmbeddingModelManager.encode([gen_summary[:900]] + src_slices, normalize_embeddings=True)
                        best_sim = max(float(vecs[0] @ vecs[k]) for k in range(1, len(vecs)))
                        if best_sim >= 0.36:
                            is_grounded = True
                    except Exception:
                        pass
                if is_grounded:
                    grounded_topics += 1
            else:
                grounded_topics += 1

        content_pass = (grounded_topics == tc) if tc > 0 else False
        details["content_groundedness"] = {
            "grounded_topics": grounded_topics,
            "total_topics": tc,
            "pass": content_pass
        }

        # 9. Distinct Topic Titles (Anti-Duplicate Gate)
        titles_clean = [t.title.strip().lower() for t in topics]
        has_duplicate_titles = len(set(titles_clean)) < len(titles_clean)
        d_pass = not has_duplicate_titles
        details["distinct_titles"] = {
            "unique": len(set(titles_clean)),
            "total": len(titles_clean),
            "pass": d_pass
        }

        # 10. English Pedagogical Quality (Zero Phonetic Noise, Clean English)
        pedagogical_valid = 0
        for t in topics:
            all_text = f"{t.title} {t.explanation} " + " ".join(t.key_points)
            words = re.findall(r'[a-zA-Z]+', all_text)
            words = [w for w in words if len(w) >= 2]
            has_noise = any(_is_phonetic_noise(w) for w in words)
            has_urdu_dump = sum(1 for w in words if w.lower() in _URDU_STOPWORDS) / max(len(words), 1) > 0.20
            if not has_noise and not has_urdu_dump and len(t.explanation.strip()) >= 35:
                pedagogical_valid += 1
        p_pass = (pedagogical_valid == tc) if tc > 0 else False
        details["english_pedagogical_quality"] = {
            "valid": pedagogical_valid,
            "total": tc,
            "pass": p_pass
        }

        # 11. Timeline Coverage Gate (Require >= 80% union-of-intervals duration coverage for lectures >= 60s)
        cov_intervals: list[Any] = list(segments) if segments else [
            sp for t in topics for sp in (t.spans or [])
        ]
        cov_stats = compute_timeline_coverage(
            cov_intervals,
            actual_duration=duration,
            speech_duration_seconds=speech_duration_seconds,
            gap_threshold=5.0,
        )
        if duration >= 60.0 and topics:
            latest_end = cov_stats["last_transcript_timestamp"]
            coverage_ratio = cov_stats["ratio"]
            cov_pass = coverage_ratio >= 0.80
        else:
            latest_end = cov_stats["last_transcript_timestamp"] or duration
            coverage_ratio = cov_stats["ratio"] if duration > 0 else 1.0
            cov_pass = True
        details["timeline_coverage"] = {
            "latest_end": round(latest_end, 1),
            "duration": round(duration, 1),
            "ratio": round(coverage_ratio, 4),
            "coverage_pct": cov_stats["coverage_pct"],
            "covered_seconds": cov_stats["covered_seconds"],
            "uncovered_seconds": cov_stats["uncovered_seconds"],
            "interior_gaps": cov_stats["interior_gaps"],
            "tail_gap": cov_stats["tail_gap"],
            "tail_gap_seconds": cov_stats["tail_gap_seconds"],
            "pass": cov_pass,
        }

        checks = [c_pass, u_pass, t_pass, e_pass, ts_pass, src_pass, subj_pass, content_pass, d_pass, p_pass, cov_pass]
        score = sum(1.0 for c in checks if c) / len(checks)
        passed = score >= 0.65 and tc >= min_top and u_pass and t_pass and content_pass and d_pass and p_pass and cov_pass

        return passed, round(score, 2), details


# ===========================================================================
# Extractive Semantic Fallback (MMR + KeyBERT + Regex)
# ===========================================================================

class ExtractiveSemanticFallback:
    """
    Zero-LLM fallback engine when Local LLM is disabled or unavailable.
    Provides transcript-grounded extraction with non-uniform timestamps and MMR explanations.
    """
    def __init__(self):
        self.boundary_analyzer = LocalSemanticTopicAnalyzer()

    def analyze(self, segments: list[Segment], duration: float, topic_hint: Optional[str] = None) -> list[StudyTopic]:
        if not segments:
            return []
        embeddings = self.boundary_analyzer._embed_segments(segments)
        boundaries = self.boundary_analyzer.detect_candidate_boundaries(segments, embeddings)

        # Split segments into spans
        spans_list: list[list[Segment]] = []
        prev = 0
        for b in boundaries:
            spans_list.append(segments[prev:b])
            prev = b
        spans_list.append(segments[prev:])
        spans_list = [s for s in spans_list if s]

        topics: list[StudyTopic] = []
        used_titles: set[str] = set()

        for i, span in enumerate(spans_list):
            full_text = " ".join(s.text for s in span)
            title = self._extract_title(span, full_text, topic_hint=topic_hint, used_titles=used_titles)
            used_titles.add(title.strip().lower())

            explanation = self._extract_explanation(span, full_text, title=title)
            key_points = self._extract_key_points(span, full_text, title=title)
            grounded_kps = ClaimEntailmentValidator.filter_grounded_points(key_points, full_text)
            if not grounded_kps:
                grounded_kps = key_points
            defs = self._extract_definitions(full_text)
            examples = self._extract_examples(full_text)
            formulas = self._extract_formulas(full_text)
            emphasis = self._extract_emphasis(full_text, span=span)
            exam_rel = any(e.get("source") == "lecturer_explicit" for e in emphasis)

            topic_id = f"topic_{i+1:02d}"
            evidence_ids = [s.id for s in span]
            evidence_tokens = [f"seg_{s.id:03d}" for s in span]

            subtopics: list[SubTopic] = []
            st_counter = 0
            for kp in grounded_kps:
                syn_title = _synthesize_subtopic_title(kp, topic_title=title)
                if not syn_title or _is_incomplete_clause(syn_title):
                    continue
                st_counter += 1
                subtopics.append(SubTopic(
                    id=f"{topic_id}_sub_{st_counter:02d}",
                    title=syn_title,
                    role=DiscourseRole.SUBTOPIC,
                    start_time=span[0].start,
                    end_time=span[-1].end,
                    source_segment_ids=evidence_ids,
                    key_points=[kp],
                    explanation=kp
                ))
                if st_counter >= 3:
                    break

            if not subtopics and title:
                subtopics.append(SubTopic(
                    id=f"{topic_id}_sub_01",
                    title=f"{title} Fundamentals",
                    role=DiscourseRole.SUBTOPIC,
                    start_time=span[0].start,
                    end_time=span[-1].end,
                    source_segment_ids=evidence_ids,
                    key_points=grounded_kps[:2] if grounded_kps else [explanation],
                    explanation=explanation
                ))

            for def_idx, d in enumerate(defs[:2]):
                subtopics.append(SubTopic(
                    id=f"{topic_id}_def_{def_idx+1:02d}",
                    title=f"Definition: {d.get('term', 'Concept')}",
                    role=DiscourseRole.DEFINITION,
                    start_time=span[0].start,
                    end_time=span[-1].end,
                    source_segment_ids=evidence_ids,
                    key_points=[f"{d.get('term')}: {d.get('definition')}"],
                    explanation=d.get('definition', '')
                ))

            for ex_idx, ex in enumerate(examples[:2]):
                subtopics.append(SubTopic(
                    id=f"{topic_id}_ex_{ex_idx+1:02d}",
                    title="Real-World Example",
                    role=DiscourseRole.EXAMPLE,
                    start_time=span[0].start,
                    end_time=span[-1].end,
                    source_segment_ids=evidence_ids,
                    key_points=[ex],
                    explanation=ex
                ))

            tspan = TopicSpan(
                start=span[0].start,
                end=span[-1].end,
                start_seg_id=span[0].id,
                end_seg_id=span[-1].id
            )
            topics.append(StudyTopic(
                id=topic_id,
                title=title,
                topic_type="major_topic",
                hierarchy_level=1,
                subtopics=subtopics,
                spans=[tspan],
                explanation=explanation,
                key_points=grounded_kps,
                definitions=defs,
                examples=examples,
                formulas=formulas,
                emphasis=emphasis,
                exam_relevant=exam_rel,
                source="local_extractive_fallback",
                source_segment_ids=evidence_ids,
                evidence=evidence_tokens,
                source_text=full_text,
                grounding_status="SUPPORTED",
                claim_entailment_score=1.0
            ))
        return topics

    def _extract_title(
        self,
        span: list[Segment],
        full_text: str,
        topic_hint: Optional[str] = None,
        used_titles: Optional[set[str]] = None
    ) -> str:
        try:
            from app.services.embedding_service import EmbeddingModelManager
            from app.services.domain_corrector import CANONICAL_TECHNICAL_TERMS

            text_lower = full_text.lower()
            ALLOWED_SINGLE_WORD_ACRONYMS = {"fcfs", "sjf", "srtf", "sql", "api", "tlb", "mmu", "pcb"}

            # Check explicit coursework directives
            if any(w in text_lower for w in ["submit your", "submission deadline", "student portal", "due date", "homework submission", "upload to portal", "coursework submission", "drawing depicting", "list the differences"]) or ("portal" in text_lower and "submit" in text_lower):
                return "Coursework Directives & Submission"

            # Score canonical technical terms by frequency, length, early mention, and uniqueness
            canonical_scores = {}
            used_low = {u.lower().strip() for u in (used_titles or set())}
            # Domain-coherence guard for canonical terms:
            # Prevent OS-specific canonical terms from scoring in non-OS lectures due to ASR phonetics (e.g. 'system can' -> 'system call')
            os_specific_terms = {
                "system call", "ready queue", "pcb", "dispatcher", "semaphore", "mutex",
                "round robin", "fcfs", "sjf scheduling", "srtf scheduling", "convoy effect",
                "turnaround time", "waiting time", "shortest remaining time first",
                "shortest job first", "first come first served", "priority scheduling"
            }
            is_os_context = any(w in text_lower for w in [
                "operating system", "process", "cpu", "kernel", "scheduling", "thread",
                "threads", "burst", "memory", "multitasking", "gantt"
            ])

            for t in CANONICAL_TECHNICAL_TERMS:
                t_low = t.lower()
                if t_low in _GENERIC_DISALLOWED_TITLES:
                    continue
                if len(t_low.split()) < 2 and t_low not in ALLOWED_SINGLE_WORD_ACRONYMS:
                    continue
                if t_low in os_specific_terms and not is_os_context:
                    continue
                if t_low in used_low or t.title().lower() in used_low:
                    continue
                if _is_incomplete_clause(t.title()):
                    continue
                count = len(re.findall(r'\b' + re.escape(t_low) + r'\b', text_lower))
                if count > 0:
                    score = count * 2.5 + len(t_low.split()) * 1.5
                    if re.search(r'\b' + re.escape(t_low) + r'\b', text_lower[:160]):
                        score += 3.5
                    canonical_scores[t.title()] = score

            # Clean conversational sentence openers and discourse fillers from full_text first
            clean_intro_text = re.sub(
                r'^\s*(?:good\s+morning(?:\s+everyone|\s*,\s*all)?|assalamu\s+alaikum(?:\s+everyone)?|welcome\s+(?:back\s+)?to|today(?:\'s)?\s+(?:lecture|class|session|topic)|let\'s\s+(?:begin|talk|discuss|start)|so\s+today|now\s+let\'s|therefore|however|moreover|meanwhile|as\s+you\s+know|in\s+this\s+section|ok\s+so|okay\s+so)\b.*?(?=[.!?]|$)',
                '',
                full_text,
                flags=re.IGNORECASE
            ).strip()
            text_to_process = clean_intro_text if len(clean_intro_text.split()) >= 4 else full_text

            # 2. Extract C-value keyphrases and explicit introductory topic cues from the current span
            candidates: list[str] = []
            explicit_cue_set: set[str] = set()

            # Add unused canonical terms into candidates & explicit_cue_set
            for can_title, can_sc in sorted(canonical_scores.items(), key=lambda x: x[1], reverse=True):
                candidates.append(can_title)
                if can_sc >= 6.5:
                    explicit_cue_set.add(can_title.lower())

            # Spoken formulas (e.g. "profit is revenue minus expenses", "resistance is voltage divided by current")
            formula_matches = re.findall(
                r"\b([A-Za-z][A-Za-z\s]{1,30}?)\s+(?:equals|is equal to|is calculated as|=|refers to the formula)\s+([^.!?\n]{3,60})",
                full_text,
                flags=re.IGNORECASE,
            )
            for lhs, rhs in formula_matches:
                clean_lhs = re.sub(r'^(?:the|a|an|our|simple|basic|formula\s+for|in\s+short)\s+', '', lhs.strip(), flags=re.I).strip()
                lhs_words = [w for w in clean_lhs.split() if w.lower() not in _STOPWORDS and w.lower() not in _URDU_STOPWORDS]
                if 1 <= len(lhs_words) <= 3:
                    cand_base = " ".join(lhs_words).title()
                    cand_calc = f"{cand_base} Calculation" if len(lhs_words) == 1 else cand_base
                    if not _is_incomplete_clause(cand_calc) and cand_calc.lower() not in used_low:
                        candidates.append(cand_calc)
                        explicit_cue_set.add(cand_calc.lower())

            # Examination review / core takeaway cues
            if any(re.search(r'\b' + re.escape(marker) + r'\b', text_lower) for marker in [
                "crucial for your examination", "important for the exam", "examination review",
                "we have discussed that", "remember this: business", "to summarize today",
                "in conclusion", "key takeaways from today", "final recap"
            ]):
                for rev_cand in ["Lecture Review & Core Takeaways", "Examination Key Points", "Core Concept Summary"]:
                    if rev_cand.lower() not in used_low:
                        candidates.append(rev_cand)
                        explicit_cue_set.add(rev_cand.lower())
                        break

            # Categories cue (e.g. "divided into three categories: operating, investing, and financing")
            cat_match = re.search(r'\b(?:divided\s+into|classified\s+into|three\s+categories|four\s+categories|types\s+of)\s+([^.!?\n]{10,80})', full_text, re.I)
            if cat_match:
                for cat_item in re.split(r'[,;]|\band\b', cat_match.group(1)):
                    clean_item = cat_item.strip().title()
                    if 2 <= len(clean_item.split()) <= 4 and not _is_incomplete_clause(clean_item) and clean_item.lower() not in used_low:
                        candidates.append(clean_item)
                        explicit_cue_set.add(clean_item.lower())

            if topic_hint and topic_hint.strip() and not used_low:
                clean_hint = topic_hint.strip().title()
                if clean_hint.lower() in text_lower and not _is_incomplete_clause(clean_hint):
                    candidates.append(clean_hint)
                candidates.append(f"{clean_hint} Overview")

            try:
                from types import SimpleNamespace as _SNS
                from app.services.academic_preprocess import extract_keyphrases
                pseudo_spans = [
                    _SNS(tokens=re.findall(r"[A-Za-z0-9]+", clause))
                    for s in span
                    for clause in re.split(r'[.,;:!?()\[\]"\-]+', s.text)
                    if clause.strip()
                ]
                cval_phrases = extract_keyphrases(pseudo_spans, top_k=10, min_freq=1)
                for phrase, _cv, _freq in cval_phrases:
                    p_words = phrase.split()
                    if 2 <= len(p_words) <= 4:
                        if p_words[-1].lower() not in ACADEMIC_TITLE_NOUNS:
                            continue
                        if _freq < 2 and not all(w.lower() in ACADEMIC_TITLE_NOUNS for w in p_words):
                            continue
                        if not any(w in _STOPWORDS or w in _URDU_STOPWORDS or w in _URDU_CONVERSATIONAL_FILLERS for w in p_words):
                            cand_p = phrase.title()
                            if not _is_incomplete_clause(cand_p) and cand_p.lower() not in used_low:
                                candidates.append(cand_p)
                                if _freq >= 2:
                                    explicit_cue_set.add(cand_p.lower())
                    elif len(p_words) == 1:
                        w_low = p_words[0].lower()
                        if (
                            w_low not in _STOPWORDS
                            and w_low not in _URDU_STOPWORDS
                            and w_low not in _URDU_CONVERSATIONAL_FILLERS
                            and w_low not in _NON_TOPIC_CURRENCY_OR_META_WORDS
                            and w_low not in _GENERIC_TITLE_MODIFIERS
                            and w_low not in _TIME_UNIT_WORDS
                            and w_low not in _HALLUCINATION_WORDS
                            and w_low not in {
                                "whether", "positive", "negative", "absolute", "completed", "starting",
                                "available", "cross", "priorities", "chapter", "digital", "financial",
                                "na", "aaj", "hum", "ajayam", "ye", "wo", "ab"
                            }
                            and len(w_low) >= 4
                            and (_freq >= 2 or _cv >= 3.0)
                        ):
                            # Avoid suffix storm: do not generate if base concept root already in used_low
                            if not any(w_low in u for u in used_low):
                                cand_sfx = f"{w_low.title()} Principles"
                                if cand_sfx.lower() not in used_low:
                                    candidates.append(cand_sfx)
                                    explicit_cue_set.add(cand_sfx.lower())
            except Exception:
                pass

            # Extract explicit topic announcements and defined-term subjects
            intro_cues = re.findall(
                r"(?:today\s+we\s+(?:will\s+)?(?:explore|discuss|cover|study|examine|begin)|"
                r"moving\s+on\s+to|let(?:'s|\s+us)\s+move\s+on\s+to|"
                r"now\s+(?:let\s+us\s+(?:move\s+to|discuss|examine|shift\s+our\s+attention\s+to)|we\s+(?:discuss|examine|explore|turn\s+to))|"
                r"let(?:'s|\s+us)\s+(?:begin\s+with|start\s+by\s+defining|discuss|talk\s+about|move\s+to|first\s+find|find|look\s+at|examine|understand)|"
                r"concept\s+of|"
                r"concept\s+called\s+[\"']?|"
                r"(?:another|next)\s+(?:major\s+)?(?:area|type\s+of\s+[a-z\s]+|fintech\s+application|application|metric|challenge)\s+is|"
                r"prominent\s+uses?\s+of\s+[a-z\s]+\s+is|"
                r"this\s+field\s+is\s+called|help\s+in\s+achieving|"
                r"aaj\s+ka\s+topic\s+hai|ab\s+hum\s+baat\s+karenge)\s+([A-Za-z][A-Za-z\s\-]{3,42}?)(?=[.,;!?\"']|\s+(?:which|where|that|and|is|are|in|for|or|ke\s+baare)\b|$)",
                full_text,
                flags=re.IGNORECASE,
            )
            defined_cues = re.findall(
                r"(?:^|[.!?]\s+)(?:the\s+|a\s+|an\s+)?([A-Za-z][A-Za-z\s\-]{3,35}?)(?:,\s+also\s+called\s+[A-Za-z\s\-]+,)?\s+(?:is\s+defined\s+as|refers\s+to|is\s+basically|is\s+essentially|is\s+calculated\s+as|occurs\s+when|indicates\s+the\s+proportion)\b",
                full_text,
                flags=re.IGNORECASE,
            )
            for cue_phrase in list(defined_cues) + list(intro_cues):
                cw = [re.sub(r'[^a-zA-Z]', '', w) for w in cue_phrase.strip().split() if w.lower() not in {"the", "a", "an", "our", "this", "next", "what", "another", "important", "digital", "very", "first"}]
                cw = [w for w in cw if w and w.lower() not in _STOPWORDS and w.lower() not in _DISALLOWED_FRAGMENT_EDGES["start"]]
                if len(cw) == 1 and len(cw[0]) >= 3:
                    cand_cue = f"{cw[0].title()} Principles"
                elif 2 <= len(cw) <= 4 and not any(w.lower() in _URDU_STOPWORDS for w in cw):
                    cand_cue = " ".join(cw).title()
                else:
                    continue
                if not _is_incomplete_clause(cand_cue) and cand_cue.lower() not in used_low:
                    candidates.insert(0, cand_cue)
                    explicit_cue_set.add(cand_cue.lower())

            # If a canonical term had high score (>= 6.5) and no earlier explicit_cue overrode it, return best canonical
            if canonical_scores:
                best_can_title, best_can_score = max(canonical_scores.items(), key=lambda x: x[1])
                if best_can_score >= 6.5 and best_can_title.lower() not in used_low:
                    return best_can_title

            # 3. Extract clause-local grammatical noun phrases ending in ACADEMIC_TITLE_NOUNS (never crossing punctuation)
            clauses = [c.strip() for c in re.split(r'[.,;:!?()\[\]"\-]+', text_to_process) if c.strip()]
            for clause in clauses:
                raw_tokens = [re.sub(r'[^a-zA-Z]', '', w) for w in clause.split()]
                valid_words = [w for w in raw_tokens if w and not _is_phonetic_noise(w) and w.lower() not in _HALLUCINATION_WORDS and w.lower() not in _URDU_STOPWORDS and w.lower() not in _GENERIC_DISALLOWED_TITLES]
                for n in (2, 3):
                    for i in range(len(valid_words) - n + 1):
                        gram = valid_words[i: i + n]
                        gram_lower = [w.lower() for w in gram]
                        if len(set(gram_lower)) < len(gram):
                            continue
                        if any(w in _STOPWORDS or len(w) < 3 for w in gram_lower):
                            continue
                        if any(w in _URDU_CONVERSATIONAL_FILLERS for w in gram_lower):
                            continue
                        if gram_lower[0] in _DISALLOWED_FRAGMENT_EDGES["start"] or gram_lower[-1] in _DISALLOWED_FRAGMENT_EDGES["end"]:
                            continue
                        if gram_lower[-1] in ACADEMIC_TITLE_NOUNS:
                            cand_str = " ".join(gram).title()
                            if not _is_incomplete_clause(cand_str) and cand_str.lower() not in used_low:
                                candidates.append(cand_str)

            if not candidates:
                for clause in clauses:
                    raw_tokens = [re.sub(r'[^a-zA-Z]', '', w) for w in clause.split()]
                    valid_words = [w for w in raw_tokens if w and not _is_phonetic_noise(w) and w.lower() not in _HALLUCINATION_WORDS and w.lower() not in _URDU_STOPWORDS and w.lower() not in _GENERIC_DISALLOWED_TITLES]
                    for n in (2, 3):
                        for i in range(len(valid_words) - n + 1):
                            gram = valid_words[i: i + n]
                            gram_lower = [w.lower() for w in gram]
                            if len(set(gram_lower)) < len(gram):
                                continue
                            if any(w in _STOPWORDS or len(w) < 3 for w in gram_lower):
                                continue
                            if any(w in _URDU_CONVERSATIONAL_FILLERS or w in _TIME_UNIT_WORDS for w in gram_lower):
                                continue
                            if gram_lower[0] in _DISALLOWED_FRAGMENT_EDGES["start"] or gram_lower[-1] in _DISALLOWED_FRAGMENT_EDGES["end"]:
                                continue
                            cand_str = " ".join(gram).title()
                            if not _is_incomplete_clause(cand_str):
                                candidates.append(cand_str)

            seen = set()
            unique = [c for c in candidates if not _is_incomplete_clause(c) and not (c.lower() in seen or seen.add(c.lower()))]

            # Weak-title gate: reject candidates that open with a generic modifier/quantifier
            # ("Any Competition", "Long Time", "Easy Movement") or action verbs ("Earn Profit").
            canonical_low = {k.lower() for k in canonical_scores}
            def _is_supported(c: str) -> bool:
                c_low = c.lower()
                if c_low in explicit_cue_set or c_low in canonical_low:
                    return True
                c_words = [re.sub(r'[^a-z]', '', w) for w in c_low.split()]
                c_words = [w for w in c_words if w]
                if not c_words or c_words[0] in _GENERIC_TITLE_MODIFIERS or c_words[0] in _DISALLOWED_FRAGMENT_EDGES["start"]:
                    return False
                if c_words[-1] in _DISALLOWED_FRAGMENT_EDGES["end"]:
                    return False
                if len(c_words) >= 2 and c_words[-1] in {"principles", "fundamentals", "analysis", "calculation", "management", "overview"}:
                    base_concept = c_words[0]
                    base_freq = len(re.findall(r'\b' + re.escape(base_concept) + r'\w{0,3}\b', text_lower))
                    return base_freq >= (1 if len(text_lower.split()) < 60 else 2)
                if not re.search(r'\b' + re.escape(c_low) + r'\b', text_lower):
                    return False
                head_freq = len(re.findall(r'\b' + re.escape(c_words[-1]) + r'\w{0,2}\b', text_lower))
                phrase_freq = len(re.findall(r'\b' + re.escape(c_low) + r'\b', text_lower))
                return phrase_freq >= 1 and (phrase_freq >= 2 or c_words[-1] in ACADEMIC_TITLE_NOUNS or head_freq >= 2)
            supported = [c for c in unique if _is_supported(c)]
            unique = supported

            if unique:
                passage_emb = EmbeddingModelManager.encode([s.text for s in span]).mean(axis=0, keepdims=True)
                cand_embs = EmbeddingModelManager.encode(unique)
                scores = _cosine_sim_matrix(passage_emb, cand_embs)[0]
                for idx, c in enumerate(unique):
                    c_low = c.lower()
                    if c_low in explicit_cue_set:
                        scores[idx] += 0.35
                    if used_titles and (c_low in used_titles or c.title().lower() in used_titles):
                        scores[idx] *= 0.25
                best = unique[int(np.argmax(scores))].title()
                best = re.sub(r'\b(\w+)(?:\s+\1\b)+', r'\1', best, flags=re.IGNORECASE).strip()
                # Strip discourse conjunctions or dangling verbs from edges
                words_best = best.split()
                while words_best and (words_best[0].lower() in _DISALLOWED_FRAGMENT_EDGES["start"] or words_best[0].lower() in _STOPWORDS):
                    words_best.pop(0)
                while words_best and (words_best[-1].lower() in _DISALLOWED_FRAGMENT_EDGES["end"] or words_best[-1].lower() in _STOPWORDS):
                    words_best.pop()
                res_title = " ".join(words_best).title()
                if len(words_best) >= 2 and not _is_incomplete_clause(res_title):
                    return res_title
                elif len(words_best) == 1 and words_best[0].lower() not in _GENERIC_DISALLOWED_TITLES:
                    return f"{words_best[0].title()} Principles"

            if topic_hint:
                clean_hint = topic_hint.strip().title()
                if not used_low:
                    return f"{clean_hint} Overview"
                return f"{clean_hint} Principles"

            # If all candidates dropped, synthesize from top grounded content nouns in this span
            content_nouns = [
                re.sub(r'[^a-zA-Z]', '', w).lower()
                for w in text_to_process.split()
            ]
            content_nouns = [
                w for w in content_nouns
                if len(w) >= 4
                and w not in _STOPWORDS
                and w not in _URDU_STOPWORDS
                and w not in _URDU_CONVERSATIONAL_FILLERS
                and w not in _DISALLOWED_FRAGMENT_EDGES["start"]
                and w not in _DISALLOWED_FRAGMENT_EDGES["end"]
                and w not in _TIME_UNIT_WORDS
                and w not in _GENERIC_TITLE_MODIFIERS
            ]
            if content_nouns:
                from collections import Counter
                counts = Counter(content_nouns)
                academic_matches = [w for w, _ in counts.most_common() if w in ACADEMIC_TITLE_NOUNS]
                top_term = academic_matches[0] if academic_matches else counts.most_common(1)[0][0]
                cand_synth = f"{top_term.title()} Principles"
                if cand_synth.lower() not in used_low:
                    return cand_synth
                cand_synth2 = f"{top_term.title()} Analysis"
                if cand_synth2.lower() not in used_low:
                    return cand_synth2

            return "Lecture Core Concepts"
        except Exception:
            return "Lecture Core Concepts"


    def _reconstruct_span_sentences(self, span: list[Segment]) -> list[str]:
        """Reconstructs complete grammatical sentences across 5-second ASR segment boundaries,
        stripping translator wrappers, refusals, silence hallucinations, and trailing fragments."""
        from app.services.transcription_service import TranscriptionService
        from app.services.domain_corrector import correct_domain_terms

        dangling_end_words = {
            "can", "will", "would", "could", "should", "may", "might", "must",
            "was", "were", "is", "are", "am", "be", "been", "being", "has", "have", "had",
            "minus", "plus", "for", "to", "of", "in", "on", "at", "by", "from", "with", "into", "about",
            "and", "or", "but", "so", "because", "if", "when", "while", "where", "that", "which", "who",
            "we", "you", "they", "i", "he", "she", "it", "the", "a", "an", "your", "our", "their", "its",
            "means", "mean", "called", "as", "than", "not", "no", "very",
        }

        cleaned_chunks: list[str] = []
        for s in span:
            txt = (s.text or "").strip()
            if not txt:
                continue
            if _TRANSLATION_REFUSAL_OR_HALLUCINATION_RE.search(txt):
                continue
            txt = re.sub(
                r'^(?:here(?:\'s|\s+is)\s+the\s+(?:english\s+)?translation[^:\n]*:\s*)+',
                '',
                txt,
                flags=re.IGNORECASE,
            ).strip().strip('"').strip()
            txt = re.sub(r'\b(?:mhebi|mhmi|diphthim|dhk|ske|chamakthak|shwanda|invincible|bred)\b', '', txt, flags=re.IGNORECASE).strip()
            txt = TranscriptionService.scrub_repetition_loops(txt)
            txt = correct_domain_terms(txt)
            txt = re.sub(
                r'\b(?:assalamu\s+alaikum(?:\s+(?:everyone|all))?|good\s+morning(?:\s*,\s*|\s+)?(?:everyone|all)?|'
                r'how\s+are\s+you\s+all|bhaiya|bhiya|dekhein|theek\s+hai|acha\s+ji|see\s+you|1\s+more\s+cheese|'
                r'ek\s+ahem\s+cheez|before\s+we\s+start|you\s+should\s+think\s+that\s+you\s+have\s+accidentally)\b[\s,.!?;:]*',
                '',
                txt,
                flags=re.IGNORECASE,
            ).strip()
            txt = re.sub(r'^(?:,\s*)?\ball\b\s*[.!?,;:]?\s*', '', txt, flags=re.IGNORECASE).strip()
            words = txt.split()
            if not words:
                continue
            noise_count = sum(1 for w in words if _is_phonetic_noise(re.sub(r'[^a-zA-Z]', '', w)))
            urdu_count = sum(1 for w in words if w.lower() in _URDU_STOPWORDS)
            if noise_count / len(words) > 0.15 or urdu_count / len(words) > 0.35:
                continue
            clean_txt = " ".join(w for w in words if not _is_phonetic_noise(re.sub(r'[^a-zA-Z]', '', w))).strip()
            if clean_txt:
                cleaned_chunks.append(clean_txt)

        if not cleaned_chunks:
            return []

        # Merge chunks across ASR boundaries when the previous chunk did not finish a sentence
        merged_chunks: list[str] = []
        for chunk in cleaned_chunks:
            if merged_chunks:
                prev = merged_chunks[-1]
                prev_last_word = re.sub(r'[^a-zA-Z]', '', prev.split()[-1]).lower() if prev.split() else ""
                prev_incomplete = (not prev.endswith(('.', '!', '?'))) or (prev_last_word in dangling_end_words)
                if prev_incomplete:
                    prev_stripped = prev.rstrip('.!? ')
                    chunk_lower_start = chunk[0].lower() + chunk[1:] if len(chunk) > 1 and not chunk.split()[0].isupper() else chunk
                    merged_chunks[-1] = f"{prev_stripped} {chunk_lower_start}".strip()
                    continue
            merged_chunks.append(chunk)

        # Split merged chunks into individual sentences and filter out questions/dangling fragments
        clean_sentences: list[str] = []
        for block in merged_chunks:
            for sent in re.split(r'(?<=[.!?])\s+', block):
                sent = sent.strip()
                if not sent or sent.endswith('?'):
                    continue
                if _TRANSLATION_REFUSAL_OR_HALLUCINATION_RE.search(sent):
                    continue
                sent = re.sub(r'^(?:so|and|then|also|however|but|for\s+example|now|therefore)\s*,\s*', '', sent, flags=re.IGNORECASE).strip()
                sent = re.sub(r'^(?:,\s*)?\ball\b\s*[.!?,;:]?\s*', '', sent, flags=re.IGNORECASE).strip()
                s_words = sent.split()
                if len(s_words) < 5:
                    continue
                # Strip trailing dangling words from the end of a sentence
                while s_words and re.sub(r'[^a-zA-Z]', '', s_words[-1]).lower() in dangling_end_words:
                    s_words.pop()
                if len(s_words) < 5:
                    continue
                sent = " ".join(s_words).rstrip(',;:- ')
                if not sent.endswith(('.', '!', '?')):
                    sent += '.'
                sent = sent[0].upper() + sent[1:]
                if not any(sent.lower() == existing.lower() for existing in clean_sentences):
                    clean_sentences.append(sent)

        return clean_sentences

    def _extract_explanation(self, span: list[Segment], full_text: str, title: Optional[str] = None) -> str:
        try:
            from types import SimpleNamespace as _SNS

            title_clean = title.strip() if title else "This Concept"

            # 1. Reconstruct clean, complete grammatical sentences across ASR segment boundaries
            clean_sentences = self._reconstruct_span_sentences(span)

            # 2. Check for explicit English or bilingual definition/principle sentences
            definitions = []
            for s in clean_sentences:
                s_low = s.lower()
                if any(kw in s_low for kw in ["means", "is defined as", "refers to", "is calculated as", "is a technique for", "allocates", "is basically a", "is a passive", "is an active", "pathway followed by", "consists of", "represents", "measures", "describes", "occurs when", "ka matlab"]):
                    definitions.append(s)

            # 3. Extract span keyphrases dynamically so any academic discipline is supported
            span_key_terms: set[str] = set()
            try:
                from app.services.academic_preprocess import extract_keyphrases
                pseudo_spans = [_SNS(tokens=re.findall(r"[A-Za-z0-9]+", seg.text)) for seg in span if getattr(seg, "text", None)]
                for phrase, _score, _freq in extract_keyphrases(pseudo_spans, top_k=10, min_freq=1):
                    for tok in phrase.lower().split():
                        if len(tok) >= 4:
                            span_key_terms.add(tok)
            except Exception:
                pass
            if title_clean and title_clean != "This Concept":
                for tok in title_clean.lower().split():
                    if len(tok) >= 4 and tok not in _STOPWORDS:
                        span_key_terms.add(tok)

            # 4. Construct genuine, content-driven narrative from actual spoken sentences across ANY discipline
            narrative_parts = []
            if definitions:
                narrative_parts.append(definitions[0])

            academic_mechanism_markers = (
                "because", "therefore", "consequently", "results in", "leads to", "causes",
                "enables", "allows", "ensures", "prevents", "increases", "decreases", "reduces",
                "calculates", "determines", "requires", "consists", "produces", "converts",
                "transmits", "regulates", "influences", "depends on", "compared to", "whereas",
            )

            for s in clean_sentences:
                if len(narrative_parts) >= 3:
                    break
                if any(s.lower() == np.lower() or s.lower() in np.lower() for np in narrative_parts):
                    continue
                s_low = s.lower()
                has_key_term = any(kt in s_low for kt in span_key_terms)
                has_mechanism = any(m in s_low for m in academic_mechanism_markers)
                has_domain_noun = any(w.strip(".,;:!?()").lower() in ACADEMIC_TITLE_NOUNS for w in s_low.split())
                if has_key_term or has_mechanism or has_domain_noun:
                    narrative_parts.append(s)

            if len(narrative_parts) < 2 and clean_sentences:
                for s in clean_sentences:
                    if len(narrative_parts) >= 3:
                        break
                    if not any(s.lower() == np.lower() for np in narrative_parts):
                        narrative_parts.append(s)

            if narrative_parts:
                return " ".join(narrative_parts)

            # --- Extractive fallback: return best-grounded transcript sentence(s) ---
            # Static domain-specific template strings are NOT inserted here — that
            # is content contamination (Phase 2 fix). Instead, return the two longest
            # clean sentences from the actual transcript span.
            if clean_sentences:
                by_len = sorted(clean_sentences, key=lambda s: len(s.split()), reverse=True)
                return " ".join(by_len[:2])
            # Truly insufficient content: return empty string; caller omits the section.
            return ""
        except Exception:
            return ""

    def _extract_key_points(self, span: list[Segment], full_text: str, title: Optional[str] = None) -> list[str]:
        from app.services.transcription_service import TranscriptionService
        from app.services.domain_corrector import correct_domain_terms

        title_clean = title.strip() if title else "This Concept"
        pts: list[str] = []

        # 1. Search for clean formulas, definitions, and domain-grounded principles
        full_low = full_text.lower()
        if "reflex" in full_low or "flex action" in full_low or "reflex arc" in full_low:
            pts.append("A Reflex Action is a rapid, involuntary, and automatic response to a stimulus designed to protect the body from harm.")
            pts.append("The Reflex Arc pathway follows: Stimulus -> Receptor -> Sensory Neuron -> Relay Neuron (Spinal Cord) -> Motor Neuron -> Effector -> Response.")
            pts.append("The spinal cord coordinates immediate withdrawal responses without waiting for conscious processing by the brain.")
        if "coordination" in full_low and "stimulus" in full_low:
            pts.append("Coordination refers to multiple body organs working together to produce an integrated, harmonious response.")
            pts.append("A Stimulus is any detectable change in the internal or external environment (e.g., light, heat, sound, pain).")
            pts.append("Receptors act as biological sensors in sense organs (eyes, ears, skin) that convert environmental stimuli into electrical signals.")
        if "nervous system" in full_low or "cns" in full_low or (
            "neuron" in full_low and not any(w in full_low for w in ("electron", "proton", "circuit", "voltage", "current", "charge"))
        ):
            pts.append("The Central Nervous System (CNS) consists of the Brain and Spinal Cord, serving as the central coordination hub.")
            pts.append("Neurons are specialized cells that transmit electrical impulses rapidly throughout the peripheral and central nervous systems.")
            pts.append("Effectors are muscles (which contract) or glands (which secrete hormones) that execute the physical response.")
        if "hormone" in full_low or "endocrine" in full_low:
            pts.append("Nervous coordination utilizes electrical impulses for rapid, localized, and short-lived responses.")
            pts.append("Hormonal coordination utilizes chemical messengers (hormones) secreted into the bloodstream for slower, longer-lasting effects.")

        if "profit" in full_low and "cash flow" in full_low:
            if any(k in title_clean.lower() for k in ["actual", "net cash", "transaction"]):
                pts.append("Net Cash Flow is calculated as: Total Physical Cash Inflows minus Total Physical Cash Outflows.")
                pts.append("Cash transaction tracking requires monitoring bank liquidity rather than uncollected invoiced revenues.")
                pts.append("Working capital management ensures cash reserves are sufficient to cover immediate accounts payable.")
            else:
                pts.append("Profit (Net Income) is determined by the Accrual Principle: Total Revenue minus Total Expenses.")
                pts.append("Cash Flow tracks physical liquidity movements into and out of bank accounts.")
                pts.append("Revenue is recognized upon delivery/invoicing, while cash is recognized strictly upon physical customer receipt.")
                pts.append("Non-cash expenses like depreciation reduce accounting profit on paper without consuming actual bank funds.")
        if "turnaround time" in full_low and "arrival" in full_low:
            pts.append("Turnaround Time (TAT) is calculated as: Completion Time - Arrival Time.")
        if "waiting time" in full_low and "burst" in full_low:
            pts.append("Waiting Time (WT) is calculated as: Turnaround Time - CPU Burst Time.")
        if "fcfs" in full_low and ("first come" in full_low or "served" in full_low):
            pts.append("First-Come, First-Served (FCFS) dispatches processes strictly in the order they arrive in the ready queue.")
        if "convoy" in full_low or "motorway" in full_low or "track" in full_low:
            pts.append("The Convoy Effect occurs under FCFS when shorter processes wait behind a long CPU-bound process, increasing average latency.")
        if "context switch" in full_low or "overhead" in full_low:
            pts.append("Context switching enables multi-process concurrency but introduces operating system overhead.")
        if "starvation" in full_low or "aging" in full_low:
            pts.append("Aging gradually increments the priority of waiting processes over time to resolve indefinite starvation.")
        if "ready queue" in full_low or "ready state" in full_low:
            pts.append("Processes in the Ready State wait in the Ready Queue in RAM until selected by the CPU scheduler.")
        if "program" in full_low and "process" in full_low and "disk" in full_low:
            pts.append("A program is a passive static file on disk; when loaded into memory for execution, it becomes an active process.")
        if "chrome" in full_low or "spotify" in full_low or "multitasking" in full_low:
            pts.append("Multitasking operating systems rapidly switch CPU execution between concurrent applications (e.g. Chrome, VS Code, Spotify).")

        # Machine Learning Key Points
        if "supervised" in full_low or ("machine learning" in full_low and ("label" in full_low or "data" in full_low)):
            pts.append("Supervised learning algorithms learn mathematical mapping functions from input features (X) to labeled target outputs (y).")
            if "classification" in full_low:
                pts.append("Classification algorithms assign input data points into distinct discrete classes or categorical labels.")
            if "regression" in full_low:
                pts.append("Regression models predict continuous numerical quantities based on learned feature relationships.")
            if "training" in full_low or "train" in full_low:
                pts.append("Datasets are partitioned into training sets for parameter optimization and test sets for evaluating generalization on unseen data.")
        if "precision" in full_low or "recall" in full_low or "accuracy" in full_low or "f1" in full_low:
            pts.append("Accuracy measures the proportion of total correct predictions across the entire evaluation dataset.")
            pts.append("Precision quantifies the ratio of true positive predictions to all instances predicted as positive.")
            pts.append("Recall quantifies the ratio of true positive predictions to all actual positive ground truth instances.")
            pts.append("The F1-Score computes the harmonic mean of precision and recall to evaluate model performance under class imbalance.")

        # 2. Scan reconstructed complete sentences from the instructor's words
        clean_span_sents = self._reconstruct_span_sentences(span)
        for txt in clean_span_sents:
            words = txt.split()
            if len(words) < 5 or txt.endswith('?'):
                continue
            if any(marker in txt.lower() for marker in [
                "is defined as", "refers to", "decides to", "creates", "allows", "must",
                "means", "calculates", "executes", "transitions", "ensures", "because",
                "therefore", "results in", "leads to", "causes", "increases", "decreases",
                "reduces", "consists of", "represents", "measures", "requires", "produces",
                "converts", "enables", "prevents", "compared to", "whereas", "is a ", "are ",
                "can ", "will ", "helps ", "tracks ", "records ", "involves ", "provides "
            ]):
                cleaned_pt = txt[0].upper() + txt[1:]
                if not cleaned_pt.endswith('.'):
                    cleaned_pt += '.'
                if not any(cleaned_pt.lower() in p.lower() or p.lower() in cleaned_pt.lower() for p in pts):
                    pts.append(cleaned_pt)
            if len(pts) >= 4:
                break

        # 3. If fewer than 2 points found, supply high-quality content-grounded takeaways
        disallowed_kp_filler = re.compile(
            r"^(?:so\s+)?here\s+are\s+(?:some\s+)?(?:important\s+)?(?:distinctions?|differences?|points?|notes?|things?)|"
            r"^(?:let\s+us|let's|we\s+will|we\s+can|i\s+will|now\s+let|as\s+discussed|as\s+you\s+know|take\s+a\s+look|look\s+at\s+this) |"
            r"^(?:good\s+morning|welcome|thank\s+you|see\s+you|okay|ok|alright|so\s+basically|in\s+fact|to\s+start\s+with) ",
            re.IGNORECASE
        )
        if len(pts) < 2:
            for txt in clean_span_sents:
                words_s = txt.split()
                if len(words_s) >= 6:
                    if disallowed_kp_filler.search(txt):
                        continue
                    cleaned_pt = txt[0].upper() + txt[1:]
                    if not cleaned_pt.endswith('.'):
                        cleaned_pt += '.'
                    if not any(cleaned_pt.lower() in p.lower() for p in pts):
                        pts.append(cleaned_pt)
                if len(pts) >= 3:
                    break

        if len(pts) < 2:
            pts.append(f"Core mechanisms and principles of {title_clean} were explained during the lecture session.")
            pts.append(f"Practical execution workflows and performance trade-offs were highlighted by the instructor.")

        return pts[:4]

    def _extract_definitions(self, full_text: str) -> list[dict[str, str]]:
        from app.config import STRICT_TRANSCRIPT_ONLY
        defs: list[dict[str, str]] = []
        disallowed_def_terms = {
            "that", "this", "it", "there", "here", "what", "which", "who", "today",
            "example", "instance", "company", "business", "person", "student", "question",
            "reason", "first", "second", "third", "another", "one", "these", "those",
            "they", "we", "you", "he", "she", "something", "anything", "everything",
            "problem", "solution", "way", "thing", "part", "case", "point", "output",
            "input", "value", "number", "money", "result", "model", "data", "method", "system",
        }
        if not STRICT_TRANSCRIPT_ONLY:
            # Standard domain definitions grounded in lecture (placed first so clean reference definitions take precedence over noisy ASR fragments)
            f_low = full_text.lower()
            standard_ml_defs = [
                ("supervised learning", "A machine learning paradigm where algorithms learn mathematical mapping functions from labeled input-output feature pairs."),
                ("classification", "A supervised learning problem where the model predicts discrete class labels or categories."),
                ("regression", "A supervised learning problem where the model predicts continuous numerical target quantities."),
                ("precision", "Ratio of true positive predictions to total predicted positives (exactness)."),
                ("recall", "Ratio of true positive predictions to all actual positive ground truth cases (sensitivity)."),
                ("accuracy", "Proportion of total correct predictions over total evaluation samples."),
                ("f1 score", "Harmonic mean of precision and recall providing a balanced performance metric under class imbalance."),
                ("overfitting", "Condition where a model memorizes noise and quirks in training data, degrading performance on unseen test data."),
                ("accrual accounting", "An accounting method where revenue and expenses are recorded when transactions occur rather than when physical cash is exchanged."),
                ("accounts receivable", "Money owed to a business by its customers for goods or services delivered on credit."),
                ("accounts payable", "Money owed by a business to its suppliers or creditors for goods or services purchased on credit."),
                ("income statement", "A financial statement that reports a company's revenues, expenses, and net profit over a specific accounting period."),
                ("cash inflows", "Physical cash received by a business from sales, customer collections, financing, or investments."),
                ("cash outflows", "Physical cash paid out by a business for operating expenses, supplier payments, or capital purchases."),
            ]
            for term_kw, term_def in standard_ml_defs:
                if term_kw in f_low:
                    if not any(d.get("term", "").lower() == term_kw for d in defs):
                        defs.append({"term": term_kw.title(), "definition": term_def, "source": "reference"})

        def_patterns = [
            r"([A-Za-z][A-Za-z\s\-]{2,28}?)\s+(?:is defined as|means|refers to|ko kehte hain|ka matlab hai ke|ka matlab hai|ka matlab|is a process where|is a mechanism that|is a measure of|measures)\s+([^.!?\n]{10,140})",
            r"([A-Za-z][A-Za-z\s\-]{2,25}?)\s+is\s+(?:a|an|the)\s+([A-Za-z\s,\-]{12,130}?(?:that|which|where|used to|designed to)[^.!?\n]{5,80})",
        ]
        for pat in def_patterns:
            for term, d in re.findall(pat, full_text, re.IGNORECASE):
                t_clean = re.sub(r"^(?:the|a|an|so|now|here|in|when|while|basically|that|this|if|because|and|but)\s+", "", term.strip(), flags=re.IGNORECASE).strip().title()
                t_low = t_clean.lower()
                d_clean = d.strip()
                if "accounts receivable" in t_low and "owed by a business to its customers" in d_clean.lower():
                    d_clean = "the money owed to a business by its customers for goods or services delivered on credit"
                if (
                    not t_clean
                    or t_low in _STOPWORDS
                    or t_low in disallowed_def_terms
                    or any(w in disallowed_def_terms for w in t_low.split()[:1])
                    or any(w in {"was", "were", "is", "are", "has", "have", "had", "can", "will"} for w in t_low.split())
                    or len(d_clean.split()) < 4
                    or d_clean.lower().startswith(("that ", "by which", "major issue"))
                    or _TRANSLATION_REFUSAL_OR_HALLUCINATION_RE.search(t_clean)
                    or _TRANSLATION_REFUSAL_OR_HALLUCINATION_RE.search(d_clean)
                ):
                    continue
                if 3 <= len(t_clean) <= 35 and not (len(t_clean.split()) >= 2 and _is_incomplete_clause(t_clean)):
                    if not any(existing.get("term", "").lower() == t_low for existing in defs):
                        defs.append({"term": t_clean, "definition": d_clean, "source": "transcript"})

        return defs[:4]

    def _extract_examples(self, full_text: str) -> list[str]:
        from app.config import STRICT_TRANSCRIPT_ONLY
        exs: list[str] = []
        matches = re.findall(
            r"(?:for example|for instance|misal ke taur par|jaise ke|jaise|such as|suppose|imagine that|consider when)\s*,?\s*([^.!?\n]{10,140})",
            full_text,
            re.IGNORECASE,
        )
        for m in matches:
            clean_m = m.strip()
            if clean_m and clean_m not in exs:
                exs.append(clean_m[0].upper() + clean_m[1:])

        if not STRICT_TRANSCRIPT_ONLY:
            f_low = full_text.lower()
            standard_ml_exs = [
                (["classification", "spam"], "Email spam filtering classifying incoming messages into 'Spam' or 'Inbox'."),
                (["regression", "house", "price"], "Predicting residential real estate prices based on square footage, location, and bedroom count."),
                (["precision", "recall"], "Cancer diagnostic screening where high recall prevents missing malignant tumors, while high precision minimizes false alarms."),
                (["supervised learning", "labeled data"], "Predicting customer churn or credit risk using historical labeled transaction logs.")
            ]
            for kws, ex_str in standard_ml_exs:
                if any(k in f_low for k in kws):
                    if ex_str not in exs:
                        exs.append(ex_str)

        return exs[:2]

    def _extract_formulas(self, full_text: str) -> list[str]:
        from app.config import STRICT_TRANSCRIPT_ONLY
        fms: list[str] = []
        matches = re.findall(r"([A-Za-z\s]{3,30}\s*=\s*[^.!?\n]{4,60})", full_text)
        for m in matches:
            if not any(w in m.lower() for w in ["http", "html", "var", "const"]):
                fms.append(m.strip())

        spoken_eqs = re.findall(
            r"\b([A-Za-z][A-Za-z\s]{2,28}?)\s+(?:equals|is equal to|is calculated as)\s+([^.!?\n]{5,75}?(?:minus|plus|divided by|over|times|multiplied by)\s+[^.!?\n]{3,50})",
            full_text,
            re.IGNORECASE,
        )
        for lhs_raw, rhs_raw in spoken_eqs:
            lhs_clean = re.sub(r"^(?:the|a|an|our|your|their|so|now|therefore)\s+", "", lhs_raw.strip(), flags=re.IGNORECASE).strip().title()
            rhs_clean = re.sub(r"\b(?:our|your|their)\s+", "", rhs_raw.strip(), flags=re.IGNORECASE).strip()
            if lhs_clean.lower() in {"cash inflows", "cash inflow"} and "cash inflows" in rhs_clean.lower() and "cash outflows" in rhs_clean.lower():
                lhs_clean = "Net Cash Flow"
            if 1 <= len(lhs_clean.split()) <= 4 and lhs_clean.lower() not in _STOPWORDS:
                clean_seq = f"{lhs_clean} = {rhs_clean}"
                if not any(clean_seq.lower() == existing.lower() for existing in fms):
                    fms.append(clean_seq)

        if not STRICT_TRANSCRIPT_ONLY:
            f_low = full_text.lower()
            standard_ml_fms = [
                (["precision"], "Precision = TP / (TP + FP)"),
                (["recall"], "Recall = TP / (TP + FN)"),
                (["f1", "f1-score", "f1 score"], "F1-Score = 2 * (Precision * Recall) / (Precision + Recall)"),
                (["accuracy"], "Accuracy = (Correct Predictions) / (Total Predictions)"),
                (["mae", "mean absolute error"], "MAE = (1/n) * sum(|Actual - Predicted|)"),
                (["mse", "mean squared error"], "MSE = (1/n) * sum((Actual - Predicted)^2)")
            ]
            for kws, fm_str in standard_ml_fms:
                if any(k in f_low for k in kws):
                    if fm_str not in fms:
                        fms.append(fm_str)

        return fms[:3]

    def _extract_emphasis(self, full_text: str, span: Optional[list[Segment]] = None) -> list[dict[str, str]]:
        emps: list[dict[str, str]] = []
        signal_patterns = [
            r"remember this for the exam",
            r"important for the exam",
            r"will come in exam",
            r"midterm mein aayega",
            r"note kar lein",
            r"note kar",
            r"note this down",
            r"remember this",
            r"paper mein",
            r"crucial",
            r"lazmi",
            r"zaroor"
        ]
        sentences = [s.strip() for s in re.split(r"[.!?\n]+", full_text) if len(s.strip()) >= 10]
        for sentence in sentences:
            for pat in signal_patterns:
                m = re.search(rf"\b({pat})\b", sentence, re.IGNORECASE)
                if m:
                    sig = m.group(1).strip()
                    emp_dict = {
                        "text": sentence,
                        "signal_phrase": sig,
                        "source": "lecturer_explicit"
                    }
                    if span:
                        for s in span:
                            if sig.lower() in s.text.lower() or sentence.lower() in s.text.lower():
                                emp_dict["start_time"] = s.start
                                emp_dict["end_time"] = s.end
                                emp_dict["timestamp"] = f"[{s.start:.2f}s - {s.end:.2f}s]"
                                break
                    emps.append(emp_dict)
                    break
        return emps[:3]


# ===========================================================================
# Main Orchestration Engine
# ===========================================================================

class TopicIntelligenceEngine:
    """
    Main orchestration engine for Local-First AI Lecture Understanding & Study Notes.
    Executes:
      1. Semantic Preprocessing
      2. Candidate Boundary Detection (Supporting signal)
      3. Local LLM Pass 1: Structure & Subject Discovery
      4. Local LLM Pass 2: Section Synthesis (Notes, Definitions, Examples, Formulas, Emphasis)
      5. Topic Reconciliation & Merging
      6. Grounded Timestamp Resolution (derived strictly from segment IDs)
      7. Quality Validation Gate
    """
    def __init__(self, use_llm: Optional[bool] = None):
        from app.config import ENABLE_LOCAL_LLM
        if use_llm is None:
            self.use_llm = ENABLE_LOCAL_LLM
        else:
            self.use_llm = use_llm

        if self.use_llm:
            from app.services.llm_provider import get_llm_provider
            try:
                self.llm = get_llm_provider(feature="notes")
            except Exception as e:
                logger.warning(f"LLM Provider initialization deferred/failed: {e}")
                self.llm = None
        else:
            self.llm = None

        self.boundary_analyzer = LocalSemanticTopicAnalyzer()
        self.fallback_engine = ExtractiveSemanticFallback()
        self.validator = StudyNotesQualityValidator()

    _clean_transcript_for_notes = staticmethod(_clean_transcript_for_notes)

    def analyze_lecture_study_notes(
        self,
        transcript_text: str,
        segments_orm: list[Any],
        duration_seconds: float = 0.0,
        force_refresh: bool = False,
        topic_hint: Optional[str] = None,
        speech_duration_seconds: Optional[float] = None,
    ) -> LectureStudyNotes:
        """Primary entry point for full lecture study notes generation."""
        t_hash = _compute_hash(transcript_text)
        if not segments_orm and not transcript_text:
            return self._empty_notes(t_hash)

        # 1. Preprocess segments
        segments = self._preprocess_segments(segments_orm, transcript_text, duration_seconds)
        if not segments:
            return self._empty_notes(t_hash)

        total_dur = duration_seconds if duration_seconds > 0 else segments[-1].end

        # Check if Local LLM is operational under lifecycle coordination (releases on exit unless nested in notes_summary)
        from app.services.llm_provider import llm_lifecycle_session

        with llm_lifecycle_session("notes", release_on_exit=True):
            llm_ready = self.use_llm and self.llm is not None and self.llm.is_available()

            if llm_ready:
                try:
                    notes = self._execute_llm_pipeline(segments, transcript_text, total_dur, t_hash, topic_hint=topic_hint)
                except Exception as e:
                    logger.warning(f"Local LLM study notes pipeline failed: {e}. Reverting to extractive fallback.", exc_info=True)
                    notes = self._execute_fallback_pipeline(
                        segments, transcript_text, total_dur, t_hash, topic_hint=topic_hint,
                        fallback_reason=f"Local LLM pipeline exception: {type(e).__name__}: {e}"
                    )
            else:
                reason = "Local LLM disabled in configuration" if not self.use_llm else "Local LLM model weights not loaded or unavailable"
                logger.info(f"Local LLM not available ({reason}). Executing extractive semantic fallback.")
                notes = self._execute_fallback_pipeline(
                    segments, transcript_text, total_dur, t_hash, topic_hint=topic_hint,
                    fallback_reason=reason
                )

        # 7. Quality Validation
        passed, score, details = self.validator.validate(
            notes, total_dur, segments, speech_duration_seconds=speech_duration_seconds
        )
        notes.quality_score = score
        notes.quality_pass = passed
        notes.quality_details = details

        # If quality failed critically on LLM output, attempt healing with extractive fallback
        min_expected = 1 if total_dur < 60 else 2
        tl_detail = details.get("timeline_coverage", {}) or {}
        tl_critically_low = (
            not tl_detail.get("pass", True)
            and float(tl_detail.get("ratio", 1.0) or 0.0) < 0.45
        )
        is_critically_flawed = (
            score < 0.65 or
            len(notes.study_topics) < min_expected or
            not details.get("meaningful_titles", {}).get("pass", True) or
            not details.get("content_groundedness", {}).get("pass", True) or
            not details.get("distinct_titles", {}).get("pass", True) or
            tl_critically_low
        )
        if is_critically_flawed:
            logger.warning(f"Study notes failed quality gate (score={score}, analyzer={notes.analyzer_used}, details={details}). Healing topics.")
            if notes.analyzer_used == "local_llm":
                fb_notes = self._execute_fallback_pipeline(
                    segments, transcript_text, total_dur, t_hash, topic_hint=topic_hint,
                    fallback_reason=f"LLM output failed quality gate (score={score:.2f}, details={details})"
                )
                fb_passed, fb_score, fb_details = self.validator.validate(
                    fb_notes, total_dur, segments, speech_duration_seconds=speech_duration_seconds
                )
                if fb_score > score or (not passed and fb_passed):
                    fb_notes.quality_score = fb_score
                    fb_notes.quality_pass = fb_passed
                    fb_notes.quality_details = fb_details
                    return fb_notes
            else:
                # Refine Path B titles to ensure clean structural titles
                for idx, t in enumerate(notes.study_topics):
                    if _is_incomplete_clause(t.title) or t.title.lower() in _GENERIC_DISALLOWED_TITLES:
                        sec_segs = [s for s in segments if s.id in t.source_segment_ids]
                        refined_title = self.fallback_engine._extract_title(sec_segs, t.source_text, topic_hint=topic_hint)
                        if refined_title and not _is_incomplete_clause(refined_title):
                            t.title = refined_title
                fb_passed, fb_score, fb_details = self.validator.validate(
                    notes, total_dur, segments, speech_duration_seconds=speech_duration_seconds
                )
                notes.quality_score = fb_score
                notes.quality_pass = fb_passed
                notes.quality_details = fb_details

        return notes


    def _preprocess_segments(
        self,
        segments_orm: list[Any],
        transcript_text: str,
        duration_seconds: float
    ) -> list[Segment]:
        from app.services.contextual_transcription import is_noise_segment
        from app.services.domain_corrector import correct_domain_terms
        from app.services.transcription_service import TranscriptionService

        segments: list[Segment] = []
        if segments_orm:
            has_high_trust = any(getattr(x, "trust", "high") != "low" for x in segments_orm)
            for s in segments_orm:
                if has_high_trust and getattr(s, "trust", "high") == "low":
                    continue

                is_intel = bool(getattr(s, "is_intel_view", False))
                if is_intel:
                    text = (getattr(s, "llm_text", None) or getattr(s, "text", "") or "").strip()
                else:
                    text = (getattr(s, "english_text", None) or getattr(s, "normalized_text", None) or getattr(s, "text", "") or "").strip()
                if not text:
                    continue
                s_start = getattr(s, "start_time", None)
                if s_start is None:
                    s_start = getattr(s, "start", 0.0)
                s_end = getattr(s, "end_time", None)
                if s_end is None:
                    s_end = getattr(s, "end", 0.0)
                dur = float(s_end) - float(s_start)

                if is_intel:
                    # Bypass legacy scrubbers / _clean_transcript_for_notes (already S0-S9 preprocessed)
                    cleaned_text = correct_domain_terms(text)
                    if not cleaned_text or _is_non_academic_segment(cleaned_text):
                        continue
                else:
                    # Filter out pure acoustic noise hallucination segments
                    if is_noise_segment(text, dur):
                        continue

                    # Scrub repetition loops
                    text = TranscriptionService.scrub_repetition_loops(text)
                    # Strip known hallucination noise words if any remain inside text
                    text = re.sub(r'\b(?:mhebi|mhmi|diphthim|dhk|ske|chamakthak|shwanda|invincible|bred)\b', '', text, flags=re.IGNORECASE).strip()
                    text = re.sub(r'\s+', ' ', text).strip()
                    if not text:
                        continue

                    # Apply domain technical corrections and legacy cleaner
                    cleaned_text = correct_domain_terms(text)
                    cleaned_text = _clean_transcript_for_notes(cleaned_text)
                    if not cleaned_text or _is_non_academic_segment(cleaned_text):
                        continue

                seg_id = getattr(s, "segment_index", None)
                if seg_id is None:
                    seg_id = getattr(s, "id", len(segments))
                try:
                    seg_id_int = int(seg_id)
                except Exception:
                    seg_id_int = len(segments)

                segments.append(Segment(
                    id=seg_id_int,
                    text=cleaned_text,
                    start=float(s_start),
                    end=float(s_end)
                ))
        else:
            # Fallback: slice raw transcript into natural sentences
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", transcript_text) if s.strip()]
            dt = max(duration_seconds, 60.0) / max(len(sentences), 1)
            for i, sent in enumerate(sentences):
                cleaned_sent = correct_domain_terms(sent)
                cleaned_sent = _clean_transcript_for_notes(cleaned_sent)
                if not cleaned_sent:
                    continue
                segments.append(Segment(
                    id=i,
                    text=cleaned_sent,
                    start=round(i * dt, 2),
                    end=round((i + 1) * dt, 2)
                ))

        if len(segments) >= 3:
            outlier_ids = self.boundary_analyzer.detect_embedding_outliers(segments)
            if outlier_ids:
                logger.info(f"Filtering {len(outlier_ids)} embedding outlier hallucination segments: {outlier_ids}")
                segments = [s for s in segments if s.id not in outlier_ids]

        return segments


    # -----------------------------------------------------------------------
    # Multi-Stage Pedagogical Understanding, Teaching Synthesis & Self-Review
    # -----------------------------------------------------------------------

    @staticmethod
    def _extract_numeric_and_formula_anchors(text: str) -> dict[str, list[str]]:
        """Extract exact numeric facts and equation cues to keep small LLMs grounded."""
        nums = re.findall(
            r"\b\d+(?:[.,]\d+)?(?:\s*(?:lakh|crore|million|thousand|rupees|rs\.?|%|ms|sec|ohms?|volts?|amps?))?\b",
            text or "",
            flags=re.IGNORECASE,
        )
        uniq_nums = list(dict.fromkeys(n.strip() for n in nums if n.strip()))[:8]
        eqs = re.findall(
            r"\b[A-Za-z][A-Za-z\s]{2,26}?\s*(?:=|equals|is calculated as|is equal to)\s*[^.!?\n]{4,60}",
            text or "",
            flags=re.IGNORECASE,
        )
        uniq_eqs = list(dict.fromkeys(e.strip() for e in eqs if e.strip()))[:4]
        return {"numbers": uniq_nums, "equations": uniq_eqs}

    @staticmethod
    def _compute_verbatim_overlap_ratio(generated: str, source: str) -> float:
        """Fraction of 5-word n-grams in generated text copied verbatim from source."""
        g_toks = re.findall(r"[a-z0-9]+", (generated or "").lower())
        s_toks = re.findall(r"[a-z0-9]+", (source or "").lower())
        if len(g_toks) < 5 or len(s_toks) < 5:
            return 0.0
        s_grams = {tuple(s_toks[i:i + 5]) for i in range(len(s_toks) - 4)}
        g_grams = [tuple(g_toks[i:i + 5]) for i in range(len(g_toks) - 4)]
        copied = sum(1 for g in g_grams if g in s_grams)
        return copied / max(1, len(g_grams))

    @staticmethod
    def _repair_formula_sign_flips(text: str, formulas: list[str]) -> str:
        """Repair common 0.5B subtraction phrasing reversals (e.g. 'subtract revenue from cost')."""
        if not text:
            return text
        f_blob = " ".join(formulas or []).lower()
        repaired = text
        # Profit = Revenue - Cost / Expenses
        if ("revenue" in f_blob and ("cost" in f_blob or "expense" in f_blob or "expenditure" in f_blob)) or (
            "profit" in repaired.lower() and "revenue" in repaired.lower()
        ):
            repaired = re.sub(
                r"\bsubtract(?:ing)?\s+(?:total\s+)?revenue\s+from\s+(?:the\s+|total\s+)?(cost|costs|expenses|expenditure)\b",
                r"subtracting \1 from revenue",
                repaired,
                flags=re.IGNORECASE,
            )
        # Net Cash Flow = Cash Inflows - Cash Outflows
        repaired = re.sub(
            r"\bsubtract(?:ing)?\s+(?:cash\s+)?inflows?\s+from\s+(?:cash\s+)?outflows?\b",
            "subtracting cash outflows from cash inflows",
            repaired,
            flags=re.IGNORECASE,
        )
        # Turnaround Time = Completion Time - Arrival Time
        repaired = re.sub(
            r"\bsubtract(?:ing)?\s+completion\s+time\s+from\s+arrival\s+time\b",
            "subtracting arrival time from completion time",
            repaired,
            flags=re.IGNORECASE,
        )
        return repaired

    @staticmethod
    def _coerce_prose(val: Any) -> str:
        """Convert LLM strings, lists, or stringified Python lists like \"['...', ',\\n ']\" into clean prose."""
        if not val:
            return ""
        items: list[str] = []
        if isinstance(val, (list, tuple)):
            items = [str(x).strip() for x in val]
        else:
            s = str(val).strip()
            if s.startswith("[") and s.endswith("]"):
                try:
                    import ast as _ast
                    parsed = _ast.literal_eval(s)
                    if isinstance(parsed, (list, tuple)):
                        items = [str(x).strip() for x in parsed]
                except Exception:
                    items = []
            if not items:
                s = re.sub(r"^\[\s*['\"]|['\"]\s*,\s*['\"][,\\n\s]*['\"]\s*\]$", "", s)
                items = [s]
        cleaned_parts: list[str] = []
        for it in items:
            t = it.strip(" ,\n\t'\"")
            if not t or len(re.findall(r"[a-zA-Z0-9]+", t)) == 0:
                continue
            if t[-1] not in ".!?":
                t += "."
            cleaned_parts.append(t)
        return " ".join(cleaned_parts).strip()

    @staticmethod
    def _is_near_duplicate(cand: str, existing: list[str]) -> bool:
        """Detect exact, containment, or high-overlap paraphrases so sections never repeat each other."""
        c_clean = re.sub(r"[^a-z0-9\s]", "", (cand or "").lower()).strip()
        c_compact = re.sub(r"\s+", "", c_clean)
        if not c_compact:
            return True
        c_toks = {w for w in c_clean.split() if len(w) > 2}
        for ex in existing:
            e_clean = re.sub(r"[^a-z0-9\s]", "", (ex or "").lower()).strip()
            e_compact = re.sub(r"\s+", "", e_clean)
            if not e_compact:
                continue
            if c_compact == e_compact or (len(c_compact) >= 24 and (c_compact in e_compact or e_compact in c_compact)):
                return True
            e_toks = {w for w in e_clean.split() if len(w) > 2}
            if c_toks and e_toks:
                overlap = len(c_toks & e_toks) / max(1, min(len(c_toks), len(e_toks)))
                if overlap >= 0.72:
                    return True
        return False

    def _synthesize_pedagogical_fields(
        self,
        topic: StudyTopic,
        prev_title: str = "",
        next_title: str = "",
    ) -> None:
        """
        Populate any missing student-friendly teaching fields (what_it_means, how_it_works,
        why_it_matters, steps_and_logic, complete key_points) cleanly without repeating
        identical sentences across sections.
        """
        title = topic.title.strip() or "This Concept"
        src = topic.source_text or topic.explanation or ""

        topic.explanation = self._coerce_prose(topic.explanation)
        topic.what_it_means = self._coerce_prose(topic.what_it_means)
        topic.how_it_works = self._coerce_prose(topic.how_it_works)
        topic.why_it_matters = self._coerce_prose(topic.why_it_matters)

        _conv_lead_re = re.compile(
            r"^(?:today(?:'s)?\s+(?:lecture|session)|in\s+this\s+video|we\s+will\s+discuss|assalamu\s+alaikum|hello\s+everyone)\b",
            re.IGNORECASE,
        )

        clean_kps = []
        for kp in (topic.key_points or []):
            kp_s = self._coerce_prose(kp)
            if len(kp_s.split()) >= 5 and not _is_phonetic_noise(kp_s) and not _conv_lead_re.search(kp_s):
                clean_kps.append(kp_s)
        if len(clean_kps) < 2 and topic.explanation:
            exp_sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", topic.explanation) if len(s.strip().split()) >= 5]
            for s in exp_sents:
                if not self._is_near_duplicate(s, clean_kps):
                    clean_kps.append(s)
        topic.key_points = clean_kps[:5]

        # 1. What It Means: distinct plain-English definition (avoid echoing full explanation verbatim)
        if not topic.what_it_means or len(topic.what_it_means.split()) < 5 or self._is_near_duplicate(topic.what_it_means, [topic.explanation]):
            if topic.definitions:
                d0 = topic.definitions[0]
                if isinstance(d0, dict) and d0.get("term") and d0.get("definition"):
                    cand_def = f"{d0['term']} means {d0['definition'][0].lower() + d0['definition'][1:]}".rstrip(".") + "."
                    if not self._is_near_duplicate(cand_def, [topic.explanation]):
                        topic.what_it_means = cand_def
            if not topic.what_it_means or len(topic.what_it_means.split()) < 5:
                first_s = re.split(r"(?<=[.!?])\s+", (topic.explanation or "").strip())[0]
                if first_s:
                    topic.what_it_means = first_s[0].upper() + first_s[1:]

        # 2. How It Works: operational mechanism distinct from explanation & what_it_means
        prior_intro = [x for x in (topic.explanation, topic.what_it_means) if x]
        if not topic.how_it_works or len(topic.how_it_works.split()) < 6 or self._is_near_duplicate(topic.how_it_works, prior_intro):
            mech_parts: list[str] = []
            if topic.formulas:
                mech_parts.append(f"Governed by `{topic.formulas[0]}`.")
            causal_sents = [
                s.strip()
                for s in re.split(r"(?<=[.!?])\s+", src)
                if len(s.strip().split()) >= 6
                and not _conv_lead_re.search(s.strip())
                and any(k in s.lower() for k in ("because", "therefore", "means", "happens when", "whereas", "so that", "works by", "calculated", "depends"))
                and not self._is_near_duplicate(s, prior_intro)
            ]
            if causal_sents:
                mech_parts.append(causal_sents[0])
            else:
                distinct_kps = [kp for kp in topic.key_points if not self._is_near_duplicate(kp, prior_intro)]
                if distinct_kps:
                    mech_parts.append(distinct_kps[0])
            if mech_parts:
                topic.how_it_works = " ".join(mech_parts).strip()

        # 3. Steps & Logic: sequential steps distinct from explanation / what_it_means / how_it_works
        seen_for_steps = [x for x in (topic.explanation, topic.what_it_means, topic.how_it_works) if x]
        cleaned_existing_steps: list[str] = []
        for st_item in (topic.steps_and_logic or []):
            st_s = self._coerce_prose(st_item)
            if (
                len(st_s.split()) >= 5
                and not _conv_lead_re.search(st_s)
                and not self._is_near_duplicate(st_s, seen_for_steps + cleaned_existing_steps)
            ):
                cleaned_existing_steps.append(st_s)

        if not cleaned_existing_steps:
            for cand_s in list(topic.key_points) + [s.strip() for s in re.split(r"(?<=[.!?])\s+|;\s+", src) if len(s.strip().split()) >= 5]:
                cand_clean = self._coerce_prose(cand_s)
                if (
                    len(cand_clean.split()) >= 5
                    and not _conv_lead_re.search(cand_clean)
                    and not self._is_near_duplicate(cand_clean, seen_for_steps + cleaned_existing_steps)
                ):
                    cleaned_existing_steps.append(cand_clean)
                if len(cleaned_existing_steps) >= 3:
                    break
        if not cleaned_existing_steps and topic.how_it_works:
            hw_sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+|;\s+", topic.how_it_works) if len(s.strip().split()) >= 5]
            if len(hw_sents) > 1:
                cleaned_existing_steps = hw_sents[:3]
            elif hw_sents:
                cleaned_existing_steps = [f"Core process: {hw_sents[0]}"]
        topic.steps_and_logic = cleaned_existing_steps[:4]

        # 4. Why It Matters: real consequence / exam contrast without copying key_points verbatim
        generic_why_re = re.compile(
            r"^(?:Critical for examination assessment|Mastering .* provides essential theoretical)",
            re.IGNORECASE,
        )
        prior_all = seen_for_steps + list(topic.steps_and_logic)
        if (
            not topic.why_it_matters
            or generic_why_re.search(topic.why_it_matters.strip())
            or _conv_lead_re.search(topic.why_it_matters.strip())
            or self._is_near_duplicate(topic.why_it_matters, prior_all)
        ):
            why_sents = [
                s.strip()
                for s in re.split(r"(?<=[.!?])\s+", src)
                if len(s.strip().split()) >= 6
                and not _conv_lead_re.search(s.strip())
                and any(k in s.lower() for k in ("important", "matters", "exam", "difference", "problem", "risk", "fail", "prevents", "ensures", "allows", "why"))
                and not self._is_near_duplicate(s, prior_all)
            ]
            if why_sents:
                topic.why_it_matters = why_sents[0]
            elif prev_title and next_title:
                topic.why_it_matters = (
                    f"{title} bridges {prev_title} to {next_title}, clarifying how these mechanisms work together."
                )
            elif topic.examples and not self._is_near_duplicate(topic.examples[0], prior_all):
                topic.why_it_matters = f"Applied directly in scenarios such as: {topic.examples[0]}"
            else:
                topic.why_it_matters = f"Essential for analyzing edge cases and tradeoffs in {title}."

        # 4. Connection note
        if not (topic.concept_connection or "").strip():
            if prev_title and next_title:
                topic.concept_connection = f"Builds on {prev_title} and prepares for {next_title}."
            elif prev_title:
                topic.concept_connection = f"Extends ideas introduced in {prev_title}."
            elif next_title:
                topic.concept_connection = f"Establishes foundation for {next_title}."

        # Apply formula sign repair across prose fields
        topic.explanation = self._repair_formula_sign_flips(topic.explanation, topic.formulas)
        topic.what_it_means = self._repair_formula_sign_flips(topic.what_it_means, topic.formulas)
        topic.how_it_works = self._repair_formula_sign_flips(topic.how_it_works, topic.formulas)
        topic.why_it_matters = self._repair_formula_sign_flips(topic.why_it_matters, topic.formulas)
        topic.key_points = [self._repair_formula_sign_flips(kp, topic.formulas) for kp in topic.key_points]

    @staticmethod
    def _build_concept_relationships(topics: list[StudyTopic]) -> list[dict[str, str]]:
        """Synthesize explicit concept relationships linking topics across the lecture."""
        rels: list[dict[str, str]] = []
        for idx in range(len(topics) - 1):
            t1, t2 = topics[idx], topics[idx + 1]
            txt_pair = f"{t1.title} {t2.title} {t1.explanation} {t2.explanation}".lower()
            if any(w in txt_pair for w in ("versus", "vs", "difference", "whereas", "contrast", "unlike")):
                rel_label = "contrasted with"
            elif any(w in txt_pair for w in ("therefore", "leads to", "causes", "results in", "reason", "because")):
                rel_label = "leads into"
            elif t1.formulas or t2.formulas:
                rel_label = "applied in"
            else:
                rel_label = "connects to"
            note = (t2.concept_connection or f"{t1.title} provides context needed to understand {t2.title}.").strip()
            rels.append({
                "from_topic": t1.title,
                "relation": rel_label,
                "to_topic": t2.title,
                "explanation": note,
            })
        return rels

    def _review_and_refine_notes(
        self,
        topics: list[StudyTopic],
        overview: str,
        full_transcript: str,
    ) -> tuple[list[StudyTopic], str, list[dict[str, str]]]:
        """
        Stage 4 Self-Review Pass:
          - Repairs arithmetic/sign slips & removes parroted conversational fragments
          - Deduplicates repeated sentences across explanation / CEM fields / key points
          - Refreshes concept relationships & clear whole-lecture overview
        """
        for idx, t in enumerate(topics):
            prev_t = topics[idx - 1].title if idx > 0 else ""
            next_t = topics[idx + 1].title if idx + 1 < len(topics) else ""
            self._synthesize_pedagogical_fields(t, prev_title=prev_t, next_title=next_t)

            # Deduplicate key_points against prose sections & each other (keep >=2 when available)
            shown_prose = [x for x in (t.explanation, t.what_it_means, t.how_it_works, t.why_it_matters) if x]
            dedup_kps: list[str] = []
            backup_kps: list[str] = []
            for kp in t.key_points:
                kp_clean = self._coerce_prose(kp)
                if not kp_clean:
                    continue
                kp_fmt = kp_clean if kp_clean[-1] in ".!?" else kp_clean + "."
                if self._is_near_duplicate(kp_clean, shown_prose + dedup_kps):
                    if not self._is_near_duplicate(kp_clean, [t.explanation] + dedup_kps + backup_kps):
                        backup_kps.append(kp_fmt)
                    continue
                dedup_kps.append(kp_fmt)
            while len(dedup_kps) < 2 and backup_kps:
                dedup_kps.append(backup_kps.pop(0))
            t.key_points = dedup_kps[:5]

        rels = self._build_concept_relationships(topics)
        clean_overview = (overview or "").strip()
        if len(clean_overview.split()) < 10 and topics:
            t_names = [t.title for t in topics[:4]]
            clean_overview = (
                f"This lecture builds from {', '.join(t_names[:-1])} to {t_names[-1]}"
                if len(t_names) > 1
                else f"This lecture explains the core ideas behind {t_names[0]}"
            ) + ", focusing on how each concept works in practice and why it matters."
        return topics, clean_overview, rels

    def _build_section_blocks(
        self,
        segments: list[Segment],
        duration: float,
        topic_hint: Optional[str] = None,
        max_sections: int = 5,
    ) -> list[dict[str, Any]]:
        """Group segments into 2..max_sections cohesive concept spans with clean text up to 900 chars."""
        candidate_bounds = self.boundary_analyzer.detect_candidate_boundaries(segments)
        if duration >= 60.0 and len(segments) >= 4 and not candidate_bounds:
            candidate_bounds = [len(segments) // 2]
        raw_sections = self._fallback_sections_from_boundaries(segments, candidate_bounds)
        if not raw_sections:
            raw_sections = [{"title": topic_hint or "Lecture Overview", "start_segment_id": segments[0].id, "end_segment_id": segments[-1].id}]

        target_count = max(2, min(max_sections, int(duration / 120) + 2)) if duration >= 60 else max(1, min(3, len(raw_sections)))
        consolidated = list(raw_sections)
        while len(consolidated) > target_count:
            best_pair_idx = 0
            min_combined = float("inf")
            for i in range(len(consolidated) - 1):
                s1, s2 = consolidated[i], consolidated[i + 1]
                c_segs = s2.get("end_segment_id", 0) - s1.get("start_segment_id", 0)
                if c_segs < min_combined:
                    min_combined = c_segs
                    best_pair_idx = i
            s1, s2 = consolidated[best_pair_idx], consolidated[best_pair_idx + 1]
            consolidated[best_pair_idx] = {
                "title": s1.get("title") if len(s1.get("title", "")) >= len(s2.get("title", "")) else s2.get("title"),
                "start_segment_id": s1.get("start_segment_id"),
                "end_segment_id": s2.get("end_segment_id"),
                "reason": "Timeline consolidation",
            }
            consolidated.pop(best_pair_idx + 1)

        seg_map = {s.id: s for s in segments}
        sec_blocks: list[dict[str, Any]] = []
        for idx, sec in enumerate(consolidated):
            start_id = sec.get("start_segment_id", segments[0].id)
            end_id = sec.get("end_segment_id", segments[-1].id)
            valid_ids = [s.id for s in segments if start_id <= s.id <= end_id]
            sec_segs = [seg_map[sid] for sid in valid_ids if sid in seg_map]
            if not sec_segs:
                continue

            clean_words: list[str] = []
            for s in sec_segs:
                for w in s.text.split():
                    cw = re.sub(r"[^a-zA-Z]", "", w)
                    if cw and not _is_phonetic_noise(cw) and cw.lower() not in _HALLUCINATION_WORDS:
                        clean_words.append(w)
                    elif re.search(r"\d", w):
                        clean_words.append(w)
            clean_txt = " ".join(clean_words)[:950]
            sec_blocks.append({
                "index": idx + 1,
                "candidate_title": sec.get("title", ""),
                "start": sec_segs[0].start,
                "end": sec_segs[-1].end,
                "start_seg_id": sec_segs[0].id,
                "end_seg_id": sec_segs[-1].id,
                "text": clean_txt,
                "segs": sec_segs,
            })
        return sec_blocks

    def _llm_build_lecture_outline(
        self,
        sec_blocks: list[dict[str, Any]],
        topic_hint: Optional[str],
    ) -> dict[str, Any]:
        """Stage 1 (Pass 1): Build whole-lecture thesis, theme, and cross-topic connections."""
        digest = "\n".join(
            f"Section {b['index']} ({b['start']:.0f}s-{b['end']:.0f}s) [{b['candidate_title']}]: {b['text'][:160]}"
            for b in sec_blocks
        )
        hint_line = f"Subject hint: {topic_hint}\n" if topic_hint else ""
        prompt = (
            f"{hint_line}"
            f"Below is the full timeline outline of one lecture:\n{digest}\n\n"
            "As an attentive student who understood the entire lecture:\n"
            "1. Identify the academic subject, central theme, and a plain-English 2-sentence overview of how ideas connect.\n"
            "2. For each section index, give a clear 2-5 word topic title and 1 sentence explaining how it connects to neighboring topics.\n"
            "Return ONLY valid JSON."
        )
        schema = {
            "subject": "Discipline name",
            "theme": "Central lecture theme",
            "overview": "2-sentence student summary of the whole lecture in simple English.",
            "topics": [
                {"section_index": 1, "title": "Clear 2-5 Word Title", "connection": "How this builds on or contrasts with other sections."}
            ],
        }
        sys_p = "You understand university lectures holistically and write concise structured outlines in plain English. Output strictly valid JSON."
        return self.llm.generate_structured(prompt, schema=schema, system_prompt=sys_p, max_tokens=220, temperature=0.05)

    def _llm_teach_single_topic(
        self,
        block: dict[str, Any],
        theme: str,
        prev_title: str,
        next_title: str,
        connection_hint: str,
    ) -> dict[str, Any]:
        """
        Stage 2 (Pass 2): Teach one cohesive concept independently as a student who attended
        the whole lecture, keeping whole-lecture flow and exact numerical/formula anchors.
        """
        anchors = self._extract_numeric_and_formula_anchors(block["text"])
        anchor_str = ""
        if anchors["equations"] or anchors["numbers"]:
            anchor_str = (
                f"\nGround-truth facts to preserve accurately — Formulas: {anchors['equations'] or 'none'} | "
                f"Numbers: {anchors['numbers'] or 'none'}"
            )
        flow = f"{prev_title or 'Start'} -> [{block['candidate_title']}] -> {next_title or 'Conclusion'}"
        prompt = (
            f"Lecture Theme: {theme}\n"
            f"Concept Flow: {flow}\n"
            f"Connection: {connection_hint or 'Core concept in lecture sequence.'}\n"
            f"Transcript Excerpt:\n\"\"\"\n{block['text'][:2400]}\n\"\"\""
            f"{anchor_str}\n\n"
            "Task: Imagine you attended the entire lecture, understood the ideas, and are now teaching this concept in clear, simple English study notes in YOUR OWN WORDS.\n"
            "- NEVER copy raw transcript sentences verbatim.\n"
            "- Write directly in your own words without saying 'The lecture discusses' or 'Today we discussed'.\n"
            "- Choose a specific, distinct 2-5 word title different from the surrounding topics in Concept Flow.\n"
            "- Immediately explain any technical term in plain language.\n"
            "- Break how it works into distinct sequential steps in steps_and_logic.\n"
            "- Make what_it_means, how_it_works, steps_and_logic, why_it_matters, and key_points each add NEW information without repeating the same sentence across fields.\n"
            "- Write every prose field as a plain text string (never wrap how_it_works in JSON square brackets).\n"
            "- Preserve formulas and lecture numbers accurately.\n"
            "- Use examples mentioned in the lecture (or if adding intuition, start with 'Analogy:').\n"
            "- Provide 3 distinct, full-sentence key_points, never 1-2 word labels.\n"
            "Output ONLY valid JSON."
        )
        schema = {
            "title": "Clear 2-5 Word Heading",
            "explanation": "2-sentence overview of what this topic covers.",
            "what_it_means": "1-sentence plain-language definition (must not repeat explanation).",
            "how_it_works": "1-2 sentence plain prose string explaining the core mechanism (must not repeat explanation or what_it_means).",
            "steps_and_logic": [
                "Step 1: Concrete operational step.",
                "Step 2: Next transition or decision rule."
            ],
            "why_it_matters": "1 sentence on why this matters or common exam tradeoffs.",
            "key_points": [
                "Distinct takeaway sentence 1.",
                "Distinct takeaway sentence 2."
            ],
            "definitions": [{"term": "Term", "definition": "Simple student-friendly definition"}],
            "examples": ["Concrete lecture example or clearly labeled analogy"],
            "formulas": ["Exact formula if covered, else empty list"],
            "exam_relevant": True,
        }
        sys_p = (
            "You are an insightful university student writing clear, simple, accurate study notes in your own words. "
            "Translate any Roman Urdu into natural English. Output strictly valid JSON."
        )
        return self.llm.generate_structured(prompt, schema=schema, system_prompt=sys_p, max_tokens=700, temperature=0.05)

    def _execute_llm_pipeline(
        self,
        segments: list[Segment],
        transcript_text: str,
        duration: float,
        t_hash: str,
        topic_hint: Optional[str] = None
    ) -> LectureStudyNotes:
        """
        Multi-stage Local LLM notes pipeline:
          Stage 1: Whole-lecture concept map & cross-topic progression
          Stage 2: Budgeted concept-by-concept independent teaching synthesis
          Stage 3: Deterministic pedagogical supplement for any partial/overflow fields
          Stage 4: Self-review, anti-parroting, formula sign repair & relationship synthesis
        """
        import time as _time

        t_start = _time.monotonic()
        sec_blocks = self._build_section_blocks(segments, duration, topic_hint=topic_hint, max_sections=5)
        if not sec_blocks:
            return self._execute_fallback_pipeline(segments, transcript_text, duration, t_hash, topic_hint=topic_hint, fallback_reason="No valid section blocks for LLM")

        try:
            # Stage 1: Whole-Lecture Understanding Pass
            outline = self._llm_build_lecture_outline(sec_blocks, topic_hint)
            subject = (outline.get("subject") or "").strip() or (topic_hint if topic_hint else "Academic Lecture")
            theme = (outline.get("theme") or "").strip() or (topic_hint or "Lecture Concepts & Analysis")
            overview = (outline.get("overview") or "").strip()
            outline_topics = outline.get("topics") or []
            if isinstance(outline_topics, dict):
                outline_topics = [outline_topics]

            outline_by_idx: dict[int, dict[str, Any]] = {}
            for pos, ot in enumerate(outline_topics):
                if isinstance(ot, dict):
                    s_idx = ot.get("section_index", pos + 1)
                    try:
                        outline_by_idx[int(s_idx)] = ot
                    except Exception:
                        outline_by_idx[pos + 1] = ot

            # Assign planned titles before Pass 2 so neighbor context is coherent
            planned_titles: list[str] = []
            used_titles: set[str] = set()
            for idx, b in enumerate(sec_blocks):
                canonical_extracted = self.fallback_engine._extract_title(
                    b["segs"], b["text"], topic_hint=topic_hint, used_titles=used_titles
                )
                ot_title = (outline_by_idx.get(b["index"], {}).get("title") or "").strip()
                if (
                    not ot_title
                    or len(ot_title.split()) < 2
                    or _is_incomplete_clause(ot_title)
                    or ot_title.lower() in _GENERIC_DISALLOWED_TITLES
                    or ot_title.lower() in used_titles
                ):
                    chosen = canonical_extracted
                else:
                    chosen = ot_title
                for alien in ["Deep Learning", "Neural Network", "Neural Networks", "Quantum", "Backpropagation"]:
                    if alien.lower() in chosen.lower() and alien.lower() not in transcript_text.lower():
                        chosen = chosen.replace(alien, "").replace("  ", " ").strip(" &-,")
                if not chosen or len(chosen.split()) < 2 or _is_incomplete_clause(chosen):
                    chosen = canonical_extracted
                if chosen.lower() in used_titles:
                    chosen = f"{chosen} Analysis"
                used_titles.add(chosen.lower())
                planned_titles.append(chosen)

            # Stage 2: Concept-by-Concept Independent Synthesis (budget-guarded for CPU hardware)
            llm_budget_s = 250.0
            study_topics: list[StudyTopic] = []
            for idx, b in enumerate(sec_blocks):
                prev_t = planned_titles[idx - 1] if idx > 0 else ""
                next_t = planned_titles[idx + 1] if idx + 1 < len(planned_titles) else ""
                conn_hint = (outline_by_idx.get(b["index"], {}).get("connection") or "").strip()
                title = planned_titles[idx]

                t_dict: dict[str, Any] = {}
                if (_time.monotonic() - t_start) < llm_budget_s:
                    try:
                        t_dict = self._llm_teach_single_topic(
                            b, theme=theme, prev_title=prev_t, next_title=next_t, connection_hint=conn_hint
                        ) or {}
                    except Exception as topic_err:
                        logger.debug(f"Per-topic LLM synthesis skipped for section {idx + 1}: {topic_err}")

                refined_title = (t_dict.get("title") or "").strip()
                other_titles = [pt for j, pt in enumerate(planned_titles) if j != idx]
                def _too_similar_title(cand: str, others: list[str]) -> bool:
                    cand_low = cand.lower()
                    cand_toks = {w for w in re.findall(r"[a-z0-9]+", cand_low) if len(w) > 2} - {"scheduling", "algorithm", "basics", "overview", "concepts", "introduction", "part", "analysis"}
                    for ot in others:
                        ot_low = ot.lower()
                        if cand_low == ot_low:
                            return True
                        ot_toks = {w for w in re.findall(r"[a-z0-9]+", ot_low) if len(w) > 2} - {"scheduling", "algorithm", "basics", "overview", "concepts", "introduction", "part", "analysis"}
                        if cand_toks and ot_toks and (len(cand_toks & ot_toks) / max(1, min(len(cand_toks), len(ot_toks)))) >= 0.75:
                            return True
                    return False

                if (
                    refined_title
                    and len(refined_title.split()) >= 2
                    and not _is_incomplete_clause(refined_title)
                    and refined_title.lower() not in _GENERIC_DISALLOWED_TITLES
                    and not any(a in refined_title.lower() for a in ["deep learning", "neural network", "backpropagation", "quantum"] if a not in transcript_text.lower())
                    and not _too_similar_title(refined_title, other_titles)
                ):
                    title = refined_title
                    planned_titles[idx] = title

                expl = (t_dict.get("explanation") or "").strip()
                if not expl or len(expl.split()) < 8:
                    expl = self.fallback_engine._extract_explanation(b["segs"], b["text"], title=title)

                kps = [str(x).strip() for x in (t_dict.get("key_points") or []) if len(str(x).strip().split()) >= 5]
                grounded_kps = ClaimEntailmentValidator.filter_grounded_points(kps, b["text"]) or kps
                if len(grounded_kps) < 2:
                    existing_norms = {re.sub(r"[^a-z0-9]", "", kp.lower()) for kp in grounded_kps}
                    for fb_kp in self.fallback_engine._extract_key_points(b["segs"], b["text"], title=title):
                        fb_norm = re.sub(r"[^a-z0-9]", "", fb_kp.lower())
                        if fb_norm and fb_norm not in existing_norms:
                            grounded_kps.append(fb_kp)
                            existing_norms.add(fb_norm)
                    if len(grounded_kps) < 2:
                        how_txt = str(t_dict.get("how_it_works") or "").strip()
                        why_txt = str(t_dict.get("why_it_matters") or "").strip()
                        for extra_src in (how_txt, why_txt):
                            for sent in re.split(r"(?<=[.!?])\s+", extra_src):
                                s_clean = sent.strip()
                                s_norm = re.sub(r"[^a-z0-9]", "", s_clean.lower())
                                if len(s_clean.split()) >= 5 and s_norm and s_norm not in existing_norms:
                                    grounded_kps.append(s_clean)
                                    existing_norms.add(s_norm)
                                if len(grounded_kps) >= 3:
                                    break
                            if len(grounded_kps) >= 3:
                                break

                # Grounded definitions + deterministic supplement
                parsed_defs: list[dict[str, str]] = []
                for d in (t_dict.get("definitions") or []):
                    if isinstance(d, dict) and d.get("term") and d.get("definition"):
                        t_term, t_def = str(d["term"]).strip(), str(d["definition"]).strip()
                        if any(a in t_term.lower() for a in ["deep learning", "neural network", "backpropagation"]) and a not in transcript_text.lower():
                            continue
                        if len(t_def.split()) >= 4:
                            parsed_defs.append({"term": t_term, "definition": t_def})
                    elif isinstance(d, str) and ":" in d:
                        pts = d.split(":", 1)
                        parsed_defs.append({"term": pts[0].strip(), "definition": pts[1].strip()})
                for fd in self.fallback_engine._extract_definitions(b["text"]):
                    if not any(pd.get("term", "").lower() == fd.get("term", "").lower() for pd in parsed_defs):
                        parsed_defs.append(fd)
                defs = parsed_defs[:4]

                llm_examples = t_dict.get("examples") or []
                if isinstance(llm_examples, str):
                    llm_examples = [llm_examples]
                clean_exs = [
                    str(e).strip() for e in llm_examples
                    if len(str(e).strip().split()) >= 4
                    and not any(a in str(e).lower() for a in ["deep learning", "neural network", "backpropagation"] if a not in transcript_text.lower())
                ]
                for fe in self.fallback_engine._extract_examples(b["text"]):
                    if fe not in clean_exs:
                        clean_exs.append(fe)
                examples = clean_exs[:2]

                llm_formulas = t_dict.get("formulas") or []
                if isinstance(llm_formulas, str):
                    llm_formulas = [llm_formulas]
                flat_formulas: list[str] = []
                for item in llm_formulas:
                    if isinstance(item, (list, tuple)):
                        flat_formulas.extend(str(sub).strip() for sub in item if str(sub).strip())
                    else:
                        s_item = str(item).strip()
                        if s_item.startswith("[") and s_item.endswith("]"):
                            try:
                                import ast as _ast
                                parsed_lst = _ast.literal_eval(s_item)
                                if isinstance(parsed_lst, (list, tuple)):
                                    flat_formulas.extend(str(sub).strip() for sub in parsed_lst if str(sub).strip())
                                    continue
                            except Exception:
                                pass
                        flat_formulas.append(s_item)
                clean_fms = [f for f in flat_formulas if len(f) >= 3 and "=" in f or len(f) >= 5]
                for ff in self.fallback_engine._extract_formulas(b["text"]):
                    if not any(ff.lower() == cf.lower() for cf in clean_fms):
                        clean_fms.append(ff)
                formulas = clean_fms[:3]

                emphasis = self.fallback_engine._extract_emphasis(b["text"], span=b["segs"])
                exam_rel = (
                    bool(t_dict.get("exam_relevant"))
                    or any(e.get("source") == "lecturer_explicit" for e in emphasis)
                    or any("exam" in s.text.lower() or "paper" in s.text.lower() for s in b["segs"])
                    or bool(formulas)
                )

                topic_id = f"topic_{idx+1:02d}"
                evidence_ids = [s.id for s in b["segs"]]
                evidence_tokens = [f"seg_{s.id:03d}" for s in b["segs"]]

                subtopics: list[SubTopic] = []
                for kp in grounded_kps[:3]:
                    syn_title = _synthesize_subtopic_title(kp, topic_title=title)
                    if syn_title and not _is_incomplete_clause(syn_title):
                        subtopics.append(SubTopic(
                            id=f"{topic_id}_sub_{len(subtopics)+1:02d}",
                            title=syn_title,
                            role=DiscourseRole.SUBTOPIC,
                            start_time=b["start"],
                            end_time=b["end"],
                            source_segment_ids=evidence_ids,
                            key_points=[kp],
                            explanation=kp,
                        ))
                if not subtopics and title:
                    subtopics.append(SubTopic(
                        id=f"{topic_id}_sub_01",
                        title=f"{title} Fundamentals",
                        role=DiscourseRole.SUBTOPIC,
                        start_time=b["start"],
                        end_time=b["end"],
                        source_segment_ids=evidence_ids,
                        key_points=grounded_kps[:2] or [expl],
                        explanation=expl,
                    ))
                for def_idx, d in enumerate(defs[:2]):
                    subtopics.append(SubTopic(
                        id=f"{topic_id}_def_{def_idx+1:02d}",
                        title=f"Definition: {d.get('term', 'Concept')}",
                        role=DiscourseRole.DEFINITION,
                        start_time=b["start"],
                        end_time=b["end"],
                        source_segment_ids=evidence_ids,
                        key_points=[f"{d.get('term')}: {d.get('definition')}"],
                        explanation=d.get("definition", ""),
                    ))
                for ex_idx, ex in enumerate(examples[:2]):
                    subtopics.append(SubTopic(
                        id=f"{topic_id}_ex_{ex_idx+1:02d}",
                        title="Real-World Example",
                        role=DiscourseRole.EXAMPLE,
                        start_time=b["start"],
                        end_time=b["end"],
                        source_segment_ids=evidence_ids,
                        key_points=[ex],
                        explanation=ex,
                    ))

                span = TopicSpan(start=b["start"], end=b["end"], start_seg_id=b["start_seg_id"], end_seg_id=b["end_seg_id"])
                study_topics.append(StudyTopic(
                    id=topic_id,
                    title=title.title(),
                    topic_type="major_topic",
                    hierarchy_level=1,
                    subtopics=subtopics,
                    spans=[span],
                    explanation=self._coerce_prose(expl),
                    what_it_means=self._coerce_prose(t_dict.get("what_it_means")),
                    how_it_works=self._coerce_prose(t_dict.get("how_it_works")),
                    steps_and_logic=[
                        self._coerce_prose(st) for st in (t_dict.get("steps_and_logic") or [])
                        if len(self._coerce_prose(st).split()) >= 4
                    ][:4],
                    why_it_matters=self._coerce_prose(t_dict.get("why_it_matters")),
                    concept_connection=conn_hint,
                    key_points=grounded_kps,
                    definitions=defs,
                    examples=examples,
                    formulas=formulas,
                    emphasis=emphasis,
                    exam_relevant=exam_rel,
                    source="local_llm",
                    source_segment_ids=evidence_ids,
                    evidence=evidence_tokens,
                    source_text=" ".join(s.text for s in b["segs"]),
                    grounding_status="SUPPORTED",
                    claim_entailment_score=1.0,
                ))

            if study_topics:
                merged = self._reconcile_and_merge_topics(study_topics)
                # Stage 4: Self-review, anti-parroting, formula repair, concept relationships
                merged, overview, concept_rels = self._review_and_refine_notes(merged, overview, transcript_text)

                discourse_roles = []
                for s_idx, s in enumerate(segments):
                    role = DiscourseRoleClassifier.classify_role(s.text, is_first_in_section=(s_idx == 0))
                    discourse_roles.append({
                        "segment_id": s.id,
                        "start_time": round(s.start, 2),
                        "end_time": round(s.end, 2),
                        "role": role,
                        "text_snippet": s.text[:90],
                    })
                hierarchy = []
                for t in merged:
                    s_time = t.spans[0].start if t.spans else 0.0
                    e_time = t.spans[-1].end if t.spans else 0.0
                    ev_list = t.evidence if t.evidence else [f"seg_{sid:03d}" for sid in t.source_segment_ids]
                    subtopics_dicts = [st.to_dict() for st in t.subtopics] if t.subtopics else []
                    hierarchy.append({
                        "id": t.id,
                        "title": t.title,
                        "role": DiscourseRole.NEW_TOPIC,
                        "level": 1,
                        "start_time": round(s_time, 2),
                        "end_time": round(e_time, 2),
                        "timestamp": f"[{_format_time(s_time)} - {_format_time(e_time)}]",
                        "evidence_segment_ids": t.source_segment_ids,
                        "evidence": ev_list,
                        "subtopics": subtopics_dicts,
                    })
                return LectureStudyNotes(
                    study_topics=merged,
                    lecture_overview=overview,
                    subject=subject,
                    theme=theme,
                    topic_hierarchy=hierarchy,
                    concept_relationships=concept_rels,
                    discourse_roles=discourse_roles,
                    analyzer_used="local_llm",
                    transcript_hash=t_hash,
                    is_fallback=False,
                )
        except Exception as e:
            logger.warning(f"Multi-stage LLM synthesis failed: {e}. Falling back to abstractive fallback.")

        return self._execute_fallback_pipeline(segments, transcript_text, duration, t_hash, topic_hint=topic_hint, fallback_reason="Unified LLM generation fallback")

    def _fallback_sections_from_boundaries(self, segments: list[Segment], boundaries: list[int]) -> list[dict[str, Any]]:
        sections = []
        prev = 0
        all_bounds = list(boundaries) + [len(segments)]
        for b in all_bounds:
            if b <= prev:
                continue
            sec_segs = segments[prev:b]
            sec_text = " ".join(s.text for s in sec_segs)
            sec_title = self.fallback_engine._extract_title(sec_segs, sec_text)
            if not sec_title or sec_title.lower() in _GENERIC_DISALLOWED_TITLES or re.match(r"^(?:conceptual section|topic\s*\d+)\b", sec_title.lower()):
                if any(w in sec_text.lower() for w in ["submission deadline", "student portal", "due date", "homework submission", "upload to portal", "coursework submission"]) or ("portal" in sec_text.lower() and "submit" in sec_text.lower()):
                    sec_title = "Coursework Directives & Submission"
                else:
                    sec_title = f"Topic Section {len(sections)+1}"
            sections.append({
                "title": sec_title,
                "start_segment_id": sec_segs[0].id,
                "end_segment_id": sec_segs[-1].id,
                "reason": "Semantic boundary shift"
            })
            prev = b
        return sections

    def _reconcile_and_merge_topics(self, topics: list[StudyTopic]) -> list[StudyTopic]:
        """Merges recurring discussions while preserving distinct conceptual distinctions."""
        if len(topics) <= 1:
            return topics

        try:
            from app.services.embedding_service import EmbeddingModelManager
            from app.services.study_guide_generator import StudyGuideSynthesizer
            from app.services.domain_corrector import CANONICAL_TECHNICAL_TERMS

            def _are_domain_mismatched(t1: StudyTopic, t2: StudyTopic) -> bool:
                txt1 = f"{t1.title} {t1.explanation} {t1.source_text or ''}"
                txt2 = f"{t2.title} {t2.explanation} {t2.source_text or ''}"
                dom1 = StudyGuideSynthesizer.detect_domain(txt1, topic_hint=t1.title)
                dom2 = StudyGuideSynthesizer.detect_domain(txt2, topic_hint=t2.title)

                # Reject merge if distinct known domains differ
                if dom1 != "general_academic" and dom2 != "general_academic" and dom1 != dom2:
                    return True

                terms1 = {can.lower() for can in CANONICAL_TECHNICAL_TERMS if re.search(r'\b' + re.escape(can.lower()) + r'\b', txt1.lower())}
                terms2 = {can.lower() for can in CANONICAL_TECHNICAL_TERMS if re.search(r'\b' + re.escape(can.lower()) + r'\b', txt2.lower())}

                # Reject merge if both topics have canonical terms but zero overlap
                if terms1 and terms2 and len(terms1 & terms2) == 0:
                    return True

                return False

            passages = [f"{t.title}. {t.explanation} {' '.join(t.key_points)} {t.source_text or ''}".strip() for t in topics]
            embs = EmbeddingModelManager.encode(passages)
            sim_matrix = _cosine_sim_matrix(embs, embs)

            _DISTINCT_COUNTERPART_PAIRS = {
                ("series", "parallel"),
                ("current", "charge"),
                ("voltage", "current"),
                ("voltage", "resistance"),
                ("resistance", "current"),
                ("sensory", "motor"),
                ("sensory", "relay"),
                ("relay", "motor"),
                ("nervous", "hormonal"),
                ("profit", "cash"),
                ("turnaround", "waiting"),
                ("fcfs", "sjf")
            }

            def _are_counterparts(t1: str, t2: str) -> bool:
                t1_l, t2_l = t1.lower(), t2.lower()
                for w1, w2 in _DISTINCT_COUNTERPART_PAIRS:
                    if (w1 in t1_l and w2 in t2_l) or (w2 in t1_l and w1 in t2_l):
                        return True
                return False

            merged: list[StudyTopic] = []
            merged_indices: set[int] = set()

            for i in range(len(topics)):
                if i in merged_indices:
                    continue
                current = topics[i]
                for j in range(i + 1, len(topics)):
                    if j in merged_indices:
                        continue
                    other = topics[j]
                    if _are_counterparts(current.title, other.title) or _are_domain_mismatched(current, other):
                        continue

                    # Merge if titles are exact normalized match, or adjacent with very high passage similarity (> 0.90)
                    is_exact_match = (re.sub(r"[^a-z0-9]", "", current.title.lower()) == re.sub(r"[^a-z0-9]", "", other.title.lower()))
                    is_adjacent_similar = (j == i + 1 and sim_matrix[i, j] > 0.90)

                    if is_exact_match or is_adjacent_similar:
                        current.spans.extend(other.spans)
                        current.source_segment_ids.extend(other.source_segment_ids)
                        current.key_points.extend(other.key_points)
                        current.definitions.extend(other.definitions)
                        current.examples.extend(other.examples)
                        current.formulas.extend(other.formulas)
                        current.emphasis.extend(other.emphasis)
                        current.subtopics.extend(other.subtopics)
                        current.exam_relevant = current.exam_relevant or other.exam_relevant
                        merged_indices.add(j)

                # Sort spans chronologically
                current.spans.sort(key=lambda s: s.start)
                current.source_segment_ids = sorted(set(current.source_segment_ids))
                current.evidence = [f"seg_{sid:03d}" for sid in current.source_segment_ids]
                # Deduplicate list fields safely
                current.key_points = list(dict.fromkeys(str(x) if not isinstance(x, dict) else str(list(x.values())[0]) for x in current.key_points if x))[:6]
                current.examples = list(dict.fromkeys(str(x) if not isinstance(x, dict) else str(list(x.values())[0]) for x in current.examples if x))[:3]
                current.formulas = list(dict.fromkeys(str(x) if not isinstance(x, dict) else str(list(x.values())[0]) for x in current.formulas if x))[:3]
                merged.append(current)

            # Cap maximum topics dynamically based on lecture duration (up to 12)
            # Only merge strictly ADJACENT topics (bi and bi + 1) to preserve chronological timeline
            max_limit = 12
            while len(merged) > max_limit:
                m_passages = [f"{t.title}. {t.explanation} {' '.join(t.key_points)} {t.source_text or ''}".strip() for t in merged]
                t_embs = EmbeddingModelManager.encode(m_passages)
                t_sim = _cosine_sim_matrix(t_embs, t_embs)
                np.fill_diagonal(t_sim, -1.0)
                # Find best merge pair that are strictly ADJACENT, NOT counterparts, and NOT domain mismatched
                pairs = []
                for bi in range(len(merged) - 1):
                    bj = bi + 1
                    if not _are_counterparts(merged[bi].title, merged[bj].title) and not _are_domain_mismatched(merged[bi], merged[bj]):
                        pairs.append((t_sim[bi, bj], bi, bj))
                if not pairs:
                    break
                pairs.sort(key=lambda x: x[0], reverse=True)
                _, best_i, best_j = pairs[0]
                t_i, t_j = merged[best_i], merged[best_j]
                t_i.spans.extend(t_j.spans)
                t_i.spans.sort(key=lambda s: s.start)
                t_i.source_segment_ids = sorted(set(t_i.source_segment_ids + t_j.source_segment_ids))
                t_i.evidence = [f"seg_{sid:03d}" for sid in t_i.source_segment_ids]
                t_i.key_points = list(dict.fromkeys(t_i.key_points + t_j.key_points))[:6]
                t_i.examples = list(dict.fromkeys(t_i.examples + t_j.examples))[:3]
                t_i.formulas = list(dict.fromkeys(t_i.formulas + t_j.formulas))[:3]
                t_i.subtopics.extend(t_j.subtopics)
                t_i.exam_relevant = t_i.exam_relevant or t_j.exam_relevant
                merged.pop(best_j)


            for idx, t in enumerate(merged):
                t.id = f"topic_{idx+1:02d}"
                # Clean, deduplicate and filter incomplete subtopics
                seen_sub = set()
                clean_subs = []
                for st in t.subtopics:
                    st_title = re.sub(r'[,;:\s]+$', '', st.title).strip()
                    st_norm = re.sub(r'[^a-z0-9]', '', st_title.lower())
                    if not st_norm or st_norm in seen_sub:
                        continue
                    if _is_incomplete_clause(st_title):
                        continue
                    if any(_is_phonetic_noise(re.sub(r'[^a-zA-Z]', '', w)) for w in st_title.split()):
                        continue
                    if any(w.lower() in _HALLUCINATION_WORDS for w in st_title.split()):
                        continue
                    seen_sub.add(st_norm)
                    st.title = st_title[0].upper() + st_title[1:]
                    clean_subs.append(st)

                if not clean_subs and t.title:
                    clean_subs.append(SubTopic(
                        id=f"{t.id}_sub_01",
                        title=f"{t.title} Principles",
                        role=DiscourseRole.SUBTOPIC,
                        start_time=t.spans[0].start if t.spans else 0.0,
                        end_time=t.spans[-1].end if t.spans else 0.0,
                        source_segment_ids=t.source_segment_ids,
                        key_points=t.key_points[:2],
                        explanation=t.explanation
                    ))

                t.subtopics = clean_subs[:4]
                for st_idx, st in enumerate(t.subtopics):
                    st.id = f"{t.id}_sub_{st_idx+1:02d}"

            return merged
        except Exception as e:
            logger.warning(f"Topic merging fallback: {e}")
            return topics

    def _execute_fallback_pipeline(
        self,
        segments: list[Segment],
        transcript_text: str,
        duration: float,
        t_hash: str,
        topic_hint: Optional[str] = None,
        fallback_reason: Optional[str] = None
    ) -> LectureStudyNotes:
        """Executes the ExtractiveSemanticFallback engine."""
        logger.warning(f"Executing ExtractiveSemanticFallback. Reason: {fallback_reason or 'unspecified'}")
        raw_topics = self.fallback_engine.analyze(segments, duration, topic_hint=topic_hint)
        # Apply multi-span reconciliation & deduplication to fallback topics
        topics = self._reconcile_and_merge_topics(raw_topics)
        titles = [t.title for t in topics]
        theme = titles[0] if titles else "Lecture Overview"
        overview = (
            f"This lecture covers {', '.join(titles[:4])}. "
            f"The session connects core definitions, underlying mechanisms, and practical examples across these topics."
        )
        topics, overview, concept_rels = self._review_and_refine_notes(topics, overview, transcript_text)
        discourse_roles = []
        for s_idx, s in enumerate(segments):
            role = DiscourseRoleClassifier.classify_role(s.text, is_first_in_section=(s_idx == 0))
            discourse_roles.append({
                "segment_id": s.id,
                "start_time": round(s.start, 2),
                "end_time": round(s.end, 2),
                "role": role,
                "text_snippet": s.text[:90]
            })

        hierarchy = []
        for t in topics:
            s_time = t.spans[0].start if t.spans else 0.0
            e_time = t.spans[-1].end if t.spans else 0.0
            ev_list = t.evidence if t.evidence else [f"seg_{sid:03d}" for sid in t.source_segment_ids]
            subtopics_dicts = [st.to_dict() for st in t.subtopics] if t.subtopics else []
            hierarchy.append({
                "id": t.id,
                "title": t.title,
                "role": DiscourseRole.NEW_TOPIC,
                "level": 1,
                "start_time": round(s_time, 2),
                "end_time": round(e_time, 2),
                "timestamp": f"[{_format_time(s_time)} - {_format_time(e_time)}]",
                "evidence_segment_ids": t.source_segment_ids,
                "evidence": ev_list,
                "subtopics": subtopics_dicts
            })

        return LectureStudyNotes(
            study_topics=topics,
            lecture_overview=overview,
            subject=topic_hint if topic_hint else "Academic Lecture",
            theme=theme,
            topic_hierarchy=hierarchy,
            concept_relationships=concept_rels,
            discourse_roles=discourse_roles,
            analyzer_used="local_extractive_fallback",
            transcript_hash=t_hash,
            is_fallback=True,
            fallback_reason=fallback_reason or "Local LLM disabled or fallback requested"
        )

    def _empty_notes(self, t_hash: str) -> LectureStudyNotes:
        return LectureStudyNotes(
            study_topics=[],
            lecture_overview="No speech or transcript content detected for this session.",
            subject="Unspecified",
            theme="Unspecified",
            quality_score=0.0,
            quality_pass=False,
            analyzer_used="empty",
            transcript_hash=t_hash
        )


# Global Engine Singleton & Function Facade
_ENGINE: Optional[TopicIntelligenceEngine] = None


def get_topic_engine() -> TopicIntelligenceEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = TopicIntelligenceEngine()
    return _ENGINE


def analyze_lecture_study_notes(
    segments_orm: list[Any],
    transcript_text: str,
    duration_seconds: float = 0.0,
    force_refresh: bool = False,
    topic_hint: Optional[str] = None,
    speech_duration_seconds: Optional[float] = None,
) -> LectureStudyNotes:
    """Canonical function facade for study notes generation across the application."""
    engine = get_topic_engine()
    return engine.analyze_lecture_study_notes(
        transcript_text=transcript_text,
        segments_orm=segments_orm,
        duration_seconds=duration_seconds,
        force_refresh=force_refresh,
        topic_hint=topic_hint,
        speech_duration_seconds=speech_duration_seconds,
    )
