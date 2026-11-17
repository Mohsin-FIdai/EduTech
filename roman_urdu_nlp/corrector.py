"""
corrector.py — Spell & Grammar Correction module for roman_urdu_nlp.

Contract:
    Function : correct(text: str) -> str
    Input    : Normalized string (output of normalizer.normalize()).
               Must be str; raises TypeError otherwise.
    Output   : Corrected string with:
               - Residual spelling errors fixed via edit-distance lookup
                 against a canonical Roman Urdu word list (max edit dist = 1)
               - Common word-order / agreement patterns fixed via rule-based
                 substitution (e.g. "hun hoon" dedup, double auxiliary removal)
               - Unknown words passed through unchanged
               Empty string → ""

Mocking note: correct() receives a *normalized* string.
Mock: normalize() = identity (pass pre-normalized strings directly in tests).

Examples:
    >>> correct("mujhe bohat bhook lagi hai")
    'mujhe bohat bhook lagi hai'

    >>> correct("mujhe bohat bhuk lagi hai")   # "bhuk" close to "bhook"
    'mujhe bohat bhook lagi hai'

    >>> correct("")
    ''
"""

from __future__ import annotations
import editdistance
from roman_urdu_nlp.detector import _ENGLISH_WORDS, _TECHNICAL_WORDS

# ---------------------------------------------------------------------------
# Canonical Roman Urdu word list (words the normalizer would output)
# Used for edit-distance spell correction of residual errors.
# ---------------------------------------------------------------------------

_CANONICAL_WORDS: frozenset[str] = frozenset({
    # Pronouns
    "mujhe", "mera", "meri", "mere", "tumhara", "tumhari", "tumhare",
    "aapka", "aapki", "aapke", "unka", "unki", "unke", "uska", "uski",
    "hum", "tum", "aap", "woh", "yeh", "main", "wo",
    # Copula / auxiliaries
    "hai", "hain", "tha", "thi", "the", "hoga", "hogi", "honge",
    "hoon", "ho", "raha", "rahi", "rahe", "gaya", "gayi", "gaye",
    "aaya", "aayi", "aaye", "jana", "aana",
    # Negation
    "nahi", "na", "mat",
    # Intensifiers & adjectives
    "bohat", "zyada", "acha", "achi", "ache", "theek", "bura", "buri",
    "bure", "mushkil", "asaan", "naya", "nayi", "purana",
    "khush", "udaas", "thaka", "thaki", "thake",
    # Nouns – common
    "bhook", "neend", "garmi", "thandi", "barish", "dhoop",
    "khana", "khaana", "chai", "pani", "doodh",
    "ghar", "school", "office", "bazar", "masjid", "hospital",
    "dost", "yaar", "bhai", "behen", "ammi", "abbu", "beta", "beti",
    "baat", "baatein", "waqt", "zindagi", "sawal", "jawab",
    "dil", "aankhein", "haath", "paon",
    "din", "raat", "subah", "shaam", "dopahar",
    "kal", "aaj", "abhi", "phir", "jaldi",
    "safar", "shehar", "gaon", "gali",
    "kaam", "kitaab", "qalam", "kagaz",
    "paisa", "rupay", "kharidna",
    # Verbs
    "karna", "karo", "karta", "karti", "karte",
    "jana", "jao", "jata", "jati", "jate",
    "aana", "aao", "aata", "aati", "aate",
    "parhna", "parha", "parhi", "parhe",
    "likhna", "likha", "likhi", "likhe",
    "bolna", "bola", "boli", "bole",
    "sunna", "suna", "suni", "sune",
    "dekhna", "dekha", "dekhi", "dekhe",
    "milna", "mila", "mili", "mile",
    "khelna", "khela", "kheli", "khele",
    "samajh", "samajhna",
    "lagi", "laga", "lage",
    "rona", "roya", "royi",
    "hona", "uthna", "bethna",
    # Connectors / prepositions
    "aur", "lekin", "magar", "agar", "to", "phir",
    "ke", "ki", "ka", "se", "ko", "mein", "par", "tak",
    "wala", "wali", "wale", "ne",
    "koi", "kuch", "sab", "pata", "bhi", "sirf",
    # Discourse markers
    "yaar", "bhai", "waise", "matlab", "basically",
    "haha", "uff", "wah", "are",
    "shukriya", "please", "sorry",
    # Greetings / religious
    "salam", "ameen",
    # Common code-switched English (pass through)
    "exam", "phone", "internet", "app", "movie", "game",
    "project", "deadline", "meeting", "office",
    "download", "upload", "online", "battery",
    "karachi", "lahore", "islamabad", "pakistan",
    "traffic", "pizza", "biryani",
    # Academic & Computer Science / Engineering Terms
    "topic", "machine", "learning", "artificial", "intelligence",
    "neural", "network", "networks", "deep", "algorithm", "algorithms",
    "dataset", "datasets", "regression", "classification", "supervised",
    "unsupervised", "model", "models", "train", "training", "test",
    "testing", "validation", "assignment", "assignments", "semester",
    "presentation", "quiz", "quizzes", "midterm", "midterms", "attendance",
    "lecture", "lectures", "slides", "slide", "professor", "teacher",
    "student", "students", "gpa", "cgpa", "degree", "thesis", "research",
    "python", "code", "coding", "database", "api", "server", "system",
    # Additional Conversational Roman Urdu (from real speech corpus)
    "bohot", "bilkul", "zaroor", "waghera", "taqreeban", "waqai", "shuru",
    "khatam", "koshish", "hamesha", "pehle", "baad", "sahi", "shayad",
    "duniya", "waqt", "zindagi", "chahiye", "samajh", "dekhna", "karne",
    "karna", "karte", "karti", "karta", "karein", "batayein", "batana",
    "bolna", "sunna", "survivability", "podcast", "clips", "basis", "idea", "support",
    "sales", "channel", "pause", "experience", "rate", "cache", "role", "form",
})

# ---------------------------------------------------------------------------
# Grammar fix patterns: (bad_phrase, corrected_phrase)
# Applied as ordered string replacements on the full text.
# ---------------------------------------------------------------------------

_GRAMMAR_FIXES: list[tuple[str, str]] = [
    # Remove double auxiliary (normalizer may produce these for "hun hoon")
    ("hoon hoon", "hoon"),
    ("hai hai", "hai"),
    ("raha raha", "raha"),
    ("rahi rahi", "rahi"),
    # Fix "phir se" ordering issues
    ("se phir", "phir se"),
    # Common agreement: "gaya hai" is correct, not "gaya hain"
    ("gaya hain", "gaya hai"),
    ("gayi hain", "gayi hai"),
    # "lagi hai" (correct) vs "lagi hain"
    ("lagi hain", "lagi hai"),
]

# ---------------------------------------------------------------------------
# Maximum edit distance for spell correction
# ---------------------------------------------------------------------------
_MAX_EDIT_DIST = 1


def _spell_correct_token(token: str) -> str:
    """
    Return the best canonical spelling for ``token`` if within edit distance 1.
    Returns token unchanged if it is already canonical, an English word, or no close match found.
    """
    lower = token.lower()
    # Skip very short tokens to avoid over-correction ('aa', 'ko', 'se' etc.)
    if len(lower) < 3:
        return token
    if lower in _CANONICAL_WORDS or lower in _ENGLISH_WORDS or lower in _TECHNICAL_WORDS:
        return token  # already valid canonical or English loanword, preserve as-is

    best: str | None = None
    best_dist = _MAX_EDIT_DIST + 1

    for word in _CANONICAL_WORDS:
        dist = editdistance.eval(lower, word)
        if dist < best_dist:
            best_dist = dist
            best = word

    if best is not None and best_dist <= _MAX_EDIT_DIST:
        # Preserve capitalisation of original token
        if token[0].isupper() and best:
            return best[0].upper() + best[1:]
        return best

    return token  # no close match — return as-is


def correct(text: str) -> str:
    """
    Correct residual spelling errors and apply grammar fix rules to a
    normalized Roman Urdu string.

    Parameters
    ----------
    text : str
        Normalized input string (output of normalizer.normalize()).

    Returns
    -------
    str
        Corrected string. Returns "" for empty input.

    Raises
    ------
    TypeError
        If ``text`` is not a ``str``.
    """
    if not isinstance(text, str):
        raise TypeError(f"correct() expects str, got {type(text).__name__!r}")

    text = text.strip()
    if not text:
        return ""

    # 1. Token-level spell correction
    tokens = text.split()
    corrected_tokens = [_spell_correct_token(tok) for tok in tokens]
    text = " ".join(corrected_tokens)

    # 2. Phrase-level grammar fixes
    for bad, good in _GRAMMAR_FIXES:
        text = text.replace(bad, good)

    return text
