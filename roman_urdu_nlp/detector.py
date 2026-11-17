"""
detector.py — Language Detection module for roman_urdu_nlp.

Contract:
    Function : detect_segments(text: str) -> list[tuple[str, str]]
    Input    : Cleaned string (output of cleaner.clean()).
               Must be str; raises TypeError otherwise.
    Output   : List of (segment, lang) tuples where:
               - segment is a contiguous substring (one or more words)
               - lang ∈ {"roman_urdu", "english"}
               Consecutive tokens with the same language label are merged
               into a single segment.
               Empty string → []

Strategy:
    - Tokenize on whitespace.
    - Each token is classified by checking membership in a curated English
      word list (~500 common words).  Words in the Roman Urdu keyword list
      are always tagged "roman_urdu" regardless of overlap.
    - Unknown/ambiguous tokens default to "roman_urdu" (pipeline bias).
    - Code-mixed sentences produce alternating segments, e.g.:
        "meeting kal hai" → [("meeting", "english"), ("kal hai", "roman_urdu")]

Examples:
    >>> detect_segments("mujhe bohat bhook lagi hai")
    [('mujhe bohat bhook lagi hai', 'roman_urdu')]

    >>> detect_segments("meeting kal hai office mein")
    [('meeting', 'english'), ('kal hai', 'roman_urdu'), ('office', 'english'), ('mein', 'roman_urdu')]

    >>> detect_segments("")
    []
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Word lists (curated)
# ---------------------------------------------------------------------------

# High-frequency English words that are NOT Roman Urdu
_ENGLISH_WORDS: frozenset[str] = frozenset({
    # Articles / determiners
    "a", "an", "the", "this", "that", "these", "those", "my", "your",
    "his", "her", "its", "our", "their", "some", "any", "all", "both",
    "each", "every", "few", "more", "most", "much", "no", "other",
    "same", "such", "which", "what", "whose",
    # Pronouns
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "us",
    "them", "who", "whom",
    # Common verbs
    "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "shall", "should", "may", "might", "must", "can", "could",
    "get", "got", "make", "made", "go", "went", "come", "came",
    "see", "saw", "know", "knew", "think", "thought", "take", "took",
    "look", "use", "find", "give", "tell", "work", "call", "try",
    "ask", "need", "feel", "become", "leave", "put", "mean", "keep",
    "let", "begin", "show", "hear", "play", "run", "move", "live",
    "believe", "hold", "bring", "happen", "write", "provide", "stand",
    "lose", "pay", "meet", "include", "continue", "set", "learn",
    "change", "lead", "understand", "watch", "follow", "stop", "create",
    "speak", "read", "spend", "grow", "open", "walk", "win", "offer",
    # Common nouns (tech / everyday)
    "game", "time", "year", "people", "way", "day", "man", "woman",
    "child", "world", "life", "hand", "part", "place", "case", "week",
    "company", "system", "program", "question", "work", "government",
    "number", "night", "point", "home", "water", "room", "mother",
    "area", "money", "story", "fact", "month", "lot", "right", "study",
    "book", "eye", "job", "word", "business", "issue", "side",
    "kind", "head", "house", "service", "friend", "father", "power",
    "hour", "game", "line", "end", "city", "community", "name",
    "second", "street", "road", "plan", "class", "order", "face",
    "food", "back", "history", "family", "event", "fire", "account",
    "car", "dog", "school", "field", "group", "body", "love", "country",
    # Tech-specific (common in Roman Urdu code-switch)
    "phone", "internet", "app", "website", "email", "download", "upload",
    "online", "offline", "battery", "charger", "screen", "update",
    "install", "password", "data", "video", "audio", "photo", "camera",
    "wifi", "bluetooth", "device", "laptop", "computer", "keyboard",
    "mouse", "network", "server", "cloud", "link", "page", "post",
    "chat", "message", "call", "text", "send", "receive", "share",
    "search", "browse", "click", "type", "login", "signup",
    "account", "profile", "settings", "menu", "button", "icon",
    "game", "level", "score", "player", "team", "match",
    # Adjectives / adverbs
    "new", "good", "high", "old", "great", "big", "american", "small",
    "large", "national", "young", "different", "long", "little", "own",
    "right", "early", "able", "best", "free", "real", "serious",
    "important", "possible", "true", "whole", "bad", "always",
    "never", "often", "sometimes", "usually", "already", "still",
    "again", "also", "just", "very", "too", "now", "only", "even",
    "here", "there", "where", "when", "how", "why", "then", "so",
    "up", "out", "about", "into", "over", "after", "above", "below",
    "through", "during", "before", "after", "between", "under",
    "around", "next", "late", "hard", "long", "little", "own",
    "last", "near", "far", "fast", "slow", "sure", "clear", "open",
    "close", "full", "happy", "funny", "finally", "seriously",
    # Prepositions / conjunctions
    "of", "in", "to", "for", "on", "with", "at", "by", "from",
    "as", "or", "and", "but", "if", "not", "than", "because",
    "while", "although", "though", "until", "unless", "since",
    "without", "within", "along", "across", "behind", "beyond",
    "except", "per", "plus", "via", "nor", "yet", "whether",
    # Greetings / interjections common in English
    "ok", "okay", "yes", "no", "hello", "hi", "bye", "oh", "wow",
    "omg", "lol", "haha", "thanks", "please", "sorry", "excuse",
    "wait", "stop", "hey", "man", "dear", "sir",
    # Common domains in Pakistani code-mixing
    "meeting", "office", "deadline", "project", "submit", "prepare",
    "presentation", "exam", "result", "class", "teacher", "student",
    "movie", "film", "song", "music", "channel", "show", "serial",
    "market", "shop", "price", "order", "delivery", "payment", "tax",
    "ticket", "hotel", "trip", "travel", "visa", "passport", "flight",
    "pizza", "burger", "coffee", "tea", "biryani",
    # Academic & Computer Science / IT / Engineering Terminology
    "topic", "machine", "learning", "artificial", "intelligence", "neural",
    "network", "networks", "deep", "algorithm", "algorithms", "dataset",
    "datasets", "regression", "classification", "supervised", "unsupervised",
    "reinforcement", "clustering", "gradient", "descent", "accuracy",
    "loss", "model", "models", "train", "training", "test", "testing",
    "validation", "epoch", "epochs", "batch", "layer", "layers",
    "activation", "optimizer", "backpropagation", "transformer",
    "transformers", "attention", "convolutional", "recurrent", "feature",
    "features", "weights", "bias", "tensor", "matrix", "vector",
    "dimension", "embedding", "embeddings", "token", "tokens",
    "tokenizer", "prompt", "inference", "pipeline", "framework",
    "library", "python", "code", "coding", "debug", "compiler",
    "function", "variable", "database", "query", "sql", "api",
    "backend", "frontend", "server", "cloud", "deployment",
    "assignment", "assignments", "semester", "quiz", "quizzes",
    "midterm", "midterms", "finals", "attendance", "lecture",
    "lectures", "slides", "slide", "syllabus", "credit", "hours",
    "gpa", "cgpa", "grade", "grades", "marks", "degree", "thesis",
    "research", "paper", "faculty", "professor", "department",
    "campus", "university", "institute", "lab", "laboratory",
    # Additional CS & code-switched vocabulary from real speech
    "rate", "cache", "role", "form", "basis", "idea", "support",
    "podcast", "clips", "experience", "sales", "channel", "survivability",
    "pause", "lock", "process", "thread", "concurrency", "deadlock",
    "semaphore", "mutex", "scheduling", "operating", "runtime",
})

# Specialized technical words subset
_TECHNICAL_WORDS: frozenset[str] = frozenset({
    "rate", "cache", "role", "form", "basis", "idea", "support",
    "lock", "process", "thread", "concurrency", "deadlock", "semaphore",
    "mutex", "scheduling", "operating",
    "machine", "learning", "artificial", "intelligence", "neural", "networks", "network",
    "deep", "algorithm", "algorithms", "dataset", "datasets", "regression", "classification",
    "supervised", "unsupervised", "clustering", "gradient", "descent", "accuracy", "loss",
    "model", "models", "training", "testing", "validation", "epoch", "epochs", "batch",
    "layer", "layers", "activation", "optimizer", "backpropagation", "transformer",
    "transformers", "attention", "convolutional", "recurrent", "feature", "features",
    "tensor", "vector", "embedding", "embeddings", "token", "tokens", "tokenizer",
    "inference", "pipeline", "framework", "database", "api", "backend", "frontend",
    "server", "topic", "lecture", "slides", "assignment", "deadline", "semester",
    "quiz", "midterm", "attendance", "gpa", "thesis", "research"
})

# Roman Urdu marker words — always classified as roman_urdu
# (overrides the English list for words that appear in both)
_ROMAN_URDU_MARKERS: frozenset[str] = frozenset({
    "hai", "hain", "tha", "thi", "the", "hoga", "hogi", "honge", "ho",
    "kya", "nahi", "nhi", "aur", "bhi", "sirf", "bohat", "boht",
    "bahut", "yaar", "yar", "bhai", "kal", "aaj", "abhi", "phir",
    "fir", "kyun", "kyu", "mujhe", "mjy", "mje", "muje", "tum",
    "aap", "woh", "ye", "yeh", "is", "us", "unka", "apna",
    "gaya", "aya", "aaya", "raha", "rahi", "rahe", "kar", "karo",
    "karna", "karte", "karta", "lagi", "laga", "lage", "ho",
    "main", "mein", "se", "ko", "ki", "ka", "ke",
    "pe", "par", "tak", "lekin", "magar", "agar", "to", "toh",
    "wala", "wali", "wale", "ne", "koi", "kuch", "sab", "pata",
    "theek", "thik", "acha", "accha", "achha", "ache",
    "mushkil", "asaan", "khana", "peena", "neend", "bhook",
    "garmi", "thandi", "barish", "ghar", "bahar", "andar",
    "dil", "yaar", "dost", "bhai", "ammi", "abbu", "beta",
    "jana", "aana", "milna", "khelna", "dekhna", "sunna",
    "parha", "parhe", "parhna", "likhna", "bolna",
    "Allah", "Ameen", "ameen", "InshAllah", "SubhanAllah",
    "hahaha", "haha", "ufff", "uff", "waah", "arrey",
    "tension", "khushi", "dukh", "gussa", "darr",
    "jaldi", "dheere", "subah", "raat", "dopahar", "shaam",
    "safar", "shahar", "gali", "mohalla", "dukaan",
    "shukriya", "shukria", "mehrbani",
})


def _classify_token(token: str) -> str:
    """Return 'english' or 'roman_urdu' for a single token."""
    lower = token.lower()
    if lower in _ROMAN_URDU_MARKERS:
        return "roman_urdu"
    if lower in _ENGLISH_WORDS:
        return "english"
    return "roman_urdu"  # default bias


def tag_tokens(text: str) -> list[dict]:
    """
    Perform fine-grained token-level language and technical term tagging.

    Returns list of dicts:
    [
        {"token": str, "lang": "ur" | "en" | "num" | "punct", "confidence": float, "is_technical": bool}
    ]
    """
    if not isinstance(text, str) or not text.strip():
        return []

    tokens = text.split()
    results = []

    for tok in tokens:
        clean_tok = tok.strip(".,!?;:\"'()[]{}")
        lower = clean_tok.lower()

        if not clean_tok:
            results.append({"token": tok, "lang": "punct", "confidence": 1.0, "is_technical": False})
            continue

        if clean_tok.isdigit():
            results.append({"token": tok, "lang": "num", "confidence": 1.0, "is_technical": False})
            continue

        is_tech = lower in _TECHNICAL_WORDS
        if lower in _ROMAN_URDU_MARKERS:
            lang = "ur"
            conf = 0.98
        elif lower in _ENGLISH_WORDS:
            lang = "en"
            conf = 0.96 if is_tech else 0.92
        else:
            lang = "ur"
            conf = 0.75

        results.append({
            "token": tok,
            "lang": lang,
            "confidence": conf,
            "is_technical": is_tech
        })

    return results


def detect_segments(text: str) -> list[tuple[str, str]]:
    """
    Detect language segments in a cleaned Roman Urdu string.

    Parameters
    ----------
    text : str
        Cleaned input string (output of cleaner.clean()).

    Returns
    -------
    list[tuple[str, str]]
        A list of (segment_text, language_tag) pairs.
        language_tag ∈ {"roman_urdu", "english"}.
        Returns [] for empty input.

    Raises
    ------
    TypeError
        If ``text`` is not a ``str``.
    """
    if not isinstance(text, str):
        raise TypeError(f"detect_segments() expects str, got {type(text).__name__!r}")

    text = text.strip()
    if not text:
        return []

    tokens = text.split()
    labelled = [(tok, _classify_token(tok)) for tok in tokens]

    segments: list[tuple[str, str]] = []
    current_words: list[str] = [labelled[0][0]]
    current_lang: str = labelled[0][1]

    for tok, lang in labelled[1:]:
        if lang == current_lang:
            current_words.append(tok)
        else:
            segments.append((" ".join(current_words), current_lang))
            current_words = [tok]
            current_lang = lang

    segments.append((" ".join(current_words), current_lang))
    return segments
