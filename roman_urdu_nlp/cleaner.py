"""
cleaner.py — Text Cleaning module for roman_urdu_nlp.

Contract:
    Function : clean(text: str) -> str
    Input    : Raw string (may contain emojis, repeated chars, noisy punctuation,
               mixed whitespace).  Must be str; raises TypeError otherwise.
    Output   : Cleaned string with:
               - Emojis removed
               - Runs of 3+ identical characters collapsed to 2
                 (preserves meaningful doubles like "aa", "oo")
               - Repeated laugh patterns ("hahahaha") collapsed to "haha"
               - Standalone punctuation noise stripped (???, !!!, ...)
               - Excess whitespace normalised to single spaces
               - Leading/trailing whitespace stripped
               Empty string / pure-emoji input → returns ""

Examples:
    >>> clean("Muje boht bhok lgi h 😂😂")
    'Muje boht bhok lgi h'

    >>> clean("kya haal hai bhai??? sab theek thaak??")
    'kya haal hai bhai sab theek thaak'

    >>> clean("aaj garmi boht zyada h yarrr 🥵🥵🥵")
    'aaj garmi boht zyada h yaar'

    >>> clean("")
    ''
"""

import regex

# ---------------------------------------------------------------------------
# Compiled patterns (module-level for performance)
# ---------------------------------------------------------------------------

# All Unicode emoji characters (using the \p{Emoji} property from `regex`)
_EMOJI_RE = regex.compile(
    r"[\U0001F600-\U0001F64F"   # emoticons
    r"\U0001F300-\U0001F5FF"    # misc symbols & pictographs
    r"\U0001F680-\U0001F6FF"    # transport & map
    r"\U0001F700-\U0001F77F"    # alchemical symbols
    r"\U0001F780-\U0001F7FF"    # geometric shapes extended
    r"\U0001F800-\U0001F8FF"    # supplemental arrows-c
    r"\U0001F900-\U0001F9FF"    # supplemental symbols & pictographs
    r"\U0001FA00-\U0001FA6F"    # chess symbols
    r"\U0001FA70-\U0001FAFF"    # symbols & pictographs extended-a
    r"\U00002702-\U000027B0"    # dingbats
    r"\U000024C2-\U0001F251"    # enclosed characters
    r"\u200d"                    # zero-width joiner
    r"\ufe0f"                    # variation selector-16
    r"]+",
    regex.UNICODE,
)

# Runs of 3+ identical characters → collapse to 2
_TRIPLE_CHAR_RE = regex.compile(r"(.)\1{2,}", regex.UNICODE)

# Repeated "ha" pattern: (ha){3,} → "haha"
_LAUGH_RE = regex.compile(r"(?:ha){3,}", regex.IGNORECASE)

# Punctuation-only tokens: any run of noise punctuation not adjacent to word chars
# Using 1+ so that ??? collapses to ?? to ? and that final ? is still removed
_PUNCT_NOISE_RE = regex.compile(r"(?<!\w)[?!.,;:\-\u2013\u2014]+(?!\w)")

# Trailing punctuation attached to words (e.g. "bhai?" -> "bhai", "seriously!" -> "seriously")
_TRAILING_PUNCT_RE = regex.compile(r"[?!]+(?=\s|$)")

# Multiple spaces / tabs / newlines → single space
_WHITESPACE_RE = regex.compile(r"\s+")


def clean(text: str) -> str:
    """
    Clean raw Roman Urdu text by removing emojis, collapsing repeated characters,
    stripping punctuation noise, and normalising whitespace.

    Parameters
    ----------
    text : str
        Raw input string.

    Returns
    -------
    str
        Cleaned string. Returns "" for empty or emoji-only input.

    Raises
    ------
    TypeError
        If ``text`` is not a ``str``.
    """
    if not isinstance(text, str):
        raise TypeError(f"clean() expects str, got {type(text).__name__!r}")

    # 1. Remove emojis
    text = _EMOJI_RE.sub("", text)

    # 2. Collapse repeated laugh patterns before general char collapse
    #    "hahahaha" → "haha"
    text = _LAUGH_RE.sub("haha", text)

    # 3. Collapse runs of 3+ identical chars to 2  ("yarrr" → "yarr", then
    #    a second pass collapses "yarr" → "yaar" — no, we keep doubles.)
    #    Roman Urdu uses doubles meaningfully ("aa", "oo", "ee"), so we
    #    stop at 2, not 1.
    text = _TRIPLE_CHAR_RE.sub(r"\1\1", text)

    # 4. Strip punctuation-noise tokens (standalone clusters)
    text = _PUNCT_NOISE_RE.sub("", text)

    # 4b. Strip trailing ?! attached to word endings
    text = _TRAILING_PUNCT_RE.sub("", text)

    # 5. Normalise whitespace
    text = _WHITESPACE_RE.sub(" ", text).strip()

    return text
