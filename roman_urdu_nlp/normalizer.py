"""
normalizer.py — Normalization + Abbreviation Expansion module for roman_urdu_nlp.

Contract:
    Function : normalize(text: str) -> str
    Input    : Cleaned string (output of cleaner.clean()).
               Must be str; raises TypeError otherwise.
    Output   : Canonical Roman Urdu string with:
               - Spelling variants mapped to canonical forms
                 (e.g. "acha"/"accha"/"achha" → "acha",
                       "boht"/"bhut"/"bahut" → "bohat")
               - Chat abbreviations expanded
                 (e.g. "mjy"/"mje"/"muje" → "mujhe",
                       "nhi"/"ni" → "nahi",
                       "h"/"ha" at end-of-token → "hai")
               - Case of first letter of sentence preserved;
                 other tokens lowercased.
               Empty string → ""

Mocking note: This module was developed against a mock clean() = identity.
The real cleaner is independent; swap is transparent since the contract is
identical (str → str).

Examples:
    >>> normalize("Muje boht bhok lgi h")
    'Mujhe bohat bhook lagi hai'

    >>> normalize("mjy neend aa rhi h yaar")
    'mujhe neend aa rahi hai yaar'

    >>> normalize("kya haal hai bhai sab theek thaak")
    'kya haal hai bhai sab theek thaak'
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Normalization dictionary — spelling variants → canonical form
# Keys are LOWERCASE. Lookup is done on lowercased tokens.
# ---------------------------------------------------------------------------

_NORM_MAP: dict[str, str] = {
    # ── Pronouns ──────────────────────────────────────────────────────────
    "mjy": "mujhe", "mje": "mujhe", "muje": "mujhe", "mjhe": "mujhe",
    "mujy": "mujhe",
    "mry": "mere", "mera": "mera", "meri": "meri",
    "tmhara": "tumhara", "tmhari": "tumhari", "tmhare": "tumhare",
    "apko": "aapko", "aapko": "aapko",
    "unhe": "unhe", "unko": "unko",

    # ── Copula / auxiliaries ──────────────────────────────────────────────
    "h": "hai", "ha": "hai", "haa": "hai",
    "hy": "hai", "hey": "hai",
    "hun": "hoon", "huu": "hoon", "hu": "hoon",
    "rhi": "rahi", "rhee": "rahi",
    "rha": "raha", "rhaa": "raha",
    "rhe": "rahe",
    "hogi": "hogi", "hoga": "hoga",

    # ── Negation ──────────────────────────────────────────────────────────
    "nhi": "nahi", "ni": "nahi", "nhe": "nahi", "nhee": "nahi",
    "nae": "nahi", "nay": "nahi", "naa": "na",

    # ── Indefinite / Quantifiers ──────────────────────────────────────────
    "kch": "kuch", "kuch": "kuch", "koi": "koi", "sb": "sab",

    # ── Intensifiers ──────────────────────────────────────────────────────
    "boht": "bohat", "bhut": "bohat", "bht": "bohat", "bohot": "bohat",
    "bahut": "bohat",
    "zyda": "zyada", "zada": "zyada", "ziada": "zyada",
    "kaafi": "kaafi",  # already canonical

    # ── Positive adjectives ───────────────────────────────────────────────
    "acha": "acha", "achha": "acha", "accha": "acha",
    "achi": "achi", "achhi": "achi", "acchi": "achi",
    "ache": "ache", "achhe": "ache", "acche": "ache",
    "theek": "theek", "thik": "theek", "thek": "theek",

    # ── Common verbs / verb forms ──────────────────────────────────────────
    "lgi": "lagi", "lge": "lage",
    "kro": "karo", "karna": "karna",
    "jana": "jana", "jao": "jao",
    "aana": "aana", "ao": "aao",
    "parha": "parha", "para": "parha", "parra": "parha",
    "parhna": "parhna",
    "smjh": "samajh", "samjh": "samajh",
    "dekho": "dekho", "dekh": "dekh",
    "suno": "suno", "sun": "sun",
    "bolo": "bolo", "bol": "bol",
    "milo": "milo", "mil": "mil",
    "jldi": "jaldi", "jldy": "jaldi",
    "uthna": "uthna", "uthana": "uthana",

    # ── Nouns ──────────────────────────────────────────────────────────────
    "bhok": "bhook", "bhookh": "bhook", "bhooka": "bhooka",
    "neend": "neend",  # already canonical
    "khaana": "khaana", "khana": "khana",
    "dost": "dost",
    "bhai": "bhai",
    "yaar": "yaar", "yar": "yaar",
    "bat": "baat", "baat": "baat",
    "ap": "aap", "aap": "aap",

    # ── Temporal ──────────────────────────────────────────────────────────
    "kal": "kal", "aaj": "aaj", "abhi": "abhi",
    "subah": "subah", "raat": "raat",

    # ── Connectors / prepositions ──────────────────────────────────────────
    "aur": "aur", "lekin": "lekin", "magar": "magar",
    "ke": "ke", "ki": "ki", "ka": "ka",
    "se": "se", "ko": "ko", "mein": "mein",
    "per": "par", "pe": "par",
    "toh": "to", "tou": "to",
    "fir": "phir", "fer": "phir",
    "phir": "phir",

    # ── Common abbreviations / chat-speak ─────────────────────────────────
    "pls": "please", "plz": "please", "plzz": "please",
    "thnx": "shukriya", "thx": "shukriya", "ty": "shukriya",
    "brb": "abhi aata hoon",
    "lol": "haha",
    "omg": "oh my god",
    "btw": "waise",
    "tbh": "sach mein",
    # ── Roman Urdu verbs & inflections ────────────────────────────────────
    "kr": "kar", "kro": "karo", "krna": "karna", "krny": "karne", "krta": "karta", "krti": "karti", "krte": "karte", "kray": "kare",
    "btao": "batao", "btain": "batayein", "btana": "batana", "btya": "bataya",
    "smjh": "samajh", "smjhe": "samjhe", "smjhna": "samjhna", "smjha": "samjha",
    "prh": "parh", "prha": "parha", "prhna": "parhna", "prhe": "parhe",
    "dkh": "dekh", "dkha": "dekha", "dkhna": "dekhna",
    "chahye": "chahiye", "chaiye": "chahiye", "chaheye": "chahiye",
    "krwao": "karwao", "krwana": "karwana",
    "bhej": "bhej", "bhjo": "bhejo", "bhjna": "bhejna",

    # ── ASR Technical Corruptions & Phonetic Artifacts ────────────────────
    "tapper": "topic", "tapek": "topic", "taapuk": "topic", "tapic": "topic",
    "artifisial": "artificial", "aratifisial": "artificial",
    "larnigwa": "learning", "lerning": "learning",
    "neorol": "neural", "nural": "neural",
    "algorthm": "algorithm", "elgorithm": "algorithm",
    "asignment": "assignment", "asignement": "assignment",
    "presintation": "presentation", "prezintation": "presentation",
    "atendance": "attendance", "semister": "semester",

    # ── English Chat Abbreviations ────────────────────────────────────────
    "pls": "please", "plz": "please", "thx": "thanks", "ty": "thank you",
    "tc": "take care", "asap": "as soon as possible",
    "idk": "mujhe nahi pata",
    "imo": "mere khayal mein",

    # ── Filler / discourse markers ─────────────────────────────────────────
    "wese": "waise", "wesy": "waise",
    "mtlb": "matlab", "mtlab": "matlab",
    "matlab": "matlab",
    "pata": "pata",
    "wala": "wala", "wali": "wali", "wale": "wale",

    # ── Greetings ─────────────────────────────────────────────────────────
    "salam": "salam", "slm": "salam",
    "assalam": "assalam",

    # ── Conversational Adverbs & Discourse Markers (Real Speech Corpus) ──
    "bohot": "bohat", "bhot": "bohat", "bht": "bohat",
    "ziada": "zyada", "zyda": "zyada", "zada": "zyada",
    "bilkul": "bilkul", "blkul": "bilkul", "bilqul": "bilkul",
    "zaroor": "zaroor", "zarur": "zaroor", "zroor": "zaroor",
    "waghera": "waghera", "wgera": "waghera", "wghera": "waghera",
    "taqreeban": "taqreeban", "taqriban": "taqreeban", "tqreeban": "taqreeban",
    "waqai": "waqai", "waqaye": "waqai", "waqayi": "waqai",
    "shuru": "shuru", "shuroo": "shuru", "shuro": "shuru",
    "khatam": "khatam", "khtm": "khatam",
    "koshish": "koshish", "koshsh": "koshish",
    "hamesha": "hamesha", "hmsha": "hamesha",
    "pehle": "pehle", "pehly": "pehle", "phle": "pehle", "pehlay": "pehle",
    "sahi": "sahi", "sahih": "sahi",
    "shayad": "shayad", "shyd": "shayad",
    "duniya": "duniya", "dunya": "duniya",
    "waqt": "waqt", "wqt": "waqt",
    "zindagi": "zindagi", "zndagi": "zindagi",
    "chahiye": "chahiye", "chahye": "chahiye", "chaiye": "chahiye", "chahey": "chahiye",
    "samajh": "samajh", "smjh": "samajh", "samjh": "samajh",
    "dekhna": "dekhna", "dkhna": "dekhna", "dekha": "dekha", "dkha": "dekha",
    "karne": "karne", "krne": "karne", "krny": "karne",
    "karna": "karna", "krna": "karna",
    "karte": "karte", "krte": "karte",
    "karti": "karti", "krti": "karti",
    "karta": "karta", "krta": "karta",
    "karein": "karein", "krain": "karein",
    "batayein": "batayein", "btain": "batayein",
    "batana": "batana", "btana": "batana",
    "bolna": "bolna", "blna": "bolna",
    "sunna": "sunna", "snna": "sunna",

    # ── Laughter / interjections ──────────────────────────────────────────
    "haha": "haha",
    "ufff": "uff", "ufffff": "uff",
    "ugh": "ugh",
    "waah": "wah", "wah": "wah",
    "arrey": "are", "arey": "are",
}

_MULTI_WORD_ASR_FIXES: dict[str, str] = {
    "arati fisial": "artificial",
    "arati fisal": "artificial",
    "aati fisial": "artificial",
    "aati fisal": "artificial",
}

# Context tokens indicating Roman Urdu clause
_URDU_CONTEXT_WORDS: frozenset[str] = frozenset({
    "aaj", "kal", "abhi", "phir", "hai", "hain", "tha", "thi", "the",
    "karna", "karo", "kare", "mujhe", "mera", "meri", "mere", "aap", "aapka",
    "tum", "tumhara", "yeh", "woh", "is", "us", "mein", "se", "ko", "par",
    "bohat", "nahi", "kuch", "sab", "yaar", "bhai", "shuru", "khatam"
})


def normalize(text: str) -> str:
    """
    Normalize a cleaned Roman Urdu string to canonical spelling, recover
    ASR phonetic corruptions, expand abbreviations, and disambiguate context homophones.

    Parameters
    ----------
    text : str
        Cleaned input string (output of cleaner.clean()).

    Returns
    -------
    str
        Normalized canonical Roman Urdu / English code-mixed string.
    """
    if not isinstance(text, str):
        raise TypeError(f"normalize() expects str, got {type(text).__name__!r}")

    text = text.strip()
    if not text:
        return ""

    # 1. Multi-word ASR corruption fixes
    processed_text = text
    lower_text = processed_text.lower()
    for raw_phrase, fix in _MULTI_WORD_ASR_FIXES.items():
        if raw_phrase in lower_text:
            # Case-insensitive replacement
            import re
            pattern = re.compile(re.escape(raw_phrase), re.IGNORECASE)
            processed_text = pattern.sub(fix, processed_text)

    tokens = processed_text.split()
    normalized_tokens: list[str] = []
    lower_tokens = [t.lower().strip(".,!?;:\"'()[]{}") for t in tokens]

    for i, token in enumerate(tokens):
        lower = lower_tokens[i]

        # Context-aware disambiguation for 'or' vs 'aur'
        if lower == "or":
            # Check surrounding context window (+/- 2 words)
            context_window = [
                lower_tokens[j]
                for j in range(max(0, i - 2), min(len(lower_tokens), i + 3))
                if j != i
            ]
            has_urdu_context = any(w in _URDU_CONTEXT_WORDS for w in context_window)
            if has_urdu_context:
                replacement = "aur"
                if i == 0 and token[0].isupper():
                    replacement = "Aur"
                normalized_tokens.append(replacement)
                continue

        replacement = _NORM_MAP.get(lower)

        if replacement is not None:
            # Preserve capitalization of the very first token
            if i == 0 and token[0].isupper():
                replacement = replacement[0].upper() + replacement[1:]
            normalized_tokens.append(replacement)
        else:
            normalized_tokens.append(token)

    return " ".join(normalized_tokens)
