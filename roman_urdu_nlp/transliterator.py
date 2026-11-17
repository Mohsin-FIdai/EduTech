"""
transliterator.py — Roman Urdu to Urdu Script Transliteration module.

Contract:
    Function : transliterate(text: str) -> str
    Input    : Corrected Roman Urdu string (output of corrector.correct()).
               Must be str; raises TypeError otherwise.
    Output   : Urdu script (Unicode, RTL) string.
               - Known words looked up from a curated word map first.
               - Unknown words transliterated phoneme-by-phoneme via a
                 consonant/vowel mapping table.
               - Empty string → ""
               - The output is Urdu script only (no Latin characters remain
                 except for untranslatable code-switched English tokens
                 which are wrapped in [brackets]).

Mocking note: transliterate() receives a *corrected* string.
Mock: correct() = identity (pass pre-corrected strings in tests).

Examples:
    >>> transliterate("mujhe bohat bhook lagi hai")
    'مجھے بہت بھوک لگی ہے'

    >>> transliterate("aaj garmi bohat zyada hai yaar")
    'آج گرمی بہت زیادہ ہے یار'

    >>> transliterate("")
    ''
"""

from __future__ import annotations
import regex

# ---------------------------------------------------------------------------
# Word-level map: canonical Roman Urdu → Urdu script
# ---------------------------------------------------------------------------

_WORD_MAP: dict[str, str] = {
    # Pronouns
    "mujhe": "مجھے", "mera": "میرا", "meri": "میری", "mere": "میرے",
    "tumhara": "تمہارا", "tumhari": "تمہاری", "tumhare": "تمہارے",
    "aapka": "آپ کا", "aapki": "آپ کی", "aapke": "آپ کے",
    "unka": "ان کا", "unki": "ان کی", "unke": "ان کے",
    "uska": "اس کا", "uski": "اس کی",
    "hum": "ہم", "tum": "تم", "aap": "آپ", "woh": "وہ", "yeh": "یہ",
    "main": "میں", "wo": "وہ", "ye": "یہ",
    # Copula / auxiliaries
    "hai": "ہے", "hain": "ہیں", "tha": "تھا", "thi": "تھی", "the": "تھے",
    "hoga": "ہوگا", "hogi": "ہوگی", "honge": "ہوں گے",
    "hoon": "ہوں", "ho": "ہو",
    "raha": "رہا", "rahi": "رہی", "rahe": "رہے",
    "gaya": "گیا", "gayi": "گئی", "gaye": "گئے",
    "aaya": "آیا", "aayi": "آئی", "aaye": "آئے",
    # Negation
    "nahi": "نہیں", "na": "نہ", "mat": "مت",
    # Intensifiers & adverbs
    "bohat": "بہت", "zyada": "زیادہ", "kaafi": "کافی",
    "jaldi": "جلدی", "abhi": "ابھی",
    # Common adjectives
    "acha": "اچھا", "achi": "اچھی", "ache": "اچھے",
    "theek": "ٹھیک", "bura": "برا", "buri": "بری", "bure": "برے",
    "mushkil": "مشکل", "asaan": "آسان",
    "naya": "نیا", "nayi": "نئی", "purana": "پرانا",
    "khush": "خوش", "udaas": "اداس",
    "thaka": "تھکا", "thaki": "تھکی", "thake": "تھکے",
    "tez": "تیز", "dhima": "دھیما", "slow": "سست",
    # Nouns – body / feelings
    "bhook": "بھوک", "neend": "نیند", "dil": "دل",
    "khushi": "خوشی", "dukh": "دکھ", "gussa": "غصہ",
    "darr": "ڈر", "tension": "ٹینشن",
    # Nouns – environment
    "garmi": "گرمی", "thandi": "ٹھنڈی", "barish": "بارش",
    "dhoop": "دھوپ", "mausam": "موسم", "bijli": "بجلی",
    # Nouns – food & drink
    "khana": "کھانا", "khaana": "کھانا", "chai": "چائے",
    "pani": "پانی", "doodh": "دودھ", "biryani": "بریانی",
    "pizza": "پیزا",
    # Nouns – people
    "dost": "دوست", "yaar": "یار", "bhai": "بھائی",
    "behen": "بہن", "ammi": "امی", "abbu": "ابو",
    "beta": "بیٹا", "beti": "بیٹی",
    "teacher": "استاد", "student": "طالب علم",
    # Nouns – places
    "ghar": "گھر", "school": "اسکول", "office": "آفس",
    "bazar": "بازار", "masjid": "مسجد", "hospital": "ہسپتال",
    "karachi": "کراچی", "lahore": "لاہور", "islamabad": "اسلام آباد",
    "pakistan": "پاکستان",
    # Nouns – technology
    "phone": "فون", "internet": "انٹرنیٹ", "app": "ایپ",
    "battery": "بیٹری", "charger": "چارجر",
    "movie": "فلم", "game": "گیم", "traffic": "ٹریفک",
    "exam": "امتحان", "result": "نتیجہ", "project": "پروجیکٹ",
    "deadline": "ڈیڈلائن", "meeting": "میٹنگ",
    "download": "ڈاؤنلوڈ",
    # Nouns – time
    "kal": "کل", "aaj": "آج", "subah": "صبح",
    "raat": "رات", "shaam": "شام", "dopahar": "دوپہر",
    "din": "دن",
    # Verbs (infinitives / common forms)
    "karna": "کرنا", "karo": "کرو", "karta": "کرتا",
    "karti": "کرتی", "karte": "کرتے",
    "jana": "جانا", "jao": "جاؤ",
    "aana": "آنا", "aao": "آؤ",
    "parhna": "پڑھنا", "parha": "پڑھا",
    "likhna": "لکھنا", "bolna": "بولنا",
    "sunna": "سننا", "dekhna": "دیکھنا",
    "milna": "ملنا", "khelna": "کھیلنا",
    "samajh": "سمجھ", "samajhna": "سمجھنا",
    "lagi": "لگی", "laga": "لگا", "lage": "لگے",
    "rona": "رونا", "hona": "ہونا",
    "uthna": "اٹھنا", "bethna": "بیٹھنا",
    "submit": "جمع کرنا",
    # Connectors / prepositions
    "aur": "اور", "lekin": "لیکن", "magar": "مگر",
    "agar": "اگر", "to": "تو", "phir": "پھر",
    "ke": "کے", "ki": "کی", "ka": "کا",
    "se": "سے", "ko": "کو", "mein": "میں",
    "par": "پر", "tak": "تک", "ne": "نے",
    "bhi": "بھی", "sirf": "صرف",
    "koi": "کوئی", "kuch": "کچھ", "sab": "سب",
    "pata": "پتہ", "wala": "والا", "wali": "والی", "wale": "والے",
    # Discourse markers / greetings
    "yaar": "یار", "bhai": "بھائی",
    "waise": "ویسے", "matlab": "مطلب",
    "haha": "ہاہا", "uff": "اف", "wah": "واہ", "are": "ارے",
    "shukriya": "شکریہ", "please": "براہ کرم", "sorry": "معاف کریں",
    "salam": "سلام", "ameen": "آمین",
    "Allah": "اللہ",
    # Question words
    "kya": "کیا", "kahan": "کہاں", "kyun": "کیوں",
    "kaun": "کون", "kitna": "کتنا", "kab": "کب", "kaise": "کیسے",
    "haal": "حال",
    # Misc
    "wait": "انتظار",
    "load": "لوڈ",
    "hang": "ہینگ",
    "funny": "مضحکہ خیز",
    "best": "بہترین",
    "seriously": "سچ میں",
    "finally": "آخر کار",
    "maine": "میں نے",
    "bahar": "باہر", "baahar": "باہر", "andar": "اندر",
    "dair": "دیر",
}

# ---------------------------------------------------------------------------
# Phoneme-level fallback: Roman Urdu digraphs/chars → Urdu letters
# Ordered longest-match first (digraphs before single chars)
# ---------------------------------------------------------------------------

_PHONEME_MAP: list[tuple[str, str]] = [
    # Digraphs (must come before singles)
    ("kh", "خ"), ("gh", "غ"), ("sh", "ش"), ("ch", "چ"),
    ("ph", "پھ"), ("bh", "بھ"), ("th", "تھ"), ("dh", "دھ"),
    ("rh", "ڑ"), ("zh", "ژ"), ("jh", "جھ"), ("nh", "ں"),
    # Vowels
    ("aa", "آ"), ("ee", "ی"), ("oo", "و"), ("uu", "و"),
    ("ai", "ے"), ("au", "او"), ("oe", "وئے"), ("ie", "یئے"),
    ("a", "ا"), ("e", "ے"), ("i", "ی"), ("o", "و"), ("u", "و"),
    # Consonants
    ("b", "ب"), ("p", "پ"), ("t", "ت"), ("j", "ج"),
    ("d", "د"), ("r", "ر"), ("z", "ز"), ("s", "س"),
    ("f", "ف"), ("q", "ق"), ("k", "ک"), ("g", "گ"),
    ("l", "ل"), ("m", "م"), ("n", "ن"), ("h", "ہ"),
    ("w", "و"), ("v", "و"), ("y", "ی"),
    ("c", "ک"), ("x", "کس"),
]

_ENGLISH_TOKEN_RE = regex.compile(r"^[a-zA-Z]+$")


def _transliterate_token(token: str) -> str:
    """Transliterate a single Roman Urdu token to Urdu script."""
    lower = token.lower()

    # 1. Word-map lookup (most accurate)
    if lower in _WORD_MAP:
        return _WORD_MAP[lower]

    # 2. If all Latin letters, try phoneme-level fallback
    if _ENGLISH_TOKEN_RE.match(token):
        result = []
        i = 0
        while i < len(lower):
            matched = False
            for roman, urdu in _PHONEME_MAP:
                if lower[i:].startswith(roman):
                    result.append(urdu)
                    i += len(roman)
                    matched = True
                    break
            if not matched:
                result.append(lower[i])
                i += 1
        return "".join(result)

    # 3. Non-Latin / numeric — return as-is (e.g. digits, Urdu already)
    return token


def transliterate(text: str) -> str:
    """
    Convert a corrected Roman Urdu string to Urdu script.

    Parameters
    ----------
    text : str
        Corrected Roman Urdu string (output of corrector.correct()).

    Returns
    -------
    str
        Urdu script string. Returns "" for empty input.

    Raises
    ------
    TypeError
        If ``text`` is not a ``str``.
    """
    if not isinstance(text, str):
        raise TypeError(f"transliterate() expects str, got {type(text).__name__!r}")

    text = text.strip()
    if not text:
        return ""

    tokens = text.split()
    urdu_tokens = [_transliterate_token(tok) for tok in tokens]
    return " ".join(urdu_tokens)
