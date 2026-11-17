"""
translator.py — English Translation module for roman_urdu_nlp.

Contract:
    Function : translate(text: str) -> str
    Input    : Corrected Roman Urdu string (output of corrector.correct()).
               Must be str; raises TypeError otherwise.
    Output   : English translation string.
               - Uses deep-translator (Google Translate) with source="ur"
                 after first converting Roman Urdu → Urdu script internally,
                 then translating Urdu script → English.
               - If the API call fails (network error, timeout, etc.), returns
                 a bracketed fallback: "[translation unavailable]"
               - Empty string → ""

Mocking note: translate() receives a *corrected* string.
Mock: correct() = identity (pass pre-corrected strings in tests).
Network calls are mocked in tests using unittest.mock.patch.

Examples:
    >>> translate("mujhe bohat bhook lagi hai")
    'I am very hungry.'

    >>> translate("aaj garmi bohat zyada hai yaar")
    'It is very hot today friend.'

    >>> translate("")
    ''
"""

from __future__ import annotations

# Lazy import of transliterator to avoid circular imports and keep module
# independently testable via mocking.
from roman_urdu_nlp.transliterator import transliterate

try:
    from deep_translator import GoogleTranslator
except ImportError:  # pragma: no cover
    GoogleTranslator = None  # type: ignore[assignment,misc]

_FALLBACK = "[translation unavailable]"


def translate(text: str) -> str:
    """
    Translate a corrected Roman Urdu string into English.

    The pipeline is:
        corrected Roman Urdu → Urdu script (via transliterate) → English (via Google Translate)

    Parameters
    ----------
    text : str
        Corrected Roman Urdu string (output of corrector.correct()).

    Returns
    -------
    str
        English translation, or "[translation unavailable]" on API failure.
        Returns "" for empty input.

    Raises
    ------
    TypeError
        If ``text`` is not a ``str``.
    """
    if not isinstance(text, str):
        raise TypeError(f"translate() expects str, got {type(text).__name__!r}")

    text = text.strip()
    if not text:
        return ""

    # Step 1: Convert Roman Urdu → Urdu script
    urdu_text = transliterate(text)

    # Step 2: Translate Urdu script → English
    from roman_urdu_nlp.utils import env_flag
    if not env_flag("ENABLE_ONLINE_TRANSLATION", default=True):
        return _FALLBACK

    try:
        if GoogleTranslator is None:
            return _FALLBACK
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(GoogleTranslator(source="ur", target="en").translate, urdu_text)
            result = future.result(timeout=5.0)
        return result if result else _FALLBACK
    except Exception:
        # Network errors, timeouts, API limits → graceful fallback
        return _FALLBACK
