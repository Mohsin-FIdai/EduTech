"""
tests/test_translation_toggle.py — Rigorous tests for the offline translation toggle.

Validates:
1. env_flag parser behavior (defaults, truthy, falsy, case-insensitivity, empty string).
2. roman_urdu_nlp.translator.translate honors ENABLE_ONLINE_TRANSLATION:
   - When True, attempts online translation.
   - When False, returns fallback immediately with NO socket/network call.
3. TextProcessingService.translate_segments_batch honors ENABLE_ONLINE_TRANSLATION:
   - When False, GoogleTranslator is NEVER instantiated or called.
   - Sockets are strictly blocked (raising on any connect attempt).
   - Provenance reason is tagged as 'skipped_online_disabled'.
   - Output routes cleanly to Local LLM or Tier 3 offline dictionary.
4. Tier 3 dictionary translation returns non-empty, valid English for known Roman-Urdu samples.
"""

import os
import socket
from unittest.mock import MagicMock, patch
import pytest

from roman_urdu_nlp.utils import env_flag
from roman_urdu_nlp.translator import translate as library_translate, _FALLBACK
from app.services.text_processor import TextProcessingService, _dictionary_translate
from app import config


class DummySegment:
    """Mock transcript segment item."""
    def __init__(self, text, normalized_text=None):
        self.text = text
        self.normalized_text = normalized_text or text
        self.urdu_text = None
        self.english_text = None
        self.translation_source = None
        self.translation_confidence = None
        self.translation_reason = None


# ─── 1. Flag Parser Unit Tests ──────────────────────────────────────

class TestEnvFlagParsing:
    """Verify boolean parser correctness according to specification."""

    def test_default_is_true(self, monkeypatch):
        monkeypatch.delenv("TEST_FLAG_KEY", raising=False)
        assert env_flag("TEST_FLAG_KEY", default=True) is True

    def test_default_is_false_when_specified(self, monkeypatch):
        monkeypatch.delenv("TEST_FLAG_KEY", raising=False)
        assert env_flag("TEST_FLAG_KEY", default=False) is False

    @pytest.mark.parametrize("val", ["true", "True", "TRUE", "1", "yes", "YES", "on", "On"])
    def test_truthy_values(self, monkeypatch, val):
        monkeypatch.setenv("TEST_FLAG_KEY", val)
        assert env_flag("TEST_FLAG_KEY") is True

    @pytest.mark.parametrize("val", ["false", "False", "FALSE", "0", "off", "Off", "no", "NO", "", "   "])
    def test_falsy_values(self, monkeypatch, val):
        monkeypatch.setenv("TEST_FLAG_KEY", val)
        assert env_flag("TEST_FLAG_KEY") is False


# ─── 2. Library-Level Offline Toggle Tests ─────────────────────────

class TestLibraryTranslatorOfflineToggle:
    """Verify roman_urdu_nlp.translator.translate honors the flag."""

    def test_offline_mode_does_not_attempt_network(self, monkeypatch):
        """When ENABLE_ONLINE_TRANSLATION is false, no network connection must occur."""
        monkeypatch.setenv("ENABLE_ONLINE_TRANSLATION", "false")

        # Patch socket connect to strictly forbid any network access
        def forbid_connect(*args, **kwargs):
            raise RuntimeError("CRITICAL: Network connection attempted while online translation is disabled!")

        monkeypatch.setattr(socket.socket, "connect", forbid_connect)

        # Also patch GoogleTranslator to guarantee it is not even instantiated or called
        with patch("roman_urdu_nlp.translator.GoogleTranslator") as mock_gt:
            result = library_translate("Muje boht bhok lgi h")
            mock_gt.assert_not_called()
            assert result == _FALLBACK

    def test_online_mode_invokes_google_translator(self, monkeypatch):
        """When ENABLE_ONLINE_TRANSLATION is true, GoogleTranslator is invoked."""
        monkeypatch.setenv("ENABLE_ONLINE_TRANSLATION", "true")

        with patch("roman_urdu_nlp.translator.GoogleTranslator") as mock_gt:
            mock_instance = MagicMock()
            mock_instance.translate.return_value = "I am very hungry"
            mock_gt.return_value = mock_instance

            result = library_translate("Muje boht bhok lgi h")
            mock_gt.assert_called_once_with(source="ur", target="en")
            mock_instance.translate.assert_called_once()
            assert result == "I am very hungry"


# ─── 3. TextProcessingService Batch Offline Toggle Tests ───────────

class TestServiceTranslationOfflineToggle:
    """Verify TextProcessingService.translate_segments_batch honors the toggle."""

    def test_offline_batch_translation_strictly_no_network(self, monkeypatch):
        """Disabling online translation completely bypasses Google Translate and forbids sockets."""
        monkeypatch.setattr(config, "ENABLE_ONLINE_TRANSLATION", False)
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")

        # Enforce hard block on socket connections
        def forbid_connect(*args, **kwargs):
            raise RuntimeError("CRITICAL: Socket connect called during offline translation test!")

        monkeypatch.setattr(socket.socket, "connect", forbid_connect)

        # Segments needing translation (Roman Urdu)
        sample_segments = [
            DummySegment("Sir process ready hai tou CPU kyu nahi mil raha?"),
            DummySegment("Agle hafte quiz hoga aur sab students ko prepare karna hai.")
        ]

        # Reset single-log indicator
        TextProcessingService._offline_logged = False

        # Deep translator must NOT be touched
        with patch("deep_translator.GoogleTranslator") as mock_gt:
            results = TextProcessingService.translate_segments_batch(sample_segments)
            mock_gt.assert_not_called()

        assert len(results) == 2
        for seg in results:
            assert seg.english_text is not None
            assert len(seg.english_text.strip()) > 0
            # Must fall back to local_llm or dictionary
            assert seg.translation_source in ("local_llm", "dictionary")
            # Provenance reason must clearly indicate online was disabled
            assert seg.translation_reason == "skipped_online_disabled"

    def test_dict_fallback_is_non_empty_for_known_sample(self):
        """Offline dictionary produces non-empty, translated English for common Roman Urdu words."""
        sample = "Sir agar process ready hai lekin CPU nahi mila"
        translated = _dictionary_translate(sample)

        assert translated is not None
        assert len(translated.strip()) > 0
        # 'hai' -> 'is', 'lekin' -> 'but', 'nahi' -> 'not', 'agar' -> 'if'
        assert "is" in translated
        assert "but" in translated
        assert "not" in translated
        assert "if" in translated

    def test_online_batch_translation_calls_google_when_enabled(self, monkeypatch):
        """When ENABLE_ONLINE_TRANSLATION is True, Google Translate is invoked."""
        monkeypatch.setattr(config, "ENABLE_ONLINE_TRANSLATION", True)

        sample_segments = [
            DummySegment("Sir process ready hai tou CPU kyu nahi mil raha?")
        ]

        with patch("deep_translator.GoogleTranslator") as mock_gt:
            mock_instance = MagicMock()
            mock_instance.translate.return_value = "[0] Sir if process is ready then why CPU is not available?"
            mock_gt.return_value = mock_instance

            results = TextProcessingService.translate_segments_batch(sample_segments)
            mock_gt.assert_called_once()
            assert results[0].translation_source == "google_translate"
            assert results[0].translation_reason == "online_success"
