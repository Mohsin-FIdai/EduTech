import pytest
from unittest.mock import patch
from deep_translator.exceptions import TooManyRequests
from app.services.text_processor import TextProcessingService


class DummySegment:
    def __init__(self, text, normalized_text=None):
        self.text = text
        self.normalized_text = normalized_text or text
        self.urdu_text = None
        self.english_text = None
        self.translation_source = None
        self.translation_confidence = None


def test_translation_fallback_chain_under_rate_limit():
    # Mix of Roman Urdu and English lecture text
    sample_segments = [
        DummySegment('Lekin jab system has 2 dozens or hundreds of processes, urgent and waiting.'),
        DummySegment('Sir agar process ready hai tou CPU kyu nahi mil raha?'),
        DummySegment('Agle hafte quiz hoga aur sab students ko prepare karna hai.')
    ]

    # Patch GoogleTranslator.translate to simulate Google rate limit (TooManyRequests)
    with patch('deep_translator.GoogleTranslator.translate', side_effect=TooManyRequests('Rate limit exceeded')):
        results = TextProcessingService.translate_segments_batch(sample_segments)

    for seg in results:
        # Check Gap 1 & Gap 5 criteria:
        assert seg.english_text is not None, 'english_text must not be None'
        assert len(seg.english_text.strip()) > 0, 'english_text must not be empty'
        assert '[translation unavailable]' not in seg.english_text, 'english_text must never be [translation unavailable]'
        assert seg.translation_source in ('local_llm', 'dictionary', 'passthrough'), f'Invalid translation_source: {seg.translation_source}'
        assert seg.translation_confidence is not None
        assert 0.0 <= seg.translation_confidence <= 1.0
        assert seg.urdu_text is not None and len(seg.urdu_text) > 0, 'urdu_text must be populated'

    # Verify at least one non-passthrough segment fell back to local_llm or dictionary
    sources = [s.translation_source for s in results]
    assert any(s in ('local_llm', 'dictionary') for s in sources), f'Expected fallback source in {sources}'
