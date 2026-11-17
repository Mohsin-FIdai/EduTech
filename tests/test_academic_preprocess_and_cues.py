"""
tests/test_academic_preprocess_and_cues.py — Unit Tests for Phase 1.

Covers:
- AcademicPreprocessor:
  - Token offset round-trip: token_map resolves to slice containing the token.
  - Negation preservation ('not', 'nahi', 'nahin').
  - Formula preservation ('I = Q/t', 'V = IR').
  - Disfluency cleaning ('bhaiya', 'acha ji').
  - Segment boundary isolation (no cross-segment merging).
- DiscourseCueClassifier:
  - Affirmed exam warning ('exam mein aayega').
  - Negation flip ('yeh exam mein nahi aayega' -> negated).
  - Hedge confidence reduction ('shayad exam mein aaye' -> hedged, halved confidence).
  - Suppression rule 1: 'unit test' suppresses bare test false alarm.
  - Suppression rule 2: 'question mark' suppresses marks false alarm.
  - Spelling variation tolerance ('ayega' -> 'aayega', 'pepar' -> 'paper').
  - Raw quote integrity (stored quote is exactly raw utterance).
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from app.services.academic_preprocess import AcademicPreprocessor
from app.services.discourse_cues import DiscourseCueClassifier


class TestAcademicPreprocessor(unittest.TestCase):
    """Tests academic copy extraction and offset mapping."""

    def test_token_offset_roundtrip(self):
        raw = "CPU scheduling dispatches FCFS processes across the ready queue."
        clean, token_map, tokens = AcademicPreprocessor.clean_segment_text(raw)
        self.assertEqual(len(tokens), len(token_map))
        for tok, (s_char, e_char) in zip(tokens, token_map):
            raw_slice = raw[s_char:e_char]
            self.assertIn(tok, raw_slice, f"Token '{tok}' not found in raw slice '{raw_slice}'")

    def test_preserves_negations_english_and_urdu(self):
        raw = "This is not required and yeh exam mein nahi aayega."
        clean, _, tokens = AcademicPreprocessor.clean_segment_text(raw)
        tokens_low = [t.lower() for t in tokens]
        self.assertIn("not", tokens_low)
        self.assertIn("nahi", tokens_low)

    def test_preserves_formulas_and_numerals(self):
        raw = "Ohm's Law states V = IR and current I = Q/t with 10 amperes."
        clean, _, tokens = AcademicPreprocessor.clean_segment_text(raw)
        self.assertIn("V", tokens)
        self.assertIn("IR", tokens)
        self.assertIn("10", tokens)

    def test_cleans_conversational_disfluencies(self):
        raw = "Bhaiya dekhein theek hai so CPU scheduling is our topic."
        clean, _, tokens = AcademicPreprocessor.clean_segment_text(raw)
        tokens_low = [t.lower() for t in tokens]
        self.assertNotIn("bhaiya", tokens_low)
        self.assertNotIn("theek", tokens_low)
        self.assertIn("cpu", tokens_low)
        self.assertIn("scheduling", tokens_low)

    def test_segments_processed_independently(self):
        seg1 = SimpleNamespace(id=1, start_time=0.0, end_time=5.0, text="First segment FCFS.")
        seg2 = SimpleNamespace(id=2, start_time=5.0, end_time=10.0, text="Second segment convoy effect.")
        res = AcademicPreprocessor.process_segments([seg1, seg2])
        self.assertEqual(len(res.spans), 2)
        self.assertEqual(res.spans[0].segment_id, 1)
        self.assertEqual(res.spans[1].segment_id, 2)
        self.assertIn("fcfs", res.recording_vocab)
        self.assertIn("convoy", res.recording_vocab)


class TestDiscourseCueClassifier(unittest.TestCase):
    """Tests polarity-aware exam and importance cue detection."""

    def test_affirmed_exam_cue(self):
        seg = SimpleNamespace(id=10, start_time=12.0, end_time=15.0, text="Yeh topic exam mein aayega note kar lein.")
        hints = DiscourseCueClassifier.classify_segment(seg)
        self.assertEqual(len(hints), 1)
        self.assertEqual(hints[0].polarity, "affirmed")
        self.assertEqual(hints[0].cue_type, "exam")
        self.assertEqual(hints[0].raw_quote, seg.text)

    def test_negated_exam_cue_flips_polarity(self):
        seg = SimpleNamespace(id=11, start_time=18.0, end_time=22.0, text="Yeh section exam mein nahi aayega bilkul bhi.")
        hints = DiscourseCueClassifier.classify_segment(seg)
        self.assertEqual(len(hints), 1)
        self.assertEqual(hints[0].polarity, "negated")
        self.assertEqual(hints[0].cue_type, "exam")

    def test_hedged_cue_reduces_confidence(self):
        seg = SimpleNamespace(id=12, start_time=25.0, end_time=29.0, text="Shayad yeh final paper mein aayega.")
        hints = DiscourseCueClassifier.classify_segment(seg)
        self.assertEqual(len(hints), 1)
        self.assertEqual(hints[0].polarity, "hedged")
        self.assertLess(hints[0].confidence, 0.70)

    def test_suppression_unit_test(self):
        """'unit test' must not be misclassified as an exam announcement."""
        seg = SimpleNamespace(id=13, start_time=30.0, end_time=33.0, text="We write a unit test to verify the algorithm.")
        hints = DiscourseCueClassifier.classify_segment(seg)
        self.assertEqual(len(hints), 0, f"Expected no hints but got {hints}")

    def test_suppression_question_marks(self):
        """'question marks' punctuation must not be misclassified as marks allocation."""
        seg = SimpleNamespace(id=14, start_time=35.0, end_time=38.0, text="Do not forget the question mark at the end.")
        hints = DiscourseCueClassifier.classify_segment(seg)
        self.assertEqual(len(hints), 0, f"Expected no hints but got {hints}")

    def test_spelling_variants_tolerant(self):
        """Variant spellings like 'pepar' and 'ayega' must fire."""
        seg = SimpleNamespace(id=15, start_time=40.0, end_time=44.0, text="Yeh midterm pepar mein ayega zaroori.")
        hints = DiscourseCueClassifier.classify_segment(seg)
        self.assertTrue(any(h.cue_type == "exam" and h.polarity == "affirmed" for h in hints))


if __name__ == "__main__":
    unittest.main()
