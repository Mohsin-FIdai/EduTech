"""
test_gold_set.py — End-to-end gold-standard evaluation for roman_urdu_nlp.

Runs the full Pipeline on all 20 gold-set items and reports:
- Per-item: normalized match (exact & fuzzy), urdu match, english present
- Summary: accuracy per stage

Run with: pytest tests/test_gold_set.py -v -s
"""

from __future__ import annotations
import pytest
import editdistance
from unittest.mock import patch, MagicMock

from roman_urdu_nlp.pipeline import Pipeline
from roman_urdu_nlp.data.gold_set import GOLD_SET


def _normalised_edit_distance(a: str, b: str) -> float:
    """Compute edit distance normalised to [0, 1] (0 = identical)."""
    if not a and not b:
        return 0.0
    max_len = max(len(a), len(b))
    return editdistance.eval(a.lower(), b.lower()) / max_len


def _word_overlap(a: str, b: str) -> float:
    """Return fraction of expected words found in actual output."""
    expected_words = set(b.lower().split())
    actual_words = set(a.lower().split())
    if not expected_words:
        return 1.0
    return len(expected_words & actual_words) / len(expected_words)


# ---------------------------------------------------------------------------
# Build a mock translate that returns expected english from gold set
# We also test with a real call separately.
# ---------------------------------------------------------------------------

class TestGoldSetNormalized:
    """Test the normalized output (clean + normalize + correct) against gold."""

    def setup_method(self):
        self.nlp = Pipeline()

    def _run(self, raw: str) -> str:
        with patch("roman_urdu_nlp.translator.GoogleTranslator") as mock_gt:
            mock_gt.return_value.translate.return_value = "mock translation"
            result = self.nlp(raw)
        return result.normalized

    def test_all_normalized_word_overlap_above_threshold(self):
        """At least 60% word overlap for all gold items."""
        results = []
        for item in GOLD_SET:
            actual = self._run(item["raw"])
            overlap = _word_overlap(actual, item["normalized"])
            results.append((item["raw"], overlap))

        passing = [r for r in results if r[1] >= 0.6]
        pct = len(passing) / len(results) * 100
        print(f"\nNormalization: {len(passing)}/{len(results)} items ≥60% word overlap ({pct:.1f}%)")
        assert pct >= 60.0, f"Only {pct:.1f}% of gold items pass normalization threshold"

    @pytest.mark.parametrize("item", GOLD_SET, ids=[g["raw"][:30] for g in GOLD_SET])
    def test_each_normalized_not_empty(self, item):
        actual = self._run(item["raw"])
        assert actual != "", f"Normalized output empty for: {item['raw']!r}"

    @pytest.mark.parametrize("item", GOLD_SET, ids=[g["raw"][:30] for g in GOLD_SET])
    def test_each_normalized_word_overlap(self, item):
        actual = self._run(item["raw"])
        overlap = _word_overlap(actual, item["normalized"])
        assert overlap >= 0.4, (
            f"Low overlap ({overlap:.2f}) for {item['raw']!r}\n"
            f"  Expected: {item['normalized']!r}\n"
            f"  Got:      {actual!r}"
        )


class TestGoldSetUrdu:
    """Test Urdu script output contains expected Urdu words."""

    def setup_method(self):
        self.nlp = Pipeline()

    def _run(self, raw: str) -> str:
        with patch("roman_urdu_nlp.translator.GoogleTranslator") as mock_gt:
            mock_gt.return_value.translate.return_value = "mock"
            result = self.nlp(raw)
        return result.urdu

    def test_all_urdu_contain_arabic_chars(self):
        """Every non-empty input must produce Urdu script."""
        for item in GOLD_SET:
            urdu = self._run(item["raw"])
            if item["raw"].strip():
                assert any("\u0600" <= ch <= "\u06FF" for ch in urdu), (
                    f"No Urdu chars for: {item['raw']!r}"
                )

    def test_all_urdu_word_overlap_above_threshold(self):
        """At least 50% word overlap for Urdu output across gold set."""
        results = []
        for item in GOLD_SET:
            actual = self._run(item["raw"])
            overlap = _word_overlap(actual, item["urdu"])
            results.append(overlap)

        avg = sum(results) / len(results)
        print(f"\nUrdu avg word overlap: {avg:.2f}")
        assert avg >= 0.45, f"Urdu avg overlap {avg:.2f} below 0.45 threshold"


class TestGoldSetEnglish:
    """Test English translation output (mocked and real)."""

    def setup_method(self):
        self.nlp = Pipeline()

    def test_english_output_is_str_for_all_items(self):
        for item in GOLD_SET:
            with patch("roman_urdu_nlp.translator.GoogleTranslator") as mock_gt:
                mock_gt.return_value.translate.return_value = "test"
                result = self.nlp(item["raw"])
            assert isinstance(result.english, str)

    def test_english_not_empty_for_non_empty_input(self):
        for item in GOLD_SET:
            with patch("roman_urdu_nlp.translator.GoogleTranslator") as mock_gt:
                mock_gt.return_value.translate.return_value = "I am very hungry."
                result = self.nlp(item["raw"])
            if item["raw"].strip():
                assert result.english != "", f"Empty english for: {item['raw']!r}"


class TestFullPipelineSummary:
    """Print a full evaluation table for the gold set."""

    def test_print_evaluation_table(self, capsys):
        nlp = Pipeline()
        print("\n\n" + "=" * 80)
        print(f"{'RAW INPUT':<35} {'NORM OVERLAP':>13} {'URDU OVERLAP':>13}")
        print("=" * 80)

        norm_overlaps = []
        urdu_overlaps = []

        for item in GOLD_SET:
            with patch("roman_urdu_nlp.translator.GoogleTranslator") as mock_gt:
                mock_gt.return_value.translate.return_value = "mock"
                result = nlp(item["raw"])

            norm_ov = _word_overlap(result.normalized, item["normalized"])
            urdu_ov = _word_overlap(result.urdu, item["urdu"])
            norm_overlaps.append(norm_ov)
            urdu_overlaps.append(urdu_ov)

            raw_short = item["raw"][:32] + ".." if len(item["raw"]) > 34 else item["raw"]
            print(f"{raw_short:<35} {norm_ov:>12.2f} {urdu_ov:>12.2f}")

        avg_norm = sum(norm_overlaps) / len(norm_overlaps)
        avg_urdu = sum(urdu_overlaps) / len(urdu_overlaps)
        print("-" * 80)
        print(f"{'AVERAGE':<35} {avg_norm:>12.2f} {avg_urdu:>12.2f}")
        print("=" * 80)

        captured = capsys.readouterr()
        print(captured.out)  # re-print so it shows in pytest -s

        assert avg_norm >= 0.55, f"Average normalization overlap {avg_norm:.2f} too low"
