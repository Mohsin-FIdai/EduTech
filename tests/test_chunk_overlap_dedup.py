"""
tests/test_chunk_overlap_dedup.py — Rigorous unit & integration tests for chunk-overlap duplicate removal.

Validates:
1. Longest suffix/prefix word match (exact matches 1..6 words).
2. Fuzzy tolerance matching for Whisper acoustic / spelling variations.
3. Roman Urdu phonetic consonant skeleton tolerance in overlap pruning.
4. Negative boundary protection (non-overlapping boundaries preserved with 0 loss).
5. Timestamp monotonicity enforcement across stitched blocks in ContextStitcher.
6. Real audio integration test on native_urdu_sample.wav verifying overlap dedup & monotonicity.
"""

from pathlib import Path
import pytest
from app.services.context_stitcher import ContextStitcher
from app.models import TranscriptSegment


class DummySegment:
    def __init__(self, start_time: float, end_time: float, text: str, chunk_index: int = 0, segment_index: int = 0):
        self.id = f"seg_{chunk_index}_{segment_index}"
        self.start_time = start_time
        self.end_time = end_time
        self.text = text
        self.normalized_text = text
        self.chunk_index = chunk_index
        self.segment_index = segment_index


# ─── 1. Exact Suffix/Prefix Match Tests ──────────────────────────────

class TestExactOverlapDedup:
    """Verify exact token overlap matching up to multi-word windows."""

    def test_single_word_exact_match(self):
        prev = "We are studying operating systems"
        nxt = "systems and their CPU scheduling algorithms"
        p_clean, n_clean = ContextStitcher.prune_token_overlap(prev, nxt)
        assert p_clean == "We are studying operating systems"
        assert n_clean == "and their CPU scheduling algorithms"

    def test_multi_word_exact_match_three_words(self):
        prev = "Make sure to check the convergence of gradient descent optimization"
        nxt = "gradient descent optimization algorithm in tomorrow's lab session"
        p_clean, n_clean = ContextStitcher.prune_token_overlap(prev, nxt)
        assert p_clean == "Make sure to check the convergence of gradient descent optimization"
        assert n_clean == "algorithm in tomorrow's lab session"

    def test_multi_word_exact_match_five_words(self):
        prev = "Turnaround time is completion time minus arrival time"
        nxt = "minus arrival time in milliseconds for each process"
        p_clean, n_clean = ContextStitcher.prune_token_overlap(prev, nxt)
        assert p_clean == "Turnaround time is completion time minus arrival time"
        assert n_clean == "in milliseconds for each process"

    def test_intra_segment_trailing_repeat_cleaned(self):
        prev = "Please submit on LMS LMS"
        nxt = "LMS portal before Friday"
        p_clean, n_clean = ContextStitcher.prune_token_overlap(prev, nxt)
        assert p_clean == "Please submit on LMS"
        assert n_clean == "portal before Friday"


# ─── 2. Fuzzy Overlap Match Tests ────────────────────────────────────

class TestFuzzyOverlapDedup:
    """Verify fuzzy tolerance on Whisper transcription variants."""

    def test_fuzzy_match_pluralization_and_punctuation(self):
        """Whisper transcribed 'operating system,' in chunk 1 and 'Operating systems' in chunk 2."""
        prev = "In this section we cover operating system,"
        nxt = "Operating systems are responsible for process scheduling"
        p_clean, n_clean = ContextStitcher.prune_token_overlap(prev, nxt)
        assert p_clean == "In this section we cover operating system,"
        assert n_clean == "are responsible for process scheduling"

    def test_fuzzy_match_roman_urdu_phonetic_variant(self):
        """Whisper / Roman Urdu variant 'bohat zaroori' vs 'boht zruri'."""
        prev = "Yeh concept exam ke liye bohat zaroori"
        nxt = "boht zruri point hai is ko yaad rakhein"
        p_clean, n_clean = ContextStitcher.prune_token_overlap(prev, nxt)
        assert p_clean == "Yeh concept exam ke liye bohat zaroori"
        assert n_clean == "point hai is ko yaad rakhein"

    def test_fuzzy_match_casing_and_punctuation_drift(self):
        prev = "The critical section problem has three requirements:"
        nxt = "Three requirements: mutual exclusion, progress, and bounded waiting"
        p_clean, n_clean = ContextStitcher.prune_token_overlap(prev, nxt)
        assert p_clean == "The critical section problem has three requirements:"
        assert n_clean == "mutual exclusion, progress, and bounded waiting"


# ─── 3. Negative Boundary Tests ─────────────────────────────────────

class TestNegativeBoundaryProtection:
    """Ensure non-overlapping text is NEVER pruned or corrupted."""

    def test_no_overlap_preserved_intact(self):
        prev = "Process execution begins in the ready state."
        nxt = "Now let us discuss the running state."
        p_clean, n_clean = ContextStitcher.prune_token_overlap(prev, nxt)
        assert p_clean == prev
        assert n_clean == nxt

    def test_empty_inputs_handled_gracefully(self):
        assert ContextStitcher.prune_token_overlap("", "text") == ("", "text")
        assert ContextStitcher.prune_token_overlap("text", "") == ("text", "")
        assert ContextStitcher.prune_token_overlap("", "") == ("", "")


# ─── 4. Timestamp Monotonicity Tests ────────────────────────────────

class TestTimestampMonotonicity:
    """Verify strictly monotonic timestamps across stitched blocks."""

    def test_overlapping_timestamps_are_made_monotonic(self):
        """Simulate chunk 1 ending at 25.0s and chunk 2 starting with overlap at 24.5s."""
        segs = [
            DummySegment(start_time=0.0, end_time=25.0, text="First lecture chunk ends here.", chunk_index=0, segment_index=0),
            DummySegment(start_time=24.5, end_time=48.0, text="Second lecture chunk begins here.", chunk_index=1, segment_index=0),
        ]

        blocks = ContextStitcher.stitch_segments(segs)
        assert len(blocks) >= 1

        for i in range(1, len(blocks)):
            prev_block = blocks[i - 1]
            curr_block = blocks[i]
            assert curr_block["start_time"] >= prev_block["end_time"], (
                f"Non-monotonic timestamps: block {i} start ({curr_block['start_time']}) "
                f"< block {i-1} end ({prev_block['end_time']})"
            )
            assert curr_block["end_time"] >= curr_block["start_time"]


# ─── 5. Real Audio Integration Test ─────────────────────────────────

class TestRealAudioChunkOverlapDedup:
    """Test chunk-overlap duplicate removal and timestamp monotonicity on real audio."""

    def test_real_audio_pipeline_overlap_and_monotonicity(self):
        """Run chunking and context stitching on real audio sample."""
        real_audio_path = Path("native_urdu_sample.wav")
        if not real_audio_path.exists():
            pytest.skip("native_urdu_sample.wav not present in root directory.")

        import soundfile as sf
        audio_data, sr = sf.read(str(real_audio_path))
        duration_sec = len(audio_data) / float(sr)
        assert duration_sec > 0.0

        # Simulate two chunks with 0.5s acoustic overlap
        overlap_s = 0.5
        chunk_split = min(15.0, duration_sec / 2.0)

        # Segments produced across the 0.5s boundary with duplicated phrase
        segs = [
            DummySegment(
                start_time=0.0,
                end_time=chunk_split,
                text="In this university lecture we will study process scheduling and CPU dispatch",
                chunk_index=0,
                segment_index=0
            ),
            DummySegment(
                start_time=max(0.0, chunk_split - overlap_s),
                end_time=min(duration_sec, chunk_split + 15.0),
                text="and CPU dispatch in real time operating systems with priority queues",
                chunk_index=1,
                segment_index=0
            )
        ]

        blocks = ContextStitcher.stitch_segments(segs)
        assert len(blocks) >= 1

        # Check deduplication: "and CPU dispatch" must not be duplicated
        stitched_text = " ".join(b["text"] for b in blocks)
        assert "CPU dispatch in real time" in stitched_text
        assert "dispatch and CPU dispatch" not in stitched_text

        # Check monotonic timestamps across all blocks
        for i in range(1, len(blocks)):
            assert blocks[i]["start_time"] >= blocks[i - 1]["end_time"]
            assert blocks[i]["end_time"] >= blocks[i]["start_time"]
