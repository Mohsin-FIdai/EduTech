"""
app/services/context_stitcher.py — Cross-chunk sentence boundary healing, repetition pruning, and discourse block assembly.
"""

from __future__ import annotations
import re
import difflib
import logging
from typing import Optional
from sqlalchemy.orm import Session
from app.models import TranscriptSegment, StitchedBlock, Recording
from app.config import (
    STITCH_MAX_OVERLAP_WORDS,
    STITCH_TIME_GAP_THRESHOLD,
    STITCH_CLOSE_GAP_THRESHOLD,
    CHUNK_OVERLAP_FUZZY_THRESHOLD,
    CHUNK_OVERLAP_MAX_WORDS,
)

logger = logging.getLogger("roman_urdu_app.context_stitcher")

# Continuation markers (English + Roman Urdu)
CONTINUATION_ENDINGS = {
    # Prepositions / Connectors (English)
    "of", "to", "in", "for", "with", "on", "from", "into", "about", "between",
    "through", "using", "by", "under", "over", "after", "before", "during",
    # Conjunctions (English)
    "and", "or", "but", "because", "although", "while", "if", "that", "which", "where",
    # Auxiliary / Verb endings (English)
    "is", "are", "was", "were", "will", "would", "should", "could", "have", "has", "had",
    # Prepositions / Postpositions / Particles (Roman Urdu)
    "ka", "ki", "ke", "kay", "mein", "se", "par", "ko", "tak", "liye", "waala", "waali",
    # Conjunctions / Connectors (Roman Urdu)
    "aur", "ya", "lekin", "magar", "kyun", "kyunke", "kyun kay", "agar", "to", "toh",
    "taakay", "jab", "jabke", "bhi", "hi", "phir", "k"
}

TERMINAL_PUNCTUATION = {".", "!", "?", "۔", "؟"}


class ContextStitcher:
    """
    Stitches transcript segments across chunk boundaries to heal sentence splices,
    prune boundary overlap tokens, and produce coherent discourse blocks.
    """

    @staticmethod
    def _token_similarity(w1: str, w2: str) -> float:
        """Compute similarity between two words with punctuation stripped."""
        c1 = re.sub(r"[^\w]", "", w1.lower())
        c2 = re.sub(r"[^\w]", "", w2.lower())
        if not c1 or not c2:
            return 1.0 if c1 == c2 else 0.0
        if c1 == c2:
            return 1.0
        # Consonant skeleton match for Roman Urdu phonetic variants (e.g. bohat/boht, zaroori/zruri, hai/h)
        skel1 = re.sub(r"[aeiouy\W]", "", c1)
        skel2 = re.sub(r"[aeiouy\W]", "", c2)
        if len(skel1) >= 2 and skel1 == skel2:
            return 0.95
        return difflib.SequenceMatcher(None, c1, c2).ratio()

    @classmethod
    def prune_token_overlap(
        cls,
        prev_text: str,
        next_text: str,
        max_overlap_words: Optional[int] = None,
        fuzzy_threshold: Optional[float] = None
    ) -> tuple[str, str]:
        """
        Detects and prunes overlapping/repeated boundary tokens between the end of prev_text
        and the start of next_text using longest suffix/prefix match with fuzzy tolerance.
        Example: prev="submit on LMS LMS", next="LMS portal" -> prev="submit on LMS", next="portal"
        """
        prev_words = prev_text.strip().split()
        next_words = next_text.strip().split()

        if not prev_words or not next_words:
            return prev_text, next_text

        if max_overlap_words is None:
            max_overlap_words = CHUNK_OVERLAP_MAX_WORDS
        if fuzzy_threshold is None:
            fuzzy_threshold = CHUNK_OVERLAP_FUZZY_THRESHOLD

        # 1. Intra-segment trailing repetition in prev_text (e.g. "LMS LMS")
        if len(prev_words) >= 2:
            sim_intra = cls._token_similarity(prev_words[-1], prev_words[-2])
            if sim_intra >= fuzzy_threshold:
                # Check if next_words starts with the same word
                if cls._token_similarity(next_words[0], prev_words[-1]) >= fuzzy_threshold:
                    prev_words.pop()

        # 2. Inter-segment overlap matching: longest suffix/prefix match with fuzzy tolerance
        max_overlap = min(len(prev_words), len(next_words), max_overlap_words)
        best_k = 0

        for k in range(max_overlap, 0, -1):
            prev_tail = prev_words[-k:]
            next_head = next_words[:k]

            similarities = [cls._token_similarity(w1, w2) for w1, w2 in zip(prev_tail, next_head)]
            min_sim = min(similarities)
            avg_sim = sum(similarities) / k

            # Require every token in the window to meet fuzzy threshold or average to be high
            if min_sim >= fuzzy_threshold and avg_sim >= (fuzzy_threshold + 0.03):
                best_k = k
                break
            elif min_sim >= 0.95:  # Exact/near-exact match across all tokens
                best_k = k
                break

        if best_k > 0:
            next_words = next_words[best_k:]

        return " ".join(prev_words), " ".join(next_words)

    @classmethod
    def should_stitch(cls, prev_seg: TranscriptSegment, next_seg: TranscriptSegment) -> bool:
        """
        Determines whether next_seg should be stitched into the same block as prev_seg.
        Conditions:
        - Gap between prev_seg.end_time and next_seg.start_time is small (<= 3.0s)
        - prev_seg has no terminal punctuation OR ends with a continuation marker
        - next_seg starts with a lowercase letter, conjunction, or subclause
        """
        time_gap = next_seg.start_time - prev_seg.end_time
        if time_gap > STITCH_TIME_GAP_THRESHOLD:
            return False

        prev_raw = prev_seg.text.strip()
        next_raw = next_seg.text.strip()

        if not prev_raw or not next_raw:
            return False

        # Check terminal punctuation of previous segment
        ends_with_terminal = prev_raw[-1] in TERMINAL_PUNCTUATION

        # Check last word of previous segment
        prev_clean_words = re.findall(r"\b[\w'-]+\b", prev_raw.lower())
        last_word = prev_clean_words[-1] if prev_clean_words else ""

        # Check first word of next segment
        next_clean_words = re.findall(r"\b[\w'-]+\b", next_raw.lower())
        first_word = next_clean_words[0] if next_clean_words else ""
        starts_lower = next_raw[0].islower() if next_raw else False

        # If previous segment ends with an ellipsis or continuation marker
        if prev_raw.endswith("...") or prev_raw.endswith("…"):
            return True

        if last_word in CONTINUATION_ENDINGS:
            return True

        if not ends_with_terminal:
            # Missing terminal punctuation + starts with lowercase or continuation word
            if starts_lower or first_word in CONTINUATION_ENDINGS or time_gap <= STITCH_CLOSE_GAP_THRESHOLD:
                return True

        # Across chunk boundary with small time gap and incomplete syntax
        if prev_seg.chunk_index != next_seg.chunk_index and not ends_with_terminal:
            return True

        return False

    @classmethod
    def stitch_segments(cls, segments: list[TranscriptSegment]) -> list[dict]:
        """
        Groups transcript segments into stitched discourse blocks.
        Returns list of structured block dictionaries.
        """
        if not segments:
            return []

        sorted_segs = sorted(segments, key=lambda s: (s.chunk_index, s.segment_index, s.start_time))
        blocks: list[dict] = []

        current_block_segs: list[TranscriptSegment] = [sorted_segs[0]]

        for seg in sorted_segs[1:]:
            prev_seg = current_block_segs[-1]
            if cls.should_stitch(prev_seg, seg):
                current_block_segs.append(seg)
            else:
                # Close current block with monotonic timestamp enforcement
                block = cls._assemble_block(len(blocks), current_block_segs)
                if blocks:
                    prev_end = blocks[-1]["end_time"]
                    if block["start_time"] < prev_end:
                        block["start_time"] = round(prev_end, 2)
                        if block["end_time"] <= block["start_time"]:
                            block["end_time"] = round(block["start_time"] + 0.1, 2)
                        block["timestamp"] = f"[{block['start_time']:.2f}s - {block['end_time']:.2f}s]"
                blocks.append(block)
                current_block_segs = [seg]

        if current_block_segs:
            block = cls._assemble_block(len(blocks), current_block_segs)
            if blocks:
                prev_end = blocks[-1]["end_time"]
                if block["start_time"] < prev_end:
                    block["start_time"] = round(prev_end, 2)
                    if block["end_time"] <= block["start_time"]:
                        block["end_time"] = round(block["start_time"] + 0.1, 2)
                    block["timestamp"] = f"[{block['start_time']:.2f}s - {block['end_time']:.2f}s]"
            blocks.append(block)

        return blocks

    @classmethod
    def _assemble_block(cls, block_idx: int, segs: list[TranscriptSegment]) -> dict:
        """
        Assembles multiple segments into a single stitched block.
        """
        start_time = float(segs[0].start_time)
        end_time = float(max(s.end_time for s in segs))
        if end_time <= start_time:
            end_time = start_time + 0.1
        chunk_indices = sorted(list({s.chunk_index for s in segs}))
        chunk_str = ",".join(str(c) for c in chunk_indices)

        # Build raw text with overlap pruning
        text_parts = [segs[0].text.strip()]
        norm_parts = [(segs[0].normalized_text or segs[0].text).strip()]

        for s in segs[1:]:
            prev_t = text_parts[-1]
            next_t = s.text.strip()
            prev_clean, next_clean = cls.prune_token_overlap(prev_t, next_t)
            text_parts[-1] = prev_clean
            if next_clean:
                text_parts.append(next_clean)

            prev_n = norm_parts[-1]
            next_n = (s.normalized_text or s.text).strip()
            prev_n_clean, next_n_clean = cls.prune_token_overlap(prev_n, next_n)
            norm_parts[-1] = prev_n_clean
            if next_n_clean:
                norm_parts.append(next_n_clean)

        joined_text = " ".join(text_parts).strip()
        joined_norm = " ".join(norm_parts).strip()

        # Clean trailing duplicate punctuation/spaces
        joined_text = re.sub(r"\s+", " ", joined_text)
        joined_norm = re.sub(r"\s+", " ", joined_norm)

        # Ensure sensible capitalization if stitched from lowercase
        if joined_text and joined_text[0].islower():
            joined_text = joined_text[0].upper() + joined_text[1:]
        if joined_norm and joined_norm[0].islower():
            joined_norm = joined_norm[0].upper() + joined_norm[1:]

        return {
            "block_index": block_idx,
            "start_time": round(start_time, 2),
            "end_time": round(end_time, 2),
            "timestamp": f"[{start_time:.2f}s - {end_time:.2f}s]",
            "text": joined_text,
            "normalized_text": joined_norm,
            "source_chunk_indices": chunk_str,
            "segment_ids": [s.id for s in segs]
        }

    @classmethod
    def process_and_persist_stitched_blocks(cls, recording_id: str, db: Session) -> list[dict]:
        """
        Fetches segments for recording_id, stitches them, and persists StitchedBlock records.
        """
        segments = (
            db.query(TranscriptSegment)
            .filter(TranscriptSegment.recording_id == recording_id)
            .order_by(TranscriptSegment.chunk_index.asc(), TranscriptSegment.segment_index.asc())
            .all()
        )

        if not segments:
            logger.warning(f"No segments found to stitch for recording '{recording_id}'.")
            return []

        # Delete existing stitched blocks for idempotent re-runs
        db.query(StitchedBlock).filter(StitchedBlock.recording_id == recording_id).delete()

        blocks_data = cls.stitch_segments(segments)
        for b in blocks_data:
            block_record = StitchedBlock(
                recording_id=recording_id,
                block_index=b["block_index"],
                start_time=b["start_time"],
                end_time=b["end_time"],
                text=b["text"],
                normalized_text=b["normalized_text"],
                source_chunk_indices=b["source_chunk_indices"]
            )
            db.add(block_record)

        db.commit()
        logger.info(f"Persisted {len(blocks_data)} stitched blocks for recording '{recording_id}'.")
        return blocks_data
