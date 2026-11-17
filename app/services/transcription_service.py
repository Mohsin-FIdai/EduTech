"""
Transcription Service — Speech-to-Text with OpenAI Whisper, Global Timeline Synchronization,
Language Detection, and Roman Urdu Normalization.

Key Capabilities:
1. Thread-safe cached Whisper model manager.
2. Sequential chunk transcription with global timeline projection:
   t_global = chunk.start_time + t_segment
3. Language detection & summary aggregation (en: X%, ur: Y%).
4. Component 0 Path A: Perso-Arabic Urdu -> Roman Urdu transliteration -> roman_urdu_nlp normalization.
5. Auto-chunking fallback if recording is not yet in CHUNKED status.
6. Database persistence (Transcript + TranscriptSegment models).
7. Status transition to TRANSCRIBED.
"""

from typing import Optional, Any
from pathlib import Path
from dataclasses import dataclass, field
import json
import logging
import threading
import numpy as np
import whisper
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

from app.config import (
    DEFAULT_WHISPER_MODEL,
    WHISPER_DEVICE,
    SUPPORTED_WHISPER_MODELS,
    WHISPER_NO_SPEECH_THRESHOLD,
    WHISPER_COMPRESSION_RATIO_THRESHOLD,
    WHISPER_LOGPROB_THRESHOLD,
    WHISPER_TEMPERATURE_FALLBACK,
    WHISPER_CONDITION_ON_PREVIOUS_TEXT,
)
from app.models import Recording, Chunk, Transcript, TranscriptSegment, TranscriptASRAudit, RecordingStatus
from app.services.audio_chunker import AudioChunker
from app.services.urdu_transliteration import UrduToRomanTransliterator
from roman_urdu_nlp import Pipeline as RomanUrduPipeline
from app.services.domain_corrector import (
    WHISPER_PROMPT_MAX_TOKENS,
    correct_domain_terms,
    get_whisper_prompt,
    is_hallucinated_noise,
)
from app.services.contextual_transcription import (
    ContextualTranscriptionRefiner, refine_transcript_segments, is_noise_segment
)
from app.services.context_stitcher import ContextStitcher


import os
import imageio_ffmpeg
import soundfile as sf

# Ensure imageio_ffmpeg directory is in PATH for any internal Whisper subprocesses
try:
    _ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    _ffmpeg_dir = os.path.dirname(_ffmpeg_exe)
    if _ffmpeg_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
except Exception:
    pass


class TranscriptionException(Exception):
    """Exception raised for errors during transcription."""
    def __init__(self, message: str, details: dict | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class WhisperModelManager:
    """Thread-safe singleton model cache for Whisper models."""
    _models: dict[str, whisper.Whisper] = {}
    _lock = threading.Lock()

    @classmethod
    def get_model(cls, model_name: str = DEFAULT_WHISPER_MODEL, device: str = WHISPER_DEVICE) -> whisper.Whisper:
        import torch
        effective_device = device
        if str(effective_device).startswith("cuda") and not torch.cuda.is_available():
            effective_device = "cpu"
        with cls._lock:
            cache_key = f"{model_name}_{effective_device}"
            if cache_key not in cls._models:
                logger.info(f"Loading Whisper model '{model_name}' on device '{effective_device}'")
                cls._models[cache_key] = whisper.load_model(model_name, device=effective_device)
            return cls._models[cache_key]


@dataclass
class SegmentItem:
    chunk_index: int
    segment_index: int
    start_time: float
    end_time: float
    text: str
    language: str
    confidence: float
    no_speech_prob: float
    raw_text: Optional[str] = None
    chunk_id: Optional[str] = None
    urdu_text: Optional[str] = None
    english_text: Optional[str] = None
    translation_source: Optional[str] = None
    translation_confidence: Optional[float] = None


@dataclass
class TranscriptionResult:
    recording_id: str
    transcript_id: str
    original_transcript: str
    normalized_transcript: str
    language_summary: str
    total_words: int
    total_segments: int
    segments: list[SegmentItem]
    urdu_transcript: Optional[str] = None
    english_transcript: Optional[str] = None
    raw_whisper_chunks: list[dict[str, Any]] = field(default_factory=list)
    rejected_segments: list[dict[str, Any]] = field(default_factory=list)


class TranscriptionService:
    _nlp_pipeline = RomanUrduPipeline()

    @staticmethod
    def scrub_repetition_loops(text: str) -> str:
        """
        Scrub degenerate ASR repetition loops (e.g. repeated words or character cycles).
        """
        if not text:
            return ""
        import re
        # 1. Scrub repeated words/phrases repeated 3+ times (1-word up to 16-word phrases)
        cleaned = re.sub(r'(\b[\w\.,]+\b(?:\s+[\w\.,]+){0,15})(?:\s+\1){2,}', r'\1', text, flags=re.IGNORECASE)
        # 2. Scrub repeated sentence-level cycles separated by punctuation
        cleaned = re.sub(r'([^.!?]{8,120}[.!?])(?:\s*\1){2,}', r'\1', cleaned, flags=re.IGNORECASE)
        # 3. Scrub degenerate repeated character sub-patterns of length 2-8 (e.g. 'cgcgcgcgcg' -> 'cg')
        cleaned = re.sub(r'(.{2,8}?)\1{3,}', r'\1', cleaned)
        # 3. Clean up excessive consecutive dots
        cleaned = re.sub(r'\.{2,}', '.', cleaned)
        cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip()
        return cleaned

    @staticmethod
    def build_effective_initial_prompt(
        topic_hint: Optional[str],
        model: Optional[Any] = None,
        model_size: str = DEFAULT_WHISPER_MODEL,
    ) -> str:
        """
        Combine general technical vocabulary with any dynamic topic hint while respecting
        Whisper's prompt-token limit (n_text_ctx // 2 - 1 = 223 tokens for standard models).
        """
        n_text_ctx = getattr(getattr(model, "dims", None), "n_text_ctx", None)
        max_tokens = (int(n_text_ctx) // 2 - 1) if isinstance(n_text_ctx, int) and n_text_ctx > 2 else WHISPER_PROMPT_MAX_TOKENS
        is_multilingual = getattr(model, "is_multilingual", None)
        if is_multilingual is None:
            is_multilingual = not str(model_size or "").endswith(".en")
        return get_whisper_prompt(
            topic_hint=topic_hint,
            max_tokens=max_tokens,
            multilingual=bool(is_multilingual),
        )

    @staticmethod
    def _snapshot_raw_whisper_chunk(
        chunk: dict[str, Any],
        whisper_result: dict[str, Any],
        raw_segments: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Create an unmodified audit snapshot of a chunk's raw Whisper output."""
        chunk_start = float(chunk["start_time"])
        chunk_end = float(chunk["end_time"])
        segment_snapshots: list[dict[str, Any]] = []
        for idx, seg in enumerate(raw_segments):
            local_start = float(seg.get("start", 0.0) or 0.0)
            local_end = float(seg.get("end", 0.0) or 0.0)
            segment_snapshots.append({
                "raw_segment_index": idx,
                "whisper_id": seg.get("id"),
                "start": round(local_start, 3),
                "end": round(local_end, 3),
                "global_start_time": round(chunk_start + local_start, 3),
                "global_end_time": round(chunk_start + local_end, 3),
                "raw_text": str(seg.get("text", "")),
                "avg_logprob": seg.get("avg_logprob"),
                "no_speech_prob": seg.get("no_speech_prob"),
                "compression_ratio": seg.get("compression_ratio"),
                "temperature": seg.get("temperature"),
            })
        return {
            "chunk_id": str(chunk["id"]),
            "chunk_index": int(chunk["chunk_index"]),
            "chunk_start_time": round(chunk_start, 3),
            "chunk_end_time": round(chunk_end, 3),
            "detected_language": whisper_result.get("language"),
            "raw_text": str(whisper_result.get("text", "")),
            "segment_count": len(segment_snapshots),
            "segments": segment_snapshots,
        }

    @staticmethod
    def _build_rejected_record(
        *,
        chunk_id: str,
        chunk_index: int,
        start_time: float,
        end_time: float,
        raw_text: Optional[str],
        rejection_reason: str,
        processed_text: Optional[str] = None,
        seg: Optional[dict[str, Any]] = None,
        segment_index: Optional[int] = None,
    ) -> dict[str, Any]:
        """Build a standardized audit record for a rejected or pruned ASR segment."""
        seg_meta = seg or {}
        return {
            "chunk_id": str(chunk_id),
            "chunk_index": int(chunk_index),
            "segment_index": segment_index,
            "start_time": round(float(start_time), 3),
            "end_time": round(float(end_time), 3),
            "raw_text": raw_text,
            "processed_text": processed_text,
            "rejection_reason": rejection_reason,
            "avg_logprob": seg_meta.get("avg_logprob"),
            "no_speech_prob": seg_meta.get("no_speech_prob"),
            "compression_ratio": seg_meta.get("compression_ratio"),
            "temperature": seg_meta.get("temperature"),
        }

    @staticmethod
    def _ensure_asr_audit_table(db: Session) -> None:
        """Create transcript_asr_audits on live SQLAlchemy binds if missing."""
        get_bind = getattr(db, "get_bind", None)
        if not callable(get_bind):
            return
        bind = get_bind()
        if bind is not None and hasattr(bind, "dialect"):
            TranscriptASRAudit.__table__.create(bind=bind, checkfirst=True)

    @classmethod
    def transcribe_recording(
        cls,
        recording_id: str,
        db: Session,
        model_size: str = DEFAULT_WHISPER_MODEL,
        language: str | None = None,
        temperature: float = 0.0,
        initial_prompt: Optional[str] = None
    ) -> TranscriptionResult:
        """
        Transcribe all chunks of a recording using Whisper:
        1. Ensure recording is chunked (trigger chunking if needed).
        2. Set status to TRANSCRIBING.
        3. Transcribe each chunk and map segment timestamps to global timeline.
        4. Transliterate Urdu script to Roman Urdu and normalize via NLP pipeline.
        5. Persist Transcript and TranscriptSegment rows in DB.
        6. Set status to TRANSCRIBED.
        """
        recording = db.query(Recording).filter(Recording.id == recording_id).first()
        if not recording:
            raise TranscriptionException(f"Recording '{recording_id}' not found.")

        # Auto-chunk ONLY if not already chunked
        existing_chunks = db.query(Chunk).filter(Chunk.recording_id == recording_id).order_by(Chunk.chunk_index.asc()).all()
        if not existing_chunks:
            AudioChunker.split_recording(recording_id, db)
            db.refresh(recording)
            existing_chunks = db.query(Chunk).filter(Chunk.recording_id == recording_id).order_by(Chunk.chunk_index.asc()).all()

        if not existing_chunks:
            raise TranscriptionException(f"No audio chunks found for recording '{recording_id}'.")

        # Snapshot chunk data as immutable dicts before committing/cleaning transcripts to prevent ObjectDeletedError
        chunks_info = [
            {
                "id": c.id,
                "chunk_index": int(c.chunk_index),
                "start_time": float(c.start_time),
                "end_time": float(c.end_time),
                "duration": float(c.duration),
                "file_path": str(c.file_path)
            }
            for c in existing_chunks
        ]

        # Update status to TRANSCRIBING
        recording.status = RecordingStatus.TRANSCRIBING
        db.commit()

        # Clean previous transcripts if re-transcribing
        cls.cleanup_transcripts(recording_id, db)

        # Validate model
        if model_size not in SUPPORTED_WHISPER_MODELS:
            model_size = DEFAULT_WHISPER_MODEL

        model = WhisperModelManager.get_model(model_size, device=WHISPER_DEVICE)

        all_segments: list[SegmentItem] = []
        raw_whisper_chunks: list[dict[str, Any]] = []
        rejected_segments: list[dict[str, Any]] = []
        global_seg_idx = 0
        lang_counts: dict[str, int] = {}

        # Combine general domain vocabulary prompt with any explicit/upload topic hint
        topic_hint = (initial_prompt or getattr(recording, "topic", None) or "").strip()
        effective_initial_prompt = cls.build_effective_initial_prompt(
            topic_hint,
            model=model,
            model_size=model_size,
        )
        rec_duration = float(getattr(recording, "duration_seconds", 0.0) or 0.0)

        for chunk in chunks_info:
            chunk_file = Path(chunk["file_path"])
            if not chunk_file.exists():
                raise TranscriptionException(f"Chunk audio file '{chunk['file_path']}' missing from storage.")

            # Load audio as float32 16kHz numpy array
            audio_data, sr = sf.read(str(chunk_file), dtype="float32")
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)

            transcribe_kwargs = {
                "audio": audio_data,
                "temperature": WHISPER_TEMPERATURE_FALLBACK,
                "condition_on_previous_text": WHISPER_CONDITION_ON_PREVIOUS_TEXT,
                "compression_ratio_threshold": WHISPER_COMPRESSION_RATIO_THRESHOLD,
                "logprob_threshold": WHISPER_LOGPROB_THRESHOLD,
                "no_speech_threshold": WHISPER_NO_SPEECH_THRESHOLD,
                "fp16": str(getattr(model, "device", WHISPER_DEVICE)).startswith("cuda"),
                "word_timestamps": False,
                "initial_prompt": effective_initial_prompt, # Context-conditioned vocabulary biasing
            }
            if language:
                transcribe_kwargs["language"] = language

            whisper_result = model.transcribe(**transcribe_kwargs)
            detected_chunk_lang = whisper_result.get("language", "en")

            raw_segments = list(whisper_result.get("segments") or [])
            # Preserve exact unmodified Whisper chunk output before any cleaning or filtering
            raw_whisper_chunks.append(
                cls._snapshot_raw_whisper_chunk(chunk, whisper_result, raw_segments)
            )

            if not raw_segments and whisper_result.get("text", "").strip():
                # Fallback if text returned without segment breakdown
                raw_segments = [{
                    "start": 0.0,
                    "end": chunk["duration"],
                    "text": whisper_result["text"],
                    "avg_logprob": 0.0,
                    "no_speech_prob": 0.0
                }]
            elif not raw_segments:
                # Whisper skipped the chunk/window (e.g. no_speech_threshold or empty decode)
                rejected_record = cls._build_rejected_record(
                    chunk_id=str(chunk["id"]),
                    chunk_index=chunk["chunk_index"],
                    start_time=chunk["start_time"],
                    end_time=chunk["end_time"],
                    raw_text=whisper_result.get("text") or None,
                    rejection_reason="whisper_no_speech_or_empty",
                )
                rejected_segments.append(rejected_record)
                logger.info(
                    "Whisper produced no segments for chunk_index=%s chunk_id=%s window=%.3f..%.3fs reason=whisper_no_speech_or_empty",
                    chunk["chunk_index"],
                    chunk["id"],
                    chunk["start_time"],
                    chunk["end_time"],
                )
                continue

            for seg in raw_segments:
                raw_seg_text = str(seg.get("text", ""))
                seg_global_start = round(chunk["start_time"] + float(seg.get("start", 0.0) or 0.0), 3)
                seg_global_end = round(chunk["start_time"] + float(seg.get("end", 0.0) or 0.0), 3)

                seg_text = cls.scrub_repetition_loops(raw_seg_text.strip())
                if is_hallucinated_noise(seg_text):
                    rejected_segments.append(cls._build_rejected_record(
                        chunk_id=str(chunk["id"]),
                        chunk_index=chunk["chunk_index"],
                        start_time=seg_global_start,
                        end_time=seg_global_end,
                        raw_text=raw_seg_text,
                        processed_text=seg_text,
                        rejection_reason="hallucinated_noise",
                        seg=seg,
                    ))
                    continue
                # Apply context-aware domain vocabulary correction
                seg_text = correct_domain_terms(seg_text)
                if not seg_text or len(seg_text) < 2:
                    rejected_segments.append(cls._build_rejected_record(
                        chunk_id=str(chunk["id"]),
                        chunk_index=chunk["chunk_index"],
                        start_time=seg_global_start,
                        end_time=seg_global_end,
                        raw_text=raw_seg_text,
                        processed_text=seg_text,
                        rejection_reason="empty_or_too_short_after_scrub",
                        seg=seg,
                    ))
                    continue

                # Deduplicate boundary tokens across chunk transitions (caused by 0.5s overlap)
                if all_segments and chunk["chunk_index"] > all_segments[-1].chunk_index:
                    prev_clean, next_clean = ContextStitcher.prune_token_overlap(all_segments[-1].text, seg_text)
                    all_segments[-1].text = prev_clean
                    seg_text = next_clean
                    if not seg_text or len(seg_text.strip()) < 2:
                        rejected_segments.append(cls._build_rejected_record(
                            chunk_id=str(chunk["id"]),
                            chunk_index=chunk["chunk_index"],
                            start_time=seg_global_start,
                            end_time=seg_global_end,
                            raw_text=raw_seg_text,
                            processed_text=seg_text,
                            rejection_reason="stitch_overlap_pruned",
                            seg=seg,
                        ))
                        continue

                g_start = seg_global_start
                g_end = seg_global_end

                # Monotonic timestamp enforcement across chunks and segments
                if all_segments:
                    prev_end = all_segments[-1].end_time
                    if g_start < prev_end:
                        g_start = round(prev_end, 3)
                    if g_end <= g_start:
                        g_end = round(g_start + max(0.05, float(seg["end"]) - float(seg["start"])), 3)

                # Cap end time to recording duration
                if rec_duration > 0 and g_end > rec_duration + 0.1:
                    g_end = round(rec_duration, 3)
                if g_start >= g_end:
                    g_start = max(0.0, round(g_end - 0.05, 3))

                avg_logprob = float(seg.get("avg_logprob", 0.0))
                confidence = round(float(np.exp(min(0.0, avg_logprob))), 4)
                no_speech_prob = round(float(seg.get("no_speech_prob", 0.0)), 4)

                seg_lang = detected_chunk_lang
                # Track language for distribution
                word_count = len(seg_text.split())
                lang_counts[seg_lang] = lang_counts.get(seg_lang, 0) + max(1, word_count)

                all_segments.append(SegmentItem(
                    chunk_index=chunk["chunk_index"],
                    segment_index=global_seg_idx,
                    start_time=g_start,
                    end_time=g_end,
                    text=seg_text,
                    language=seg_lang,
                    confidence=confidence,
                    no_speech_prob=no_speech_prob,
                    raw_text=raw_seg_text,
                    chunk_id=str(chunk["id"]),
                ))
                global_seg_idx += 1

        # Assemble original transcript
        original_transcript = " ".join([s.text for s in all_segments]).strip()

        # Component 0 Path A: Convert any Perso-Arabic Urdu to Roman Urdu and normalize
        # 1. Transliterate Urdu script tokens to Roman Urdu
        roman_urdu_intermediate = UrduToRomanTransliterator.transliterate(original_transcript)
        # 2. Run Roman Urdu NLP cleaner/normalizer/corrector pipeline (skip redundant online translation)
        nlp_res = cls._nlp_pipeline(roman_urdu_intermediate, skip_translation=True)
        normalized_transcript = nlp_res.normalized if nlp_res.normalized else roman_urdu_intermediate
        normalized_transcript = correct_domain_terms(normalized_transcript)

        # Compute Language Distribution Summary
        total_lang_words = sum(lang_counts.values())
        if total_lang_words > 0:
            lang_summary_parts = [
                f"{lang}: {round((count / total_lang_words) * 100)}%"
                for lang, count in sorted(lang_counts.items(), key=lambda x: x[1], reverse=True)
            ]
            language_summary = ", ".join(lang_summary_parts)
        else:
            language_summary = "en: 100%"

        total_words = len(original_transcript.split())

        # Persist to Database
        db_transcript = Transcript(
            recording_id=recording_id,
            original_transcript=original_transcript,
            normalized_transcript=normalized_transcript,
            language_summary=language_summary
        )
        db.add(db_transcript)
        db.flush()

        # --- Pass: Per-segment basic normalization ---
        seg_norm_map: dict[int, str] = {}  # segment_index -> basic normalized text
        for seg_item in all_segments:
            seg_raw_text = seg_item.text
            if UrduToRomanTransliterator.is_urdu_script(seg_raw_text):
                seg_raw_text = UrduToRomanTransliterator.transliterate(seg_raw_text)
            seg_nlp = cls._nlp_pipeline(seg_raw_text, skip_translation=True)
            seg_norm_text = seg_nlp.normalized if seg_nlp.normalized else seg_raw_text
            seg_norm_text = correct_domain_terms(seg_norm_text)
            seg_norm_map[seg_item.segment_index] = seg_norm_text

        # --- Pass: Context-aware semantic refinement ---
        refiner_input = [
            {
                "segment_index": seg_item.segment_index,
                "text": seg_norm_map.get(seg_item.segment_index, seg_item.text),
                "start_time": seg_item.start_time,
                "end_time": seg_item.end_time,
            }
            for seg_item in all_segments
        ]
        refined_results = refine_transcript_segments(
            refiner_input, full_transcript=original_transcript
        )
        refined_map: dict[int, str] = {}
        pruned_indices: set[int] = set()
        prune_reasons: dict[int, str] = {}
        for r in refined_results:
            if r.was_pruned:
                pruned_indices.add(r.segment_index)
                prune_reasons[r.segment_index] = ", ".join(r.changes_made or []) or "contextual_refiner_pruned"
            else:
                refined_map[r.segment_index] = r.refined_text

        created_segments = []
        for seg_item in all_segments:
            if seg_item.segment_index in pruned_indices:
                rejected_segments.append(cls._build_rejected_record(
                    chunk_id=seg_item.chunk_id or "",
                    chunk_index=seg_item.chunk_index,
                    segment_index=seg_item.segment_index,
                    start_time=seg_item.start_time,
                    end_time=seg_item.end_time,
                    raw_text=seg_item.raw_text,
                    processed_text=seg_item.text,
                    rejection_reason=f"contextual_refiner_pruned:{prune_reasons.get(seg_item.segment_index, '')}".rstrip(":"),
                ))
                continue  # Skip noise/hallucinated segments

            final_norm = refined_map.get(
                seg_item.segment_index,
                seg_norm_map.get(seg_item.segment_index, seg_item.text)
            )

            db_seg = TranscriptSegment(
                transcript_id=db_transcript.id,
                recording_id=recording_id,
                chunk_index=seg_item.chunk_index,
                segment_index=seg_item.segment_index,
                start_time=seg_item.start_time,
                end_time=seg_item.end_time,
                text=seg_item.text,
                normalized_text=final_norm,
                language=seg_item.language,
                confidence=seg_item.confidence,
                no_speech_prob=seg_item.no_speech_prob
            )
            db.add(db_seg)
            created_segments.append(db_seg)

        # Batch translate segments with multi-tier fallback & provenance
        from app.services.text_processor import TextProcessingService
        TextProcessingService.translate_segments_batch(created_segments)

        full_urdu = " ".join([s.urdu_text for s in created_segments if s.urdu_text]).strip()
        full_english = " ".join([s.english_text for s in created_segments if s.english_text]).strip()
        db_transcript.urdu_transcript = full_urdu
        db_transcript.english_transcript = full_english

        # Persist untouched raw Whisper chunk outputs and rejected/pruned segment audit trail
        cls._ensure_asr_audit_table(db)
        db.add(TranscriptASRAudit(
            transcript_id=db_transcript.id,
            recording_id=recording_id,
            raw_whisper_chunks_json=json.dumps(raw_whisper_chunks, ensure_ascii=False),
            rejected_segments_json=json.dumps(rejected_segments, ensure_ascii=False),
        ))

        recording.status = RecordingStatus.TRANSCRIBED
        db.commit()
        db.refresh(recording)
        db.refresh(db_transcript)

        created_by_idx = {s.segment_index: s for s in created_segments}
        for s_item in all_segments:
            created_seg = created_by_idx.get(s_item.segment_index)
            if created_seg is not None:
                s_item.urdu_text = created_seg.urdu_text
                s_item.english_text = created_seg.english_text
                s_item.translation_source = created_seg.translation_source
                s_item.translation_confidence = created_seg.translation_confidence

        return TranscriptionResult(
            recording_id=recording_id,
            transcript_id=db_transcript.id,
            original_transcript=original_transcript,
            normalized_transcript=normalized_transcript,
            language_summary=language_summary,
            total_words=total_words,
            total_segments=len(all_segments),
            segments=all_segments,
            urdu_transcript=full_urdu,
            english_transcript=full_english,
            raw_whisper_chunks=raw_whisper_chunks,
            rejected_segments=rejected_segments,
        )

    @classmethod
    def cleanup_transcripts(cls, recording_id: str, db: Session) -> None:
        """
        Clean up all downstream rows for a recording before re-transcribing.
        Order matters: delete leaves first, then parents, to satisfy FK constraints.
        Use synchronize_session='fetch' so SQLAlchemy's identity map stays in sync
        and the subsequent db.delete(transcript) flush doesn't violate FK checks.
        """
        from app.models import IntelligenceReport, SegmentEmbedding, StitchedBlock, EntityMention, Entity
        try:
            # 0. Raw Whisper / rejected-segment ASR audit (FK -> transcripts.id and recordings.id)
            get_bind = getattr(db, "get_bind", None)
            bind = get_bind() if callable(get_bind) else None
            has_audit_table = True
            if bind is not None and hasattr(bind, "dialect"):
                from sqlalchemy import inspect as sa_inspect
                has_audit_table = sa_inspect(bind).has_table(TranscriptASRAudit.__tablename__)
            if has_audit_table:
                db.query(TranscriptASRAudit).filter(
                    TranscriptASRAudit.recording_id == recording_id
                ).delete(synchronize_session="fetch")

            # 1. Embeddings (FK -> transcript_segments.id and recordings.id)
            db.query(SegmentEmbedding).filter(
                SegmentEmbedding.recording_id == recording_id
            ).delete(synchronize_session="fetch")

            # 2. Entity mentions (FK -> transcript_segments.id, entities.id, recordings.id)
            db.query(EntityMention).filter(
                EntityMention.recording_id == recording_id
            ).delete(synchronize_session="fetch")

            # 3. Entities (FK -> recordings.id)
            db.query(Entity).filter(
                Entity.recording_id == recording_id
            ).delete(synchronize_session="fetch")

            # 4. Intelligence reports (FK -> transcripts.id and recordings.id) — must go before transcripts
            db.query(IntelligenceReport).filter(
                IntelligenceReport.recording_id == recording_id
            ).delete(synchronize_session="fetch")

            # 5. Stitched blocks (FK -> recordings.id)
            db.query(StitchedBlock).filter(
                StitchedBlock.recording_id == recording_id
            ).delete(synchronize_session="fetch")

            # 6. Transcript segments (FK -> transcripts.id and recordings.id) — must go before transcripts
            db.query(TranscriptSegment).filter(
                TranscriptSegment.recording_id == recording_id
            ).delete(synchronize_session="fetch")

            # 7. Finally transcripts (all children cleared above)
            db.query(Transcript).filter(
                Transcript.recording_id == recording_id
            ).delete(synchronize_session="fetch")

            db.commit()
        except Exception:
            db.rollback()
            raise

    @classmethod
    def refine_existing_transcription(cls, recording_id: str, db: Session) -> dict:
        """
        Apply contextual semantic refinement to an already-transcribed recording.
        Updates normalized_text for each segment and normalized_transcript for the transcript.
        Does NOT re-run Whisper — only refines existing text via context-aware rules.

        Returns dict with summary: {refined_count, pruned_count, total_segments, changes}
        """
        import logging
        logger = logging.getLogger(__name__)

        transcript = db.query(Transcript).filter(
            Transcript.recording_id == recording_id
        ).first()
        if not transcript:
            raise TranscriptionException(f"No transcript found for recording '{recording_id}'.")

        segments = db.query(TranscriptSegment).filter(
            TranscriptSegment.recording_id == recording_id
        ).order_by(TranscriptSegment.segment_index.asc()).all()

        if not segments:
            raise TranscriptionException(f"No segments found for recording '{recording_id}'.")

        # Build refiner input from existing normalized_text (or raw text as fallback)
        refiner_input = [
            {
                "segment_index": seg.segment_index,
                "text": seg.normalized_text or seg.text,
                "start_time": seg.start_time,
                "end_time": seg.end_time,
            }
            for seg in segments
        ]

        full_text = transcript.original_transcript or ""
        refined_results = refine_transcript_segments(refiner_input, full_transcript=full_text)

        refined_count = 0
        pruned_count = 0
        changes_log = []

        for r in refined_results:
            # Find the matching DB segment
            matching_seg = next(
                (s for s in segments if s.segment_index == r.segment_index), None
            )
            if not matching_seg:
                continue

            if r.was_pruned:
                pruned_count += 1
                # Mark pruned segments with empty normalized_text
                if matching_seg.normalized_text:
                    old = matching_seg.normalized_text
                    matching_seg.normalized_text = ""
                    changes_log.append({
                        "segment_index": r.segment_index,
                        "action": "pruned",
                        "old": old,
                    })
            elif r.refined_text != (matching_seg.normalized_text or matching_seg.text):
                refined_count += 1
                old = matching_seg.normalized_text or matching_seg.text
                matching_seg.normalized_text = r.refined_text
                changes_log.append({
                    "segment_index": r.segment_index,
                    "action": "refined",
                    "old": old,
                    "new": r.refined_text,
                    "rules": r.changes_made,
                })

        # Rebuild normalized_transcript from non-pruned segments
        norm_parts = []
        for seg in segments:
            if seg.normalized_text:
                norm_parts.append(seg.normalized_text)
        transcript.normalized_transcript = " ".join(norm_parts)

        db.commit()

        summary = {
            "recording_id": recording_id,
            "total_segments": len(segments),
            "refined_count": refined_count,
            "pruned_count": pruned_count,
            "changes": changes_log[:50],  # Limit for response size
        }
        logger.info(
            f"Contextual refinement complete for {recording_id}: "
            f"{refined_count} refined, {pruned_count} pruned out of {len(segments)} segments."
        )
        return summary

    @classmethod
    def update_segment_correction(
        cls,
        recording_id: str,
        segment_id: str,
        db: Session,
        normalized_text: Optional[str] = None,
        text: Optional[str] = None,
        scope: str = "full"
    ) -> dict:
        """
        Manually correct a transcript segment, persist to database against exact
        chunk and timestamp, and automatically propagate to all downstream intelligence layers:
          1. Persist updated text and/or normalized_text
          2. Re-stitch parent Transcript full texts
          3. Downstream: Recompute vector embeddings via EmbeddingService (selective delta)
          4. Downstream: Re-run intelligence generation (summary, notes, assignments, quizzes)
             - scope='full': Full recording re-synthesis (100% global consistency)
             - scope='chunk': Fast localized re-extraction (sub-second)
          5. Grounded Q&A immediately reflects the updated report & segments
        """
        import json
        import logging
        from app.models import IntelligenceReport

        logger = logging.getLogger(__name__)

        # 1. Fetch Segment
        segment = db.query(TranscriptSegment).filter(
            TranscriptSegment.id == segment_id,
            TranscriptSegment.recording_id == recording_id
        ).first()

        # Fallback to segment_index match if segment_id looks like an index
        if not segment and segment_id.isdigit():
            seg_idx = int(segment_id)
            segment = db.query(TranscriptSegment).filter(
                TranscriptSegment.segment_index == seg_idx,
                TranscriptSegment.recording_id == recording_id
            ).first()

        if not segment:
            raise TranscriptionException(f"Segment '{segment_id}' not found for recording '{recording_id}'.")

        old_norm = segment.normalized_text
        old_raw = segment.text

        # 2. Persist Corrections
        if normalized_text is not None:
            segment.normalized_text = normalized_text.strip()
        if text is not None:
            segment.text = text.strip()

        db.flush()

        # 3. Update Parent Transcript
        transcript = db.query(Transcript).filter(Transcript.id == segment.transcript_id).first()
        if transcript:
            all_segs = db.query(TranscriptSegment).filter(
                TranscriptSegment.transcript_id == transcript.id
            ).order_by(TranscriptSegment.segment_index.asc()).all()

            norm_parts = [s.normalized_text for s in all_segs if s.normalized_text]
            transcript.normalized_transcript = " ".join(norm_parts)

            raw_parts = [s.text for s in all_segs if s.text]
            transcript.original_transcript = " ".join(raw_parts)
            db.flush()
        else:
            all_segs = [segment]

        db.commit()

        # 4. Downstream Step A: Selective Embedding Recomputation
        recomputed_embeddings = False
        try:
            from app.services.embedding_service import EmbeddingService
            EmbeddingService.generate_and_store_embeddings(recording_id, db, force_regenerate=False)
            recomputed_embeddings = True
        except Exception as e:
            logger.warning(f"Downstream embedding recomputation skipped or failed: {e}")

        # 5. Downstream Step B: Intelligence Recomputation (Summary, Notes, Assignments, Quizzes)
        from app.services.intelligence_service import LectureIntelligenceService

        intelligence_data = None
        if scope == "full":
            try:
                intelligence_data = LectureIntelligenceService.generate_intelligence_report(recording_id, db)
                logger.info(f"Full downstream intelligence successfully regenerated for recording '{recording_id}'.")
            except Exception as e:
                logger.error(f"Downstream intelligence generation error: {e}")
                raise
        else:
            # Chunk-scoped recomputation: Update assignments & action items without full document re-synthesis
            existing_report = db.query(IntelligenceReport).filter(IntelligenceReport.recording_id == recording_id).first()
            if existing_report and transcript:
                full_text = transcript.normalized_transcript or transcript.original_transcript
                assignments = LectureIntelligenceService.extract_assignments(full_text, all_segs)
                action_items = LectureIntelligenceService.extract_action_items(full_text, all_segs)
                existing_report.assignments_json = json.dumps(assignments)
                existing_report.action_items_json = json.dumps(action_items)
                db.commit()
                db.refresh(existing_report)

                notes = json.loads(existing_report.lecture_notes_json) if existing_report.lecture_notes_json else {}
                quizzes = json.loads(existing_report.quizzes_json) if existing_report.quizzes_json else []

                created_at_str = (
                    existing_report.created_at.isoformat()
                    if hasattr(existing_report.created_at, "isoformat")
                    else str(existing_report.created_at)
                )

                intelligence_data = {
                    "status": "success",
                    "recording_id": recording_id,
                    "transcript_id": transcript.id,
                    "summary_short": existing_report.summary_short,
                    "summary_detailed": existing_report.summary_detailed,
                    "lecture_notes": notes,
                    "assignments": assignments,
                    "action_items": action_items,
                    "quizzes": quizzes,
                    "created_at": created_at_str
                }
                logger.info(f"Chunk-scoped downstream intelligence updated for recording '{recording_id}'.")

        return {
            "status": "success",
            "message": "Segment corrected and downstream intelligence propagated.",
            "recording_id": recording_id,
            "segment": {
                "id": segment.id,
                "segment_index": segment.segment_index,
                "chunk_index": segment.chunk_index,
                "start_time": segment.start_time,
                "end_time": segment.end_time,
                "text": segment.text,
                "normalized_text": segment.normalized_text,
                "old_normalized_text": old_norm,
                "old_text": old_raw
            },
            "scope": scope,
            "recomputed": {
                "embeddings": recomputed_embeddings,
                "summary": scope == "full",
                "lecture_notes": scope == "full",
                "assignments": True,
                "action_items": True,
                "quizzes": scope == "full",
                "qa_grounding": True
            },
            "intelligence": intelligence_data
        }
