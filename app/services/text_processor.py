"""
text_processor.py — Text Processing & Correction Service for Pakistani English / Roman Urdu.

Key Capabilities:
1. Orchestration of roman_urdu_nlp pipeline (cleaning, token tagging, normalization, correction, transliteration, translation).
2. Context-aware homophone disambiguation and academic/technical lexicon preservation.
3. Batch enhancement and persistence of recording transcript segments in the database.
4. Token-level language and technical metadata extraction.
"""

from __future__ import annotations

import logging
import re
from typing import Any
from sqlalchemy.orm import Session

from app import config
from app.config import (
    ENGLISH_PASSTHROUGH_RATIO,
    TRANSLATION_CONFIDENCE_GOOGLE,
    TRANSLATION_CONFIDENCE_LLM,
    TRANSLATION_CONFIDENCE_DICTIONARY,
)
from app.models import Recording, Transcript, TranscriptSegment
from app.services.urdu_transliteration import UrduToRomanTransliterator
from roman_urdu_nlp import Pipeline as RomanUrduPipeline
from roman_urdu_nlp.detector import tag_tokens
from roman_urdu_nlp.normalizer import normalize
from roman_urdu_nlp.transliterator import transliterate as roman_to_urdu_script

logger = logging.getLogger(__name__)

_ROMAN_TO_EN_DICT = {
    "hai": "is", "hain": "are", "tha": "was", "thi": "was", "the": "were",
    "aur": "and", "ya": "or", "lekin": "but", "agar": "if", "mein": "in",
    "se": "from", "ka": "of", "ki": "of", "ke": "of", "ko": "to",
    "par": "on", "pe": "on", "bhi": "also", "bohat": "very", "zyada": "much",
    "sirf": "only", "nahi": "not", "ye": "this", "wo": "that", "kuch": "some",
    "aaj": "today", "kal": "tomorrow", "ab": "now", "phir": "then",
    "karna": "to do", "karega": "will do", "hoga": "will be", "hogi": "will be",
    "hota": "happens", "hoti": "happens", "hote": "happen",
    "kyun": "why", "kyu": "why", "kab": "when", "kahan": "where", "kaise": "how",
    "sab": "all", "har": "every", "koi": "any", "unka": "their", "mera": "my"
}


def _dictionary_translate(text: str) -> str:
    if not text:
        return ""
    tokens = text.split()
    res = []
    for tok in tokens:
        clean = tok.lower().strip(".,!?;:\"'()[]{}")
        if clean in _ROMAN_TO_EN_DICT:
            res.append(_ROMAN_TO_EN_DICT[clean])
        else:
            res.append(tok)
    return " ".join(res)


class TextProcessingException(Exception):
    """Exception raised for text processing failures."""
    def __init__(self, message: str, details: dict | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class TextProcessingService:
    _pipeline = RomanUrduPipeline()
    _offline_logged: bool = False

    @classmethod
    def is_passthrough_english(cls, text: str) -> bool:
        """
        Check if text qualifies as native English (>= 80% English tokens).
        """
        if not text or not text.strip():
            return False
        tags = tag_tokens(text)
        word_tags = [t for t in tags if t["lang"] in ("en", "ur")]
        if not word_tags:
            return bool(re.search(r"[a-zA-Z]", text))
        en_count = sum(1 for t in word_tags if t["lang"] == "en")
        ratio = en_count / len(word_tags)
        return ratio >= ENGLISH_PASSTHROUGH_RATIO

    @classmethod
    def process_text(cls, raw_text: str) -> dict:
        """
        Run raw code-mixed text through the full NLP pipeline.

        Returns dictionary matching TextProcessResponse schema.
        """
        if not raw_text or not raw_text.strip():
            return {
                "status": "success",
                "original_text": raw_text or "",
                "cleaned_text": "",
                "normalized_text": "",
                "urdu_text": "",
                "english_text": "",
                "token_tags": [],
                "language_distribution": {"ur": 1.0},
                "stages": {}
            }

        # If input contains Perso-Arabic Urdu script, transliterate first
        if UrduToRomanTransliterator.is_urdu_script(raw_text):
            intermediate_text = UrduToRomanTransliterator.transliterate(raw_text)
        else:
            intermediate_text = raw_text

        res = cls._pipeline(intermediate_text)

        return {
            "status": "success",
            "original_text": raw_text,
            "cleaned_text": res.stages.get("cleaned", ""),
            "normalized_text": res.normalized,
            "urdu_text": res.urdu,
            "english_text": res.english,
            "token_tags": res.token_tags,
            "language_distribution": res.language_distribution,
            "stages": res.stages
        }

    @classmethod
    def translate_segments_batch(cls, segments: list[Any]) -> list[Any]:
        """
        Batch-translate a list of TranscriptSegment instances or dicts.
        Adheres to 3-tier fallback chain:
          Tier 1: Google Translate (batched 10-15 segments)
          Tier 2: Local LLM (Qwen/Qwen2.5-0.5B-Instruct) on rate limit / error
          Tier 3: Bilingual dictionary mapping
        Also attaches Perso-Arabic Urdu transliteration and full provenance.
        """
        if not segments:
            return []

        # 1. First pass: generate urdu_text and evaluate passthrough
        needs_translation_indices: list[int] = []

        for idx, seg in enumerate(segments):
            raw = seg.text if hasattr(seg, "text") else seg.get("text", "")
            norm = getattr(seg, "normalized_text", None) or (seg.get("normalized_text") if isinstance(seg, dict) else None) or raw

            # Perso-Arabic script conversion if raw had Urdu script
            if UrduToRomanTransliterator.is_urdu_script(raw):
                norm = UrduToRomanTransliterator.transliterate(raw)

            # Urdu script transliteration
            try:
                urdu_script = roman_to_urdu_script(norm)
            except Exception:
                urdu_script = norm

            if hasattr(seg, "urdu_text"):
                seg.urdu_text = urdu_script
            elif isinstance(seg, dict):
                seg["urdu_text"] = urdu_script

            # Check passthrough
            if cls.is_passthrough_english(norm):
                if hasattr(seg, "english_text"):
                    seg.english_text = norm
                    seg.translation_source = "passthrough"
                    seg.translation_confidence = 1.0
                elif isinstance(seg, dict):
                    seg["english_text"] = norm
                    seg["translation_source"] = "passthrough"
                    seg["translation_confidence"] = 1.0
            else:
                needs_translation_indices.append(idx)

        online_enabled = getattr(config, "ENABLE_ONLINE_TRANSLATION", True)
        if not online_enabled and not cls._offline_logged:
            logger.info(
                "Online translation is disabled (ENABLE_ONLINE_TRANSLATION=False). "
                "Routing directly to offline Local LLM and dictionary fallback."
            )
            cls._offline_logged = True

        # 2. Translate non-passthrough segments in batches of 10-15
        batch_size = 12
        for b_start in range(0, len(needs_translation_indices), batch_size):
            batch_idxs = needs_translation_indices[b_start:b_start + batch_size]
            batch_texts = []
            for b_idx in batch_idxs:
                s = segments[b_idx]
                t = getattr(s, "normalized_text", None) or (s.get("normalized_text") if isinstance(s, dict) else None) or (s.text if hasattr(s, "text") else s.get("text", ""))
                batch_texts.append(t)

            # Try Tier 1: Google Translate (only if online translation enabled)
            google_success = False
            if online_enabled:
                try:
                    import concurrent.futures
                    from deep_translator import GoogleTranslator
                    prompt_lines = [f"[{i}] {txt}" for i, txt in enumerate(batch_texts)]
                    batch_input = "\n".join(prompt_lines)
                    gt = GoogleTranslator(source="auto", target="en")
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        gt_future = executor.submit(gt.translate, batch_input)
                        gt_output = gt_future.result(timeout=8.0)

                    parsed: dict[int, str] = {}
                    if gt_output:
                        for line in gt_output.split("\n"):
                            m = re.match(r"^\[(\d+)\]\s*(.*)$", line.strip())
                            if m:
                                parsed[int(m.group(1))] = m.group(2).strip()

                    if len(parsed) == len(batch_texts):
                        google_success = True
                        for i, seg_i in enumerate(batch_idxs):
                            translated_val = parsed.get(i, batch_texts[i])
                            seg = segments[seg_i]
                            if hasattr(seg, "english_text"):
                                seg.english_text = translated_val
                                seg.translation_source = "google_translate"
                                seg.translation_confidence = TRANSLATION_CONFIDENCE_GOOGLE
                                seg.translation_reason = "online_success"
                            elif isinstance(seg, dict):
                                seg["english_text"] = translated_val
                                seg["translation_source"] = "google_translate"
                                seg["translation_confidence"] = TRANSLATION_CONFIDENCE_GOOGLE
                                seg["translation_reason"] = "online_success"
                except Exception as e:
                    logger.warning(f"Google Translate batch failed or timed out ({e}). Falling back to Local LLM / Dictionary.")
                    google_success = False

            if not google_success:
                fallback_reason = "skipped_online_disabled" if not online_enabled else "google_translate_failed"
                # Tier 2 & 3: Local LLM (batched on GPU) with instant circuit-breaker & Dictionary Fallback
                llm_circuit_open = getattr(cls, "_llm_translation_circuit_open", False)
                for i, seg_i in enumerate(batch_idxs):
                    txt_to_trans = batch_texts[i]
                    seg = segments[seg_i]
                    llm_success = False
                    if not llm_circuit_open:
                        try:
                            import concurrent.futures
                            from app.services.llm_provider import get_llm_lifecycle_manager, get_llm_provider
                            llm = get_llm_provider(feature="translation")
                            llm_prompt = f"Translate the following Roman Urdu lecture sentence into clean English:\n\n{txt_to_trans}"
                            sys_prompt = "You are an academic translation assistant. Translate into fluent English. Output ONLY the English translation without quotes or preamble."
                            if get_llm_lifecycle_manager().current_thread_holds_lock():
                                translated = llm.generate_text(llm_prompt, system_prompt=sys_prompt, max_tokens=120)
                            else:
                                executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                                try:
                                    llm_future = executor.submit(llm.generate_text, llm_prompt, system_prompt=sys_prompt, max_tokens=120)
                                    translated = llm_future.result(timeout=25.0)
                                finally:
                                    executor.shutdown(wait=False, cancel_futures=True)
                            translated = re.sub(
                                r"^(?:here\s+(?:is|'s)\s+(?:a\s+|the\s+)?[^:\n]{0,80}:\s*)",
                                "",
                                translated.strip(),
                                flags=re.IGNORECASE,
                            )
                            translated = translated.strip().strip('"').strip("'").strip()
                            if translated and len(translated) > 2 and translated.lower() != txt_to_trans.lower():
                                if hasattr(seg, "english_text"):
                                    seg.english_text = translated
                                    seg.translation_source = "local_llm"
                                    seg.translation_confidence = TRANSLATION_CONFIDENCE_LLM
                                    seg.translation_reason = fallback_reason
                                elif isinstance(seg, dict):
                                    seg["english_text"] = translated
                                    seg["translation_source"] = "local_llm"
                                    seg["translation_confidence"] = TRANSLATION_CONFIDENCE_LLM
                                    seg["translation_reason"] = fallback_reason
                                llm_success = True
                        except Exception as llm_err:
                            logger.warning(f"Local LLM translation unavailable/timed out on segment {seg_i} ({llm_err}); switching remaining segments in batch to fast dictionary fallback.")
                            llm_circuit_open = True
                            llm_success = False

                    if not llm_success:
                        # Tier 3: Dictionary fallback
                        dict_res = _dictionary_translate(txt_to_trans)
                        if hasattr(seg, "english_text"):
                            seg.english_text = dict_res
                            seg.translation_source = "dictionary"
                            seg.translation_confidence = TRANSLATION_CONFIDENCE_DICTIONARY
                            seg.translation_reason = fallback_reason
                        elif isinstance(seg, dict):
                            seg["english_text"] = dict_res
                            seg["translation_source"] = "dictionary"
                            seg["translation_confidence"] = TRANSLATION_CONFIDENCE_DICTIONARY
                            seg["translation_reason"] = fallback_reason

        return segments

    @classmethod
    def process_recording_transcript(cls, recording_id: str, db: Session) -> dict:
        """
        Batch process and enhance all transcript segments of a recording.
        Updates normalized_transcript, urdu_transcript, english_transcript,
        and segment level translations with provenance.
        """
        recording = db.query(Recording).filter(Recording.id == recording_id).first()
        if not recording:
            raise TextProcessingException(f"Recording '{recording_id}' not found.")

        transcript = db.query(Transcript).filter(Transcript.recording_id == recording_id).first()
        if not transcript:
            raise TextProcessingException(f"Transcript for recording '{recording_id}' not found. Please transcribe first.")

        segments = db.query(TranscriptSegment).filter(
            TranscriptSegment.transcript_id == transcript.id
        ).order_by(TranscriptSegment.segment_index.asc()).all()

        normalized_segment_texts = []
        lang_counts: dict[str, int] = {}

        for seg in segments:
            seg_raw = seg.text
            if UrduToRomanTransliterator.is_urdu_script(seg_raw):
                seg_raw = UrduToRomanTransliterator.transliterate(seg_raw)

            seg_res = cls._pipeline(seg_raw)
            seg_norm = seg_res.normalized if seg_res.normalized else seg_raw
            seg.normalized_text = seg_norm
            normalized_segment_texts.append(seg_norm)

            for tag in seg_res.token_tags:
                lang = tag["lang"]
                if lang in ("ur", "en"):
                    lang_counts[lang] = lang_counts.get(lang, 0) + 1

        # Batch translate all segments
        cls.translate_segments_batch(segments)

        full_normalized = " ".join(normalized_segment_texts).strip()
        full_urdu = " ".join([s.urdu_text for s in segments if s.urdu_text]).strip()
        full_english = " ".join([s.english_text for s in segments if s.english_text]).strip()

        # Update language summary based on fine-grained token tags
        total_tokens = sum(lang_counts.values())
        if total_tokens > 0:
            lang_summary_parts = [
                f"{lang}: {round((count / total_tokens) * 100)}%"
                for lang, count in sorted(lang_counts.items(), key=lambda x: x[1], reverse=True)
            ]
            transcript.language_summary = ", ".join(lang_summary_parts)

        transcript.normalized_transcript = full_normalized
        transcript.urdu_transcript = full_urdu
        transcript.english_transcript = full_english
        db.commit()
        db.refresh(transcript)

        return {
            "status": "success",
            "recording_id": recording_id,
            "transcript_id": transcript.id,
            "original_transcript": transcript.original_transcript,
            "normalized_transcript": transcript.normalized_transcript,
            "urdu_transcript": transcript.urdu_transcript,
            "english_transcript": transcript.english_transcript,
            "language_summary": transcript.language_summary,
            "total_segments": len(segments)
        }

    @classmethod
    def backfill_recording(cls, recording_id: str, db: Session, regenerate_intelligence: bool = True) -> dict:
        """
        Backfill translations and provenance for all segments of a recording.
        If regenerate_intelligence is True, regenerates study notes, assignments,
        action items, quizzes, and summaries from the clean English transcript.
        """
        from app.services.intelligence_service import LectureIntelligenceService

        transcript = db.query(Transcript).filter(Transcript.recording_id == recording_id).first()
        if not transcript:
            raise TextProcessingException(f"Transcript for recording '{recording_id}' not found.")

        segments = db.query(TranscriptSegment).filter(
            TranscriptSegment.transcript_id == transcript.id
        ).order_by(TranscriptSegment.segment_index.asc()).all()

        cls.translate_segments_batch(segments)

        full_urdu = " ".join([s.urdu_text for s in segments if s.urdu_text]).strip()
        full_english = " ".join([s.english_text for s in segments if s.english_text]).strip()

        transcript.urdu_transcript = full_urdu
        transcript.english_transcript = full_english
        db.commit()
        db.refresh(transcript)

        if regenerate_intelligence:
            LectureIntelligenceService.generate_intelligence_report(recording_id, db)

        return {
            "status": "success",
            "recording_id": recording_id,
            "transcript_id": transcript.id,
            "total_segments": len(segments),
            "english_transcript_words": len(full_english.split()),
            "intelligence_regenerated": regenerate_intelligence
        }

    @classmethod
    def get_tagged_segments(cls, recording_id: str, db: Session) -> list[dict]:
        """
        Retrieve all transcript segments with token-level language and technical tags.
        """
        transcript = db.query(Transcript).filter(Transcript.recording_id == recording_id).first()
        if not transcript:
            raise TextProcessingException(f"Transcript for recording '{recording_id}' not found.")

        segments = db.query(TranscriptSegment).filter(
            TranscriptSegment.transcript_id == transcript.id
        ).order_by(TranscriptSegment.segment_index.asc()).all()

        results = []
        for seg in segments:
            seg_text = seg.text
            if UrduToRomanTransliterator.is_urdu_script(seg_text):
                seg_text = UrduToRomanTransliterator.transliterate(seg_text)

            token_tags = tag_tokens(seg_text)
            norm_text = normalize(seg_text)

            results.append({
                "segment_id": seg.id,
                "chunk_index": seg.chunk_index,
                "segment_index": seg.segment_index,
                "start_time": seg.start_time,
                "end_time": seg.end_time,
                "original_text": seg.text,
                "normalized_text": norm_text,
                "urdu_text": seg.urdu_text,
                "english_text": seg.english_text,
                "translation_source": seg.translation_source,
                "translation_confidence": seg.translation_confidence,
                "token_tags": token_tags
            })

        return results
