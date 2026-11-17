"""
app/services/academic_pipeline.py — Academic Intelligence v2 orchestration.

Runs strictly downstream of transcription: it only READS transcript segments (never writes them) and
produces enriched notes + MCQs that are validated before they are returned.

Output states (meta.status): complete | partial | insufficient_content | failed.
`failed` returns no notes content and no quizzes; unvalidated content is never emitted.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any, Callable, Optional, Sequence

from app import config as cfg
from app.services.academic_preprocess import (
    AcademicPreprocessor, extract_keyphrases, score_importance, importance_label,
    IMPORTANCE_WEIGHTS, IMPORTANCE_CUTS,
)
from app.services.discourse_cues import DiscourseCueClassifier
from app.services.lecture_context import LectureContext
from app.services.pedagogical_validator import (
    PedagogicalValidator, ContaminationAuditor, MCQStructureValidator,
)
from app.services.study_guide_generator import (
    AcademicMCQEngine, align_sentence, DISTRACTOR_BAND, DISTRACTOR_BAND_RELAXED,
)

logger = logging.getLogger(__name__)

PIPELINE_VERSION = "academic-v2.0"
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_ANALOGY_RE = re.compile(r"\b(?:like|jaise|similar to|for example|analogous|just as)\b", re.I)


def compute_transcript_hash(segments: Sequence[Any]) -> str:
    h = hashlib.sha256()
    for s in segments:
        sid = getattr(s, "id", None) or getattr(s, "segment_index", 0)
        if getattr(s, "is_intel_view", False):
            txt = getattr(s, "llm_text", "") or ""
        else:
            txt = getattr(s, "english_text", None) or getattr(s, "normalized_text", None) or getattr(s, "text", "") or ""
        h.update(f"{sid}\x1f{txt}\x1e".encode("utf-8"))
    return h.hexdigest()


def compute_config_hash() -> str:
    from app.services.intelligence_preprocessor import PREPROCESSING_VERSION

    payload = {
        "align": (cfg.ALIGN_COS_MIN, cfg.ALIGN_OVERLAP_MIN), "dedupe": cfg.MCQ_DEDUPE_COS,
        "min_words": cfg.MIN_WORDS_FOR_10, "cue": cfg.EXAM_CUE_MIN_CONFIDENCE,
        "strict": cfg.STRICT_TRANSCRIPT_ONLY, "weights": IMPORTANCE_WEIGHTS, "cuts": IMPORTANCE_CUTS,
        "bands": (DISTRACTOR_BAND, DISTRACTOR_BAND_RELAXED), "repair": cfg.MAX_REPAIR_PER_ITEM,
        "universal_notes_rev": "2026-11-10-v3",
        "intel_preproc_ver": PREPROCESSING_VERSION,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def stored_result_is_current(
    stored_notes: Optional[dict],
    segments: Sequence[Any],
    transcript_hash_override: Optional[str] = None,
) -> bool:
    """Idempotency check: same transcript hash + pipeline version + config hash => reuse the stored result."""
    meta = (stored_notes or {}).get("meta") or {}
    expected_hash = transcript_hash_override or compute_transcript_hash(segments)
    return (
        meta.get("pipeline_version") == PIPELINE_VERSION
        and meta.get("transcript_hash") == expected_hash
        and meta.get("config_hash") == compute_config_hash()
        and meta.get("status") in {"complete", "insufficient_content"}
    )


class _Budget:
    def __init__(self, seconds: float, t0: Optional[float] = None):
        self.limit, self.t0 = seconds, (t0 if t0 is not None else time.monotonic())

    def expired(self) -> bool:
        return (time.monotonic() - self.t0) > self.limit

    def elapsed(self) -> float:
        return round(time.monotonic() - self.t0, 2)


def _topic_window(topic: dict) -> tuple[float, float]:
    return float(topic.get("start_time") or 0.0), float(topic.get("end_time") or 0.0)


def _filter_sentences(text: str, keep: Callable[[str], bool]) -> tuple[str, int]:
    parts = [p for p in _SENT_SPLIT.split((text or "").strip()) if p.strip()]
    kept = [p for p in parts if keep(p)]
    return " ".join(kept), len(parts) - len(kept)


def run_academic_v2(
    notes: dict,
    segments: Sequence[Any],
    transcript_text: str,
    recording_id: str,
    recording_type: str = "Lecture",
    embed_fn: Optional[Callable] = None,
    t0: Optional[float] = None,
    transcript_hash_override: Optional[str] = None,
) -> dict:
    """
    Returns {"notes": dict, "quizzes": list, "meta": dict}. Never raises: failures yield status "failed"
    with empty quizzes and no unvalidated content.
    """
    budget = _Budget(cfg.INTEL_TIME_BUDGET_S, t0)
    t_hash = transcript_hash_override or compute_transcript_hash(segments)
    meta: dict[str, Any] = {
        "pipeline_version": PIPELINE_VERSION, "transcript_hash": t_hash, "config_hash": compute_config_hash(),
        "created_at": datetime.now(timezone.utc).isoformat(), "status": "complete",
        "unfinished_stages": [], "dropped": {"note_sentences": 0, "topics": 0, "mcq": 0}, "timings_s": {},
    }
    try:
        stage = time.monotonic()
        pre = AcademicPreprocessor.process_segments(segments)
        kps = extract_keyphrases(pre.spans, top_k=30)
        hints = DiscourseCueClassifier.extract_cues(segments)
        meta["timings_s"]["analysis"] = round(time.monotonic() - stage, 2)
        vocab = pre.recording_vocab
        out = dict(notes)

        # ---- notes: importance, hints, alignment, contamination gate -----------------
        stage = time.monotonic()
        topics_out, scores = [], []
        for topic in out.get("study_topics", []):
            if budget.expired():
                meta["unfinished_stages"].append("notes_alignment")
                topics_out.append(topic)  # keep legacy topic untouched rather than emit half-processed content
                continue
            t = dict(topic)
            st, et = _topic_window(t)
            t_spans = [s for s in pre.spans if s.raw_end >= st and s.raw_start <= et] or pre.spans[:1]

            imp = score_importance(t_spans, pre.spans, kps)
            t["inferred_importance"] = imp["score"]
            t["inferred_exam_relevance"] = importance_label(imp["score"])  # derived label (D3)
            t["importance_components"] = imp["components"]
            t["explicit_exam_hints"] = [
                {"polarity": h.polarity, "cue_type": h.cue_type, "raw_quote": h.raw_quote, "segment_id": h.segment_id,
                 "start_time": h.start_time, "end_time": h.end_time, "confidence": h.confidence}
                for h in hints if h.end_time >= st and h.start_time <= et
            ]
            scores.append(imp["score"])

            evidence: list[dict] = []
            ov_min = 0.05 if t.get("source") == "local_llm" else None

            def keep(sentence: str, window=(st, et), ov_floor=ov_min) -> bool:
                al = align_sentence(sentence, pre.spans, embed_fn=embed_fn, overlap_min=ov_floor, window=window)
                if al is None:
                    return False
                a = ContaminationAuditor.audit_text(sentence, vocab)
                if not a["passed"] and not _ANALOGY_RE.search(sentence):
                    return False
                evidence.append({"segment_ids": al["segment_ids"], "start": al["start_time"], "end": al["end_time"]})
                return True

            expl, d1 = _filter_sentences(t.get("explanation", ""), keep)
            kp_new, d2 = [], 0
            for kp in t.get("key_points", []) or []:
                if keep(str(kp)):
                    kp_new.append(kp)
                else:
                    d2 += 1
            meta["dropped"]["note_sentences"] += d1 + d2
            if not expl and not kp_new:
                meta["dropped"]["topics"] += 1
                continue
            t["explanation"], t["key_points"] = expl, kp_new
            t["thin_topic"] = len(kp_new) < 2
            t["evidence_timestamps"] = evidence
            topics_out.append(t)

        out["study_topics"] = topics_out
        if topics_out:
            out["topics"] = [t.get("title", "") for t in topics_out if t.get("title")]
            synced_subtopics: list[str] = []
            for t in topics_out:
                for st in t.get("subtopics", []) or []:
                    st_title = st.get("title", "") if isinstance(st, dict) else getattr(st, "title", "")
                    if st_title and not st_title.startswith(("Definition:", "Real-World Example", "Formula:")) and st_title not in synced_subtopics:
                        synced_subtopics.append(st_title)
            if synced_subtopics:
                out["subtopics"] = synced_subtopics
            if not out.get("markdown_study_guide") or meta["dropped"]["topics"] > 0 or meta["dropped"]["note_sentences"] > 0:
                from app.services.study_guide_generator import _build_extractive_fallback_guide
                out["markdown_study_guide"] = _build_extractive_fallback_guide(
                    topics_out,
                    lecture_overview=out.get("lecture_overview", ""),
                    concept_relationships=out.get("concept_relationships", []),
                )
        out["explicit_exam_hints"] = [
            {"polarity": h.polarity, "cue_type": h.cue_type, "raw_quote": h.raw_quote, "segment_id": h.segment_id,
             "start_time": h.start_time, "end_time": h.end_time, "confidence": h.confidence} for h in hints
        ]
        if scores:
            out["inferred_importance"] = max(1, min(5, round(sum(scores) / len(scores))))
            out["inferred_exam_relevance"] = importance_label(out["inferred_importance"])
        meta["timings_s"]["notes"] = round(time.monotonic() - stage, 2)

        # ---- MCQs --------------------------------------------------------------------
        quizzes: list[dict] = []
        if recording_type != "Meeting":
            if budget.expired():
                meta["unfinished_stages"].append("mcq")
            else:
                stage = time.monotonic()
                rng = LectureContext.from_transcript(recording_id, transcript_text).get_rng("mcq")
                res = AcademicMCQEngine.generate(pre.spans, kps, rng=rng, embed_fn=embed_fn, transcript_hash=t_hash)
                quizzes, dropped_answers = _gate_quizzes(res["mcqs"], transcript_text, vocab)
                meta["dropped"]["mcq"] += len(res["mcqs"]) - len(quizzes)
                # one bounded global regeneration pass if validation removed questions
                if len(quizzes) < 10 and dropped_answers and res["mcq_status"] != "insufficient_content" \
                        and not budget.expired() and cfg.MAX_REPAIR_PER_ITEM > 0:
                    res2 = AcademicMCQEngine.generate(pre.spans, kps, rng=rng, embed_fn=embed_fn,
                                                      transcript_hash=t_hash, exclude_answers=dropped_answers)
                    q2, _ = _gate_quizzes(res2["mcqs"], transcript_text, vocab)
                    if len(q2) > len(quizzes):
                        quizzes = q2
                if len(quizzes) == 10 and not res["reason"]:
                    meta.update(mcq_status="complete", mcq_count=10, mcq_reason=None)
                else:
                    meta.update(mcq_status="insufficient_content", mcq_count=len(quizzes),
                                mcq_reason=res["reason"] or "validation_attrition")
                    meta["status"] = "insufficient_content"
                meta["distribution_relaxed"] = res.get("distribution_relaxed", False)
                meta["timings_s"]["mcq"] = round(time.monotonic() - stage, 2)
                for q in quizzes:
                    q["mcq_status"] = meta["mcq_status"]

        if meta["unfinished_stages"]:
            meta["status"] = "partial"
        meta["elapsed_s"] = budget.elapsed()
        from app.services.pedagogical_validator import PreSubmissionQualityController
        out = PreSubmissionQualityController.audit_and_heal_notes(out, transcript_text=transcript_text, segments=segments)
        out["meta"] = meta
        return {"notes": out, "quizzes": quizzes, "meta": meta}
    except Exception as exc:  # never emit unvalidated content
        logger.exception("Academic v2 pipeline failed")
        meta.update(status="failed", reason=f"{type(exc).__name__}: {exc}", elapsed_s=budget.elapsed())
        empty = {"topics": [], "subtopics": [], "definitions": [], "key_concepts": [], "important_points": [],
                 "study_topics": [], "meta": meta}
        return {"notes": empty, "quizzes": [], "meta": meta}


def _gate_quizzes(mcqs: list[dict], transcript_text: str, vocab: set) -> tuple[list[dict], set]:
    """Validation gate: Bloom policy + structure + contamination. Returns (kept, answers_of_dropped)."""
    kept, dropped = [], set()
    bloom = PedagogicalValidator.validate_quizzes(mcqs, transcript_text, bloom_policy="quiz")
    rejected_stems = {r["question"] for r in bloom["rejected"]}
    for q in mcqs:
        bad = (
            q["question"] in rejected_stems
            or bool(MCQStructureValidator.validate_question(q))
            or not ContaminationAuditor.audit_quizzes([q], vocab)["passed"]
        )
        if bad:
            dropped.add(q["correct_answer"])
        else:
            kept.append(q)
    return kept, dropped
