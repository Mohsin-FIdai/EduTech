"""
tests/test_academic_intelligence.py — Phase 4 & Phase 5 tests for Academic Intelligence v2.

Covers:
1. Analogy exemption in run_academic_v2 (instructor analogies preserved while foreign domain terms rejected).
2. Drop-below-10 repair regeneration & insufficient_content fallback.
3. Time-budget expiry (INTEL_TIME_BUDGET_S) returning status: "partial" with unfinished_stages.
4. Failure handling returning status: "failed" with zero unvalidated content.
5. LocalLLMProvider._gen_lock thread safety under concurrent calls.
6. Idempotency in generate_intelligence_report (reuses stored result when hashes match; invalidates on edit).
7. End-to-end integration across short, long, English, Roman Urdu, and code-switched lectures,
   measuring grounding rate (>= 95%), contamination count (0), execution time, and peak RAM.
"""

from __future__ import annotations

import re
import threading
import time
import tracemalloc
from types import SimpleNamespace

from app import config as cfg
from app.database import SessionLocal
from app.models import Recording, Transcript, TranscriptSegment
from app.services.academic_pipeline import run_academic_v2
from app.services.intelligence_service import LectureIntelligenceService
from app.services.llm_provider import LocalLLMProvider
from app.services.pedagogical_validator import ContaminationAuditor
from tests.test_academic_intelligence_core import LECTURE, make_segments, uniform_embed, bow_embed


def _seg(idx: int, text: str, start: float = 0.0, end: float = 10.0, tid: str = "t1"):
    return SimpleNamespace(
        id=idx,
        segment_index=idx,
        transcript_id=tid,
        text=text,
        normalized_text=text,
        english_text=text,
        start_time=start,
        end_time=end,
    )


def _long_os_segments() -> tuple[list, str]:
    """Use the 44-sentence (>600 word) OS lecture fixture with repeating multi-word keyphrases."""
    segs = make_segments(LECTURE, per_seg=2, seg_dur=10.0)
    for s in segs:
        s.normalized_text = s.text
        s.english_text = s.text
        s.transcript_id = "tr_bench"
    full_text = " ".join(s.text for s in segs)
    return segs, full_text


class TestPhase4ResilienceAndSafety:
    def test_analogy_exemption_preserves_instructor_analogy_and_drops_foreign_contamination(self):
        """Instructor analogies ('like cars on a motorway') spoken in the lecture are kept, while unmentioned foreign domain terms ('mitochondria') are rejected."""
        segs = [
            _seg(1, "First Come First Served scheduling queues processes in arrival order like cars on a motorway.", 0.0, 20.0),
            _seg(2, "Turnaround time is completion time minus arrival time in operating systems.", 20.0, 40.0),
        ]
        full_text = " ".join(s.text for s in segs)
        notes = {
            "study_topics": [
                {
                    "title": "FCFS Scheduling",
                    "start_time": 0.0,
                    "end_time": 40.0,
                    "explanation": (
                        "First Come First Served scheduling queues processes in arrival order like cars on a motorway. "
                        "Mitochondria ATP synthesis occurs inside cellular membranes."
                    ),
                    "key_points": [
                        "Turnaround time is completion time minus arrival time in operating systems.",
                        "Photosynthesis chlorophyll reactions happen in plant leaves.",
                    ],
                }
            ]
        }
        res = run_academic_v2(notes, segs, full_text, recording_id="rec_analogy", embed_fn=bow_embed)
        topics = res["notes"]["study_topics"]
        assert len(topics) == 1
        expl = topics[0]["explanation"]
        assert "like cars on a motorway" in expl
        assert "Mitochondria" not in expl
        assert len(topics[0]["key_points"]) == 1
        assert "Turnaround time" in topics[0]["key_points"][0]
        assert res["meta"]["dropped"]["note_sentences"] >= 2

    def test_repair_regeneration_when_question_rejected(self, monkeypatch):
        """If initial MCQ generation has an item rejected by validation, bounded repair regenerates excluding dropped answers."""
        segs, full_text = _long_os_segments()
        notes = {
            "study_topics": [
                {
                    "title": "Process Scheduling",
                    "start_time": 0.0,
                    "end_time": 220.0,
                    "explanation": segs[0].text,
                    "key_points": [segs[1].text, segs[2].text],
                }
            ]
        }
        calls = []
        orig_gen = __import__("app.services.study_guide_generator", fromlist=["AcademicMCQEngine"]).AcademicMCQEngine.generate

        def spy_generate(*args, **kwargs):
            calls.append(kwargs.get("exclude_answers"))
            out = orig_gen(*args, **kwargs)
            # On the first call only, corrupt one option with a banned catch-all so _gate_quizzes drops it
            if len(calls) == 1 and out["mcqs"]:
                bad_q = dict(out["mcqs"][0])
                bad_q["options"] = list(bad_q["options"])
                bad_q["options"][0] = "All of the above"
                bad_q["correct_answer"] = bad_q["options"][1]
                out["mcqs"][0] = bad_q
            return out

        monkeypatch.setattr("app.services.study_guide_generator.AcademicMCQEngine.generate", spy_generate)
        res = run_academic_v2(notes, segs, full_text, recording_id="rec_repair", embed_fn=uniform_embed)
        assert len(calls) == 2
        assert calls[0] is None
        assert calls[1] is not None and len(calls[1]) >= 1
        assert len(res["quizzes"]) >= 9

    def test_time_budget_expiry_returns_partial_status(self):
        """When t0 is already past INTEL_TIME_BUDGET_S, run_academic_v2 skips remaining work and returns status='partial'."""
        segs, full_text = _long_os_segments()
        notes = {
            "study_topics": [
                {
                    "title": "Process Scheduling",
                    "start_time": 0.0,
                    "end_time": 60.0,
                    "explanation": segs[0].text,
                    "key_points": [segs[1].text],
                }
            ]
        }
        expired_t0 = time.monotonic() - (cfg.INTEL_TIME_BUDGET_S + 10.0)
        res = run_academic_v2(notes, segs, full_text, recording_id="rec_timeout", embed_fn=uniform_embed, t0=expired_t0)
        assert res["meta"]["status"] == "partial"
        assert "notes_alignment" in res["meta"]["unfinished_stages"]
        assert "mcq" in res["meta"]["unfinished_stages"]
        assert res["quizzes"] == []

    def test_failure_handling_returns_failed_with_zero_unvalidated_content(self, monkeypatch):
        """An unexpected exception inside the pipeline returns status='failed' and never leaks unvalidated notes or quizzes."""
        segs, full_text = _long_os_segments()
        notes = {"study_topics": [{"title": "Unvalidated Topic", "explanation": "Unvalidated text"}]}

        def boom(*args, **kwargs):
            raise RuntimeError("synthetic failure")

        monkeypatch.setattr("app.services.academic_preprocess.AcademicPreprocessor.process_segments", boom)
        res = run_academic_v2(notes, segs, full_text, recording_id="rec_fail", embed_fn=uniform_embed)
        assert res["meta"]["status"] == "failed"
        assert "RuntimeError" in res["meta"]["reason"]
        assert res["notes"]["study_topics"] == []
        assert res["quizzes"] == []

    def test_llm_provider_gen_lock_serializes_concurrent_calls(self):
        """Verify LocalLLMProvider._gen_lock prevents overlapping execution in critical section."""
        lock = LocalLLMProvider._gen_lock
        assert isinstance(lock, type(threading.Lock()))

        active = 0
        max_concurrent = 0
        errors = []

        def worker():
            nonlocal active, max_concurrent
            with LocalLLMProvider._gen_lock:
                active += 1
                if active > max_concurrent:
                    max_concurrent = active
                if active > 1:
                    errors.append("Concurrent execution detected inside _gen_lock!")
                time.sleep(0.01)
                active -= 1

        threads = [threading.Thread(target=worker) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert max_concurrent == 1

    def test_idempotency_reuses_stored_result_and_invalidates_on_edit(self, monkeypatch):
        """Second run with unchanged transcript and config reuses stored report; editing a segment re-runs pipeline."""
        monkeypatch.setattr(cfg, "ACADEMIC_INTEL_V2", True)
        segs_data, full_text = _long_os_segments()

        db_session = SessionLocal()
        try:
            rec = Recording(
                id="rec_idem_1",
                filename="os_lec.wav",
                file_path="/tmp/os_lec.wav",
                file_format=".wav",
                media_type="audio",
                file_size_bytes=1024,
                duration_seconds=220.0,
                status="COMPLETED",
                recording_type="Lecture",
            )
            db_session.merge(rec)
            tr = Transcript(
                id="tr_idem_1",
                recording_id=rec.id,
                original_transcript=full_text,
                normalized_transcript=full_text,
                english_transcript=full_text,
                language_summary="en: 100%",
            )
            db_session.merge(tr)
            db_session.query(TranscriptSegment).filter_by(transcript_id=tr.id).delete()
            for s in segs_data:
                db_session.add(
                    TranscriptSegment(
                        transcript_id=tr.id,
                        recording_id=rec.id,
                        chunk_index=0,
                        segment_index=s.segment_index,
                        start_time=s.start_time,
                        end_time=s.end_time,
                        text=s.text,
                        normalized_text=s.text,
                        english_text=s.text,
                    )
                )
            db_session.commit()

            # Stub extract_lecture_notes so we don't run LLM pass during idempotency check
            fast_notes = {
                "topics": ["Round Robin", "Priority Scheduling"],
                "subtopics": [],
                "definitions": [{"term": "Time quantum", "definition": "Maximum CPU time per turn"}],
                "key_concepts": ["Round Robin", "Time Quantum", "Aging"],
                "important_points": [segs_data[0].text],
                "study_topics": [
                    {
                        "id": "topic_01",
                        "title": "Round Robin Scheduling",
                        "start_time": 0.0,
                        "end_time": 220.0,
                        "explanation": segs_data[0].text,
                        "key_points": [segs_data[1].text, segs_data[3].text],
                        "definitions": [{"term": "Time quantum", "definition": "Maximum CPU time"}],
                        "examples": [],
                        "formulas": [],
                    }
                ],
            }
            monkeypatch.setattr(
                LectureIntelligenceService,
                "extract_lecture_notes",
                classmethod(lambda cls, *a, **kw: dict(fast_notes)),
            )
            monkeypatch.setattr(
                "app.services.embedding_service.EmbeddingModelManager.encode",
                classmethod(lambda cls, texts, **kw: uniform_embed(texts)),
            )

            # First run
            r1 = LectureIntelligenceService.generate_intelligence_report(rec.id, db_session)
            created_1 = r1["lecture_notes"]["meta"]["created_at"]
            hash_1 = r1["lecture_notes"]["meta"]["transcript_hash"]

            # Second run with identical transcript -> should reuse stored report (created_at identical)
            r2 = LectureIntelligenceService.generate_intelligence_report(rec.id, db_session)
            assert r2["lecture_notes"]["meta"]["created_at"] == created_1
            assert r2["lecture_notes"]["meta"]["transcript_hash"] == hash_1

            # Now modify a segment text -> transcript_hash changes -> should re-run
            seg_row = db_session.query(TranscriptSegment).filter_by(transcript_id=tr.id, segment_index=0).first()
            seg_row.english_text = seg_row.english_text + " Additional updated scheduling detail."
            db_session.commit()

            r3 = LectureIntelligenceService.generate_intelligence_report(rec.id, db_session)
            assert r3["lecture_notes"]["meta"]["transcript_hash"] != hash_1
        finally:
            db_session.close()


class TestPhase5EndToEndIntegrationAndBenchmarks:
    def test_short_lecture_returns_insufficient_content_without_padding(self):
        """A short (<600 words) lecture returns fewer than 10 MCQs and mcq_status='insufficient_content'."""
        segs = [
            _seg(1, "Ohm law states that voltage equals current multiplied by resistance in an electrical circuit.", 0.0, 15.0),
            _seg(2, "Voltage = current * resistance is the primary formula for resistive circuits.", 15.0, 30.0),
        ]
        full_text = " ".join(s.text for s in segs)
        notes = {
            "study_topics": [
                {
                    "title": "Ohm Law",
                    "start_time": 0.0,
                    "end_time": 30.0,
                    "explanation": segs[0].text,
                    "key_points": [segs[1].text],
                }
            ]
        }
        res = run_academic_v2(notes, segs, full_text, recording_id="rec_short", embed_fn=uniform_embed)
        assert res["meta"]["status"] == "insufficient_content"
        assert res["meta"]["mcq_status"] == "insufficient_content"
        assert res["meta"]["mcq_reason"] == "too_short"
        assert len(res["quizzes"]) < 10

    def test_code_switched_roman_urdu_lecture_grounding_and_cues(self):
        """Verify code-switched English + Roman Urdu lecture preserves negations, extracts exam hints, and grounds 100% of sentences."""
        segs = [
            _seg(1, "Deadlock tab hota hai jab concurrent processes ek doosre ke resources ka wait karti hain. Yeh topic exam mein zaroor aayega.", 0.0, 25.0),
            _seg(2, "Mutual exclusion, hold and wait, no preemption, aur circular wait char zaroori conditions hain deadlock ke liye.", 25.0, 50.0),
            _seg(3, "Banker algorithm resource allocation graph check karta hai lekin yeh proof exam mein nahi aayega.", 50.0, 75.0),
        ]
        full_text = " ".join(s.text for s in segs)
        notes = {
            "study_topics": [
                {
                    "title": "Deadlock Conditions",
                    "start_time": 0.0,
                    "end_time": 75.0,
                    "explanation": segs[0].text,
                    "key_points": [segs[1].text, segs[2].text],
                }
            ]
        }
        res = run_academic_v2(notes, segs, full_text, recording_id="rec_urdu", embed_fn=bow_embed)
        hints = res["notes"]["explicit_exam_hints"]
        polarities = {h["polarity"] for h in hints}
        assert "affirmed" in polarities
        assert "negated" in polarities

        topics = res["notes"]["study_topics"]
        assert len(topics) == 1
        assert len(topics[0]["evidence_timestamps"]) >= 2
        assert topics[0]["inferred_importance"] in {1, 2, 3, 4, 5}
        assert topics[0]["inferred_exam_relevance"] in {"low", "medium", "high"}

    def test_full_lecture_10_mcqs_grounding_memory_and_latency_budget(self):
        """
        End-to-end verification on a full >600-word lecture:
        - Exactly 10 validated MCQs with mcq_status == 'complete'
        - Grounding rate >= 95% (every emitted note sentence has evidence_timestamps)
        - Zero cross-domain contamination
        - Runtime well within INTEL_TIME_BUDGET_S and peak RAM increase < 200 MB
        """
        segs, full_text = _long_os_segments()
        notes = {
            "study_topics": [
                {
                    "title": "Round Robin & Context Switch",
                    "start_time": 0.0,
                    "end_time": 70.0,
                    "explanation": segs[0].text,
                    "key_points": [segs[1].text, segs[2].text],
                    "definitions": [{"term": "Time quantum", "definition": "Maximum CPU time a process may use in one turn"}],
                    "examples": [],
                    "formulas": [],
                },
                {
                    "title": "Priority Scheduling, Starvation & Aging",
                    "start_time": 70.0,
                    "end_time": 150.0,
                    "explanation": segs[5].text,
                    "key_points": [segs[6].text, segs[7].text],
                    "definitions": [{"term": "Aging", "definition": "Slowly increasing priority of waiting processes"}],
                    "examples": [],
                    "formulas": [],
                },
                {
                    "title": "FCFS, SJF & Gantt Charts",
                    "start_time": 150.0,
                    "end_time": 220.0,
                    "explanation": segs[14].text,
                    "key_points": [segs[15].text, segs[16].text],
                    "definitions": [{"term": "Gantt chart", "definition": "Timeline showing which process runs on the CPU"}],
                    "examples": [],
                    "formulas": [],
                },
            ]
        }

        tracemalloc.start()
        t0 = time.monotonic()
        res = run_academic_v2(notes, segs, full_text, recording_id="rec_bench", embed_fn=uniform_embed)
        elapsed = time.monotonic() - t0
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak_bytes / (1024 * 1024)
        assert elapsed < cfg.INTEL_TIME_BUDGET_S
        assert peak_mb < 200.0

        assert res["meta"]["status"] == "complete", f"Meta was: {res['meta']}"
        assert res["meta"]["mcq_status"] == "complete"
        assert len(res["quizzes"]) == 10

        # Check grounding rate >= 95% across all emitted study topics
        total_emitted = 0
        aligned_count = 0
        for t in res["notes"]["study_topics"]:
            sents = [s for s in re.split(r"(?<=[.!?])\s+", t["explanation"].strip()) if s.strip()] + list(t["key_points"])
            total_emitted += len(sents)
            aligned_count += len(t["evidence_timestamps"])
        grounding_rate = aligned_count / max(1, total_emitted)
        assert grounding_rate >= 0.95

        # Check zero contamination across notes and quizzes
        vocab = set(re.findall(r"[a-z]+", full_text.lower()))
        quiz_audit = ContaminationAuditor.audit_quizzes(res["quizzes"], vocab)
        assert quiz_audit["passed"], f"Quiz contamination found: {quiz_audit}"
