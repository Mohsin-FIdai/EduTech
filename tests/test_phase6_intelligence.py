"""
test_phase6_intelligence.py — Acceptance tests for Phase 6: Lecture & Meeting Intelligence Layer.

Acceptance Matrix:
1. Planted-Content Assignment Extraction Accuracy (Exact field match & 100% 'Not specified' rate on unstated fields).
2. Planted-Content Action Items & Meeting Mode (Named owners vs unstated owners).
3. Grounded Q&A Verification (Answerable + Completely Unmentioned + Topical Near-Miss).
4. Global Multi-Chunk Summary Coverage (Spanning Early, Middle, Late sections).
5. Explicit Exam-Relevance Signal Phrase Gating (Signal phrase required for exam flag).
6. Quiz Groundedness & REST API Endpoints (/generate, /intelligence, /qa, /quizzes).
"""

import tempfile
import os
import pytest
from fastapi.testclient import TestClient
import soundfile as sf
import numpy as np

from app.main import app
from app.database import SessionLocal
from app.models import Recording, Transcript, TranscriptSegment
from app.services.intelligence_service import LectureIntelligenceService


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def _create_mock_recording_with_segments(segments_data: list[dict], duration: float = 30.0) -> tuple[str, str]:
    """Helper to create a recording, transcript, and segments in the database."""
    tmp_wav = tempfile.mktemp(suffix=".wav")
    sr = 16000
    sf.write(tmp_wav, np.zeros(int(sr * 1.0), dtype=np.float32), sr, subtype="PCM_16")

    db = SessionLocal()
    try:
        rec = Recording(
            filename="intelligence_test.wav",
            file_path=tmp_wav,
            file_format=".wav",
            media_type="audio",
            file_size_bytes=1024,
            duration_seconds=duration,
            status="transcribed"
        )
        db.add(rec)
        db.flush()

        full_text = " ".join([s["text"] for s in segments_data])
        tx = Transcript(
            recording_id=rec.id,
            original_transcript=full_text,
            normalized_transcript=full_text,
            language_summary="en: 80%, ur: 20%"
        )
        db.add(tx)
        db.flush()

        for i, s in enumerate(segments_data):
            seg = TranscriptSegment(
                transcript_id=tx.id,
                recording_id=rec.id,
                chunk_index=s.get("chunk_index", 0),
                segment_index=i,
                start_time=s["start_time"],
                end_time=s["end_time"],
                text=s["text"],
                normalized_text=s.get("normalized_text", s["text"]),
                language="en",
                confidence=0.95
            )
            db.add(seg)

        db.commit()
        return rec.id, tx.id
    finally:
        db.close()
        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)


# ============================================================================
# Test 1: Planted-Content Assignment Extraction Accuracy
# ============================================================================
def test_planted_assignment_extraction_accuracy():
    """
    Field-by-field verification against planted assignments:
    - Assignment 1: Fully specified.
    - Assignment 2: Partially specified (deliberately unstated format & submission method).
    - Assignment 3: Ambiguous deadline ('next class') preserved verbatim.
    """
    planted_segments = [
        TranscriptSegment(
            chunk_index=0, segment_index=0, start_time=5.0, end_time=15.0,
            text="Assignment 1 is on Neural Networks, you have to implement a multi-layer perceptron in Python, format must be Jupyter notebook, submit through LMS portal before Friday midnight.",
            normalized_text="Assignment 1 is on Neural Networks, you have to implement a multi-layer perceptron in Python, format must be Jupyter notebook, submit through LMS portal before Friday midnight."
        ),
        TranscriptSegment(
            chunk_index=1, segment_index=1, start_time=20.0, end_time=30.0,
            text="Assignment 2 is on Support Vector Machines, deadline is next week Monday, all students must complete it individually.",
            normalized_text="Assignment 2 is on Support Vector Machines, deadline is next week Monday, all students must complete it individually."
        ),
        TranscriptSegment(
            chunk_index=2, segment_index=2, start_time=35.0, end_time=45.0,
            text="Homework 3 is about Deep Learning models, submit by next class, group of 3 students.",
            normalized_text="Homework 3 is about Deep Learning models, submit by next class, group of 3 students."
        )
    ]

    extracted = LectureIntelligenceService.extract_assignments(" ".join([s.text for s in planted_segments]), planted_segments)
    assert len(extracted) == 3

    # Assignment 1 Verification
    a1 = extracted[0]
    assert "Neural Networks" in a1["topic"]
    assert a1["format"] == "Jupyter Notebook (.ipynb)"
    assert a1["submission_method"] == "LMS Portal"
    assert "Friday" in a1["deadline"]
    assert "Python" in a1["required_tools"]
    assert a1["length_requirements"] == "Not specified"  # Unstated
    assert a1["individual_or_group"] == "Not specified"   # Unstated
    assert a1["responsible_person_or_group"] == "Not specified" # Unstated
    assert a1["confidence"] == "High"

    # Assignment 2 Verification (Format & Submission Method deliberately unstated)
    a2 = extracted[1]
    assert "Support Vector Machines" in a2["topic"]
    assert a2["format"] == "Not specified"               # MUST NOT hallucinate
    assert a2["submission_method"] == "Not specified"    # MUST NOT hallucinate
    assert "Monday" in a2["deadline"]
    assert a2["individual_or_group"] == "Individual"
    assert a2["responsible_person_or_group"] == "All Students"

    # Assignment 3 Verification (Ambiguous deadline preserved verbatim)
    a3 = extracted[2]
    assert "Deep Learning" in a3["topic"]
    assert "next class" in a3["deadline"].lower()         # Preserved verbatim, not guessed date
    assert "Group Of 3" in a3["individual_or_group"]
    assert a3["format"] == "Not specified"
    assert a3["submission_method"] == "Not specified"


# ============================================================================
# Test 2: Planted-Content Action Items & Meeting Mode
# ============================================================================
def test_planted_action_items_meeting_mode():
    """
    Verify meeting action item extraction with named owners vs unstated owners.
    """
    planted_segments = [
        TranscriptSegment(
            chunk_index=0, segment_index=0, start_time=2.0, end_time=8.0,
            text="Ali will prepare the slides by Monday.",
            normalized_text="Ali will prepare the slides by Monday."
        ),
        TranscriptSegment(
            chunk_index=1, segment_index=1, start_time=10.0, end_time=16.0,
            text="We decided that everyone will review the report before next meeting.",
            normalized_text="We decided that everyone will review the report before next meeting."
        ),
        TranscriptSegment(
            chunk_index=2, segment_index=2, start_time=18.0, end_time=24.0,
            text="Usman will finalize the excel sheet tomorrow.",
            normalized_text="Usman will finalize the excel sheet tomorrow."
        )
    ]

    action_items = LectureIntelligenceService.extract_action_items(" ".join([s.text for s in planted_segments]), planted_segments)
    assert len(action_items) == 3

    # Item 1: Named owner
    ai1 = action_items[0]
    assert ai1["owner"] == "Ali"
    assert "Monday" in ai1["deadline"]
    assert ai1["deliverables"] == "Slides"
    assert ai1["confidence"] == "High"

    # Item 2: Unstated owner (MUST be 'Not specified')
    ai2 = action_items[1]
    assert ai2["owner"] == "Not specified"                # MUST NOT invent name
    assert "before next meeting" in ai2["deadline"]
    assert ai2["deliverables"] == "Report"

    # Item 3: Named owner
    ai3 = action_items[2]
    assert ai3["owner"] == "Usman"
    assert "tomorrow" in ai3["deadline"]
    assert ai3["deliverables"] == "Excel Sheet"


# ============================================================================
# Test 3: Grounded Q&A Verification (Answerable + Unmentioned + Near-Miss)
# ============================================================================
def test_grounded_qa_verification():
    """
    Test 6 grounded questions:
    Q1: Early segment (definition) -> Found + timestamp.
    Q2: Middle segment (technical details) -> Found + timestamp.
    Q3: Late segment (assignment deadline) -> Found + timestamp.
    Q4: Code-switched Urdu segment -> Found + timestamp.
    Q5: Completely unmentioned topic -> 'Not found in the recording'.
    Q6: Topical near-miss (unsupervised clustering vs supervised classification) -> 'Not found in the recording'.
    """
    segments_data = [
        {"start_time": 0.0, "end_time": 10.0, "text": "Machine Learning is defined as the study of computer algorithms that improve automatically through experience.", "normalized_text": "Machine Learning is defined as the study of computer algorithms that improve automatically through experience."},
        {"start_time": 10.0, "end_time": 20.0, "text": "In supervised classification, the model is trained on labeled feature datasets.", "normalized_text": "In supervised classification, the model is trained on labeled feature datasets."},
        {"start_time": 20.0, "end_time": 30.0, "text": "Assignment 1 deadline is Friday midnight through LMS portal.", "normalized_text": "Assignment 1 deadline is Friday midnight through LMS portal."},
        {"start_time": 30.0, "end_time": 40.0, "text": "Agle hafte quiz hoga aur sab ko prepare karna hai.", "normalized_text": "Agle hafte quiz hoga aur sab ko prepare karna hai."}
    ]

    rec_id, tx_id = _create_mock_recording_with_segments(segments_data, duration=40.0)
    db = SessionLocal()
    try:
        # Q1: Early
        res1 = LectureIntelligenceService.answer_grounded_question("What is the definition of Machine Learning?", rec_id, db)
        assert res1["found"] is True
        assert "algorithms" in res1["answer"].lower()
        assert res1["timestamp"] == "[0.00s - 10.00s]"

        # Q2: Middle
        res2 = LectureIntelligenceService.answer_grounded_question("How is the model trained in supervised classification?", rec_id, db)
        assert res2["found"] is True
        assert "labeled" in res2["answer"].lower()
        assert res2["timestamp"] == "[10.00s - 20.00s]"

        # Q3: Late
        res3 = LectureIntelligenceService.answer_grounded_question("What is the submission deadline for Assignment 1?", rec_id, db)
        assert res3["found"] is True
        assert "friday midnight" in res3["answer"].lower()
        assert res3["timestamp"] == "[20.00s - 30.00s]"

        # Q4: Code-switched Urdu
        res4 = LectureIntelligenceService.answer_grounded_question("Agle hafte quiz hoga kya?", rec_id, db)
        assert res4["found"] is True
        assert "quiz" in res4["answer"].lower()
        assert res4["timestamp"] == "[30.00s - 40.00s]"

        # Q5: Completely unmentioned
        res5 = LectureIntelligenceService.answer_grounded_question("What is quantum computing error mitigation protocol?", rec_id, db)
        assert res5["found"] is False
        assert res5["answer"] == "Not found in the recording"
        assert res5["timestamp"] is None

        # Q6: Topical near-miss (asking about unsupervised clustering when only supervised classification was taught)
        res6 = LectureIntelligenceService.answer_grounded_question("What algorithm is used for unsupervised clustering?", rec_id, db)
        assert res6["found"] is False
        assert res6["answer"] == "Not found in the recording"
        assert res6["timestamp"] is None
    finally:
        db.close()


# ============================================================================
# Test 4: Global Multi-Chunk Summary Temporal Coverage
# ============================================================================
def test_global_multichunk_summary_coverage():
    """
    Verify summary integrates content across early, middle, and late temporal sections.
    """
    segments = [
        TranscriptSegment(chunk_index=0, segment_index=0, start_time=0.0, end_time=25.0, text="Introduction to Artificial Intelligence and history.", normalized_text="Introduction to Artificial Intelligence and history."),
        TranscriptSegment(chunk_index=1, segment_index=1, start_time=25.0, end_time=50.0, text="Neural network backpropagation mathematics and loss gradient descent.", normalized_text="Neural network backpropagation mathematics and loss gradient descent."),
        TranscriptSegment(chunk_index=2, segment_index=2, start_time=50.0, end_time=75.0, text="Final project submission guidelines and semester closing remarks.", normalized_text="Final project submission guidelines and semester closing remarks.")
    ]

    short_sum, detailed_sum = LectureIntelligenceService.generate_global_summary("full transcript", segments)

    # Verify both early, middle, and late content are reflected in detailed summary
    assert "Artificial Intelligence" in detailed_sum or "Introduction" in detailed_sum
    assert "Neural network" in detailed_sum or "backpropagation" in detailed_sum
    assert "project" in detailed_sum or "closing" in detailed_sum


# ============================================================================
# Test 5: Explicit Exam-Relevance Signal Phrase Gating
# ============================================================================
def test_explicit_exam_relevance_signal_phrase_gating():
    """
    Verify points are flagged as exam-relevant ONLY when an explicit signal phrase is present.
    """
    segments = [
        TranscriptSegment(
            chunk_index=0, segment_index=0, start_time=5.0, end_time=15.0,
            text="Remember this for the exam, backpropagation calculates the gradient of the loss function.",
            normalized_text="Remember this for the exam, backpropagation calculates the gradient of the loss function."
        ),
        TranscriptSegment(
            chunk_index=1, segment_index=1, start_time=15.0, end_time=25.0,
            text="Convolutional neural networks use kernel filters across feature maps.",
            normalized_text="Convolutional neural networks use kernel filters across feature maps."
        ),
        TranscriptSegment(
            chunk_index=2, segment_index=2, start_time=25.0, end_time=35.0,
            text="Ye note kar lein, yeh topic midterm mein aayega lazmi.",
            normalized_text="Ye note kar lein, yeh topic midterm mein aayega lazmi."
        )
    ]

    notes = LectureIntelligenceService.extract_lecture_notes("full text", segments)
    imp_points = notes["important_points"]

    assert len(imp_points) == 2  # Only 2 have explicit signal phrases

    # Point 1 (English signal)
    p1 = imp_points[0]
    assert p1["is_exam_relevant"] is True
    assert "remember this for the exam" in p1["signal_phrase"].lower()

    # Point 2 (Roman Urdu signal)
    p2 = imp_points[1]
    assert p2["is_exam_relevant"] is True
    assert "note kar" in p2["signal_phrase"].lower() or "midterm mein aayega" in p2["signal_phrase"].lower()


# ============================================================================
# Test 6: Quiz Groundedness & REST API Endpoints
# ============================================================================
def test_quiz_groundedness_and_rest_endpoints(client):
    """
    Test REST endpoints:
    1. POST /api/recordings/{id}/intelligence/generate
    2. GET /api/recordings/{id}/intelligence
    3. POST /api/recordings/{id}/intelligence/qa
    4. GET /api/recordings/{id}/intelligence/quizzes
    And verify generated quiz questions trace strictly to extracted concepts.
    """
    segments_data = [
        {"start_time": 0.0, "end_time": 10.0, "text": "Deep Learning is defined as representation learning with artificial neural networks.", "normalized_text": "Deep Learning is defined as representation learning with artificial neural networks."},
        {"start_time": 10.0, "end_time": 20.0, "text": "Ali will prepare the dataset report by Monday.", "normalized_text": "Ali will prepare the dataset report by Monday."},
        {"start_time": 20.0, "end_time": 30.0, "text": "Assignment 1 on Machine Learning is due next class.", "normalized_text": "Assignment 1 on Machine Learning is due next class."}
    ]

    rec_id, tx_id = _create_mock_recording_with_segments(segments_data, duration=30.0)

    # 1. POST /api/recordings/{id}/intelligence/generate
    gen_res = client.post(f"/api/recordings/{rec_id}/intelligence/generate")
    assert gen_res.status_code == 200
    gen_data = gen_res.json()
    assert gen_data["status"] == "success"
    assert gen_data["summary_short"]
    assert len(gen_data["assignments"]) >= 1
    assert len(gen_data["action_items"]) >= 1
    assert len(gen_data["quizzes"]) >= 1

    # Verify quiz groundedness
    quiz = gen_data["quizzes"][0]
    grounded_concept = quiz["grounded_concept"]
    all_concepts = [c.lower() for c in gen_data["lecture_notes"]["key_concepts"]] + [d["term"].lower() for d in gen_data["lecture_notes"]["definitions"]]
    assert grounded_concept.lower() in all_concepts

    # 2. GET /api/recordings/{id}/intelligence
    get_res = client.get(f"/api/recordings/{rec_id}/intelligence")
    assert get_res.status_code == 200
    assert get_res.json()["recording_id"] == rec_id

    # 3. POST /api/recordings/{id}/intelligence/qa
    qa_res = client.post(f"/api/recordings/{rec_id}/intelligence/qa", json={"question": "What is the definition of Deep Learning?"})
    assert qa_res.status_code == 200
    assert qa_res.json()["found"] is True
    assert "[0.00s - 10.00s]" in qa_res.json()["timestamp"]

    # 4. GET /api/recordings/{id}/intelligence/quizzes
    quiz_res = client.get(f"/api/recordings/{rec_id}/intelligence/quizzes")
    assert quiz_res.status_code == 200
    assert quiz_res.json()["total_quizzes"] >= 1
