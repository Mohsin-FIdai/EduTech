"""
tests/test_manual_correction.py — Unit & Integration tests for Manual Correction Workflow with Downstream Recomputation.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.database import SessionLocal
from app.models import Recording, Transcript, TranscriptSegment, IntelligenceReport, SegmentEmbedding
from app.services.transcription_service import TranscriptionService
from app.services.embedding_service import EmbeddingService
from app.services.intelligence_service import LectureIntelligenceService


@pytest.fixture
def db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def test_data(db_session):
    # Setup isolated recording, transcript, and segments
    rec = Recording(
        filename="test_os_lecture.mp4",
        file_path="storage/uploads/test_os_lecture.mp4",
        file_format=".mp4",
        media_type="video/mp4",
        duration_seconds=120.0,
        file_size_bytes=1024 * 100,
        status="COMPLETED"
    )
    db_session.add(rec)
    db_session.flush()

    tr = Transcript(
        recording_id=rec.id,
        original_transcript="Today we discuss process scheduling. Paging can reduce starvation. The assignment is due on 18 September.",
        normalized_transcript="Today we discuss process scheduling. aging can reduce starvation. The assignment is due on 18 September.",
        language_summary="en: 100%"
    )
    db_session.add(tr)
    db_session.flush()

    seg1 = TranscriptSegment(
        transcript_id=tr.id,
        recording_id=rec.id,
        chunk_index=0,
        segment_index=0,
        start_time=0.0,
        end_time=10.0,
        text="Today we discuss process scheduling.",
        normalized_text="Today we discuss process scheduling.",
        language="en",
        confidence=0.95
    )
    seg2 = TranscriptSegment(
        transcript_id=tr.id,
        recording_id=rec.id,
        chunk_index=0,
        segment_index=1,
        start_time=10.0,
        end_time=25.0,
        text="Paging can reduce starvation in priority scheduling.",
        normalized_text="aging can reduce starvation in priority scheduling.",
        language="en",
        confidence=0.92
    )
    seg3 = TranscriptSegment(
        transcript_id=tr.id,
        recording_id=rec.id,
        chunk_index=1,
        segment_index=2,
        start_time=25.0,
        end_time=45.0,
        text="The assignment is due on 18 September on the portal.",
        normalized_text="The assignment is due on 18 September on the portal.",
        language="en",
        confidence=0.98
    )
    db_session.add_all([seg1, seg2, seg3])
    db_session.commit()

    return {
        "recording": rec,
        "transcript": tr,
        "segments": [seg1, seg2, seg3]
    }


def test_segment_correction_persistence_and_timestamp_integrity(db_session, test_data):
    rec = test_data["recording"]
    seg = test_data["segments"][1]

    orig_start = seg.start_time
    orig_end = seg.end_time
    orig_chunk = seg.chunk_index
    orig_seg_idx = seg.segment_index

    test_norm_text = "aging solves starvation in priority scheduling through gradual priority elevation."
    test_raw_text = "Paging solves starvation in priority scheduling."

    res = TranscriptionService.update_segment_correction(
        recording_id=rec.id,
        segment_id=seg.id,
        db=db_session,
        normalized_text=test_norm_text,
        text=test_raw_text,
        scope="chunk"
    )

    assert res["status"] == "success"
    assert res["segment"]["id"] == seg.id
    assert res["segment"]["start_time"] == orig_start
    assert res["segment"]["end_time"] == orig_end
    assert res["segment"]["chunk_index"] == orig_chunk
    assert res["segment"]["segment_index"] == orig_seg_idx
    assert res["segment"]["normalized_text"] == test_norm_text
    assert res["segment"]["text"] == test_raw_text

    # Verify DB persistence
    db_session.refresh(seg)
    assert seg.normalized_text == test_norm_text
    assert seg.text == test_raw_text
    assert seg.start_time == orig_start
    assert seg.end_time == orig_end

    # Verify parent Transcript re-stitching
    parent_transcript = db_session.query(Transcript).filter(Transcript.id == seg.transcript_id).first()
    assert parent_transcript is not None
    assert test_norm_text in parent_transcript.normalized_transcript
    assert test_raw_text in parent_transcript.original_transcript


def test_downstream_full_scope_recomputation(db_session, test_data):
    rec = test_data["recording"]
    seg = test_data["segments"][2]

    # Correct assignment deadline
    corrected_assignment = "The homework assignment deadline is 25 September at 11:59 PM on the LMS portal."
    res = TranscriptionService.update_segment_correction(
        recording_id=rec.id,
        segment_id=seg.id,
        db=db_session,
        normalized_text=corrected_assignment,
        scope="full"
    )

    assert res["status"] == "success"
    assert res["scope"] == "full"
    assert res["recomputed"]["summary"] is True
    assert res["recomputed"]["lecture_notes"] is True
    assert res["recomputed"]["assignments"] is True
    assert res["recomputed"]["quizzes"] is True
    assert res["recomputed"]["embeddings"] is True
    assert res["intelligence"] is not None
    assert "summary_short" in res["intelligence"]

    # Verify IntelligenceReport persisted in DB
    report = db_session.query(IntelligenceReport).filter(IntelligenceReport.recording_id == rec.id).first()
    assert report is not None
    assert len(report.summary_short) > 10


def test_api_endpoint_segment_correction(client, test_data):
    rec = test_data["recording"]
    seg = test_data["segments"][0]

    payload = {
        "normalized_text": "Today we begin CPU Process Scheduling and Round Robin algorithms.",
        "text": "Today we begin CPU Process Scheduling.",
        "scope": "full"
    }

    resp = client.post(f"/api/recordings/{rec.id}/segments/{seg.id}/correct", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["segment"]["normalized_text"] == payload["normalized_text"]
    assert data["recomputed"]["embeddings"] is True
    assert data["recomputed"]["summary"] is True


def test_qa_grounding_reflects_updated_correction(db_session, test_data):
    rec = test_data["recording"]
    seg = test_data["segments"][0]

    # Run full recompute on segment
    TranscriptionService.update_segment_correction(
        recording_id=rec.id,
        segment_id=seg.id,
        db=db_session,
        normalized_text="Today we study CPU process scheduling algorithms and turnaround time.",
        scope="full"
    )

    # Test Q&A on the recording
    qa_res = LectureIntelligenceService.answer_grounded_question(
        "What is process scheduling?",
        rec.id,
        db_session
    )
    assert qa_res is not None
    assert qa_res["found"] is True
    assert len(qa_res["answer"]) > 20
