"""
Phase 13 Test Suite: Timestamp Traceability (Master Spec Section 22 + Phase 13)

Validates:
1. Timestamp accuracy spot-check across >=5 extracted items (assignments, action items, important points, Q&A).
2. Merged-record multi-chunk timestamp completeness & field-level attribution accuracy (field_timestamps).
3. API consistency across all response schemas (timestamp, start_time, end_time, source_timestamps).
4. False-merge timestamp isolation (distinct items maintain isolated intervals).
5. Grounded Q&A & Semantic Q&A citation fidelity and threshold gating.
6. REST API JSON serialization of all timestamp metadata.
"""

import os
import uuid
import pytest
from datetime import datetime, timezone
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base, get_db
from app.main import app
from app.models import Recording, Transcript, TranscriptSegment
from app.services.intelligence_service import LectureIntelligenceService
from app.services.embedding_service import EmbeddingService
from app.services.context_stitcher import ContextStitcher
from app.schemas import (
    AssignmentItem,
    ActionItem,
    ImportantPointItem,
    QAResponse,
    SemanticQAResponse,
    StitchedBlockSchema,
    SemanticSearchResultItem
)

# In-memory test SQLite session fixture
TEST_DB_URL = "sqlite:///./test_phase13_traceability.db"
engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)
    if os.path.exists("test_phase13_traceability.db"):
        try:
            os.remove("test_phase13_traceability.db")
        except Exception:
            pass


@pytest.fixture
def db_session():
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture
def client(db_session):
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


# ============================================================================
# TEST 1: Timestamp Accuracy Spot-Check (5 Extracted Items with Exact Spoken Verification)
# ============================================================================
def test_timestamp_accuracy_spot_check(db_session):
    """
    Spot-checks 5 distinct extracted items against known transcript locations:
    1. Assignment Topic (Chunk 0: [0.00s - 12.35s])
    2. Assignment Format & Deadline (Chunk 1: [12.80s - 25.40s])
    3. ImportantPointItem with Exam Signal (Chunk 2: [25.90s - 38.65s])
    4. ActionItem with Owner & Deliverable (Chunk 3: [39.10s - 51.75s])
    5. Grounded Q&A Answer Citation (Chunk 2: [25.90s - 38.65s])
    """
    rec_id = str(uuid.uuid4())
    rec = Recording(
        id=rec_id,
        filename="lecture_traceability.wav",
        file_path="uploads/lecture_traceability.wav",
        file_size_bytes=1024 * 50,
        duration_seconds=65.0,
        file_format=".wav",
        media_type="audio",
        status="PROCESSED"
    )
    db_session.add(rec)
    db_session.commit()

    transcript = Transcript(
        id=str(uuid.uuid4()),
        recording_id=rec_id,
        original_transcript="",
        normalized_transcript=""
    )
    db_session.add(transcript)
    db_session.commit()

    segments_data = [
        (0, 0.00, 12.35, "Assignment 1 is regarding Deep Learning algorithms."),
        (1, 12.80, 25.40, "Submit a 10-page PDF report for this assignment before Friday midnight on LMS portal."),
        (2, 25.90, 38.65, "Remember this concept because this will come in exam: Supervised Learning uses labeled datasets for training."),
        (3, 39.10, 51.75, "Bilal will prepare the architecture diagram by Tuesday."),
        (4, 52.20, 64.80, "Next week we will cover reinforcement learning principles.")
    ]

    segments = []
    for idx, st, et, txt in segments_data:
        seg = TranscriptSegment(
            id=str(uuid.uuid4()),
            transcript_id=transcript.id,
            recording_id=rec_id,
            chunk_index=idx,
            segment_index=idx,
            start_time=st,
            end_time=et,
            text=txt,
            normalized_text=txt,
            language="en"
        )
        db_session.add(seg)
        segments.append(seg)
    db_session.commit()

    # 1 & 2: Extract assignments
    assignments = LectureIntelligenceService.extract_assignments("", segments)
    assert len(assignments) == 1
    asgn = assignments[0]

    # Item 1: Topic timestamp verification
    assert "Deep Learning" in asgn["topic"]
    assert asgn["field_timestamps"]["topic"] == "[0.00s - 12.35s]"

    # Item 2: Format and deadline timestamp verification
    assert asgn["format"] == "PDF"
    assert asgn["field_timestamps"]["format"] == "[12.80s - 25.40s]"
    assert asgn["field_timestamps"]["deadline"] == "[12.80s - 25.40s]"
    assert asgn["field_timestamps"]["length_requirements"] == "[12.80s - 25.40s]"
    assert asgn["start_time"] == 0.00
    assert asgn["end_time"] == 25.40

    # Item 3: ImportantPointItem with exam signal verification
    notes = LectureIntelligenceService.extract_lecture_notes("", segments)
    assert len(notes["important_points"]) >= 1
    pt = notes["important_points"][0]
    assert pt["is_exam_relevant"] is True
    assert "will come in exam" in pt["signal_phrase"].lower()
    assert pt["timestamp"] == "[25.90s - 38.65s]"
    assert pt["start_time"] == 25.90
    assert pt["end_time"] == 38.65

    # Item 4: ActionItem with owner & deliverable verification
    action_items = LectureIntelligenceService.extract_action_items("", segments)
    assert len(action_items) == 1
    act = action_items[0]
    assert act["owner"] == "Bilal"
    assert "Tuesday" in act["deadline"]
    assert act["timestamp"] == "[39.10s - 51.75s]"
    assert act["start_time"] == 39.10
    assert act["end_time"] == 51.75
    assert act["field_timestamps"]["owner"] == "[39.10s - 51.75s]"
    assert act["field_timestamps"]["deadline"] == "[39.10s - 51.75s]"

    # Item 5: Grounded Q&A timestamp verification
    qa_res = LectureIntelligenceService.answer_grounded_question(
        question="What does supervised learning use for training?",
        recording_id=rec_id,
        db=db_session
    )
    assert qa_res["found"] is True
    assert qa_res["timestamp"] == "[25.90s - 38.65s]"
    assert qa_res["start_time"] == 25.90
    assert qa_res["end_time"] == 38.65
    assert "Supervised Learning" in qa_res["answer"]


# ============================================================================
# TEST 2: Merged-Record Multi-Chunk Timestamp Completeness & Field-Level Accuracy
# ============================================================================
def test_merged_record_field_level_timestamp_accuracy(db_session):
    """
    Validates that merged records preserve ALL contributing source timestamps,
    and each individual field strictly maps to the exact chunk where it was spoken:
    - Topic -> Chunk 0 ([0.00s - 18.25s])
    - Format / Length -> Chunk 1 ([18.80s - 37.10s])
    - Submission Method / Deadline -> Chunk 2 ([37.65s - 56.40s])
    """
    rec_id = str(uuid.uuid4())
    rec = Recording(
        id=rec_id,
        filename="lecture_merge.wav",
        file_path="uploads/lecture_merge.wav",
        file_size_bytes=1024 * 50,
        duration_seconds=60.0,
        file_format=".wav",
        media_type="audio",
        status="PROCESSED"
    )
    db_session.add(rec)
    db_session.commit()

    transcript = Transcript(
        id=str(uuid.uuid4()),
        recording_id=rec_id,
        original_transcript="",
        normalized_transcript=""
    )
    db_session.add(transcript)
    db_session.commit()

    seg0 = TranscriptSegment(
        id=str(uuid.uuid4()),
        transcript_id=transcript.id,
        recording_id=rec_id,
        chunk_index=0,
        segment_index=0,
        start_time=0.00,
        end_time=18.25,
        text="For Assignment 1, you have to implement gradient descent algorithm.",
        normalized_text="For Assignment 1, you have to implement gradient descent algorithm.",
        language="en"
    )
    seg1 = TranscriptSegment(
        id=str(uuid.uuid4()),
        transcript_id=transcript.id,
        recording_id=rec_id,
        chunk_index=1,
        segment_index=1,
        start_time=18.80,
        end_time=37.10,
        text="Please write a 5-page report in PDF format for this assignment.",
        normalized_text="Please write a 5-page report in PDF format for this assignment.",
        language="en"
    )
    seg2 = TranscriptSegment(
        id=str(uuid.uuid4()),
        transcript_id=transcript.id,
        recording_id=rec_id,
        chunk_index=2,
        segment_index=2,
        start_time=37.65,
        end_time=56.40,
        text="Isko Friday midnight tak LMS par submit karwana hai.",
        normalized_text="Isko Friday midnight tak LMS par submit karwana hai.",
        language="ur"
    )
    db_session.add_all([seg0, seg1, seg2])
    db_session.commit()

    assignments = LectureIntelligenceService.extract_assignments("", [seg0, seg1, seg2])
    assert len(assignments) == 1
    merged = assignments[0]

    # 1. Multi-chunk timestamp completeness
    assert len(merged["source_timestamps"]) == 3
    assert "[0.00s - 18.25s]" in merged["source_timestamps"]
    assert "[18.80s - 37.10s]" in merged["source_timestamps"]
    assert "[37.65s - 56.40s]" in merged["source_timestamps"]
    assert merged["start_time"] == 0.00
    assert merged["end_time"] == 56.40
    assert "[0.00s - 18.25s]" in merged["timestamp"]
    assert "[37.65s - 56.40s]" in merged["timestamp"]

    # 2. Strict field-level timestamp accuracy
    fts = merged["field_timestamps"]
    assert fts["topic"] == "[0.00s - 18.25s]"
    assert fts["format"] == "[18.80s - 37.10s]"
    assert fts["length_requirements"] == "[18.80s - 37.10s]"
    assert fts["submission_method"] == "[37.65s - 56.40s]"
    assert fts["deadline"] == "[37.65s - 56.40s]"

    # Action item 2-chunk merge field timestamp accuracy
    act_seg0 = TranscriptSegment(
        id=str(uuid.uuid4()),
        transcript_id=transcript.id,
        recording_id=rec_id,
        chunk_index=3,
        segment_index=3,
        start_time=60.00,
        end_time=74.50,
        text="Bilal will prepare the data preprocessing pipeline.",
        normalized_text="Bilal will prepare the data preprocessing pipeline.",
        language="en"
    )
    act_seg1 = TranscriptSegment(
        id=str(uuid.uuid4()),
        transcript_id=transcript.id,
        recording_id=rec_id,
        chunk_index=4,
        segment_index=4,
        start_time=75.10,
        end_time=89.20,
        text="He will send the csv file by Monday.",
        normalized_text="He will send the csv file by Monday.",
        language="en"
    )
    action_items = LectureIntelligenceService.extract_action_items("", [act_seg0, act_seg1])
    assert len(action_items) == 1
    act_merged = action_items[0]

    assert len(act_merged["source_timestamps"]) == 2
    assert act_merged["start_time"] == 60.00
    assert act_merged["end_time"] == 89.20
    assert act_merged["field_timestamps"]["owner"] == "[60.00s - 74.50s]"
    assert act_merged["field_timestamps"]["deliverables"] == "[75.10s - 89.20s]"
    assert act_merged["field_timestamps"]["deadline"] == "[75.10s - 89.20s]"


# ============================================================================
# TEST 3: Schema API Consistency
# ============================================================================
def test_schema_api_timestamp_consistency():
    """
    Validates uniform presence and typing of timestamp fields across all intelligence schemas:
    - timestamp (str)
    - start_time (Optional[float])
    - end_time (Optional[float])
    """
    asgn = AssignmentItem(
        topic="Gradient Descent",
        timestamp="[0.00s - 18.25s]",
        start_time=0.00,
        end_time=18.25,
        source_timestamps=["[0.00s - 18.25s]"],
        field_timestamps={"topic": "[0.00s - 18.25s]"}
    )
    assert asgn.start_time == 0.00
    assert asgn.end_time == 18.25
    assert len(asgn.source_timestamps) == 1

    act = ActionItem(
        task="Prepare slides",
        owner="Ali",
        timestamp="[10.00s - 20.00s]",
        start_time=10.00,
        end_time=20.00
    )
    assert act.start_time == 10.00
    assert act.end_time == 20.00

    point = ImportantPointItem(
        point="Linear regression is a linear model",
        timestamp="[5.00s - 15.00s]",
        start_time=5.00,
        end_time=15.00
    )
    assert point.start_time == 5.00
    assert point.end_time == 15.00

    qa = QAResponse(
        found=True,
        question="What is linear regression?",
        answer="A linear model",
        timestamp="[5.00s - 15.00s]",
        start_time=5.00,
        end_time=15.00
    )
    assert qa.start_time == 5.00
    assert qa.end_time == 15.00

    sqa = SemanticQAResponse(
        found=True,
        question="Explain linear regression",
        answer="A linear model",
        timestamp="[5.00s - 15.00s]",
        start_time=5.00,
        end_time=15.00,
        similarity_score=0.88
    )
    assert sqa.start_time == 5.00
    assert sqa.end_time == 15.00

    block = StitchedBlockSchema(
        block_index=0,
        start_time=0.00,
        end_time=30.00,
        timestamp="[0.00s - 30.00s]",
        text="Sample text",
        normalized_text="Sample text",
        source_chunk_indices="0,1"
    )
    assert block.start_time == 0.00
    assert block.end_time == 30.00

    search_item = SemanticSearchResultItem(
        segment_id="seg1",
        chunk_index=0,
        segment_index=0,
        start_time=0.00,
        end_time=15.00,
        timestamp="[0.00s - 15.00s]",
        text="Sample",
        normalized_text="Sample",
        similarity_score=0.75
    )
    assert search_item.start_time == 0.00
    assert search_item.end_time == 15.00


# ============================================================================
# TEST 4: False-Merge Timestamp Isolation
# ============================================================================
def test_false_merge_timestamp_isolation(db_session):
    """
    Validates that distinct items (Assignment 1 vs Assignment 2) maintain
    strictly isolated, non-overlapping timestamps and field attributions.
    """
    rec_id = str(uuid.uuid4())
    rec = Recording(
        id=rec_id,
        filename="lecture_isolation.wav",
        file_path="uploads/lecture_isolation.wav",
        file_size_bytes=1024 * 50,
        duration_seconds=70.0,
        file_format=".wav",
        media_type="audio",
        status="PROCESSED"
    )
    db_session.add(rec)
    db_session.commit()

    transcript = Transcript(
        id=str(uuid.uuid4()),
        recording_id=rec_id,
        original_transcript="",
        normalized_transcript=""
    )
    db_session.add(transcript)
    db_session.commit()

    seg1 = TranscriptSegment(
        id=str(uuid.uuid4()),
        transcript_id=transcript.id,
        recording_id=rec_id,
        chunk_index=0,
        segment_index=0,
        start_time=0.00,
        end_time=19.45,
        text="Assignment 1 is due Friday midnight on LMS.",
        normalized_text="Assignment 1 is due Friday midnight on LMS.",
        language="en"
    )
    seg2 = TranscriptSegment(
        id=str(uuid.uuid4()),
        transcript_id=transcript.id,
        recording_id=rec_id,
        chunk_index=1,
        segment_index=1,
        start_time=20.10,
        end_time=39.80,
        text="Assignment 2 will be a group report due next month.",
        normalized_text="Assignment 2 will be a group report due next month.",
        language="en"
    )
    db_session.add_all([seg1, seg2])
    db_session.commit()

    assignments = LectureIntelligenceService.extract_assignments("", [seg1, seg2])
    assert len(assignments) == 2

    # Assignment 1 timestamps
    a1 = assignments[0]
    assert a1["start_time"] == 0.00
    assert a1["end_time"] == 19.45
    assert a1["source_timestamps"] == ["[0.00s - 19.45s]"]
    assert a1["field_timestamps"]["deadline"] == "[0.00s - 19.45s]"

    # Assignment 2 timestamps
    a2 = assignments[1]
    assert a2["start_time"] == 20.10
    assert a2["end_time"] == 39.80
    assert a2["source_timestamps"] == ["[20.10s - 39.80s]"]
    assert a2["field_timestamps"]["deadline"] == "[20.10s - 39.80s]"


# ============================================================================
# TEST 5: Semantic Q&A Acceptance Gating & Citation Fidelity
# ============================================================================
def test_semantic_qa_threshold_gating_and_citation(db_session):
    """
    Validates that Q&A citations are strictly emitted when score >= 0.50 threshold,
    with exact start_time and end_time metadata. Unmentioned queries return None.
    """
    rec_id = str(uuid.uuid4())
    rec = Recording(
        id=rec_id,
        filename="lecture_qa.wav",
        file_path="uploads/lecture_qa.wav",
        file_size_bytes=1024 * 50,
        duration_seconds=50.0,
        file_format=".wav",
        media_type="audio",
        status="PROCESSED"
    )
    db_session.add(rec)
    db_session.commit()

    transcript = Transcript(
        id=str(uuid.uuid4()),
        recording_id=rec_id,
        original_transcript="",
        normalized_transcript=""
    )
    db_session.add(transcript)
    db_session.commit()

    seg = TranscriptSegment(
        id=str(uuid.uuid4()),
        transcript_id=transcript.id,
        recording_id=rec_id,
        chunk_index=0,
        segment_index=0,
        start_time=14.20,
        end_time=29.80,
        text="Convolutional Neural Networks are primarily used for image processing and computer vision tasks.",
        normalized_text="Convolutional Neural Networks are primarily used for image processing and computer vision tasks.",
        language="en"
    )
    db_session.add(seg)
    db_session.commit()

    # Generate embeddings
    EmbeddingService.generate_and_store_embeddings(recording_id=rec_id, db=db_session)

    # 1. Genuine query (should pass >= 0.50 threshold and return exact citation)
    res = EmbeddingService.answer_semantic_qa(
        recording_id=rec_id,
        question="What are CNN models applied to in computer vision?",
        threshold=0.50,
        db=db_session
    )
    assert res["found"] is True
    assert res["similarity_score"] >= 0.50
    assert res["timestamp"] == "[14.20s - 29.80s]"
    assert res["start_time"] == 14.20
    assert res["end_time"] == 29.80

    # 2. Unmentioned query (must return found=False, timestamp=None, start_time=None)
    unmentioned_res = EmbeddingService.answer_semantic_qa(
        recording_id=rec_id,
        question="What is the weather forecast for tomorrow?",
        threshold=0.50,
        db=db_session
    )
    assert unmentioned_res["found"] is False
    assert unmentioned_res["timestamp"] is None
    assert unmentioned_res["start_time"] is None
    assert unmentioned_res["end_time"] is None


# ============================================================================
# TEST 6: REST API JSON Serialization of Timestamp Fields
# ============================================================================
def test_rest_api_timestamp_serialization(client, db_session):
    """
    Validates REST API responses correctly serialize start_time, end_time,
    and field_timestamps in JSON format.
    """
    rec_id = str(uuid.uuid4())
    rec = Recording(
        id=rec_id,
        filename="lecture_api.wav",
        file_path="uploads/lecture_api.wav",
        file_size_bytes=1024 * 50,
        duration_seconds=40.0,
        file_format=".wav",
        media_type="audio",
        status="PROCESSED"
    )
    db_session.add(rec)
    db_session.commit()

    transcript = Transcript(
        id=str(uuid.uuid4()),
        recording_id=rec_id,
        original_transcript="",
        normalized_transcript=""
    )
    db_session.add(transcript)
    db_session.commit()

    seg = TranscriptSegment(
        id=str(uuid.uuid4()),
        transcript_id=transcript.id,
        recording_id=rec_id,
        chunk_index=0,
        segment_index=0,
        start_time=5.50,
        end_time=19.75,
        text="Assignment 1 is about Neural Networks due Friday midnight.",
        normalized_text="Assignment 1 is about Neural Networks due Friday midnight.",
        language="en"
    )
    db_session.add(seg)
    db_session.commit()

    # Generate embeddings
    EmbeddingService.generate_and_store_embeddings(recording_id=rec_id, db=db_session)

    # 1. POST /api/recordings/{id}/intelligence/generate
    intel_res = client.post(f"/api/recordings/{rec_id}/intelligence/generate")
    assert intel_res.status_code == 200
    intel_data = intel_res.json()
    assert len(intel_data["assignments"]) == 1
    asgn = intel_data["assignments"][0]
    assert asgn["start_time"] == 5.50
    assert asgn["end_time"] == 19.75
    assert asgn["timestamp"] == "[5.50s - 19.75s]"
    assert "topic" in asgn["field_timestamps"]

    # 2. POST /api/recordings/{id}/intelligence/qa
    qa_res = client.post(
        f"/api/recordings/{rec_id}/intelligence/qa",
        json={"question": "When is Assignment 1 due?"}
    )
    assert qa_res.status_code == 200
    qa_data = qa_res.json()
    assert qa_data["found"] is True
    assert qa_data["start_time"] == 5.50
    assert qa_data["end_time"] == 19.75

    # 3. POST /api/recordings/{id}/qa-semantic
    sqa_res = client.post(
        f"/api/recordings/{rec_id}/qa-semantic",
        json={"question": "What is the deadline for Neural Networks assignment?", "threshold": 0.50}
    )
    assert sqa_res.status_code == 200
    sqa_data = sqa_res.json()
    assert sqa_data["found"] is True
    assert sqa_data["start_time"] == 5.50
    assert sqa_data["end_time"] == 19.75
