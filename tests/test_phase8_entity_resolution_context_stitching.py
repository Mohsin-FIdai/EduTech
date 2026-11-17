"""
tests/test_phase8_entity_resolution_context_stitching.py — Acceptance Test Suite for Phase 8:
Cross-Chunk Entity Resolution & Context Stitching.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.main import app
from app.database import SessionLocal, Base, engine
from app.models import Recording, Transcript, TranscriptSegment, Entity, EntityMention, StitchedBlock, RecordingStatus
from app.services.context_stitcher import ContextStitcher
from app.services.entity_resolver import EntityResolver

client = TestClient(app)


@pytest.fixture(autouse=True)
def init_db():
    Base.metadata.create_all(bind=engine)
    yield


def _create_mock_recording_with_segments(segments_data: list[dict], duration: float = 60.0) -> tuple[str, str]:
    """Helper to populate a mock recording and transcript with explicit chunk indices."""
    db: Session = SessionLocal()
    try:
        rec = Recording(
            filename="test_phase8_lecture.wav",
            file_path="mock/path/test_phase8_lecture.wav",
            file_format=".wav",
            media_type="audio",
            file_size_bytes=1024000,
            duration_seconds=duration,
            status=RecordingStatus.COMPLETED
        )
        db.add(rec)
        db.flush()

        tx = Transcript(
            recording_id=rec.id,
            original_transcript=" ".join(s["text"] for s in segments_data),
            normalized_transcript=" ".join(s.get("normalized_text", s["text"]) for s in segments_data),
            language_summary="en: 70%, ur: 30%"
        )
        db.add(tx)
        db.flush()

        for idx, s in enumerate(segments_data):
            seg = TranscriptSegment(
                transcript_id=tx.id,
                recording_id=rec.id,
                chunk_index=s.get("chunk_index", 0),
                segment_index=s.get("segment_index", idx),
                start_time=s["start_time"],
                end_time=s["end_time"],
                text=s["text"],
                normalized_text=s.get("normalized_text", s["text"]),
                language=s.get("language", "en"),
                confidence=0.95
            )
            db.add(seg)

        db.commit()
        return rec.id, tx.id
    finally:
        db.close()


# ============================================================================
# Test 1: Cross-Chunk Boundary Stitching
# ============================================================================
def test_cross_chunk_boundary_stitching():
    """
    Verify sentence split across chunk boundaries without terminal punctuation is stitched.
    Uses realistic non-round chunker timestamps matching Phase 3 silence-split outputs.
    """
    segments_data = [
        {
            "chunk_index": 0,
            "segment_index": 0,
            "start_time": 0.00,
            "end_time": 21.34,
            "text": "In today's lecture we will explore gradient",
            "normalized_text": "In today's lecture we will explore gradient"
        },
        {
            "chunk_index": 1,
            "segment_index": 0,
            "start_time": 21.82,
            "end_time": 43.17,
            "text": "descent optimization and its convergence properties.",
            "normalized_text": "descent optimization and its convergence properties."
        }
    ]

    rec_id, _ = _create_mock_recording_with_segments(segments_data, duration=43.17)
    db = SessionLocal()
    try:
        blocks = ContextStitcher.process_and_persist_stitched_blocks(rec_id, db)
        assert len(blocks) == 1

        b = blocks[0]
        assert b["start_time"] == 0.00
        assert b["end_time"] == 43.17
        assert b["source_chunk_indices"] == "0,1"
        assert "gradient descent optimization and its convergence properties." in b["text"]
    finally:
        db.close()


# ============================================================================
# Test 2: Repetition and Overlap Pruning
# ============================================================================
def test_repetition_and_overlap_pruning():
    """
    Verify duplicated boundary tokens from Whisper acoustic overlap are deduplicated.
    """
    segments_data = [
        {
            "chunk_index": 0,
            "segment_index": 0,
            "start_time": 0.0,
            "end_time": 14.0,
            "text": "Please make sure you submit your assignment on LMS LMS",
            "normalized_text": "Please make sure you submit your assignment on LMS LMS"
        },
        {
            "chunk_index": 1,
            "segment_index": 0,
            "start_time": 14.5,
            "end_time": 28.0,
            "text": "LMS portal before Friday midnight.",
            "normalized_text": "LMS portal before Friday midnight."
        }
    ]

    rec_id, _ = _create_mock_recording_with_segments(segments_data, duration=28.0)
    db = SessionLocal()
    try:
        blocks = ContextStitcher.process_and_persist_stitched_blocks(rec_id, db)
        assert len(blocks) == 1

        b = blocks[0]
        # Should not have triple "LMS"
        assert "LMS LMS LMS" not in b["text"]
        assert "on LMS portal before Friday midnight." in b["text"]
    finally:
        db.close()


# ============================================================================
# Test 3: Cross-Chunk Named Entity Alias Clustering
# ============================================================================
def test_cross_chunk_named_entity_alias_clustering():
    """
    Verify mentions of an entity across separate chunks are clustered under canonical entity.
    """
    segments_data = [
        {
            "chunk_index": 0,
            "segment_index": 0,
            "start_time": 0.0,
            "end_time": 10.0,
            "text": "Professor Tariq introduced the course grading policy.",
            "normalized_text": "Professor Tariq introduced the course grading policy."
        },
        {
            "chunk_index": 1,
            "segment_index": 0,
            "start_time": 10.5,
            "end_time": 20.0,
            "text": "Linear regression will be covered in week two.",
            "normalized_text": "Linear regression will be covered in week two."
        },
        {
            "chunk_index": 2,
            "segment_index": 0,
            "start_time": 20.5,
            "end_time": 30.0,
            "text": "Tariq mentioned that quizzes will be conducted on LMS.",
            "normalized_text": "Tariq mentioned that quizzes will be conducted on LMS."
        }
    ]

    rec_id, _ = _create_mock_recording_with_segments(segments_data, duration=30.0)
    db = SessionLocal()
    try:
        stats = EntityResolver.process_and_persist_entities(rec_id, db)
        assert stats["total_entities"] >= 1

        entities = db.query(Entity).filter(Entity.recording_id == rec_id).all()
        ent_names = {e.canonical_name: e for e in entities}

        # Verify Professor Tariq cluster
        tariq_ent = next((e for name, e in ent_names.items() if "Tariq" in name), None)
        assert tariq_ent is not None
        assert tariq_ent.mention_count >= 2
        assert tariq_ent.entity_type == "PERSON"

        # Verify LMS Tool entity
        lms_ent = next((e for name, e in ent_names.items() if "LMS" in name), None)
        assert lms_ent is not None
        assert lms_ent.entity_type == "TOOL_TECH"
    finally:
        db.close()


# ============================================================================
# Test 4: Cross-Chunk Pronoun Anaphora Resolution
# ============================================================================
def test_cross_chunk_pronoun_anaphora_resolution():
    """
    Verify English and Roman Urdu pronouns resolve to the nearest antecedent entity.
    """
    segments_data = [
        {
            "chunk_index": 0,
            "segment_index": 0,
            "start_time": 0.0,
            "end_time": 12.0,
            "text": "Bilal is responsible for data preprocessing pipeline.",
            "normalized_text": "Bilal is responsible for data preprocessing pipeline."
        },
        {
            "chunk_index": 1,
            "segment_index": 0,
            "start_time": 12.5,
            "end_time": 25.0,
            "text": "He will share the cleaned CSV file on Monday.",
            "normalized_text": "He will share the cleaned CSV file on Monday."
        },
        {
            "chunk_index": 2,
            "segment_index": 0,
            "start_time": 25.5,
            "end_time": 38.0,
            "text": "Sara ne PyTorch model train kiya.",
            "normalized_text": "Sara ne PyTorch model train kiya."
        },
        {
            "chunk_index": 3,
            "segment_index": 0,
            "start_time": 38.5,
            "end_time": 50.0,
            "text": "Unhon ne 95% accuracy achieve ki hai.",
            "normalized_text": "Unhon ne 95% accuracy achieve ki hai."
        }
    ]

    rec_id, _ = _create_mock_recording_with_segments(segments_data, duration=50.0)
    db = SessionLocal()
    try:
        stats = EntityResolver.process_and_persist_entities(rec_id, db)
        assert stats["total_anaphors_resolved"] >= 2

        # Check Bilal mentions
        bilal_mentions = (
            db.query(EntityMention)
            .join(Entity, EntityMention.entity_id == Entity.id)
            .filter(Entity.recording_id == rec_id, Entity.canonical_name.like("%Bilal%"))
            .all()
        )
        surfaces = [m.surface_text.lower() for m in bilal_mentions]
        assert "bilal" in surfaces
        assert "he" in surfaces

        # Check Sara mentions
        sara_mentions = (
            db.query(EntityMention)
            .join(Entity, EntityMention.entity_id == Entity.id)
            .filter(Entity.recording_id == rec_id, Entity.canonical_name.like("%Sara%"))
            .all()
        )
        s_surfaces = [m.surface_text.lower() for m in sara_mentions]
        assert "sara" in s_surfaces
        assert any("unhon" in s for s in s_surfaces)
    finally:
        db.close()


# ============================================================================
# Test 5: Ambiguous Pronoun Non-Hallucination
# ============================================================================
def test_ambiguous_pronoun_non_hallucination():
    """
    Verify coreference engine does not hallucinate single owner when multiple equal candidates exist.
    """
    segments_data = [
        {
            "chunk_index": 0,
            "segment_index": 0,
            "start_time": 0.0,
            "end_time": 10.0,
            "text": "Ali met Usman in the computer lab.",
            "normalized_text": "Ali met Usman in the computer lab."
        },
        {
            "chunk_index": 1,
            "segment_index": 0,
            "start_time": 10.5,
            "end_time": 20.0,
            "text": "He explained the simulation results.",
            "normalized_text": "He explained the simulation results."
        }
    ]

    rec_id, _ = _create_mock_recording_with_segments(segments_data, duration=20.0)
    db = SessionLocal()
    try:
        stats = EntityResolver.process_and_persist_entities(rec_id, db)
        # Because Ali and Usman are both in chunk 0 with near identical timestamps, "He" is ambiguous
        resolved_anaphors = (
            db.query(EntityMention)
            .filter(EntityMention.recording_id == rec_id, EntityMention.is_resolved_anaphor.is_(True))
            .all()
        )
        # Zero hallucination: ambiguous pronoun should not be forcibly mapped
        assert len(resolved_anaphors) == 0
    finally:
        db.close()


# ============================================================================
# Test 6: Database Persistence and Cascading Deletion
# ============================================================================
def test_database_persistence_and_cascading_deletion():
    """
    Verify StitchedBlock, Entity, and EntityMention are saved and deleted on cascade.
    """
    segments_data = [
        {"chunk_index": 0, "segment_index": 0, "start_time": 0.0, "end_time": 15.0, "text": "Dr. Ahmed gave the lecture."},
        {"chunk_index": 1, "segment_index": 0, "start_time": 15.5, "end_time": 30.0, "text": "He covered backpropagation."}
    ]

    rec_id, _ = _create_mock_recording_with_segments(segments_data, duration=30.0)
    db = SessionLocal()
    try:
        ContextStitcher.process_and_persist_stitched_blocks(rec_id, db)
        EntityResolver.process_and_persist_entities(rec_id, db)

        assert db.query(StitchedBlock).filter(StitchedBlock.recording_id == rec_id).count() >= 1
        assert db.query(Entity).filter(Entity.recording_id == rec_id).count() >= 1
        assert db.query(EntityMention).filter(EntityMention.recording_id == rec_id).count() >= 1

        # Delete recording
        rec = db.query(Recording).filter(Recording.id == rec_id).first()
        db.delete(rec)
        db.commit()

        # Check cascading deletion
        assert db.query(StitchedBlock).filter(StitchedBlock.recording_id == rec_id).count() == 0
        assert db.query(Entity).filter(Entity.recording_id == rec_id).count() == 0
        assert db.query(EntityMention).filter(EntityMention.recording_id == rec_id).count() == 0
    finally:
        db.close()


# ============================================================================
# Test 7: REST API Endpoints Integration
# ============================================================================
def test_stitching_and_entity_rest_endpoints():
    """
    Verify POST /stitch, GET /stitched-transcript, and GET /entities API endpoints.
    """
    segments_data = [
        {
            "chunk_index": 0,
            "segment_index": 0,
            "start_time": 0.0,
            "end_time": 15.0,
            "text": "Professor Tariq announced Assignment 1 deadline on LMS.",
            "normalized_text": "Professor Tariq announced Assignment 1 deadline on LMS."
        },
        {
            "chunk_index": 1,
            "segment_index": 0,
            "start_time": 15.5,
            "end_time": 30.0,
            "text": "He mentioned that late submissions carry a penalty.",
            "normalized_text": "He mentioned that late submissions carry a penalty."
        }
    ]

    rec_id, _ = _create_mock_recording_with_segments(segments_data, duration=30.0)

    # 1. POST /stitch
    stitch_res = client.post(f"/api/recordings/{rec_id}/stitch")
    assert stitch_res.status_code == 200
    stitch_json = stitch_res.json()
    assert stitch_json["status"] == "success"
    assert stitch_json["total_blocks_created"] >= 1
    assert stitch_json["total_entities_extracted"] >= 1

    # 2. GET /stitched-transcript
    tx_res = client.get(f"/api/recordings/{rec_id}/stitched-transcript")
    assert tx_res.status_code == 200
    tx_json = tx_res.json()
    assert tx_json["status"] == "success"
    assert tx_json["total_blocks"] >= 1
    assert len(tx_json["blocks"]) >= 1

    # 3. GET /entities
    ent_res = client.get(f"/api/recordings/{rec_id}/entities")
    assert ent_res.status_code == 200
    ent_json = ent_res.json()
    assert ent_json["status"] == "success"
    assert ent_json["total_entities"] >= 1
    names = [e["canonical_name"] for e in ent_json["entities"]]
    assert any("Tariq" in n for n in names)


# ============================================================================
# Test 8: True Merge — 3 Duplicate Mentions Across 3 Chunks -> 1 Merged Record
# ============================================================================
def test_assignment_duplicate_merging_true_merge():
    """
    Verify 3 mentions of the same assignment across 3 different chunks collapse into exactly 1 record,
    retaining the most complete/accurate combination of fields from its source mentions.
    Scenario:
    - Chunk 0: 'For Assignment 1, you have to implement gradient descent algorithm.'
    - Chunk 1: 'Please write a 5-page report in PDF format for this assignment.'
    - Chunk 2: 'Isko Friday midnight tak LMS par submit karwana hai.'
    """
    segments_data = [
        {
            "chunk_index": 0,
            "segment_index": 0,
            "start_time": 0.00,
            "end_time": 18.25,
            "text": "For Assignment 1, you have to implement gradient descent algorithm.",
            "normalized_text": "For Assignment 1, you have to implement gradient descent algorithm."
        },
        {
            "chunk_index": 1,
            "segment_index": 0,
            "start_time": 18.80,
            "end_time": 37.10,
            "text": "Please write a 5-page report in PDF format for this assignment.",
            "normalized_text": "Please write a 5-page report in PDF format for this assignment."
        },
        {
            "chunk_index": 2,
            "segment_index": 0,
            "start_time": 37.65,
            "end_time": 56.40,
            "text": "Isko Friday midnight tak LMS par submit karwana hai.",
            "normalized_text": "Isko Friday midnight tak LMS par submit karwana hai."
        }
    ]

    rec_id, tx_id = _create_mock_recording_with_segments(segments_data, duration=56.40)
    db = SessionLocal()
    try:
        from app.services.intelligence_service import IntelligenceService
        segments = db.query(TranscriptSegment).filter(TranscriptSegment.transcript_id == tx_id).all()
        transcript_text = " ".join(s.text for s in segments)

        # 1. Verify raw extraction count before merging
        raw_items = []
        for s in segments:
            # Check individual segment extraction
            seg_items = IntelligenceService.extract_assignments(s.text, [s])
            raw_items.extend(seg_items)
        assert len(raw_items) == 3, f"Expected 3 raw records before merge, got {len(raw_items)}"

        # 2. Verify full multi-chunk extraction collapses to exactly 1 record
        merged_assignments = IntelligenceService.extract_assignments(transcript_text, segments)
        assert len(merged_assignments) == 1, f"Expected 1 merged record after deduplication, got {len(merged_assignments)}"

        merged = merged_assignments[0]
        # Verify retained most complete combination of fields
        assert "Gradient Descent" in merged["topic"]
        assert merged["format"] == "PDF"
        assert "5" in merged["length_requirements"]
        assert "LMS" in merged["submission_method"]
        assert "Friday" in merged["deadline"] or "midnight" in merged["deadline"]
        assert merged["confidence"] == "High"
    finally:
        db.close()


# ============================================================================
# Test 9: False-Merge Guard — 2 Genuinely Different Assignments Stay Separate
# ============================================================================
def test_assignment_duplicate_merging_false_merge_guard():
    """
    Verify 2 genuinely different assignments mentioned close together in time remain 2 distinct records.
    Scenario:
    - Chunk 0: 'Assignment 1 is due Friday midnight on LMS.'
    - Chunk 1: 'Assignment 2 will be a group report due next month.'
    """
    segments_data = [
        {
            "chunk_index": 0,
            "segment_index": 0,
            "start_time": 0.00,
            "end_time": 19.50,
            "text": "Assignment 1 is due Friday midnight on LMS.",
            "normalized_text": "Assignment 1 is due Friday midnight on LMS."
        },
        {
            "chunk_index": 1,
            "segment_index": 0,
            "start_time": 20.10,
            "end_time": 39.80,
            "text": "Assignment 2 will be a group report due next month.",
            "normalized_text": "Assignment 2 will be a group report due next month."
        }
    ]

    rec_id, tx_id = _create_mock_recording_with_segments(segments_data, duration=39.80)
    db = SessionLocal()
    try:
        from app.services.intelligence_service import IntelligenceService
        segments = db.query(TranscriptSegment).filter(TranscriptSegment.transcript_id == tx_id).all()
        transcript_text = " ".join(s.text for s in segments)

        # Extraction must maintain 2 separate distinct records
        assignments = IntelligenceService.extract_assignments(transcript_text, segments)
        assert len(assignments) == 2, f"Expected 2 separate records (False-merge guard), got {len(assignments)}"

        a1, a2 = assignments[0], assignments[1]
        assert "Assignment 1" in a1["description"]
        assert "LMS" in a1["submission_method"]
        assert "Friday" in a1["deadline"] or "midnight" in a1["deadline"]

        assert "Assignment 2" in a2["description"]
        assert a2["individual_or_group"] == "Group"
        assert "next month" in a2["deadline"].lower()
    finally:
        db.close()


# ============================================================================
# Test 10: Action Item Merging — True Merge & False-Merge Guard
# ============================================================================
def test_action_item_duplicate_merging_true_and_false_merge():
    """
    Verify ActionItem merging:
    - True merge: Bilal assigned task + deliverable/deadline with pronoun 'He' -> collapses 2 records -> 1 record.
    - False-merge guard: Bilal assigned task 1 vs Sara assigned task 2 -> remains 2 separate records.
    """
    from app.services.intelligence_service import IntelligenceService

    # 1. True merge scenario
    true_merge_segs = [
        TranscriptSegment(
            id="s1", transcript_id="tx1", recording_id="r1", chunk_index=0, segment_index=0,
            start_time=0.00, end_time=15.00,
            text="Bilal will prepare the data preprocessing pipeline.",
            normalized_text="Bilal will prepare the data preprocessing pipeline."
        ),
        TranscriptSegment(
            id="s2", transcript_id="tx1", recording_id="r1", chunk_index=1, segment_index=0,
            start_time=15.50, end_time=30.00,
            text="He will send the csv file by Monday.",
            normalized_text="He will send the csv file by Monday."
        )
    ]
    txt1 = " ".join(s.text for s in true_merge_segs)
    merged_items = IntelligenceService.extract_action_items(txt1, true_merge_segs)
    assert len(merged_items) == 1, f"Expected 1 merged action item, got {len(merged_items)}"
    assert merged_items[0]["owner"] == "Bilal"
    assert "Monday" in merged_items[0]["deadline"]
    assert "Csv File" in merged_items[0]["deliverables"]

    # 2. False-merge guard scenario
    false_merge_segs = [
        TranscriptSegment(
            id="s3", transcript_id="tx2", recording_id="r2", chunk_index=0, segment_index=0,
            start_time=0.00, end_time=15.00,
            text="Bilal will prepare the data preprocessing pipeline by Monday.",
            normalized_text="Bilal will prepare the data preprocessing pipeline by Monday."
        ),
        TranscriptSegment(
            id="s4", transcript_id="tx2", recording_id="r2", chunk_index=1, segment_index=0,
            start_time=15.50, end_time=30.00,
            text="Sara will train the neural network model by Friday.",
            normalized_text="Sara will train the neural network model by Friday."
        )
    ]
    txt2 = " ".join(s.text for s in false_merge_segs)
    separate_items = IntelligenceService.extract_action_items(txt2, false_merge_segs)
    assert len(separate_items) == 2, f"Expected 2 distinct action items, got {len(separate_items)}"
    assert separate_items[0]["owner"] == "Bilal"
    assert separate_items[1]["owner"] == "Sara"

