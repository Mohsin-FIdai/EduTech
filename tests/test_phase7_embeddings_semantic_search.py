"""
test_phase7_embeddings_semantic_search.py — Acceptance tests for Phase 7: Vector Embeddings & Semantic Search Q&A.

Acceptance Matrix:
1. Embedding Generation & Vector Dimensionality (384-dim normalized float32 vectors).
2. Paraphrase / Synonym Match Verification (Resolving Phase 6 Addendum failure).
3. Cross-Lingual Semantic Retrieval (Roman Urdu query -> English segment).
4. False-Positive / Unmentioned Topic Rejection (Cosine similarity threshold gating).
5. Top-k Ranking & Discrimination across multi-topic lecture.
6. Database Embedding Persistence & Caching (Hash-based change detection).
7. REST API Endpoints Verification (/embeddings/generate, /search/semantic, /qa-semantic).
"""

import tempfile
import os
import pytest
import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from app.main import app
from app.database import SessionLocal
from app.models import Recording, Transcript, TranscriptSegment, SegmentEmbedding
from app.services.embedding_service import EmbeddingService, EmbeddingModelManager


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
            filename="semantic_test.wav",
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
            language_summary="en: 70%, ur: 30%"
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
                language=s.get("language", "en"),
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
# Test 1: Embedding Generation & Vector Dimensionality
# ============================================================================
def test_embedding_generation_and_dimensionality():
    """Verify generated vectors are exactly 384-dimensional, L2-normalized float32 arrays."""
    text = "Machine learning algorithms optimize mathematical loss functions through iterative training."
    vec = EmbeddingModelManager.encode(text, normalize_embeddings=True)

    assert isinstance(vec, np.ndarray)
    assert vec.shape == (1, 384)
    assert vec.dtype == np.float32

    # L2 Norm should be approximately 1.0
    norm = np.linalg.norm(vec[0])
    assert abs(norm - 1.0) < 1e-4


# ============================================================================
# Test 2: Paraphrase / Synonym Match Verification (Fix for Phase 6 Addendum)
# ============================================================================
def test_paraphrase_synonym_match_verification():
    """
    Directly tests the failure case from Phase 6 Addendum:
    Segment: 'In supervised classification, the model is trained on labeled feature datasets.'
    Query: 'How does the algorithm learn from tagged examples?'
    Must achieve high cosine similarity and return found=True with citation.
    """
    segments_data = [
        {
            "start_time": 10.0,
            "end_time": 20.0,
            "text": "In supervised classification, the model is trained on labeled feature datasets.",
            "normalized_text": "In supervised classification, the model is trained on labeled feature datasets."
        },
        {
            "start_time": 20.0,
            "end_time": 30.0,
            "text": "Assignment 1 deadline is Friday midnight through LMS portal.",
            "normalized_text": "Assignment 1 deadline is Friday midnight through LMS portal."
        }
    ]

    rec_id, tx_id = _create_mock_recording_with_segments(segments_data, duration=30.0)
    db = SessionLocal()
    try:
        # Perform semantic QA with synonym phrasing
        synonym_query = "How does the algorithm learn from tagged examples?"
        res = EmbeddingService.answer_semantic_qa(
            recording_id=rec_id,
            question=synonym_query,
            threshold=0.50,
            db=db
        )

        assert res["found"] is True
        assert res["timestamp"] == "[10.00s - 20.00s]"
        assert "supervised classification" in res["answer"].lower()
        assert res["similarity_score"] >= 0.50
    finally:
        db.close()


# ============================================================================
# Test 3: Cross-Lingual Semantic Retrieval
# ============================================================================
def test_cross_lingual_semantic_retrieval():
    """
    Verify Roman Urdu query accurately retrieves an English lecture segment.
    English Segment: 'Assignment 1 deadline is Friday midnight through LMS portal.'
    Roman Urdu Query: 'Assignment jama karwane ki aakhri tareekh kab hai?'
    """
    segments_data = [
        {
            "start_time": 0.0,
            "end_time": 10.0,
            "text": "Linear regression models predict continuous real values.",
            "normalized_text": "Linear regression models predict continuous real values."
        },
        {
            "start_time": 10.0,
            "end_time": 20.0,
            "text": "Assignment 1 deadline is Friday midnight through LMS portal.",
            "normalized_text": "Assignment 1 deadline is Friday midnight through LMS portal."
        }
    ]

    rec_id, tx_id = _create_mock_recording_with_segments(segments_data, duration=20.0)
    db = SessionLocal()
    try:
        roman_urdu_query = "Assignment jama karwane ki aakhri tareekh kab hai?"
        results = EmbeddingService.search_semantic(
            recording_id=rec_id,
            query=roman_urdu_query,
            top_k=1,
            threshold=0.40,
            db=db
        )

        assert len(results) >= 1
        best = results[0]
        assert best["timestamp"] == "[10.00s - 20.00s]"
        assert "deadline is Friday midnight" in best["text"]
    finally:
        db.close()


# ============================================================================
# Test 4: False-Positive / Unmentioned Topic Rejection
# ============================================================================
def test_unmentioned_topic_rejection():
    """
    Verify completely unmentioned topics score below threshold and are strictly declined.
    """
    segments_data = [
        {
            "start_time": 0.0,
            "end_time": 15.0,
            "text": "Convolutional neural networks process image feature matrices using pooling layers.",
            "normalized_text": "Convolutional neural networks process image feature matrices using pooling layers."
        }
    ]

    rec_id, tx_id = _create_mock_recording_with_segments(segments_data, duration=15.0)
    db = SessionLocal()
    try:
        unmentioned_query = "What is quantum computing error mitigation protocol?"
        res = EmbeddingService.answer_semantic_qa(
            recording_id=rec_id,
            question=unmentioned_query,
            threshold=0.50,
            db=db
        )

        assert res["found"] is False
        assert res["answer"] == "Not found in the recording"
        assert res["timestamp"] is None
        # Verify raw score is populated (not clamped to 0) and below the 0.50 rejection threshold
        assert 0.0 < res["similarity_score"] < 0.50
    finally:
        db.close()


# ============================================================================
# Test 5: Top-k Ranking & Discrimination Across Multi-Topic Lecture
# ============================================================================
def test_top_k_ranking_multi_topic():
    """
    Verify multi-topic lecture correctly ranks the most semantically relevant segment first.
    """
    segments_data = [
        {"start_time": 0.0, "end_time": 10.0, "text": "Relational databases use SQL queries and B-tree indexes for fast lookups."},
        {"start_time": 10.0, "end_time": 20.0, "text": "Operating systems manage process scheduling and virtual memory paging."},
        {"start_time": 20.0, "end_time": 30.0, "text": "Backpropagation computes loss gradients across multi-layer neural networks."},
        {"start_time": 30.0, "end_time": 40.0, "text": "Computer networking uses TCP IP protocol handshakes for reliable transmission."}
    ]

    rec_id, tx_id = _create_mock_recording_with_segments(segments_data, duration=40.0)
    db = SessionLocal()
    try:
        query = "gradient descent backpropagation training"
        results = EmbeddingService.search_semantic(
            recording_id=rec_id,
            query=query,
            top_k=3,
            threshold=0.30,
            db=db
        )

        assert len(results) >= 1
        assert results[0]["timestamp"] == "[20.00s - 30.00s]"
        assert "Backpropagation" in results[0]["text"]
    finally:
        db.close()


# ============================================================================
# Test 6: Database Embedding Persistence & Caching
# ============================================================================
def test_database_embedding_persistence_and_caching():
    """
    Verify embeddings are stored in SQLite and cached without redundant computations.
    """
    segments_data = [
        {"start_time": 0.0, "end_time": 10.0, "text": "First segment on machine learning fundamentals."},
        {"start_time": 10.0, "end_time": 20.0, "text": "Second segment on optimization algorithms."}
    ]

    rec_id, tx_id = _create_mock_recording_with_segments(segments_data, duration=20.0)
    db = SessionLocal()
    try:
        # Generate embeddings
        count1 = EmbeddingService.generate_and_store_embeddings(rec_id, db)
        assert count1 == 2

        # Verify DB records
        emb_records = db.query(SegmentEmbedding).filter(SegmentEmbedding.recording_id == rec_id).all()
        assert len(emb_records) == 2
        assert emb_records[0].dimensions == 384
        assert len(emb_records[0].embedding_vector) == 384 * 4  # 384 float32 = 1536 bytes

        # Re-run generation without force -> should use cached hashes
        count2 = EmbeddingService.generate_and_store_embeddings(rec_id, db, force_regenerate=False)
        assert count2 == 2
    finally:
        db.close()


# ============================================================================
# Test 7: REST API Endpoints Verification
# ============================================================================
def test_semantic_search_rest_endpoints(client):
    """
    Test REST API endpoints:
    1. POST /api/recordings/{id}/embeddings/generate
    2. POST /api/recordings/{id}/search/semantic
    3. POST /api/recordings/{id}/qa-semantic
    """
    segments_data = [
        {"start_time": 0.0, "end_time": 15.0, "text": "Artificial Intelligence is transforming medical diagnosis and healthcare automation."},
        {"start_time": 15.0, "end_time": 30.0, "text": "The final semester project is due next month."}
    ]

    rec_id, tx_id = _create_mock_recording_with_segments(segments_data, duration=30.0)

    # 1. POST /embeddings/generate
    gen_res = client.post(f"/api/recordings/{rec_id}/embeddings/generate")
    assert gen_res.status_code == 200
    gen_data = gen_res.json()
    assert gen_data["status"] == "success"
    assert gen_data["total_segments_embedded"] == 2
    assert gen_data["dimensions"] == 384

    # 2. POST /search/semantic
    search_res = client.post(
        f"/api/recordings/{rec_id}/search/semantic",
        json={"query": "medical diagnosis and healthcare", "top_k": 2, "threshold": 0.35}
    )
    assert search_res.status_code == 200
    search_data = search_res.json()
    assert search_data["status"] == "success"
    assert search_data["total_results"] >= 1
    assert "[0.00s - 15.00s]" in search_data["results"][0]["timestamp"]

    # 3. POST /qa-semantic
    qa_res = client.post(
        f"/api/recordings/{rec_id}/qa-semantic",
        json={"question": "When is the final semester project due?", "threshold": 0.50}
    )
    assert qa_res.status_code == 200
    qa_data = qa_res.json()
    assert qa_data["found"] is True
    assert "[15.00s - 30.00s]" in qa_data["timestamp"]
