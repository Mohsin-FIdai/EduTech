"""
test_phase5_text_processing.py — Acceptance tests for Phase 5: Pakistani English / Roman-Urdu Text Processing & Correction.

Acceptance Matrix:
1. Code-Mixed Segment Detection & Token-Level Tagging (Pakistani English + Roman Urdu).
2. Technical Dictionary Preservation (CS/IT and academic terms preserved).
3. Roman Urdu Spelling Variant Normalization & Abbreviation Expansion.
4. Context-Aware Homophone Disambiguation ('or' vs 'aur').
5. Full NLP Pipeline End-to-End Execution (PipelineResult with tags & stages).
6. Recording Transcript Batch Enhancement (POST /api/recordings/{id}/process-text).
7. Noisy ASR Error Recovery (handling Whisper phonetic corruptions).
8. REST API Endpoints (/api/text/process, /api/text/normalize, /api/text/detect, /api/recordings/{id}/segments/tagged).
"""

import tempfile
import os
import pytest
from fastapi.testclient import TestClient
import soundfile as sf
import numpy as np

from app.main import app
from app.database import get_db
from app.models import Recording, Transcript, TranscriptSegment, RecordingStatus
from roman_urdu_nlp import Pipeline as RomanUrduPipeline
from roman_urdu_nlp.cleaner import clean
from roman_urdu_nlp.detector import detect_segments, tag_tokens
from roman_urdu_nlp.normalizer import normalize
from roman_urdu_nlp.corrector import correct


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


# ============================================================================
# Test 1: Code-Mixed Segment Detection & Token Tagging
# ============================================================================
def test_code_mixed_detection_and_token_tagging():
    """
    Test segment-level and token-level language identification on real code-mixed sentences.
    """
    text = "Aaj ka topic machine learning aur neural networks hai"
    segments = detect_segments(text)
    tags = tag_tokens(text)

    # Verify segments
    assert len(segments) >= 3
    segment_langs = [s[1] for s in segments]
    assert "roman_urdu" in segment_langs
    assert "english" in segment_langs

    # Verify token tags
    tag_map = {t["token"].lower(): t for t in tags}
    assert tag_map["aaj"]["lang"] == "ur"
    assert tag_map["ka"]["lang"] == "ur"
    assert tag_map["topic"]["lang"] == "en"
    assert tag_map["topic"]["is_technical"] is True
    assert tag_map["machine"]["lang"] == "en"
    assert tag_map["machine"]["is_technical"] is True
    assert tag_map["learning"]["lang"] == "en"
    assert tag_map["learning"]["is_technical"] is True
    assert tag_map["aur"]["lang"] == "ur"
    assert tag_map["neural"]["lang"] == "en"
    assert tag_map["neural"]["is_technical"] is True
    assert tag_map["networks"]["lang"] == "en"
    assert tag_map["networks"]["is_technical"] is True
    assert tag_map["hai"]["lang"] == "ur"


# ============================================================================
# Test 2: Academic & Technical Dictionary Preservation
# ============================================================================
def test_technical_dictionary_preservation():
    """
    Verify CS/IT academic loanwords are preserved through the full normalization and correction chain.
    """
    sentence = "Machine learning models dataset par train hotay hain aur accuracy evaluate kartay hain."
    cleaned = clean(sentence)
    normalized = normalize(cleaned)
    corrected = correct(normalized)

    # Ensure technical terms are NOT corrupted by edit-distance spell correction
    assert "machine" in corrected.lower()
    assert "learning" in corrected.lower()
    assert "models" in corrected.lower()
    assert "dataset" in corrected.lower()
    assert "accuracy" in corrected.lower()
    assert "evaluate" in corrected.lower()


# ============================================================================
# Test 3: Roman Urdu Spelling Variant Normalization & Abbreviations
# ============================================================================
def test_roman_urdu_normalization_and_abbreviations():
    """
    Test normalization of informal Roman Urdu spelling variants and chat abbreviations.
    """
    raw_input = "mjy boht bhok lgi h aur kal exam h pls btao kch smjh nhi aa rha"
    cleaned = clean(raw_input)
    normalized = normalize(cleaned)

    # Assert canonical mappings
    assert "mujhe" in normalized.lower()
    assert "bohat" in normalized.lower()
    assert "bhook" in normalized.lower()
    assert "lagi" in normalized.lower()
    assert "hai" in normalized.lower()
    assert "please" in normalized.lower()
    assert "batao" in normalized.lower()
    assert "kuch" in normalized.lower()
    assert "samajh" in normalized.lower()
    assert "nahi" in normalized.lower()
    assert "raha" in normalized.lower()


# ============================================================================
# Test 4: Context-Aware Homophone Disambiguation ('or' vs 'aur')
# ============================================================================
def test_context_aware_homophone_disambiguation():
    """
    Disambiguate 'or' (English conjunction) vs 'aur' (Urdu conjunction) based on context.
    """
    # Case A: Roman Urdu context -> 'or' should become 'aur'
    urdu_context = "kal assignment submit karni hai or quiz ki tayari karni hai"
    norm_urdu = normalize(urdu_context)
    assert "aur" in norm_urdu.lower()

    # Case B: English technical list -> 'or' should be preserved
    english_context = "you can choose machine learning or deep learning"
    norm_english = normalize(english_context)
    assert " or " in norm_english.lower()


# ============================================================================
# Test 5: Full NLP Pipeline End-to-End
# ============================================================================
def test_full_nlp_pipeline_end_to_end():
    """
    Verify roman_urdu_nlp.Pipeline() returns structured PipelineResult with token_tags and stages.
    """
    nlp = RomanUrduPipeline()
    result = nlp("Muje boht bhok lgi h 😂😂 aur kal exam hai")

    assert result.normalized
    assert "mujhe" in result.normalized.lower()
    assert "bohat" in result.normalized.lower()
    assert "bhook" in result.normalized.lower()
    assert result.urdu
    assert result.english
    assert isinstance(result.token_tags, list)
    assert len(result.token_tags) >= 5
    assert "ur" in result.language_distribution
    assert "cleaned" in result.stages
    assert "normalized" in result.stages
    assert "token_tags" in result.stages


# ============================================================================
# Test 6: Recording Transcript Batch Enhancement
# ============================================================================
def test_recording_transcript_batch_enhancement(client):
    """
    Test POST /api/recordings/{id}/process-text on a recording in the database.
    """
    tmp_wav = tempfile.mktemp(suffix=".wav")
    sr = 16000
    sf.write(tmp_wav, np.zeros(int(sr * 3.0), dtype=np.float32), sr, subtype="PCM_16")

    try:
        with open(tmp_wav, "rb") as f:
            up_res = client.post("/api/upload", files={"file": ("batch_text_test.wav", f.read(), "audio/wav")})
        rec_id = up_res.json()["data"]["id"]

        # Manually create mock transcript and segment for test
        from app.database import SessionLocal
        db = SessionLocal()
        try:
            tx = Transcript(
                recording_id=rec_id,
                original_transcript="mjy boht bhok lgi h or kal presentation hai",
                normalized_transcript="mjy boht bhok lgi h or kal presentation hai",
                language_summary="ur: 100%"
            )
            db.add(tx)
            db.flush()

            seg = TranscriptSegment(
                transcript_id=tx.id,
                recording_id=rec_id,
                chunk_index=0,
                segment_index=0,
                start_time=0.0,
                end_time=3.0,
                text="mjy boht bhok lgi h or kal presentation hai",
                language="ur",
                confidence=0.95
            )
            db.add(seg)
            db.commit()
        finally:
            db.close()

        # Call batch enhancement endpoint
        enh_res = client.post(f"/api/recordings/{rec_id}/process-text")
        assert enh_res.status_code == 200
        data = enh_res.json()
        assert data["status"] == "success"
        assert "mujhe" in data["data"]["normalized_transcript"].lower()
        assert "bohat" in data["data"]["normalized_transcript"].lower()
        assert "presentation" in data["data"]["normalized_transcript"].lower()
    finally:
        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)


# ============================================================================
# Test 7: Noisy ASR Error Recovery
# ============================================================================
def test_noisy_asr_error_recovery():
    """
    Verify recovery of phonetic Whisper ASR corruptions for technical vocabulary.
    """
    noisy_asr = "aaj ka tapek machine learning aur arati fisial intelligence hai"
    cleaned = clean(noisy_asr)
    normalized = normalize(cleaned)
    corrected = correct(normalized)

    # 'tapek' -> 'topic'
    assert "topic" in corrected.lower()
    # 'arati fisial' -> 'artificial'
    assert "artificial" in corrected.lower()
    assert "learning" in corrected.lower()
    assert "intelligence" in corrected.lower()


# ============================================================================
# Test 8: REST API Endpoints Verification
# ============================================================================
def test_text_processing_rest_endpoints(client):
    """
    Test REST endpoints:
    1. POST /api/text/process
    2. POST /api/text/normalize
    3. POST /api/text/detect
    4. GET /api/recordings/{id}/segments/tagged
    """
    # 1. POST /api/text/process
    res1 = client.post("/api/text/process", json={"text": "Muje boht bhok lgi h 😂😂 or kal exam hai"})
    assert res1.status_code == 200
    data1 = res1.json()
    assert data1["status"] == "success"
    assert "mujhe" in data1["normalized_text"].lower()
    assert "bohat" in data1["normalized_text"].lower()
    assert data1["urdu_text"]
    assert data1["english_text"]
    assert len(data1["token_tags"]) > 0

    # 2. POST /api/text/normalize
    res2 = client.post("/api/text/normalize", json={"text": "mjy smjh nhi aa rha"})
    assert res2.status_code == 200
    data2 = res2.json()
    assert "mujhe" in data2["normalized_text"].lower()
    assert "samajh" in data2["normalized_text"].lower()
    assert "nahi" in data2["normalized_text"].lower()

    # 3. POST /api/text/detect
    res3 = client.post("/api/text/detect", json={"text": "today meeting kal hai office mein"})
    assert res3.status_code == 200
    data3 = res3.json()
    assert len(data3["segments"]) >= 2
    assert "en" in data3["language_distribution"]
    assert "ur" in data3["language_distribution"]
