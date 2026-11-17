"""
tests/test_config.py — Sanity tests for centralised configuration constants.

Validates:
  - All config values have expected defaults
  - Env-var overrides work via monkeypatch
  - Type and range checks for every constant
"""

import importlib
import os

import pytest


def _reload_config(monkeypatch=None, overrides: dict | None = None):
    """Reload app.config module with optional env-var overrides."""
    if overrides:
        for k, v in overrides.items():
            monkeypatch.setenv(k, v)
    import app.config as cfg
    importlib.reload(cfg)
    return cfg


# ─── Default value tests ───────────────────────────────────────────

class TestConfigDefaults:
    """Every constant has the expected default value."""

    def test_chunking_defaults(self):
        from app import config as cfg
        assert cfg.TARGET_CHUNK_DURATION_SECONDS == 25.0
        assert cfg.MAX_CHUNK_DURATION_SECONDS == 30.0
        assert cfg.MIN_CHUNK_DURATION_SECONDS == 2.0
        assert cfg.DEFAULT_CHUNK_OVERLAP_SECONDS == 0.5
        assert cfg.SILENCE_THRESHOLD_DBFS == -30.0
        assert cfg.MIN_PAUSE_DURATION_SECONDS == 0.20

    def test_whisper_defaults(self):
        from app import config as cfg
        assert cfg.DEFAULT_WHISPER_MODEL == "base"
        assert cfg.WHISPER_NO_SPEECH_THRESHOLD == 0.6
        assert cfg.WHISPER_COMPRESSION_RATIO_THRESHOLD == 2.2
        assert cfg.WHISPER_LOGPROB_THRESHOLD == -1.0

    def test_llm_defaults(self):
        from app import config as cfg
        assert cfg.LOCAL_LLM_MODEL == "Qwen/Qwen2.5-0.5B-Instruct"
        assert cfg.LOCAL_LLM_MAX_TOKENS == 512
        assert cfg.LOCAL_LLM_TEMPERATURE == 0.1

    def test_rag_defaults(self):
        from app import config as cfg
        assert cfg.RAG_MIN_TOKEN_OVERLAP == 0.80
        assert cfg.MAX_CONVERSATION_TURNS == 5
        assert cfg.RAG_SLIDING_WINDOW_THRESHOLD == 3.5
        assert cfg.RAG_TOPIC_MATCH_MIN_SCORE == 4.0

    def test_pedagogical_defaults(self):
        from app import config as cfg
        assert cfg.NOTES_GROUNDING_FLOOR == 0.25
        assert cfg.NOTES_GROUNDING_NORMALIZATION == 0.50

    def test_stitching_defaults(self):
        from app import config as cfg
        assert cfg.STITCH_MAX_OVERLAP_WORDS == 4
        assert cfg.STITCH_TIME_GAP_THRESHOLD == 3.0
        assert cfg.STITCH_CLOSE_GAP_THRESHOLD == 1.5

    def test_embedding_defaults(self):
        from app import config as cfg
        assert cfg.SEMANTIC_SEARCH_THRESHOLD == 0.40
        assert cfg.SEMANTIC_QA_THRESHOLD == 0.50

    def test_translation_defaults(self):
        from app import config as cfg
        assert cfg.ENGLISH_PASSTHROUGH_RATIO == 0.80
        assert cfg.TRANSLATION_CONFIDENCE_GOOGLE == 0.95
        assert cfg.TRANSLATION_CONFIDENCE_LLM == 0.85
        assert cfg.TRANSLATION_CONFIDENCE_DICTIONARY == 0.60

    def test_audio_preprocessing_defaults(self):
        from app import config as cfg
        assert cfg.AUDIO_TARGET_PEAK_NORM == 0.95
        assert cfg.AUDIO_DENOISE_PROP_DECREASE == 0.75
        assert cfg.VAD_TOP_DB == 20.0
        assert cfg.VAD_MIN_SPEECH_DURATION_SEC == 0.1

    def test_domain_corrector_defaults(self):
        from app import config as cfg
        assert cfg.DOMAIN_CORRECTOR_MIN_SCORE == 0.82
        assert cfg.DOMAIN_CORRECTOR_BIGRAM_MIN_SCORE == 0.84

    def test_hallucination_ratio_default(self):
        from app import config as cfg
        assert cfg.HALLUCINATION_TOKEN_RATIO == 0.35


# ─── Env-var override tests ────────────────────────────────────────

class TestConfigEnvOverrides:
    """Env-var overrides change the constant value."""

    def test_override_rag_min_token_overlap(self, monkeypatch):
        cfg = _reload_config(monkeypatch, {"RAG_MIN_TOKEN_OVERLAP": "0.55"})
        assert cfg.RAG_MIN_TOKEN_OVERLAP == 0.55

    def test_override_max_conversation_turns(self, monkeypatch):
        cfg = _reload_config(monkeypatch, {"MAX_CONVERSATION_TURNS": "10"})
        assert cfg.MAX_CONVERSATION_TURNS == 10

    def test_override_whisper_no_speech(self, monkeypatch):
        cfg = _reload_config(monkeypatch, {"WHISPER_NO_SPEECH_THRESHOLD": "0.8"})
        assert cfg.WHISPER_NO_SPEECH_THRESHOLD == 0.8

    def test_override_stitch_time_gap(self, monkeypatch):
        cfg = _reload_config(monkeypatch, {"STITCH_TIME_GAP_THRESHOLD": "5.0"})
        assert cfg.STITCH_TIME_GAP_THRESHOLD == 5.0

    def test_override_semantic_search(self, monkeypatch):
        cfg = _reload_config(monkeypatch, {"SEMANTIC_SEARCH_THRESHOLD": "0.60"})
        assert cfg.SEMANTIC_SEARCH_THRESHOLD == 0.60

    def test_override_notes_grounding(self, monkeypatch):
        cfg = _reload_config(monkeypatch, {"NOTES_GROUNDING_FLOOR": "0.30"})
        assert cfg.NOTES_GROUNDING_FLOOR == 0.30

    def test_override_hallucination_ratio(self, monkeypatch):
        cfg = _reload_config(monkeypatch, {"HALLUCINATION_TOKEN_RATIO": "0.50"})
        assert cfg.HALLUCINATION_TOKEN_RATIO == 0.50

    def test_override_vad_top_db(self, monkeypatch):
        cfg = _reload_config(monkeypatch, {"VAD_TOP_DB": "30.0"})
        assert cfg.VAD_TOP_DB == 30.0


# ─── Type and range sanity ─────────────────────────────────────────

class TestConfigTypes:
    """All constants have correct types and sensible ranges."""

    def test_float_thresholds_are_floats(self):
        from app import config as cfg
        floats = [
            cfg.SILENCE_THRESHOLD_DBFS,
            cfg.RAG_MIN_TOKEN_OVERLAP,
            cfg.NOTES_GROUNDING_FLOOR,
            cfg.NOTES_GROUNDING_NORMALIZATION,
            cfg.SEMANTIC_SEARCH_THRESHOLD,
            cfg.SEMANTIC_QA_THRESHOLD,
            cfg.ENGLISH_PASSTHROUGH_RATIO,
            cfg.AUDIO_TARGET_PEAK_NORM,
            cfg.VAD_TOP_DB,
            cfg.DOMAIN_CORRECTOR_MIN_SCORE,
            cfg.HALLUCINATION_TOKEN_RATIO,
            cfg.WHISPER_NO_SPEECH_THRESHOLD,
            cfg.WHISPER_COMPRESSION_RATIO_THRESHOLD,
        ]
        for v in floats:
            assert isinstance(v, float), f"Expected float, got {type(v)} for {v}"

    def test_int_constants_are_ints(self):
        from app import config as cfg
        ints = [
            cfg.MAX_CONVERSATION_TURNS,
            cfg.STITCH_MAX_OVERLAP_WORDS,
            cfg.LOCAL_LLM_MAX_TOKENS,
        ]
        for v in ints:
            assert isinstance(v, int), f"Expected int, got {type(v)} for {v}"

    def test_overlap_ratios_in_0_1(self):
        from app import config as cfg
        ratios = [
            cfg.RAG_MIN_TOKEN_OVERLAP,
            cfg.NOTES_GROUNDING_FLOOR,
            cfg.NOTES_GROUNDING_NORMALIZATION,
            cfg.SEMANTIC_SEARCH_THRESHOLD,
            cfg.SEMANTIC_QA_THRESHOLD,
            cfg.ENGLISH_PASSTHROUGH_RATIO,
            cfg.AUDIO_TARGET_PEAK_NORM,
            cfg.HALLUCINATION_TOKEN_RATIO,
            cfg.WHISPER_NO_SPEECH_THRESHOLD,
        ]
        for v in ratios:
            assert 0.0 <= v <= 1.0, f"Ratio {v} out of [0,1] range"

    def test_silence_threshold_negative(self):
        from app import config as cfg
        assert cfg.SILENCE_THRESHOLD_DBFS < 0, "Silence threshold should be negative dBFS"

    def test_positive_durations(self):
        from app import config as cfg
        durations = [
            cfg.TARGET_CHUNK_DURATION_SECONDS,
            cfg.MAX_CHUNK_DURATION_SECONDS,
            cfg.MIN_CHUNK_DURATION_SECONDS,
            cfg.DEFAULT_CHUNK_OVERLAP_SECONDS,
            cfg.MIN_PAUSE_DURATION_SECONDS,
            cfg.STITCH_TIME_GAP_THRESHOLD,
            cfg.STITCH_CLOSE_GAP_THRESHOLD,
            cfg.VAD_MIN_SPEECH_DURATION_SEC,
        ]
        for v in durations:
            assert v > 0, f"Duration {v} should be positive"

    def test_max_gt_min_chunk(self):
        from app import config as cfg
        assert cfg.MAX_CHUNK_DURATION_SECONDS > cfg.MIN_CHUNK_DURATION_SECONDS
        assert cfg.TARGET_CHUNK_DURATION_SECONDS >= cfg.MIN_CHUNK_DURATION_SECONDS
        assert cfg.TARGET_CHUNK_DURATION_SECONDS <= cfg.MAX_CHUNK_DURATION_SECONDS

    def test_stitch_gap_ordering(self):
        from app import config as cfg
        assert cfg.STITCH_CLOSE_GAP_THRESHOLD <= cfg.STITCH_TIME_GAP_THRESHOLD

    def test_translation_confidence_ordering(self):
        from app import config as cfg
        assert cfg.TRANSLATION_CONFIDENCE_GOOGLE > cfg.TRANSLATION_CONFIDENCE_LLM
        assert cfg.TRANSLATION_CONFIDENCE_LLM > cfg.TRANSLATION_CONFIDENCE_DICTIONARY


class TestConfigDocumentation:
    """Ensure documentation cannot drift from the codebase."""

    def test_all_constants_documented_in_thresholds_md(self):
        from pathlib import Path
        from app import config as cfg

        thresholds_path = Path(__file__).resolve().parent.parent / "eval_results" / "thresholds.md"
        assert thresholds_path.exists(), "eval_results/thresholds.md does not exist!"

        thresholds_content = thresholds_path.read_text(encoding="utf-8")
        all_constants = [k for k in dir(cfg) if k.isupper() and not k.startswith("_")]

        missing = [c for c in all_constants if c not in thresholds_content]
        assert not missing, f"The following constants from app/config.py are missing in thresholds.md: {missing}"
