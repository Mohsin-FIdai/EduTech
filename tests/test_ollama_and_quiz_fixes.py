"""Targeted regression tests for Ollama integration and Phase C quiz validation fixes."""

from __future__ import annotations

import os
import random

import pytest
from app.services.llm_provider import (
    LLMException,
    LocalLLMProvider,
    OllamaLLMProvider,
    get_llm_provider,
)
from app.services.pedagogical_validator import PedagogicalValidator
from app.services.study_guide_generator import StudyGuideSynthesizer


def test_ollama_provider_selection_and_fallback(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3.5:4b")
    monkeypatch.delenv("ALLOW_OLLAMA_TO_LOCAL_FALLBACK", raising=False)
    OllamaLLMProvider._instance = None

    monkeypatch.setattr(OllamaLLMProvider, "is_available", lambda self: True)
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setattr("app.config.ENABLE_LOCAL_LLM", True)

    # All intelligence features route to Ollama (qwen3.5:4b) by default
    for feat in ("notes", "notes_summary", "summary", "translation", "quiz", "rag"):
        assert isinstance(get_llm_provider(feat), OllamaLLMProvider)
    assert isinstance(get_llm_provider(), OllamaLLMProvider)

    # Explicit override to 'local' routes protected features to LocalLLMProvider
    monkeypatch.setenv("NOTES_LLM_PROVIDER", "local")
    assert isinstance(get_llm_provider("notes"), LocalLLMProvider)

    # No silent fallback from Ollama to Local when ALLOW_OLLAMA_TO_LOCAL_FALLBACK is False
    monkeypatch.setattr(OllamaLLMProvider, "is_available", lambda self: False)
    with pytest.raises(LLMException):
        get_llm_provider("quiz")

    # Explicit fallback succeeds when ALLOW_OLLAMA_TO_LOCAL_FALLBACK=true
    monkeypatch.setenv("ALLOW_OLLAMA_TO_LOCAL_FALLBACK", "true")
    p_fb = get_llm_provider("quiz")
    assert isinstance(p_fb, LocalLLMProvider)

    OllamaLLMProvider._instance = None


def test_validate_and_refine_mcq_rejects_placeholders_and_mismatched_answer():
    rng = random.Random(7)
    transcript_low = "supervised learning labeled data features target regression classification training testing"
    seen_stems: list[str] = []
    seen_concepts: set[str] = set()

    # 1. Reject copied prompt stem placeholder
    q_stem_ph = {
        "question": "First exam question?",
        "options": ["Labeled inputs", "Unlabeled clusters", "Reward signal", "Random noise"],
        "correct_answer": "Labeled inputs",
        "explanation": "Supervised learning uses labeled inputs to map features to targets.",
    }
    res, err = StudyGuideSynthesizer._validate_and_refine_mcq(q_stem_ph, transcript_low, seen_stems, seen_concepts, rng)
    assert res is None and "placeholder" in (err or "").lower()

    # 2. Reject "Opt 1..4" placeholder options
    q_opt_ph = {
        "question": "What is the role of features and targets in supervised learning?",
        "options": ["Opt 1", "Opt 2", "Opt 3", "Opt 4"],
        "correct_answer": "Opt 2",
        "explanation": "Features are input variables and targets are the predicted outcomes.",
    }
    res, err = StudyGuideSynthesizer._validate_and_refine_mcq(q_opt_ph, transcript_low, seen_stems, seen_concepts, rng)
    assert res is None and err is not None

    # 3. Strip "Opt D: None of the above" and reject as banned catch-all
    q_prefixed_banned = {
        "question": "How do classification and regression differ in supervised learning?",
        "options": [
            "Opt A: Classification predicts discrete categories while regression predicts continuous values",
            "Opt B: Both tasks require unlabeled clusters",
            "Opt C: Regression only applies to text classification",
            "Opt D: None of the above",
        ],
        "correct_answer": "Opt A: Classification predicts discrete categories while regression predicts continuous values",
        "explanation": "Classification outputs discrete labels whereas regression predicts continuous numerical values.",
    }
    res, err = StudyGuideSynthesizer._validate_and_refine_mcq(q_prefixed_banned, transcript_low, seen_stems, seen_concepts, rng)
    assert res is None and "Banned" in (err or "")

    # 4. Never overwrite non-empty mismatched text answer with correct_option_letter
    q_mismatch = {
        "question": "Which regularization method penalizes large model weights during training?",
        "options": ["Dropout", "L2 weight decay", "Mini-batch SGD", "Early stopping"],
        "correct_answer": "Both A and B are correct",
        "correct_option_letter": "B",
        "explanation": "L2 weight decay penalizes large weights directly in the loss function.",
    }
    res, err = StudyGuideSynthesizer._validate_and_refine_mcq(q_mismatch, transcript_low, seen_stems, seen_concepts, rng)
    assert res is None and "does not match" in (err or "")

    # 5. Strip valid "Opt A: " prefixes cleanly on valid questions
    q_prefixed_good = {
        "question": "What distinguishes classification from regression in supervised learning?",
        "options": [
            "Opt A: Classification predicts categories whereas regression predicts continuous numbers",
            "Opt B: Classification requires no training data unlike regression",
            "Opt C: Regression only works on binary outcomes",
            "Opt D: Classification cannot evaluate test accuracy",
        ],
        "correct_answer": "Opt A: Classification predicts categories whereas regression predicts continuous numbers",
        "explanation": "Classification predicts discrete class labels while regression predicts continuous numerical quantities.",
    }
    res, err = StudyGuideSynthesizer._validate_and_refine_mcq(q_prefixed_good, transcript_low, seen_stems, seen_concepts, rng)
    assert res is not None and err is None
    assert all(not o.startswith("Opt ") for o in res["options"])
    assert res["correct_answer"] in res["options"]


def test_pedagogical_validator_rejects_opt_1_to_4():
    res = PedagogicalValidator.validate_quizzes(
        [
            {
                "question": "What is the role of features in supervised learning models?",
                "options": ["Opt 1", "Opt 2", "Opt 3", "Opt 4"],
                "correct_answer": "Opt 1",
            }
        ],
        "supervised learning features target",
        bloom_policy="quiz",
    )
    assert not res["passed"]


def test_lifecycle_reentrant_leases_mutual_exclusion_and_concurrency(monkeypatch):
    import threading
    import time
    from app.services.llm_provider import get_llm_lifecycle_manager, llm_lifecycle_session

    mgr = get_llm_lifecycle_manager()
    local_p = LocalLLMProvider.get_instance()
    ollama_p = OllamaLLMProvider.get_instance()

    events: list[str] = []
    evict_calls: list[str] = []
    unload_calls: list[str] = []

    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("NOTES_LLM_PROVIDER", "local")
    monkeypatch.setenv("SUMMARY_LLM_PROVIDER", "local")
    monkeypatch.setattr("app.config.ENABLE_LOCAL_LLM", True)
    monkeypatch.setattr(ollama_p, "is_available", lambda: True)
    monkeypatch.setattr(ollama_p, "evict_loaded_models", lambda reason="": evict_calls.append(reason) or ["qwen2.5:1.5b"])

    # Simulate LocalLLMProvider weight load/unload without loading real Torch weights
    monkeypatch.setattr(local_p, "is_loaded", lambda: getattr(local_p, "_test_loaded", False))

    def fake_unload(reason="manual"):
        if mgr.active_leases_snapshot().get("local", 0) > 0:
            return False
        if getattr(local_p, "_test_loaded", False):
            local_p._test_loaded = False
            unload_calls.append(reason)
            return True
        return False

    monkeypatch.setattr(local_p, "unload", fake_unload)

    # 1. Nested notes_summary -> notes -> summary must not deadlock and keeps weights until outer exit
    with llm_lifecycle_session("notes_summary", release_on_exit=True):
        assert mgr.current_thread_holds_lock()
        assert mgr.active_leases_snapshot()["local"] == 1
        local_p._test_loaded = True
        with llm_lifecycle_session("notes", release_on_exit=True):
            assert mgr.active_leases_snapshot()["local"] == 2
            # Must block unload while active lease > 0
            assert local_p.unload("mid_notes") is False
        assert mgr.active_leases_snapshot()["local"] == 1
        assert getattr(local_p, "_test_loaded", False) is True
        with llm_lifecycle_session("summary", release_on_exit=True):
            assert mgr.active_leases_snapshot()["local"] == 2
        assert getattr(local_p, "_test_loaded", False) is True
    assert mgr.active_leases_snapshot()["local"] == 0
    assert getattr(local_p, "_test_loaded", False) is False
    assert len(unload_calls) == 1
    assert evict_calls  # Evicted Ollama before Local loaded

    # 2. Switch to quiz (ollama) must serialize and unload local if somehow still loaded
    local_p._test_loaded = True
    with llm_lifecycle_session("quiz", release_on_exit=False):
        assert getattr(local_p, "_test_loaded", False) is False
        assert mgr.active_leases_snapshot()["ollama"] == 1
    assert mgr.active_leases_snapshot()["ollama"] == 0

    # 3. Simultaneous threads must serialize strictly (concurrent_active <= 1) and recover after error
    active_now = 0
    max_active = 0
    lock_check = threading.Lock()

    def worker(feat: str, fail: bool = False):
        nonlocal active_now, max_active
        try:
            with llm_lifecycle_session(feat, release_on_exit=True):
                with lock_check:
                    active_now += 1
                    max_active = max(max_active, active_now)
                events.append(f"enter:{feat}")
                time.sleep(0.05)
                if fail:
                    raise RuntimeError("simulated provider error")
                events.append(f"exit:{feat}")
        except RuntimeError:
            events.append(f"err:{feat}")
        finally:
            with lock_check:
                active_now -= 1

    t1 = threading.Thread(target=worker, args=("notes", True))
    t2 = threading.Thread(target=worker, args=("quiz", False))
    t1.start()
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)
    assert max_active == 1
    assert mgr.active_leases_snapshot() == {"local": 0, "ollama": 0}
    assert "err:notes" in events and "exit:quiz" in events


def test_translation_threadpool_no_deadlock_when_lock_held(monkeypatch):
    from app.services.llm_provider import get_llm_lifecycle_manager, llm_lifecycle_session
    from app.services.text_processor import TextProcessingService

    mgr = get_llm_lifecycle_manager()
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("NOTES_LLM_PROVIDER", "local")
    monkeypatch.setenv("TRANSLATION_LLM_PROVIDER", "local")
    monkeypatch.setattr("app.config.ENABLE_LOCAL_LLM", True)
    monkeypatch.setattr("app.config.ENABLE_ONLINE_TRANSLATION", False)
    monkeypatch.setattr(
        LocalLLMProvider,
        "generate_text",
        lambda self, prompt, system_prompt="", max_tokens=40, temperature=0.1: "Gradient descent minimizes loss.",
    )

    class Seg:
        def __init__(self, txt: str):
            self.text = txt
            self.normalized_text = txt
            self.english_text = ""
            self.translation_source = ""
            self.translation_confidence = 0.0
            self.translation_reason = ""

    seg = Seg("aaj hum gradient descent parhenge")
    with llm_lifecycle_session("notes", release_on_exit=False):
        assert mgr.current_thread_holds_lock()
        TextProcessingService.translate_segments_batch([seg])
    assert seg.english_text == "Gradient descent minimizes loss."
    assert seg.translation_source == "local_llm"
