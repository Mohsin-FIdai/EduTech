import logging
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import whisper.tokenizer as wtok

from app.services.domain_corrector import (
    WHISPER_DOMAIN_PROMPT,
    WHISPER_PROMPT_MAX_TOKENS,
    build_whisper_initial_prompt,
    count_whisper_prompt_tokens,
    get_whisper_prompt,
)
from app.services.transcription_service import TranscriptionService


def _actual_whisper_prompt_tokens(text: str, multilingual: bool = True) -> int:
    tok = wtok.get_tokenizer(multilingual=multilingual)
    return len(tok.encode(" " + text.strip()))


def test_topic_hint_absent_uses_general_vocabulary_prompt():
    """Empty, whitespace-only, or None topic hints must fall back to the full general vocabulary prompt."""
    for blank in (None, "", "   \t\n "):
        prompt = build_whisper_initial_prompt(blank)
        assert prompt == WHISPER_DOMAIN_PROMPT
        assert "process scheduling" in prompt
        assert "FCFS" in prompt and "Round Robin" in prompt
        assert _actual_whisper_prompt_tokens(prompt) <= WHISPER_PROMPT_MAX_TOKENS


def test_topic_hint_present_retains_both_general_vocab_and_topic_hint():
    """When topic hint is present, both general technical vocabulary and topic hint must be retained."""
    topic = "Operating systems lecture: process scheduling, SRTF, PCB, waiting time"
    prompt = build_whisper_initial_prompt(topic)

    # Retains both general vocabulary and topic-specific hint
    assert WHISPER_DOMAIN_PROMPT in prompt
    assert "Gantt chart" in prompt
    assert "Round Robin" in prompt
    assert "backpropagation" in prompt
    assert topic in prompt
    assert _actual_whisper_prompt_tokens(prompt) <= WHISPER_PROMPT_MAX_TOKENS


def test_oversized_topic_hint_respects_whisper_223_token_limit_and_retains_both(caplog):
    """Even with a very long topic hint, prompt never exceeds 223 Whisper tokens and retains both sources."""
    long_topic = ", ".join(
        [
            "Operating System Process Scheduling",
            "Shortest Remaining Time First (SRTF)",
            "Process Control Block (PCB)",
            "Context Switching Overhead",
            "Ready Queue Dispatching",
            "Blocked I/O State Transitions",
            "Turnaround Time vs Waiting Time",
            "Response Time Optimization",
            "Convoy Effect in First Come First Served",
            "Aging to Prevent Starvation",
        ]
        + [f"Scheduling Subtopic {i} with preemptive quantum analysis" for i in range(1, 35)]
    )
    assert _actual_whisper_prompt_tokens(long_topic) > WHISPER_PROMPT_MAX_TOKENS

    with caplog.at_level(logging.WARNING):
        prompt = build_whisper_initial_prompt(long_topic)

    actual_tokens = _actual_whisper_prompt_tokens(prompt)
    assert actual_tokens <= WHISPER_PROMPT_MAX_TOKENS
    # General technical terms retained
    assert "operating systems" in prompt
    assert "FCFS" in prompt and "Round Robin" in prompt
    assert "turnaround time" in prompt
    # Leading topic-specific concepts retained
    assert "Operating System Process Scheduling" in prompt
    assert "Shortest Remaining Time First (SRTF)" in prompt
    assert "Process Control Block (PCB)" in prompt


def test_tight_custom_token_budget_splits_fairly_without_dropping_either_source(caplog):
    """Under a tighter n_text_ctx budget, both general technical terms and topic hint remain represented."""
    topic = "Computer Architecture Pipelining Hazards Forwarding"
    with caplog.at_level(logging.WARNING):
        prompt = build_whisper_initial_prompt(topic, max_tokens=80)

    assert _actual_whisper_prompt_tokens(prompt) <= 80
    assert "operating systems" in prompt
    assert "Computer Architecture" in prompt


def test_build_effective_initial_prompt_reads_model_dims_and_multilingual():
    """TranscriptionService.build_effective_initial_prompt inspects model.dims.n_text_ctx when available."""
    mock_model = SimpleNamespace(
        dims=SimpleNamespace(n_text_ctx=448),
        is_multilingual=True,
    )
    prompt = TranscriptionService.build_effective_initial_prompt(
        "OS Scheduling & Deadlocks", model=mock_model, model_size="base"
    )
    assert WHISPER_DOMAIN_PROMPT in prompt
    assert "OS Scheduling & Deadlocks" in prompt
    assert count_whisper_prompt_tokens(prompt, multilingual=True) <= 223


def test_transcribe_recording_passes_combined_prompt_without_changing_decode_settings(tmp_path):
    """End-to-end TranscriptionService wiring combines domain prompt + recording.topic cleanly."""
    chunk_path = tmp_path / "chunk_0.wav"
    chunk_path.write_bytes(b"RIFF")

    captured_kwargs = {}

    class FakeWhisperModel:
        dims = SimpleNamespace(n_text_ctx=448)
        is_multilingual = True

        def transcribe(self, **kwargs):
            captured_kwargs.update(kwargs)
            return {
                "language": "en",
                "segments": [
                    {
                        "start": 0.0,
                        "end": 4.0,
                        "text": "Today we discuss operating systems and process scheduling.",
                        "avg_logprob": -0.2,
                        "no_speech_prob": 0.01,
                    }
                ],
            }

    fake_recording = SimpleNamespace(
        id="rec-t4",
        topic="Operating Systems: Process Scheduling & SRTF",
        duration_seconds=10.0,
        status="CHUNKED",
    )
    fake_chunk = SimpleNamespace(
        id="chk-0",
        chunk_index=0,
        start_time=0.0,
        end_time=10.0,
        duration=10.0,
        file_path=str(chunk_path),
    )

    class _Query:
        def __init__(self, obj):
            self._obj = obj

        def filter(self, *args, **kwargs):
            return self

        def order_by(self, *args, **kwargs):
            return self

        def first(self):
            return self._obj

        def all(self):
            return self._obj if isinstance(self._obj, list) else []

    class FakeDB:
        def __init__(self):
            self.added = []

        def query(self, model_cls):
            name = getattr(model_cls, "__name__", "")
            if name == "Recording":
                return _Query(fake_recording)
            if name == "Chunk":
                return _Query([fake_chunk])
            return _Query([])

        def commit(self):
            pass

        def flush(self):
            pass

        def refresh(self, obj):
            pass

        def add(self, obj):
            if getattr(obj, "id", None) is None:
                obj.id = f"id-{len(self.added)}"
            self.added.append(obj)

    fake_audio = np.zeros(16000, dtype=np.float32)
    with (
        patch("app.services.transcription_service.TranscriptionService.cleanup_transcripts"),
        patch("app.services.transcription_service.WhisperModelManager.get_model", return_value=FakeWhisperModel()),
        patch("app.services.transcription_service.sf.read", return_value=(fake_audio, 16000)),
        patch("app.services.transcription_service.refine_transcript_segments", side_effect=lambda segs, **kw: []),
        patch("app.services.transcription_service.TranscriptionService._nlp_pipeline", return_value=SimpleNamespace(normalized="Today we discuss operating systems and process scheduling.")),
    ):
        TranscriptionService.transcribe_recording("rec-t4", FakeDB(), model_size="base")

    eff_prompt = captured_kwargs["initial_prompt"]
    assert WHISPER_DOMAIN_PROMPT in eff_prompt
    assert "Operating Systems: Process Scheduling & SRTF" in eff_prompt
    assert _actual_whisper_prompt_tokens(eff_prompt) <= 223
    # Decoding settings: language detection untouched; T5 sets temperature=(0.0, 0.2) and condition_on_previous_text=False
    assert captured_kwargs["condition_on_previous_text"] is False
    assert captured_kwargs["temperature"] == (0.0, 0.2)
    assert "language" not in captured_kwargs
