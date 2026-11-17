"""
tests/test_contextual_transcription.py — Unit & Integration tests for Context-Aware High-Precision Transcription.
"""

import pytest
from app.services.contextual_transcription import (
    detect_lecture_domain,
    is_noise_segment,
    ContextualTranscriptionRefiner,
    refine_transcript_segments,
)


def test_detect_lecture_domain_os():
    text = "Today we discuss process scheduling, CPU burst time, Round Robin, Gantt chart and FCFS."
    domain = detect_lecture_domain(text)
    assert domain == "operating_systems"


def test_detect_lecture_domain_ml():
    text = "We will optimize neural networks using gradient descent and backpropagation with cross-entropy loss."
    domain = detect_lecture_domain(text)
    assert domain == "machine_learning"


def test_detect_lecture_domain_fallback():
    text = "Good morning everyone. Let us start today's lecture on general computing."
    domain = detect_lecture_domain(text)
    assert domain == "general_cs"


def test_is_noise_segment():
    assert is_noise_segment("dhk", duration=1.0) is True
    assert is_noise_segment("se", duration=0.8) is True
    assert is_noise_segment("re", duration=0.5) is True
    assert is_noise_segment("Baba Chamakthak", duration=2.0) is True
    assert is_noise_segment("bk уже", duration=1.0) is True
    assert is_noise_segment("", duration=0.0) is True

    # Valid technical short terms must NOT be pruned
    assert is_noise_segment("Gantt chart", duration=1.5) is False
    assert is_noise_segment("Formulas", duration=1.0) is False
    assert is_noise_segment("thank you", duration=1.0) is False


def test_homophone_disambiguation_aging_vs_paging():
    refiner = ContextualTranscriptionRefiner(domain="operating_systems")

    # In starvation context, "paging" should be corrected to "aging"
    segments = [
        {"segment_index": 0, "text": "context switch creates overhead, paging can reduce starvation.", "start_time": 0.0, "end_time": 4.0}
    ]
    refined = refiner.refine_segments(segments)
    assert not refined[0].was_pruned
    assert "aging can reduce starvation" in refined[0].refined_text

    # In virtual memory context, "paging" must remain "paging"
    vm_segments = [
        {"segment_index": 0, "text": "virtual memory uses paging and page replacement algorithms.", "start_time": 0.0, "end_time": 4.0}
    ]
    vm_refined = refiner.refine_segments(vm_segments)
    assert "paging" in vm_refined[0].refined_text


def test_contextual_thread_vs_threat():
    refiner = ContextualTranscriptionRefiner(domain="operating_systems")
    segments = [
        {"segment_index": 0, "text": "Program process or threat, program basically instructions hai", "start_time": 0.0, "end_time": 5.0}
    ]
    refined = refiner.refine_segments(segments)
    assert "thread" in refined[0].refined_text
    assert "threat" not in refined[0].refined_text


def test_contextual_compute_and_waiting_time():
    refiner = ContextualTranscriptionRefiner(domain="operating_systems")
    segments = [
        {"segment_index": 0, "text": "Running process is on CPU, waiting for event.", "start_time": 0.0, "end_time": 3.0},
        {"segment_index": 1, "text": "Likhe disk IO computer and time written main process is finished.", "start_time": 3.0, "end_time": 7.0}
    ]
    refined = refiner.refine_segments(segments)
    assert "compute time" in refined[1].refined_text
    assert "waiting mein process" in refined[1].refined_text
    assert "Like" in refined[1].refined_text


def test_garbled_phrase_repair():
    refiner = ContextualTranscriptionRefiner(domain="operating_systems")
    segments = [
        {"segment_index": 0, "text": "skajilad kohe kpallus si follow kurni ogin", "start_time": 0.0, "end_time": 3.0}
    ]
    refined = refiner.refine_segments(segments)
    assert "scheduler ko ek policy follow karni hogi" in refined[0].refined_text


def test_a_priori_knowledge_and_sjf_terms():
    refiner = ContextualTranscriptionRefiner(domain="operating_systems")
    segments = [
        {"segment_index": 0, "text": "predict future CPU burst time", "start_time": 0.0, "end_time": 3.0},
        {"segment_index": 1, "text": "The future of operating system is not a project knowledge, so practical system cannot predict perfectly.", "start_time": 3.0, "end_time": 7.0},
        {"segment_index": 2, "text": "SGF non-cleanable is the one-size process gets CPU until it finishes.", "start_time": 7.0, "end_time": 10.0}
    ]
    refined = refiner.refine_segments(segments)
    assert "a priori knowledge" in refined[1].refined_text
    assert "SJF non-preemptive" in refined[2].refined_text
    assert "once a process" in refined[2].refined_text


def test_multi_segment_sliding_window():
    segments = [
        {"segment_index": 0, "text": "We observe starvation in priority scheduling.", "start_time": 0.0, "end_time": 3.0},
        {"segment_index": 1, "text": "To reduce this problem, paging is introduced.", "start_time": 3.0, "end_time": 6.0},
    ]
    results = refine_transcript_segments(segments)
    assert "aging is introduced" in results[1].refined_text


def test_pruning_noise_segments():
    segments = [
        {"segment_index": 0, "text": "Process scheduling in OS.", "start_time": 0.0, "end_time": 4.0},
        {"segment_index": 1, "text": "dhk", "start_time": 4.1, "end_time": 4.8},
        {"segment_index": 2, "text": "se", "start_time": 4.9, "end_time": 5.4},
        {"segment_index": 3, "text": "Gantt chart shows the timeline.", "start_time": 5.5, "end_time": 9.0},
    ]
    results = refine_transcript_segments(segments)
    assert results[0].was_pruned is False
    assert results[1].was_pruned is True
    assert results[2].was_pruned is True
    assert results[3].was_pruned is False
