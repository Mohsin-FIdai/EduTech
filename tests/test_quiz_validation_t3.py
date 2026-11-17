import json
import random
from pathlib import Path

import pytest

from app.prompts.pedagogical_prompts import EXAM_MCQ_BATCH_PROMPT
from app.services.study_guide_generator import StudyGuideSynthesizer


OS_TRANSCRIPT_LOW = (
    "an operating system manages processes in states including new, ready, running, "
    "waiting or blocked, and terminated. a newly created process enters the new state. "
    "a process waiting for i/o completion is in the blocked state. turnaround time equals "
    "completion time minus arrival time. aging prevents starvation in scheduling."
)


def test_quiz_prompt_has_no_literal_angle_bracket_placeholders():
    """EXAM_MCQ_BATCH_PROMPT must not leak <concept> or <topic name> placeholders in JSON schema."""
    rendered = EXAM_MCQ_BATCH_PROMPT.format(
        batch_count=3,
        subject_domain="Operating System",
        section_transcript="Sample lecture excerpt.",
        blueprint_items_json="[]",
        covered_concepts="None",
        retry_feedback="",
    )
    assert "<topic name>" not in rendered
    assert "<choice 1 text>" not in rendered
    assert "<exam question" not in rendered
    assert '"grounded_concept": "<concept>"' not in rendered


@pytest.mark.parametrize("field", ["question", "options", "topic_area", "grounded_concept", "explanation"])
def test_validate_and_refine_rejects_unresolved_placeholders_everywhere(field):
    """Unresolved <...> placeholders must be rejected across stems, options, tags, and explanations."""
    q = {
        "category": "Conceptual",
        "difficulty": "Medium",
        "topic_area": "Process Scheduling",
        "grounded_concept": "New State",
        "question": "Which process state indicates that a process was just created?",
        "options": ["New", "Ready", "Running", "Terminated"],
        "correct_answer": "New",
        "explanation": "A newly created process enters the New state before entering the ready queue.",
    }
    if field == "options":
        q["options"] = ["New", "<choice 2 text>", "Running", "Terminated"]
    elif field == "topic_area":
        q["topic_area"] = "<concept>"
    elif field == "grounded_concept":
        q["grounded_concept"] = "<concept>"
    elif field == "question":
        q["question"] = "Which state represents <concept> in operating systems?"
    elif field == "explanation":
        q["explanation"] = "Because <concept> holds, the other choices are incorrect."

    res, err = StudyGuideSynthesizer._validate_and_refine_mcq(
        q, OS_TRANSCRIPT_LOW, [], set(), random.Random(42)
    )
    assert res is None
    assert err is not None and "placeholder" in err.lower()


def test_rejects_combined_ready_running_process_state():
    """Combined process states such as 'Ready Running' must be rejected."""
    q1_baseline = {
        "category": "Application",
        "difficulty": "Medium",
        "topic_area": "Running Instance",
        "question": "What is the state of a process when it is in the 'running' instance?",
        "options": ["Terminated", "New", "Ready Running", "Waiting Block"],
        "correct_answer": "Ready Running",
        "explanation": "A running instance is currently executing in the CPU.",
        "grounded_concept": "Running Instance",
    }
    res, err = StudyGuideSynthesizer._validate_and_refine_mcq(
        q1_baseline, OS_TRANSCRIPT_LOW, [], set(), random.Random(42)
    )
    assert res is None
    assert err is not None and "combined process-state" in err.lower()


def test_duplicate_process_state_blocked_from_first_question():
    """Duplicate concept/process-state questions must be blocked immediately from question #1."""
    seen_stems: list[str] = []
    seen_concepts: set[str] = set()
    accepted: list[dict] = []
    rng = random.Random(42)

    q_new_1 = {
        "category": "Conceptual",
        "difficulty": "Medium",
        "topic_area": "Process States",
        "grounded_concept": "New Process State",
        "question": "What does a process state of 'new' indicate?",
        "options": [
            "The process has terminated",
            "The process is executing",
            "The process is waiting for I/O",
            "The process has been created",
        ],
        "correct_answer": "The process has been created",
        "explanation": "The New state indicates the process has been created and is awaiting admission.",
    }
    q_new_2 = {
        "category": "Application",
        "difficulty": "Medium",
        "topic_area": "Operating System",
        "grounded_concept": "Process Creation",
        "question": "What is the state of a process after it has been created?",
        "options": ["Ready", "New", "Running", "Terminated"],
        "correct_answer": "New",
        "explanation": "A process enters the New state right after creation before joining the ready queue.",
    }
    q_new_same_concept = {
        "category": "Conceptual",
        "difficulty": "Easy",
        "topic_area": "Process States",
        "grounded_concept": "New Process State",
        "question": "Before joining the ready queue, which initial phase holds newly allocated processes?",
        "options": ["Ready", "New", "Running", "Blocked"],
        "correct_answer": "New",
        "explanation": "Newly initialized processes begin in the New state prior to admission.",
    }

    r1, err1 = StudyGuideSynthesizer._validate_and_refine_mcq(
        q_new_1, OS_TRANSCRIPT_LOW, seen_stems, seen_concepts, rng, seen_accepted=accepted
    )
    assert r1 is not None and err1 is None
    accepted.append(r1)

    # Blocked immediately as question #2 (previously bypassed when len(seen_stems) < 3)
    r2, err2 = StudyGuideSynthesizer._validate_and_refine_mcq(
        q_new_2, OS_TRANSCRIPT_LOW, seen_stems, seen_concepts, rng, seen_accepted=accepted
    )
    assert r2 is None and err2 is not None

    r3, err3 = StudyGuideSynthesizer._validate_and_refine_mcq(
        q_new_same_concept, OS_TRANSCRIPT_LOW, seen_stems, seen_concepts, rng, seen_accepted=accepted
    )
    assert r3 is None and err3 is not None


def test_rejects_turnaround_time_topic_mismatch_on_blocked_question():
    """A Blocked I/O question mislabeled with topic_area='Turnaround Time' must be caught."""
    q3_baseline = {
        "category": "Application",
        "difficulty": "Medium",
        "topic_area": "Turnaround Time",
        "question": "What is the state of a process when it is waiting for I/O to complete?",
        "options": ["Blocked", "Ready", "Terminated", "Running"],
        "correct_answer": "Blocked",
        "explanation": "A process waiting for an external event such as I/O completion is blocked.",
        "grounded_concept": "Turnaround Time",
    }
    res, err = StudyGuideSynthesizer._validate_and_refine_mcq(
        q3_baseline, OS_TRANSCRIPT_LOW, [], set(), random.Random(42)
    )
    assert res is None
    assert err is not None and "mismatch" in err.lower()


def test_baseline_failing_quizzes_all_caught():
    """Replay the 5 captured baseline quizzes and confirm the known defects are rejected."""
    base = Path("audit_baseline")
    if not (base / "educational_outputs.json").exists():
        pytest.skip("audit_baseline artifacts not present")

    qs = json.loads((base / "educational_outputs.json").read_text(encoding="utf-8"))["quizzes"]
    seen_stems: list[str] = []
    seen_concepts: set[str] = set()
    accepted: list[dict] = []
    rng = random.Random(42)

    outcomes = []
    for q in qs:
        res, err = StudyGuideSynthesizer._validate_and_refine_mcq(
            q, OS_TRANSCRIPT_LOW, seen_stems, seen_concepts, rng, seen_accepted=accepted
        )
        if res is not None:
            accepted.append(res)
        outcomes.append((res is not None, err))

    # Q1 Ready Running, Q2 <concept>, Q3 Turnaround Time mismatch, Q4 <concept> all rejected
    assert outcomes[0][0] is False
    assert outcomes[1][0] is False
    assert outcomes[2][0] is False
    assert outcomes[3][0] is False
    # At most Q5 ("New" with valid fields) could pass once, and never a duplicate
    assert len(accepted) <= 1


def test_llm_generate_quizzes_safely_regenerates_rejected_questions():
    """_llm_generate_quizzes retries rejected batch items cleanly without leaking defects."""
    class FakeQuizProvider:
        def __init__(self):
            self.calls = 0

        def is_available(self):
            return True

        def generate_structured(self, prompt, schema=None, system_prompt="", max_tokens=500, temperature=0.2):
            self.calls += 1
            if "examinable_items" in (schema or {}) or "examinable_items" in prompt:
                return {
                    "subject_domain": "Operating System",
                    "lecture_style": "systems",
                    "has_numerical_content": True,
                    "examinable_items": [
                        {"concept": "New State", "section": "early", "topic_area": "Process States", "category": "Conceptual", "difficulty": "Easy", "angle": "Initial state"},
                        {"concept": "Blocked State", "section": "middle", "topic_area": "Process States", "category": "Application", "difficulty": "Medium", "angle": "I/O wait"},
                        {"concept": "Turnaround Time", "section": "late", "topic_area": "Turnaround Time", "category": "Numerical", "difficulty": "Medium", "angle": "TAT formula"},
                        {"concept": "Aging", "section": "late", "topic_area": "Starvation & Aging", "category": "Conceptual", "difficulty": "Hard", "angle": "Preventing starvation"},
                    ],
                }
            # Batch call 1: returns 1 valid + 2 invalid (<concept> placeholder & Ready Running)
            if self.calls == 2:
                return {
                    "quizzes": [
                        {
                            "category": "Conceptual", "difficulty": "Easy", "topic_area": "Process States",
                            "question": "Which process state holds a newly created process prior to admission?",
                            "options": ["New", "Ready", "Running", "Terminated"],
                            "correct_answer": "New",
                            "explanation": "A newly created process begins in the New state before entering ready queue.",
                            "grounded_concept": "New State",
                        },
                        {
                            "category": "Application", "difficulty": "Medium", "topic_area": "<concept>",
                            "question": "What happens when a process is in the running state?",
                            "options": ["Executes on CPU", "Waits on disk", "Is deleted", "Created only"],
                            "correct_answer": "Executes on CPU",
                            "explanation": "In running state instructions execute on the CPU.",
                            "grounded_concept": "<concept>",
                        },
                        {
                            "category": "Application", "difficulty": "Medium", "topic_area": "Process States",
                            "question": "What is the state when a process executes on CPU?",
                            "options": ["Ready Running", "New", "Blocked", "Terminated"],
                            "correct_answer": "Ready Running",
                            "explanation": "Invalid merged state example.",
                            "grounded_concept": "Running State",
                        },
                    ]
                }
            # Batch call 2 (regeneration for remaining 2 items) + later batch/topup: clean distinct questions
            return {
                "quizzes": [
                    {
                        "category": "Application", "difficulty": "Medium", "topic_area": "Process States",
                        "question": "Which state describes a process waiting for an I/O event to finish?",
                        "options": ["Blocked", "Ready", "Running", "New"],
                        "correct_answer": "Blocked",
                        "explanation": "Waiting for I/O places a process into the Blocked state until I/O completes.",
                        "grounded_concept": "Blocked State",
                    },
                    {
                        "category": "Numerical", "difficulty": "Medium", "topic_area": "Turnaround Time",
                        "question": "If arrival time is 2 ms and completion time is 10 ms, what is turnaround time?",
                        "options": ["8 ms", "12 ms", "5 ms", "20 ms"],
                        "correct_answer": "8 ms",
                        "explanation": "Turnaround time equals completion time minus arrival time: 10 ms - 2 ms = 8 ms.",
                        "grounded_concept": "Turnaround Time",
                    },
                    {
                        "category": "Conceptual", "difficulty": "Hard", "topic_area": "Starvation & Aging",
                        "question": "How does aging prevent indefinite starvation in CPU scheduling?",
                        "options": [
                            "Gradually raises priority of waiting processes",
                            "Deletes long-waiting background processes",
                            "Forces immediate termination after I/O wait",
                            "Doubles arrival times across queues",
                        ],
                        "correct_answer": "Gradually raises priority of waiting processes",
                        "explanation": "Aging increases priority over wait duration so waiting processes eventually run.",
                        "grounded_concept": "Aging",
                    },
                ]
            }

    provider = FakeQuizProvider()
    quizzes = StudyGuideSynthesizer.generate_conceptual_quizzes(
        transcript_text=OS_TRANSCRIPT_LOW * 3,
        topic_hint="Operating System",
        lecture_notes={"study_topics": [{"title": "Process States"}, {"title": "Turnaround Time"}, {"title": "Starvation & Aging"}]},
        llm_provider=provider,
        allow_extractive_fallback=False,
    )
    assert len(quizzes) >= 4
    assert all("<" not in q["grounded_concept"] and "<" not in q["topic_area"] for q in quizzes)
    assert all(q["correct_answer"] != "Ready Running" for q in quizzes)
    assert len({q["grounded_concept"] for q in quizzes}) == len(quizzes)


@pytest.mark.parametrize(
    "expr_q,expr_expl",
    [
        (
            "What queue structure is preferred when n < 10 processes arrive concurrently?",
            "When n < 10 holds, a small array avoids heap maintenance overhead.",
        ),
        (
            "Which process runs first if burst times satisfy x < y and y > z?",
            "Because x < y and y > z holds, the scheduler picks the smaller burst time.",
        ),
        (
            "Which invariant remains safe when x<y and y>z and 0 < n and k < m?",
            "Given O(n log n) bounds with n <= 10 and x<y and y>z, indices remain valid.",
        ),
    ],
)
def test_legitimate_math_inequalities_not_rejected_as_placeholders(expr_q, expr_expl):
    """Legitimate inequalities such as n < 10 and x < y and y > z must pass placeholder detection."""
    q = {
        "category": "Technical",
        "difficulty": "Medium",
        "topic_area": "CPU Scheduling",
        "grounded_concept": "Bounds Condition",
        "question": expr_q,
        "options": [
            "Smallest remaining burst runs first",
            "Largest arrival timestamp always wins",
            "Process enters terminated state immediately",
            "Scheduler halts all I/O interrupts",
        ],
        "correct_answer": "Smallest remaining burst runs first",
        "explanation": expr_expl,
    }
    res, err = StudyGuideSynthesizer._validate_and_refine_mcq(
        q, OS_TRANSCRIPT_LOW, [], set(), random.Random(42)
    )
    assert res is not None, f"Math inequality falsely rejected: {err}"


@pytest.mark.parametrize(
    "echoed_field,echoed_value",
    [
        ("question", "Complete self-contained exam question ending with ?"),
        ("options", ["First complete answer choice", "Second complete answer choice", "Third complete answer choice", "Fourth complete answer choice"]),
        ("correct_answer", "Exact verbatim text of the winning choice"),
        ("topic_area", "Specific matching topic title from the lecture"),
        ("grounded_concept", "Specific distinct concept tested by this question"),
    ],
)
def test_echoed_prompt_schema_descriptions_explicitly_rejected(echoed_field, echoed_value):
    """Literal schema template descriptions echoed by small LLMs must be rejected as placeholders."""
    assert StudyGuideSynthesizer._has_unresolved_placeholder(
        echoed_value[0] if isinstance(echoed_value, list) else echoed_value
    )
    q = {
        "category": "Conceptual",
        "difficulty": "Medium",
        "topic_area": "Process States",
        "grounded_concept": "New State",
        "question": "Which process state holds a newly initialized process?",
        "options": ["First complete answer choice", "Ready", "Running", "Terminated"] if echoed_field == "correct_answer" else ["New", "Ready", "Running", "Terminated"],
        "correct_answer": "First complete answer choice" if echoed_field == "correct_answer" else "New",
        "explanation": "A newly created process begins in the New state prior to admission.",
    }
    if echoed_field == "options":
        q["options"] = echoed_value
        q["correct_answer"] = echoed_value[0]
    elif echoed_field == "correct_answer":
        # Test exact echoed winning-choice description even if placed inside options
        q["options"] = [echoed_value, "Ready", "Running", "Terminated"]
        q["correct_answer"] = echoed_value
    else:
        q[echoed_field] = echoed_value

    res, err = StudyGuideSynthesizer._validate_and_refine_mcq(
        q, OS_TRANSCRIPT_LOW, [], set(), random.Random(42)
    )
    assert res is None
    assert err is not None and "placeholder" in err.lower()


@pytest.mark.parametrize(
    "concept,topic,q_phrase,transcript_extra",
    [
        ("SRTF", "Shortest Remaining Time First", "Shortest Remaining Time First", "shortest remaining time first srtf preemption"),
        ("Shortest Remaining Time First", "SRTF", "SRTF", "shortest remaining time first srtf preemption"),
        ("SJF", "Shortest Job First", "Shortest Job First", "shortest job first sjf scheduling"),
        ("FCFS", "First Come First Served", "First Come First Served", "first come first served fcfs scheduling"),
        ("TLB", "Translation Lookaside Buffer", "Translation Lookaside Buffer", "translation lookaside buffer tlb paging"),
        ("PCB", "Process Control Block", "Process Control Block", "process control block pcb"),
    ],
)
def test_canonical_alias_equivalence_bidirectional(concept, topic, q_phrase, transcript_extra):
    """Acronyms and full technical names must validate consistently across concept, topic, and question."""
    study_topics = [
        {"title": topic, "explanation": f"Covers {q_phrase} operation and design.", "key_points": [q_phrase]},
    ]
    q = {
        "category": "Conceptual",
        "difficulty": "Medium",
        "topic_area": topic,
        "grounded_concept": concept,
        "question": f"How does {q_phrase} handle scheduling decisions in the operating system?",
        "options": [
            f"{q_phrase} uses specific criteria to select execution order",
            "Randomly drops memory pages on interrupt",
            "Forces every task into Blocked state at boot",
            "Merges Ready and Running into one queue",
        ],
        "correct_answer": f"{q_phrase} uses specific criteria to select execution order",
        "explanation": f"{q_phrase} operates according to the lecture mechanism for CPU/memory management.",
    }
    res, err = StudyGuideSynthesizer._validate_and_refine_mcq(
        q, f"{OS_TRANSCRIPT_LOW} {transcript_extra}", [], set(), random.Random(42), study_topics=study_topics
    )
    assert res is not None, f"Canonical alias failed for {concept} / {topic}: {err}"


def test_generic_keyword_overlap_does_not_remap_to_wrong_topic():
    """Generic words like 'process', 'waiting', 'system' must not trigger false automatic remapping."""
    study_topics = [
        {"title": "Turnaround Time", "explanation": "Completion time minus arrival time calculation."},
        {"title": "Deadlock Prevention", "explanation": "Occurs when a process is waiting indefinitely in the system queue."},
    ]
    q = {
        "category": "Application",
        "difficulty": "Medium",
        "topic_area": "Turnaround Time",
        "grounded_concept": "Blocked State",
        "question": "What is the state of a process when it is waiting for I/O completion?",
        "options": ["Blocked", "Ready", "Running", "Terminated"],
        "correct_answer": "Blocked",
        "explanation": "Waiting for I/O completion places the process in the Blocked state.",
    }
    res, err = StudyGuideSynthesizer._validate_and_refine_mcq(
        q, OS_TRANSCRIPT_LOW, [], set(), random.Random(42), study_topics=study_topics
    )
    assert res is None
    assert err is not None and "topic mismatch" in err.lower()


def test_rejects_ready_process_completed_burst_factual_error_and_allows_negation():
    """Known OS factual error ('ready process has completed its burst time') must be rejected; negation passes."""
    q_bad = {
        "category": "Application",
        "difficulty": "Medium",
        "topic_area": "Process Scheduling",
        "grounded_concept": "Process States",
        "question": "What distinguishes a ready process from a blocked process?",
        "options": [
            "The ready process can execute when dispatched",
            "The ready process waits indefinitely for disk spin-up",
            "The ready process has already exited memory",
            "The ready process cannot receive CPU cycles",
        ],
        "correct_answer": "The ready process can execute when dispatched",
        "explanation": (
            "A blocked process waits for an I/O event, whereas a ready process has completed its burst time "
            "and waits only for the next event."
        ),
    }
    res_bad, err_bad = StudyGuideSynthesizer._validate_and_refine_mcq(
        q_bad, OS_TRANSCRIPT_LOW, [], set(), random.Random(42)
    )
    assert res_bad is None
    assert err_bad is not None and "contradiction" in err_bad.lower()

    q_neg = dict(q_bad)
    q_neg["explanation"] = (
        "A blocked process waits for I/O, whereas a ready process has not completed its burst time "
        "and is waiting for CPU allocation."
    )
    res_ok, err_ok = StudyGuideSynthesizer._validate_and_refine_mcq(
        q_neg, OS_TRANSCRIPT_LOW, [], set(), random.Random(42)
    )
    assert res_ok is not None, f"Valid negated statement rejected: {err_ok}"


def test_insufficient_1_of_3_quizzes_fails_without_padding():
    """Accepting only 1 of 3 requested questions below minimum floor must fail without padding invalid items."""
    class OneValidThenInvalidProvider:
        def __init__(self):
            self.calls = 0

        def is_available(self):
            return True

        def generate_structured(self, prompt, schema=None, system_prompt="", max_tokens=500, temperature=0.2):
            self.calls += 1
            if "examinable_items" in prompt:
                return {
                    "subject_domain": "Operating System",
                    "lecture_style": "systems",
                    "has_numerical_content": False,
                    "examinable_items": [
                        {"concept": "Process States", "section": "early", "topic_area": "Process Scheduling", "category": "Conceptual", "difficulty": "Medium", "angle": "Ready vs Blocked"},
                        {"concept": "Turnaround Time", "section": "middle", "topic_area": "Turnaround Time", "category": "Numerical", "difficulty": "Medium", "angle": "Formula"},
                        {"concept": "Waiting Time", "section": "middle", "topic_area": "Waiting Time", "category": "Application", "difficulty": "Medium", "angle": "Relation"},
                    ],
                }
            if self.calls == 2:
                return {
                    "quizzes": [
                        {
                            "category": "Application",
                            "difficulty": "Medium",
                            "topic_area": "Process Scheduling",
                            "grounded_concept": "Process States",
                            "question": "What is the main difference between a ready process and a blocked process?",
                            "options": [
                                "The ready process is awaiting CPU dispatch while blocked awaits an event",
                                "The ready process is awaiting slow disk interrupts",
                                "The ready process has terminated and freed its PCB",
                                "The ready process holds an exclusive disk lock",
                            ],
                            "correct_answer": "The ready process is awaiting CPU dispatch while blocked awaits an event",
                            "explanation": "A ready process waits in memory for the CPU scheduler, whereas a blocked process waits for I/O or an external event.",
                        },
                        {
                            "category": "Numerical",
                            "difficulty": "Medium",
                            "topic_area": "Turnaround Time",
                            "grounded_concept": "Turnaround Time",
                            "question": "Compute turnaround time when burst is 10 ms and wait is 5 ms.",
                            "options": ["15 ms", "5 ms", "20 ms", "50 ms"],
                            "correct_answer": "Exact verbatim text of the winning choice",
                            "explanation": "Bad schema placeholder in correct_answer.",
                        },
                        {
                            "category": "Application",
                            "difficulty": "Medium",
                            "topic_area": "Waiting Time",
                            "grounded_concept": "Waiting Time",
                            "question": "Complete self-contained exam question ending with ?",
                            "options": ["First complete answer choice", "Second complete answer choice", "Third complete answer choice", "Fourth complete answer choice"],
                            "correct_answer": "First complete answer choice",
                            "explanation": "Echoed template placeholders.",
                        },
                    ]
                }
            return {
                "quizzes": [
                    {
                        "category": "Application",
                        "difficulty": "Medium",
                        "topic_area": "<concept>",
                        "grounded_concept": "<concept>",
                        "question": "Invalid placeholder retry should be rejected?",
                        "options": ["First complete answer choice", "B", "C", "D"],
                        "correct_answer": "B",
                        "explanation": "Invalid retry item.",
                    }
                ]
            }

    status_out: dict = {}
    quizzes = StudyGuideSynthesizer.generate_conceptual_quizzes(
        transcript_text=OS_TRANSCRIPT_LOW * 3,
        topic_hint="Operating System",
        lecture_notes={"study_topics": [{"title": "Process Scheduling"}, {"title": "Turnaround Time"}, {"title": "Waiting Time"}]},
        llm_provider=OneValidThenInvalidProvider(),
        allow_extractive_fallback=False,
        status_out=status_out,
    )
    assert quizzes == []
    assert status_out.get("status") == "failed"
    assert "insufficient" in str(status_out.get("reason", "")).lower()


def test_ready_state_immediate_execution_claim_rejected_and_dispatch_wording_accepted():
    """Reject unconditional Ready immediate-execution claim; accept scheduler-dispatch wording."""
    q_uncond = {
        "category": "Application",
        "difficulty": "Medium",
        "topic_area": "Process Scheduling",
        "grounded_concept": "Process States",
        "question": "What is the primary difference between a process that is ready and one that is blocked?",
        "options": [
            "The ready process can execute immediately without waiting",
            "The ready process awaits slow disk completion",
            "The ready process has already exited memory",
            "The ready process has permanently blocked I/O",
        ],
        "correct_answer": "The ready process can execute immediately without waiting",
        "explanation": "The ready process can execute immediately without waiting while a blocked process awaits an event.",
    }
    res_bad, err_bad = StudyGuideSynthesizer._validate_and_refine_mcq(
        q_uncond, OS_TRANSCRIPT_LOW, [], set(), random.Random(42)
    )
    assert res_bad is None
    assert err_bad is not None and "contradiction" in err_bad.lower()

    q_valid = {
        "category": "Application",
        "difficulty": "Medium",
        "topic_area": "Process Scheduling",
        "grounded_concept": "Process States",
        "question": "What distinguishes a Ready process from a Blocked process?",
        "options": [
            "A Ready process waits in the ready queue until CPU dispatch",
            "A Ready process waits indefinitely for disk interrupts",
            "A Ready process has exited all execution queues",
            "A Ready process holds all physical I/O devices",
        ],
        "correct_answer": "A Ready process waits in the ready queue until CPU dispatch",
        "explanation": (
            "A Ready process is waiting in the ready queue and can execute when the CPU scheduler dispatches it."
        ),
    }
    res_ok, err_ok = StudyGuideSynthesizer._validate_and_refine_mcq(
        q_valid, OS_TRANSCRIPT_LOW, [], set(), random.Random(42)
    )
    assert res_ok is not None, f"Valid Ready-state dispatch wording rejected: {err_ok}"


@pytest.mark.parametrize(
    "placeholder_text",
    [
        "<concept> and its application",
        "<topic name> or another concept",
    ],
)
def test_placeholders_followed_by_conjunctions_still_rejected(placeholder_text):
    """Genuine <...> placeholders followed by 'and'/'or' conjunctions outside '>' must still be rejected."""
    assert StudyGuideSynthesizer._has_unresolved_placeholder(placeholder_text)
    q = {
        "category": "Conceptual",
        "difficulty": "Medium",
        "topic_area": "Process States",
        "grounded_concept": placeholder_text,
        "question": "Which process state holds newly created programs before admission?",
        "options": ["New", "Ready", "Running", "Terminated"],
        "correct_answer": "New",
        "explanation": "Newly created processes enter the New state prior to ready admission.",
    }
    res, err = StudyGuideSynthesizer._validate_and_refine_mcq(
        q, OS_TRANSCRIPT_LOW, [], set(), random.Random(42)
    )
    assert res is None
    assert err is not None and "placeholder" in err.lower()
