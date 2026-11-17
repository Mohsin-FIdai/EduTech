"""
tests/test_topic_intelligence.py -- Comprehensive Validation & Regression Test Suite
for Local-First AI Lecture Understanding & Study Notes (v3).

Covers:
  - Test 1: Computer Science Lecture
  - Test 2: Biology Lecture
  - Test 3: Business / Economics Lecture
  - Test 4: Statistics Lecture
  - Test 5: Bilingual Roman Urdu + English Code-Switching
  - Test 6: Long Lecture (> 35 segments)
  - Test 7: Lecture with Repeated Concepts (Merging verification)
  - Test 8: Lecture with Concrete Examples
  - Test 9: Lecture with Mathematical Formulas
  - Test 10: Lecture with Few Obvious Headings
  - Test 11: Original 45-Second Fixed Interval Bug Regression
  - Test 12: Grounding and Quality Gate Verification
"""

from app.services.topic_intelligence import (
    Segment, TopicSpan, StudyTopic, SubTopic,
    LectureStudyNotes, DiscourseRole, ContextWindowProcessor,
    DiscourseRoleClassifier, ClaimEntailmentValidator,
    TopicIntelligenceEngine, StudyNotesQualityValidator,
    analyze_lecture_study_notes
)


# ---------------------------------------------------------------------------
# Test 1: Computer Science Lecture
# ---------------------------------------------------------------------------
def test_1_computer_science_lecture():
    engine = TopicIntelligenceEngine(use_llm=False)
    data = [
        (0, 0.0, 20.0, "Today we begin CPU process scheduling in modern operating systems."),
        (1, 20.0, 45.0, "Processes wait in the ready queue until the kernel dispatcher grants execution context."),
        (2, 45.0, 75.0, "CPU burst time is the continuous duration a thread executes before voluntary or forced yielding."),
        (3, 75.0, 110.0, "Next we examine race conditions and deadlock in multi-threaded programs."),
        (4, 110.0, 145.0, "Deadlock occurs when multiple threads wait indefinitely on mutual exclusion locks."),
        (5, 145.0, 180.0, "To prevent deadlock, we can enforce strict lock ordering or use Peterson's algorithm.")
    ]
    segs = [Segment(id=d[0], start=d[1], end=d[2], text=d[3]) for d in data]
    text = " ".join(s.text for s in segs)
    notes = engine.analyze_lecture_study_notes(text, segs, 180.0)

    assert len(notes.study_topics) >= 2
    for t in notes.study_topics:
        assert len(t.title.split()) >= 1
        assert t.title.lower() not in ["algorithm", "deep", "feature", "lecture"]
        assert len(t.explanation) >= 35
        assert t.spans[0].start < t.spans[-1].end


# ---------------------------------------------------------------------------
# Test 2: Biology Lecture (Subject Independence)
# ---------------------------------------------------------------------------
def test_2_biology_lecture():
    engine = TopicIntelligenceEngine(use_llm=False)
    data = [
        (0, 0.0, 25.0, "In today's cellular biology lecture, we examine eukaryotic cell division and the mitotic phase."),
        (1, 25.0, 55.0, "During prophase, chromatin fibers condense into distinct chromosomes with paired sister chromatids."),
        (2, 55.0, 90.0, "The nuclear envelope breaks down as spindle fibers attach to the kinetochore protein complexes."),
        (3, 90.0, 125.0, "Moving to cellular respiration, mitochondria generate adenosine triphosphate through oxidative phosphorylation."),
        (4, 125.0, 160.0, "Electrons pass down the electron transport chain, creating a proton gradient across the inner membrane."),
        (5, 160.0, 195.0, "ATP synthase uses this electrochemical chemiosmotic gradient to phosphorylate ADP into ATP.")
    ]
    segs = [Segment(id=d[0], start=d[1], end=d[2], text=d[3]) for d in data]
    text = " ".join(s.text for s in segs)
    notes = engine.analyze_lecture_study_notes(text, segs, 195.0)

    assert len(notes.study_topics) >= 2
    # Ensure no hardcoded CS assumptions
    all_titles = " ".join(t.title for t in notes.study_topics).lower()
    assert not any(cs_term in all_titles for cs_term in ["algorithm", "scheduling", "deadlock", "cpu"])


# ---------------------------------------------------------------------------
# Test 3: Business / Economics Lecture
# ---------------------------------------------------------------------------
def test_3_business_economics_lecture():
    engine = TopicIntelligenceEngine(use_llm=False)
    data = [
        (0, 0.0, 30.0, "Welcome to microeconomics. Today we explore market equilibrium and price elasticity of demand."),
        (1, 30.0, 65.0, "Demand curves slope downward because marginal utility diminishes as consumption increases."),
        (2, 65.0, 100.0, "When quantity supplied equals quantity demanded, the clearing price establishes market equilibrium."),
        (3, 100.0, 135.0, "Now consider corporate capital structure and debt financing versus equity issuance."),
        (4, 135.0, 170.0, "Leverage increases return on equity during expansionary periods but escalates bankruptcy risk during recessions.")
    ]
    segs = [Segment(id=d[0], start=d[1], end=d[2], text=d[3]) for d in data]
    text = " ".join(s.text for s in segs)
    notes = engine.analyze_lecture_study_notes(text, segs, 170.0)

    assert len(notes.study_topics) >= 2
    for t in notes.study_topics:
        assert len(t.explanation) >= 35


# ---------------------------------------------------------------------------
# Test 4: Statistics Lecture
# ---------------------------------------------------------------------------
def test_4_statistics_lecture():
    engine = TopicIntelligenceEngine(use_llm=False)
    data = [
        (0, 0.0, 25.0, "Today we cover inferential statistics and statistical hypothesis testing fundamentals."),
        (1, 25.0, 60.0, "The null hypothesis assumes no effect or difference exists between study populations."),
        (2, 60.0, 95.0, "If the computed p-value is strictly below the significance threshold alpha, we reject the null hypothesis."),
        (3, 95.0, 130.0, "Next we discuss ordinary least squares linear regression and variance analysis."),
        (4, 130.0, 165.0, "The coefficient of determination R-squared measures the proportion of dependent variable variance explained.")
    ]
    segs = [Segment(id=d[0], start=d[1], end=d[2], text=d[3]) for d in data]
    text = " ".join(s.text for s in segs)
    notes = engine.analyze_lecture_study_notes(text, segs, 165.0)

    assert len(notes.study_topics) >= 2
    assert any("p-value" in t.explanation.lower() or "hypothesis" in t.explanation.lower() or "linear" in t.explanation.lower() for t in notes.study_topics)


# ---------------------------------------------------------------------------
# Test 5: Bilingual Roman Urdu + English Code-Switching Lecture
# ---------------------------------------------------------------------------
def test_5_bilingual_code_switching_lecture():
    engine = TopicIntelligenceEngine(use_llm=False)
    data = [
        (0, 0.0, 22.0, "Aaj ka topic CPU process scheduling hai. Multiprogramming environment mein CPU time divide hota hai."),
        (1, 22.0, 50.0, "Operating system decide karta hai ke ready queue mein se kaunsa process pehle execute hoga."),
        (2, 50.0, 80.0, "Turnaround time calculate karne ke liye completion time minus arrival time use hota hai."),
        (3, 80.0, 115.0, "Ab Gantt chart par aate hain. Gantt chart se timeline plot ki jati hai."),
        (4, 115.0, 150.0, "Ye chart show karta hai ke har process kis time start hua aur kab finish hua. Paper ke liye bohot zaroori hai.")
    ]
    segs = [Segment(id=d[0], start=d[1], end=d[2], text=d[3]) for d in data]
    text = " ".join(s.text for s in segs)
    notes = engine.analyze_lecture_study_notes(text, segs, 150.0)

    assert len(notes.study_topics) >= 2
    # Check that bilingual concepts are extracted without crash
    assert all(len(t.explanation) >= 30 for t in notes.study_topics)


# ---------------------------------------------------------------------------
# Test 6: Long Lecture (> 35 Segments)
# ---------------------------------------------------------------------------
def test_6_long_lecture():
    engine = TopicIntelligenceEngine(use_llm=False)
    segs = []
    # Create 36 segments across 4 distinct thematic blocks
    themes = [
        "Network fundamentals and packet routing across autonomous systems.",
        "Transmission Control Protocol three-way handshake and sequence acknowledgement.",
        "Domain Name System hierarchical resolution and recursive caching servers.",
        "Transport Layer Security cryptographic handshakes and public key certificates."
    ]
    for i in range(36):
        theme_idx = min(i // 9, len(themes) - 1)
        segs.append(Segment(
            id=i,
            start=float(i * 20),
            end=float((i + 1) * 20),
            text=f"Segment {i}: {themes[theme_idx]} Additional detail about network packet protocol layer {i}."
        ))

    text = " ".join(s.text for s in segs)
    notes = engine.analyze_lecture_study_notes(text, segs, 720.0)

    assert 2 <= len(notes.study_topics) <= 10
    assert notes.study_topics[-1].spans[-1].end <= 720.0


# ---------------------------------------------------------------------------
# Test 7: Lecture with Repeated Concepts (Topic Merging)
# ---------------------------------------------------------------------------
def test_7_repeated_concepts_merging():
    t1 = StudyTopic(
        id="t1",
        title="Round Robin Scheduling",
        spans=[TopicSpan(0.0, 30.0, 0, 2)],
        explanation="Round Robin algorithm with time quantum rotation.",
        key_points=["Time quantum concept"],
        source_segment_ids=[0, 1, 2]
    )
    t2 = StudyTopic(
        id="t2",
        title="Round Robin Scheduling",
        spans=[TopicSpan(80.0, 110.0, 6, 8)],
        explanation="Further analysis of Round Robin scheduling.",
        key_points=["Preemption rules"],
        source_segment_ids=[6, 7, 8]
    )
    engine = TopicIntelligenceEngine(use_llm=False)
    merged = engine._reconcile_and_merge_topics([t1, t2])

    assert len(merged) == 1
    assert len(merged[0].spans) == 2
    assert merged[0].spans[0].start == 0.0
    assert merged[0].spans[1].start == 80.0
    assert merged[0].source_segment_ids == [0, 1, 2, 6, 7, 8]


# ---------------------------------------------------------------------------
# Test 8: Lecture Containing Examples
# ---------------------------------------------------------------------------
def test_8_lecture_with_examples():
    engine = TopicIntelligenceEngine(use_llm=False)
    data = [
        (0, 0.0, 30.0, "Operating systems manage concurrent software applications."),
        (1, 30.0, 60.0, "For example Chrome Spotify and VS Code can run concurrently on modern desktop computers."),
        (2, 60.0, 90.0, "The operating system schedules these processes so they share the physical microprocessor.")
    ]
    segs = [Segment(id=d[0], start=d[1], end=d[2], text=d[3]) for d in data]
    text = " ".join(s.text for s in segs)
    notes = engine.analyze_lecture_study_notes(text, segs, 90.0)

    assert len(notes.study_topics) >= 1
    has_ex = any(len(t.examples) > 0 or "chrome" in t.explanation.lower() for t in notes.study_topics)
    assert has_ex, "Example should be captured in examples list or explanation"


# ---------------------------------------------------------------------------
# Test 9: Lecture Containing Mathematical Formulas
# ---------------------------------------------------------------------------
def test_9_lecture_with_formulas():
    engine = TopicIntelligenceEngine(use_llm=False)
    data = [
        (0, 0.0, 30.0, "Now we compute scheduling performance metrics for process turnaround."),
        (1, 30.0, 60.0, "Turnaround time equals completion time minus arrival time. Turnaround time = Completion - Arrival."),
        (2, 60.0, 90.0, "Waiting time equals turnaround time minus burst time. Waiting time = Turnaround - Burst."),
        (3, 90.0, 120.0, "Note these equations down, they are critical for the exam.")
    ]
    segs = [Segment(id=d[0], start=d[1], end=d[2], text=d[3]) for d in data]
    text = " ".join(s.text for s in segs)
    notes = engine.analyze_lecture_study_notes(text, segs, 120.0)

    assert len(notes.study_topics) >= 1
    has_formula = any(len(t.formulas) > 0 or "=" in t.explanation for t in notes.study_topics)
    assert has_formula, "Formula should be extracted and preserved"


# ---------------------------------------------------------------------------
# Test 10: Lecture with Few Obvious Headings (Latent Concept Discovery)
# ---------------------------------------------------------------------------
def test_10_lecture_without_obvious_headings():
    engine = TopicIntelligenceEngine(use_llm=False)
    # Conversational narrative with no explicit "Topic 1" or "Next section" cues
    data = [
        (0, 0.0, 35.0, "Whenever you deploy containers to Kubernetes, pods get distributed across worker nodes by kube-scheduler."),
        (1, 35.0, 70.0, "Node affinity and taints determine whether a pod can be scheduled on a specific machine."),
        (2, 70.0, 105.0, "Later if traffic spikes, horizontal pod autoscalers monitor CPU utilization metrics and spin up replicas."),
        (3, 105.0, 140.0, "Persistent volumes decouple storage lifecycles from pod crashes using CSI storage drivers.")
    ]
    segs = [Segment(id=d[0], start=d[1], end=d[2], text=d[3]) for d in data]
    text = " ".join(s.text for s in segs)
    notes = engine.analyze_lecture_study_notes(text, segs, 140.0)

    assert len(notes.study_topics) >= 1
    assert all(t.title != "Lecture Topic" for t in notes.study_topics)


# ---------------------------------------------------------------------------
# Test 11: ORIGINAL BUG REGRESSION TEST (Anti-Uniform 45-Second Interval)
# ---------------------------------------------------------------------------
def test_11_original_bug_regression_no_45s_intervals():
    """
    CRITICAL REGRESSION TEST:
    Proves that the old bug (chopping transcripts into fixed 45-second windows:
    00:00-00:45, 00:45-01:30, 01:30-02:15) is completely eliminated.
    """
    engine = TopicIntelligenceEngine(use_llm=False)
    # 15 segments with natural speech durations (total ~300 seconds)
    data = [
        (0, 0.0, 18.2, "Introduction to process scheduling concepts."),
        (1, 18.2, 42.1, "Ready queue and dispatcher mechanics."),
        (2, 42.1, 79.5, "Detailed explanation of FCFS and non-preemptive execution."),
        (3, 79.5, 115.0, "Convoy effect problem when long burst processes execute."),
        (4, 115.0, 162.3, "Round Robin scheduling with time quantum rotation."),
        (5, 162.3, 210.0, "Context switch overhead analysis for small time slices."),
        (6, 210.0, 255.4, "Gantt chart drawing and turnaround time calculations."),
        (7, 255.4, 300.0, "Waiting time formulas and exam summary points.")
    ]
    segs = [Segment(id=d[0], start=d[1], end=d[2], text=d[3]) for d in data]
    text = " ".join(s.text for s in segs)
    notes = engine.analyze_lecture_study_notes(text, segs, 300.0)

    assert len(notes.study_topics) >= 2
    starts = [t.spans[0].start for t in notes.study_topics]
    ends = [t.spans[-1].end for t in notes.study_topics]

    # Check 1: Intervals must NOT all be 45.0 seconds
    diffs = [starts[i+1] - starts[i] for i in range(len(starts)-1)]
    assert not all(abs(d - 45.0) < 1.0 for d in diffs), f"Found uniform 45s intervals: {diffs}"

    # Check 2: Durations must vary naturally
    durations = [e - s for s, e in zip(starts, ends)]
    assert max(durations) != min(durations), "Topic durations must not all be identical"

    # Check 3: No topic title should be a crude single noun
    for t in notes.study_topics:
        assert t.title.lower() not in ["algorithm", "deep", "feature", "lecture", "process", "system"]


# ---------------------------------------------------------------------------
# Test 12: Grounding and Quality Gate Verification
# ---------------------------------------------------------------------------
def test_12_grounding_and_quality_gate():
    engine = TopicIntelligenceEngine(use_llm=False)
    data = [
        (0, 0.0, 30.0, "Database transaction management and ACID properties."),
        (1, 30.0, 65.0, "Atomicity ensures all operations in a transaction succeed or none execute."),
        (2, 65.0, 100.0, "Consistency guarantees database invariants remain valid before and after commits."),
        (3, 100.0, 140.0, "Isolation prevents concurrent transactions from observing intermediate states."),
        (4, 140.0, 180.0, "Durability ensures committed changes survive system crashes or power failures.")
    ]
    segs = [Segment(id=d[0], start=d[1], end=d[2], text=d[3]) for d in data]
    text = " ".join(s.text for s in segs)
    notes = engine.analyze_lecture_study_notes(text, segs, 180.0)

    validator = StudyNotesQualityValidator()
    passed, score, details = validator.validate(notes, 180.0)

    assert score >= 0.65, f"Quality score too low: {score}"
    assert passed is True
    assert details["topic_count"]["pass"] is True
    assert details["anti_uniform_interval"]["pass"] is True
    assert details["meaningful_titles"]["pass"] is True
    assert details["non_tautological_explanations"]["pass"] is True
    assert details["timestamp_grounding"]["pass"] is True
    assert details["source_grounding"]["pass"] is True
    assert details["content_groundedness"]["pass"] is True


# ---------------------------------------------------------------------------
# Test 13: Content Groundedness Gate Catches Fabricated Topic #4
# ---------------------------------------------------------------------------
def test_13_groundedness_check_fails_on_fabricated_baba_chamakthak_topic():
    """
    CRITICAL QUALITY GATE AUDIT TEST:
    Verifies that the new content groundedness check in StudyNotesQualityValidator
    specifically catches and FAILS the confabulated Topic #4:
    "Baba Chamakthak Konvai Effect: A Critical Analysis Of Process Optimization Techniques".
    """
    # Create the exact 4 topics as previously produced by Qwen2.5-0.5B
    t1 = StudyTopic(
        id="topic_01",
        title="Operating Systems: Process Scheduling",
        spans=[TopicSpan(0.0, 38.8, 0, 5)],
        explanation="This section introduces fundamental concepts of CPU scheduling and process priority.",
        key_points=["Processes allocated CPU time based on priority", "Operating system decides process ordering"],
        source="local_llm",
        source_segment_ids=[0, 1, 2, 3, 4, 5]
    )
    t2 = StudyTopic(
        id="topic_02",
        title="Understanding Process Management In Windows Systems",
        spans=[TopicSpan(48.5, 168.1, 6, 29)],
        explanation="The lecture explains how processes work in Windows systems, focusing on lifecycle.",
        key_points=["Processes created when program starts", "CreateProcess function in Windows API"],
        source="local_llm",
        source_segment_ids=list(range(6, 30))
    )
    t3 = StudyTopic(
        id="topic_03",
        title="Cpu Burst Time",
        spans=[TopicSpan(169.0, 259.8, 30, 35)],
        explanation="Round Robin algorithm and CPU burst times where processes wait their turn.",
        key_points=["Ensures no process waits longer than turn", "Minimizes waiting time"],
        source="local_llm",
        source_segment_ids=list(range(30, 36))
    )
    # The problematic Topic #4
    t4_fabricated = StudyTopic(
        id="topic_04",
        title="Baba Chamakthak Konvai Effect: A Critical Analysis Of Process Optimization Techniques",
        spans=[TopicSpan(259.8, 332.9, 36, 40)],
        explanation="The Baba Chamakthak Konvai Effect refers to a scenario where a long process is stacked with multiple shorter processes, leading to theoretical faster overall performance.",
        key_points=["Stacking Multiple Processes", "Baba Chamakthak Effect", "Optimization Techniques"],
        source="local_llm",
        source_segment_ids=[36, 37, 38, 39, 40]
    )

    unhealed_notes = LectureStudyNotes(
        study_topics=[t1, t2, t3, t4_fabricated],
        lecture_overview="The lecture discusses CPU process scheduling.",
        subject="Operating Systems",
        theme="Process Scheduling",
        analyzer_used="local_llm"
    )

    validator = StudyNotesQualityValidator()
    passed, score, details = validator.validate(unhealed_notes, 510.9)

    # Must FAIL because Topic #4 contains hallucinated noise entity 'Baba Chamakthak'
    # and ungrounded 'Process Optimization Techniques'
    assert details["content_groundedness"]["pass"] is False
    assert details["content_groundedness"]["grounded_topics"] < details["content_groundedness"]["total_topics"]
    assert passed is False, f"Fabricated topic #4 should have failed quality gate, but passed={passed}"


# ---------------------------------------------------------------------------
# Test 14: Generalized Groundedness Passes Paraphrase & Fails Novel Fabrications
# ---------------------------------------------------------------------------
def test_14_generalized_groundedness_passes_legitimate_paraphrase_and_fails_novel_synthetic_fabrications():
    """
    VERIFICATION OF GENERALIZED GROUNDING (NO BLOCKLIST RELIANCE):
    Proves that:
    1. Legitimate academic paraphrasing (e.g. 'Convoy Effect in FCFS Scheduling' where audio said 'Konvai') PASSES.
    2. Novel synthetic fabrications (Quantum Hyperthreading, Mitochondrial Respiration, Alien Terms) FAIL.
    """
    source_convoy_text = (
        "Konvai Effect Basically short process, a long process is stack in the feature line. "
        "Likhe motorway, X load truck and rear of it, the entire line is stack. "
        "The par is theoretically fast, practical speed slow, truck is ahead. "
        "Game concept, CPU scheduling also applies. FCFS advantage is tha algorithm is very simple, "
        "implementation, starvation normally not because process is Q and eventually turned on."
    )

    validator = StudyNotesQualityValidator()

    # Case 1: Legitimate Paraphrase (Convoy Effect in FCFS) -> MUST PASS
    legitimate_topic = StudyTopic(
        id="top_01",
        title="Convoy Effect In FCFS Scheduling",
        explanation="The lecturer explains the convoy effect where short processes are delayed behind a long process in First-Come First-Served scheduling, using a motorway truck analogy.",
        key_points=["Long processes block subsequent short processes", "FCFS algorithm causes high average waiting times"],
        source="local_llm",
        source_text=source_convoy_text
    )
    legit_notes = LectureStudyNotes(
        study_topics=[legitimate_topic],
        lecture_overview="Overview of convoy effect in CPU scheduling.",
        subject="Operating Systems",
        theme="Process Scheduling"
    )
    p_legit, s_legit, d_legit = validator.validate(legit_notes, 120.0)
    assert d_legit["content_groundedness"]["pass"] is True, f"Legitimate paraphrase should pass groundedness, details={d_legit}"

    # Case 2: Synthetic Fabrication A (Quantum Architecture on OS transcript) -> MUST FAIL
    syn_a = StudyTopic(
        id="top_syn_a",
        title="Quantum Hyperthreading Architecture: Vector Acceleration",
        explanation="Introduces quantum entanglement registers and vector pipeline acceleration units inside superscalar processors.",
        key_points=["Entangled qubit registers reduce gate propagation delay", "Vector acceleration pipelines"],
        source="local_llm",
        source_text=source_convoy_text
    )
    notes_a = LectureStudyNotes(study_topics=[syn_a], lecture_overview="Overview", subject="OS")
    p_a, s_a, d_a = validator.validate(notes_a, 120.0)
    assert d_a["content_groundedness"]["pass"] is False, "Synthetic fabrication A must fail groundedness"

    # Case 3: Synthetic Fabrication B (Cellular Biology on OS transcript) -> MUST FAIL
    syn_b = StudyTopic(
        id="top_syn_b",
        title="Mitochondrial Cellular Respiration",
        explanation="Examines oxidative phosphorylation and electron transport chains producing ATP in the inner membrane.",
        key_points=["Proton gradients drive ATP synthase rotary catalysis", "NADH and FADH2 donate electrons"],
        source="local_llm",
        source_text=source_convoy_text
    )
    notes_b = LectureStudyNotes(study_topics=[syn_b], lecture_overview="Overview", subject="OS")
    p_b, s_b, d_b = validator.validate(notes_b, 120.0)
    assert d_b["content_groundedness"]["pass"] is False, "Synthetic fabrication B must fail groundedness"


# ---------------------------------------------------------------------------
# Test 15: Context Window Processor & Discourse Role Classification
# ---------------------------------------------------------------------------
def test_15_context_window_and_discourse_roles():
    segs = [
        Segment(0, "Welcome to operating systems lecture today.", 0.0, 15.0),
        Segment(1, "Let's say a user is touching a hot pan or object.", 15.0, 35.0),
        Segment(2, "A reflex action is defined as an involuntary automatic response.", 35.0, 60.0),
        Segment(3, "Now let's move to our next topic on neuronal pathways.", 60.0, 85.0),
        Segment(4, "Please remember to submit your assignment by Friday midnight on the portal.", 85.0, 110.0),
    ]

    proc = ContextWindowProcessor(window_size=1)
    win = proc.build_window(segs, 1, global_outline="Nervous System Overview")
    assert win["target_id"] == 1
    assert "touching a hot pan" in win["target_text"]
    assert "Welcome to operating systems" in win["prev_text"]
    assert "reflex action is defined as" in win["next_text"]
    assert win["global_outline"] == "Nervous System Overview"

    # Discourse Role Classification
    role_0 = DiscourseRoleClassifier.classify_role(segs[0].text, is_first_in_section=True)
    assert role_0 == DiscourseRole.NEW_TOPIC

    role_1 = DiscourseRoleClassifier.classify_role(segs[1].text)
    assert role_1 == DiscourseRole.EXAMPLE

    role_2 = DiscourseRoleClassifier.classify_role(segs[2].text)
    assert role_2 == DiscourseRole.DEFINITION

    role_3 = DiscourseRoleClassifier.classify_role(segs[3].text)
    assert role_3 == DiscourseRole.TRANSITION

    role_4 = DiscourseRoleClassifier.classify_role(segs[4].text)
    assert role_4 == DiscourseRole.ASSIGNMENT


# ---------------------------------------------------------------------------
# Test 16: Claim Entailment Validator
# ---------------------------------------------------------------------------
def test_16_claim_entailment_validation():
    evidence = (
        "Electric current is the rate of flow of electric charges through a conductor. "
        "It is measured in Amperes where one Ampere equals one Coulomb per second. "
        "Ohm's Law states that voltage equals current multiplied by resistance."
    )

    # 1. Grounded claim -> SUPPORTED
    status, score = ClaimEntailmentValidator.validate_claim(
        "Electric current measures the flow of electric charges in Amperes.",
        evidence
    )
    assert status == "SUPPORTED"
    assert score >= 0.50

    # 2. Fabricated claim with alien tokens -> UNSUPPORTED
    status_un, score_un = ClaimEntailmentValidator.validate_claim(
        "Mitochondrial oxidative phosphorylation generates quantum synaptic vesicles in prokaryotes.",
        evidence
    )
    assert status_un == "UNSUPPORTED"

    # 3. Filter grounded points
    points = [
        "Current is the flow of electric charges measured in Amperes.",
        "Quantum entanglement drives interstellar dark matter propulsion.",
        "Voltage is related to current and resistance through Ohm's law."
    ]
    filtered = ClaimEntailmentValidator.filter_grounded_points(points, evidence)
    assert len(filtered) == 2
    assert "Current is the flow of electric charges" in filtered[0]
    assert "Voltage is related to current" in filtered[1]


# ---------------------------------------------------------------------------
# Test 17: Hierarchical SubTopics, Evidence Tokens & Discourse Roles
# ---------------------------------------------------------------------------
def test_17_hierarchical_subtopics_and_evidence_graph():
    engine = TopicIntelligenceEngine(use_llm=False)
    data = [
        (0, 0.0, 25.0, "Electric charge is a fundamental property of matter measured in Coulombs."),
        (1, 25.0, 55.0, "Electric current is defined as the rate of flow of charge, where I = Q/t in Amperes."),
        (2, 55.0, 85.0, "For example in domestic household wiring free electrons drift along copper conductors."),
        (3, 85.0, 120.0, "Next we discuss Ohm's Law where potential difference V equals current I times resistance R."),
        (4, 120.0, 155.0, "Resistance R depends on wire length and cross-sectional area."),
        (5, 155.0, 180.0, "Remember this equation for the exam, it is crucial.")
    ]
    segs = [Segment(id=d[0], start=d[1], end=d[2], text=d[3]) for d in data]
    text = " ".join(s.text for s in segs)
    notes = engine.analyze_lecture_study_notes(text, segs, 180.0)

    # Verify Topic Hierarchy
    assert len(notes.topic_hierarchy) >= 2
    for node in notes.topic_hierarchy:
        assert node["level"] == 1
        assert "evidence" in node
        assert len(node["evidence"]) > 0
        assert all(ev.startswith("seg_") for ev in node["evidence"])

    # Verify SubTopics on StudyTopics
    assert len(notes.study_topics) >= 2
    for topic in notes.study_topics:
        assert len(topic.subtopics) >= 1
        assert topic.grounding_status == "SUPPORTED"
        assert len(topic.evidence) > 0
        assert topic.source_segment_ids == [int(e.replace("seg_", "")) for e in topic.evidence]
        for st in topic.subtopics:
            assert st.id.startswith(topic.id)
            assert st.role in [DiscourseRole.SUBTOPIC, DiscourseRole.DEFINITION, DiscourseRole.EXAMPLE, DiscourseRole.EXPLANATION]

    # Verify Discourse Roles
    assert len(notes.discourse_roles) == len(segs)
    assert notes.discourse_roles[0]["role"] == DiscourseRole.NEW_TOPIC
    assert notes.discourse_roles[2]["role"] == DiscourseRole.EXAMPLE


# ---------------------------------------------------------------------------
# Test 18: Notes Dictionary Serialization Backward Compatibility
# ---------------------------------------------------------------------------
def test_18_notes_serialization_and_backward_compatibility():
    engine = TopicIntelligenceEngine(use_llm=False)
    data = [
        (0, 0.0, 30.0, "Microeconomics covers supply and demand equilibrium."),
        (1, 30.0, 60.0, "Price elasticity of demand measures consumer responsiveness to price fluctuations."),
        (2, 60.0, 90.0, "When demand is inelastic, firms can increase revenue by raising prices.")
    ]
    segs = [Segment(id=d[0], start=d[1], end=d[2], text=d[3]) for d in data]
    text = " ".join(s.text for s in segs)
    notes = engine.analyze_lecture_study_notes(text, segs, 90.0)
    d = notes.to_dict()

    # Legacy fields
    assert "topics" in d and isinstance(d["topics"], list)
    assert "subtopics" in d and isinstance(d["subtopics"], list)
    assert "definitions" in d and isinstance(d["definitions"], list)
    assert "key_concepts" in d and isinstance(d["key_concepts"], list)
    assert "important_points" in d and isinstance(d["important_points"], list)
    assert "study_topics" in d and isinstance(d["study_topics"], list)

    # New Hierarchical & Grounding fields
    assert "topic_hierarchy" in d and isinstance(d["topic_hierarchy"], list)
    assert "discourse_roles" in d and isinstance(d["discourse_roles"], list)
    assert "quality_score" in d and isinstance(d["quality_score"], float)

    st0 = d["study_topics"][0]
    assert "evidence" in st0 and isinstance(st0["evidence"], list)
    assert "grounding_status" in st0
    assert "subtopics" in st0 and isinstance(st0["subtopics"], list)


# ---------------------------------------------------------------------------
# Test 19: Four Issue Pipeline Fixes Regression Test
# ---------------------------------------------------------------------------
def test_19_four_issue_pipeline_fixes():
    from app.services.topic_intelligence import _is_incomplete_clause

    # Issue 1: Cross-subject topic merging protection
    t_elec = StudyTopic(
        id="t1",
        title="Introduction to Electricity & Power Sources",
        explanation="Electric current is the rate of flow of charge (I = Q/t) through conductors.",
        key_points=["Electric Current (I = Q/t)", "Ohm's Law"],
        source_text="That is electricity and electric circuits. Electric power is utilized. Electric current is the flow of charge."
    )
    t_acct = StudyTopic(
        id="t2",
        title="Introduction to Accounting & Financial Liquidity",
        explanation="Profit is accrual based while cash flow tracks liquid bank funds.",
        key_points=["Net Cash Flow", "Accrual Principle"],
        source_text="Difference between profit and cash flow. Working capital and liquid bank funds accounts receivable."
    )
    engine = TopicIntelligenceEngine(use_llm=False)
    merged = engine._reconcile_and_merge_topics([t_elec, t_acct])
    assert len(merged) == 2, "Electricity and Accounting topics should NOT be merged together!"

    # Issue 2: Incomplete clause / opener title rejection
    assert _is_incomplete_clause("Good morning everyone, so today") is True
    assert _is_incomplete_clause("A company can sell a") is True
    assert _is_incomplete_clause("Therefore") is True
    assert _is_incomplete_clause("Electric Current & Charge Flow") is False
    assert _is_incomplete_clause("Ohm's Law Principles") is False

    # Issue 3: Embedding-outlier noise detection
    segs = [
        Segment(id=0, start=0.0, end=10.0, text="Today we discuss CPU process scheduling and ready queue management."),
        Segment(id=1, start=10.0, end=20.0, text="Processes in the ready queue wait for dispatcher execution."),
        Segment(id=2, start=20.0, end=25.0, text="Dursa if Pasaata Investing Gating"),  # Novel hallucination
        Segment(id=3, start=25.0, end=35.0, text="Turnaround time is completion time minus arrival time.")
    ]
    outliers = engine.boundary_analyzer.detect_embedding_outliers(segs)
    assert 2 in outliers, "Hallucination segment 2 should be flagged as an embedding outlier!"

    # Issue 4: Quality gate validation runs on Path B output
    notes = engine.analyze_lecture_study_notes(" ".join(s.text for s in segs if s.id != 2), [s for s in segs if s.id != 2], 35.0)
    assert notes.quality_score > 0.0
    for t in notes.study_topics:
        assert not _is_incomplete_clause(t.title)


def test_multistage_llm_understanding_synthesis_and_self_review():
    """
    Verify 4-stage student-understood notes generation:
      - Pass 1 whole-lecture outline & concept relationships
      - Pass 2 concept-by-concept synthesis with continuity context & exact numbers/formulas
      - Stage 4 self-review repairs sign flips, drops 1-word bullets, avoids boilerplate overwrites
    """
    from app.services.study_guide_generator import StudyGuideSynthesizer
    from app.services.pedagogical_validator import PreSubmissionQualityController

    prompts_seen = []

    class FakeQwenProvider:
        def is_available(self):
            return True

        def generate_structured(self, prompt, schema=None, system_prompt="", max_tokens=256, temperature=0.05):
            prompts_seen.append(prompt)
            if "timeline outline" in prompt.lower():
                return {
                    "subject": "Accounting & Finance",
                    "theme": "Profit vs Cash Flow",
                    "overview": "This lecture explains why earning accounting profit does not automatically mean having physical cash in the bank.",
                    "topics": [
                        {"section_index": 1, "title": "Accrual Profit Basics", "connection": "Introduces profit measured when sales happen, setting up the contrast with actual cash receipts."},
                        {"section_index": 2, "title": "Cash Flow & Credit Sales", "connection": "Explains why credit sales boost Accrual Profit before any cash enters the bank."},
                    ],
                }
            excerpt = prompt.split("Transcript Excerpt:")[-1].lower()
            if "credit" in excerpt:
                return {
                    "title": "Cash Flow vs Credit Sales",
                    "explanation": "Businesses often look profitable on paper while still running short on bank money when customers pay later.",
                    "what_it_means": "Cash inflow tracks real money entering your bank account, unlike credit revenue recorded on invoices.",
                    "how_it_works": "First a store delivers goods on credit and books revenue right away, then waits weeks for actual customer cash to arrive.",
                    "why_it_matters": "A profitable company can miss rent or payroll if customers take too long to pay.",
                    "key_points": [
                        "Liquidity",  # 1-word label -> Stage 4 must drop it!
                        "Credit sales increase reported profit immediately without adding cash to the bank.",
                        "Net cash flow compares physical cash inflows against cash outflows.",
                    ],
                    "definitions": [{"term": "Accounts Receivable", "definition": "Unpaid money customers owe for delivered credit sales."}],
                    "examples": ["Selling 2 lakh rupees on 60-day credit yields profit on paper but zero cash today."],
                    "formulas": ["Net Cash Flow = Cash Inflows - Cash Outflows"],
                    "exam_relevant": True,
                }
            return {
                "title": "Accrual Profit Calculation",
                "explanation": "Accounting profit shows whether sales exceed costs during a period, calculated by subtracting revenue from cost.",
                "what_it_means": "Revenue is money earned from delivering goods, and cost is what it took to produce them.",
                "how_it_works": "First add total earned sales such as 2 lakh rupees, then deduct expenses like 1 lakh 20,000 rupees to leave 80,000 rupees profit.",
                "why_it_matters": "Separating earned revenue from expense timing stops owners from confusing paper profit with bank balance.",
                "key_points": [
                    "Profit equals earned revenue minus total expenses rather than cash collected.",
                    "With 2 lakh revenue and 1 lakh 20,000 cost, profit equals 80,000 rupees.",
                ],
                "definitions": [{"term": "Profit", "definition": "Earned revenue minus total incurred expenses."}],
                "examples": ["Clothing store earning 2 lakh revenue against 1 lakh 20,000 cost makes 80,000 profit."],
                "formulas": ["Profit = Revenue - Expenses"],
                "exam_relevant": True,
            }

    segs = [
        Segment(id=0, start=0.0, end=25.0, text="Today we compare accounting profit and cash flow. Profit equals revenue minus expenses."),
        Segment(id=1, start=25.0, end=55.0, text="Suppose your online clothing revenue is 2 lakh rupees and cost is 1 lakh 20,000 rupees, so profit is 80,000 rupees."),
        Segment(id=2, start=55.0, end=90.0, text="Now suppose sales are made on credit, so cash inflows are low and accounts receivable rise while bank cash stays negative."),
        Segment(id=3, start=90.0, end=125.0, text="Net cash flow equals cash inflows minus cash outflows, showing why a profitable firm can lack bank cash for rent."),
    ]
    full_txt = " ".join(s.text for s in segs)

    engine = TopicIntelligenceEngine(use_llm=True)
    engine.llm = FakeQwenProvider()
    notes = engine.analyze_lecture_study_notes(full_txt, segs, 125.0, topic_hint="Accounting and Finance")

    assert notes.analyzer_used == "local_llm"
    assert len(prompts_seen) >= 2
    assert any("Concept Flow:" in p for p in prompts_seen[1:])
    assert len(notes.concept_relationships) >= 1

    all_expl = " ".join(t.explanation for t in notes.study_topics)
    assert "subtracting revenue from cost" not in all_expl.lower()
    assert "subtracting cost from revenue" in all_expl.lower()
    for t in notes.study_topics:
        assert all(len(kp.split()) >= 5 for kp in t.key_points)

    as_dict = notes.to_dict()
    enriched = StudyGuideSynthesizer.enrich_study_topics(as_dict["study_topics"], full_txt, topic_hint="Accounting")
    as_dict["study_topics"] = enriched
    healed = PreSubmissionQualityController.audit_and_heal_notes(as_dict, transcript_text=full_txt)
    for ht in healed["study_topics"]:
        assert "Critical for examination assessment" not in ht.get("why_it_matters", "")
        assert not ht["explanation"].lower().startswith("this section explores")
        assert ht.get("what_it_means")
        assert ht.get("how_it_works")
        assert ht.get("steps_and_logic")

    guide = StudyGuideSynthesizer.generate_master_study_guide(
        transcript_text=full_txt,
        study_topics=healed["study_topics"],
        lecture_overview=notes.lecture_overview,
        concept_relationships=notes.concept_relationships,
    )
    assert "## 1. The Main Idea" in guide
    assert "**Why It Matters**:" in guide
    assert "**How Concepts Connect**:" in guide


