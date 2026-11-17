"""
Tests and multi-domain quality evaluation for the redesigned Exam-Grade Quiz Engine
(StudyGuideSynthesizer + PedagogicalValidator + IntelligenceService).
"""

import json
import pytest

from app.services.study_guide_generator import StudyGuideSynthesizer
from app.services.pedagogical_validator import PedagogicalValidator
from app.services.intelligence_service import IntelligenceService


# ---------------------------------------------------------------------------
# Mock Provider simulating Qwen Blueprint + Batched Exam MCQ Generation
# ---------------------------------------------------------------------------

class _MockExamLLMProvider:
    def __init__(self, blueprint_payload: dict, batches_payload: list[dict], fail_first_batch: bool = False):
        self.blueprint_payload = blueprint_payload
        self.batches_payload = list(batches_payload)
        self.fail_first_batch = fail_first_batch
        self.calls: list[dict] = []
        self._batch_idx = 0

    def is_available(self) -> bool:
        return True

    def generate_structured(self, prompt: str, schema=None, system_prompt="", max_tokens=512, temperature=0.2):
        self.calls.append({"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature})
        if "examinable_items" in (schema or {}) or "examinable_items" in prompt:
            return self.blueprint_payload
        if self.fail_first_batch:
            self.fail_first_batch = False
            raise RuntimeError("Simulated transient JSON parse failure")
        if self._batch_idx < len(self.batches_payload):
            out = self.batches_payload[self._batch_idx]
            self._batch_idx += 1
            return out
        return {"quizzes": []}


class _FailingLLMProvider:
    def is_available(self) -> bool:
        return True

    def generate_structured(self, *args, **kwargs):
        raise RuntimeError("Model inference failed")


# ---------------------------------------------------------------------------
# 1. Adaptive Question Count & Full-Lecture Chronological Coverage
# ---------------------------------------------------------------------------

def test_adaptive_quiz_target_plan_scales_with_length():
    short_text = "word " * 400
    short_plan = StudyGuideSynthesizer._compute_quiz_target_plan(
        short_text, {"study_topics": [{"title": f"T{i}"} for i in range(3)]}, segments=[{"end": 360}]
    )
    assert short_plan["tier"] == "short"
    assert short_plan["min_questions"] >= 5

    med_text = "word " * 2000
    med_plan = StudyGuideSynthesizer._compute_quiz_target_plan(
        med_text, {"study_topics": [{"title": f"T{i}"} for i in range(6)]}, segments=[{"end": 1200}]
    )
    assert med_plan["tier"] == "medium"
    assert med_plan["min_questions"] >= 10

    long_text = "word " * 5500
    long_plan = StudyGuideSynthesizer._compute_quiz_target_plan(
        long_text, {"study_topics": [{"title": f"T{i}"} for i in range(10)]}, segments=[{"end": 2700}]
    )
    assert long_plan["tier"] == "long"
    assert 15 <= long_plan["target_questions"] <= 25


def test_chronological_digest_covers_early_middle_and_late_sections():
    early = "EARLY_MARKER " + ("alpha probability distribution variance " * 120)
    middle = ("bayes theorem conditional likelihood " * 60) + " MIDDLE_MARKER " + ("bayes theorem conditional likelihood " * 60)
    late = ("central limit theorem confidence interval " * 115) + " LATE_MARKER"
    full_text = early + "\n" + middle + "\n" + late

    digest, windows = StudyGuideSynthesizer._build_chronological_digest(full_text, [])
    assert "EARLY_MARKER" in digest
    assert "MIDDLE_MARKER" in digest
    assert "LATE_MARKER" in digest
    assert "EARLY_MARKER" in windows["early"]
    assert "MIDDLE_MARKER" in windows["middle"]
    assert "LATE_MARKER" in windows["late"]


# ---------------------------------------------------------------------------
# 2. Distractor Quality, Anti-Pattern Rejection & Answer Balancing
# ---------------------------------------------------------------------------

def test_validate_and_refine_mcq_rejects_banned_patterns_and_repetition():
    import random
    rng = random.Random(123)
    transcript_low = "bayes theorem conditional probability prior likelihood posterior variance"
    seen_stems: list[str] = []
    seen_concepts: set[str] = set()

    # Reject "All of the above"
    q_banned = {
        "question": "Which property holds for conditional probability in Bayes theorem?",
        "options": ["P(A|B) scales with P(B|A)", "P(A|B) = P(A) + P(B)", "P(A|B) ignores prior", "All of the above"],
        "correct_answer": "P(A|B) scales with P(B|A)",
        "explanation": "Bayes theorem relates posterior to likelihood and prior.",
        "grounded_concept": "Bayes theorem",
    }
    res, err = StudyGuideSynthesizer._validate_and_refine_mcq(q_banned, transcript_low, seen_stems, seen_concepts, rng)
    assert res is None and "Banned" in err

    # Reject recall phrasing
    q_recall = {
        "question": "According to the lecture, what is Bayes theorem used for?",
        "options": ["Updating beliefs with evidence", "Sorting arrays", "Paging memory", "Edge detection"],
        "correct_answer": "Updating beliefs with evidence",
        "explanation": "It updates prior probability using likelihood.",
        "grounded_concept": "Bayes theorem",
    }
    res, err = StudyGuideSynthesizer._validate_and_refine_mcq(q_recall, transcript_low, seen_stems, seen_concepts, rng)
    assert res is None and "Recall" in err

    # Accept valid exam-level question
    q_good = {
        "category": "Conceptual",
        "difficulty": "Hard",
        "topic_area": "Bayesian Inference",
        "question": "If the prior probability of a rare disease is very low, why can a positive test with 95% sensitivity still yield a low posterior probability?",
        "options": [
            "False positives from the large healthy population outnumber true positives",
            "Sensitivity directly equals the posterior probability regardless of base rate",
            "The likelihood ratio becomes negative when prevalence drops below 5%",
            "Conditional independence forces the prior to cancel out of the denominator",
        ],
        "correct_answer": "False positives from the large healthy population outnumber true positives",
        "explanation": "By Bayes theorem, P(D|+) depends on P(D). Low prevalence means even a small false-positive rate produces many false alarms.",
        "distractor_rationales": {"Sensitivity directly equals...": "Confuses P(+|D) with P(D|+)."},
        "grounded_concept": "Bayes theorem",
    }
    res, err = StudyGuideSynthesizer._validate_and_refine_mcq(q_good, transcript_low, seen_stems, seen_concepts, rng)
    assert res is not None and err is None
    assert res["correct_answer"] in res["options"]
    assert res["options"][["A", "B", "C", "D"].index(res["correct_option_letter"])] == res["correct_answer"]

    # Reject near-duplicate stem
    res_dup, err_dup = StudyGuideSynthesizer._validate_and_refine_mcq(q_good, transcript_low, seen_stems, seen_concepts, rng)
    assert res_dup is None and "Duplicate" in err_dup


def test_pedagogical_validator_rejects_catchall_and_duplicate_options():
    transcript = "operating systems cpu scheduling round robin turnaround time"
    bad_quizzes = [
        {
            "question": "When evaluating Round Robin CPU scheduling, what happens as the time quantum approaches infinity?",
            "options": ["It behaves like FCFS", "It causes starvation", "Context switches double", "All of the above"],
            "correct_answer": "It behaves like FCFS",
        }
    ]
    res = PedagogicalValidator.validate_quizzes(bad_quizzes, transcript, bloom_policy="quiz")
    assert not res["passed"]
    assert any("Banned" in r for r in res["rejected"][0]["reasons"])


# ---------------------------------------------------------------------------
# 3. No Silent Extractive Fallback When LLM Fails in Strict Mode
# ---------------------------------------------------------------------------

def test_no_silent_fallback_when_llm_fails_in_strict_mode():
    status = {}
    quizzes = StudyGuideSynthesizer.generate_conceptual_quizzes(
        transcript_text="This lecture covers hypothesis testing, p-values, confidence intervals, and type I errors in detail.",
        topic_hint="Statistics",
        lecture_notes={"study_topics": [{"title": "Hypothesis Testing"}], "definitions": [{"term": "P-Value", "definition": "Probability under H0"}]},
        llm_provider=_FailingLLMProvider(),
        allow_extractive_fallback=False,
        status_out=status,
    )
    assert quizzes == []
    assert status["status"] == "failed"


def test_intelligence_service_reports_failure_instead_of_extractive_templates(monkeypatch):
    monkeypatch.setattr(
        StudyGuideSynthesizer,
        "generate_conceptual_quizzes",
        staticmethod(lambda *a, **kw: (kw["status_out"].update({"status": "failed", "reason": "LLM disabled"}) or [])),
    )
    notes = {"study_topics": [{"title": "Scheduling"}], "definitions": [{"term": "Deadlock", "definition": "Circular wait"}]}
    quizzes = IntelligenceService.generate_quizzes(
        lecture_notes=notes,
        assignments=[],
        segments=[],
        transcript_text="CPU scheduling and deadlock prevention in operating systems.",
    )
    assert quizzes == []
    assert notes["meta"]["quiz_generation"]["status"] == "failed"


# ---------------------------------------------------------------------------
# 4. Multi-Domain End-to-End Evaluation (Stats, OS, CV, Unfamiliar Subject)
# ---------------------------------------------------------------------------

def _make_distinct_batch(subject: str, concepts: list[tuple[str, str, str, str, str | None]]) -> dict:
    """Builds a batch of distinct exam MCQs from (category, difficulty, topic, stem, calc)."""
    quizzes = []
    for cat, diff, topic, stem, calc in concepts:
        correct = f"Primary mechanism governing {topic} in {subject}"
        quizzes.append({
            "category": cat,
            "difficulty": diff,
            "topic_area": topic,
            "question": stem,
            "options": [
                correct,
                f"Common misconception swapping cause and effect in {topic}",
                f"Boundary assumption that invalidates {topic} under constraints",
                f"Competing sibling model that omits {topic} parameters",
            ],
            "correct_answer": correct,
            "correct_option_letter": "A",
            "explanation": f"{correct} explains the observed behavior in {topic}, whereas the other options confuse boundary or causal assumptions.",
            "calculation": calc,
            "grounded_concept": topic,
        })
    return {"quizzes": quizzes}


@pytest.mark.parametrize("subject,transcript,concepts", [
    (
        "Statistics",
        ("Normal distribution mean standard deviation z-score p-value null hypothesis type I error central limit theorem confidence interval variance " * 40),
        [
            ("Numerical", "Medium", "Z-Score Standardization", "With mean 50 and standard deviation 10, what z-score corresponds to an observation of 75?", "z = (75 - 50) / 10 = 2.5"),
            ("Conceptual", "Hard", "Central Limit Theorem", "Why does the sampling distribution of the sample mean approach normality even for skewed populations when n is large?", None),
            ("Application", "Medium", "P-Value Interpretation", "When a two-tailed hypothesis test yields p = 0.03 at alpha = 0.05, what conclusion is statistically valid?", None),
            ("Numerical", "Hard", "Standard Error Scaling", "If sample size increases from 25 to 100 while population variance remains fixed, how does the standard error change?", "SE_new = sigma / sqrt(100) = 0.5 * SE_old"),
            ("Conceptual", "Medium", "Type I vs Type II Error", "Lowering the significance threshold alpha from 0.05 to 0.01 has what direct trade-off on statistical power?", None),
            ("Application", "Hard", "Confidence Interval Width", "A researcher switches from a 95% to a 99% confidence interval on the same sample; how does the margin of error respond?", None),
            ("Numerical", "Medium", "Conditional Probability", "Given P(A) = 0.4, P(B|A) = 0.5, and P(B) = 0.25, what is P(A|B) via Bayes rule?", "P(A|B) = (0.5 * 0.4) / 0.25 = 0.80"),
            ("Conceptual", "Easy", "Variance vs Standard Deviation", "Why is standard deviation often preferred over variance when reporting spread alongside the sample mean?", None),
            ("Application", "Hard", "Independent Events", "If two events A and B are mutually exclusive with non-zero probabilities, why can they never be independent?", None),
            ("Technical", "Medium", "Degrees of Freedom", "Why does sample variance divide by n - 1 rather than n when estimating population variance?", None),
        ],
    ),
    (
        "Operating Systems",
        ("Round robin scheduling time quantum context switch deadlock banker algorithm virtual memory page fault thrashing semaphore mutex " * 40),
        [
            ("Technical", "Medium", "Round Robin Quantum", "If the Round Robin time quantum becomes much larger than typical CPU bursts, which scheduling behavior emerges?", None),
            ("Application", "Hard", "Thrashing Collapse", "During heavy multiprogramming, when total working-set demand exceeds physical frames, why does CPU utilization plummet?", None),
            ("Conceptual", "Medium", "Deadlock Coffman Conditions", "Breaking circular wait by imposing a total ordering on resource acquisition prevents deadlock by violating which requirement?", None),
            ("Technical", "Hard", "Banker Safety Algorithm", "Before granting a resource request, what must the Banker algorithm verify to guarantee the system stays in a safe state?", None),
            ("Application", "Medium", "TLB Hit Ratio", "How does a Translation Lookaside Buffer reduce effective memory access time during virtual-to-physical address translation?", None),
            ("Conceptual", "Easy", "Process vs Thread", "Why is switching between threads of the same process cheaper than switching between separate processes?", None),
            ("Technical", "Hard", "Priority Inversion", "In real-time scheduling, how does priority inheritance resolve unbounded priority inversion on a shared mutex?", None),
            ("Application", "Medium", "Copy-on-Write Fork", "After fork(), when does the kernel actually duplicate physical memory pages between parent and child processes?", None),
            ("Conceptual", "Medium", "Internal vs External Fragmentation", "Why does fixed-size paging eliminate external fragmentation while still allowing internal fragmentation?", None),
            ("Technical", "Hard", "Page Replacement Belady", "Why can FIFO page replacement suffer Belady's anomaly when extra physical frames are allocated, unlike LRU?", None),
        ],
    ),
    (
        "Computer Vision",
        ("Gaussian blur Sobel edge detection convolution kernel stride padding Canny hysteresis thresholding SIFT keypoint homography " * 40),
        [
            ("Application", "Medium", "Gaussian Pre-Smoothing", "Why is Gaussian smoothing applied before Sobel differentiation in noisy edge-detection pipelines?", None),
            ("Technical", "Hard", "Canny Hysteresis", "How does Canny hysteresis thresholding use dual high/low thresholds to preserve weak true edges without keeping noise?", None),
            ("Numerical", "Medium", "Convolution Output Size", "For a 32x32 feature map, 5x5 kernel, stride 1, and padding 0, what is the spatial output dimension?", "(32 - 5) / 1 + 1 = 28x28"),
            ("Conceptual", "Medium", "Separable Filters", "Why does decomposing a 2D Gaussian kernel into two 1D convolutions reduce computational complexity?", None),
            ("Technical", "Hard", "Non-Maximum Suppression", "During edge thinning, what criterion determines whether a gradient magnitude pixel survives non-maximum suppression?", None),
            ("Application", "Hard", "SIFT Scale Invariance", "How does Difference-of-Gaussians across octave pyramids enable SIFT keypoints to remain invariant to image scale?", None),
            ("Conceptual", "Easy", "Padding Borders", "What artifact occurs at image boundaries when convolution is run without zero or reflection padding?", None),
            ("Technical", "Medium", "RANSAC Homography", "Why is RANSAC paired with feature matching instead of ordinary least squares when estimating a planar homography?", None),
            ("Application", "Medium", "Histogram Equalization", "Under what lighting condition does global histogram equalization degrade local contrast in bright regions?", None),
            ("Conceptual", "Hard", "Harris Corner Response", "How do the eigenvalues of the second-moment matrix distinguish a corner from a flat region or straight edge?", None),
        ],
    ),
    (
        "Microbial Bioelectrochemistry",
        ("Geobacter sulfurreducens extracellular electron transfer cytochrome nanowire anode biofilm overpotential coulombic efficiency mediator " * 40),
        [
            ("Conceptual", "Hard", "Biofilm Diffusion Limit", "Why does excessive thickening of an inactive outer biofilm layer reduce coulombic efficiency at the anode?", None),
            ("Technical", "Medium", "Direct Electron Transfer", "How do outer-membrane c-type cytochromes enable Geobacter to respire onto a solid electrode without soluble mediators?", None),
            ("Application", "Hard", "Activation Overpotential", "When operating a microbial fuel cell at low current density, which loss dominates the drop below theoretical open-circuit voltage?", None),
            ("Conceptual", "Medium", "Syntrophic Methanogenesis", "Why does methanogen competition in an unbuffered anode chamber lower coulombic efficiency?", None),
            ("Technical", "Hard", "Conductive Pili Nanowires", "What role do electrically conductive pili play for cells located several micrometers away from the electrode surface?", None),
            ("Application", "Medium", "Redox Mediator Shuttle", "Under what condition does adding an exogenous flavin shuttle improve power density compared to direct contact alone?", None),
            ("Conceptual", "Easy", "Anode vs Cathode", "In a microbial electrolysis cell, why must external voltage be supplied to evolve hydrogen at the cathode?", None),
            ("Technical", "Hard", "Proton Mass Transport", "How does sluggish proton migration across an ion-exchange membrane create localized pH splitting between chambers?", None),
            ("Application", "Medium", "Poised Potential Chronoamperometry", "Why do researchers use a three-electrode potentiostat setup rather than a two-electrode cell to isolate anode kinetics?", None),
            ("Conceptual", "Hard", "Substrate Saturation Monod", "Above saturating acetate concentrations, what limits further increases in steady-state current density?", None),
        ],
    ),
], ids=["Statistics", "OperatingSystems", "ComputerVision", "Bioelectrochemistry"])
def test_multi_domain_exam_quiz_generation(subject, transcript, concepts):
    items = [
        {
            "concept": topic,
            "section": ["early", "middle", "late"][i % 3],
            "topic_area": topic,
            "category": cat,
            "difficulty": diff,
            "angle": stem[:60],
        }
        for i, (cat, diff, topic, stem, _calc) in enumerate(concepts)
    ]
    batches = [
        _make_distinct_batch(subject, concepts[0:3]),
        _make_distinct_batch(subject, concepts[3:6]),
        _make_distinct_batch(subject, concepts[6:9]),
        _make_distinct_batch(subject, concepts[9:10]),
    ]
    blueprint = {
        "subject_domain": subject,
        "lecture_style": "mixed",
        "has_numerical_content": any(c[0] == "Numerical" for c in concepts),
        "examinable_items": items,
    }
    provider = _MockExamLLMProvider(blueprint, batches, fail_first_batch=True)
    status = {}
    quizzes = StudyGuideSynthesizer.generate_conceptual_quizzes(
        transcript_text=transcript,
        topic_hint=subject,
        lecture_notes={"study_topics": [{"title": f"{subject} Topic {i}"} for i in range(6)]},
        segments=[{"end": 1200}],
        llm_provider=provider,
        allow_extractive_fallback=False,
        status_out=status,
    )
    assert status["status"] == "ok"
    assert len(quizzes) >= 10
    letters = {q["correct_option_letter"] for q in quizzes}
    assert len(letters) >= 2, "Answer positions should be randomized across A-D"
    for q in quizzes:
        assert len(q["options"]) == 4
        assert q["correct_answer"] in q["options"]
        assert q["difficulty"] in ("Easy", "Medium", "Hard")
        assert q["category"] in ("Conceptual", "Application", "Numerical", "Technical")
        assert len(q["explanation"]) > 15
