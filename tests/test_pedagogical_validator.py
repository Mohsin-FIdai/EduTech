"""
tests/test_pedagogical_validator.py — Unit and integration tests for PedagogicalValidator,
CEM note hierarchy validation, Bloom's quiz auditing, and RAG grounding verification.
"""

import pytest
from app.services.pedagogical_validator import PedagogicalValidator
from app.services.topic_intelligence import _clean_transcript_for_notes
from app.services.rag_service import LectureRAGService
from app.config import RAG_MIN_TOKEN_OVERLAP


def test_validate_study_notes_detects_cem_structure():
    """Verifies that PedagogicalValidator detects CEM headings and bullet format."""
    valid_notes = """# Master Study Guide

## 1. The Main Idea
- This lecture examines Supervised Learning, focusing on Classification and Regression.

## 2. Core Concepts Overview
### 2.1 Classification
- **Definition**: Supervised learning task predicting discrete categories.
- **Mechanism**: Maps feature vectors to discrete target labels using boundary optimization.
- **Example**: Spam email detection.

## 3. Key Definitions
- **Classification**: Supervised learning task predicting discrete categories.
- **Precision**: Ratio of true positives to total predicted positives.

## 4. Governing Formulas
- `Precision = TP / (TP + FP)`

## 5. Worked Example(s)
1. **Classification**: Spam filter sorting incoming email.
"""
    transcript = "We discuss Supervised Learning, classification, and regression. Classification maps feature vectors to discrete target labels. Precision is true positives over predicted positives."

    res = PedagogicalValidator.validate_study_notes(valid_notes, transcript)
    assert res["passed"] is True
    assert res["score"] >= 0.8
    assert len(res["issues"]) == 0


def test_validate_study_notes_flags_missing_sections():
    """Verifies that PedagogicalValidator flags missing CEM sections."""
    poor_notes = "Some random text without any headings or definitions."
    transcript = "We discuss Supervised Learning and classification."

    res = PedagogicalValidator.validate_study_notes(poor_notes, transcript)
    assert res["passed"] is False
    assert any("Core Concepts" in issue or "Main Idea" in issue for issue in res["issues"])
    assert any("Definitions" in issue for issue in res["issues"])


def test_validate_quizzes_rejects_recall_phrasing():
    """Verifies that PedagogicalValidator rejects questions that test verbatim memory."""
    recall_quizzes = [
        {
            "category": "Recall",
            "question": "What did the speaker say about gradient descent in the lecture?",
            "options": ["It minimizes loss", "It diverges", "It stops", "It repeats"],
            "correct_answer": "It minimizes loss",
            "correct_option_letter": "A",
            "explanation": "The speaker mentioned it.",
        },
        {
            "category": "Recall",
            "question": "According to this session, what is Precision?",
            "options": ["TP / (TP + FP)", "TN / (TN + FP)", "TP / FN", "TN / FP"],
            "correct_answer": "TP / (TP + FP)",
            "correct_option_letter": "A",
            "explanation": "Defined in session.",
        }
    ]
    transcript = "Gradient descent minimizes loss. Precision is TP over TP plus FP."

    res = PedagogicalValidator.validate_quizzes(recall_quizzes, transcript)
    assert res["passed"] is False
    assert len(res["rejected"]) == 2
    assert any("Recall phrasing" in r["reasons"][0] for r in res["rejected"])


def test_validate_quizzes_accepts_scenario_phrasing():
    """Verifies that PedagogicalValidator passes application-level questions."""
    application_quizzes = [
        {
            "category": "Application",
            "question": "A practitioner applying Precision in a real-world diagnostic scenario needs to evaluate which metric?",
            "options": [
                "Ratio of true positive predictions to total predicted positives",
                "Ratio of true negatives to all negatives",
                "Total elapsed calculation time across iterations",
                "Unweighted harmonic mean of training loss residuals",
            ],
            "correct_answer": "Ratio of true positive predictions to total predicted positives",
            "correct_option_letter": "A",
            "explanation": "Precision evaluates prediction exactness.",
            "grounded_concept": "Precision",
        },
        {
            "category": "Analysis",
            "question": "A practitioner working on Classification needs to evaluate system behavior. Which principle should guide their analysis?",
            "options": [
                "Models map feature vectors to discrete target categories using decision boundaries",
                "Continuous numerical outputs are estimated via polynomial regression",
                "Parameters must be adjusted without empirical loss verification",
                "System throughput is prioritized by disabling test validation splits",
            ],
            "correct_answer": "Models map feature vectors to discrete target categories using decision boundaries",
            "correct_option_letter": "A",
            "explanation": "Classification predicts discrete categories.",
            "grounded_concept": "Classification",
        }
    ]
    transcript = "In this session we discuss Classification and Precision. Classification predicts discrete categories using labeled data. Precision is the ratio of true positive predictions to total predicted positives."

    res = PedagogicalValidator.validate_quizzes(application_quizzes, transcript)
    assert res["passed"] is True
    assert res["valid_count"] == 2
    assert len(res["rejected"]) == 0


def test_validate_quizzes_rejects_alien_concepts():
    """Verifies that PedagogicalValidator rejects quizzes referencing terms absent from transcript."""
    alien_quizzes = [
        {
            "category": "Application",
            "question": "A practitioner designing a Deep Learning neural network architecture should use which optimizer?",
            "options": ["SGD", "Adam", "RMSProp", "Adagrad"],
            "correct_answer": "SGD",
            "correct_option_letter": "A",
            "explanation": "Deep learning models train with backpropagation.",
        }
    ]
    transcript = "This lecture covers basic decision trees and linear regression."

    res = PedagogicalValidator.validate_quizzes(alien_quizzes, transcript)
    assert res["passed"] is False
    assert any("Alien concept" in r["reasons"][0] for r in res["rejected"])


def test_verify_grounding_passes_high_overlap():
    """Verifies that high-overlap answers pass grounding check."""
    context = "Profit equals revenue minus expenditure. For March, revenue was Rs. 500,000 and expenditure was Rs. 300,000."
    answer = "Profit is calculated as revenue minus expenditure. Your profit is Rs. 200,000."

    assert PedagogicalValidator.verify_grounding(answer, context, min_overlap=0.70) is True


def test_verify_grounding_fails_low_overlap():
    """Verifies that ungrounded / hallucinated answers fail grounding check."""
    context = "We discussed First-Come First-Served process scheduling and the convoy effect."
    hallucinated_answer = "Quantum entanglement wormholes allow superluminal communication between superconducting qubits."

    assert PedagogicalValidator.verify_grounding(hallucinated_answer, context, min_overlap=0.80) is False


def test_verify_grounding_uses_config_threshold():
    """Verifies that verify_grounding respects default RAG_MIN_TOKEN_OVERLAP from config."""
    context = "A decision tree recursively splits feature space."
    answer = "Decision trees split space recursively."
    # Both tokens are in context -> 100% overlap
    assert PedagogicalValidator.verify_grounding(answer, context) is True


def test_disfluency_cleaner_removes_fillers_and_duplicates():
    """Verifies that _clean_transcript_for_notes strips fillers and repeated words without deep pronoun rewriting."""
    raw = "Um, so basically acha theek hai, we have the the decision tree which is like really useful."
    cleaned = _clean_transcript_for_notes(raw)

    assert "um" not in cleaned.lower().split()
    assert "the the" not in cleaned
    assert "theek hai" not in cleaned.lower()
    assert "acha" not in cleaned.lower().split()
    assert "decision tree" in cleaned


def test_rag_service_verify_grounding_integration():
    """Verifies that LectureRAGService._verify_grounding correctly delegates to PedagogicalValidator."""
    context = "Turnaround Time equals completion time minus arrival time."
    answer = "Turnaround Time is completion time minus arrival time."

    assert LectureRAGService._verify_grounding(answer, context) is True
    assert LectureRAGService._verify_grounding("Quantum entanglement", context) is False


def test_pre_submission_quality_controller_heals_all_edge_cases():
    """
    Verifies that PreSubmissionQualityController eliminates:
      1. Fake or currency titles ('Lakh Rupees', 'Translate Please Provide')
      2. Awkward template prefixes ('This section introduces X, explaining how for instance...')
      3. Action verb titles ('Earn Profit' -> 'Profit Generation')
      4. Duplicated definitions/examples across topic cards
      5. Guarantees executive summary, learning objectives, revision summary, and chapters
    """
    from app.services.pedagogical_validator import PreSubmissionQualityController

    raw_notes = {
        "study_topics": [
            {
                "title": "Cash Inflows",
                "explanation": "This section introduces Cash Inflows, explaining how for instance, a manufacturing company collects revenue from clients.",
                "key_points": [
                    "This section introduces Cash Inflows, explaining how accounts receivable are converted to cash.",
                    "Cash inflows represent actual liquid currency received by the firm.",
                ],
                "definitions": [
                    {"term": "Cash Inflows", "definition": "Physical liquid currency received by a company."}
                ],
                "formulas": ["Net Cash Flow = Cash Inflows - Cash Outflows"],
                "examples": ["A firm receiving payment 30 days after delivering goods."],
                "start_time": 0.0,
                "end_time": 90.0,
            },
            {
                "title": "Lakh Rupees",
                "explanation": "A company made five lakh rupees in revenue during March.",
                "key_points": ["Two lakh rupees were on credit."],
                "start_time": 90.0,
                "end_time": 150.0,
            },
            {
                "title": "Translate Please Provide",
                "explanation": "Your message is incomplete, please provide a complete sentence for translation.",
                "key_points": ["Incomplete sentence."],
                "start_time": 150.0,
                "end_time": 180.0,
            },
            {
                "title": "Earn Profit",
                "explanation": "Profit calculation involves subtracting expenses from total earned income.",
                "key_points": [
                    "Operating revenue minus operating expenses yields net accounting profit.",
                    "Accounting profit differs fundamentally from immediate liquid cash flow.",
                ],
                "definitions": [
                    {"term": "Cash Inflows", "definition": "Duplicate definition that should be pruned."},
                    {"term": "Accounting Profit", "definition": "Total revenue minus expenses."}
                ],
                "formulas": [
                    "Net Cash Flow = Cash Inflows - Cash Outflows",
                    "Net Profit = Revenue - Expenses"
                ],
                "examples": ["A firm receiving payment 30 days after delivering goods."],
                "start_time": 180.0,
                "end_time": 260.0,
            },
            {
                "title": "Through Each Component",
                "explanation": "If any component in a simple series circuit breaks, the circuit interrupts.",
                "key_points": ["Broken component stops current."],
                "start_time": 260.0,
                "end_time": 300.0,
            },
            {
                "title": "Reason Why Profits",
                "explanation": "Revenue recognition occurs when goods are delivered on credit.",
                "key_points": ["Revenue recognition principles apply."],
                "start_time": 300.0,
                "end_time": 360.0,
            }
        ],
        "key_concepts": [
            "Whereas Response",
            "Profit equals our income minus expenses",
            "Net Profit",
            "Lakh Rupees",
            "Working Capital"
        ]
    }

    healed = PreSubmissionQualityController.audit_and_heal_notes(raw_notes)

    # 1. Fake/currency topics must be pruned
    titles = [t["title"] for t in healed["study_topics"]]
    assert "Lakh Rupees" not in titles
    assert "Translate Please Provide" not in titles
    assert not any("please provide" in t.lower() for t in titles)

    # 2. Action verb converted and clause fragments healed
    assert "Earn Profit" not in titles
    assert "Profit Generation" in titles
    assert "Through Each Component" not in titles
    assert "Series Circuit Continuity" in titles
    assert "Reason Why Profits" not in titles
    assert "Revenue Recognition Principles" in titles

    # 3. Awkward prefixes eliminated
    cash_inflows = next(t for t in healed["study_topics"] if t["title"] == "Cash Inflows")
    assert "explaining how for instance" not in cash_inflows["explanation"].lower()
    assert "This section introduces" not in cash_inflows["explanation"]

    # 4. Cross-topic deduplication
    profit_topic = next(t for t in healed["study_topics"] if t["title"] == "Profit Generation")
    # Duplicate 'Cash Inflows' definition pruned from profit_topic
    assert len(profit_topic["definitions"]) == 1
    assert profit_topic["definitions"][0]["term"] == "Accounting Profit"
    # Duplicate formula pruned from profit_topic
    assert len(profit_topic["formulas"]) == 1
    assert "Net Profit" in profit_topic["formulas"][0]

    # 5. Key concepts sanitized
    clean_kcs = healed.get("key_concepts", [])
    assert "Whereas Response" not in clean_kcs
    assert "Lakh Rupees" not in clean_kcs
    assert "Net Profit" in clean_kcs
    assert "Working Capital" in clean_kcs

    # 6. CEM fields distinct and populated
    assert cash_inflows["what_it_means"] != cash_inflows["how_it_works"]
    assert len(cash_inflows["steps_and_logic"]) > 0

    # 7. Pedagogical summaries and chapters
    assert healed.get("executive_summary")
    assert len(healed.get("learning_objectives", [])) >= 2
    assert len(healed.get("revision_summary", [])) >= 2
    assert len(healed.get("chapters", [])) == len(healed["study_topics"])
    assert healed["universal_notes_rev"] == "2026-11-10-v3"

