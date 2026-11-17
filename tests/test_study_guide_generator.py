"""
test_study_guide_generator.py — Regression and unit tests for dynamic master study guide synthesis.
"""
from unittest.mock import MagicMock
from types import SimpleNamespace

from app.services.study_guide_generator import StudyGuideSynthesizer


def test_two_lectures_same_domain_produce_different_guides():
    """
    Regression test: Verifies that two distinct lectures in the same domain
    (e.g., Machine Learning: Decision Trees vs Gradient Descent) produce
    different study guides whose contents reflect their specific lecture material.
    """
    # Lecture A: Decision Trees & Gini Impurity
    transcript_a = (
        "In this lecture we discuss Decision Trees and Gini Impurity. "
        "A Decision Tree recursively splits data based on feature thresholds. "
        "Gini Impurity measures how often a randomly chosen element would be incorrectly labeled. "
        "Information Gain is calculated using Entropy and Gini Index."
    )
    topics_a = [
        {
            "title": "Decision Trees & Feature Splits",
            "explanation": "Decision trees partition data space into rectangular decision regions.",
            "key_points": [
                "Recursive binary splitting creates leaf nodes",
                "Pruning prevents tree overfitting"
            ],
            "definitions": [
                {"term": "Gini Impurity", "definition": "A metric of node impurity between 0 and 0.5."}
            ],
            "examples": ["Classifying loan applicants by income and credit score"]
        }
    ]

    # Lecture B: Gradient Descent & Optimization
    transcript_b = (
        "Today we examine Gradient Descent and Learning Rate in deep neural networks. "
        "Gradient Descent iteratively updates model parameters in the direction of steepest descent. "
        "If the learning rate is too large, the optimizer diverges; if too small, convergence is slow. "
        "Stochastic Gradient Descent uses mini-batches to estimate the gradient."
    )
    topics_b = [
        {
            "title": "Gradient Descent & Optimization",
            "explanation": "Iterative optimization algorithm for minimizing objective loss functions.",
            "key_points": [
                "Step size is determined by the learning rate alpha",
                "Batch vs Stochastic gradient computation trade-offs"
            ],
            "definitions": [
                {"term": "Learning Rate", "definition": "Hyperparameter controlling the step size per iteration."}
            ],
            "examples": ["Optimizing weights of linear regression with MSE loss"]
        }
    ]

    guide_a = StudyGuideSynthesizer.generate_master_study_guide(
        transcript_text=transcript_a,
        study_topics=topics_a,
        domain="machine_learning",
        llm_provider=None  # triggers extractive assembly fallback
    )

    guide_b = StudyGuideSynthesizer.generate_master_study_guide(
        transcript_text=transcript_b,
        study_topics=topics_b,
        domain="machine_learning",
        llm_provider=None  # triggers extractive assembly fallback
    )

    # 1. Guides must be strictly distinct
    assert guide_a != guide_b

    # 2. Guide A must contain Lecture A concepts and NO Lecture B concepts
    assert "Decision Trees" in guide_a or "Gini Impurity" in guide_a
    assert "Recursive binary splitting" in guide_a
    assert "Gradient Descent" not in guide_a
    assert "Learning Rate" not in guide_a

    # 3. Guide B must contain Lecture B concepts and NO Lecture A concepts
    assert "Gradient Descent" in guide_b or "Learning Rate" in guide_b
    assert "learning rate alpha" in guide_b
    assert "Decision Trees" not in guide_b
    assert "Gini Impurity" not in guide_b

    # 4. Guides must adhere to the CEM (Concept-Example-Mechanism) Markdown hierarchy
    for guide in [guide_a, guide_b]:
        assert "## 1. The Main Idea" in guide
        assert "## 2. Core Concepts Overview" in guide
        assert "- **Definition**:" in guide
        assert "- **Mechanism**:" in guide
        assert "## 3. Key Definitions" in guide
        assert "## 4. Key Points & Principles" in guide


def test_builder_forwarders_produce_lecture_grounded_content():
    """
    Verifies that calling the domain builder method _build_machine_learning_study_guide
    uses the actual lecture topics instead of a static hardcoded string.
    """
    transcript = "We analyze Convolutional Neural Networks and pooling layers for image classification."
    topics = [
        {
            "title": "Convolutional Neural Networks",
            "explanation": "CNNs use 2D convolution filters to capture spatial spatial hierarchies.",
            "key_points": ["Kernel convolution preserves translation invariance", "Max pooling reduces spatial dimensions"],
            "definitions": [{"term": "Receptive Field", "definition": "Area of input space that affects a feature."}],
            "examples": ["LeNet-5 for digit classification"]
        }
    ]

    guide = StudyGuideSynthesizer._build_machine_learning_study_guide(
        transcript_text=transcript,
        study_topics=topics,
        llm_provider=None
    )

    assert "Convolutional Neural Networks" in guide
    assert "Max pooling reduces spatial dimensions" in guide
    # Ensure it's not the old static hardcoded string (which was about Precision vs Recall)
    assert "Cancer Detection (High Recall Priority)" not in guide


def test_llm_synthesis_flow_with_mock_provider():
    """
    Verifies that when an LLM provider is available, generate_master_study_guide
    calls complete(), performs entailment validation, and returns the generated content.
    """
    mock_provider = MagicMock()
    mock_provider.is_available.return_value = True
    mock_provider.complete.return_value = (
        "# 1. Core Concepts Overview\n"
        "Decision Trees recursively split feature space to classify inputs accurately.\n"
        "# 2. Key Definitions\n"
        "Gini Impurity measures classification error probability.\n"
    )

    mock_validator = MagicMock()
    mock_validator.validate.return_value = SimpleNamespace(passed=True, reason="Grounded")

    transcript = "Decision Trees recursively split feature space to classify inputs. Gini Impurity is covered."
    topics = [{"title": "Decision Trees", "explanation": "Feature splitting"}]

    guide = StudyGuideSynthesizer.generate_master_study_guide(
        transcript_text=transcript,
        study_topics=topics,
        domain="machine_learning",
        llm_provider=mock_provider,
        entailment_validator=mock_validator
    )

    assert mock_provider.complete.called
    assert mock_validator.validate.called
    assert "Decision Trees recursively split feature space" in guide


def test_validation_failure_triggers_extractive_fallback():
    """
    Verifies that if the LLM produces ungrounded claims failing validation,
    the pipeline falls back to the deterministic extractive guide.
    """
    mock_provider = MagicMock()
    mock_provider.is_available.return_value = True
    mock_provider.complete.return_value = (
        "# 1. Core Concepts Overview\n"
        "Quantum flux supercomputers use tachyonic wormholes for time travel.\n"
    )

    mock_validator = MagicMock()
    mock_validator.validate.return_value = SimpleNamespace(passed=False, reason="Severe hallucination")

    transcript = "We cover elementary accounting and cash transactions."
    topics = [{"title": "Cash Transactions", "explanation": "Physical money flow"}]

    guide = StudyGuideSynthesizer.generate_master_study_guide(
        transcript_text=transcript,
        study_topics=topics,
        domain="accounting",
        llm_provider=mock_provider,
        entailment_validator=mock_validator
    )

    # Must fall back to extractive guide with the real topics
    assert "Cash Transactions" in guide
    assert "tachyonic wormholes" not in guide


def test_quizzes_have_zero_alien_concepts_for_supervised_learning():
    """
    Regression test: Verifies that quiz generation for a lecture on supervised learning
    never outputs questions or options mentioning Deep Learning or Neural Networks
    when those terms are absent from the transcript.
    """
    transcript = (
        "In this session we examine Supervised Learning, focusing on Classification and Regression. "
        "Classification predicts discrete labels while Regression estimates continuous values. "
        "We evaluate models using Precision, Recall, Accuracy, and F1-score. "
        "For regression we calculate Mean Absolute Error and Mean Squared Error."
    )
    topics = [
        {
            "title": "Supervised Learning Fundamentals",
            "explanation": "Supervised learning maps feature vectors to output targets using labeled data.",
            "key_points": [
                "Classification predicts categorical targets",
                "Regression predicts continuous numerical values"
            ],
            "definitions": [
                {"term": "Classification", "definition": "Supervised learning task predicting discrete categories."},
                {"term": "Precision", "definition": "Ratio of true positive predictions to total predicted positives."}
            ]
        },
        {
            "title": "Model Evaluation Metrics",
            "explanation": "Quantitative assessment of model performance on test splits.",
            "key_points": [
                "Precision and Recall balance sensitivity and exactness",
                "MAE and MSE evaluate regression residuals"
            ],
            "definitions": [
                {"term": "Recall", "definition": "Ratio of true positive predictions to all actual positive cases."}
            ]
        }
    ]

    quizzes = StudyGuideSynthesizer.generate_conceptual_quizzes(
        transcript_text=transcript,
        lecture_notes={"study_topics": topics, "definitions": topics[0]["definitions"] + topics[1]["definitions"]},
        topic_hint="Supervised Machine Learning"
    )

    assert len(quizzes) >= 4
    for q in quizzes:
        # Schema validation
        assert "question" in q and len(q["question"]) > 10
        assert "options" in q and len(q["options"]) == 4
        assert "correct_answer" in q and q["correct_answer"] in q["options"]
        assert "correct_option_letter" in q and q["correct_option_letter"] in ["A", "B", "C", "D"]
        assert "explanation" in q and len(q["explanation"]) > 5

        # Bloom's Taxonomy: No verbatim recall phrasing
        q_low = q["question"].lower()
        assert "what did the speaker say" not in q_low
        assert "according to this session, what is" not in q_low
        assert "what was said about" not in q_low
        assert "which statement accurately reflects" not in q_low

        # Alien concept filtering
        q_full = f"{q['question']} {' '.join(q['options'])} {q.get('explanation', '')} {q.get('grounded_concept', '')}".lower()
        assert "deep learning" not in q_full, f"Alien concept 'deep learning' found in quiz: {q}"
        assert "neural network" not in q_full, f"Alien concept 'neural network' found in quiz: {q}"
        assert "backpropagation" not in q_full, f"Alien concept 'backpropagation' found in quiz: {q}"

    # Verify questions use scenario-based framing
    has_scenario = any("practitioner" in q["question"].lower() or "scenario" in q["question"].lower() for q in quizzes)
    assert has_scenario, "Quizzes should include scenario-based questions"


def test_executive_briefing_grounded_in_script_with_no_conversational_artifacts():
    """
    Verifies that generate_executive_briefing uses the transcript topics and strips
    conversational openers and noise.
    """
    transcript = "Assalamu alaikum everyone, how are you. Today we discuss Supervised Learning and Evaluation Metrics."
    topics = [
        {
            "title": "Supervised Learning",
            "explanation": "Assalamu alaikum everyone. This section examines supervised learning with labeled data."
        },
        {
            "title": "Evaluation Metrics",
            "explanation": "How are you all. Model evaluation relies on precision, recall, and accuracy."
        }
    ]

    short, detailed = StudyGuideSynthesizer.generate_executive_briefing(
        transcript_text=transcript,
        segments=[],
        recording_type="Lecture",
        study_topics=topics
    )

    assert "assalamu alaikum" not in short.lower()
    assert "how are you" not in short.lower()
    assert "assalamu alaikum" not in detailed.lower()
    assert "Supervised Learning" in short or "Evaluation Metrics" in short
    assert "Supervised Learning" in detailed or "Evaluation Metrics" in detailed


def test_topic_cards_have_exact_rich_format_and_no_alien_concepts():
    """
    Verifies that enrich_study_topics produces the exact rich card structure:
    - Authoritative 2-5 word conceptual title
    - Narrative starting with 'This section...'
    - 3-4 key points
    - 1-3 definitions
    - Concrete practical examples
    - Formulas where applicable (e.g. Precision / Recall / Accuracy)
    - Exam relevance set to True where applicable
    - ZERO alien concepts (no Deep Learning / Neural Networks when absent from transcript)
    """
    transcript = (
        "In this lecture on supervised machine learning, we discuss classification and regression models. "
        "Supervised learning trains on labeled data. Classification predicts discrete classes, whereas regression predicts continuous numbers. "
        "We evaluate classification using accuracy, precision, and recall. "
        "Precision is calculated as true positives over total predicted positives. "
        "Recall measures the ratio of true positives to actual positives. "
        "Remember this for the exam because evaluation metrics are heavily tested."
    )
    raw_topics = [
        {
            "title": "Supervised Learning",
            "explanation": "Supervised learning trains on labeled data where inputs map to target outputs.",
            "key_points": [
                "Models learn mathematical mapping functions from features to targets",
                "Classification predicts discrete categories while regression predicts numbers"
            ],
            "definitions": [],
            "examples": [],
            "formulas": [],
            "exam_relevant": False
        },
        {
            "title": "Evaluation Metrics",
            "explanation": "Model evaluation assesses accuracy, precision, and recall on test data.",
            "key_points": [
                "Accuracy calculates correct predictions over total predictions",
                "Precision and recall balance prediction exactness and sensitivity"
            ],
            "definitions": [],
            "examples": [],
            "formulas": [],
            "exam_relevant": True
        }
    ]

    enriched = StudyGuideSynthesizer.enrich_study_topics(
        study_topics=raw_topics,
        transcript_text=transcript,
        topic_hint="Machine Learning"
    )

    assert len(enriched) >= 2
    for t in enriched:
        # Title
        assert len(t["title"].split()) >= 2
        assert not any(alien in t["title"].lower() for alien in ["deep learning", "neural network", "backpropagation"])

        # Narrative starts with "This section..."
        assert t["explanation"].lower().startswith("this section"), f"Explanation does not start with 'This section': {t['explanation']}"
        assert not any(alien in t["explanation"].lower() for alien in ["deep learning", "neural network", "backpropagation"])

        # Key points
        assert len(t["key_points"]) >= 2
        for kp in t["key_points"]:
            assert not any(alien in kp.lower() for alien in ["deep learning", "neural network"])

        # Definitions
        assert len(t["definitions"]) >= 1
        for d in t["definitions"]:
            term = d.get("term", "").lower()
            assert not any(alien in term for alien in ["deep learning", "neural network"])

        # Examples
        assert len(t["examples"]) >= 1
        for ex in t["examples"]:
            assert not any(alien in ex.lower() for alien in ["deep learning", "neural network"])

    # Formulas in evaluation metrics
    metrics_topic = next(t for t in enriched if "evaluation" in t["title"].lower() or "metrics" in t["title"].lower())
    assert len(metrics_topic["formulas"]) >= 1
    assert any("Precision" in f or "Recall" in f or "Accuracy" in f for f in metrics_topic["formulas"])

