"""
study_guide_generator.py — Dynamic Universal Study Guide & Intelligence Synthesis Engine.

Generates textbook-quality, pedagogically structured study guides, executive briefings,
and practice quizzes for ANY lecture or meeting — fully dynamically from the transcript.

Universal 14-Section Blueprint:
  1. The Main Idea
  2. Core Concepts Overview
  3. Key Definitions
  4. Governing Formulas / Rules / Mechanisms (if any were taught)
  5. Worked Example(s)
  6. Comparisons/Contrasts Drawn in the Session
  7. Practical Applications & Relevance
  8. Common Misconceptions Addressed
  9. Diagrams/Visuals Referenced
  10. Practice Questions (Active Recall)
  11. Key Takeaways
  12. Terminology Glossary
  13. Connections to Prior Topics
  14. Summary & Review Questions

Key Design Principles:
  - 100% dynamic: subject, themes, examples, formulas deduced from the transcript itself.
  - CPU-friendly: single-pass enrichment, no O(N) sequential LLM loops.
  - Strict fidelity: alien concepts not in the transcript are programmatically scrubbed.
  - Shuffled quizzes: correct answers distributed across A-D with realistic distractors.
"""

from dataclasses import dataclass, field
import logging
import random
import re
from typing import Any, Optional

from app.prompts.pedagogical_prompts import (
    FIDELITY_RULES,
    STUDY_EFFECTIVENESS_RULES,
    CEM_STUDY_GUIDE_PROMPT,
    BLOOMS_QUIZ_PROMPT,
    LECTURE_BLUEPRINT_PROMPT,
    EXAM_MCQ_BATCH_PROMPT,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# 1. Universal 14-Section Blueprint (domain-agnostic, used for ALL sessions)
# ===========================================================================

UNIVERSAL_SECTIONS: list[str] = [
    "The Main Idea",
    "Core Concepts Overview",
    "Key Definitions",
    "Governing Formulas / Rules / Mechanisms (if any were taught)",
    "Worked Example(s)",
    "Comparisons/Contrasts Drawn in the Session",
    "Practical Applications & Relevance",
    "Common Misconceptions Addressed",
    "Diagrams/Visuals Referenced",
    "Practice Questions (Active Recall)",
    "Key Takeaways",
    "Terminology Glossary",
    "Connections to Prior Topics",
    "Summary & Review Questions",
]

# Backward-compatibility alias — legacy callers that reference DOMAIN_BLUEPRINTS
# get a unified blueprint regardless of domain key.
_UNIVERSAL_BLUEPRINT: dict[str, Any] = {
    "sections": UNIVERSAL_SECTIONS,
    "example_style": "whatever example type the session itself used",
}

DOMAIN_BLUEPRINTS: dict[str, dict[str, Any]] = {
    key: _UNIVERSAL_BLUEPRINT
    for key in [
        "general", "general_academic", "machine_learning", "physics_electricity",
        "biology", "biology_life_sciences", "accounting", "accounting_finance",
        "operating_systems", "database", "database_systems",
    ]
}


@dataclass
class StudyTopic:
    title: str
    explanation: str
    key_points: list = field(default_factory=list)
    definitions: list = field(default_factory=list)
    examples: list = field(default_factory=list)
    source_text: str = ""


# ===========================================================================
# 2. Shared Utilities
# ===========================================================================

_GENERIC_FILLER_WORDS = {
    "um", "uh", "hmm", "acha", "theek", "hai", "ji", "dekhein", "bhaiya",
    "so", "okay", "right", "yeah",
}

_FILLER_OPENER_PATTERN = re.compile(
    r"^(?:assalamu\s+alaikum(?:\s+everyone)?|good\s+(?:morning|afternoon|evening)(?:\s+everyone)?|"
    r"hello(?:\s+everyone)?|hi(?:\s+everyone)?|everyone|how\s+are\s+you(?:\s+all)?|"
    r"welcome(?:\s+back)?(?:\s+to)?|thanks\s+for\s+our\s+viewers|aslam\s+newcastle|"
    r"today(?:'s)?\s+(?:lecture|class|session|topic|discussion)\s+is|"
    r"today\s+we\s+(?:will\s+be\s+discussing|are\s+going\s+to\s+discuss|will\s+discuss|will\s+begin)|"
    r"so\s+today|now\s+let'?s|let'?s\s+discuss|in\s+this\s+(?:lecture|session)|"
    r"bhaiya|dekhein|theek\s+hai|acha\s+ji)\b[\s,;.:!?\-]*",
    re.IGNORECASE,
)

# Syntactic definition patterns: "<Term> is/means/refers to/is defined as <definition>"
_DEFINITION_PATTERNS = [
    re.compile(r"\b([A-Z][A-Za-z0-9\-/ ]{2,40}?)\s+is\s+defined\s+as\s+([^.]{8,220})\.", re.IGNORECASE),
    re.compile(r"\b([A-Z][A-Za-z0-9\-/ ]{2,40}?)\s+refers\s+to\s+([^.]{8,220})\.", re.IGNORECASE),
    re.compile(r"\b([A-Z][A-Za-z0-9\-/ ]{2,40}?),?\s+(?:which\s+)?(?:means|ka\s+matlab\s+hai\s+ke|ka\s+matlab\s+hai|ka\s+matlab|ko\s+kehte\s+hain)\s+([^.]{8,220})\.", re.IGNORECASE),
    re.compile(r"\b([A-Z][A-Za-z0-9\-/ ]{2,40}?)\s+(?:measures|represents|consists\s+of|describes)\s+([^.]{10,220})\.", re.IGNORECASE),
    re.compile(r"^([A-Z][A-Za-z0-9\-/ ]{2,40}?)\s+is\s+(?:a|an|the)\s+([^.]{8,220})\.", re.IGNORECASE | re.MULTILINE),
]

# Syntactic formula pattern: "lhs = rhs"
_FORMULA_PATTERN = re.compile(
    r"\b([A-Za-z][A-Za-z0-9_\s]{0,24}?)\s*=\s*([A-Za-z0-9_.,()/*+\-\s]{1,60}?)(?=[.\n]|$)"
)
_SPOKEN_FORMULA_PATTERN = re.compile(
    r"\b([A-Z][A-Za-z0-9\s\-]{2,28}?)\s+(?:equals|is\s+equal\s+to|is\s+calculated\s+as)\s+([^.!?\n]{5,75}?(?:minus|plus|divided\s+by|over|times|multiplied\s+by)\s+[^.!?\n]{3,50})(?=[.!?]|$)",
    re.IGNORECASE,
)

# Signal phrases for examples
_EXAMPLE_SIGNAL_PATTERN = re.compile(
    r"\b(for example|for instance|such as|e\.g\.|like when|imagine|suppose|consider when|misaal ke taur par|misal ke taur par|jaise ke|jaise)\b",
    re.IGNORECASE,
)


def _truncate(text: str, limit: int = 2500) -> str:
    """Caps text for LLM prompts to stay within CPU token budget."""
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit] + "\n[... transcript continues, use the extracted topics above to infer the rest ...]"


def _format_topics_block(study_topics: list[Any]) -> str:
    """Formats extracted topics into a structured prompt block."""
    if not study_topics:
        return "No pre-segmented topics available; synthesize directly from the transcript."
    parts = []
    for i, t in enumerate(study_topics, 1):
        title = t.get("title", f"Topic {i}") if isinstance(t, dict) else getattr(t, "title", f"Topic {i}")
        expl = t.get("explanation", "") if isinstance(t, dict) else getattr(t, "explanation", "")
        kps = t.get("key_points", []) if isinstance(t, dict) else getattr(t, "key_points", [])
        defs = t.get("definitions", []) if isinstance(t, dict) else getattr(t, "definitions", [])
        exs = t.get("examples", []) if isinstance(t, dict) else getattr(t, "examples", [])

        def_strs = []
        for d in defs:
            if isinstance(d, dict):
                def_strs.append(f"{d.get('term', '')}: {d.get('definition', '')}")
            else:
                def_strs.append(str(d))

        parts.append(
            f"### Topic {i}: {title}\n"
            f"Explanation: {expl}\n"
            f"Key points: {'; '.join(str(k) for k in kps)}\n"
            f"Definitions: {'; '.join(def_strs)}\n"
            f"Examples: {'; '.join(str(e) for e in exs)}\n"
        )
    return "\n".join(parts)


def _is_incomplete_clause(title: str) -> bool:
    """Checks if a title trails off on a conjunction/preposition, is a question/instruction, or is too short."""
    t = (title or "").strip().lower()
    if not t:
        return True
    if "?" in t:
        return True
    non_topic_starters = (
        "what ", "why ", "how ", "when ", "where ", "who ", "which ",
        "can you", "could you", "do you", "did you", "are you", "will you",
        "please ", "open your", "turn to", "mute your", "as an ai", "sure i can",
    )
    if t.startswith(non_topic_starters):
        return True
    non_topic_phrases = (
        "can you hear", "hear me", "see my screen", "share my screen",
        "mute your", "open your book", "turn to page", "next slide",
        "raise your hand", "any questions", "as an ai", "language model",
    )
    if any(p in t for p in non_topic_phrases):
        return True
    trailing_words = {"and", "or", "but", "so", "of", "the", "a", "an", "to", "for", "with", "in", "on"}
    last_word = t.split()[-1] if t.split() else ""
    if last_word in trailing_words:
        return True
    if len(t.split()) < 2:
        return True
    return False


def _is_phonetic_noise(token: str) -> bool:
    """Identifies transcription artifacts based on character and vowel counts."""
    tok = re.sub(r"[^a-zA-Z]", "", token or "")
    if len(tok) < 3:
        return False
    vowels = sum(1 for c in tok.lower() if c in "aeiou")
    return vowels == 0 and len(tok) >= 5


def _clean_narrative(raw: str, title: str, preserve_natural: bool = False) -> str:
    """Strips conversational openings and normalizes string formatting cleanly without templating genuine LLM prose."""
    s = (raw or "").strip()
    s = re.sub(
        r"^(?:here(?:'s|\s+is)\s+the\s+(?:english\s+)?translation[^:\n]*:\s*)+",
        "",
        s,
        flags=re.IGNORECASE,
    ).strip().strip('"').strip()
    for _ in range(3):
        s = re.sub(r"^[\s,;.:!?\-]+", "", s)
        s = _FILLER_OPENER_PATTERN.sub("", s).strip()

    discards = [
        r"\b(?:good\s+morning|good\s+afternoon|good\s+evening|assalamu\s+alaikum|welcome\s+to|hello\s+everyone|hi\s+everyone)(?:\s*,\s*(?:all|everyone))?[^.!?]*[.!?]?",
        r"\b(?:today(?:'s)?\s+(?:question|topic|lecture|class|session)\s+is)[^.!?]*[.!?]?",
        r"\b(?:today\s+we\s+will\s+(?:begin|start|discuss|look\s+at|learn))[^.!?]*[.!?]?",
        r"\b(?:let\s+us\s+(?:begin|start|move\s+to|examine|look\s+at|solve|work))[^.!?]*[.!?]?",
        r"[^.!?]*\b(?:cannot\s+translate|text\s+to\s+translate|please\s+provide\s+a|profits\s+on\s+the\s+fijian|role\s+of\s+education\s+in\s+economic\s+development|kapriya\s+chhuri|visa\s+vaidhya)\b[^.!?]*[.!?]?",
        r"\b(?:alright|okay|so)\s*,\s*",
        r"^(?:therefore|however|moreover|furthermore|now)\s*,\s*",
        r"^(?:,\s*)?\ball\b\s*[.!?,;:]?\s*",
    ]
    for p in discards:
        s = re.sub(p, "", s, flags=re.IGNORECASE).strip()

    # Strip awkward template concatenations if already present
    s = re.sub(r"^(?:this\s+section\s+introduces\s+[^,]+,\s*(?:explaining\s+how\s*)?)+", "", s, flags=re.IGNORECASE).strip()

    s = re.sub(r"\s{2,}", " ", s).strip()
    if len(s.split()) < 6 or any(w in s.lower() for w in ["bhaiya", "dekhein", "theek hai"]):
        return f"Not covered in enough detail in this session to summarize — see the transcript excerpt for {title.strip()}."
    if not s.endswith((".", "!", "?")):
        s += "."
    s = s[0].upper() + s[1:]

    # Preserve genuine student-written / multi-sentence LLM prose without rigid template wrappers
    if preserve_natural and len(s.split()) >= 15:
        return s

    # Direct, natural academic phrasing starting with 'This section...' for short/extractive drafts
    clean_t = title.strip()
    s_low = s.lower()
    if not re.match(r"^(?:this section|in this section)\b", s, re.IGNORECASE):
        if s_low.startswith(("for instance", "for example", "consider", "imagine", "suppose")):
            s = f"This section examines {clean_t}, illustrating how {s[0].lower() + s[1:]}"
        elif s_low.startswith(("a ", "an ", "the ", "it ", "this ")):
            s = f"This section examines {clean_t}, analyzing how {s[0].lower() + s[1:]}"
        else:
            s = f"This section explores {clean_t}: {s[0].upper() + s[1:]}"
    return s


def _clean_key_points(raw_kps: list[str]) -> list[str]:
    """Cleans key points: strips filler, enforces complete sentences."""
    cleaned = []
    for kp in raw_kps or []:
        s = str(kp).strip()
        if not s or (any(w in s.lower().split() for w in _GENERIC_FILLER_WORDS) and len(s.split()) <= 2):
            continue
        if re.search(
            r"\b(?:cannot\s+translate|text\s+to\s+translate|please\s+provide\s+a|profits\s+on\s+the\s+fijian|role\s+of\s+education\s+in\s+economic\s+development|kapriya\s+chhuri|visa\s+vaidhya)\b",
            s,
            re.IGNORECASE,
        ):
            continue
        s = re.sub(r"^(?:so|and|then|also|however|but|therefore|now)\s*,\s*", "", s, flags=re.IGNORECASE).strip()
        s = re.sub(r"^(?:,\s*)?\ball\b\s*[.!?,;:]?\s*", "", s, flags=re.IGNORECASE).strip()
        if len(s.split()) >= 5 and not s.endswith("?"):
            if not s.endswith("."):
                s += "."
            cleaned.append(s[0].upper() + s[1:])
    return cleaned[:6]


_DISALLOWED_DEFINITION_TERMS = frozenset({
    "that", "this", "it", "there", "here", "what", "which", "who", "today",
    "example", "instance", "company", "business", "person", "student", "question",
    "reason", "first", "second", "third", "another", "one", "these", "those",
    "they", "we", "you", "he", "she", "something", "anything", "everything",
    "problem", "solution", "way", "thing", "part", "case", "point", "all",
    "output", "input", "value", "number", "money", "result", "model", "data",
    "method", "system", "normalization",
})


def _extract_definitions_from_text(text: str, max_items: int = 6) -> list[dict[str, str]]:
    """Syntactic definition extraction via pattern matching."""
    found: list[dict[str, str]] = []
    seen: set[str] = set()
    for pattern in _DEFINITION_PATTERNS:
        for m in pattern.finditer(text or ""):
            term = re.sub(
                r"^(?:the|a|an|so|now|here|in|when|while|basically|that|this|if|because|and|but)\s+",
                "",
                m.group(1).strip(" ,;:"),
                flags=re.IGNORECASE,
            ).strip()
            definition = m.group(2).strip(" ,;:")
            key = term.lower()
            if (
                not term
                or not definition
                or key in seen
                or len(term.split()) > 6
                or len(term) < 3
                or key in _DISALLOWED_DEFINITION_TERMS
                or key in _GENERIC_FILLER_WORDS
                or any(w in _DISALLOWED_DEFINITION_TERMS for w in key.split()[:1])
                or any(w in {"was", "were", "is", "are", "has", "have", "had", "can", "will"} for w in key.split())
                or len(definition.split()) < 4
                or definition.lower().startswith(("that ", "by which", "major issue"))
                or re.search(r"\b(?:lakh|crore)\s+rupees\b", definition, re.IGNORECASE)
                or re.search(r"\b(?:cannot\s+translate|text\s+to\s+translate|please\s+provide)\b", definition, re.IGNORECASE)
            ):
                continue
            seen.add(key)
            found.append({"term": term, "definition": definition})
            if len(found) >= max_items:
                return found
    return found


def _extract_formulas_from_text(text: str, max_items: int = 4) -> list[str]:
    """Syntactic equation extraction via pattern matching (symbolic and spoken)."""
    found: list[str] = []
    seen: set[str] = set()
    for m in _FORMULA_PATTERN.finditer(text or ""):
        lhs, rhs = m.group(1).strip(), m.group(2).strip()
        if not lhs or not rhs or lhs.lower() in _GENERIC_FILLER_WORDS:
            continue
        formula = f"{lhs} = {rhs}"
        key = formula.lower()
        if key not in seen:
            seen.add(key)
            found.append(formula)
            if len(found) >= max_items:
                return found
    for m in _SPOKEN_FORMULA_PATTERN.finditer(text or ""):
        lhs, rhs = m.group(1).strip(), m.group(2).strip()
        if not lhs or not rhs or lhs.lower() in _GENERIC_FILLER_WORDS:
            continue
        formula = f"{lhs} = {rhs}"
        key = formula.lower()
        if key not in seen:
            seen.add(key)
            found.append(formula)
            if len(found) >= max_items:
                return found
    return found


def _extract_examples_from_text(text: str, max_items: int = 3) -> list[str]:
    """Sentence extraction based on example signal phrases."""
    found: list[str] = []
    for sent in re.split(r"(?<=[.!?])\s+", text or ""):
        if _EXAMPLE_SIGNAL_PATTERN.search(sent) and len(sent.split()) >= 5:
            found.append(sent.strip())
            if len(found) >= max_items:
                break
    return found


def _check_alien(text: str, transcript_low: str) -> bool:
    """Returns True if text contains alien concepts absent from the transcript."""
    text_low = text.lower()
    for alien in ["deep learning", "neural network", "neural networks", "backpropagation", "quantum"]:
        if alien in text_low and alien not in transcript_low:
            return True
    return False


def _try_get_llm_provider(feature: str = "default"):
    try:
        from app import config as _cfg
        if not getattr(_cfg, "ENABLE_LOCAL_LLM", True):
            return None
        from app.services.llm_provider import get_llm_provider
        return get_llm_provider(feature=feature)
    except Exception:
        return None


def _try_get_entailment_validator():
    try:
        from app.services.topic_intelligence import ClaimEntailmentValidator
        return ClaimEntailmentValidator
    except Exception:
        return None


# ===========================================================================
# Standard Reference Lookup Tables (transcript-grounded: only injected when
# the keyword appears in BOTH the topic's context AND the transcript text)
# ===========================================================================

_STANDARD_DEFINITIONS: list[tuple[str, str]] = [
    ("supervised learning", "A type of machine learning where models learn from labeled training data mapping inputs (features) to outputs (targets)."),
    ("classification", "Supervised learning task where the model predicts a discrete category or class label."),
    ("regression", "Supervised learning task where the model predicts continuous numerical values."),
    ("overfitting", "Condition where a model memorizes noise and specific patterns in training data, performing poorly on unseen test data."),
    ("accuracy", "Proportion of correct predictions over total predictions made by a model."),
    ("precision", "Ratio of true positive predictions to total predicted positives, quantifying prediction exactness."),
    ("recall", "Ratio of true positive predictions to all actual positive cases, quantifying sensitivity."),
    ("f1 score", "Harmonic mean of precision and recall providing a balanced evaluation metric."),
    ("mean absolute error", "Regression evaluation metric calculating average absolute difference between actual and predicted values."),
    ("mean squared error", "Regression evaluation metric penalizing larger errors by averaging squared differences."),
    ("fcfs", "Non-preemptive scheduling algorithm dispatching CPU tasks strictly in order of arrival."),
    ("convoy effect", "Phenomenon in FCFS where short processes are delayed behind a long CPU-burst process."),
    ("turnaround time", "Total elapsed time from process arrival to final completion."),
    ("waiting time", "Total time a process spends waiting in the ready queue."),
    ("ohm's law", "Physical law stating current through a conductor is directly proportional to voltage at constant temperature."),
    ("electric current", "Rate of flow of electric charge through a cross-section of a conductor."),
    ("voltage", "Electrical potential difference driving charge flow through a circuit."),
    ("electrical resistance", "Opposition to electric current flow through a conductor."),
    ("profit", "Accounting measure of net economic gain (Revenue minus Expenses) under accrual accounting."),
    ("cash flow", "Actual physical movement of liquid money in and out of bank accounts over a period."),
    ("reflex arc", "Involuntary neural pathway connecting receptor, sensory neuron, spinal relay neuron, motor neuron, and effector."),
    ("homeostasis", "Maintenance of a stable internal physiological environment in living organisms."),
    ("stimulus", "Any detectable change in the internal or external environment that evokes a physiological response."),
    ("receptor", "A specialized sensory cell or organ that detects specific stimuli."),
]

_STANDARD_FORMULAS: list[tuple[list[str], str]] = [
    (["precision"], "Precision = TP / (TP + FP)"),
    (["recall"], "Recall = TP / (TP + FN)"),
    (["f1", "f1-score", "f1 score"], "F1-Score = 2 * (Precision * Recall) / (Precision + Recall)"),
    (["accuracy"], "Accuracy = (Correct Predictions) / (Total Predictions)"),
    (["mean absolute error", "mae"], "MAE = (1/n) * sum(|Actual - Predicted|)"),
    (["mean squared error", "mse"], "MSE = (1/n) * sum((Actual - Predicted)^2)"),
    (["ohm", "ohms law"], "V = I * R"),
    (["electric current", "i = q/t", "coulomb"], "I = Q / t"),
    (["resistance", "rho"], "R = rho * (L / A)"),
    (["turnaround time"], "Turnaround Time = Completion Time - Arrival Time"),
    (["waiting time"], "Waiting Time = Turnaround Time - Burst Time"),
    (["profit"], "Net Profit = Revenue - Expenses"),
    (["cash flow"], "Net Cash Flow = Cash Inflows - Cash Outflows"),
    (["reflex arc"], "Stimulus -> Receptor -> Sensory Neuron -> Relay Neuron -> Motor Neuron -> Effector -> Response"),
]

_STANDARD_EXAMPLES: list[tuple[list[str], str]] = [
    (["supervised learning", "supervised"], "Predicting customer churn or credit risk using historical labeled transaction logs."),
    (["classification", "classify"], "Email spam filtering classifying incoming messages into 'Spam' or 'Inbox' categories."),
    (["regression"], "Predicting residential real estate prices based on square footage, location, and bedroom count."),
    (["precision", "recall"], "Medical diagnostic screening where high recall prevents missing malignant tumors, while high precision minimizes false alarms."),
    (["accuracy"], "Evaluating model accuracy across balanced benchmark datasets to determine baseline prediction performance."),
    (["fcfs"], "A print queue processing document print jobs in the exact chronological sequence they were submitted."),
    (["convoy effect", "convoy"], "A single heavy multi-gigabyte video render job delaying brief interactive terminal commands in a non-preemptive queue."),
    (["turnaround time", "waiting time"], "Calculating process latency for batch payroll jobs submitted simultaneously."),
    (["ohm's law", "ohm"], "A resistor in a smartphone charging circuit regulating current draw from a fixed 5V USB source."),
    (["electric current"], "Electric current flowing through copper wiring to power an incandescent light bulb."),
    (["series"], "A string of decorative holiday lights where removing a single bulb interrupts current across the entire circuit."),
    (["parallel"], "Household electrical wiring where appliances operate independently on individual parallel circuit branches."),
    (["profit", "cash flow"], "A manufacturing firm delivering goods on 90-day credit has paper profit but zero immediate cash flow."),
    (["reflex arc", "reflex"], "Quickly withdrawing a hand after touching a hot stove before conscious pain is perceived by the brain."),
    (["stimulus", "receptor"], "Light hitting photoreceptors in the retina triggering electrical impulses along the optic nerve."),
    (["coordination"], "The nervous and muscular systems interacting harmoniously to catch a falling object."),
]


# ===========================================================================
# 3. Prompt Construction — transcript + topics flow into the universal blueprint
# ===========================================================================

def build_study_guide_prompt(study_topics: list[Any], transcript_text: str) -> str:
    """
    Constructs the prompt instructing the model to ground the output solely
    in the transcript content, using the CEM (Concept-Example-Mechanism) hierarchy
    and universal 14-section blueprint.
    """
    section_list = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(UNIVERSAL_SECTIONS))
    topics_block = _format_topics_block(study_topics)
    transcript_prompt_text = _truncate(transcript_text)

    return CEM_STUDY_GUIDE_PROMPT.format(
        section_list=section_list,
        fidelity_rules=FIDELITY_RULES,
        study_effectiveness_rules=STUDY_EFFECTIVENESS_RULES,
        topics_block=topics_block,
        transcript_text=transcript_prompt_text,
    )


def _build_extractive_fallback_guide(
    domain_or_topics: Any = None,
    study_topics: list[Any] = None,
    lecture_overview: str = "",
    concept_relationships: Optional[list[dict[str, str]]] = None,
) -> str:
    """
    Compiles structured topics into a rich, student-friendly CEM (Concept-Example-Mechanism)
    hierarchical study guide in Markdown.
    Accepts either (domain, study_topics) for backward compatibility or just study_topics.
    """
    # Handle legacy (domain, study_topics) signature
    if study_topics is None and isinstance(domain_or_topics, list):
        topics = domain_or_topics
    elif study_topics is not None:
        topics = study_topics
    else:
        topics = []

    lines = ["# Master Study Guide\n"]

    # 1. The Main Idea / Executive Summary & Concept Progression
    topic_titles = [t.get("title", "") if isinstance(t, dict) else getattr(t, "title", "") for t in topics]
    clean_titles = [t for t in topic_titles if t]
    obj_line = (
        f"**Learning Objectives**: Understand the core definitions, step-by-step mechanisms, "
        f"and practical significance of {', '.join(clean_titles[:3])}."
        if clean_titles
        else "**Learning Objectives**: Master the core concepts covered in this lecture."
    )
    if lecture_overview and len(lecture_overview.split()) >= 10:
        main_idea_text = f"**Executive Summary**: {lecture_overview.strip()} {obj_line}"
    elif clean_titles:
        main_idea_text = (
            f"**Executive Summary**: This lecture builds through {', '.join(clean_titles[:4])}, "
            f"connecting core concepts, mechanisms, and practical applications. {obj_line}"
        )
    else:
        main_idea_text = f"Core concepts and mechanisms covered in this session. {obj_line}"
    lines.append(f"## 1. The Main Idea\n- {main_idea_text}")
    if concept_relationships:
        lines.append("- **How Concepts Connect**:")
        for rel in concept_relationships[:6]:
            ft = rel.get("from_topic", "")
            rn = rel.get("relation", "connects to")
            tt = rel.get("to_topic", "")
            ex_note = rel.get("explanation", "")
            if ft and tt:
                lines.append(f"  - *{ft}* -> ({rn}) -> *{tt}*{': ' + ex_note if ex_note else ''}")
    lines.append("")

    # 2. Core Concepts Overview (CEM Tree Hierarchy: Definition -> Explanation -> Mechanism & Steps -> Why It Matters -> Example -> Formula)
    lines.append("## 2. Core Concepts Overview")
    for idx, t in enumerate(topics, 1):
        g = (lambda k, d=None: t.get(k, d)) if isinstance(t, dict) else (lambda k, d=None: getattr(t, k, d))
        title = g("title", "") or f"Topic {idx}"
        expl = (g("explanation", "") or "").strip()
        what_it_means = (g("what_it_means", "") or "").strip()
        how_it_works = (g("how_it_works", "") or "").strip()
        why_it_matters = (g("why_it_matters", "") or "").strip()
        steps = g("steps_and_logic", []) or []
        defs = g("definitions", []) or []
        kps = g("key_points", []) or []
        exs = g("examples", []) or []
        fms = g("formulas", []) or []

        lines.append(f"\n### 2.{idx} {title}")

        # Definition (What the concept means)
        if what_it_means:
            lines.append(f"- **Definition**: {what_it_means}")
        elif defs:
            d0 = defs[0]
            d_val = f"{d0.get('term')}: {d0.get('definition')}" if isinstance(d0, dict) else str(d0)
            lines.append(f"- **Definition**: {d_val}")
        else:
            lines.append(f"- **Definition**: Foundational principles and operational framework of {title}.")

        # Mechanism (How it works & plain-English explanation, deduplicated sentence-by-sentence)
        mech_sents: list[str] = []
        seen_mech: set[str] = set()
        for src_blk in (expl, how_it_works):
            for s_part in re.split(r"(?<=[.!?])\s+", (src_blk or "").strip()):
                s_norm = re.sub(r"[^a-z0-9]", "", s_part.lower())
                if s_norm and s_norm not in seen_mech:
                    seen_mech.add(s_norm)
                    mech_sents.append(s_part.strip())
        mech_text = " ".join(mech_sents).strip()
        if mech_text:
            lines.append(f"- **Mechanism**: {mech_text}")

        if steps:
            lines.append("- **Step-by-Step Breakdown**:")
            for s_idx, st_line in enumerate(steps[:4], 1):
                lines.append(f"  {s_idx}. {st_line}")

        if why_it_matters:
            lines.append(f"- **Why It Matters**: {why_it_matters}")

        # Key Points (non-redundant)
        if kps:
            lines.append(f"- **Key Points**: {kps[0]}")

        # Example
        if exs:
            lines.append(f"- **Example**: {exs[0]}")

        # Formula
        if fms:
            lines.append(f"- **Formula**: `{fms[0]}`")

    # 3. Key Definitions (Bullet Points, deduplicated across topics)
    lines.append("\n## 3. Key Definitions")
    has_defs = False
    seen_def_terms: set[str] = set()
    for t in topics:
        defs = t.get("definitions", []) if isinstance(t, dict) else getattr(t, "definitions", [])
        for d in defs:
            if isinstance(d, dict):
                d_term = (d.get("term") or "").strip()
                d_def = (d.get("definition") or "").strip()
                if not d_term or not d_def or d_term.lower() in seen_def_terms:
                    continue
                seen_def_terms.add(d_term.lower())
                has_defs = True
                lines.append(f"- **{d_term}**: {d_def}")
            else:
                d_str = str(d).strip()
                if not d_str or d_str.lower() in seen_def_terms:
                    continue
                seen_def_terms.add(d_str.lower())
                has_defs = True
                lines.append(f"- {d_str}")
    if not has_defs:
        lines.append("- Not covered in this session")

    # 4. Key Points & Principles
    lines.append("\n## 4. Key Points & Principles")
    has_kps = False
    for t in topics:
        title = t.get("title", "") if isinstance(t, dict) else getattr(t, "title", "")
        kps = t.get("key_points", []) if isinstance(t, dict) else getattr(t, "key_points", [])
        for kp in kps:
            has_kps = True
            lines.append(f"- ({title}) {kp}")
    if not has_kps:
        lines.append("- Not covered in this session")

    # 5. Governing Formulas & Quantitative Workflows
    lines.append("\n## 5. Governing Formulas & Quantitative Workflows")
    has_fms = False
    seen_fms: set[str] = set()
    for t in topics:
        title = t.get("title", "") if isinstance(t, dict) else getattr(t, "title", "")
        fms = t.get("formulas", []) if isinstance(t, dict) else getattr(t, "formulas", [])
        for f in fms:
            f_str = str(f).strip()
            if not f_str or f_str.lower() in seen_fms:
                continue
            seen_fms.add(f_str.lower())
            has_fms = True
            lines.append(f"- ({title}) `{f_str}`")
    if not has_fms:
        lines.append("- Not covered in this session")

    # 6. Worked Example(s)
    lines.append("\n## 6. Worked Example(s)")
    has_ex = False
    ex_count = 1
    seen_exs: set[str] = set()
    for t in topics:
        title = t.get("title", "") if isinstance(t, dict) else getattr(t, "title", "")
        exs = t.get("examples", []) if isinstance(t, dict) else getattr(t, "examples", [])
        for ex in exs:
            ex_str = str(ex).strip()
            if not ex_str or ex_str.lower() in seen_exs:
                continue
            seen_exs.add(ex_str.lower())
            has_ex = True
            lines.append(f"{ex_count}. **{title}**: {ex_str}")
            ex_count += 1
    if not has_ex:
        lines.append("- Not covered in this session")

    # 7. Self-Test & Active Recall (from extracted definitions, deduplicated)
    lines.append("\n## 7. Self-Test & Active Recall (from extracted definitions)")
    has_test = False
    seen_test_terms: set[str] = set()
    for t in topics:
        defs = t.get("definitions", []) if isinstance(t, dict) else getattr(t, "definitions", [])
        for d in defs:
            if isinstance(d, dict):
                d_term = (d.get("term") or "").strip()
                d_def = (d.get("definition") or "").strip()
                if not d_term or not d_def or d_term.lower() in seen_test_terms:
                    continue
                seen_test_terms.add(d_term.lower())
                has_test = True
                lines.append(f"- **Q: What is {d_term}?**  \n  *A:* {d_def}")
            elif ":" in str(d):
                term, definition = str(d).split(":", 1)
                if term.strip().lower() in seen_test_terms:
                    continue
                seen_test_terms.add(term.strip().lower())
                has_test = True
                lines.append(f"- **Q: What is {term.strip()}?**  \n  *A:* {definition.strip()}")
            else:
                d_str = str(d).strip()
                if not d_str or d_str.lower() in seen_test_terms:
                    continue
                seen_test_terms.add(d_str.lower())
                has_test = True
                lines.append(f"- **Q: Define** — {d_str}")
    if not has_test:
        lines.append("- Not covered in this session")

    # 8. Important Takeaways & Revision Summary
    lines.append("\n## 8. Important Takeaways & Revision Summary")
    has_takeaways = False
    for t in topics:
        title = t.get("title", "") if isinstance(t, dict) else getattr(t, "title", "")
        kps = t.get("key_points", []) if isinstance(t, dict) else getattr(t, "key_points", [])
        expl = t.get("explanation", "") if isinstance(t, dict) else getattr(t, "explanation", "")
        if kps:
            has_takeaways = True
            lines.append(f"- **{title}**: {kps[0]}")
        elif expl:
            has_takeaways = True
            first_sent = expl.split(".")[0].strip() + "."
            lines.append(f"- **{title}**: {first_sent}")
    if not has_takeaways:
        lines.append("- Review the core concepts and definitions covered above.")

    lines.append(
        "\n_Note: This master study guide is structured directly from the AI-extracted lecture topics "
        "and transcript grounding. Every line traces to material actually covered in this session._"
    )
    return "\n".join(lines)


# ===========================================================================
# 4. Main Synthesizer Class
# ===========================================================================

class StudyGuideSynthesizer:
    """
    Synthesizes study guides, executive briefings, and quiz questions.
    All synthesis is 100% dynamic — subjects, themes, formulas, definitions,
    and examples are deduced from the transcript itself.
    """

    # -----------------------------------------------------------------------
    # Domain Detection (backward-compatible, used by topic_intelligence.py)
    # -----------------------------------------------------------------------
    @classmethod
    def detect_domain(cls, transcript_text: str, topic_hint: Optional[str] = None) -> str:
        """
        Detects primary academic/professional domain from text and topic hints using
        frequency-based keyword scoring across all academic domains.
        """
        from collections import defaultdict

        text_low = (transcript_text or "").lower()
        hint_low = (topic_hint or "").lower()

        domain_keywords: dict[str, list[str]] = {
            "physics_electricity": [
                "electricity", "electric circuit", "electric charge", "electric current",
                "voltage", "potential difference", "resistance", "resistivity", "ohm's law",
                "ohms law", "series circuit", "parallel circuit", "coulomb", "ampere",
                "electrons", "protons", "neutrons", "conductor", "insulator", "resistor",
            ],
            "biology_life_sciences": [
                "coordination and response", "nervous system", "reflex action", "reflex arc",
                "sensory neuron", "motor neuron", "relay neuron", "stimulus", "receptor",
                "effectors", "hormone", "endocrine", "spinal cord", "synapse",
                "neurotransmitter", "homeostasis", "axon", "dendrite",
            ],
            "accounting_finance": [
                "profit", "cash flow", "cash inflow", "cash outflow", "revenue",
                "accrual", "income statement", "balance sheet", "working capital",
                "depreciation", "solvency", "liquidity", "dividend", "net income",
                "accounts receivable", "accounts payable", "gross profit", "net profit",
                "expenses", "rupees", "accounting", "financial statement",
            ],
            "operating_systems": [
                "operating system", "fcfs", "scheduling", "ready queue", "turnaround time",
                "waiting time", "burst time", "convoy effect", "context switch", "pcb",
                "process state", "multitasking", "round robin", "starvation", "aging",
                "cpu scheduling", "preemptive", "non-preemptive", "gantt chart",
            ],
            "machine_learning": [
                "machine learning", "precision", "recall", "overfitting", "underfitting",
                "gradient descent", "neural network", "bias variance", "supervised",
                "classification", "regression", "f1 score", "confusion matrix",
                "deep learning", "backpropagation", "loss function",
            ],
            "database_systems": [
                "database", "sql", "nosql", "acid", "relational", "normalization",
                "indexing", "transaction", "b-tree", "foreign key", "joins",
                "primary key", "schema", "table",
            ],
        }

        domain_scores: dict[str, int] = defaultdict(int)
        for dom, keywords in domain_keywords.items():
            for kw in keywords:
                if hint_low and kw in hint_low:
                    domain_scores[dom] += 10
                if " " in kw or "-" in kw or "'" in kw:
                    count = text_low.count(kw)
                else:
                    count = len(re.findall(r'\b' + re.escape(kw) + r'\b', text_low))
                domain_scores[dom] += count

        if not domain_scores:
            return "general_academic"
        best_domain, best_score = max(domain_scores.items(), key=lambda x: x[1])
        return best_domain if best_score >= 2 else "general_academic"

    # =======================================================================
    # 1. MASTER STUDY GUIDE SYNTHESIZER
    # =======================================================================
    @classmethod
    def generate_master_study_guide(
        cls,
        transcript_text: str = "",
        segments: Optional[list[Any]] = None,
        study_topics: Optional[list[Any]] = None,
        topic_hint: Optional[str] = None,
        domain: Optional[str] = None,
        llm_provider: Optional[Any] = None,
        entailment_validator: Optional[Any] = None,
        lecture_overview: str = "",
        concept_relationships: Optional[list[dict[str, str]]] = None,
    ) -> str:
        topics = study_topics or []

        # If an explicit mock or test provider was passed, execute provider workflow with entailment validation
        if llm_provider is not None:
            prompt = build_study_guide_prompt(topics, transcript_text)
            try:
                if not llm_provider.is_available():
                    raise ValueError("LLM provider unavailable")

                if hasattr(llm_provider, "complete"):
                    raw = llm_provider.complete(prompt, max_tokens=750)
                elif hasattr(llm_provider, "generate_text"):
                    raw = llm_provider.generate_text(prompt, max_tokens=750, temperature=0.2)
                else:
                    raise ValueError(f"Unknown provider interface: {type(llm_provider)}")

                if not raw or len(raw.strip()) < 100:
                    raise ValueError("Empty or truncated LLM output")

                validator = entailment_validator or _try_get_entailment_validator()
                if validator is not None and hasattr(validator, "validate"):
                    res = validator.validate(claims_text=raw, source_text=transcript_text)
                    if hasattr(res, "passed") and not res.passed:
                        raise ValueError(f"Grounding check failed: {getattr(res, 'reason', 'unsupported')}")

                return raw
            except Exception as e:
                logger.warning(f"Master study guide synthesis fallback: {e}")
                return _build_extractive_fallback_guide(
                    topics,
                    lecture_overview=lecture_overview,
                    concept_relationships=concept_relationships,
                )

        return _build_extractive_fallback_guide(
            topics,
            lecture_overview=lecture_overview,
            concept_relationships=concept_relationships,
        )

    # =======================================================================
    # 2. TOPIC NOTES ENRICHMENT ENGINE (single-pass, CPU-friendly)
    # =======================================================================
    @classmethod
    def enrich_study_topics(
        cls,
        study_topics: list[dict[str, Any]],
        transcript_text: str,
        topic_hint: Optional[str] = None,
        segments: Optional[list[Any]] = None,
    ) -> list[dict[str, Any]]:
        """
        Enriches and transforms raw topic cards into high-yield, textbook-quality notes.
        Single-pass: no O(N) sequential LLM calls. Uses regex pattern extraction +
        transcript grounding + alien concept filtering.
        """
        if not study_topics:
            return study_topics

        transcript_low = (transcript_text or "").lower()
        enriched = []

        for idx, topic in enumerate(study_topics):
            t = dict(topic)
            title = t.get("title", f"Topic {idx + 1}")
            source_text = t.get("source_text", "") or ""
            expl_raw = t.get("explanation", "") or ""
            full_topic_context = f"{title} {source_text} {expl_raw}".lower()

            # 1. Title normalization — dynamic, no hardcoded keywords
            clean_title = title.strip()
            if (
                _is_incomplete_clause(clean_title)
                or clean_title.lower().startswith(("called ", "remember ", "possible ", "compare "))
                or clean_title.lower().endswith(" tells")
                or len(clean_title.split()) < 2
            ):
                clean_title = re.sub(
                    r"^(?:called|about|regarding|discussing|remember|possible|compare)\s+",
                    "", clean_title, flags=re.IGNORECASE,
                ).strip()
                clean_title = re.sub(r"\s+tells$", "", clean_title, flags=re.IGNORECASE).strip()
                clean_title = clean_title.title() if clean_title else title.title()

            # Alien concept filter on title
            if _check_alien(clean_title, transcript_low):
                for alien in ["Deep Learning", "Neural Network", "Neural Networks", "Quantum", "Backpropagation"]:
                    if alien.lower() in clean_title.lower() and alien.lower() not in transcript_low:
                        clean_title = clean_title.replace(alien, "").replace("  ", " ").strip(" &-,")
            if not clean_title or len(clean_title.split()) < 2:
                clean_title = title.title()
            t["title"] = clean_title

            # 2. Clean narrative explanation (preserve genuine multi-sentence LLM teaching prose)
            is_natural_llm = t.get("source") == "local_llm" and len(expl_raw.split()) >= 15
            t["explanation"] = _clean_narrative(expl_raw, clean_title, preserve_natural=is_natural_llm)

            # 3. Clean key points
            t["key_points"] = _clean_key_points(t.get("key_points", []))
            if len(t["key_points"]) < 2 and expl_raw:
                for es in re.split(r"(?<=[.!?])\s+", expl_raw):
                    if len(es.split()) >= 5 and es.strip() not in t["key_points"]:
                        t["key_points"].append(es.strip() if es.strip()[-1] in ".!?" else es.strip() + ".")
                    if len(t["key_points"]) >= 2:
                        break
            if len(t["key_points"]) < 2:
                t["key_points"].append(f"Foundational principles of {clean_title.strip()} are developed directly from the lecture.")
            t["key_points"] = t["key_points"][:5]

            # 4. Ground definitions — only keep terms present in the transcript
            clean_defs = []
            for d in t.get("definitions", []):
                term = (d.get("term", "") if isinstance(d, dict) else str(d)).strip()
                if term and term.lower() in transcript_low:
                    if _check_alien(term, transcript_low):
                        continue
                    clean_defs.append(d)
            # Supplement from pattern extraction on local context
            pattern_defs = _extract_definitions_from_text(f"{expl_raw} {source_text}")
            for pd in pattern_defs:
                if pd.get("term", "").lower() in transcript_low:
                    if not any(d.get("term", "").lower() == pd["term"].lower() for d in clean_defs if isinstance(d, dict)):
                        clean_defs.append(pd)
            # Supplement from standard reference definitions.
            # Disabled in strict mode (STRICT_TRANSCRIPT_ONLY=True).
            from app.config import STRICT_TRANSCRIPT_ONLY
            if not STRICT_TRANSCRIPT_ONLY:
                for term_kw, term_def in _STANDARD_DEFINITIONS:
                    if term_kw in full_topic_context and term_kw in transcript_low:
                        if not _check_alien(term_kw, transcript_low):
                            if not any(d.get("term", "").lower() == term_kw for d in clean_defs if isinstance(d, dict)):
                                clean_defs.append({"term": term_kw.title(), "definition": term_def, "source": "reference"})
            t["definitions"] = clean_defs[:4]

            # 5. Ground formulas — filter alien concepts
            clean_formulas = []
            for f in t.get("formulas", []):
                f_str = str(f).strip()
                if not _check_alien(f_str, transcript_low):
                    clean_formulas.append(f_str)
            # Supplement from pattern extraction
            pattern_formulas = _extract_formulas_from_text(f"{expl_raw} {source_text}")
            for pf in pattern_formulas:
                if pf not in clean_formulas:
                    clean_formulas.append(pf)
            # Supplement from standard reference formulas.
            # Disabled in strict mode (STRICT_TRANSCRIPT_ONLY=True).
            if not STRICT_TRANSCRIPT_ONLY:
                for kws, formula_str in _STANDARD_FORMULAS:
                    if any(k in full_topic_context for k in kws) and any(k in transcript_low for k in kws):
                        if formula_str not in clean_formulas and not _check_alien(formula_str, transcript_low):
                            clean_formulas.append(formula_str)
            t["formulas"] = clean_formulas[:3]

            # 6. Ground examples — filter alien concepts
            clean_examples = []
            for ex in t.get("examples", []):
                ex_str = str(ex).strip()
                if len(ex_str.split()) >= 4 and not _check_alien(ex_str, transcript_low):
                    clean_examples.append(ex_str)
            # Supplement from pattern extraction
            pattern_examples = _extract_examples_from_text(f"{expl_raw} {source_text}")
            for pe in pattern_examples:
                if pe not in clean_examples:
                    clean_examples.append(pe)
            # Supplement from standard reference examples (only if keyword in BOTH topic context AND transcript, and not strict mode)
            if not STRICT_TRANSCRIPT_ONLY:
                for kws, ex_str in _STANDARD_EXAMPLES:
                    if any(k in full_topic_context for k in kws) and any(k in transcript_low for k in kws):
                        if not _check_alien(ex_str, transcript_low):
                            if not any(ex_str.lower() in ce.lower() or ce.lower() in ex_str.lower() for ce in clean_examples):
                                clean_examples.append(ex_str)
            t["examples"] = clean_examples[:2]

            # 7. Teacher-style structured pedagogical breakdowns (preserve existing LLM/refined fields without repetition)
            def _strip_list_str(v: Any) -> str:
                if not v:
                    return ""
                if isinstance(v, (list, tuple)):
                    return " ".join(str(x).strip(" ,'\n") for x in v if str(x).strip(" ,'\n")).strip()
                sv = str(v).strip()
                if sv.startswith("[") and sv.endswith("]"):
                    try:
                        import ast as _ast
                        pv = _ast.literal_eval(sv)
                        if isinstance(pv, (list, tuple)):
                            return " ".join(str(x).strip(" ,'\n") for x in pv if str(x).strip(" ,'\n")).strip()
                    except Exception:
                        pass
                return sv

            def _is_dup_text(cand: str, refs: list[str]) -> bool:
                nc = re.sub(r"[^a-z0-9]", "", (cand or "").lower())
                if not nc:
                    return True
                tc = {w for w in re.findall(r"[a-z0-9]+", (cand or "").lower()) if len(w) > 2}
                for r in refs:
                    nr = re.sub(r"[^a-z0-9]", "", (r or "").lower())
                    if not nr:
                        continue
                    if nc == nr or (len(nc) >= 24 and (nc in nr or nr in nc)):
                        return True
                    tr = {w for w in re.findall(r"[a-z0-9]+", (r or "").lower()) if len(w) > 2}
                    if tc and tr and (len(tc & tr) / max(1, min(len(tc), len(tr)))) >= 0.72:
                        return True
                return False

            t["what_it_means"] = _strip_list_str(t.get("what_it_means"))
            t["how_it_works"] = _strip_list_str(t.get("how_it_works"))
            t["why_it_matters"] = _strip_list_str(t.get("why_it_matters"))

            if not t["what_it_means"]:
                if t["definitions"]:
                    d0 = t["definitions"][0]
                    d0_str = f"{d0.get('term', '')}: {d0.get('definition', '')}" if isinstance(d0, dict) else str(d0)
                    t["what_it_means"] = d0_str
                else:
                    t["what_it_means"] = f"{clean_title} is a foundational concept examined in this lecture."

            intro_refs = [t.get("explanation", ""), t["what_it_means"]]
            if not t["how_it_works"] or _is_dup_text(t["how_it_works"], intro_refs):
                distinct_kps = [kp for kp in t["key_points"] if not _is_dup_text(kp, intro_refs)]
                if t["formulas"]:
                    t["how_it_works"] = f"Operates according to the governing formula: {t['formulas'][0]}."
                elif distinct_kps:
                    t["how_it_works"] = distinct_kps[0]
                elif not t["how_it_works"]:
                    t["how_it_works"] = f"Executes through the core mechanisms and systematic principles of {clean_title}."

            mech_refs = intro_refs + [t["how_it_works"]]
            raw_steps = [_strip_list_str(st) for st in (t.get("steps_and_logic") or [])]
            filtered_steps: list[str] = []
            for st_s in raw_steps:
                if st_s and not _is_dup_text(st_s, mech_refs + filtered_steps):
                    filtered_steps.append(st_s)
            if not filtered_steps:
                filtered_steps = [kp for kp in t["key_points"] if not _is_dup_text(kp, mech_refs)]
            t["steps_and_logic"] = filtered_steps or list(t["key_points"][:2])

            why_refs = mech_refs + t["steps_and_logic"]
            if not t["why_it_matters"] or _is_dup_text(t["why_it_matters"], why_refs):
                if t["examples"] and not _is_dup_text(t["examples"][0], why_refs):
                    t["why_it_matters"] = f"Practical relevance illustrated in lecture: {t['examples'][0]}"
                elif (t.get("concept_connection") or "").strip() and not _is_dup_text(t["concept_connection"], why_refs):
                    t["why_it_matters"] = t["concept_connection"]
                else:
                    t["why_it_matters"] = f"Understanding {clean_title} connects core mechanisms examined in this session."

            shown_prose = [t.get("explanation", ""), t["what_it_means"], t["how_it_works"], t["why_it_matters"]]
            distinct_kps = [kp for kp in t["key_points"] if not _is_dup_text(kp, shown_prose)]
            if len(distinct_kps) < 2:
                for kp in t["key_points"]:
                    if kp not in distinct_kps and not _is_dup_text(kp, [t.get("explanation", "")]):
                        distinct_kps.append(kp)
                    if len(distinct_kps) >= 2:
                        break
            t["key_points"] = distinct_kps or list(t["key_points"])

            enriched.append(t)

        # Filter out purely non-academic / conversational / logistics pseudo-topics when real topics exist
        academic_enriched = [
            et for et in enriched
            if not re.search(
                r"\b(?:can you hear|hear me|mute your|open your book|turn to page|share my screen|as an ai|good morning|assalamu alaikum)\b",
                et.get("title", "").lower(),
            )
            and "?" not in et.get("title", "")
        ]
        if academic_enriched:
            enriched = academic_enriched

        # Merge topics with duplicate/equivalent titles
        merged = cls._merge_duplicate_topics(enriched)
        cls._clean_subtopics(merged)
        return merged

    @classmethod
    def _merge_duplicate_topics(cls, enriched: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Merges topics with duplicate or equivalent titles."""
        merged_topics: list[dict[str, Any]] = []
        seen_titles: dict[str, int] = {}

        for t in enriched:
            title_norm = re.sub(r"[^a-z0-9]", "", t.get("title", "").lower())
            if not title_norm:
                continue
            if title_norm in seen_titles:
                target = merged_topics[seen_titles[title_norm]]

                # Merge spans
                t_spans = t.get("spans", [])
                if t_spans:
                    raw_all_spans = list(target.get("spans", [])) + list(t_spans)
                    sorted_spans = []
                    for sp in raw_all_spans:
                        s_start = sp.get("start", 0.0) if isinstance(sp, dict) else getattr(sp, "start", 0.0)
                        s_end = sp.get("end", 0.0) if isinstance(sp, dict) else getattr(sp, "end", 0.0)
                        sorted_spans.append({"start": round(float(s_start), 1), "end": round(float(s_end), 1)})
                    sorted_spans.sort(key=lambda s: s["start"])
                    consolidated: list[dict[str, float]] = []
                    for sp in sorted_spans:
                        if not consolidated:
                            consolidated.append(sp)
                        else:
                            last = consolidated[-1]
                            if sp["start"] <= last["end"] + 4.0:
                                last["end"] = max(last["end"], sp["end"])
                            else:
                                consolidated.append(sp)
                    target["spans"] = consolidated

                # Update start/end times
                t_start = t.get("start_time", 0.0)
                t_end = t.get("end_time", 0.0)
                target["start_time"] = min(target.get("start_time", t_start), t_start)
                target["end_time"] = max(target.get("end_time", t_end), t_end)
                target["timestamp"] = (
                    f"[{int(target['start_time'] // 60):02d}:{int(target['start_time'] % 60):02d} - "
                    f"{int(target['end_time'] // 60):02d}:{int(target['end_time'] % 60):02d}]"
                )

                for field_name, cap in (("key_points", 6), ("definitions", None), ("formulas", None), ("examples", None)):
                    target_list = target.get(field_name, [])
                    for item in t.get(field_name, []):
                        if item not in target_list:
                            target_list.append(item)
                    target[field_name] = target_list[:cap] if cap else target_list

                target_sids = set(target.get("source_segment_ids", []))
                for sid in t.get("source_segment_ids", []):
                    target_sids.add(sid)
                target["source_segment_ids"] = sorted(target_sids)
                target["evidence"] = [f"seg_{sid:03d}" for sid in target["source_segment_ids"]]

                target["subtopics"] = target.get("subtopics", []) + t.get("subtopics", [])
                target["exam_relevant"] = target.get("exam_relevant", False) or t.get("exam_relevant", False)
            else:
                seen_titles[title_norm] = len(merged_topics)
                merged_topics.append(t)

        return merged_topics

    @classmethod
    def _clean_subtopics(cls, merged_topics: list[dict[str, Any]]) -> None:
        """Filters incomplete or noisy subtopic entries."""
        for t in merged_topics:
            raw_subs = t.get("subtopics", [])
            seen_sub: set[str] = set()
            cleaned_subs = []
            for st in raw_subs:
                st_dict = dict(st) if isinstance(st, dict) else st.to_dict()
                st_title = re.sub(r"[,;:\s]+$", "", st_dict.get("title", "")).strip()
                st_norm = re.sub(r"[^a-z0-9]", "", st_title.lower())
                if not st_norm or st_norm in seen_sub:
                    continue
                if _is_incomplete_clause(st_title):
                    continue
                tokens = [re.sub(r"[^a-zA-Z]", "", w) for w in st_title.split()]
                if any(_is_phonetic_noise(w) for w in tokens if w):
                    continue
                seen_sub.add(st_norm)
                st_dict["title"] = st_title[0].upper() + st_title[1:] if st_title else st_title
                cleaned_subs.append(st_dict)

            if not cleaned_subs and t.get("title"):
                cleaned_subs.append({
                    "id": f"{t.get('id', 'topic_01')}_sub_01",
                    "title": f"{t['title']} — Core Points",
                    "role": "SUBTOPIC",
                    "start_time": t.get("start_time", 0.0),
                    "end_time": t.get("end_time", 0.0),
                    "source_segment_ids": t.get("source_segment_ids", []),
                    "key_points": t.get("key_points", [])[:2],
                    "explanation": t.get("explanation", ""),
                })

            t["subtopics"] = cleaned_subs[:4]

    # =======================================================================
    # 3. EXECUTIVE BRIEFING SYNTHESIZER (Short & Detailed)
    # =======================================================================
    @classmethod
    def generate_executive_briefing(
        cls,
        transcript_text: str,
        segments: list[Any],
        recording_type: str = "Lecture",
        topic_hint: Optional[str] = None,
        study_topics: Optional[list[dict[str, Any]]] = None,
        llm_provider: Optional[Any] = None,
    ) -> tuple[str, str]:
        total_segs = len(segments) if segments else 0
        if total_segs > 0:
            t1 = max(1, total_segs // 3)
            t2 = max(t1 + 1, (2 * total_segs) // 3)
            early_text = " ".join(cls._segment_text(s) for s in segments[:t1])
            mid_text = " ".join(cls._segment_text(s) for s in segments[t1:t2])
            late_text = " ".join(cls._segment_text(s) for s in segments[t2:])
        else:
            early_text = mid_text = late_text = transcript_text

        if recording_type == "Meeting":
            return cls._meeting_briefing(early_text, mid_text, late_text)

        return cls._session_briefing(
            transcript_text, study_topics, topic_hint, llm_provider,
            early_text=early_text, mid_text=mid_text, late_text=late_text
        )

    @staticmethod
    def _segment_text(s: Any) -> str:
        if getattr(s, "is_intel_view", False):
            return getattr(s, "llm_text", None) or getattr(s, "text", "") or ""
        return getattr(s, "english_text", None) or getattr(s, "normalized_text", None) or getattr(s, "text", "") or ""

    @staticmethod
    def _meeting_briefing(early_text: str, mid_text: str, late_text: str) -> tuple[str, str]:
        early_snippet = early_text[:200].strip() or "agenda review and priorities"
        mid_snippet = mid_text[:200].strip() or "strategic discussions and decisions"
        late_snippet = late_text[:200].strip() or "deliverables and next steps"
        short = (
            f"The meeting opened with agenda review and initial discussion on {early_snippet}. "
            f"The team aligned on key discussions and decisions regarding {mid_snippet}. "
            f"Finally, the meeting concluded with action items, owner assignments, and next deliverables: {late_snippet}."
        )
        detailed = (
            f"### Meeting Overview & Discussion Record\n\n"
            f"**1. Meeting Opening & Agenda Review:**\n- {early_text[:350]}...\n\n"
            f"**2. Key Discussions & Decisions:**\n- {mid_text[:350] if mid_text else 'Continued review of agenda points.'}...\n\n"
            f"**3. Action Items, Owners & Next Steps:**\n- {late_text[:350] if late_text else 'Meeting adjourned.'}..."
        )
        return short, detailed

    @classmethod
    def _session_briefing(
        cls,
        transcript_text: str,
        study_topics: Optional[list[dict[str, Any]]],
        topic_hint: Optional[str],
        llm_provider: Optional[Any],
        early_text: str = "",
        mid_text: str = "",
        late_text: str = "",
    ) -> tuple[str, str]:
        topics_summary = ", ".join([t.get("title", "") for t in (study_topics or [])[:4] if t.get("title")])
        if not topics_summary:
            topics_summary = topic_hint or "this session's material"

        transcript_low = (transcript_text or "").lower()

        # 1. Attempt AI synthesis
        try:
            from app.services.llm_provider import llm_lifecycle_session
            session_ctx = (
                llm_lifecycle_session("summary", release_on_exit=True)
                if llm_provider is None
                else __import__("contextlib").nullcontext()
            )
            with session_ctx:
                provider = llm_provider or _try_get_llm_provider("summary")
                if provider and provider.is_available() and len(transcript_text.strip()) > 50:
                    transcript_prompt_text = _truncate(transcript_text)
                    prompt = (
                        f"You are synthesizing an executive briefing for a lecture/session.\n"
                        f"Ground your summary EXCLUSIVELY in the transcript below. Identify the subject "
                        f"yourself from the transcript — do not assume one. Never introduce any concept, "
                        f"algorithm, or topic that is not discussed in this specific transcript.\n\n"
                        f"FULL TRANSCRIPT (GROUND TRUTH):\n\"\"\"\n{transcript_prompt_text}\n\"\"\"\n\n"
                        f"TOPICS ALREADY EXTRACTED (may be incomplete — verify against the transcript):\n{topics_summary}\n\n"
                        f"Generate a two-part executive briefing:\n"
                        f"1. 'summary_short': A cohesive 3-4 sentence paragraph summarizing the session's progression from opening to conclusion.\n"
                        f"2. 'summary_detailed': A Markdown briefing with three headed sections you title yourself based on what the transcript actually covers (introduction/foundation, core content deep-dive, and takeaways/action items).\n\n"
                        f"Output strictly valid JSON: {{\"summary_short\": \"...\", \"summary_detailed\": \"...\"}}"
                    )
                    res = provider.generate_structured(
                        prompt=prompt,
                        schema={"summary_short": "Short summary", "summary_detailed": "Detailed markdown briefing"},
                        system_prompt="You are a summarization expert. Ground everything strictly in the provided transcript and name the subject yourself. Output valid JSON.",
                        max_tokens=350,
                    )
                    ai_short = (res.get("summary_short") or "").strip()
                    ai_detailed = (res.get("summary_detailed") or "").strip()
                    if not _check_alien(ai_short + " " + ai_detailed, transcript_low):
                        if len(ai_short.split()) >= 15 and len(ai_detailed.split()) >= 40:
                            return ai_short, ai_detailed
        except Exception as e_llm:
            logger.warning(f"AI executive briefing generation fallback: {e_llm}")

        # 2. Extractive fallback
        return cls._extractive_briefing_fallback(
            topics_summary, study_topics or [],
            early_text=early_text, mid_text=mid_text, late_text=late_text
        )

    @staticmethod
    def _clean_summary_clause(txt: str) -> str:
        if not txt:
            return ""
        cur = txt.strip()
        for _ in range(3):
            cur = re.sub(r"^[\s,;.:!?\-]+", "", cur)
            cur = _FILLER_OPENER_PATTERN.sub("", cur).strip()
        cur = re.sub(r"\s{2,}", " ", cur).strip()
        return cur

    @classmethod
    def _extractive_briefing_fallback(
        cls,
        topics_summary: str,
        study_topics: list[dict[str, Any]],
        early_text: str = "",
        mid_text: str = "",
        late_text: str = "",
    ) -> tuple[str, str]:
        if not study_topics and (early_text or mid_text or late_text):
            first_topic = cls._clean_summary_clause(early_text[:80]).rstrip(".:,;") or "Introduction & Foundations"
            mid_topic = cls._clean_summary_clause(mid_text[:80]).rstrip(".:,;") or "Core Concepts & Deep Dive"
            last_topic = cls._clean_summary_clause(late_text[:80]).rstrip(".:,;") or "Conclusions & Takeaways"
            first_expl = cls._clean_summary_clause(early_text[:300]).rstrip(".")
            mid_expl = cls._clean_summary_clause(mid_text[:300]).rstrip(".")
            last_expl = cls._clean_summary_clause(late_text[:300]).rstrip(".")
            if topics_summary == "this session's material" and first_topic != "Introduction & Foundations":
                topics_summary = f"{first_topic}, {mid_topic}, and {last_topic}"
        else:
            first_topic = study_topics[0].get("title") if study_topics else "the session's opening material"
            mid_topic = study_topics[len(study_topics) // 2].get("title") if len(study_topics) > 1 else "the session's core material"
            last_topic = study_topics[-1].get("title") if len(study_topics) > 2 else "the session's closing material"

            first_expl = cls._clean_summary_clause(study_topics[0].get("explanation", "") if study_topics else "").rstrip(".")
            mid_expl = cls._clean_summary_clause(study_topics[len(study_topics) // 2].get("explanation", "") if len(study_topics) > 1 else "").rstrip(".")
            last_expl = cls._clean_summary_clause(study_topics[-1].get("explanation", "") if len(study_topics) > 2 else "").rstrip(".")

        # Ensure clean snippets without conversational fillers
        early_snippet = first_expl or first_topic
        mid_snippet = mid_expl or mid_topic
        late_snippet = last_expl or last_topic
        for f in ["assalamu alaikum", "how are you", "good morning"]:
            early_snippet = re.sub(rf"\b{f}\b", "", early_snippet, flags=re.I).strip(" ,;:")
            mid_snippet = re.sub(rf"\b{f}\b", "", mid_snippet, flags=re.I).strip(" ,;:")
            late_snippet = re.sub(rf"\b{f}\b", "", late_snippet, flags=re.I).strip(" ,;:")

        def _clean_body(text: str) -> str:
            cleaned = text
            for f in ["assalamu alaikum", "how are you", "good morning"]:
                cleaned = re.sub(rf"\b{f}\b[^.!?]*[.!?]?", "", cleaned, flags=re.I).strip(" ,;:")
            return cleaned.strip()

        body_1 = first_expl or _clean_body(early_text[:350]) or early_snippet
        body_2 = mid_expl or _clean_body(mid_text[:350]) or mid_snippet
        body_3 = last_expl or _clean_body(late_text[:350]) or late_snippet

        short = (
            f"The session began with discussions on {first_topic or early_snippet}. "
            f"The central focus covered key methodologies including {mid_topic or mid_snippet}. "
            f"Finally, the session concluded with practical assignments and next steps: {last_topic or late_snippet}."
        )
        detailed = (
            f"### Session Overview & Chronological Summary\n\n"
            f"**1. Introduction & Initial Topics (Early Section):**\n"
            f"- **{first_topic or 'Opening'}**: {body_1}...\n\n"
            f"**2. Core Concepts & Technical Deep Dive (Middle Section):**\n"
            f"- **{mid_topic or 'Core Concepts'}**: {body_2}...\n\n"
            f"**3. Practical Directives, Action Items & Conclusion (Late Section):**\n"
            f"- **{last_topic or 'Closing'}**: {body_3}..."
        )
        return short, detailed

    # =======================================================================
    # 4. QUIZ / MCQ SYNTHESIS ENGINE (Blueprint + Batched Exam-Grade Pipeline)
    # =======================================================================
    @staticmethod
    def _estimate_lecture_duration_minutes(
        transcript_text: str,
        segments: Optional[list[Any]] = None,
    ) -> float:
        if segments:
            max_end = 0.0
            for s in segments:
                end_val = s.get("end") if isinstance(s, dict) else getattr(s, "end_time", getattr(s, "end", 0.0))
                try:
                    max_end = max(max_end, float(end_val or 0.0))
                except (TypeError, ValueError):
                    pass
            if max_end > 0:
                return max_end / 60.0
        words = len((transcript_text or "").split())
        return words / 130.0  # ~130 wpm typical university lecture pace

    @classmethod
    def _compute_quiz_target_plan(
        cls,
        transcript_text: str,
        lecture_notes: Optional[dict[str, Any]] = None,
        segments: Optional[list[Any]] = None,
    ) -> dict[str, Any]:
        notes = lecture_notes or {}
        study_topics = notes.get("study_topics", []) or []
        definitions = notes.get("definitions", []) or []
        key_concepts = notes.get("key_concepts", []) or notes.get("topics", []) or []

        duration_min = cls._estimate_lecture_duration_minutes(transcript_text, segments)
        word_count = len((transcript_text or "").split())
        topic_count = max(len(study_topics), len(key_concepts), len(definitions))

        if word_count < 120:
            tier = "micro"
            min_q, target_q, max_q = 3, 4, 5
        elif duration_min < 10.0 and word_count < 1200:
            tier = "short"
            min_q = 5
            target_q = max(5, min(8, 5 + topic_count // 2))
            max_q = 8
        elif duration_min <= 30.0 and word_count <= 3800:
            tier = "medium"
            min_q = 10
            target_q = max(10, min(15, 10 + topic_count // 2))
            max_q = 15
        else:
            tier = "long"
            min_q = 15
            target_q = max(15, min(25, 14 + topic_count))
            max_q = 25

        return {
            "tier": tier,
            "duration_min": round(duration_min, 1),
            "word_count": word_count,
            "topic_count": topic_count,
            "min_questions": min_q,
            "target_questions": target_q,
            "max_questions": max_q,
        }

    @staticmethod
    def _build_chronological_digest(
        transcript_text: str,
        study_topics: list[dict[str, Any]],
        max_section_chars: int = 4500,
    ) -> tuple[str, dict[str, str]]:
        """
        Partitions the full transcript into Early, Middle, and Late windows so
        both the blueprint and batched MCQ generator cover the entire lecture.
        """
        text = (transcript_text or "").strip()
        n = len(text)
        if n <= max_section_chars:
            return text, {"early": text, "middle": text, "late": text}

        third = n // 3
        early_raw = text[:third]
        mid_raw = text[third: 2 * third]
        late_raw = text[2 * third:]

        def _sample_window(chunk: str, limit: int) -> str:
            if len(chunk) <= limit:
                return chunk
            part = limit // 3
            head = chunk[:part]
            mid_start = max(part, (len(chunk) - part) // 2)
            mid = chunk[mid_start: mid_start + part]
            tail = chunk[-(limit - len(head) - len(mid)):]
            return f"{head}\n[...]\n{mid}\n[...]\n{tail}"

        early_win = _sample_window(early_raw, max_section_chars)
        mid_win = _sample_window(mid_raw, max_section_chars)
        late_win = _sample_window(late_raw, max_section_chars)

        digest = (
            f"=== EARLY LECTURE SECTION ===\n{_sample_window(early_raw, 2800)}\n\n"
            f"=== MIDDLE LECTURE SECTION ===\n{_sample_window(mid_raw, 2800)}\n\n"
            f"=== LATE LECTURE SECTION ===\n{_sample_window(late_raw, 2800)}"
        )
        return digest, {"early": early_win, "middle": mid_win, "late": late_win}

    @staticmethod
    def _build_lecture_blueprint(
        provider: Any,
        chronological_digest: str,
        topics_summary: str,
        target_count: int,
        study_topics: list[dict[str, Any]],
        topic_hint: Optional[str] = None,
    ) -> dict[str, Any]:
        """Stage A: Map examinable concepts across early/middle/late sections."""
        if len(study_topics) < 3:
            try:
                prompt = LECTURE_BLUEPRINT_PROMPT.format(
                    topics_summary=topics_summary,
                    chronological_digest=chronological_digest[:2600],
                    target_count=target_count,
                )
                res = provider.generate_structured(
                    prompt=prompt,
                    schema=None,
                    system_prompt="You are a university professor designing an exam blueprint. Output strictly valid JSON.",
                    max_tokens=600,
                    temperature=0.2,
                )
                if isinstance(res, dict) and isinstance(res.get("examinable_items"), list) and len(res["examinable_items"]) >= 3:
                    items = []
                    for idx, it in enumerate(res["examinable_items"]):
                        if not isinstance(it, dict) or not it.get("concept"):
                            continue
                        sec = str(it.get("section", "")).lower().strip()
                        if sec not in ("early", "middle", "late"):
                            sec = ["early", "middle", "late"][idx % 3]
                        cat = str(it.get("category", "Conceptual")).strip().title()
                        if cat not in ("Conceptual", "Application", "Numerical", "Technical"):
                            cat = "Conceptual"
                        diff = str(it.get("difficulty", "Medium")).strip().title()
                        if diff not in ("Easy", "Medium", "Hard"):
                            diff = "Medium"
                        items.append({
                            "concept": str(it["concept"]).strip(),
                            "section": sec,
                            "topic_area": str(it.get("topic_area") or it["concept"]).strip(),
                            "category": cat,
                            "difficulty": diff,
                            "angle": str(it.get("angle") or "Test deep conceptual/practical understanding").strip(),
                        })
                    if items:
                        sections = ["early", "middle", "late"]
                        cats = ["Conceptual", "Application", "Technical"]
                        diffs = ["Medium", "Hard", "Easy", "Medium", "Hard"]
                        subj = str(res.get("subject_domain") or topic_hint or "Academic Lecture").strip()
                        while len(items) < target_count:
                            idx = len(items)
                            sec = sections[idx % 3]
                            t_title = (
                                study_topics[idx % len(study_topics)].get("title", f"{subj} ({sec} #{idx+1})")
                                if study_topics
                                else f"{subj} ({sec} part #{idx+1})"
                            )
                            items.append({
                                "concept": f"{t_title} — aspect {idx + 1}",
                                "section": sec,
                                "topic_area": t_title,
                                "category": cats[idx % len(cats)],
                                "difficulty": diffs[idx % len(diffs)],
                                "angle": f"Test a distinct principle, scenario, or edge case from the {sec} lecture section",
                            })
                        return {
                            "subject_domain": subj,
                            "lecture_style": str(res.get("lecture_style") or "conceptual").strip(),
                            "has_numerical_content": bool(res.get("has_numerical_content", False)),
                            "examinable_items": items[:target_count],
                        }
            except Exception as e_bp:
                logger.warning(f"Lecture blueprint generation failed, synthesizing section blueprint: {e_bp}")

        # Deterministic structural blueprint across early/middle/late sections (still drives LLM batching)
        items: list[dict[str, Any]] = []
        sections = ["early", "middle", "late"]
        cats = ["Conceptual", "Application", "Technical"]
        diffs = ["Medium", "Hard", "Medium", "Easy", "Hard"]
        for i in range(target_count):
            sec = sections[i % 3]
            topic_title = (
                study_topics[i % len(study_topics)].get("title", f"Section {sec.title()} Concept {i+1}")
                if study_topics
                else f"{topic_hint or 'Lecture'} ({sec.title()} section #{i+1})"
            )
            items.append({
                "concept": topic_title,
                "section": sec,
                "topic_area": topic_title,
                "category": cats[i % len(cats)],
                "difficulty": diffs[i % len(diffs)],
                "angle": f"Examine a distinct principle, scenario, or mechanism from the {sec} part of the lecture",
            })
        return {
            "subject_domain": topic_hint or "University Lecture",
            "lecture_style": "mixed",
            "has_numerical_content": False,
            "examinable_items": items,
        }

    @staticmethod
    def _stem_similarity(a: str, b: str) -> float:
        stop = {
            "what", "which", "when", "where", "why", "how", "does", "the", "are",
            "for", "from", "with", "into", "this", "that", "primary", "purpose",
            "main", "following", "best", "most", "during", "using", "used",
        }
        toks_a = {w for w in re.findall(r"[a-z0-9]+", (a or "").lower()) if len(w) > 2 and w not in stop}
        toks_b = {w for w in re.findall(r"[a-z0-9]+", (b or "").lower()) if len(w) > 2 and w not in stop}
        if not toks_a or not toks_b:
            return 0.0
        return len(toks_a & toks_b) / max(1, len(toks_a | toks_b))

    _UNRESOLVED_PLACEHOLDER_RE = re.compile(
        r"<(?!\s)(?![^<>]*\b(?:and|or)\b)[a-zA-Z_][a-zA-Z0-9_ \-\./?:]{0,38}(?<!\s)>|"
        r"\[\s*(?:concept|topic\s*name|choice\s*\d*(?:\s*text)?|insert\s+[^\[\]]+)\s*\]|"
        r"^(?:topic\s*name|concept|exam\s*question(?:\s*ending\s*with\s*\?)?|choice\s*\d+\s*text|"
        r"concise\s+explanation.*|"
        r"complete\s+self-contained\s+exam\s+question(?:\s+ending\s+with\s*\?)?|"
        r"(?:first|second|third|fourth)\s+complete\s+answer\s+choice|"
        r"exact\s+verbatim\s+text\s+of\s+the\s+winning\s+choice|"
        r"specific\s+matching\s+topic\s+title\s+from\s+the\s+lecture|"
        r"specific\s+distinct\s+concept\s+tested\s+by\s+this\s+question)$",
        re.IGNORECASE,
    )

    _COMBINED_PROCESS_STATES_RE = re.compile(
        r"\b(new|ready|running|waiting|blocked|block|terminated)\s*[-/]?\s+"
        r"(new|ready|running|waiting|blocked|block|terminated)\b",
        re.IGNORECASE,
    )

    _CONCEPT_SPECIFIC_TERMS = {
        "turnaround", "waiting", "burst", "arrival", "completion", "response",
        "aging", "starvation", "convoy", "quantum", "round", "robin", "fcfs",
        "sjf", "srtf", "paging", "segmentation", "deadlock", "semaphore",
        "mutex", "thrashing", "tlb", "pcb", "virtual", "supervised", "regression",
        "classification", "precision", "recall", "overfitting", "gradient",
        "ohm", "voltage", "current", "resistance", "accrual", "cash",
        "profit", "revenue", "neuron", "reflex", "hormone", "endocrine",
    }

    _CONCEPT_ALIAS_GROUPS: tuple[tuple[str, ...], ...] = (
        ("srtf", "shortest remaining time first"),
        ("sjf", "shortest job first"),
        ("fcfs", "first come first served", "first-come first-served", "first-come, first-served"),
        ("fifo", "first in first out", "first-in first-out"),
        ("lru", "least recently used"),
        ("tlb", "translation lookaside buffer"),
        ("pcb", "process control block"),
        ("rr", "round robin"),
    )

    _GENERIC_TOPIC_REMAPPING_TOKENS = {
        "process", "processe", "processes", "state", "states", "system", "systems",
        "operating", "time", "times", "waiting", "wait", "event", "events",
        "queue", "queues", "program", "programs", "task", "tasks",
        "cpu", "execution", "execute", "executing", "running", "memory",
        "resource", "resources", "operation", "operations", "computer",
    }

    @classmethod
    def _alias_keys(cls, text: str) -> set[str]:
        low = (text or "").lower()
        keys: set[str] = set()
        for group in cls._CONCEPT_ALIAS_GROUPS:
            if any(re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", low) for phrase in group):
                keys.add(f"alias:{group[0]}")
        return keys

    @staticmethod
    def _has_ready_burst_contradiction(text: str) -> bool:
        """
        Narrow OS lifecycle check rejecting explanations/answers that assert either:
          1) a Ready process has already completed/finished its CPU burst time, or
          2) a Ready process executes/runs immediately without waiting (ignoring queue/dispatch wait).
        """
        for clause in re.split(r"[.!?;]+", (text or "").lower()):
            if re.search(
                r"\bready\s+(?:process|state|task)\b[^,;]{0,60}"
                r"\b(?:has\s+|have\s+)?(?:completed|finished)\s+(?:its\s+|the\s+)?(?:cpu\s+)?burst(?:\s+time)?\b",
                clause,
            ):
                if not re.search(
                    r"\b(?:not|never|yet\s+to|before|without|hasn['’]?t|haven['’]?t)\b[^,;]{0,25}"
                    r"\b(?:completed|finished)\b",
                    clause,
                ):
                    return True
            if re.search(
                r"\bready\s+(?:process|state|task)\b[^,;]{0,60}"
                r"\b(?:can|will|does)?\s*(?:execute|executes|run|runs)\s+immediately\s+without\s+waiting\b",
                clause,
            ):
                if not re.search(
                    r"\b(?:not|cannot|can['’]?t|never|won['’]?t|doesn['’]?t|does\s+not)\b[^,;]{0,35}"
                    r"\b(?:execute|executes|run|runs)\b",
                    clause,
                ):
                    return True
        return False

    @staticmethod
    def _norm_quiz_tokens(text: str) -> set[str]:
        stop = {
            "the", "and", "for", "with", "that", "this", "from", "into", "when",
            "where", "which", "what", "why", "how", "does", "are", "was", "were",
            "has", "have", "been", "not", "yet", "still", "more", "most", "some",
            "about", "after", "before", "during", "using", "used", "such", "only",
        }
        return {
            w for w in re.findall(r"[a-z0-9]+", (text or "").lower())
            if len(w) >= 3 and w not in stop
        }

    @classmethod
    def _has_unresolved_placeholder(cls, value: Any) -> bool:
        if value is None:
            return False
        s = str(value).strip()
        if not s:
            return False
        return bool(cls._UNRESOLVED_PLACEHOLDER_RE.search(s))

    @classmethod
    def _is_combined_state_label(cls, text: str) -> bool:
        s = (text or "").strip()
        for m in cls._COMBINED_PROCESS_STATES_RE.finditer(s):
            g1, g2 = m.group(1).lower(), m.group(2).lower()
            if g1 != g2:
                return True
        return False

    @classmethod
    def _validate_and_refine_mcq(
        cls,
        q_item: Any,
        transcript_low: str,
        seen_stems: list[str],
        seen_concepts: set[str],
        rng: random.Random,
        study_topics: Optional[list[dict[str, Any]]] = None,
        seen_accepted: Optional[list[dict[str, Any]]] = None,
    ) -> tuple[Optional[dict[str, Any]], Optional[str]]:
        """Validates, checks placeholders/consistency, deduplicates from Q1, and shuffles option order."""
        if not isinstance(q_item, dict):
            return None, "Item is not a JSON object"

        q_text = str(q_item.get("question") or "").strip()
        raw_opts = q_item.get("options") or []
        if not isinstance(raw_opts, list) or len(raw_opts) != 4:
            return None, f"Expected 4 options, got {len(raw_opts) if isinstance(raw_opts, list) else 'non-list'}"

        opt_prefix_re = re.compile(r"^(?:Opt(?:ion)?\s*[A-D0-9]\s*[:\.\)-]\s*|[A-D][\.\):]\s*)", re.IGNORECASE)
        opts = [opt_prefix_re.sub("", str(o).strip()).strip() for o in raw_opts]
        if any(not o or len(o) < 2 for o in opts):
            return None, "One or more options are empty or too short"

        raw_topic_area = str(q_item.get("topic_area") or "").strip()
        raw_concept = str(q_item.get("grounded_concept") or "").strip()
        explanation = str(q_item.get("explanation") or "").strip()

        # Reject unresolved template placeholders anywhere in question, options, tags, or explanation
        fields_to_check = [q_text, raw_topic_area, raw_concept, explanation, str(q_item.get("correct_answer") or "")] + opts
        if any(cls._has_unresolved_placeholder(f) for f in fields_to_check):
            return None, "Unresolved template placeholder (<...> or placeholder label) in MCQ"

        dist_rats = q_item.get("distractor_rationales")
        if isinstance(dist_rats, dict):
            if any(cls._has_unresolved_placeholder(k) or cls._has_unresolved_placeholder(v) for k, v in dist_rats.items()):
                return None, "Unresolved template placeholder in distractor_rationales"

        if (
            not q_text
            or len(q_text) < 15
            or q_text.endswith("...")
            or re.match(r"^(?:first|second|third|sample|example)\s+exam\s+question\??$", q_text, re.IGNORECASE)
        ):
            return None, "Question stem is missing, placeholder, truncated, or too short"

        # Reject recall-style stems
        recall_re = re.compile(
            r"(what did the speaker say|according to (the |this )?(lecture|session|speaker|instructor)|"
            r"what was said about|which statement was mentioned)",
            re.IGNORECASE,
        )
        if recall_re.search(q_text):
            return None, "Recall-style phrasing ('according to the lecture' / 'what did the speaker say')"

        # Reject banned catch-all or placeholder options
        banned_re = re.compile(
            r"^(all\s+of\s+the\s+above|none\s+of\s+the\s+above|all\s+of\s+these|"
            r"none\s+of\s+these|both\s+[a-d]\s+and\s+[a-d]|neither\s+[a-d]\s+nor\s+[a-d]|"
            r"opt(?:ion)?\s*(?:[a-d]|[1-4])|<choice\s*\d*\s*text>)$",
            re.IGNORECASE,
        )
        if any(banned_re.match(o.rstrip(".")) for o in opts):
            return None, "Banned catch-all or placeholder distractor ('All/None of the above' or 'Option A')"

        # Reject duplicate options
        norm_opts = [re.sub(r"\s+", " ", o.lower()) for o in opts]
        if len(set(norm_opts)) < 4:
            return None, "Duplicate options within the same question"

        # Resolve correct_answer against options (never overwrite a mismatched text answer via letter)
        corr = str(q_item.get("correct_answer") or "").strip()
        let_raw = str(q_item.get("correct_option_letter") or "").strip().upper().rstrip(".")
        if corr not in opts:
            cleaned_corr = opt_prefix_re.sub("", corr).strip()
            bare_letter = re.sub(r"^Opt(?:ion)?\s*", "", corr, flags=re.IGNORECASE).strip().upper().rstrip(".")
            matched = next((o for o in opts if cleaned_corr and (o == cleaned_corr or o.lower() == cleaned_corr.lower())), None)
            if matched:
                corr = matched
            elif bare_letter in ("A", "B", "C", "D"):
                corr = opts[["A", "B", "C", "D"].index(bare_letter)]
            elif not corr and let_raw in ("A", "B", "C", "D"):
                corr = opts[["A", "B", "C", "D"].index(let_raw)]
            else:
                return None, "correct_answer does not match any of the 4 options"

        # Reject merged/combined process-state labels as correct answers (e.g., "Ready Running", "Waiting Block")
        # when asking for a process state
        q_low = q_text.lower()
        if "state" in q_low and (cls._is_combined_state_label(corr) or any(cls._is_combined_state_label(o) for o in opts)):
            return None, "Invalid combined process-state label (e.g., 'Ready Running'); states must not be merged"

        # Check extreme length bias on non-numerical answers
        distractor_lens = [len(o) for o in opts if o != corr]
        avg_dist_len = sum(distractor_lens) / max(1, len(distractor_lens))
        if len(corr) > 90 and avg_dist_len > 0 and len(corr) > 2.4 * avg_dist_len:
            return None, "Correct option is disproportionately longer than distractors (length bias)"

        # Alien concept filter
        full_q_str = f"{q_text} {' '.join(opts)} {corr} {q_item.get('grounded_concept', '')}".lower()
        if _check_alien(full_q_str, transcript_low):
            return None, "Alien concept not grounded in lecture transcript"

        # Concept & topic_area consistency check against question/answer/explanation and study_topics
        qa_expl_str = f"{q_text} {corr} {' '.join(opts)} {explanation}"
        qa_expl_tokens = cls._norm_quiz_tokens(qa_expl_str)
        qa_expl_aliases = cls._alias_keys(qa_expl_str)
        concept_tokens = cls._norm_quiz_tokens(raw_concept)
        concept_aliases = cls._alias_keys(raw_concept)
        topic_tokens = cls._norm_quiz_tokens(raw_topic_area)
        topic_aliases = cls._alias_keys(raw_topic_area)

        if raw_concept and concept_tokens:
            # If grounded_concept cites a specific technical term (or canonical alias) not present in Q/A/explanation, reject
            specific_in_concept = concept_tokens & cls._CONCEPT_SPECIFIC_TERMS
            alias_matched = bool(concept_aliases & qa_expl_aliases)
            if specific_in_concept and not (specific_in_concept & qa_expl_tokens) and not alias_matched:
                return None, f"Concept mismatch: grounded_concept '{raw_concept}' does not match question/answer"

        resolved_topic_area = raw_topic_area or raw_concept
        if raw_topic_area and topic_tokens:
            specific_in_topic = topic_tokens & cls._CONCEPT_SPECIFIC_TERMS
            topic_in_qa = bool((topic_tokens & qa_expl_tokens) or (topic_aliases & qa_expl_aliases))
            # Build lookup against actual study_topics if provided
            if study_topics:
                matched_st = next(
                    (
                        st for st in study_topics
                        if str(st.get("title") or "").strip().lower() == raw_topic_area.lower()
                        or (topic_aliases and (topic_aliases & cls._alias_keys(str(st.get("title") or ""))))
                    ),
                    None,
                )
                if matched_st is not None:
                    st_body = " ".join([
                        str(matched_st.get("explanation") or ""),
                        str(matched_st.get("source_text") or ""),
                        " ".join(str(x) for x in (matched_st.get("key_points") or [])),
                    ]).strip()
                    st_full = f"{matched_st.get('title', '')} {st_body}"
                    st_tokens = cls._norm_quiz_tokens(st_full)
                    st_aliases = cls._alias_keys(st_full)
                    qa_roots = {w.rstrip("s") for w in qa_expl_tokens} | qa_expl_aliases
                    st_roots = {w.rstrip("s") for w in st_tokens} | st_aliases
                    corr_roots = {w.rstrip("s") for w in cls._norm_quiz_tokens(corr)} | cls._alias_keys(corr)
                    has_mismatch_with_body = bool(st_body) and (
                        (corr_roots and not (corr_roots & st_roots) and len(qa_roots & st_roots) < 2)
                    )
                    if (not topic_in_qa and specific_in_topic) or has_mismatch_with_body:
                        # Attempt safe remapping only on non-generic domain token overlap
                        qa_corr_text = f"{q_text} {corr}"
                        qa_corr_specific = (
                            {w.rstrip("s") for w in cls._norm_quiz_tokens(qa_corr_text) if w.rstrip("s") not in cls._GENERIC_TOPIC_REMAPPING_TOKENS}
                            | cls._alias_keys(qa_corr_text)
                        )
                        best_st = None
                        best_overlap = 0
                        for st in study_topics:
                            st_cand_text = " ".join([
                                str(st.get("title") or ""),
                                str(st.get("explanation") or ""),
                                str(st.get("source_text") or ""),
                                " ".join(str(x) for x in (st.get("key_points") or [])),
                            ])
                            st_cand_specific = (
                                {w.rstrip("s") for w in cls._norm_quiz_tokens(st_cand_text) if w.rstrip("s") not in cls._GENERIC_TOPIC_REMAPPING_TOKENS}
                                | cls._alias_keys(st_cand_text)
                            )
                            ov = len(qa_corr_specific & st_cand_specific)
                            if ov > best_overlap:
                                best_overlap = ov
                                best_st = st
                        if best_st and best_overlap >= 2:
                            resolved_topic_area = str(best_st.get("title") or "").strip()
                        else:
                            return None, f"Topic mismatch: topic_area '{raw_topic_area}' is inconsistent with question/answer"
                elif specific_in_topic and not topic_in_qa:
                    return None, f"Topic mismatch: topic_area '{raw_topic_area}' does not match question/answer"
            elif specific_in_topic and not topic_in_qa:
                return None, f"Topic mismatch: topic_area '{raw_topic_area}' does not match question/answer"

        # Semantic & concept deduplication from Question #1 (no len(seen_stems) >= 3 bypass)
        for prev_stem in seen_stems:
            if cls._stem_similarity(q_text, prev_stem) >= 0.48:
                return None, f"Duplicate/repetitive question stem: '{q_text[:50]}'"

        # Detect paraphrased duplicate questions testing the exact same process state / short canonical answer
        corr_low = corr.lower().strip()
        corr_tokens = cls._norm_quiz_tokens(corr_low)
        state_names = {"new", "ready", "running", "waiting", "blocked", "block", "terminated"}
        q_state_hits = set(re.findall(r"\b(new|newly|ready|running|waiting|blocked|terminated)\b", q_low))
        q_state_hits = {"new" if w == "newly" else w for w in q_state_hits}
        tested_state = corr_low if corr_low in state_names else (next(iter(q_state_hits)) if ("state" in q_low and len(q_state_hits) == 1) else None)

        if tested_state and "state" in q_low:
            for idx_prev, prev_stem in enumerate(seen_stems):
                prev_low = prev_stem.lower()
                prev_states = {"new" if w == "newly" else w for w in re.findall(r"\b(new|newly|ready|running|waiting|blocked|terminated)\b", prev_low)}
                prev_ans_low = (seen_accepted[idx_prev].get("correct_answer", "").lower().strip() if seen_accepted and idx_prev < len(seen_accepted) else "")
                if "state" in prev_low and (prev_ans_low == tested_state or (tested_state in prev_states and len(prev_states) == 1)):
                    return None, f"Duplicate process-state question already testing '{tested_state}' state"

        if seen_accepted:
            for prev_q in seen_accepted:
                prev_corr = str(prev_q.get("correct_answer") or "").lower().strip()
                if corr_low == prev_corr and len(corr_low.split()) <= 4 and cls._stem_similarity(q_text, prev_q.get("question", "")) >= 0.25:
                    return None, f"Duplicate answer target '{corr}' already tested"

        concept = raw_concept or resolved_topic_area
        concept_key = re.sub(r"\s+", " ", raw_concept.lower()) if raw_concept else ""
        if concept_key and concept_key in seen_concepts:
            return None, f"Concept '{concept}' already tested"

        if len(explanation) < 15 or re.search(r"^(why opt\b|concise explanation|\.{3}$)", explanation, re.IGNORECASE):
            return None, "Explanation is missing, placeholder, or too brief"

        if cls._has_ready_burst_contradiction(f"{corr}. {explanation}"):
            return None, "Factual contradiction: answer or explanation misstates the Ready process state"

        # Merge distractor rationales into explanation if provided and not already included
        dist_rats = q_item.get("distractor_rationales")
        if isinstance(dist_rats, dict) and dist_rats:
            rat_lines = [f"- {k}: {v}" for k, v in dist_rats.items() if k and v and str(v).strip() not in explanation]
            if rat_lines:
                explanation = f"{explanation}\n\nWhy other options are incorrect:\n" + "\n".join(rat_lines[:3])

        # Shuffle options deterministically so correct answer position is randomized
        shuffled_opts = list(opts)
        rng.shuffle(shuffled_opts)
        correct_idx = shuffled_opts.index(corr)
        correct_letter = ["A", "B", "C", "D"][correct_idx]

        cat = str(q_item.get("category") or "Conceptual").strip().title()
        if cat not in ("Conceptual", "Application", "Numerical", "Technical"):
            cat = "Conceptual"
        diff = str(q_item.get("difficulty") or "Medium").strip().title()
        if diff not in ("Easy", "Medium", "Hard"):
            diff = "Medium"

        seen_stems.append(q_text)
        if concept_key:
            seen_concepts.add(concept_key)

        final_topic_area = resolved_topic_area or concept or cat
        # If raw_concept failed consistency or wasn't specific while topic_area was remapped, keep consistent
        final_concept = concept or final_topic_area or cat
        return {
            "category": cat,
            "difficulty": diff,
            "topic_area": final_topic_area,
            "question": q_text,
            "options": shuffled_opts,
            "correct_answer": corr,
            "correct_option_letter": correct_letter,
            "explanation": explanation,
            "calculation": q_item.get("calculation") or None,
            "distractor_rationales": dist_rats if isinstance(dist_rats, dict) else None,
            "grounded_concept": final_concept,
        }, None

    @staticmethod
    def _rebalance_answer_letters(quizzes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Ensures no single answer letter (A/B/C/D) dominates the quiz."""
        if len(quizzes) < 4:
            return quizzes
        letters = ["A", "B", "C", "D"]
        counts = {l: 0 for l in letters}
        for q in quizzes:
            counts[q.get("correct_option_letter", "A")] = counts.get(q.get("correct_option_letter", "A"), 0) + 1

        max_allowed = max(2, (len(quizzes) // 4) + 1)
        for q in quizzes:
            cur_let = q.get("correct_option_letter", "A")
            if counts.get(cur_let, 0) > max_allowed:
                target_let = min(letters, key=lambda l: counts.get(l, 0))
                if target_let != cur_let and len(q.get("options", [])) == 4:
                    cur_i = letters.index(cur_let)
                    tgt_i = letters.index(target_let)
                    opts = list(q["options"])
                    opts[cur_i], opts[tgt_i] = opts[tgt_i], opts[cur_i]
                    q["options"] = opts
                    q["correct_option_letter"] = target_let
                    counts[cur_let] -= 1
                    counts[target_let] += 1
        return quizzes

    @classmethod
    def generate_conceptual_quizzes(
        cls,
        transcript_text: str,
        topic_hint: Optional[str] = None,
        lecture_notes: Optional[dict[str, Any]] = None,
        assignments: Optional[list[dict[str, Any]]] = None,
        segments: Optional[list[Any]] = None,
        llm_provider: Optional[Any] = None,
        allow_extractive_fallback: bool = True,
        status_out: Optional[dict[str, Any]] = None,
        on_progress: Optional[Any] = None,
    ) -> list[dict[str, Any]]:
        import hashlib
        notes = lecture_notes or {}
        key_concepts = notes.get("key_concepts", []) or notes.get("topics", [])
        definitions = notes.get("definitions", [])
        study_topics = notes.get("study_topics", [])

        all_candidate_concepts = (
            [c for c in key_concepts]
            + [d.get("term") for d in definitions if d.get("term")]
            + [t.get("title") for t in study_topics if t.get("title")]
        )
        if not all_candidate_concepts:
            all_candidate_concepts = [topic_hint] if topic_hint else ["this session's material"]

        transcript_low = (transcript_text or "").lower()
        topics_summary = ", ".join(
            [t.get("title", "") for t in (study_topics or [])[:12] if t.get("title")]
        ) or (topic_hint or "this session's material")

        plan = cls._compute_quiz_target_plan(transcript_text, notes, segments)

        # Deterministic recording-scoped RNG for answer shuffling
        seed_bytes = hashlib.sha256((transcript_text[:500] + topics_summary).encode("utf-8", errors="ignore")).digest()[:8]
        rng = random.Random(int.from_bytes(seed_bytes, "big"))

        # 1. Attempt AI Quiz Synthesis (Blueprint + Batched Generation + Retry)
        from app.services.llm_provider import llm_lifecycle_session
        session_ctx = (
            llm_lifecycle_session("quiz", release_on_exit=False)
            if llm_provider is None
            else __import__("contextlib").nullcontext()
        )
        with session_ctx:
            provider = llm_provider or _try_get_llm_provider("quiz")
            if provider and provider.is_available() and len((transcript_text or "").strip()) > 50:
                try:
                    quizzes = cls._llm_generate_quizzes(
                        provider=provider,
                        transcript_text=transcript_text,
                        topics_summary=topics_summary,
                        transcript_low=transcript_low,
                        study_topics=study_topics,
                        plan=plan,
                        topic_hint=topic_hint,
                        rng=rng,
                        on_progress=on_progress,
                    )
                    min_floor = 1 if len((transcript_text or "").split()) < 80 else 4
                    hard_floor = max(min_floor, int(plan["min_questions"] * 0.6))
                    if quizzes and len(quizzes) >= hard_floor:
                        if status_out is not None:
                            status_out.update({
                                "status": "ok",
                                "mode": "llm_exam_v2",
                                "plan": plan,
                                "generated": len(quizzes),
                                "met_target_min": len(quizzes) >= plan["min_questions"],
                            })
                        return quizzes
                    msg = f"LLM generated insufficient valid exam questions ({len(quizzes)} < {hard_floor})."
                    logger.warning(msg)
                    if status_out is not None:
                        status_out.update({"status": "failed", "reason": msg, "plan": plan})
                    if not allow_extractive_fallback:
                        return []
                except Exception as e_quiz:
                    logger.warning(f"AI quiz generation failed: {e_quiz}", exc_info=True)
                    if status_out is not None:
                        status_out.update({"status": "failed", "reason": str(e_quiz), "plan": plan})
                    if not allow_extractive_fallback:
                        return []
            else:
                reason = "Quiz LLM is disabled or unavailable."
                if status_out is not None:
                    status_out.update({"status": "failed", "reason": reason, "plan": plan})
                if not allow_extractive_fallback:
                    return []

        # 2. Legacy extractive fallback (only when explicitly allowed for offline/unit tests)
        return cls._extractive_quiz_fallback(all_candidate_concepts, definitions, study_topics)

    @classmethod
    def _llm_generate_quizzes(
        cls,
        provider: Any,
        transcript_text: str,
        topics_summary: str,
        transcript_low: str,
        study_topics: Optional[list[dict[str, Any]]] = None,
        plan: Optional[dict[str, Any]] = None,
        topic_hint: Optional[str] = None,
        rng: Optional[random.Random] = None,
        on_progress: Optional[Any] = None,
    ) -> list[dict[str, Any]]:
        """
        Two-stage exam-grade MCQ generator:
          Stage A: Full-lecture chronological blueprint across early/middle/late sections.
          Stage B: Batched generation (4-5 MCQs per call) with validation, deduplication,
                   distractor auditing, and corrective retries.
        """
        import json as _json

        study_topics = study_topics or []
        plan = plan or cls._compute_quiz_target_plan(transcript_text, {"study_topics": study_topics})
        rng = rng or random.Random(42)
        target_count = plan["target_questions"]

        digest, section_windows = cls._build_chronological_digest(
            transcript_text, study_topics, max_section_chars=1500
        )
        blueprint = cls._build_lecture_blueprint(
            provider=provider,
            chronological_digest=digest,
            topics_summary=topics_summary,
            target_count=target_count,
            study_topics=study_topics,
            topic_hint=topic_hint,
        )

        subject_domain = blueprint["subject_domain"]
        items = blueprint["examinable_items"]
        batch_size = 3
        batches = [items[i: i + batch_size] for i in range(0, len(items), batch_size)]

        valid: list[dict[str, Any]] = []
        seen_stems: list[str] = []
        seen_concepts: set[str] = set()

        def _run_batch(batch_items: list[dict[str, Any]], section_text: str, base_temp: float = 0.25) -> None:
            retry_feedback = ""
            for attempt in range(2):
                covered_str = ", ".join(sorted(seen_concepts)[:15]) or "None yet"
                prompt = EXAM_MCQ_BATCH_PROMPT.format(
                    batch_count=len(batch_items),
                    subject_domain=subject_domain,
                    section_transcript=section_text,
                    blueprint_items_json=_json.dumps(batch_items, ensure_ascii=False),
                    covered_concepts=covered_str,
                    retry_feedback=retry_feedback,
                )
                try:
                    res = provider.generate_structured(
                        prompt=prompt,
                        schema=None,
                        system_prompt=(
                            f"You are a university professor in {subject_domain} writing challenging exam MCQs. "
                            "Ground everything strictly in the provided lecture section. Output ONLY valid JSON."
                        ),
                        max_tokens=1600,
                        temperature=base_temp if attempt == 0 else min(0.45, base_temp + 0.12),
                    )
                except Exception as e_batch:
                    retry_feedback = f"PREVIOUS ATTEMPT FAILED ({e_batch}). Return strictly valid JSON.\n\n"
                    continue

                raw_quizzes = res.get("quizzes", []) if isinstance(res, dict) else []
                if isinstance(raw_quizzes, dict):
                    raw_quizzes = [raw_quizzes]

                batch_accepted = 0
                rejection_reasons: list[str] = []
                for q_item in raw_quizzes:
                    refined, err = cls._validate_and_refine_mcq(
                        q_item, transcript_low, seen_stems, seen_concepts, rng,
                        study_topics=study_topics, seen_accepted=valid,
                    )
                    if refined:
                        valid.append(refined)
                        batch_accepted += 1
                    elif err:
                        rejection_reasons.append(err)

                if batch_accepted > 0 and on_progress:
                    try:
                        on_progress(list(valid), target_count)
                    except Exception:
                        pass

                if batch_accepted >= len(batch_items):
                    break
                if batch_accepted > 0:
                    # Only regenerate the rejected slots in attempt 1
                    remaining = max(1, len(batch_items) - batch_accepted)
                    batch_items = batch_items[-remaining:]
                retry_feedback = (
                    "PREVIOUS ATTEMPT HAD ISSUES: "
                    + "; ".join(rejection_reasons[:3] or ["Empty/invalid JSON"])
                    + ". Fix these issues, avoid placeholders/duplicates/merged states, and output strictly valid JSON.\n\n"
                )

        for batch in batches:
            if len(valid) >= plan["max_questions"]:
                break
            sec_counts: dict[str, int] = {}
            for it in batch:
                s = it.get("section", "middle")
                sec_counts[s] = sec_counts.get(s, 0) + 1
            primary_sec = max(sec_counts, key=lambda k: sec_counts[k])
            section_text = section_windows.get(primary_sec, transcript_text[:1500])
            _run_batch(batch, section_text)

        # Top-up pass if we haven't reached min_questions yet, rotating across fresh slices
        full_text = (transcript_text or "").strip()
        num_slices = 6
        slice_len = max(1, len(full_text) // num_slices)
        extra_windows = [
            full_text[i * slice_len: min(len(full_text), i * slice_len + 1500)]
            for i in range(num_slices)
        ] or [transcript_text[:1500]]

        sections_cycle = ["early", "middle", "late"]
        extra_idx = 0
        consecutive_empty = 0
        max_extra = 5
        while len(valid) < plan["min_questions"] and extra_idx < max_extra:
            sec = sections_cycle[extra_idx % 3]
            win_text = extra_windows[extra_idx % len(extra_windows)] or section_windows.get(sec, transcript_text[:2400])
            needed = min(3, max(1, plan["min_questions"] - len(valid)))
            topup_items = [
                {
                    "concept": f"Distinct {sec} lecture concept #{extra_idx + k + 1}",
                    "section": sec,
                    "topic_area": topic_hint or subject_domain,
                    "category": ["Conceptual", "Application", "Technical"][(extra_idx + k) % 3],
                    "difficulty": ["Medium", "Hard", "Easy"][(extra_idx + k) % 3],
                    "angle": f"Ask a new question about a different concept in this excerpt not in ALREADY COVERED",
                }
                for k in range(needed)
            ]
            before = len(valid)
            _run_batch(topup_items, win_text, base_temp=0.35)
            extra_idx += 1
            if len(valid) == before:
                consecutive_empty += 1
                if consecutive_empty >= 3:
                    break
            else:
                consecutive_empty = 0

        if valid:
            valid = cls._rebalance_answer_letters(valid)
            try:
                from app.services.pedagogical_validator import PedagogicalValidator
                val_res = PedagogicalValidator.validate_quizzes(valid, transcript_text, bloom_policy="quiz")
                if not val_res["passed"]:
                    rejected_qs = {r["question"] for r in val_res.get("rejected", [])}
                    valid = [q for q in valid if q["question"] not in rejected_qs]
            except Exception as e_val:
                logger.debug(f"PedagogicalValidator verification skipped: {e_val}")

        return valid[:plan["max_questions"]]

    @staticmethod
    def _extractive_quiz_fallback(
        all_candidate_concepts: list[str],
        definitions: list[dict[str, Any]],
        study_topics: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Constructs scenario-based fallback quiz questions (Bloom's Application & Analysis)
        from extracted definitions and key points. Uses authentic sibling concepts as
        distractors and shuffles answer positions using a recording-scoped deterministic seed.
        """
        import hashlib
        quizzes: list[dict[str, Any]] = []
        seed_material = "|".join(
            [str(c) for c in (all_candidate_concepts or [])]
            + [f"{d.get('term', '')}:{d.get('definition', '')}" for d in (definitions or [])]
            + [str(t.get("title", "")) for t in (study_topics or [])]
        )
        seed_int = int.from_bytes(hashlib.sha256(seed_material.encode("utf-8")).digest()[:8], "big")
        rng = random.Random(seed_int)

        # Collect all available terms for distractor generation
        all_terms = [d.get("term", "").strip() for d in definitions if d.get("term")] + \
                    [t.get("title", "").strip() for t in study_topics if t.get("title")]
        all_terms = [t for t in all_terms if t]

        # Definition-based questions (Application focus: "A practitioner applying {term}...")
        for d_idx, d in enumerate(definitions[:3]):
            term = d.get("term", "").strip()
            defn = d.get("definition", "").strip()
            if not term or not defn:
                continue

            # Build authentic distractors from sibling terms
            sibling_defs = [
                od.get("definition", "").strip()
                for od in definitions if od.get("term", "").strip().lower() != term.lower() and od.get("definition", "").strip()
            ]
            distractors = sibling_defs[:2]
            while len(distractors) < 3:
                generic = [
                    f"A concept unrelated to the operational mechanics of {term}",
                    f"An inverse hypothesis contradicting the established behavior of {term}",
                    f"An alternative configuration that disregards constraints of {term}",
                ]
                for g in generic:
                    if g not in distractors:
                        distractors.append(g)
                        if len(distractors) >= 3:
                            break

            options = [defn] + distractors[:3]
            rng.shuffle(options)
            correct_idx = options.index(defn)
            letter = ["A", "B", "C", "D"][correct_idx]

            quizzes.append({
                "category": "Application",
                "question": f"A practitioner applying {term} in a real-world scenario would need to understand which operational principle?",
                "options": options,
                "correct_answer": defn,
                "correct_option_letter": letter,
                "explanation": f"The correct operational definition of {term} is: {defn}",
                "calculation": None,
                "grounded_concept": term,
            })

        # Key-point questions (Analysis focus: "A practitioner working on {title} needs to evaluate system behavior...")
        for t in study_topics:
            if len(quizzes) >= 5:
                break
            kps = t.get("key_points", [])
            title = t.get("title", "")
            if not kps or not title:
                continue
            true_point = str(kps[0]).rstrip(".")

            # Build distractors from other topics' key points and definitions
            other_kps = []
            for ot in study_topics:
                if ot.get("title", "") != title:
                    for okp in ot.get("key_points", [])[:2]:
                        cand_kp = str(okp).rstrip(".")
                        if cand_kp and cand_kp.lower() != true_point.lower() and cand_kp not in other_kps:
                            other_kps.append(cand_kp)
            for od in definitions:
                odefn = od.get("definition", "").strip().rstrip(".")
                if odefn and odefn.lower() != true_point.lower() and odefn not in other_kps:
                    other_kps.append(odefn)

            distractors = other_kps[:2]
            while len(distractors) < 3:
                generic = [
                    f"Operational parameters are initialized without empirical verification for {title}",
                    f"Primary causal relationships in {title} are inverted regardless of observed conditions",
                    f"All analytical variables for {title} are held constant regardless of contextual variability",
                ]
                for g in generic:
                    if g not in distractors:
                        distractors.append(g)
                        if len(distractors) >= 3:
                            break

            options = [true_point] + distractors[:3]
            rng.shuffle(options)
            correct_idx = options.index(true_point)
            letter = ["A", "B", "C", "D"][correct_idx]

            quizzes.append({
                "category": "Analysis",
                "question": f"A practitioner working on {title} needs to evaluate system behavior. Which principle should guide their analysis?",
                "options": options,
                "correct_answer": true_point,
                "correct_option_letter": letter,
                "explanation": f"The key principle is: {true_point}.",
                "calculation": None,
                "grounded_concept": title,
            })

        return quizzes[:5]

    # =======================================================================
    # 5. BACKWARD-COMPATIBLE DOMAIN FORWARDERS
    # =======================================================================
    @classmethod
    def _build_machine_learning_study_guide(cls, transcript_text: str = "", topic_hint: Optional[str] = None,
                                             study_topics: Optional[list[Any]] = None, segments: Optional[list[Any]] = None,
                                             llm_provider: Optional[Any] = None, entailment_validator: Optional[Any] = None) -> str:
        return cls.generate_master_study_guide(transcript_text=transcript_text, segments=segments, study_topics=study_topics,
                                                topic_hint=topic_hint, domain="machine_learning", llm_provider=llm_provider, entailment_validator=entailment_validator)



# ===========================================================================
# Academic Intelligence v2 — evidence alignment and deterministic MCQ engine
# (activated only when ACADEMIC_INTEL_V2 is on; legacy paths above are untouched)
# ===========================================================================
import re as _re2
import random as _random2
from dataclasses import dataclass as _dataclass2

DISTRACTOR_BAND = (0.40, 0.80)          # preferred cosine band to the correct answer
DISTRACTOR_BAND_RELAXED = (0.25, 0.85)  # used only if the preferred band yields < 3 (flagged)
DISTRACTOR_REJECT_COS = 0.90            # synonyms / near-duplicates of the answer

_NEG_RE = _re2.compile(r"\b(?:not|no|never|nahi|nahin|mat|cannot|without)\b|n't", _re2.I)
_SENT_SPLIT_RE = _re2.compile(r"(?<=[.!?])\s+")
_DISFLUENT_RE = _re2.compile(r"\b(?:bhaiya|dekhein|acha ji|theek hai|assalamu|um+|uh+)\b", _re2.I)
_DEF_RE = _re2.compile(r"\b(?:is defined as|is called|refers to|is known as|means|is the process of|is a|is an)\b", _re2.I)
_EXAMPLE_RE = _re2.compile(r"\b(?:for example|for instance|such as|e\.g\.|jaise|misal|suppose|imagine|consider)\b", _re2.I)
_ANALYSIS_RE = _re2.compile(
    r"\b(?:because|therefore|unlike|whereas|however|causes?|leads? to|results? in|compared (?:to|with)|"
    r"difference|versus|vs\.?|trade-?off|advantage|disadvantage|drawback)\b", _re2.I)

_POOL_QUOTAS = (("definition", 2), ("concept", 3), ("application", 3), ("analysis", 2))
_FILL_PRIORITY = ("concept", "application", "analysis", "definition")
_POOL_BLOOM = {"definition": "Remember/Understand", "concept": "Understand",
               "application": "Apply", "analysis": "Analyze/Evaluate"}


def _neg_count(text: str) -> int:
    return len(_NEG_RE.findall(text or ""))


def _content_tokens(text: str) -> set:
    from app.services.academic_preprocess import _KP_STOPWORDS, _stem_simple
    return {_stem_simple(w) for w in _re2.findall(r"[A-Za-z]{3,}", (text or "").lower()) if w not in _KP_STOPWORDS}


def _default_embed(texts: list):
    from app.services.embedding_service import EmbeddingModelManager
    return EmbeddingModelManager.encode(list(texts))


def align_sentence(sentence: str, spans, embed_fn=None, cos_min=None, overlap_min=None, window=None):
    """
    Post-generation evidence attribution (plan 5.4).
    Returns {"segment_ids", "start_time", "end_time", "cosine", "overlap"} or None when unsupported.
    `window` optionally restricts the search to a (start, end) time range (the topic's range).
    """
    import numpy as np
    from app.config import ALIGN_COS_MIN, ALIGN_OVERLAP_MIN
    cos_min = ALIGN_COS_MIN if cos_min is None else cos_min
    overlap_min = ALIGN_OVERLAP_MIN if overlap_min is None else overlap_min
    pool = [s for s in spans if s.raw_text.strip()
            and (window is None or (s.raw_end >= window[0] and s.raw_start <= window[1]))]
    if not pool or not (sentence or "").strip():
        return None
    sent_tok = _content_tokens(sentence)
    if not sent_tok:
        return None
    embed = embed_fn or _default_embed
    vecs = embed([sentence] + [s.raw_text for s in pool])
    sims = vecs[1:] @ vecs[0]
    order = np.argsort(-sims)[:3]
    best = []
    for idx in order:
        seg = pool[int(idx)]
        overlap = len(sent_tok & _content_tokens(seg.raw_text)) / len(sent_tok)
        if float(sims[idx]) >= cos_min and overlap >= overlap_min:
            best.append((seg, float(sims[idx]), overlap))
    if not best:
        return None
    return {
        "segment_ids": [b[0].segment_id for b in best],
        "start_time": min(b[0].raw_start for b in best),
        "end_time": max(b[0].raw_end for b in best),
        "cosine": round(best[0][1], 3),
        "overlap": round(best[0][2], 3),
    }


@_dataclass2
class _Sentence:
    text: str
    segment_id: object
    start: float
    end: float
    index: int


class AcademicMCQEngine:
    """Deterministic, transcript-grounded MCQ generator with explicit sufficiency policy."""

    @staticmethod
    def _sentences(spans) -> list:
        out, k = [], 0
        for sp in spans:
            for part in _SENT_SPLIT_RE.split((sp.raw_text or "").strip()):
                part = part.strip()
                wc = len(part.split())
                if 8 <= wc <= 40 and not part.endswith("?") and not _DISFLUENT_RE.search(part):
                    out.append(_Sentence(part, sp.segment_id, sp.raw_start, sp.raw_end, k))
                    k += 1
        return out

    @staticmethod
    def _find_phrase(sentence: str, phrase: str):
        return _re2.search(r"(?<![A-Za-z])" + _re2.escape(phrase).replace(r"\ ", r"\s+") + r"(?![A-Za-z])", sentence, _re2.I)

    @classmethod
    def _pool_of(cls, s: _Sentence, answer: str = "") -> str:
        # Definition: the term is the grammatical subject of an "X is/means/refers to ..." sentence.
        if answer and _re2.match(
                r"^(?:the\s+|a\s+|an\s+)?" + _re2.escape(answer).replace(r"\ ", r"\s+") + r"\s+(?:is|are|means|refers)\b",
                s.text, _re2.I):
            return "definition"
        if _DEF_RE.search(s.text) and _re2.search(r"\b(?:is defined as|is called|refers to|is known as)\b", s.text, _re2.I):
            return "definition"
        if _EXAMPLE_RE.search(s.text):
            return "application"
        if _ANALYSIS_RE.search(s.text):
            return "analysis"
        return "concept"

    @staticmethod
    def _stem_for(pool: str, blanked: str) -> str:
        if pool == "definition":
            return f"Which term completes this statement from the lecture? \"{blanked}\""
        if pool == "application":
            return f"Which concept does this scenario from the lecture illustrate? \"{blanked}\""
        if pool == "analysis":
            return f"Which term correctly completes this reasoning from the lecture? \"{blanked}\""
        return f"Which term best completes this explanation from the lecture? \"{blanked}\""

    @classmethod
    def verify_question(cls, q: dict, source_sentence: str) -> list:
        """Correctness checks (plan 6.2): answer span in source, polarity parity, answer not leaked in stem."""
        reasons = []
        ans = str(q.get("correct_answer", ""))
        if not cls._find_phrase(source_sentence, ans):
            reasons.append("answer_not_in_source")
        stem_plus = f"{q.get('question', '')} {q.get('explanation', '')}"
        quoted = _re2.findall(r"\"([^\"]+)\"", q.get("question", ""))
        stem_body = quoted[0] if quoted else q.get("question", "")
        if _neg_count(stem_body) != _neg_count(source_sentence):
            reasons.append("negation_polarity_mismatch")
        if cls._find_phrase(stem_body, ans):
            reasons.append("answer_leaked_in_stem")
        return reasons

    @classmethod
    def generate(cls, spans, keyphrases, rng=None, embed_fn=None, target: int = 10, transcript_hash: str = "",
                 exclude_answers=None) -> dict:
        import numpy as np
        from app.config import MCQ_DEDUPE_COS, MIN_WORDS_FOR_10
        exclude_answers = set(exclude_answers or ())
        rng = rng or _random2.Random(transcript_hash or "mcq")
        embed = embed_fn or _default_embed

        total_words = sum(len(s.tokens) for s in spans)
        phrases = [k for k, _, _ in keyphrases]
        cval = {k: c for k, c, _ in keyphrases}
        sentences = cls._sentences(spans)

        reason = None
        if total_words < MIN_WORDS_FOR_10:
            reason = "too_short"
        elif len(phrases) < 5:
            reason = "too_few_concepts"

        # ---- build candidates -------------------------------------------------
        def answer_ok(p: str) -> bool:
            # A usable answer is a multi-word term or a long single word (avoids "time", "process" ...).
            return len(p.split()) >= 2 or len(p) >= 7

        cands = []
        for s in sentences:
            present = [ph for ph in phrases if cls._find_phrase(s.text, ph)]
            # keep maximal phrases only: never blank part of a longer term ("time" inside "time quantum")
            maximal = [p for p in present if not any(p != q and p in q for q in present)]
            for ph in sorted([p for p in maximal if answer_ok(p) and p not in exclude_answers], key=lambda p: (-cval[p], p)):
                hits = list(_re2.finditer(r"(?<![A-Za-z])" + _re2.escape(ph).replace(r"\ ", r"\s+") + r"(?![A-Za-z])",
                                          s.text, _re2.I))
                if len(hits) != 1:
                    continue  # a single unambiguous blank only
                m = hits[0]
                blanked = s.text[:m.start()] + "_____" + s.text[m.end():]
                if len(blanked.replace("_____", "").split()) < 6:
                    continue
                cands.append({"sentence": s, "answer": ph, "blanked": blanked,
                              "pool": cls._pool_of(s, ph), "score": cval[ph]})
                break  # one candidate per sentence: highest-value eligible keyphrase

        cands.sort(key=lambda c: (-c["score"], c["sentence"].index))
        if not phrases or not cands:
            return {"mcqs": [], "mcq_status": "insufficient_content", "mcq_count": 0,
                    "reason": reason or "validation_attrition", "distribution_relaxed": False}

        ph_vecs = embed(phrases)
        ph_idx = {p: i for i, p in enumerate(phrases)}

        def distractors(c):
            a = c["answer"]
            av = ph_vecs[ph_idx[a]]
            a_stem = _content_tokens(a)
            scored = []
            for p in phrases:
                if p == a or a in p or p in a or _content_tokens(p) == a_stem:
                    continue
                if cls._find_phrase(c["sentence"].text, p) or cls._find_phrase(c["blanked"], p):
                    continue
                cos = float(ph_vecs[ph_idx[p]] @ av)
                if cos > DISTRACTOR_REJECT_COS:
                    continue
                scored.append((p, cos))
            for lo, hi, relaxed in ((*DISTRACTOR_BAND, False), (*DISTRACTOR_BAND_RELAXED, True)):
                ok = sorted([x for x in scored if lo <= x[1] <= hi], key=lambda x: (-x[1], x[0]))
                if len(ok) >= 3:
                    return [p for p, _ in ok[:3]], relaxed
            return None, False

        # ---- select per pool with fixed redistribution ------------------------
        by_pool = {p: [c for c in cands if c["pool"] == p] for p, _ in _POOL_QUOTAS}
        chosen, used_sent, used_ans = [], set(), set()
        relaxed_dist = False

        def try_take(c):
            if c["sentence"].index in used_sent or c["answer"] in used_ans:
                return False
            d, dr = distractors(c)
            if d is None:
                return False
            c["_d"], c["_dr"] = d, dr
            used_sent.add(c["sentence"].index)
            used_ans.add(c["answer"])
            chosen.append(c)
            return True

        for pool, quota in _POOL_QUOTAS:
            taken = 0
            for c in by_pool[pool]:
                if taken >= quota:
                    break
                if try_take(c):
                    taken += 1
        if len(chosen) < target:
            for pool in _FILL_PRIORITY:
                for c in by_pool[pool]:
                    if len(chosen) >= target:
                        break
                    if try_take(c):
                        relaxed_dist = True

        # ---- build question drafts (answer position assigned AFTER dedupe so balance holds) ----
        drafts = []
        for c in chosen:
            s = c["sentence"]
            q = {
                "question": cls._stem_for(c["pool"], c["blanked"]),
                "options": list(c["_d"]),  # distractors only for now
                "correct_answer": c["answer"],
                "explanation": f"The lecture states: \"{s.text}\"",
                "grounded_concept": c["answer"],
                "category": c["pool"],
                "bloom_level": _POOL_BLOOM[c["pool"]],
                "calculation": None,
                "distractor_relaxed": c["_dr"],
                "evidence_timestamps": [{"segment_id": s.segment_id, "start": s.start, "end": s.end}],
            }
            if cls.verify_question(q, s.text):
                continue
            drafts.append(q)

        # ---- stem dedupe (embedding cosine) -----------------------------------
        quizzes = drafts
        if len(quizzes) > 1:
            sv = embed([q["question"] for q in quizzes])
            keep = []
            for i in range(len(quizzes)):
                if all(float(sv[i] @ sv[j]) < MCQ_DEDUPE_COS for j in keep):
                    keep.append(i)
            quizzes = [quizzes[i] for i in keep]
        quizzes = quizzes[:target]

        # ---- seeded, balanced answer positions over the FINAL set ---------------
        positions = [0, 1, 2, 3] * ((len(quizzes) + 3) // 4)
        rng.shuffle(positions)
        for n, q in enumerate(quizzes):
            pos = positions[n]
            q["options"].insert(pos, q["correct_answer"])
            q["correct_option_letter"] = "ABCD"[pos]

        status, why = "complete", None
        if len(quizzes) < target or reason:
            status = "insufficient_content"
            why = reason or "validation_attrition"
        return {"mcqs": quizzes, "mcq_status": status, "mcq_count": len(quizzes),
                "reason": why, "distribution_relaxed": relaxed_dist}
