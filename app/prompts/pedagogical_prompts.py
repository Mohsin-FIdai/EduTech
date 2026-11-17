"""
app/prompts/pedagogical_prompts.py — Centralized LLM prompt templates for
pedagogical synthesis across the study guide, quiz, and RAG pipelines.

All prompt constants are imported by the logic files
(study_guide_generator.py, rag_service.py) so that prompt text is
independently auditable and versionable without touching business logic.
"""

# ===========================================================================
# 1. TRANSCRIPTION-FIDELITY RULES (non-negotiable, shared across prompts)
# ===========================================================================

FIDELITY_RULES = """\
TRANSCRIPTION-FIDELITY RULES (non-negotiable):
- The transcript is ground truth. If the extracted topics summary and the transcript ever disagree or the topics summary is vague, defer to the transcript's actual wording.
- For anything precise — formulas, definitions, numeric values, named laws/theorems, exact terminology the instructor used — reproduce it the way the instructor stated it, not a paraphrase that could drift the meaning. Do not silently "correct" or generalize a formula/definition even if you believe a textbook version differs.
- Identify the subject matter, the right worked-example style, and every term/formula/example directly from THIS transcript. Do not reach for a "typical" example, formula, or definition for the subject you think this is — use only what this specific session actually said.
- If something is ambiguous, garbled, or inaudible-seeming in the transcript (e.g. cut off, contradictory, clearly a transcription error), say so explicitly in the guide (e.g. "[unclear in recording]") rather than guessing or smoothing it over.
- If a section of the blueprint has no supporting material in this particular session, write "Not covered in this session" — do not fill it with generic domain knowledge to avoid an empty section.
- Anything you add that is NOT from the transcript (e.g. a standard definition to clarify jargon the session used but didn't define) must be clearly labeled "Beyond the session (background context):" so the student can tell session content apart from your additions.
- Never introduce a concept, term, algorithm, or example that does not appear in the transcript below, even if it is a common companion topic to what's being discussed.
"""

# ===========================================================================
# 2. UNIVERSAL INSTRUCTIONS FOR LECTURE TOPIC EXTRACTION & NOTE GENERATION
# ===========================================================================

UNIVERSAL_LECTURE_NOTES_INSTRUCTIONS = """\
UNIVERSAL INSTRUCTIONS FOR LECTURE TOPIC EXTRACTION AND NOTE GENERATION:
Apply these rules to EVERY lecture uploaded, regardless of its subject or academic field:

1. Accurate Topic Extraction:
- Identify only genuine academic topics and concepts being taught in the lecture.
- Never convert random transcript sentences, conversational phrases, classroom instructions, questions, or AI-generated responses into topics.
- Do not create meaningless or irrelevant headings.
- Identify concepts based on their actual meaning and context, not merely the words spoken.
- Organise related concepts together and use meaningful subheadings.

2. Proper Structure:
- Always begin the notes with an Introduction / Executive Summary explaining the lecture's overall subject, main ideas, and learning objectives.
- Organise the actual topics and subtopics in a logical sequence.
- End with Important Takeaways or a concise Revision Summary.

3. Actual Educational Notes:
- Do not simply copy, extract, or rephrase sentences from the transcript. First understand what is being taught, then explain it in your own words, just as a good teacher would teach a student.
- For every topic, explain:
  * What the concept means.
  * How it works.
  * Why it is important.
  * Relevant steps, details, and underlying logic.
  * Examples and applications wherever helpful.
- Use simple, natural, student-friendly language. Explain difficult terminology and build understanding from basic concepts to more advanced ones.

4. Quality and Accuracy:
- Cover all important concepts without unnecessary repetition.
- Do not invent topics or unrelated information.
- Preserve the lecturer's actual meaning and important details.
- Exclude irrelevant transcript content rather than turning it into notes.
- If something is unclear in the transcript, do not guess.
- Use headings, subheadings, bullet points, tables, formulas, and examples wherever appropriate.
"""

# ===========================================================================
# 2b. STUDY-EFFECTIVENESS RULES (write for retention, not just accuracy)
# ===========================================================================

STUDY_EFFECTIVENESS_RULES = """\
STUDY-EFFECTIVENESS RULES (write for retention, not just accuracy):
- Always begin with an Introduction / Executive Summary covering the overall subject, main ideas, and learning objectives, and conclude with Important Takeaways / Revision Summary.
- Teach the lecture like a clear, student-friendly professor: for each topic explain what it means, how it works, why it is important, its steps/logic, and concrete examples.
- Flag high-yield material: if the instructor repeated something, said "this will be on the exam," emphasized it, or spent disproportionate time on it, mark it (e.g. "⭐ Emphasized in session").
- Every worked example must show the reasoning chain step-by-step, not just the final answer — the student should be able to redo it, not just recognize it.
- End each major section with 2-4 short retrieval-practice questions (active recall), with answers given directly underneath — never make the student hunt for the answer elsewhere in the guide.
- Terminology Glossary entries should be written flashcard-style: "Term — one-sentence definition in plain language," short enough to actually be used for spaced repetition.
- In "Key Takeaways," phrase each point as something the student should be able to recall unprompted (a testable claim), not a topic label.
- Explicitly name common misconceptions AND why they're wrong, not just that they exist — a student can't self-correct from a vague warning.
- Prefer plain, concrete language over dense jargon when restating an idea; jargon from the transcript should still appear (fidelity), but followed by a plain-language gloss.
"""

# ===========================================================================
# 3. CEM STUDY GUIDE PROMPT — Concept-Example-Mechanism hierarchy
# ===========================================================================

CEM_STUDY_GUIDE_PROMPT = """\
You are generating a study guide structured as a Concept-Example-Mechanism (CEM) \
hierarchy for ONE specific lecture or meeting session.

Ground every section ONLY in the material below — do not introduce facts, formulas, \
numbers, terms, or examples that are not supported by the transcript or extracted topics.

First, work out from the transcript itself what subject/domain this session covers \
and what kind of worked example fits it (a numeric calculation, a process/cause-effect \
walkthrough, a transaction trace, a query/derivation, a scenario — whatever the \
transcript actually supports). Do not assume a subject in advance; read the transcript first.

CEM HIERARCHY STRUCTURE (use these exact section headings, in this order; \
skip a heading's content with "Not covered in this session" if nothing supports it):
{section_list}

For each Core Concept, teach the student clearly (do not merely copy raw transcript lines) and provide:
- **Definition**: What the concept means, in clear, student-friendly language (explaining any technical terminology).
- **Mechanism**: How it works step by step and its underlying logic.
- **Why It Matters**: Why the concept is important and where it applies.
- **Example**: A concrete example or application from the lecture illustrating it.
- **Formula**: Any governing equation or workflow pathway, if taught (otherwise omit).

{fidelity_rules}
{study_effectiveness_rules}

EXTRACTED TOPICS FOR THIS SESSION:
{topics_block}

FULL TRANSCRIPT (ground-truth source — resolve any ambiguity in the topics above \
by checking this):
\"\"\"
{transcript_text}
\"\"\"

Write the complete study guide now, section by section, using only the content above, \
following both the fidelity rules and the study-effectiveness rules.\
"""

# ===========================================================================
# 4. BLOOM'S TAXONOMY QUIZ PROMPT — Application & Analysis levels
# ===========================================================================

BLOOMS_QUIZ_PROMPT = """\
You are synthesizing academic practice quiz questions for ONE specific session.

CRITICAL RULES:
1. Ground every question, option, correct answer, and explanation EXCLUSIVELY \
in the transcript below.
2. Identify the subject yourself from the transcript — never introduce a concept, \
model, formula, or term not present in it.
3. Do NOT ask questions about what the speaker "said" or "mentioned." Do NOT use \
phrasing like "According to the lecture, what is..." or "What did the session say about..."
4. Generate questions that test APPLICATION and ANALYSIS of the concept (Bloom's \
Taxonomy levels 3-4). Example: "If a practitioner needs to achieve X, which method \
should they use based on the principles covered?"
5. Generate 3 plausible but factually wrong distractors based on the lecture's logic. \
Distractors must be realistic sibling concepts, NOT trivial placeholders like "Option A."

FULL TRANSCRIPT (GROUND TRUTH):
\"\"\"
{transcript_text}
\"\"\"

TOPICS ALREADY EXTRACTED (may be incomplete):
{topics_summary}

Generate 3 to 4 multiple-choice questions covering key concepts, definitions, \
numerical derivations (only if formulas exist in the transcript), and practical \
scenarios FROM THE TRANSCRIPT.

Each question needs exactly 4 options, a correct_answer matching one option, \
a correct_option_letter in A-D, an explanation citing the session's own reasoning, \
and a grounded_concept naming the specific transcript concept it tests.

Output strictly valid JSON: {{"quizzes": [{{"category": "...", "question": "...", \
"options": ["...", "...", "...", "..."], "correct_answer": "...", \
"correct_option_letter": "A", "explanation": "...", "calculation": null, \
"grounded_concept": "..."}}]}}\
"""


# ===========================================================================
# 4b. EXAM-GRADE QUIZ BLUEPRINT & BATCHED GENERATION PROMPTS
# ===========================================================================

LECTURE_BLUEPRINT_PROMPT = """\
You are an experienced university professor designing a rigorous examination blueprint for a lecture.

Analyze the FULL lecture overview and chronological excerpts below, then identify all distinct, \
high-value examinable concepts across the ENTIRE lecture (early, middle, and late sections).

LECTURE METADATA & TOPICS:
{topics_summary}

CHRONOLOGICAL LECTURE EXCERPTS (EARLY / MIDDLE / LATE):
\"\"\"
{chronological_digest}
\"\"\"

TASK:
1. Identify the academic subject and lecture style (e.g., mathematical/numerical, systems/algorithmic, \
conceptual/theoretical, experimental/applied).
2. Produce a non-redundant list of {target_count} distinct examinable items spanning ALL sections of \
the lecture (early, middle, late).
3. For each item, choose the most appropriate question category for THIS subject:
   - "Conceptual": underlying mechanisms, why something works, cause-effect, comparing closely related ideas.
   - "Application": real-world scenario, predicting outcomes, choosing the right method, diagnosing an error.
   - "Numerical": ONLY if formulas, equations, probabilities, or quantitative calculations appear in the lecture.
   - "Technical": algorithms, execution steps, system components, code/data-structure reasoning, implementation details.
4. Assign a difficulty distribution across the blueprint: ~20% "Easy" (fundamental principle, NOT word recall), \
~45% "Medium" (application & interpretation), ~35% "Hard" (multi-step reasoning, subtle distinctions, edge cases).

Output strictly valid JSON:
{{"subject_domain": "...", "lecture_style": "...", "has_numerical_content": true, \
"examinable_items": [{{"concept": "...", "section": "early|middle|late", "topic_area": "...", \
"category": "Conceptual|Application|Numerical|Technical", "difficulty": "Easy|Medium|Hard", \
"angle": "Brief note on what tricky distinction, scenario, or calculation to test"}}]}}
"""


EXAM_MCQ_BATCH_PROMPT = """\
Write {batch_count} challenging university exam MCQs from this lecture excerpt ({subject_domain}).
Rules:
- Ground strictly in the excerpt. Never say "According to the lecture" or "What did the speaker say".
- Test understanding, application, or calculation. Every question must end with "?".
- 4 substantive options of similar length (write full option text WITHOUT "A.", "Opt A", or "1)" prefixes). No "All/None of the above".
- Copy category/difficulty from ITEMS TO TEST. Do not repeat ALREADY COVERED concepts.
- Copy the exact winning option text verbatim into "correct_answer".
- Explain why the answer is right and why wrong options are misconceptions. Put math steps in "calculation".

EXCERPT:
\"\"\"
{section_transcript}
\"\"\"

ITEMS TO TEST: {blueprint_items_json}
ALREADY COVERED: {covered_concepts}

{retry_feedback}Return ONLY a valid JSON object matching this schema (replace every field description with real lecture-derived content; never output template text or angle brackets like <concept>):
- "grounded_concept" and "topic_area" must name the specific concept and matching lecture topic actually tested by that question (never output placeholder strings, and never attach an unrelated topic such as "Turnaround Time" to a process-state question).
- Keep distinct states and concepts separate (never combine two distinct states like "Ready Running" or "Waiting Block" into one option).
{{"quizzes": [{{"category": "Application", "difficulty": "Medium", "topic_area": "Specific matching topic title from the lecture", "question": "Complete self-contained exam question ending with ?", "options": ["First complete answer choice", "Second complete answer choice", "Third complete answer choice", "Fourth complete answer choice"], "correct_answer": "Exact verbatim text of the winning choice", "correct_option_letter": "A", "explanation": "Concise explanation of why the winning choice holds and why distractors fail", "calculation": null, "grounded_concept": "Specific distinct concept tested by this question"}}]}}
"""

# ===========================================================================
# 5. RAG GROUNDING SYSTEM PROMPT — strict lecture-grounded teaching assistant
# ===========================================================================

GROUNDING_SYSTEM_PROMPT = """\
You are an expert AI Teaching Assistant for a university course on {domain_name}.
Your role is to help students deeply understand lecture concepts by combining the \
instructor's lecture discussion with foundational domain principles.

Rules:
1. Answer naturally, clearly, concisely, and pedagogically.
2. Ground your explanation strictly in what the instructor taught. Do NOT introduce \
concepts, terms, formulas, or examples that were not discussed in this lecture.
3. Always reference the exact lecture timestamp range.
4. If the question asks about something not covered in the lecture, say so honestly \
rather than hallucinating an answer.
5. If you are unsure, prefer "This was not covered in detail in the lecture" over \
a speculative answer.\
"""

FREE_TUTOR_SYSTEM_PROMPT = """\
You are an expert AI University Tutor for {domain_name}.
Combine what the instructor covered in the lecture recording with your broader academic \
knowledge to give clear, professionally formatted, easy-to-read educational answers.

Formatting & Response Rules:
1. Start IMMEDIATELY with a clean `### Title` heading—never write greetings, filler preambles, or clutter.
2. Organize answers consistently with `###` main headings, `####` subheadings, blank lines between sections, bullet lists (`- **Key Term:** Explanation`), numbered steps (`1. `, `2. `), and **bold** highlights for crucial takeaways.
3. ALWAYS write every mathematical formula, equation, calculation, and variable using proper LaTeX:
   - Display formulas on their own line: `$$TAT_i = C_i - A_i$$`
   - Inline variables/symbols inside sentences: `$C_i$`, `$A_i$`, `$O(n \\log n)$`
4. Weave relevant lecture details and timestamps naturally into your explanation when excerpts are provided, and freely add complete textbook context, examples, and step-by-step clarity beyond the audio.
5. Never refuse a question—if not in the recording, briefly state that in one phrase and give a complete, well-structured explanation from domain knowledge.\
"""
