"""
Pydantic schemas for request/response serialization and error models.
"""

from datetime import datetime
from typing import Optional, Any, Union
from pydantic import BaseModel, ConfigDict, field_validator


class ErrorResponse(BaseModel):
    status: str = "error"
    code: str
    message: str
    details: Optional[Any] = None


class RecordingBase(BaseModel):
    filename: str
    file_format: str
    media_type: str
    file_size_bytes: int
    duration_seconds: float
    status: str
    sample_rate: Optional[int] = 16000
    channels: Optional[int] = 1
    speech_duration_seconds: Optional[float] = None
    silence_duration_seconds: Optional[float] = None
    preprocessed_file_path: Optional[str] = None
    topic: Optional[str] = None
    recording_type: Optional[str] = "Lecture"


class RecordingResponse(RecordingBase):
    id: str
    created_at: datetime
    updated_at: datetime
    error_message: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class PreprocessingResponse(BaseModel):
    status: str = "success"
    message: str = "Audio extracted and preprocessed successfully"
    recording_id: str
    preprocessed_file_path: str
    sample_rate: int
    channels: int
    total_duration_seconds: float
    speech_duration_seconds: float
    silence_duration_seconds: float
    speech_intervals: list[tuple[float, float]]


class UploadSuccessResponse(BaseModel):
    status: str = "success"
    message: str = "File uploaded and validated successfully"
    data: RecordingResponse


class RecordingListResponse(BaseModel):
    status: str = "success"
    total: int
    data: list[RecordingResponse]


# Phase 3 Chunking Schemas
class ChunkItemResponse(BaseModel):
    id: str
    recording_id: str
    chunk_index: int
    start_time: float
    end_time: float
    duration: float
    file_path: str
    overlap_start: float = 0.0
    overlap_end: float = 0.0
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ChunkListResponse(BaseModel):
    status: str = "success"
    recording_id: str
    total_chunks: int
    data: list[ChunkItemResponse]


class ChunkingRequest(BaseModel):
    target_duration: Optional[float] = 25.0
    max_duration: Optional[float] = 30.0
    min_duration: Optional[float] = 2.0
    overlap_seconds: Optional[float] = 0.5


class ChunkingResponse(BaseModel):
    status: str = "success"
    message: str = "Audio partitioned into intelligent chunks successfully"
    recording_id: str
    total_chunks: int
    chunks: list[ChunkItemResponse]


# Phase 4 Transcription Schemas
class TranscriptSegmentResponse(BaseModel):
    id: str
    chunk_index: int
    segment_index: int
    start_time: float
    end_time: float
    text: str
    normalized_text: Optional[str] = None
    urdu_text: Optional[str] = None
    english_text: Optional[str] = None
    translation_source: Optional[str] = None
    translation_confidence: Optional[float] = None
    language: Optional[str] = None
    confidence: Optional[float] = None
    no_speech_prob: Optional[float] = None

    model_config = ConfigDict(from_attributes=True)


class TranscriptResponse(BaseModel):
    id: str
    recording_id: str
    original_transcript: str
    normalized_transcript: str
    urdu_transcript: Optional[str] = None
    english_transcript: Optional[str] = None
    language_summary: Optional[str] = None
    total_words: int
    total_segments: int
    segments: list[TranscriptSegmentResponse]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class TranscriptionRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_size: Optional[str] = "base"
    language: Optional[str] = None
    temperature: Optional[float] = 0.0
    topic: Optional[str] = None


class TranscriptionResponse(BaseModel):
    status: str = "success"
    message: str = "Recording transcribed successfully"
    recording_id: str
    transcript_id: str
    total_segments: int
    total_words: int
    language_summary: Optional[str] = None
    original_transcript: str
    normalized_transcript: str
    urdu_transcript: Optional[str] = None
    english_transcript: Optional[str] = None
    segments: list[TranscriptSegmentResponse]


# Phase 5 Text Processing Schemas
class TokenTagItem(BaseModel):
    token: str
    lang: str
    confidence: float
    is_technical: bool


class TextProcessRequest(BaseModel):
    text: str


class TextProcessResponse(BaseModel):
    status: str = "success"
    original_text: str
    cleaned_text: str
    normalized_text: str
    urdu_text: str
    english_text: str
    token_tags: list[TokenTagItem]
    language_distribution: dict[str, float]
    stages: dict[str, Any]


class TaggedSegmentItem(BaseModel):
    segment_id: str
    chunk_index: int
    segment_index: int
    start_time: float
    end_time: float
    original_text: str
    normalized_text: str
    token_tags: list[TokenTagItem]


class TaggedSegmentsResponse(BaseModel):
    status: str = "success"
    recording_id: str
    total_segments: int
    segments: list[TaggedSegmentItem]


# Phase 6 Intelligence Schemas
class AssignmentItem(BaseModel):
    topic: str = "Not specified"
    description: str = "Not specified"
    required_tasks: str = "Not specified"
    format: str = "Not specified"
    submission_method: str = "Not specified"
    deadline: str = "Not specified"
    length_requirements: str = "Not specified"
    individual_or_group: str = "Not specified"
    required_tools: str = "Not specified"
    responsible_person_or_group: str = "Not specified"
    timestamp: str = "Not specified"
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    source_timestamps: list[str] = []
    field_timestamps: dict[str, str] = {}
    confidence: str = "Medium"  # High, Medium, Low


class ActionItem(BaseModel):
    task: str
    owner: str = "Not specified"
    deadline: str = "Not specified"
    deliverables: str = "Not specified"
    timestamp: str = "Not specified"
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    source_timestamps: list[str] = []
    field_timestamps: dict[str, str] = {}
    confidence: str = "Medium"  # High, Medium, Low


class ImportantPointItem(BaseModel):
    point: str
    is_exam_relevant: bool = False
    signal_phrase: Optional[str] = None
    timestamp: str = "Not specified"
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    source_segment: Optional[str] = None
    confidence: float = 0.0
    citations: list[dict] = []


class LectureNotes(BaseModel):
    model_config = ConfigDict(extra="allow")

    topics: list[str] = []
    subtopics: list[str] = []
    definitions: list[dict[str, str]] = []
    key_concepts: list[str] = []
    important_points: list[ImportantPointItem] = []
    study_topics: list[dict[str, Any]] = []
    lecture_overview: Optional[str] = None
    markdown_study_guide: Optional[str] = None
    executive_summary: Optional[str] = None
    learning_objectives: list[str] = []
    revision_summary: Optional[Union[str, list[str]]] = None
    theme: Optional[str] = None

    @field_validator("revision_summary", "executive_summary", "lecture_overview", "markdown_study_guide", mode="before")
    @classmethod
    def coerce_text_fields(cls, v: Any) -> Optional[str]:
        if isinstance(v, list):
            return "\n\n".join(str(item) for item in v if item)
        return v
    subject: Optional[str] = None
    chapters: list[dict[str, Any]] = []
    quality_score: Optional[float] = None
    quality_pass: Optional[bool] = None
    quality_details: Optional[dict[str, Any]] = None
    analyzer_used: Optional[str] = None
    transcript_hash: Optional[str] = None
    meta: Optional[dict[str, Any]] = None
    explicit_exam_hints: list[dict[str, Any]] = []
    inferred_importance: Optional[int] = None
    inferred_exam_relevance: Optional[str] = None
    universal_notes_rev: Optional[str] = None


class QuizQuestion(BaseModel):
    question: str
    options: list[str]
    correct_answer: str
    explanation: str
    grounded_concept: str
    category: Optional[str] = None
    difficulty: Optional[str] = None
    topic_area: Optional[str] = None
    calculation: Optional[str] = None
    distractor_rationales: Optional[dict[str, str]] = None
    correct_option_letter: Optional[str] = None
    mcq_status: Optional[str] = "complete"
    evidence_timestamps: list[dict[str, Any]] = []


class EvidenceItem(BaseModel):
    start: float
    end: float
    segment_id: Optional[Union[int, str]] = None
    text: str = ""


class QARequest(BaseModel):
    question: str
    top_k: Optional[int] = 3
    eli5: Optional[bool] = False
    session_id: Optional[str] = None


class QAResponse(BaseModel):
    status: str = "success"
    found: bool
    question: str
    answer: str
    timestamp: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    source_segment: Optional[str] = None
    confidence: float = 0.0
    citations: list[dict] = []
    evidence: list[EvidenceItem] = []
    session_id: Optional[str] = None


class IntelligenceResponse(BaseModel):
    status: str = "success"
    recording_id: str
    transcript_id: str
    summary_short: str
    summary_detailed: str
    lecture_notes: LectureNotes
    assignments: list[AssignmentItem]
    action_items: list[ActionItem]
    quizzes: list[QuizQuestion]
    created_at: datetime


# Phase 7 Vector Embeddings & Semantic Search Schemas
class SemanticSearchRequest(BaseModel):
    query: str
    top_k: int = 5
    threshold: float = 0.40


class SemanticSearchResultItem(BaseModel):
    segment_id: str
    chunk_index: int
    segment_index: int
    start_time: float
    end_time: float
    timestamp: str
    text: str
    normalized_text: str
    similarity_score: float


class SemanticSearchResponse(BaseModel):
    status: str = "success"
    recording_id: str
    query: str
    total_results: int
    results: list[SemanticSearchResultItem]


class SemanticQARequest(BaseModel):
    question: str
    threshold: float = 0.50


class SemanticQAResponse(BaseModel):
    status: str = "success"
    found: bool
    question: str
    answer: str
    timestamp: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    source_segment: Optional[str] = None
    similarity_score: float = 0.0


class EmbeddingGenerationResponse(BaseModel):
    status: str = "success"
    recording_id: str
    total_segments_embedded: int
    dimensions: int = 384


# Phase 8 Cross-Chunk Entity Resolution & Context Stitching Schemas
class EntityMentionSchema(BaseModel):
    id: Optional[str] = None
    surface_text: str
    entity_id: Optional[str] = None
    chunk_index: int
    start_time: float
    end_time: float
    is_resolved_anaphor: bool = False
    anaphor_type: Optional[str] = None


class EntitySchema(BaseModel):
    id: Optional[str] = None
    name: str
    canonical_name: str
    entity_type: str
    mention_count: int = 1
    first_seen_timestamp: Optional[float] = None
    mentions: list[EntityMentionSchema] = []


class StitchedBlockSchema(BaseModel):
    block_index: int
    start_time: float
    end_time: float
    timestamp: str
    text: str
    normalized_text: str
    source_chunk_indices: str
    entity_mentions: list[EntityMentionSchema] = []


class StitchedTranscriptResponse(BaseModel):
    status: str = "success"
    recording_id: str
    total_blocks: int
    blocks: list[StitchedBlockSchema]


class EntityCatalogResponse(BaseModel):
    status: str = "success"
    recording_id: str
    total_entities: int
    entities: list[EntitySchema]


class StitchingProcessResponse(BaseModel):
    status: str = "success"
    recording_id: str
    total_blocks_created: int
    total_entities_extracted: int
    total_anaphors_resolved: int


class SegmentCorrectionRequest(BaseModel):
    normalized_text: Optional[str] = None
    text: Optional[str] = None
    scope: str = "full"  # "full" or "chunk"


class SegmentCorrectionResponse(BaseModel):
    status: str = "success"
    message: str
    recording_id: str
    segment: dict[str, Any]
    scope: str = "full"
    recomputed: dict[str, bool]
    intelligence: Optional[dict[str, Any]] = None
