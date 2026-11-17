"""
SQLAlchemy ORM models for Recordings and downstream intelligence pipeline stages.
"""

from datetime import datetime, timezone
import uuid
from sqlalchemy import (
    Column,
    String,
    Float,
    Integer,
    DateTime,
    Text,
    ForeignKey,
    Boolean,
    LargeBinary,
    Enum as SQLEnum
)
from sqlalchemy.orm import relationship
from app.database import Base


def generate_uuid():
    return str(uuid.uuid4())


class RecordingStatus:
    UPLOADED = "UPLOADED"
    PREPROCESSED = "PREPROCESSED"
    CHUNKING = "CHUNKING"
    CHUNKED = "CHUNKED"
    TRANSCRIBING = "TRANSCRIBING"
    TRANSCRIBED = "TRANSCRIBED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class Recording(Base):
    __tablename__ = "recordings"

    id = Column(String(36), primary_key=True, default=generate_uuid, index=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    file_format = Column(String(20), nullable=False)       # e.g., ".mp3", ".mp4"
    media_type = Column(String(20), nullable=False)        # "audio" or "video"
    file_size_bytes = Column(Integer, nullable=False)
    duration_seconds = Column(Float, nullable=False)
    preprocessed_file_path = Column(String(512), nullable=True)
    sample_rate = Column(Integer, nullable=True, default=16000)
    channels = Column(Integer, nullable=True, default=1)
    speech_duration_seconds = Column(Float, nullable=True)
    silence_duration_seconds = Column(Float, nullable=True)
    status = Column(String(50), default=RecordingStatus.UPLOADED, nullable=False)
    error_message = Column(Text, nullable=True)
    topic = Column(String(512), nullable=True)
    recording_type = Column(String(50), default="Lecture", nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False
    )

    # Relationships for downstream phases (Phases 3 to 11)
    chunks = relationship("Chunk", back_populates="recording", cascade="all, delete-orphan")
    transcripts = relationship("Transcript", back_populates="recording", cascade="all, delete-orphan")
    intelligence_reports = relationship("IntelligenceReport", back_populates="recording", cascade="all, delete-orphan")
    segment_embeddings = relationship("SegmentEmbedding", back_populates="recording", cascade="all, delete-orphan")
    entities = relationship("Entity", back_populates="recording", cascade="all, delete-orphan")
    entity_mentions = relationship("EntityMention", back_populates="recording", cascade="all, delete-orphan")
    stitched_blocks = relationship("StitchedBlock", back_populates="recording", cascade="all, delete-orphan")
    conversation_sessions = relationship("ConversationSession", back_populates="recording", cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(String(36), primary_key=True, default=generate_uuid, index=True)
    recording_id = Column(String(36), ForeignKey("recordings.id"), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)
    duration = Column(Float, nullable=False)
    file_path = Column(String(512), nullable=False)
    overlap_start = Column(Float, default=0.0)
    overlap_end = Column(Float, default=0.0)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    recording = relationship("Recording", back_populates="chunks")


class Transcript(Base):
    __tablename__ = "transcripts"

    id = Column(String(36), primary_key=True, default=generate_uuid, index=True)
    recording_id = Column(String(36), ForeignKey("recordings.id"), nullable=False, index=True)
    original_transcript = Column(Text, nullable=False)     # As spoken/converted
    normalized_transcript = Column(Text, nullable=False)   # For internal NLP matching
    urdu_transcript = Column(Text, nullable=True)          # Assembled Perso-Arabic Urdu
    english_transcript = Column(Text, nullable=True)       # Assembled English translation
    language_summary = Column(String(255), nullable=True)  # e.g., "en: 60%, ur: 40%"
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    recording = relationship("Recording", back_populates="transcripts")
    segments = relationship("TranscriptSegment", back_populates="transcript", cascade="all, delete-orphan")


class TranscriptSegment(Base):
    __tablename__ = "transcript_segments"

    id = Column(String(36), primary_key=True, default=generate_uuid, index=True)
    transcript_id = Column(String(36), ForeignKey("transcripts.id"), nullable=False, index=True)
    recording_id = Column(String(36), ForeignKey("recordings.id"), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    segment_index = Column(Integer, nullable=False)
    start_time = Column(Float, nullable=False)   # Global recording timeline start
    end_time = Column(Float, nullable=False)     # Global recording timeline end
    text = Column(Text, nullable=False)          # Whisper raw unmodified output
    normalized_text = Column(Text, nullable=True) # Normalized Roman Urdu text
    urdu_text = Column(Text, nullable=True)      # Perso-Arabic Urdu transliteration
    english_text = Column(Text, nullable=True)   # English translation (pivot for downstream NLP)
    translation_source = Column(String(50), nullable=True) # "passthrough", "google_translate", "local_llm", "dictionary"
    translation_confidence = Column(Float, nullable=True)  # Translation confidence score
    language = Column(String(10), nullable=True) # Detected language code (e.g., "en", "ur")
    confidence = Column(Float, nullable=True)    # Probability / logprob
    no_speech_prob = Column(Float, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    transcript = relationship("Transcript", back_populates="segments")
    embedding = relationship("SegmentEmbedding", back_populates="segment", uselist=False, cascade="all, delete-orphan")
    entity_mentions = relationship("EntityMention", back_populates="segment", cascade="all, delete-orphan")


class TranscriptASRAudit(Base):
    """Preserves unmodified raw Whisper chunk outputs and rejected/pruned segments per transcript."""
    __tablename__ = "transcript_asr_audits"

    id = Column(String(36), primary_key=True, default=generate_uuid, index=True)
    transcript_id = Column(String(36), ForeignKey("transcripts.id"), nullable=False, index=True)
    recording_id = Column(String(36), ForeignKey("recordings.id"), nullable=False, index=True)
    raw_whisper_chunks_json = Column(Text, nullable=False)
    rejected_segments_json = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    transcript = relationship("Transcript", backref="asr_audit")


class IntelligenceReport(Base):
    __tablename__ = "intelligence_reports"

    id = Column(String(36), primary_key=True, default=generate_uuid, index=True)
    recording_id = Column(String(36), ForeignKey("recordings.id"), nullable=False, index=True)
    transcript_id = Column(String(36), ForeignKey("transcripts.id"), nullable=False, index=True)
    summary_short = Column(Text, nullable=False)
    summary_detailed = Column(Text, nullable=False)
    lecture_notes_json = Column(Text, nullable=False)    # JSON string: topics, definitions, key concepts
    important_points_json = Column(Text, nullable=False) # JSON string: points with exam_relevance & signal phrase
    assignments_json = Column(Text, nullable=False)      # JSON string: extracted assignments with confidence
    action_items_json = Column(Text, nullable=False)     # JSON string: decisions & action items with owner/deadline
    quizzes_json = Column(Text, nullable=False)          # JSON string: practice quiz questions
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    recording = relationship("Recording", back_populates="intelligence_reports")
    transcript = relationship("Transcript")


class SegmentEmbedding(Base):
    __tablename__ = "segment_embeddings"

    id = Column(String(36), primary_key=True, default=generate_uuid, index=True)
    recording_id = Column(String(36), ForeignKey("recordings.id"), nullable=False, index=True)
    segment_id = Column(String(36), ForeignKey("transcript_segments.id"), nullable=False, index=True)
    embedding_vector = Column(LargeBinary, nullable=False) # Serialized float32 NumPy array
    dimensions = Column(Integer, default=384, nullable=False)
    text_hash = Column(String(64), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    recording = relationship("Recording", back_populates="segment_embeddings")
    segment = relationship("TranscriptSegment", back_populates="embedding")


class Entity(Base):
    __tablename__ = "entities"

    id = Column(String(36), primary_key=True, default=generate_uuid, index=True)
    recording_id = Column(String(36), ForeignKey("recordings.id"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    canonical_name = Column(String(255), nullable=False)
    entity_type = Column(String(50), nullable=False) # PERSON, CONCEPT, TOOL_TECH, DEADLINE_EVENT, ORGANIZATION
    mention_count = Column(Integer, default=1, nullable=False)
    first_seen_timestamp = Column(Float, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    recording = relationship("Recording", back_populates="entities")
    mentions = relationship("EntityMention", back_populates="entity", cascade="all, delete-orphan")


class EntityMention(Base):
    __tablename__ = "entity_mentions"

    id = Column(String(36), primary_key=True, default=generate_uuid, index=True)
    recording_id = Column(String(36), ForeignKey("recordings.id"), nullable=False, index=True)
    entity_id = Column(String(36), ForeignKey("entities.id"), nullable=False, index=True)
    segment_id = Column(String(36), ForeignKey("transcript_segments.id"), nullable=True, index=True)
    chunk_index = Column(Integer, nullable=False)
    surface_text = Column(String(255), nullable=False)
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)
    is_resolved_anaphor = Column(Boolean, default=False, nullable=False)
    anaphor_type = Column(String(50), nullable=True) # e.g. "pronoun_he", "pronoun_woh"
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    recording = relationship("Recording", back_populates="entity_mentions")
    entity = relationship("Entity", back_populates="mentions")
    segment = relationship("TranscriptSegment", back_populates="entity_mentions")


class StitchedBlock(Base):
    __tablename__ = "stitched_blocks"

    id = Column(String(36), primary_key=True, default=generate_uuid, index=True)
    recording_id = Column(String(36), ForeignKey("recordings.id"), nullable=False, index=True)
    block_index = Column(Integer, nullable=False)
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)
    text = Column(Text, nullable=False)
    normalized_text = Column(Text, nullable=True)
    source_chunk_indices = Column(String(255), nullable=False) # e.g. "0,1"
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    recording = relationship("Recording", back_populates="stitched_blocks")


class ConversationSession(Base):
    __tablename__ = "conversation_sessions"

    id = Column(String(36), primary_key=True, default=generate_uuid, index=True)
    recording_id = Column(String(36), ForeignKey("recordings.id"), nullable=True, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    recording = relationship("Recording", back_populates="conversation_sessions")
    turns = relationship("ConversationTurn", back_populates="session", cascade="all, delete-orphan", order_by="ConversationTurn.created_at.asc()")


class ConversationTurn(Base):
    __tablename__ = "conversation_turns"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(36), ForeignKey("conversation_sessions.id"), nullable=False, index=True)
    role = Column(String(20), nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    session = relationship("ConversationSession", back_populates="turns")




