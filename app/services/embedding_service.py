"""
app/services/embedding_service.py — Cross-Lingual Vector Embedding & Semantic Search Service.

Uses sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (384 dimensions)
to generate dense embeddings, persist vector blobs in SQLite, and perform dense
semantic search & grounded Q&A across English, Roman Urdu, and Urdu code-mixed transcripts.
"""

import hashlib
import numpy as np
import logging
from typing import Optional, Any
from sqlalchemy.orm import Session

from app.models import Recording, Transcript, TranscriptSegment, SegmentEmbedding
from app.config import SEMANTIC_SEARCH_THRESHOLD, SEMANTIC_QA_THRESHOLD

logger = logging.getLogger(__name__)


class EmbeddingException(Exception):
    """Base exception for embedding and semantic search failures."""
    pass


class EmbeddingModelManager:
    """Singleton manager for the multilingual embedding model using PyTorch and Transformers."""
    _tokenizer: Optional[Any] = None
    _model: Optional[Any] = None
    _device: str = "cpu"
    MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    DIMENSIONS = 384

    @classmethod
    def _load(cls):
        if cls._model is None:
            import torch
            from transformers import AutoTokenizer, AutoModel
            from app.config import EMBEDDING_DEVICE
            cls._device = "cuda" if (EMBEDDING_DEVICE != "cpu" and torch.cuda.is_available()) else "cpu"
            logger.info(f"Loading embedding model via transformers: {cls.MODEL_NAME} on {cls._device}")
            dtype = torch.float16 if cls._device == "cuda" else torch.float32
            try:
                cls._tokenizer = AutoTokenizer.from_pretrained(cls.MODEL_NAME, local_files_only=True)
                cls._model = AutoModel.from_pretrained(cls.MODEL_NAME, torch_dtype=dtype, local_files_only=True)
            except Exception:
                try:
                    fallback_model = "BAAI/bge-m3"
                    cls._tokenizer = AutoTokenizer.from_pretrained(fallback_model, local_files_only=True)
                    cls._model = AutoModel.from_pretrained(fallback_model, torch_dtype=dtype, local_files_only=True)
                    logger.info(f"Using locally cached {fallback_model} on {cls._device} for instant GPU embeddings")
                except Exception:
                    cls._tokenizer = AutoTokenizer.from_pretrained(cls.MODEL_NAME)
                    cls._model = AutoModel.from_pretrained(cls.MODEL_NAME, torch_dtype=dtype)
            cls._model.eval()
            if cls._device == "cuda":
                cls._model.to("cuda")

    @classmethod
    def get_model(cls) -> Any:
        cls._load()
        return cls._model

    @classmethod
    def encode(cls, texts: list[str] | str, normalize_embeddings: bool = True) -> np.ndarray:
        """Encode text or list of texts into L2-normalized float32 vectors without pyarrow dependency."""
        cls._load()
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return np.empty((0, cls.DIMENSIONS), dtype=np.float32)

        import torch
        import torch.nn.functional as F

        encoded = cls._tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=256
        )
        if cls._device == "cuda" and torch.cuda.is_available():
            encoded = {k: v.to("cuda") for k, v in encoded.items()}
        with torch.no_grad():
            output = cls._model(**encoded)

        token_embeddings = output.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * mask, 1)
        sum_mask = torch.clamp(mask.sum(1), min=1e-9)
        mean_pooled = (sum_embeddings / sum_mask)[:, :cls.DIMENSIONS]

        if normalize_embeddings:
            mean_pooled = F.normalize(mean_pooled, p=2, dim=1)

        return mean_pooled.cpu().numpy().astype(np.float32)


class EmbeddingService:
    """Service for computing, storing, and searching dense transcript embeddings."""

    @classmethod
    def compute_text_hash(cls, text: str) -> str:
        return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

    @classmethod
    def generate_and_store_embeddings(
        cls, recording_id: str, db: Session, force_regenerate: bool = False
    ) -> int:
        """
        Generate and persist 384-dimensional embeddings for all segments in a recording.
        Caches embeddings and only recalculates for new/modified segments unless forced.
        """
        recording = db.query(Recording).filter(Recording.id == recording_id).first()
        if not recording:
            raise EmbeddingException(f"Recording '{recording_id}' not found.")

        transcript = db.query(Transcript).filter(Transcript.recording_id == recording_id).first()
        if not transcript:
            raise EmbeddingException(f"Transcript for recording '{recording_id}' not found.")

        segments = (
            db.query(TranscriptSegment)
            .filter(TranscriptSegment.recording_id == recording_id)
            .order_by(TranscriptSegment.segment_index.asc())
            .all()
        )

        if not segments:
            logger.warning(f"No transcript segments found for recording '{recording_id}'.")
            return 0

        from app.services.intelligence_preprocessor import IntelligencePreprocessor

        # F6: Strictly ONE SegmentEmbedding row per source TranscriptSegment,
        # embedding that segment's repaired `llm_text` (NEVER `match_text`).
        emb_entries = IntelligencePreprocessor.build_per_source_embedding_texts(
            segments,
            topic_hint=getattr(recording, "topic", None),
        )

        # Existing embeddings mapping: segment_id -> SegmentEmbedding
        existing_embeddings = {
            emb.segment_id: emb
            for emb in db.query(SegmentEmbedding).filter(SegmentEmbedding.recording_id == recording_id).all()
        }

        segments_to_encode: list[TranscriptSegment] = []
        hashes_to_store: list[str] = []
        texts_to_encode: list[str] = []

        for seg, entry in zip(segments, emb_entries):
            text = entry["llm_text"] or (seg.normalized_text or seg.text).strip()
            thash = entry["versioned_hash"]

            existing = existing_embeddings.get(seg.id)
            if force_regenerate or existing is None or existing.text_hash != thash:
                segments_to_encode.append(seg)
                hashes_to_store.append(thash)
                try:
                    from roman_urdu_nlp.transliterator import transliterate
                    u_text = transliterate(text)
                    if u_text and u_text.strip() != text:
                        combined = f"{text} {u_text}"
                    else:
                        combined = text
                except Exception:
                    combined = text
                texts_to_encode.append(combined)

        if not segments_to_encode:
            logger.info(f"All {len(segments)} segment embeddings already up to date for recording '{recording_id}'.")
            return len(segments)

        logger.info(f"Computing embeddings for {len(segments_to_encode)} segments (recording '{recording_id}')...")
        vectors = EmbeddingModelManager.encode(texts_to_encode, normalize_embeddings=True)

        for seg, thash, vec in zip(segments_to_encode, hashes_to_store, vectors):
            vec_bytes = vec.tobytes()

            existing = existing_embeddings.get(seg.id)
            if existing:
                existing.embedding_vector = vec_bytes
                existing.dimensions = EmbeddingModelManager.DIMENSIONS
                existing.text_hash = thash
            else:
                new_emb = SegmentEmbedding(
                    recording_id=recording_id,
                    segment_id=seg.id,
                    embedding_vector=vec_bytes,
                    dimensions=EmbeddingModelManager.DIMENSIONS,
                    text_hash=thash
                )
                db.add(new_emb)

        db.commit()
        logger.info(f"Successfully stored embeddings for {len(segments)} segments.")
        return len(segments)

    @classmethod
    def search_semantic(
        cls,
        recording_id: str,
        query: str,
        top_k: int = 5,
        threshold: float = SEMANTIC_SEARCH_THRESHOLD,
        db: Session = None
    ) -> list[dict]:
        """
        Perform dense semantic search for a query across all segments of a recording.
        Returns top-k segments sorted by cosine similarity, filtered by threshold.
        """
        if db is None:
            raise EmbeddingException("Database session required for semantic search.")

        # Ensure embeddings exist AND refresh if stale across edits or preprocessing_version changes
        cls.generate_and_store_embeddings(recording_id, db, force_regenerate=False)

        # Retrieve stored embeddings with their segments
        embeddings_records = (
            db.query(SegmentEmbedding, TranscriptSegment)
            .join(TranscriptSegment, SegmentEmbedding.segment_id == TranscriptSegment.id)
            .filter(SegmentEmbedding.recording_id == recording_id)
            .order_by(TranscriptSegment.segment_index.asc())
            .all()
        )

        if not embeddings_records:
            return []

        # Stack segment vectors
        doc_vectors = np.array([
            np.frombuffer(emb.embedding_vector, dtype=np.float32)
            for emb, _ in embeddings_records
        ])

        # Multi-representation Query Encoding (supports Roman Urdu -> Urdu script bridge)
        try:
            from roman_urdu_nlp.transliterator import transliterate
            urdu_query = transliterate(query)
            if urdu_query and urdu_query.strip() != query.strip():
                q_vectors = EmbeddingModelManager.encode([query, urdu_query], normalize_embeddings=True)
                sim_direct = np.dot(doc_vectors, q_vectors[0])
                sim_translit = np.dot(doc_vectors, q_vectors[1])
                similarities = np.maximum(sim_direct, sim_translit)
            else:
                query_vector = EmbeddingModelManager.encode(query, normalize_embeddings=True)[0]
                similarities = np.dot(doc_vectors, query_vector)
        except Exception:
            query_vector = EmbeddingModelManager.encode(query, normalize_embeddings=True)[0]
            similarities = np.dot(doc_vectors, query_vector)

        results = []
        for (emb, seg), sim in zip(embeddings_records, similarities):
            score = float(sim)
            if score >= threshold:
                results.append({
                    "segment_id": seg.id,
                    "chunk_index": seg.chunk_index,
                    "segment_index": seg.segment_index,
                    "start_time": seg.start_time,
                    "end_time": seg.end_time,
                    "timestamp": f"[{seg.start_time:.2f}s - {seg.end_time:.2f}s]",
                    "text": seg.text,
                    "normalized_text": seg.normalized_text or seg.text,
                    "similarity_score": round(score, 4)
                })

        # Sort by similarity descending
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results[:top_k]

    @classmethod
    def answer_semantic_qa(
        cls,
        recording_id: str,
        question: str,
        threshold: float = SEMANTIC_QA_THRESHOLD,
        db: Session = None
    ) -> dict:
        """
        Grounded semantic Q&A using dense embeddings.
        Resolves paraphrases and cross-lingual questions with source timestamp citations.
        Strictly returns 'Not found in the recording' if no segment meets the similarity threshold.
        """
        # Search with baseline threshold 0.0 to capture true raw score
        all_results = cls.search_semantic(
            recording_id=recording_id,
            query=question,
            top_k=1,
            threshold=0.0,
            db=db
        )

        if all_results:
            best = all_results[0]
            if best["similarity_score"] >= threshold:
                return {
                    "status": "success",
                    "found": True,
                    "question": question,
                    "answer": best["normalized_text"] or best["text"],
                    "timestamp": best["timestamp"],
                    "start_time": round(best["start_time"], 2),
                    "end_time": round(best["end_time"], 2),
                    "source_segment": best["text"],
                    "similarity_score": best["similarity_score"]
                }
            else:
                return {
                    "status": "success",
                    "found": False,
                    "question": question,
                    "answer": "Not found in the recording",
                    "timestamp": None,
                    "start_time": None,
                    "end_time": None,
                    "source_segment": None,
                    "similarity_score": best["similarity_score"]
                }

        return {
            "status": "success",
            "found": False,
            "question": question,
            "answer": "Not found in the recording",
            "timestamp": None,
            "start_time": None,
            "end_time": None,
            "source_segment": None,
            "similarity_score": 0.0
        }
