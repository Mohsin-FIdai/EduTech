"""
app/routers/stitching.py — FastAPI endpoints for cross-chunk boundary stitching and entity resolution.
"""

from __future__ import annotations
import logging
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Recording, StitchedBlock, Entity, EntityMention
from app.schemas import (
    StitchingProcessResponse,
    StitchedTranscriptResponse,
    StitchedBlockSchema,
    EntityCatalogResponse,
    EntitySchema,
    EntityMentionSchema
)
from app.services.context_stitcher import ContextStitcher
from app.services.entity_resolver import EntityResolver

logger = logging.getLogger("roman_urdu_app.stitching_router")

router = APIRouter(prefix="/api/recordings", tags=["Context Stitching & Entity Resolution"])


@router.post("/{recording_id}/stitch", response_model=StitchingProcessResponse)
def stitch_recording_transcript(recording_id: str, db: Session = Depends(get_db)):
    """
    Executes cross-chunk sentence boundary stitching and entity resolution.
    Heals split clauses, prunes repetition, clusters aliases, and resolves coreferences.
    """
    rec = db.query(Recording).filter(Recording.id == recording_id).first()
    if not rec:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Recording with ID '{recording_id}' not found."
        )

    try:
        blocks_data = ContextStitcher.process_and_persist_stitched_blocks(recording_id, db)
        ent_stats = EntityResolver.process_and_persist_entities(recording_id, db)

        return StitchingProcessResponse(
            status="success",
            recording_id=recording_id,
            total_blocks_created=len(blocks_data),
            total_entities_extracted=ent_stats["total_entities"],
            total_anaphors_resolved=ent_stats["total_anaphors_resolved"]
        )
    except Exception as e:
        logger.error(f"Error executing stitching pipeline for recording '{recording_id}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stitch recording transcript: {str(e)}"
        )


@router.get("/{recording_id}/stitched-transcript", response_model=StitchedTranscriptResponse)
def get_stitched_transcript(recording_id: str, db: Session = Depends(get_db)):
    """
    Returns the stitched continuous transcript blocks with attached entity mentions.
    """
    rec = db.query(Recording).filter(Recording.id == recording_id).first()
    if not rec:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Recording with ID '{recording_id}' not found."
        )

    blocks = (
        db.query(StitchedBlock)
        .filter(StitchedBlock.recording_id == recording_id)
        .order_by(StitchedBlock.block_index.asc())
        .all()
    )

    if not blocks:
        # Run stitching on-demand
        ContextStitcher.process_and_persist_stitched_blocks(recording_id, db)
        EntityResolver.process_and_persist_entities(recording_id, db)
        blocks = (
            db.query(StitchedBlock)
            .filter(StitchedBlock.recording_id == recording_id)
            .order_by(StitchedBlock.block_index.asc())
            .all()
        )

    # Fetch entity mentions for this recording
    mentions = (
        db.query(EntityMention)
        .filter(EntityMention.recording_id == recording_id)
        .order_by(EntityMention.start_time.asc())
        .all()
    )

    blocks_schemas: list[StitchedBlockSchema] = []
    for b in blocks:
        # Find mentions falling within this block's timeframe
        b_mentions = [
            EntityMentionSchema(
                id=m.id,
                surface_text=m.surface_text,
                entity_id=m.entity_id,
                chunk_index=m.chunk_index,
                start_time=m.start_time,
                end_time=m.end_time,
                is_resolved_anaphor=m.is_resolved_anaphor,
                anaphor_type=m.anaphor_type
            )
            for m in mentions
            if (b.start_time - 0.5) <= m.start_time and m.end_time <= (b.end_time + 0.5)
        ]

        blocks_schemas.append(
            StitchedBlockSchema(
                block_index=b.block_index,
                start_time=b.start_time,
                end_time=b.end_time,
                timestamp=f"[{b.start_time:.2f}s - {b.end_time:.2f}s]",
                text=b.text,
                normalized_text=b.normalized_text or b.text,
                source_chunk_indices=b.source_chunk_indices,
                entity_mentions=b_mentions
            )
        )

    return StitchedTranscriptResponse(
        status="success",
        recording_id=recording_id,
        total_blocks=len(blocks_schemas),
        blocks=blocks_schemas
    )


@router.get("/{recording_id}/entities", response_model=EntityCatalogResponse)
def get_recording_entities(recording_id: str, db: Session = Depends(get_db)):
    """
    Returns the extracted entity catalog with resolved alias clusters and mention frequencies.
    """
    rec = db.query(Recording).filter(Recording.id == recording_id).first()
    if not rec:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Recording with ID '{recording_id}' not found."
        )

    entities = (
        db.query(Entity)
        .filter(Entity.recording_id == recording_id)
        .order_by(Entity.mention_count.desc())
        .all()
    )

    if not entities:
        # Run entity resolution on-demand
        EntityResolver.process_and_persist_entities(recording_id, db)
        entities = (
            db.query(Entity)
            .filter(Entity.recording_id == recording_id)
            .order_by(Entity.mention_count.desc())
            .all()
        )

    entity_schemas: list[EntitySchema] = []
    for ent in entities:
        mentions = (
            db.query(EntityMention)
            .filter(EntityMention.entity_id == ent.id)
            .order_by(EntityMention.start_time.asc())
            .all()
        )

        m_schemas = [
            EntityMentionSchema(
                id=m.id,
                surface_text=m.surface_text,
                entity_id=m.entity_id,
                chunk_index=m.chunk_index,
                start_time=m.start_time,
                end_time=m.end_time,
                is_resolved_anaphor=m.is_resolved_anaphor,
                anaphor_type=m.anaphor_type
            )
            for m in mentions
        ]

        entity_schemas.append(
            EntitySchema(
                id=ent.id,
                name=ent.name,
                canonical_name=ent.canonical_name,
                entity_type=ent.entity_type,
                mention_count=ent.mention_count,
                first_seen_timestamp=ent.first_seen_timestamp,
                mentions=m_schemas
            )
        )

    return EntityCatalogResponse(
        status="success",
        recording_id=recording_id,
        total_entities=len(entity_schemas),
        entities=entity_schemas
    )
