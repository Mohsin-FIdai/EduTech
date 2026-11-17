"""
app/services/entity_resolver.py — Cross-chunk entity extraction, alias clustering, and coreference resolution.
"""

from __future__ import annotations
import re
import logging
from sqlalchemy.orm import Session
from app.models import TranscriptSegment, Entity, EntityMention

logger = logging.getLogger("roman_urdu_app.entity_resolver")

# Common Pakistani & English First Names for PERSON extraction
KNOWN_PERSON_NAMES = {
    "ali", "ahmed", "usman", "bilal", "sara", "fatima", "zainab", "hassan",
    "hussain", "ayesha", "hamza", "tariq", "saad", "zubair", "omer", "farhan",
    "naveed", "sana", "hira", "kiran", "maryam", "asif", "kashif", "waqas",
    "imran", "shahid", "kamran", "adnan", "salman", "faisal", "rashid",
    "john", "alice", "bob", "david", "michael", "sarah", "alex"
}

FEMININE_NAMES = {"sara", "fatima", "zainab", "ayesha", "sana", "hira", "kiran", "maryam", "alice", "sarah"}
MASCULINE_NAMES = {
    "ali", "ahmed", "usman", "bilal", "hassan", "hussain", "hamza", "tariq", "saad",
    "zubair", "omer", "farhan", "naveed", "asif", "kashif", "waqas", "imran",
    "shahid", "kamran", "adnan", "salman", "faisal", "rashid", "john", "bob", "david", "michael", "alex"
}

HONORIFICS = r"(?:Prof\.|Professor|Dr\.|Doctor|Engr\.|Engineer|Sir|Madam|Mr\.|Ms\.|Mrs\.)"

# Technical Concepts & Topics
KNOWN_CONCEPTS = [
    "gradient descent", "linear regression", "logistic regression", "neural network",
    "neural networks", "convolutional neural network", "backpropagation", "support vector machine",
    "random forest", "decision tree", "data preprocessing", "feature scaling", "overfitting",
    "underfitting", "cross validation", "k-means", "tcp handshake", "b-tree", "virtual memory",
    "process scheduling", "loss function", "supervised learning", "unsupervised learning"
]

# Tools, Technologies & Platforms
KNOWN_TOOLS = [
    "lms", "pytorch", "tensorflow", "keras", "scikit-learn", "docker", "kubernetes",
    "postgresql", "mysql", "mongodb", "github", "gitlab", "jupyter", "google colab",
    "linux", "vs code", "zoom", "slack", "teams", "python", "fastapi"
]

# Deadlines & Events
KNOWN_EVENTS = [
    "assignment 1", "assignment 2", "assignment 3", "assignment 4", "assignment 5",
    "quiz 1", "quiz 2", "quiz 3", "midterm exam", "final exam", "lab 1", "lab 2",
    "project proposal", "sprint review", "standup meeting", "midterm", "final"
]


class EntityResolver:
    """
    Identifies entities across chunks, clusters aliases, and resolves anaphoric pronouns.
    """

    @classmethod
    def extract_raw_mentions(cls, segment: TranscriptSegment) -> list[dict]:
        """
        Extracts entity mentions from a single transcript segment.
        """
        text = segment.text
        norm_text = segment.normalized_text or text
        mentions: list[dict] = []

        # 1. PERSON with Honorifics (e.g., "Professor Tariq", "Dr. Ahmed")
        hon_pattern = rf"\b({HONORIFICS}\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b"
        for match in re.finditer(hon_pattern, text):
            surface = match.group(1).strip()
            # Clean canonical name by extracting root name
            root_name = re.sub(rf"^{HONORIFICS}\s+", "", surface).strip()
            mentions.append({
                "surface_text": surface,
                "root_name": root_name,
                "canonical_name": surface,
                "entity_type": "PERSON",
                "start_time": segment.start_time,
                "end_time": segment.end_time,
                "chunk_index": segment.chunk_index,
                "segment_id": segment.id,
                "is_resolved_anaphor": False,
                "anaphor_type": None
            })

        # 2. Standalone Known Person Names (e.g., "Bilal", "Sara")
        for word_match in re.finditer(r"\b([A-Z][a-z]+|[a-z]+)\b", norm_text):
            word = word_match.group(1).strip()
            w_lower = word.lower()
            if w_lower in KNOWN_PERSON_NAMES:
                # Avoid re-adding if already part of an honorific match
                already_matched = any(
                    w_lower in m["surface_text"].lower() and m["entity_type"] == "PERSON"
                    for m in mentions
                )
                if not already_matched:
                    cap_name = word.capitalize()
                    mentions.append({
                        "surface_text": word,
                        "root_name": cap_name,
                        "canonical_name": cap_name,
                        "entity_type": "PERSON",
                        "start_time": segment.start_time,
                        "end_time": segment.end_time,
                        "chunk_index": segment.chunk_index,
                        "segment_id": segment.id,
                        "is_resolved_anaphor": False,
                        "anaphor_type": None
                    })

        # 3. DEADLINE_EVENT
        for ev in KNOWN_EVENTS:
            pattern = rf"\b({re.escape(ev)})\b"
            for match in re.finditer(pattern, text, re.IGNORECASE):
                surface = match.group(1).strip()
                mentions.append({
                    "surface_text": surface,
                    "root_name": ev.title(),
                    "canonical_name": ev.title(),
                    "entity_type": "DEADLINE_EVENT",
                    "start_time": segment.start_time,
                    "end_time": segment.end_time,
                    "chunk_index": segment.chunk_index,
                    "segment_id": segment.id,
                    "is_resolved_anaphor": False,
                    "anaphor_type": None
                })

        # 4. TOOL_TECH
        for tool in KNOWN_TOOLS:
            pattern = rf"\b({re.escape(tool)})\b"
            for match in re.finditer(pattern, text, re.IGNORECASE):
                surface = match.group(1).strip()
                # Determine display name
                display_name = surface.upper() if len(surface) <= 3 else surface.title()
                mentions.append({
                    "surface_text": surface,
                    "root_name": display_name,
                    "canonical_name": display_name,
                    "entity_type": "TOOL_TECH",
                    "start_time": segment.start_time,
                    "end_time": segment.end_time,
                    "chunk_index": segment.chunk_index,
                    "segment_id": segment.id,
                    "is_resolved_anaphor": False,
                    "anaphor_type": None
                })

        # 5. CONCEPT
        for concept in KNOWN_CONCEPTS:
            pattern = rf"\b({re.escape(concept)})\b"
            for match in re.finditer(pattern, text, re.IGNORECASE):
                surface = match.group(1).strip()
                mentions.append({
                    "surface_text": surface,
                    "root_name": concept.title(),
                    "canonical_name": concept.title(),
                    "entity_type": "CONCEPT",
                    "start_time": segment.start_time,
                    "end_time": segment.end_time,
                    "chunk_index": segment.chunk_index,
                    "segment_id": segment.id,
                    "is_resolved_anaphor": False,
                    "anaphor_type": None
                })

        return mentions

    @classmethod
    def resolve_coreferences_and_cluster(
        cls,
        segments: list[TranscriptSegment]
    ) -> tuple[list[dict], list[dict]]:
        """
        Extracts mentions across all segments, clusters aliases, and performs cross-chunk
        anaphora resolution.
        Returns:
            canonical_entities: list of dicts with entity info and mention counts
            all_mentions: list of dicts representing all individual occurrences
        """
        sorted_segs = sorted(segments, key=lambda s: (s.chunk_index, s.segment_index, s.start_time))

        # Collect raw named entity mentions
        raw_mentions: list[dict] = []
        for seg in sorted_segs:
            raw_mentions.extend(cls.extract_raw_mentions(seg))

        # Step 1: Cluster Named Entities by Root Name & Type
        clusters: dict[str, dict] = {} # cluster_key -> {canonical_name, entity_type, mentions, first_seen}

        for m in raw_mentions:
            root = m["root_name"].lower()
            etype = m["entity_type"]
            key = f"{etype}:{root}"

            if key not in clusters:
                clusters[key] = {
                    "name": m["surface_text"],
                    "canonical_name": m["canonical_name"],
                    "entity_type": etype,
                    "first_seen_timestamp": m["start_time"],
                    "mentions": []
                }
            else:
                # Update canonical name if current mention has honorific / more detailed title
                if len(m["canonical_name"]) > len(clusters[key]["canonical_name"]):
                    clusters[key]["canonical_name"] = m["canonical_name"]

            clusters[key]["mentions"].append(m)

        # Step 2: Cross-Chunk Anaphora Resolution
        # Pronoun patterns: English + Roman Urdu
        # (pattern, gender/type, anaphor_label)
        ANAPHOR_PATTERNS = [
            (r"\b(he|him|his)\b", "masculine", "pronoun_he"),
            (r"\b(she|her|hers)\b", "feminine", "pronoun_she"),
            (r"\b(woh|us\s+ne|unhon\s+ne|unka|unke|unki)\b", "urdu_person", "pronoun_urdu_person"),
        ]

        resolved_anaphor_mentions: list[dict] = []

        for seg in sorted_segs:
            seg_text = (seg.normalized_text or seg.text).lower()

            for pattern, target_gender, label in ANAPHOR_PATTERNS:
                for match in re.finditer(pattern, seg_text):
                    pronoun_str = match.group(1)

                    # Look back in history up to 3 chunks or 60 seconds
                    candidate_clusters = []
                    for c_key, c_data in clusters.items():
                        if c_data["entity_type"] != "PERSON":
                            continue

                        # Find most recent mention of this entity before this segment
                        past_mentions = [
                            pm for pm in c_data["mentions"]
                            if pm["start_time"] < seg.start_time and (seg.start_time - pm["end_time"]) <= 60.0
                        ]
                        if past_mentions:
                            latest_pm = max(past_mentions, key=lambda pm: pm["end_time"])
                            root_lower = c_data["canonical_name"].lower()

                            # Gender / Number compatibility
                            is_compatible = False
                            if target_gender == "masculine" and any(m in root_lower for m in MASCULINE_NAMES):
                                is_compatible = True
                            elif target_gender == "feminine" and any(f in root_lower for f in FEMININE_NAMES):
                                is_compatible = True
                            elif target_gender == "urdu_person":
                                is_compatible = True

                            if is_compatible:
                                candidate_clusters.append((c_key, latest_pm))

                    if not candidate_clusters:
                        continue

                    # Sort by temporal recency
                    candidate_clusters.sort(key=lambda item: item[1]["end_time"], reverse=True)

                    # Ambiguity check: if top 2 candidates occurred in the same sentence or same chunk without distance gap
                    if len(candidate_clusters) >= 2:
                        top_time = candidate_clusters[0][1]["end_time"]
                        second_time = candidate_clusters[1][1]["end_time"]
                        if abs(top_time - second_time) < 2.0:
                            # Ambiguous antecedent -> do not hallucinate single resolution
                            logger.info(f"Ambiguous pronoun '{pronoun_str}' at {seg.start_time:.2f}s between multiple candidates. Skipping.")
                            continue

                    # Resolve to nearest candidate
                    chosen_key, chosen_pm = candidate_clusters[0]
                    resolved_mention = {
                        "surface_text": pronoun_str,
                        "root_name": clusters[chosen_key]["canonical_name"],
                        "canonical_name": clusters[chosen_key]["canonical_name"],
                        "entity_type": "PERSON",
                        "start_time": seg.start_time,
                        "end_time": seg.end_time,
                        "chunk_index": seg.chunk_index,
                        "segment_id": seg.id,
                        "is_resolved_anaphor": True,
                        "anaphor_type": label
                    }
                    clusters[chosen_key]["mentions"].append(resolved_mention)
                    resolved_anaphor_mentions.append(resolved_mention)

        # Assemble final canonical entities list
        canonical_entities: list[dict] = []
        all_mentions: list[dict] = []

        for key, c_data in clusters.items():
            # Sort mentions chronologically
            c_mentions = sorted(c_data["mentions"], key=lambda m: (m["chunk_index"], m["start_time"]))
            entity_dict = {
                "name": c_data["name"],
                "canonical_name": c_data["canonical_name"],
                "entity_type": c_data["entity_type"],
                "mention_count": len(c_mentions),
                "first_seen_timestamp": round(c_data["first_seen_timestamp"], 2) if c_data["first_seen_timestamp"] is not None else None,
                "mentions": c_mentions
            }
            canonical_entities.append(entity_dict)
            all_mentions.extend(c_mentions)

        # Sort entities by mention frequency descending
        canonical_entities.sort(key=lambda e: e["mention_count"], reverse=True)
        return canonical_entities, all_mentions

    @classmethod
    def process_and_persist_entities(cls, recording_id: str, db: Session) -> dict:
        """
        Extracts entities, clusters aliases, resolves cross-chunk coreferences,
        and persists Entity and EntityMention records in the database.
        """
        segments = (
            db.query(TranscriptSegment)
            .filter(TranscriptSegment.recording_id == recording_id)
            .order_by(TranscriptSegment.chunk_index.asc(), TranscriptSegment.segment_index.asc())
            .all()
        )

        if not segments:
            logger.warning(f"No segments found for entity resolution in recording '{recording_id}'.")
            return {
                "total_entities": 0,
                "total_mentions": 0,
                "total_anaphors_resolved": 0
            }

        # Clear existing entities and mentions for idempotent re-runs
        db.query(Entity).filter(Entity.recording_id == recording_id).delete()
        db.query(EntityMention).filter(EntityMention.recording_id == recording_id).delete()

        canonical_entities, all_mentions = cls.resolve_coreferences_and_cluster(segments)

        anaphors_count = 0
        for ent_data in canonical_entities:
            entity_record = Entity(
                recording_id=recording_id,
                name=ent_data["name"],
                canonical_name=ent_data["canonical_name"],
                entity_type=ent_data["entity_type"],
                mention_count=ent_data["mention_count"],
                first_seen_timestamp=ent_data["first_seen_timestamp"]
            )
            db.add(entity_record)
            db.flush() # populate entity_record.id

            for m in ent_data["mentions"]:
                if m["is_resolved_anaphor"]:
                    anaphors_count += 1
                mention_record = EntityMention(
                    recording_id=recording_id,
                    entity_id=entity_record.id,
                    segment_id=m.get("segment_id"),
                    chunk_index=m["chunk_index"],
                    surface_text=m["surface_text"],
                    start_time=m["start_time"],
                    end_time=m["end_time"],
                    is_resolved_anaphor=m["is_resolved_anaphor"],
                    anaphor_type=m.get("anaphor_type")
                )
                db.add(mention_record)

        db.commit()
        logger.info(
            f"Persisted {len(canonical_entities)} entities ({len(all_mentions)} mentions, {anaphors_count} anaphors resolved) "
            f"for recording '{recording_id}'."
        )

        return {
            "total_entities": len(canonical_entities),
            "total_mentions": len(all_mentions),
            "total_anaphors_resolved": anaphors_count
        }
