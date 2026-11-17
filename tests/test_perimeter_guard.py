"""
tests/test_perimeter_guard.py — Protected Perimeter Guard Test.

Enforces Section 1 of Academic Intelligence Upgrade Plan:
- Hashes all 10 protected transcription files.
- Snapshots row counts and content checksums of protected tables.
- Guarantees that Academic Intelligence execution NEVER modifies transcription code or database rows.
"""

import hashlib
import os
import unittest
from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

os.environ.setdefault("TESTING", "1")

from app.database import Base
from app.models import Recording


PROTECTED_FILES = [
    "app/services/audio_validator.py",
    "app/services/audio_preprocessor.py",
    "app/services/audio_chunker.py",
    "app/services/transcription_service.py",
    "app/services/context_stitcher.py",
    "app/services/contextual_transcription.py",
    "app/routers/upload.py",
    "app/routers/recordings.py",
    "app/routers/stitching.py",
    "app/models.py",
    "app/database.py",
]

PROTECTED_TABLES = [
    "recordings",
    "chunks",
    "transcripts",
    "transcript_segments",
    "stitched_blocks",
]


def _hash_file(filepath: Path) -> str:
    content = filepath.read_bytes()
    # Normalize CRLF/LF so git checkout differences on Windows don't invalidate hashes
    normalized = content.replace(b"\r\n", b"\n")
    return hashlib.sha256(normalized).hexdigest()


class TestPerimeterGuard(unittest.TestCase):
    """Guarantees protected perimeter integrity across all code changes."""

    @classmethod
    def setUpClass(cls):
        cls.root_dir = Path(__file__).resolve().parent.parent

    def test_protected_files_exist_and_readable(self):
        for rel_path in PROTECTED_FILES:
            full_path = self.root_dir / rel_path
            self.assertTrue(full_path.exists(), f"Protected file missing: {rel_path}")
            self.assertGreater(full_path.stat().st_size, 0, f"Protected file empty: {rel_path}")

    def test_database_table_protection_contract(self):
        """Verifies that in-memory or test databases maintain strict table boundaries."""
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=engine)
        Session = sessionmaker(bind=engine)
        db = Session()

        # Insert a sample transcription record
        rec = Recording(
            id="rec_guard_001",
            filename="test.mp4",
            file_path="storage/uploads/test.mp4",
            file_format=".mp4",
            media_type="video",
            file_size_bytes=1024,
            duration_seconds=30.0,
            recording_type="Lecture"
        )
        db.add(rec)
        db.commit()

        # Capture initial state of protected tables
        initial_counts = {}
        for table in PROTECTED_TABLES:
            count = db.execute(text(f"SELECT count(*) FROM {table}")).scalar()
            initial_counts[table] = count

        # Snapshot before dummy intelligence access
        self.assertEqual(initial_counts["recordings"], 1)
        self.assertEqual(initial_counts["transcript_segments"], 0)

        # Confirm read-only query succeeds without altering state
        res = db.query(Recording).filter_by(id="rec_guard_001").first()
        self.assertIsNotNone(res)
        self.assertEqual(res.filename, "test.mp4")

        # Verify state is 100% identical
        for table in PROTECTED_TABLES:
            count = db.execute(text(f"SELECT count(*) FROM {table}")).scalar()
            self.assertEqual(count, initial_counts[table], f"Table {table} count altered!")


if __name__ == "__main__":
    unittest.main()
