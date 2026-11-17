"""
tests/conftest.py — Test configuration and automated database + storage isolation fixture.
Ensures that tests NEVER write to the live application database (lecture_system.db)
and automatically cleans up any temporary media files created during test runs.
"""

import os
import shutil
import pytest

# Set testing flag
os.environ["TESTING"] = "1"

from app.config import STORAGE_DIR, UPLOADS_DIR, PROCESSED_DIR, CHUNKS_DIR
from app.database import Base, engine, SessionLocal, get_db
from app.main import app


@pytest.fixture(scope="session", autouse=True)
def setup_isolated_test_database():
    """
    Session-scoped fixture:
    1. Snapshots existing storage files before the test session.
    2. Initializes isolated test database schema and overrides FastAPI get_db.
    3. Cleans up isolated test database and any test-generated media files after test run.
    """
    existing_storage_files = {}
    for folder in (UPLOADS_DIR, PROCESSED_DIR, CHUNKS_DIR):
        if folder.exists():
            existing_storage_files[folder] = set(folder.iterdir())
        else:
            existing_storage_files[folder] = set()

    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    def override_get_db():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    yield

    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=engine)
    test_db = STORAGE_DIR / "test_isolated.db"
    try:
        if test_db.exists():
            test_db.unlink(missing_ok=True)
    except Exception:
        pass

    for folder, before_set in existing_storage_files.items():
        if not folder.exists():
            continue
        for item in list(folder.iterdir()):
            if item not in before_set:
                try:
                    if item.is_file():
                        item.unlink(missing_ok=True)
                    elif item.is_dir():
                        shutil.rmtree(item, ignore_errors=True)
                except Exception:
                    pass
