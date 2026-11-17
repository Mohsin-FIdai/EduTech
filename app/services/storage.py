"""
Secure storage service for managing uploaded and processed files.
"""

from pathlib import Path
import uuid
from fastapi import UploadFile
from app.config import UPLOADS_DIR, PROCESSED_DIR


class StorageService:
    def __init__(self, upload_dir: Path = UPLOADS_DIR, processed_dir: Path = PROCESSED_DIR):
        self.upload_dir = upload_dir
        self.processed_dir = processed_dir
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def generate_unique_filename(self, original_filename: str) -> tuple[str, str, Path]:
        """
        Generate a unique storage filename using UUID4 while preserving extension.
        Returns (recording_id, safe_filename, full_path)
        """
        recording_id = str(uuid.uuid4())
        suffix = Path(original_filename).suffix.lower()
        safe_filename = f"{recording_id}{suffix}"
        file_path = self.upload_dir / safe_filename
        return recording_id, safe_filename, file_path

    async def save_upload_file(self, upload_file: UploadFile, target_path: Path) -> int:
        """
        Stream an uploaded file to disk asynchronously and return written byte count.
        """
        bytes_written = 0
        chunk_size = 1024 * 1024  # 1 MB chunks

        # Reset cursor
        await upload_file.seek(0)

        with open(target_path, "wb") as buffer:
            while chunk := await upload_file.read(chunk_size):
                buffer.write(chunk)
                bytes_written += len(chunk)

        return bytes_written

    def delete_file(self, file_path: Path | str) -> bool:
        """Safely delete a file if it exists."""
        p = Path(file_path)
        if p.exists() and p.is_file():
            p.unlink()
            return True
        return False
