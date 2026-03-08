"""
tests/test_pipeline.py

Core unit tests covering schemas, ingestion label derivation,
metadata hashing, and validation logic.

Run: pytest tests/ -v
"""

from __future__ import annotations

import hashlib
import tempfile
import wave
from pathlib import Path

import numpy as np
import pytest

from src.ingestion import _derive_label, _media_type, ingest_data
from src.metadata import _sha256, compute_dataset_version
from src.schemas import (
    Label,
    LabelSource,
    MediaType,
    ValidationStatus,
)
from src.validation import validate_sample


# ── Ingestion ──────────────────────────────────────────────────────────────

class TestMediaTypeDetection:
    def test_image_extensions(self):
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            assert _media_type(Path(f"sample{ext}")) == MediaType.IMAGE

    def test_video_extensions(self):
        for ext in [".mp4", ".avi", ".mov"]:
            assert _media_type(Path(f"sample{ext}")) == MediaType.VIDEO

    def test_audio_extensions(self):
        for ext in [".wav", ".mp3", ".flac"]:
            assert _media_type(Path(f"sample{ext}")) == MediaType.AUDIO

    def test_unknown_extension(self):
        assert _media_type(Path("sample.txt")) is None
        assert _media_type(Path("sample.pdf")) is None


class TestLabelDerivation:
    def test_directory_real(self):
        label, source = _derive_label(Path("data/raw/real/portrait.jpg"))
        assert label == Label.REAL
        assert source == LabelSource.DIRECTORY_CONVENTION

    def test_directory_synthetic(self):
        label, source = _derive_label(Path("data/raw/synthetic/faceswap.jpg"))
        assert label == Label.SYNTHETIC
        assert source == LabelSource.DIRECTORY_CONVENTION

    def test_directory_fake(self):
        label, source = _derive_label(Path("data/raw/fake/video.mp4"))
        assert label == Label.SYNTHETIC
        assert source == LabelSource.DIRECTORY_CONVENTION

    def test_filename_suffix_real(self):
        label, source = _derive_label(Path("data/raw/unknown/portrait_real.jpg"))
        assert label == Label.REAL
        assert source == LabelSource.FILENAME_CONVENTION

    def test_filename_suffix_fake(self):
        label, source = _derive_label(Path("data/raw/unknown/portrait_fake.jpg"))
        assert label == Label.SYNTHETIC
        assert source == LabelSource.FILENAME_CONVENTION

    def test_unknown_label(self):
        label, source = _derive_label(Path("data/raw/misc/unlabelled.jpg"))
        assert label == Label.UNKNOWN
        assert source == LabelSource.INFERRED


class TestIngestion:
    def test_empty_directory(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ingest_data(Path("/nonexistent/path"))

    def test_discovers_supported_files(self, tmp_path):
        (tmp_path / "real").mkdir()
        (tmp_path / "synthetic").mkdir()
        (tmp_path / "real" / "a.jpg").touch()
        (tmp_path / "synthetic" / "b.png").touch()
        (tmp_path / "real" / "ignore.txt").touch()

        result = ingest_data(tmp_path)
        assert len(result) == 2

    def test_label_derived_from_directory(self, tmp_path):
        (tmp_path / "real").mkdir()
        (tmp_path / "synthetic").mkdir()
        (tmp_path / "real" / "a.jpg").touch()
        (tmp_path / "synthetic" / "b.jpg").touch()

        result = ingest_data(tmp_path)
        labels = {f.path.name: f.label for f in result}
        assert labels["a.jpg"] == Label.REAL
        assert labels["b.jpg"] == Label.SYNTHETIC


# ── Metadata ───────────────────────────────────────────────────────────────

class TestHashing:
    def test_sha256_consistent(self, tmp_path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"hello deepfake world")
        h1 = _sha256(f)
        h2 = _sha256(f)
        assert h1 == h2
        assert len(h1) == 64

    def test_sha256_changes_with_content(self, tmp_path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"content_a")
        h1 = _sha256(f)
        f.write_bytes(b"content_b")
        h2 = _sha256(f)
        assert h1 != h2

    def test_dataset_version_deterministic(self):
        hashes = ["abc123", "def456", "ghi789"]
        v1 = compute_dataset_version(hashes)
        v2 = compute_dataset_version(hashes)
        assert v1 == v2
        assert len(v1) == 16

    def test_dataset_version_order_independent(self):
        hashes = ["abc123", "def456", "ghi789"]
        v1 = compute_dataset_version(hashes)
        v2 = compute_dataset_version(list(reversed(hashes)))
        assert v1 == v2

    def test_dataset_version_changes_on_new_sample(self):
        hashes = ["abc123", "def456"]
        v1 = compute_dataset_version(hashes)
        v2 = compute_dataset_version(hashes + ["new000"])
        assert v1 != v2


# ── Validation ─────────────────────────────────────────────────────────────

class TestValidation:
    def _make_sample(self, tmp_path) -> object:
        """Create a minimal valid Sample for testing."""
        from src.schemas import (
            FileMetadata, ImageMetadata, Label, LabelSource,
            MediaType, Sample, ValidationStatus
        )

        # Create a real file for existence checks
        f = tmp_path / "test.jpg"
        f.write_bytes(b"\xff\xd8\xff" + b"\x00" * 100)

        meta = FileMetadata(
            file_name="test.jpg",
            file_path=str(f),
            media_type=MediaType.IMAGE,
            file_size_bytes=103,
            sha256_hash="a" * 64,
            media_metadata=ImageMetadata(width=224, height=224, channels=3, format="JPEG"),
        )
        return Sample(
            sample_id="test01",
            file_path=str(f),
            media_type=MediaType.IMAGE,
            label=Label.REAL,
            label_source=LabelSource.DIRECTORY_CONVENTION,
            metadata=meta,
        )

    def test_valid_sample_passes(self, tmp_path):
        sample = self._make_sample(tmp_path)
        result = validate_sample(sample)
        assert result.validation_status == ValidationStatus.PASS

    def test_missing_file_fails(self, tmp_path):
        from src.schemas import (
            FileMetadata, ImageMetadata, Label, LabelSource,
            MediaType, Sample
        )
        meta = FileMetadata(
            file_name="ghost.jpg",
            file_path="/nonexistent/ghost.jpg",
            media_type=MediaType.IMAGE,
            file_size_bytes=100,
            sha256_hash="a" * 64,
            media_metadata=ImageMetadata(width=224, height=224, channels=3, format="JPEG"),
        )
        sample = Sample(
            sample_id="ghost",
            file_path="/nonexistent/ghost.jpg",
            media_type=MediaType.IMAGE,
            label=Label.REAL,
            label_source=LabelSource.DIRECTORY_CONVENTION,
            metadata=meta,
        )
        result = validate_sample(sample)
        assert result.validation_status == ValidationStatus.FAIL

    def test_unknown_label_warns(self, tmp_path):
        sample = self._make_sample(tmp_path)
        updated = sample.model_copy(update={"label": Label.UNKNOWN})
        result = validate_sample(updated)
        assert result.validation_status == ValidationStatus.WARN
        assert any(i.issue_type == "unknown_label" for i in result.validation_issues)
