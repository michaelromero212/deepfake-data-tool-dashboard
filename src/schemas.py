"""
Pydantic schemas for deepfake-data-forge.
All data contracts used across the pipeline are defined here.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class MediaType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class Label(str, Enum):
    REAL = "real"
    SYNTHETIC = "synthetic"
    UNKNOWN = "unknown"


class LabelSource(str, Enum):
    """How the label was assigned — critical for dataset provenance."""
    DIRECTORY_CONVENTION = "directory_convention"  # /real/ or /synthetic/ folder
    FILENAME_CONVENTION = "filename_convention"    # _real or _fake suffix
    MANIFEST_ANNOTATION = "manifest_annotation"   # pre-labelled manifest
    INFERRED = "inferred"                          # derived from detection score


class ImageMetadata(BaseModel):
    width: int
    height: int
    channels: int
    format: str


class VideoMetadata(BaseModel):
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float
    codec: Optional[str] = None


class AudioMetadata(BaseModel):
    sample_rate: int
    duration_seconds: float
    channels: int
    format: str


class FileMetadata(BaseModel):
    file_name: str
    file_path: str
    media_type: MediaType
    file_size_bytes: int
    sha256_hash: str
    media_metadata: ImageMetadata | VideoMetadata | AudioMetadata


class DetectionResult(BaseModel):
    """Result of running a pre-trained deepfake detector on a sample."""
    model_name: str
    detection_score: float = Field(ge=0.0, le=1.0, description="0=real, 1=synthetic")
    model_version: str
    inference_time_ms: float


class ValidationStatus(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class ValidationIssue(BaseModel):
    issue_type: str
    message: str
    severity: ValidationStatus


class Sample(BaseModel):
    """A single dataset sample — the atomic unit of the manifest."""
    sample_id: str
    file_path: str
    media_type: MediaType
    label: Label
    label_source: LabelSource
    metadata: FileMetadata
    detection_result: Optional[DetectionResult] = None
    validation_status: ValidationStatus = ValidationStatus.PASS
    validation_issues: list[ValidationIssue] = Field(default_factory=list)
    processed_path: Optional[str] = None


class DatasetManifest(BaseModel):
    """Top-level manifest — the final output artifact of the pipeline."""
    dataset_name: str
    dataset_version: str  # SHA256 of all sample hashes combined
    pipeline_version: str = "0.1.0"
    created_at: str
    total_samples: int
    label_distribution: dict[str, int]
    media_type_distribution: dict[str, int]
    samples: list[Sample]

    @field_validator("label_distribution", "media_type_distribution")
    @classmethod
    def non_empty_distribution(cls, v: dict) -> dict:
        if not v:
            raise ValueError("Distribution must not be empty")
        return v


class ValidationReport(BaseModel):
    """Summary report produced by the validation stage."""
    total_samples: int
    passed: int
    warned: int
    failed: int
    pass_rate: float
    corruption_rate: float
    schema_violation_count: int
    issues_by_type: dict[str, int]
    failed_samples: list[str]
