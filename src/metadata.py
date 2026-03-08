"""
metadata.py — Metadata extraction stage.

Extracts file-level and media-level metadata for each sample.
SHA-256 hashing enables dataset versioning and deduplication.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from loguru import logger

from src.schemas import (
    AudioMetadata,
    FileMetadata,
    ImageMetadata,
    MediaType,
    VideoMetadata,
)

CHUNK_SIZE = 65_536  # 64KB chunks for hashing large files


def _sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file in chunks (memory-safe for large files)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK_SIZE):
            h.update(chunk)
    return h.hexdigest()


def _image_metadata(path: Path) -> ImageMetadata:
    import cv2
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Cannot read image: {path.name}")
    h, w, c = img.shape
    return ImageMetadata(width=w, height=h, channels=c, format=path.suffix.lstrip(".").upper())


def _video_metadata(path: Path) -> VideoMetadata:
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path.name}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    codec_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join([chr((codec_int >> 8 * i) & 0xFF) for i in range(4)])
    cap.release()
    duration = frames / fps if fps > 0 else 0.0
    return VideoMetadata(
        width=w, height=h, fps=round(fps, 3),
        frame_count=frames, duration_seconds=round(duration, 3), codec=codec.strip()
    )


def _audio_metadata(path: Path) -> AudioMetadata:
    import librosa
    y, sr = librosa.load(str(path), sr=None, mono=False)
    duration = librosa.get_duration(y=y, sr=sr)
    channels = 1 if y.ndim == 1 else y.shape[0]
    return AudioMetadata(
        sample_rate=sr,
        duration_seconds=round(duration, 3),
        channels=channels,
        format=path.suffix.lstrip(".").upper(),
    )


def extract_metadata(path: Path, media_type: MediaType) -> FileMetadata:
    """
    Extract all metadata for a given file.
    Called on the ORIGINAL raw file (not the processed version).
    """
    try:
        sha = _sha256(path)

        if media_type == MediaType.IMAGE:
            media_meta = _image_metadata(path)
        elif media_type == MediaType.VIDEO:
            media_meta = _video_metadata(path)
        elif media_type == MediaType.AUDIO:
            media_meta = _audio_metadata(path)
        else:
            raise ValueError(f"Unsupported media type: {media_type}")

        return FileMetadata(
            file_name=path.name,
            file_path=str(path),
            media_type=media_type,
            file_size_bytes=path.stat().st_size,
            sha256_hash=sha,
            media_metadata=media_meta,
        )

    except Exception as e:
        logger.error(f"Metadata extraction failed for {path.name}: {e}")
        raise


def compute_dataset_version(sample_hashes: list[str]) -> str:
    """
    Compute a deterministic dataset-level version hash.
    Any change to any sample (add, remove, modify) changes this hash.
    Critical for tracking dev vs training data separation.
    """
    combined = hashlib.sha256()
    for h in sorted(sample_hashes):  # sorted = order-independent
        combined.update(h.encode())
    return combined.hexdigest()[:16]  # 16 chars is readable but still unique
