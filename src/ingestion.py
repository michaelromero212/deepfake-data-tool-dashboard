"""
ingestion.py — Data ingestion stage.

Scans raw input directories, derives labels from directory/filename conventions,
and returns a flat list of discovered files ready for preprocessing.

Label derivation logic (in order of precedence):
  1. Parent directory named 'real' or 'synthetic' (or 'fake')
  2. Filename suffix _real / _fake / _synthetic
  3. Falls back to UNKNOWN with a warning
"""

from __future__ import annotations

import uuid
from pathlib import Path

from loguru import logger

from src.schemas import Label, LabelSource, MediaType

# ── Supported extensions ─────────────────────────────────────────────────────
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

SYNTHETIC_DIR_NAMES = {"synthetic", "fake", "deepfake", "generated", "ai"}
REAL_DIR_NAMES = {"real", "genuine", "authentic", "original"}


def _media_type(path: Path) -> MediaType | None:
    ext = path.suffix.lower()
    if ext in IMAGE_EXTS:
        return MediaType.IMAGE
    if ext in VIDEO_EXTS:
        return MediaType.VIDEO
    if ext in AUDIO_EXTS:
        return MediaType.AUDIO
    return None


def _derive_label(path: Path) -> tuple[Label, LabelSource]:
    """
    Attempt to derive a real/synthetic label from the file path.
    Returns (Label, LabelSource) so downstream consumers know how confident to be.
    """
    # Check all parent directory names
    for parent in path.parts:
        lower = parent.lower()
        if lower in SYNTHETIC_DIR_NAMES:
            return Label.SYNTHETIC, LabelSource.DIRECTORY_CONVENTION
        if lower in REAL_DIR_NAMES:
            return Label.REAL, LabelSource.DIRECTORY_CONVENTION

    # Check filename suffixes (e.g. face_swap_fake.mp4, portrait_real.jpg)
    stem = path.stem.lower()
    if any(stem.endswith(f"_{s}") for s in SYNTHETIC_DIR_NAMES):
        return Label.SYNTHETIC, LabelSource.FILENAME_CONVENTION
    if any(stem.endswith(f"_{r}") for r in REAL_DIR_NAMES):
        return Label.REAL, LabelSource.FILENAME_CONVENTION

    logger.warning(f"Could not derive label for {path.name} — marking UNKNOWN")
    return Label.UNKNOWN, LabelSource.INFERRED


class DiscoveredFile:
    """Lightweight container used between ingestion and preprocessing."""
    __slots__ = ("path", "media_type", "label", "label_source", "sample_id")

    def __init__(
        self,
        path: Path,
        media_type: MediaType,
        label: Label,
        label_source: LabelSource,
    ) -> None:
        self.path = path
        self.media_type = media_type
        self.label = label
        self.label_source = label_source
        self.sample_id = str(uuid.uuid4())[:8]


def ingest_data(raw_root: Path) -> list[DiscoveredFile]:
    """
    Walk raw_root recursively, discover all supported media files,
    and assign labels. Returns a list of DiscoveredFile objects.
    """
    if not raw_root.exists():
        raise FileNotFoundError(f"Raw data root not found: {raw_root}")

    discovered: list[DiscoveredFile] = []
    skipped = 0

    for path in sorted(raw_root.rglob("*")):
        if not path.is_file():
            continue

        media_type = _media_type(path)
        if media_type is None:
            skipped += 1
            continue

        label, label_source = _derive_label(path)
        discovered.append(DiscoveredFile(path, media_type, label, label_source))

    logger.info(
        f"Ingestion complete — {len(discovered)} files discovered, {skipped} skipped"
    )
    _log_label_distribution(discovered)
    return discovered


def _log_label_distribution(files: list[DiscoveredFile]) -> None:
    counts: dict[str, int] = {}
    for f in files:
        counts[f.label.value] = counts.get(f.label.value, 0) + 1
    logger.info(f"Label distribution: {counts}")
