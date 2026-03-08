"""
preprocessing.py — Media preprocessing stage.

Handles image resizing/normalization, video frame extraction,
and audio resampling + spectrogram generation.
Each processor returns the output path of the processed file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from loguru import logger

from src.schemas import MediaType

# ── Constants ──────────────────────────────────────────────────────────────
TARGET_IMAGE_SIZE = (224, 224)   # Standard input size for most vision models
TARGET_SAMPLE_RATE = 16_000      # 16kHz — standard for audio deepfake models
FRAMES_PER_VIDEO = 16            # How many evenly-spaced frames to extract


# ── Image ──────────────────────────────────────────────────────────────────

def preprocess_image(src: Path, out_dir: Path) -> Path:
    """
    Resize to TARGET_IMAGE_SIZE, normalize to [0, 1], save as PNG.
    Returns path to processed file.
    """
    try:
        import cv2

        img = cv2.imread(str(src))
        if img is None:
            raise ValueError(f"OpenCV could not decode {src.name}")

        img_resized = cv2.resize(img, TARGET_IMAGE_SIZE, interpolation=cv2.INTER_AREA)

        # Normalize to float32 [0, 1] — save as uint8 PNG for storage efficiency
        img_normalized = (img_resized.astype(np.float32) / 255.0 * 255).astype(np.uint8)

        out_path = out_dir / f"{src.stem}_processed.png"
        cv2.imwrite(str(out_path), img_normalized)
        return out_path

    except Exception as e:
        logger.error(f"Image preprocessing failed for {src.name}: {e}")
        raise


# ── Video ──────────────────────────────────────────────────────────────────

def preprocess_video(src: Path, out_dir: Path) -> list[Path]:
    """
    Extract FRAMES_PER_VIDEO evenly-spaced frames, resize each to TARGET_IMAGE_SIZE.
    Returns list of frame paths.
    """
    try:
        import cv2

        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {src.name}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            raise ValueError(f"Video has 0 frames: {src.name}")

        frame_indices = np.linspace(0, total_frames - 1, FRAMES_PER_VIDEO, dtype=int)
        out_paths: list[Path] = []

        for i, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Could not read frame {frame_idx} from {src.name}")
                continue

            frame_resized = cv2.resize(frame, TARGET_IMAGE_SIZE, interpolation=cv2.INTER_AREA)
            out_path = out_dir / f"{src.stem}_frame{i:03d}.png"
            cv2.imwrite(str(out_path), frame_resized)
            out_paths.append(out_path)

        cap.release()
        return out_paths

    except Exception as e:
        logger.error(f"Video preprocessing failed for {src.name}: {e}")
        raise


# ── Audio ──────────────────────────────────────────────────────────────────

def preprocess_audio(src: Path, out_dir: Path) -> Path:
    """
    Resample to TARGET_SAMPLE_RATE, generate mel spectrogram, save as PNG.
    The spectrogram is what deepfake audio models typically operate on.
    Returns path to spectrogram image.
    """
    try:
        import librosa
        import librosa.display
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt

        y, sr = librosa.load(str(src), sr=TARGET_SAMPLE_RATE, mono=True)

        # Mel spectrogram — standard feature for audio deepfake detection
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        fig, ax = plt.subplots(figsize=(4, 4), dpi=56)  # ~224x224 output
        librosa.display.specshow(mel_spec_db, sr=sr, x_axis="time", y_axis="mel", ax=ax)
        ax.axis("off")
        plt.tight_layout(pad=0)

        out_path = out_dir / f"{src.stem}_spectrogram.png"
        fig.savefig(str(out_path), bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        return out_path

    except Exception as e:
        logger.error(f"Audio preprocessing failed for {src.name}: {e}")
        raise


# ── Dispatcher ─────────────────────────────────────────────────────────────

def preprocess_file(
    src: Path,
    media_type: MediaType,
    processed_root: Path,
) -> Path | list[Path]:
    """
    Route a file to the correct preprocessor based on media type.
    Returns output path(s).
    """
    out_dirs = {
        MediaType.IMAGE: processed_root / "images",
        MediaType.VIDEO: processed_root / "video_frames",
        MediaType.AUDIO: processed_root / "audio_features",
    }
    out_dir = out_dirs[media_type]
    out_dir.mkdir(parents=True, exist_ok=True)

    if media_type == MediaType.IMAGE:
        return preprocess_image(src, out_dir)
    elif media_type == MediaType.VIDEO:
        return preprocess_video(src, out_dir)
    elif media_type == MediaType.AUDIO:
        return preprocess_audio(src, out_dir)
    else:
        raise ValueError(f"Unsupported media type: {media_type}")
