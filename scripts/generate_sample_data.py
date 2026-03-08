"""
scripts/generate_sample_data.py

Generates a small set of synthetic + real sample media files
so the pipeline can be demoed without requiring a real dataset download.

Run: python scripts/generate_sample_data.py
"""

from __future__ import annotations

import struct
import wave
from pathlib import Path

import numpy as np

RAW_ROOT = Path("data/raw")


def create_sample_images() -> None:
    """Create simple solid-color PNG images using only stdlib + numpy."""
    try:
        from PIL import Image

        dirs = {
            RAW_ROOT / "images" / "real": (100, 180, 100),       # green-ish
            RAW_ROOT / "images" / "synthetic": (180, 100, 100),  # red-ish
        }
        for out_dir, base_color in dirs.items():
            out_dir.mkdir(parents=True, exist_ok=True)
            for i in range(3):
                rng = np.random.default_rng(i)
                noise = rng.integers(-20, 20, (128, 128, 3), dtype=np.int16)
                pixel = np.clip(np.array(base_color, dtype=np.int16) + noise, 0, 255).astype(np.uint8)
                img_array = np.broadcast_to(pixel, (128, 128, 3)).copy()
                img = Image.fromarray(img_array, "RGB")
                img.save(out_dir / f"sample_{i:02d}.jpg")

        print(f"✓ Created sample images in {RAW_ROOT / 'images'}")

    except ImportError:
        print("Pillow not installed — skipping image generation")


def create_sample_audio() -> None:
    """Create short WAV files using only stdlib."""
    dirs = {
        RAW_ROOT / "audio" / "real": 440.0,       # A4 tone
        RAW_ROOT / "audio" / "synthetic": 880.0,  # A5 tone (different freq)
    }
    sample_rate = 16_000
    duration = 1.0  # 1 second

    for out_dir, freq in dirs.items():
        out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(2):
            t = np.linspace(0, duration, int(sample_rate * duration))
            signal = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)

            out_path = out_dir / f"tone_{i:02d}.wav"
            with wave.open(str(out_path), "w") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(signal.tobytes())

    print(f"✓ Created sample audio in {RAW_ROOT / 'audio'}")


def create_sample_video() -> None:
    """Create minimal MP4 files using OpenCV if available."""
    try:
        import cv2

        dirs = {
            RAW_ROOT / "videos" / "real": (0, 200, 0),
            RAW_ROOT / "videos" / "synthetic": (0, 0, 200),
        }
        for out_dir, color in dirs.items():
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "clip_00.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, 10.0, (128, 128))
            frame = np.full((128, 128, 3), color, dtype=np.uint8)
            for _ in range(20):  # 2 seconds at 10fps
                writer.write(frame)
            writer.release()

        print(f"✓ Created sample videos in {RAW_ROOT / 'videos'}")

    except ImportError:
        print("OpenCV not installed — skipping video generation")


if __name__ == "__main__":
    print("Generating sample data...\n")
    create_sample_images()
    create_sample_audio()
    create_sample_video()
    print("\nDone. Run the pipeline with:")
    print("  python -m src.pipeline run")
