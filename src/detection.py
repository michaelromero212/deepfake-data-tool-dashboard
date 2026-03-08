"""
detection.py — Deepfake detection inference stage.

Three modes, in order of precedence:

  1. HuggingFace mode (default when transformers is installed)
     Uses dima806/deepfake_vs_real_image_detection — a ViT-base model
     fine-tuned on real vs. AI-generated face images. Runs on CPU.
     First run downloads ~330MB model weights (cached after that).

  2. ONNX mode (set FORGE_USE_ONNX=1 + drop model in models/)
     Loads any EfficientNet-based ONNX deepfake classifier.

  3. Mock mode (fallback, no ML dependencies needed)
     Simulates realistic score distributions using beta distributions.

Audio samples are scored via their mel spectrogram PNG (produced by
preprocessing.py), which is consistent with how audio deepfake
detection models typically operate.
"""

from __future__ import annotations

import os
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from loguru import logger

from src.schemas import DetectionResult, Label, MediaType

# ── Constants ──────────────────────────────────────────────────────────────
HF_MODEL_ID = "dima806/deepfake_vs_real_image_detection"
HF_MODEL_VERSION = "1.0.0"
MOCK_MODEL_NAME = "mock-efficientnet-dfdc-v1"
MOCK_MODEL_VERSION = "0.1.0-mock"
ONNX_MODEL_PATH = Path("models/deepfake_detector.onnx")


# ── HuggingFace detector ───────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_hf_pipeline():
    """
    Load the HuggingFace image classification pipeline.
    Cached so the model is only loaded once per process (~330MB, CPU).
    """
    from transformers import pipeline
    logger.info(f"Loading HuggingFace model: {HF_MODEL_ID} (CPU)")
    pipe = pipeline(
        "image-classification",
        model=HF_MODEL_ID,
        device=-1,  # CPU
    )
    logger.info("Model loaded and ready.")
    return pipe


def _hf_score(processed_path: Path) -> float:
    """
    Run inference with the HuggingFace deepfake detector.
    Returns score in [0, 1] where 1.0 = synthetic/fake.
    Model outputs 'Fake' and 'Real' labels — we return the Fake confidence.
    """
    from PIL import Image

    pipe = _load_hf_pipeline()
    img = Image.open(processed_path).convert("RGB")
    results = pipe(img)

    # results: [{"label": "Fake", "score": 0.91}, {"label": "Real", "score": 0.09}]
    score_map = {r["label"].lower(): r["score"] for r in results}
    fake_score = score_map.get("fake", score_map.get("deepfake", 0.5))
    return float(fake_score)


# ── ONNX detector ──────────────────────────────────────────────────────────

def _onnx_score(processed_path: Path) -> float:
    """Run inference with a real ONNX model. Expects a 224x224 PNG image."""
    try:
        import cv2
        import onnxruntime as ort

        sess = ort.InferenceSession(str(ONNX_MODEL_PATH))
        img = cv2.imread(str(processed_path))
        if img is None:
            raise ValueError(f"Cannot load image: {processed_path}")

        img = cv2.resize(img, (224, 224)).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]  # NCHW

        input_name = sess.get_inputs()[0].name
        result = sess.run(None, {input_name: img})
        logits = result[0][0]

        score = float(np.exp(logits[1]) / np.sum(np.exp(logits)))
        return score

    except ImportError:
        logger.warning("onnxruntime not installed — falling back to mock mode")
        return _mock_score(Label.UNKNOWN, MediaType.IMAGE)


# ── Mock detector ──────────────────────────────────────────────────────────

def _mock_score(label: Label, media_type: MediaType) -> float:
    """
    Simulate realistic detection scores using beta distributions.
    Real samples cluster low (0.05-0.25), synthetic cluster high (0.70-0.95).
    """
    rng = np.random.default_rng()

    if label == Label.REAL:
        base = rng.beta(a=1.5, b=8.0)
    elif label == Label.SYNTHETIC:
        base = rng.beta(a=8.0, b=1.5)
    else:
        base = rng.uniform(0.3, 0.7)

    if media_type in (MediaType.AUDIO, MediaType.VIDEO):
        base = np.clip(base + rng.normal(0, 0.05), 0.0, 1.0)

    return float(np.clip(base, 0.0, 1.0))


# ── Mode detection ─────────────────────────────────────────────────────────

def _detect_mode() -> str:
    if os.getenv("FORGE_USE_ONNX", "0") == "1" and ONNX_MODEL_PATH.exists():
        return "onnx"
    if os.getenv("FORGE_USE_MOCK", "0") == "1":
        return "mock"
    try:
        import transformers  # noqa: F401
        import PIL  # noqa: F401
        return "huggingface"
    except ImportError:
        logger.warning(
            "transformers/PIL not installed — using mock mode. "
            "Run: pip install transformers pillow"
        )
        return "mock"


# ── Public API ─────────────────────────────────────────────────────────────

def run_detection(
    processed_path: Path | None,
    label: Label,
    media_type: MediaType,
) -> DetectionResult:
    """
    Run deepfake detection on a processed sample.
    Returns a DetectionResult with score, model provenance, and timing.
    """
    mode = _detect_mode()
    start = time.perf_counter()

    try:
        if mode == "huggingface" and processed_path and processed_path.exists():
            score = _hf_score(processed_path)
            model_name = HF_MODEL_ID
            model_version = HF_MODEL_VERSION
            logger.debug(f"HF inference: score={score:.3f} [{processed_path.name}]")

        elif mode == "onnx" and processed_path and processed_path.exists():
            score = _onnx_score(processed_path)
            model_name = "onnx-efficientnet-dfdc"
            model_version = "1.0.0"
            logger.debug(f"ONNX inference: score={score:.3f} [{processed_path.name}]")

        else:
            score = _mock_score(label, media_type)
            model_name = MOCK_MODEL_NAME
            model_version = MOCK_MODEL_VERSION
            logger.debug(f"Mock inference: score={score:.3f} (label={label.value})")

    except Exception as e:
        logger.warning(f"Detection failed ({e}) — falling back to mock")
        score = _mock_score(label, media_type)
        model_name = f"{MOCK_MODEL_NAME}-fallback"
        model_version = MOCK_MODEL_VERSION

    elapsed_ms = (time.perf_counter() - start) * 1000

    return DetectionResult(
        model_name=model_name,
        detection_score=round(score, 4),
        model_version=model_version,
        inference_time_ms=round(elapsed_ms, 2),
    )
