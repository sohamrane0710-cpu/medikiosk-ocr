"""PaddleOCR 3.x OCR Engine Service.

Provides an isolated, reusable OCR inference interface with structured outputs,
confidence reporting, and safe error handling.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

# Configure Paddle cache paths to workspace by default to ensure write access and offline caching
_WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
_CACHE_ROOT = _WORKSPACE_ROOT / ".cache"
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(_CACHE_ROOT / "paddlex"))
os.environ.setdefault("PADDLE_HOME", str(_CACHE_ROOT / "paddle"))
os.environ.setdefault("PADDLEOCR_HOME", str(_CACHE_ROOT / "paddleocr"))
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

logger = logging.getLogger(__name__)


@dataclass
class OCRTextLine:
    """Represents a single recognized line of text with bounding polygon and confidence."""

    text: str
    confidence: float
    polygon: list[list[int]] = field(default_factory=list)
    box: list[int] = field(default_factory=list)  # [x_min, y_min, x_max, y_max]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OCRResult:
    """Complete result from an OCR extraction run."""

    image_path: str
    lines: list[OCRTextLine]
    raw_text: str
    inference_time: float
    success: bool = True
    error_message: str | None = None
    preprocessing_applied: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_path": self.image_path,
            "lines": [line.to_dict() for line in self.lines],
            "raw_text": self.raw_text,
            "inference_time": round(self.inference_time, 4),
            "success": self.success,
            "error_message": self.error_message,
            "preprocessing_applied": self.preprocessing_applied,
        }


class PaddleOCREngine:
    """Singleton-capable PaddleOCR 3.x Engine Wrapper."""

    _instance: PaddleOCREngine | None = None

    def __init__(self, lang: str = "en", **kwargs: Any) -> None:
        """Initialize PaddleOCR engine once."""
        from paddleocr import PaddleOCR

        logger.info("Initializing PaddleOCR engine (lang=%s)...", lang)
        t0 = time.perf_counter()
        self.engine = PaddleOCR(lang=lang, **kwargs)
        self.init_duration = time.perf_counter() - t0
        self.lang = lang
        logger.info("PaddleOCR engine initialized in %.2fs", self.init_duration)

    @classmethod
    def get_instance(cls, lang: str = "en", **kwargs: Any) -> PaddleOCREngine:
        """Retrieve or create the shared singleton engine instance."""
        if cls._instance is None:
            cls._instance = cls(lang=lang, **kwargs)
        return cls._instance

    @staticmethod
    def preprocess_image(
        image: np.ndarray,
        to_grayscale: bool = False,
        adjust_contrast: bool = False,
        denoise: bool = False,
        deskew: bool = False,
        resize_scale: float = 1.0,
    ) -> tuple[np.ndarray, list[str]]:
        """Apply optional image preprocessing steps without mutating the input image."""
        applied: list[str] = []
        processed = image.copy()

        # 1. Resize scaling
        if resize_scale != 1.0 and resize_scale > 0:
            h, w = processed.shape[:2]
            new_w, new_h = int(w * resize_scale), int(h * resize_scale)
            processed = cv2.resize(processed, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            applied.append(f"resize_{resize_scale}x")

        # 2. Grayscale conversion
        if to_grayscale and len(processed.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            # PaddleOCR expects 3 channels in many pipeline stages; convert back to 3-channel gray
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
            applied.append("grayscale")

        # 3. Contrast adjustment (CLAHE)
        if adjust_contrast:
            if len(processed.shape) == 3:
                lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
                l_channel, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                cl = clahe.apply(l_channel)
                merged = cv2.merge((cl, a, b))
                processed = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
            else:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                processed = cv2.cvtColor(clahe.apply(processed), cv2.COLOR_GRAY2BGR)
            applied.append("clahe_contrast")

        # 4. Denoising
        if denoise:
            if len(processed.shape) == 3:
                processed = cv2.fastNlMeansDenoisingColored(processed, None, 10, 10, 7, 21)
            else:
                processed = cv2.fastNlMeansDenoising(processed, None, 10, 7, 21)
            applied.append("denoise")

        # 5. Deskewing
        if deskew:
            gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY) if len(processed.shape) == 3 else processed
            coords = np.column_stack(np.where(gray < 250))
            if coords.shape[0] > 10:
                angle = cv2.minAreaRect(coords)[-1]
                if angle < -45:
                    angle = -(90 + angle)
                else:
                    angle = -angle
                if abs(angle) > 0.5:
                    h, w = processed.shape[:2]
                    center = (w // 2, h // 2)
                    m = cv2.getRotationMatrix2D(center, angle, 1.0)
                    processed = cv2.warpAffine(
                        processed, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
                    )
                    applied.append(f"deskew_{angle:.1f}deg")

        return processed, applied

    def run_ocr(
        self,
        image_input: str | Path | np.ndarray,
        apply_preprocessing: bool = False,
        preprocessing_kwargs: dict[str, Any] | None = None,
    ) -> OCRResult:
        """Run OCR on an image file or numpy array and return structured OCRResult."""
        image_str_path = str(image_input) if isinstance(image_input, (str, Path)) else "<in-memory array>"
        t_start = time.perf_counter()

        # Validate input existence if path
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists():
                return OCRResult(
                    image_path=image_str_path,
                    lines=[],
                    raw_text="",
                    inference_time=time.perf_counter() - t_start,
                    success=False,
                    error_message=f"Image file not found: {path}",
                )
            if path.stat().st_size == 0:
                return OCRResult(
                    image_path=image_str_path,
                    lines=[],
                    raw_text="",
                    inference_time=time.perf_counter() - t_start,
                    success=False,
                    error_message="Image file is empty (0 bytes).",
                )
            img = cv2.imread(str(path))
            if img is None:
                return OCRResult(
                    image_path=image_str_path,
                    lines=[],
                    raw_text="",
                    inference_time=time.perf_counter() - t_start,
                    success=False,
                    error_message=f"Failed to decode image file: {path}",
                )
        elif isinstance(image_input, np.ndarray):
            if image_input.size == 0:
                return OCRResult(
                    image_path=image_str_path,
                    lines=[],
                    raw_text="",
                    inference_time=time.perf_counter() - t_start,
                    success=False,
                    error_message="Provided numpy array is empty.",
                )
            img = image_input.copy()
        else:
            return OCRResult(
                image_path=image_str_path,
                lines=[],
                raw_text="",
                inference_time=time.perf_counter() - t_start,
                success=False,
                error_message=f"Unsupported image input type: {type(image_input)}",
            )

        applied_preproc: list[str] = []
        if apply_preprocessing:
            kwargs = preprocessing_kwargs or {}
            img, applied_preproc = self.preprocess_image(img, **kwargs)

        try:
            # Execute inference using modern predict() API
            predict_gen = self.engine.predict(img)
            prediction_list = list(predict_gen)

            if not prediction_list:
                return OCRResult(
                    image_path=image_str_path,
                    lines=[],
                    raw_text="",
                    inference_time=time.perf_counter() - t_start,
                    success=True,
                    preprocessing_applied=applied_preproc,
                )

            res = prediction_list[0]
            rec_texts: Sequence[str] = res.get("rec_texts", [])
            rec_scores: Sequence[float] = res.get("rec_scores", [])
            rec_polys: Sequence[Any] = res.get("rec_polys", [])
            rec_boxes: Sequence[Any] = res.get("rec_boxes", [])

            lines: list[OCRTextLine] = []
            for i, text in enumerate(rec_texts):
                score = float(rec_scores[i]) if i < len(rec_scores) else 0.0
                poly: list[list[int]] = []
                if i < len(rec_polys):
                    poly_raw = rec_polys[i]
                    if hasattr(poly_raw, "tolist"):
                        poly = poly_raw.tolist()
                    elif isinstance(poly_raw, (list, tuple)):
                        poly = [list(pt) for pt in poly_raw]

                box: list[int] = []
                if i < len(rec_boxes):
                    box_raw = rec_boxes[i]
                    if hasattr(box_raw, "tolist"):
                        box = [int(v) for v in box_raw.tolist()]
                    elif isinstance(box_raw, (list, tuple)):
                        box = [int(v) for v in box_raw]

                lines.append(OCRTextLine(text=str(text), confidence=score, polygon=poly, box=box))

            raw_text = "\n".join([line.text for line in lines])
            inference_duration = time.perf_counter() - t_start

            return OCRResult(
                image_path=image_str_path,
                lines=lines,
                raw_text=raw_text,
                inference_time=inference_duration,
                success=True,
                preprocessing_applied=applied_preproc,
            )

        except Exception as exc:
            logger.exception("OCR inference error on %s: %s", image_str_path, exc)
            return OCRResult(
                image_path=image_str_path,
                lines=[],
                raw_text="",
                inference_time=time.perf_counter() - t_start,
                success=False,
                error_message=str(exc),
                preprocessing_applied=applied_preproc,
            )
