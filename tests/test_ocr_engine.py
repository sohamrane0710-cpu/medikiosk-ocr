"""Automated tests for PaddleOCREngine.

Includes:
- Lightweight unit tests (data structures, invalid paths, empty files, preprocessing)
- Integration inference tests (initialization, repeated inference, confidence scores, bounding boxes)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from app.services.ocr_engine import OCRResult, OCRTextLine, PaddleOCREngine

SAMPLE_IMAGE = Path(__file__).resolve().parent / "fixtures" / "ocr" / "images" / "printed_01.png"


# ------------------------------------------------------------------------------
# UNIT TESTS (Fast, no model weights required)
# ------------------------------------------------------------------------------


@pytest.mark.unit
def test_ocr_data_structures():
    """Verify serialization and structure of OCRTextLine and OCRResult."""
    line = OCRTextLine(
        text="Paracetamol 500mg",
        confidence=0.985,
        polygon=[[10, 10], [100, 10], [100, 30], [10, 30]],
        box=[10, 10, 100, 30],
    )
    assert line.text == "Paracetamol 500mg"
    assert line.confidence == 0.985
    assert len(line.polygon) == 4

    line_dict = line.to_dict()
    assert line_dict["text"] == "Paracetamol 500mg"
    assert line_dict["box"] == [10, 10, 100, 30]

    res = OCRResult(
        image_path="test.png",
        lines=[line],
        raw_text="Paracetamol 500mg",
        inference_time=0.45,
        success=True,
    )
    res_dict = res.to_dict()
    assert res_dict["success"] is True
    assert res_dict["inference_time"] == 0.45
    assert len(res_dict["lines"]) == 1


@pytest.mark.unit
def test_nonexistent_image_handling():
    """Verify graceful failure on non-existent image path without throwing exception."""
    engine = PaddleOCREngine.get_instance()
    res = engine.run_ocr("non_existent_image_12345.png")
    assert res.success is False
    assert "not found" in (res.error_message or "").lower()
    assert res.lines == []
    assert res.raw_text == ""


@pytest.mark.unit
def test_empty_image_handling():
    """Verify graceful failure on 0-byte image file without crashing."""
    engine = PaddleOCREngine.get_instance()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp_name = f.name

    try:
        res = engine.run_ocr(tmp_name)
        assert res.success is False
        assert "empty" in (res.error_message or "").lower()
        assert res.lines == []
    finally:
        Path(tmp_name).unlink(missing_ok=True)


@pytest.mark.unit
def test_empty_numpy_array_handling():
    """Verify handling of empty numpy array."""
    engine = PaddleOCREngine.get_instance()
    empty_arr = np.array([])
    res = engine.run_ocr(empty_arr)
    assert res.success is False
    assert "empty" in (res.error_message or "").lower()


@pytest.mark.unit
def test_image_preprocessing_non_mutating():
    """Verify preprocessing transforms image correctly without mutating original array."""
    original = np.full((100, 100, 3), 128, dtype=np.uint8)
    copy_orig = original.copy()

    processed, applied = PaddleOCREngine.preprocess_image(
        original, to_grayscale=True, adjust_contrast=True, resize_scale=1.5
    )

    # Verify original unchanged
    assert np.array_equal(original, copy_orig)
    # Verify dimensions changed due to scaling
    assert processed.shape == (150, 150, 3)
    assert "grayscale" in applied
    assert "clahe_contrast" in applied
    assert "resize_1.5x" in applied


# ------------------------------------------------------------------------------
# INTEGRATION / INFERENCE TESTS (Require loaded model)
# ------------------------------------------------------------------------------


@pytest.mark.inference
def test_engine_singleton_initialization():
    """Verify PaddleOCREngine behaves as a singleton and preserves instance."""
    inst1 = PaddleOCREngine.get_instance()
    inst2 = PaddleOCREngine.get_instance()
    assert inst1 is inst2
    assert hasattr(inst1, "engine")


@pytest.mark.inference
def test_inference_valid_fixture():
    """Verify inference runs successfully on a valid test fixture."""
    if not SAMPLE_IMAGE.exists():
        pytest.skip(f"Fixture not found: {SAMPLE_IMAGE}")

    engine = PaddleOCREngine.get_instance()
    res = engine.run_ocr(SAMPLE_IMAGE)

    assert res.success is True
    assert res.error_message is None
    assert len(res.lines) > 0
    assert "Amoxicillin" in res.raw_text
    assert res.inference_time > 0.0

    # Verify bounding polygons and confidences
    for line in res.lines:
        assert isinstance(line.text, str)
        assert 0.0 <= line.confidence <= 1.0
        assert len(line.polygon) == 4
        for pt in line.polygon:
            assert len(pt) == 2


@pytest.mark.inference
def test_repeated_inference_same_engine():
    """Verify repeated inference on the same engine maintains consistency."""
    if not SAMPLE_IMAGE.exists():
        pytest.skip(f"Fixture not found: {SAMPLE_IMAGE}")

    engine = PaddleOCREngine.get_instance()
    res1 = engine.run_ocr(SAMPLE_IMAGE)
    res2 = engine.run_ocr(SAMPLE_IMAGE)

    assert res1.success is True and res2.success is True
    assert res1.raw_text == res2.raw_text
    assert len(res1.lines) == len(res2.lines)
    # Warm inference should be efficient
    assert res2.inference_time < 5.0
