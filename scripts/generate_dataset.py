"""Generate synthetic prescription evaluation dataset for Phase 1 OCR testing.

Creates 30 prescription test images categorized into:
- Category A: Printed prescriptions (10 images)
- Category B: Handwritten-style prescriptions (10 images)
- Category C: Degraded document prescriptions (10 images)

Ground truth metadata and critical entity annotations are saved to
tests/fixtures/ocr/dataset_manifest.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

DATASET_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "ocr"
IMAGES_DIR = DATASET_DIR / "images"
FONTS_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "fonts"

FONT_PRINT_SANS = "/usr/share/fonts/TTF/DejaVuSans.ttf"
FONT_PRINT_SERIF = "/usr/share/fonts/TTF/DejaVuSerif.ttf"
FONT_HAND_CAVEAT = str(FONTS_DIR / "Caveat-Regular.ttf")
FONT_HAND_KALAM = str(FONTS_DIR / "Kalam-Regular.ttf")
FONT_HAND_DANCING = str(FONTS_DIR / "DancingScript-Regular.ttf")

PRESCRIPTION_SPECS: list[dict[str, Any]] = [
    # --------------------------------------------------------------------------
    # CATEGORY A: PRINTED PRESCRIPTIONS (10 Images)
    # --------------------------------------------------------------------------
    {
        "id": "printed_01",
        "category": "printed",
        "degradation_type": "none",
        "font_path": FONT_PRINT_SANS,
        "doctor": "Dr. Sarah Jenkins, MD - Internal Medicine",
        "clinic": "Metro General Hospital Clinic",
        "medicine": "Amoxicillin 500mg",
        "instructions": "Take 1 capsule TDS for 7 days after meals",
        "critical": {
            "medicine_name": "Amoxicillin",
            "dosage": "500mg",
            "frequency": "TDS",
            "duration": "7 days",
            "numeric_instruction": "1 capsule",
        },
    },
    {
        "id": "printed_02",
        "category": "printed",
        "degradation_type": "none",
        "font_path": FONT_PRINT_SERIF,
        "doctor": "Dr. Robert Chen, MBBS - Endocrinology",
        "clinic": "Apex Diabetes & Wellness Center",
        "medicine": "Metformin HCl 850mg",
        "instructions": "Take 1 tablet BD with meals for 30 days",
        "critical": {
            "medicine_name": "Metformin HCl",
            "dosage": "850mg",
            "frequency": "BD",
            "duration": "30 days",
            "numeric_instruction": "1 tablet",
        },
    },
    {
        "id": "printed_03",
        "category": "printed",
        "degradation_type": "none",
        "font_path": FONT_PRINT_SANS,
        "doctor": "Dr. Emily Taylor, MD - Cardiology",
        "clinic": "Highland Heart Institute",
        "medicine": "Atorvastatin 20mg",
        "instructions": "Take 1 tablet OD at bedtime for 60 days",
        "critical": {
            "medicine_name": "Atorvastatin",
            "dosage": "20mg",
            "frequency": "OD",
            "duration": "60 days",
            "numeric_instruction": "1 tablet",
        },
    },
    {
        "id": "printed_04",
        "category": "printed",
        "degradation_type": "none",
        "font_path": FONT_PRINT_SERIF,
        "doctor": "Dr. Mark Davis, MD - Pulmonology",
        "clinic": "Valley Respiratory Associates",
        "medicine": "Azithromycin 250mg",
        "instructions": "Take 2 tablets OD on day 1 then 1 tablet OD for 4 days",
        "critical": {
            "medicine_name": "Azithromycin",
            "dosage": "250mg",
            "frequency": "OD",
            "duration": "4 days",
            "numeric_instruction": "1 tablet",
        },
    },
    {
        "id": "printed_05",
        "category": "printed",
        "degradation_type": "none",
        "font_path": FONT_PRINT_SANS,
        "doctor": "Dr. Priya Patel, MD - Gastroenterology",
        "clinic": "Sunset Digestive Care",
        "medicine": "Omeprazole 20mg",
        "instructions": "Take 1 capsule OD before breakfast for 14 days",
        "critical": {
            "medicine_name": "Omeprazole",
            "dosage": "20mg",
            "frequency": "OD",
            "duration": "14 days",
            "numeric_instruction": "1 capsule",
        },
    },
    {
        "id": "printed_06",
        "category": "printed",
        "degradation_type": "none",
        "font_path": FONT_PRINT_SERIF,
        "doctor": "Dr. John Wilson, MD - Infectious Disease",
        "clinic": "St. Jude Community Health",
        "medicine": "Ciprofloxacin 500mg",
        "instructions": "Take 1 tablet BD for 5 days with full glass of water",
        "critical": {
            "medicine_name": "Ciprofloxacin",
            "dosage": "500mg",
            "frequency": "BD",
            "duration": "5 days",
            "numeric_instruction": "1 tablet",
        },
    },
    {
        "id": "printed_07",
        "category": "printed",
        "degradation_type": "none",
        "font_path": FONT_PRINT_SANS,
        "doctor": "Dr. Alan Brooks, MD - Nephrology",
        "clinic": "Cedar Valley Renal Clinic",
        "medicine": "Losartan Potassium 50mg",
        "instructions": "Take 1 tablet OD every morning for 30 days",
        "critical": {
            "medicine_name": "Losartan Potassium",
            "dosage": "50mg",
            "frequency": "OD",
            "duration": "30 days",
            "numeric_instruction": "1 tablet",
        },
    },
    {
        "id": "printed_08",
        "category": "printed",
        "degradation_type": "none",
        "font_path": FONT_PRINT_SERIF,
        "doctor": "Dr. Lisa Wong, MD - Family Medicine",
        "clinic": "Bayside Medical Group",
        "medicine": "Paracetamol 650mg",
        "instructions": "Take 1 tablet SOS every 6 hours for fever for 3 days",
        "critical": {
            "medicine_name": "Paracetamol",
            "dosage": "650mg",
            "frequency": "SOS",
            "duration": "3 days",
            "numeric_instruction": "1 tablet",
        },
    },
    {
        "id": "printed_09",
        "category": "printed",
        "degradation_type": "none",
        "font_path": FONT_PRINT_SANS,
        "doctor": "Dr. Karen White, MD - Orthopedics",
        "clinic": "Summit Joint & Spine Clinic",
        "medicine": "Ibuprofen 400mg",
        "instructions": "Take 1 tablet TDS after meals for 5 days",
        "critical": {
            "medicine_name": "Ibuprofen",
            "dosage": "400mg",
            "frequency": "TDS",
            "duration": "5 days",
            "numeric_instruction": "1 tablet",
        },
    },
    {
        "id": "printed_10",
        "category": "printed",
        "degradation_type": "none",
        "font_path": FONT_PRINT_SERIF,
        "doctor": "Dr. David Miller, MD - Allergy & Immunology",
        "clinic": "Northwest Asthma Center",
        "medicine": "Montelukast 10mg",
        "instructions": "Take 1 tablet OD at bedtime for 30 days",
        "critical": {
            "medicine_name": "Montelukast",
            "dosage": "10mg",
            "frequency": "OD",
            "duration": "30 days",
            "numeric_instruction": "1 tablet",
        },
    },
    # --------------------------------------------------------------------------
    # CATEGORY B: HANDWRITTEN-STYLE PRESCRIPTIONS (10 Images)
    # --------------------------------------------------------------------------
    {
        "id": "handwritten_01",
        "category": "handwritten",
        "degradation_type": "neat_handwriting_font",
        "font_path": FONT_HAND_CAVEAT,
        "doctor": "Dr. Arthur Conan, MD",
        "clinic": "St. Bartholomew Medical Ward",
        "medicine": "Amoxicillin 250mg",
        "instructions": "1 cap TDS for 5 days pc",
        "critical": {
            "medicine_name": "Amoxicillin",
            "dosage": "250mg",
            "frequency": "TDS",
            "duration": "5 days",
            "numeric_instruction": "1 cap",
        },
    },
    {
        "id": "handwritten_02",
        "category": "handwritten",
        "degradation_type": "moderate_handwriting_font",
        "font_path": FONT_HAND_KALAM,
        "doctor": "Dr. Gregory House, MD",
        "clinic": "Princeton Plainsboro Hospital",
        "medicine": "Cetirizine 10mg",
        "instructions": "1 tab OD hs for 10 days",
        "critical": {
            "medicine_name": "Cetirizine",
            "dosage": "10mg",
            "frequency": "OD",
            "duration": "10 days",
            "numeric_instruction": "1 tab",
        },
    },
    {
        "id": "handwritten_03",
        "category": "handwritten",
        "degradation_type": "cursive_handwriting_font",
        "font_path": FONT_HAND_DANCING,
        "doctor": "Dr. Leonard McCoy, MD",
        "clinic": "Starfleet Medical Services",
        "medicine": "Doxycycline 100mg",
        "instructions": "1 cap BD pc for 7 days",
        "critical": {
            "medicine_name": "Doxycycline",
            "dosage": "100mg",
            "frequency": "BD",
            "duration": "7 days",
            "numeric_instruction": "1 cap",
        },
    },
    {
        "id": "handwritten_04",
        "category": "handwritten",
        "degradation_type": "neat_handwriting_font",
        "font_path": FONT_HAND_CAVEAT,
        "doctor": "Dr. Beverly Crusher, MD",
        "clinic": "Enterprise Healthcare Facility",
        "medicine": "Pantoprazole 40mg",
        "instructions": "1 tab OD ac for 14 days",
        "critical": {
            "medicine_name": "Pantoprazole",
            "dosage": "40mg",
            "frequency": "OD",
            "duration": "14 days",
            "numeric_instruction": "1 tab",
        },
    },
    {
        "id": "handwritten_05",
        "category": "handwritten",
        "degradation_type": "moderate_handwriting_font",
        "font_path": FONT_HAND_KALAM,
        "doctor": "Dr. John Watson, MD",
        "clinic": "Baker Street Clinic",
        "medicine": "Metoprolol 25mg",
        "instructions": "1 tab BD for 30 days",
        "critical": {
            "medicine_name": "Metoprolol",
            "dosage": "25mg",
            "frequency": "BD",
            "duration": "30 days",
            "numeric_instruction": "1 tab",
        },
    },
    {
        "id": "handwritten_06",
        "category": "handwritten",
        "degradation_type": "cursive_handwriting_font",
        "font_path": FONT_HAND_DANCING,
        "doctor": "Dr. Michaela Quinn, MD",
        "clinic": "Colorado Springs Medical",
        "medicine": "Salbutamol 4mg",
        "instructions": "1 tab TDS prn for 5 days",
        "critical": {
            "medicine_name": "Salbutamol",
            "dosage": "4mg",
            "frequency": "TDS",
            "duration": "5 days",
            "numeric_instruction": "1 tab",
        },
    },
    {
        "id": "handwritten_07",
        "category": "handwritten",
        "degradation_type": "neat_handwriting_font",
        "font_path": FONT_HAND_CAVEAT,
        "doctor": "Dr. James Wilson, MD",
        "clinic": "Mercy General Hospital",
        "medicine": "Prednisolone 5mg",
        "instructions": "2 tabs OD in morning for 5 days",
        "critical": {
            "medicine_name": "Prednisolone",
            "dosage": "5mg",
            "frequency": "OD",
            "duration": "5 days",
            "numeric_instruction": "2 tabs",
        },
    },
    {
        "id": "handwritten_08",
        "category": "handwritten",
        "degradation_type": "moderate_handwriting_font",
        "font_path": FONT_HAND_KALAM,
        "doctor": "Dr. Allison Cameron, MD",
        "clinic": "St. Luke Diagnostic Center",
        "medicine": "Amoxicillin Clavulanate 625mg",
        "instructions": "1 tab BD pc for 7 days",
        "critical": {
            "medicine_name": "Amoxicillin Clavulanate",
            "dosage": "625mg",
            "frequency": "BD",
            "duration": "7 days",
            "numeric_instruction": "1 tab",
        },
    },
    {
        "id": "handwritten_09",
        "category": "handwritten",
        "degradation_type": "cursive_handwriting_font",
        "font_path": FONT_HAND_DANCING,
        "doctor": "Dr. Robert Chase, MD",
        "clinic": "Hope Valley Medical Office",
        "medicine": "Tramadol 50mg",
        "instructions": "1 cap SOS for severe pain for 3 days",
        "critical": {
            "medicine_name": "Tramadol",
            "dosage": "50mg",
            "frequency": "SOS",
            "duration": "3 days",
            "numeric_instruction": "1 cap",
        },
    },
    {
        "id": "handwritten_10",
        "category": "handwritten",
        "degradation_type": "neat_handwriting_font",
        "font_path": FONT_HAND_CAVEAT,
        "doctor": "Dr. Eric Foreman, MD",
        "clinic": "Downtown Health Pavilion",
        "medicine": "Levothyroxine 50mcg",
        "instructions": "1 tab OD empty stomach for 30 days",
        "critical": {
            "medicine_name": "Levothyroxine",
            "dosage": "50mcg",
            "frequency": "OD",
            "duration": "30 days",
            "numeric_instruction": "1 tab",
        },
    },
    # --------------------------------------------------------------------------
    # CATEGORY C: DEGRADED DOCUMENTS (10 Images)
    # --------------------------------------------------------------------------
    {
        "id": "degraded_01",
        "category": "degraded",
        "degradation_type": "skew_rotation",
        "font_path": FONT_PRINT_SANS,
        "doctor": "Dr. Sarah Jenkins, MD - Internal Medicine",
        "clinic": "Metro General Hospital Clinic",
        "medicine": "Amoxicillin 500mg",
        "instructions": "Take 1 capsule TDS for 7 days after meals",
        "critical": {
            "medicine_name": "Amoxicillin",
            "dosage": "500mg",
            "frequency": "TDS",
            "duration": "7 days",
            "numeric_instruction": "1 capsule",
        },
    },
    {
        "id": "degraded_02",
        "category": "degraded",
        "degradation_type": "low_resolution",
        "font_path": FONT_PRINT_SERIF,
        "doctor": "Dr. Robert Chen, MBBS - Endocrinology",
        "clinic": "Apex Diabetes & Wellness Center",
        "medicine": "Metformin HCl 500mg",
        "instructions": "Take 1 tablet BD with meals for 30 days",
        "critical": {
            "medicine_name": "Metformin HCl",
            "dosage": "500mg",
            "frequency": "BD",
            "duration": "30 days",
            "numeric_instruction": "1 tablet",
        },
    },
    {
        "id": "degraded_03",
        "category": "degraded",
        "degradation_type": "gaussian_blur",
        "font_path": FONT_PRINT_SANS,
        "doctor": "Dr. Emily Taylor, MD - Cardiology",
        "clinic": "Highland Heart Institute",
        "medicine": "Atorvastatin 10mg",
        "instructions": "Take 1 tablet OD at bedtime for 60 days",
        "critical": {
            "medicine_name": "Atorvastatin",
            "dosage": "10mg",
            "frequency": "OD",
            "duration": "60 days",
            "numeric_instruction": "1 tablet",
        },
    },
    {
        "id": "degraded_04",
        "category": "degraded",
        "degradation_type": "motion_blur",
        "font_path": FONT_PRINT_SERIF,
        "doctor": "Dr. John Wilson, MD - Infectious Disease",
        "clinic": "St. Jude Community Health",
        "medicine": "Ciprofloxacin 250mg",
        "instructions": "Take 1 tablet BD for 5 days with water",
        "critical": {
            "medicine_name": "Ciprofloxacin",
            "dosage": "250mg",
            "frequency": "BD",
            "duration": "5 days",
            "numeric_instruction": "1 tablet",
        },
    },
    {
        "id": "degraded_05",
        "category": "degraded",
        "degradation_type": "uneven_lighting",
        "font_path": FONT_PRINT_SANS,
        "doctor": "Dr. Mark Davis, MD - Pulmonology",
        "clinic": "Valley Respiratory Associates",
        "medicine": "Azithromycin 500mg",
        "instructions": "Take 1 tablet OD for 3 days after meals",
        "critical": {
            "medicine_name": "Azithromycin",
            "dosage": "500mg",
            "frequency": "OD",
            "duration": "3 days",
            "numeric_instruction": "1 tablet",
        },
    },
    {
        "id": "degraded_06",
        "category": "degraded",
        "degradation_type": "shadow_occlusion",
        "font_path": FONT_PRINT_SANS,
        "doctor": "Dr. Priya Patel, MD - Gastroenterology",
        "clinic": "Sunset Digestive Care",
        "medicine": "Omeprazole 40mg",
        "instructions": "Take 1 capsule OD before breakfast for 14 days",
        "critical": {
            "medicine_name": "Omeprazole",
            "dosage": "40mg",
            "frequency": "OD",
            "duration": "14 days",
            "numeric_instruction": "1 capsule",
        },
    },
    {
        "id": "degraded_07",
        "category": "degraded",
        "degradation_type": "jpeg_compression",
        "font_path": FONT_PRINT_SERIF,
        "doctor": "Dr. Alan Brooks, MD - Nephrology",
        "clinic": "Cedar Valley Renal Clinic",
        "medicine": "Losartan 25mg",
        "instructions": "Take 1 tablet OD in morning for 30 days",
        "critical": {
            "medicine_name": "Losartan",
            "dosage": "25mg",
            "frequency": "OD",
            "duration": "30 days",
            "numeric_instruction": "1 tablet",
        },
    },
    {
        "id": "degraded_08",
        "category": "degraded",
        "degradation_type": "low_contrast",
        "font_path": FONT_PRINT_SERIF,
        "doctor": "Dr. Lisa Wong, MD - Family Medicine",
        "clinic": "Bayside Medical Group",
        "medicine": "Paracetamol 500mg",
        "instructions": "Take 1 tablet SOS every 6 hours for 3 days",
        "critical": {
            "medicine_name": "Paracetamol",
            "dosage": "500mg",
            "frequency": "SOS",
            "duration": "3 days",
            "numeric_instruction": "1 tablet",
        },
    },
    {
        "id": "degraded_09",
        "category": "degraded",
        "degradation_type": "noise_speckle",
        "font_path": FONT_PRINT_SANS,
        "doctor": "Dr. Karen White, MD - Orthopedics",
        "clinic": "Summit Joint & Spine Clinic",
        "medicine": "Ibuprofen 200mg",
        "instructions": "Take 1 tablet TDS after meals for 5 days",
        "critical": {
            "medicine_name": "Ibuprofen",
            "dosage": "200mg",
            "frequency": "TDS",
            "duration": "5 days",
            "numeric_instruction": "1 tablet",
        },
    },
    {
        "id": "degraded_10",
        "category": "degraded",
        "degradation_type": "compound_degradation",
        "font_path": FONT_PRINT_SANS,
        "doctor": "Dr. David Miller, MD - Allergy & Immunology",
        "clinic": "Northwest Asthma Center",
        "medicine": "Montelukast 5mg",
        "instructions": "Take 1 tablet OD at bedtime for 30 days",
        "critical": {
            "medicine_name": "Montelukast",
            "dosage": "5mg",
            "frequency": "OD",
            "duration": "30 days",
            "numeric_instruction": "1 tablet",
        },
    },
]


def render_base_prescription(spec: dict[str, Any]) -> Image.Image:
    """Render a clean synthetic prescription document."""
    width, height = 750, 480
    image = Image.new("RGB", (width, height), color=(252, 252, 250))
    draw = ImageDraw.Draw(image)

    # Load fonts
    font_path = spec["font_path"]
    is_handwritten = spec["category"] == "handwritten"
    body_font_size = 28 if is_handwritten else 24
    header_font_size = 20

    header_font = ImageFont.truetype(FONT_PRINT_SANS, header_font_size)
    body_font = ImageFont.truetype(font_path, body_font_size)
    bold_rx_font = ImageFont.truetype(FONT_PRINT_SERIF, 34)

    # Draw border
    draw.rectangle([(20, 20), (width - 20, height - 20)], outline=(180, 180, 180), width=2)

    # Clinic Header (Always printed)
    draw.text((40, 35), spec["clinic"], font=header_font, fill=(40, 40, 40))
    draw.text((40, 65), spec["doctor"], font=header_font, fill=(80, 80, 80))
    draw.line([(35, 95), (width - 35, 95)], fill=(200, 200, 200), width=2)

    # Rx Symbol
    draw.text((45, 115), "Rx", font=bold_rx_font, fill=(20, 20, 20))

    # Medicine Line
    draw.text((110, 125), spec["medicine"], font=body_font, fill=(15, 15, 15))

    # Instructions Line
    draw.text((110, 180), spec["instructions"], font=body_font, fill=(25, 25, 25))

    # Footer note & Date
    footer_font = ImageFont.truetype(FONT_PRINT_SANS, 16)
    draw.line([(35, 390), (width - 35, 390)], fill=(220, 220, 220), width=1)
    draw.text((40, 410), "Date: 15/11/2026", font=footer_font, fill=(100, 100, 100))
    draw.text((width - 250, 410), "Signature: ____________", font=footer_font, fill=(100, 100, 100))

    return image


def apply_degradation(image: Image.Image, degradation_type: str) -> Image.Image:
    """Apply controlled physical degradation effects."""
    img_np = np.array(image)

    if degradation_type == "skew_rotation":
        # 4.5 degree rotation
        h, w = img_np.shape[:2]
        center = (w // 2, h // 2)
        m = cv2.getRotationMatrix2D(center, 4.5, 1.0)
        img_np = cv2.warpAffine(img_np, m, (w, h), borderValue=(255, 255, 255))
        return Image.fromarray(img_np)

    elif degradation_type == "low_resolution":
        # Downsample to 35% then resize back
        h, w = img_np.shape[:2]
        down = cv2.resize(img_np, (int(w * 0.35), int(h * 0.35)), interpolation=cv2.INTER_AREA)
        up = cv2.resize(down, (w, h), interpolation=cv2.INTER_LINEAR)
        return Image.fromarray(up)

    elif degradation_type == "gaussian_blur":
        blurred = cv2.GaussianBlur(img_np, (7, 7), sigmaX=2.0)
        return Image.fromarray(blurred)

    elif degradation_type == "motion_blur":
        # Horizontal motion blur
        kernel_size = 9
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        kernel /= kernel_size
        blurred = cv2.filter2D(img_np, -1, kernel)
        return Image.fromarray(blurred)

    elif degradation_type == "uneven_lighting":
        # Linear lighting gradient from top-left to bottom-right
        h, w = img_np.shape[:2]
        gradient = np.linspace(0.4, 1.1, w)
        gradient = np.tile(gradient, (h, 1))
        gradient = np.stack([gradient] * 3, axis=-1)
        shaded = np.clip(img_np.astype(np.float32) * gradient, 0, 255).astype(np.uint8)
        return Image.fromarray(shaded)

    elif degradation_type == "shadow_occlusion":
        # Diagonal dark shadow mask
        h, w = img_np.shape[:2]
        mask = np.ones((h, w), dtype=np.float32)
        for y in range(h):
            for x in range(w):
                if x + y < (w + h) * 0.45:
                    mask[y, x] = 0.55
        mask = cv2.GaussianBlur(mask, (31, 31), 10)
        mask = np.stack([mask] * 3, axis=-1)
        shaded = np.clip(img_np.astype(np.float32) * mask, 0, 255).astype(np.uint8)
        return Image.fromarray(shaded)

    elif degradation_type == "jpeg_compression":
        # Heavy JPEG compression
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 12]
        _, encimg = cv2.imencode(".jpg", img_np, encode_param)
        decimg = cv2.imdecode(encimg, 1)
        return Image.fromarray(decimg)

    elif degradation_type == "low_contrast":
        # Washed out low contrast
        alpha = 0.45  # Contrast control
        beta = 110  # Brightness control
        adjusted = cv2.convertScaleAbs(img_np, alpha=alpha, beta=beta)
        return Image.fromarray(adjusted)

    elif degradation_type == "noise_speckle":
        # Salt & pepper speckle noise
        noisy = img_np.copy()
        h, w, c = noisy.shape
        num_salt = int(0.005 * h * w)
        num_pepper = int(0.005 * h * w)
        # Salt
        for _ in range(num_salt):
            y = np.random.randint(0, h)
            x = np.random.randint(0, w)
            noisy[y, x] = [255, 255, 255]
        # Pepper
        for _ in range(num_pepper):
            y = np.random.randint(0, h)
            x = np.random.randint(0, w)
            noisy[y, x] = [30, 30, 30]
        return Image.fromarray(noisy)

    elif degradation_type == "compound_degradation":
        # Low contrast + mild blur + 3 degree rotation
        h, w = img_np.shape[:2]
        m = cv2.getRotationMatrix2D((w // 2, h // 2), 3.0, 1.0)
        rotated = cv2.warpAffine(img_np, m, (w, h), borderValue=(255, 255, 255))
        blurred = cv2.GaussianBlur(rotated, (5, 5), 1.2)
        low_con = cv2.convertScaleAbs(blurred, alpha=0.6, beta=70)
        return Image.fromarray(low_con)

    return image


def generate_all_fixtures() -> list[dict[str, Any]]:
    """Generate all 30 prescription evaluation images and build ground-truth manifest."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []

    for spec in PRESCRIPTION_SPECS:
        item_id = spec["id"]
        img = render_base_prescription(spec)

        if spec["category"] == "degraded":
            img = apply_degradation(img, spec["degradation_type"])

        filename = f"{item_id}.png"
        file_path = IMAGES_DIR / filename
        img.save(file_path, "PNG")

        # Ground truth lines exactly as rendered on the image
        ground_truth_lines = [
            spec["clinic"],
            spec["doctor"],
            "Rx",
            spec["medicine"],
            spec["instructions"],
            "Date: 15/11/2026",
            "Signature: ____________",
        ]
        ground_truth_text = "\n".join(ground_truth_lines)

        manifest.append(
            {
                "id": item_id,
                "filename": filename,
                "relative_path": f"tests/fixtures/ocr/images/{filename}",
                "category": spec["category"],
                "degradation_type": spec["degradation_type"],
                "doctor": spec["doctor"],
                "clinic": spec["clinic"],
                "medicine": spec["medicine"],
                "instructions": spec["instructions"],
                "ground_truth_lines": ground_truth_lines,
                "ground_truth_text": ground_truth_text,
                "critical_entities": spec["critical"],
            }
        )

    manifest_file = DATASET_DIR / "dataset_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Generated {len(manifest)} test images in {IMAGES_DIR}")
    print(f"Manifest written to {manifest_file}")
    return manifest


if __name__ == "__main__":
    generate_all_fixtures()
