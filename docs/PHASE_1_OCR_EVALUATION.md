# Phase 1 Evaluation Report: PaddleOCR 3.x Evaluation & Benchmark for Medikiosk OCR

**Document Version:** 1.0.0  
**Date:** 2026-11-15  
**Author:** Medikiosk OCR Microservice Engineering  
**Scope:** OCR Engine Feasibility, Compatibility, Performance, and Preprocessing Benchmark  
**Artifact Path:** [docs/PHASE_1_OCR_EVALUATION.md](file:///home/asta/Projects/medikiosk-ocr/docs/PHASE_1_OCR_EVALUATION.md)  
**Status:** Evaluation Completed — Decision Gate: **CONDITIONAL PASS**

---

## Table of Contents

1. [Executive Summary & Decision Gate Recommendation](#1-executive-summary--decision-gate-recommendation)
2. [Environment & Hardware Specifications](#2-environment--hardware-specifications)
3. [Python Runtime & Isolated Environment Setup](#3-python-runtime--isolated-environment-setup)
4. [PaddlePaddle & PaddleOCR Version Resolution](#4-paddlepaddle--paddleocr-version-resolution)
5. [Installation & Verification Commands](#5-installation--verification-commands)
6. [Compatibility Hurdles & Technical Resolutions](#6-compatibility-hurdles--technical-resolutions)
7. [PaddleOCR 3.x Architecture & Model Pipeline](#7-paddleocr-3x-architecture--model-pipeline)
8. [Evaluation Dataset & Test Corpus](#8-evaluation-dataset--test-corpus)
9. [Ground-Truth Evaluation Methodology & Metrics Definitions](#9-ground-truth-evaluation-methodology--metrics-definitions)
10. [Category A: Printed Prescription Results](#10-category-a-printed-prescription-results)
11. [Category B: Handwritten-Style Prescription Results](#11-category-b-handwritten-style-prescription-results)
12. [Category C: Degraded Document Results](#12-category-c-degraded-document-results)
13. [Overall Accuracy Metrics & Failure Mode Classification](#13-overall-accuracy-metrics--failure-mode-classification)
14. [Critical Healthcare Entity Accuracy Breakdown](#14-critical-healthcare-entity-accuracy-breakdown)
15. [Performance, Latency & Memory Footprint Analysis](#15-performance-latency--memory-footprint-analysis)
16. [Image Preprocessing Benchmarks & Comparison](#16-image-preprocessing-benchmarks--comparison)
17. [Automated Test Suite Results](#17-automated-test-suite-results)
18. [Docker Containerization & Deployment Feasibility](#18-docker-containerization--deployment-feasibility)
19. [Known Limitations, Clinical Safety Warnings & Phase 2 Recommendations](#19-known-limitations-clinical-safety-warnings--phase-2-recommendations)

---

## 1. Executive Summary & Decision Gate Recommendation

### 1.1 Executive Summary
During Phase 1 of the **Medikiosk OCR** microservice project, we conducted an empirical feasibility study and quantitative evaluation of **PaddleOCR 3.x** on CPU inference. The objective was to determine whether PaddleOCR can reliably extract clinical information from patient-uploaded prescriptions—spanning printed computer-generated documents, handwritten prescription scripts, and degraded/noisy camera captures—while operating within memory, latency, and containerization constraints suitable for future microservice deployment on Render.

An automated test harness was constructed evaluating **30 synthetic prescription images** containing **150 annotated critical healthcare entities** (medicine names, dosages, frequencies, durations, and numeric administration instructions). Across the baseline evaluation, PaddleOCR 3.x achieved:
- **Overall Mean Character Error Rate (CER):** **1.10%** (Median: 0.00%)
- **Overall Mean Word Error Rate (WER):** **4.69%** (Median: 0.00%)
- **Critical Entity Exact-Match Accuracy:** **92.67%** (139 / 150 critical fields perfectly matched)
- **Mean Warm Inference Latency:** **2.458 seconds** on an Intel Core i5 CPU
- **Containerized Footprint:** 2.3 GB uncompressed Docker image (533 MB compressed layer transfer size)

### 1.2 Decision Gate Recommendation: CONDITIONAL PASS

| Decision Gate Criteria | Evaluated Result | Threshold | Status |
|---|---|---|---|
| Printed Prescription CER | **0.36%** | < 2.0% | **PASS** |
| Printed Critical Entity Accuracy | **96.00%** | > 95.0% | **PASS** |
| Degraded Document Tolerance | **2.38%** CER (96.0% Critical Accuracy) | < 5.0% CER | **PASS** |
| CPU Inference Latency (Warm) | **2.46 s** (Median: 2.38 s) | < 5.0 s | **PASS** |
| Automated Test Pass Rate | **100%** (8/8 passed in 7.49 s) | 100% | **PASS** |
| Docker Container Inference | Validated (`medikiosk-ocr:paddle-test`) | Functional | **PASS** |
| Handwritten Prescription CER | **0.55%** (Caveat / Kalam / Dancing Script) | < 5.0% | **CONDITIONAL** |
| Handwritten Critical Entity Accuracy | **86.00%** (Frequency recognition: 60.0%) | > 90.0% | **CONDITIONAL** |
| Memory Footprint (RSS) | **2.13 GB** peak memory consumption | < 2.5 GB | **CONDITIONAL** |

#### Why CONDITIONAL PASS?
1. **Clinical Field Ambiguity in Handwriting:** PaddleOCR demonstrates remarkable structural document orientation and printed text extraction. However, on handwriting scripts, OCR engines suffer from systematic alphanumeric character substitutions that are clinically hazardous:
   - Recognizing **"0D"** instead of **"OD"** (*once daily*), dropping frequency accuracy to **60.0%**.
   - Recognizing **"I tab"** instead of **"1 tab"** (letter *I* instead of digit *1*).
   - Recognizing **"Smg"** instead of **"5mg"** (letter *S* instead of digit *5*).
   - Severe failure under directional motion blur (`degraded_04`: **19.87% CER**, misidentifying *"Ciprofloxacin"* as *"Ciprofloracin"* and *"250mg"* as *"250ng"*).
2. **Clinical Safety Mandate:** Raw OCR output **cannot** be directly committed into Cloud Firestore or patient electronic health records (EHR) without a deterministic post-processing normalization layer, clinical fuzzy dictionary matching, and mandatory **Human-in-the-Loop (HITL) pharmacist verification**.
3. **Phase 2 Condition:** Proceeding to Phase 2 (Architecture, Storage & API Foundations) is approved under the condition that PaddleOCR is paired with:
   - A dedicated regex/fuzzy clinical post-processor for dosage forms and frequencies.
   - Low-confidence alert flags signaling manual human review when entity extraction confidence drops below 0.90.
   - Restricting automated image preprocessing to mild grayscale or CLAHE, while strictly prohibiting external affine deskewing which interferes with PaddleOCR's internal orientation classifier.

---

## 2. Environment & Hardware Specifications

The evaluation was executed on the local development machine under real CPU constraints:

- **Operating System:** Linux x86_64 (CachyOS / Arch Linux kernel 6.17)
- **CPU:** Intel® Core™ i5-13420H (8 cores / 12 threads: 4 Performance cores up to 4.6 GHz, 4 Efficient cores up to 3.4 GHz)
- **Host System RAM:** 16 GB DDR5 (Available headroom: ~11 GB)
- **Dedicated GPU:** None (CPU inference exclusively)
- **Terminal Isolation:** Sandboxed Linux execution environment
- **Container Engine:** Docker version 28.5.1
- **File System Working Directory:** `/home/asta/Projects/medikiosk-ocr`

---

## 3. Python Runtime & Isolated Environment Setup

### 3.1 Preservation of Phase 0 Environment
Phase 0 configured the repository foundation with a FastAPI skeleton in `.venv` running Python 3.14.0. Because PaddlePaddle 3.x C-extensions do not support Python 3.14, `.venv` was left strictly unmodified and preserved for microservice API development.

### 3.2 Python 3.11 Standalone Runtime
PaddleOCR 3.x and PaddlePaddle 3.2.0 require Python 3.8–3.12, with 3.11 being the officially verified release. Because the host system only provided Python 3.14 in its package manager, we provisioned an isolated standalone CPython runtime:
- **Standalone Runtime:** CPython 3.11.10 (x86_64-unknown-linux-gnu) located in `.python311-runtime/`
- **Dedicated Virtual Environment:** `.venv-paddle` created at `/home/asta/Projects/medikiosk-ocr/.venv-paddle`
- **Interpreter Path:** `/home/asta/Projects/medikiosk-ocr/.venv-paddle/bin/python3.11`
- **Pip Version:** Pip 26.3

---

## 4. PaddlePaddle & PaddleOCR Version Resolution

Official documentation from PaddleOCR 3.x and PaddlePaddle was consulted. The following exact version pinned set was resolved and documented in [requirements-ocr.txt](file:///home/asta/Projects/medikiosk-ocr/requirements-ocr.txt):

```text
paddlepaddle==3.2.0
paddleocr==3.7.0
paddlex==3.7.2
numpy==1.26.4
opencv-contrib-python==4.10.0.84
pillow==11.1.0
pytest==9.2.0
psutil==7.2.2
```

### Dependency Rationale:
- `paddlepaddle==3.2.0`: Official CPU release for Linux x86_64, compiled against OpenBLAS and oneDNN with AVX2 instruction acceleration.
- `paddleocr==3.7.0`: The stable PaddleOCR 3.x library utilizing PaddleX modular pipelines.
- `paddlex==3.7.2`: The core execution engine for PaddleOCR 3.x model downloading, caching, and inference graphs.
- `numpy==1.26.4`: Strict pin to NumPy 1.x ABI to prevent fatal C-extension crashes (detailed below).
- `opencv-contrib-python==4.10.0.84`: Headless computer vision library for document image manipulation and preprocessing.

---

## 5. Installation & Verification Commands

Installation into `.venv-paddle` was performed using direct wheel downloads from PyPI:

```bash
# 1. Create Python 3.11 virtual environment
.python311-runtime/bin/python3.11 -m venv .venv-paddle

# 2. Upgrade core packaging tools
.venv-paddle/bin/pip install --upgrade pip setuptools wheel

# 3. Install pinned dependencies
.venv-paddle/bin/pip install -r requirements-ocr.txt

# 4. Verify PaddlePaddle and PaddleX installation
.venv-paddle/bin/python -c "import paddle, paddlex; print('Paddle:', paddle.__version__, '| PaddleX:', paddlex.__version__)"
```

**Verification Output:**
```text
Paddle: 3.2.0 | PaddleX: 3.7.2
```

---

## 6. Compatibility Hurdles & Technical Resolutions

During installation and initialization, four significant technical hurdles were identified and resolved:

### 6.1 NumPy 2.x C-ABI Incompatibility
- **Issue:** Modern pip installations default to installing `numpy>=2.0.0`. When `paddlepaddle` or `cv2` imported NumPy 2.x, it failed immediately with:
  `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x as it may crash.`
- **Resolution:** Explicitly pinned `numpy==1.26.4` in [requirements-ocr.txt](file:///home/asta/Projects/medikiosk-ocr/requirements-ocr.txt). All C-extensions compiled and linked cleanly against NumPy 1.26.

### 6.2 Home Directory Cache Read-Only Permissions in Sandbox
- **Issue:** PaddleX defaults to creating and writing downloaded model weights to `~/.paddlex/` and `~/.paddle/`. In restricted sandboxed environments and Docker containers running as unprivileged users, writing to the root user's home directory triggers `OSError: [Errno 30] Read-only file system`.
- **Resolution:** Configured workspace-relative environment variables in [app/services/ocr_engine.py](file:///home/asta/Projects/medikiosk-ocr/app/services/ocr_engine.py):
  ```python
  _WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
  _CACHE_ROOT = _WORKSPACE_ROOT / ".cache"
  os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(_CACHE_ROOT / "paddlex"))
  os.environ.setdefault("PADDLE_HOME", str(_CACHE_ROOT / "paddle"))
  os.environ.setdefault("PADDLEOCR_HOME", str(_CACHE_ROOT / "paddleocr"))
  ```
  This guarantees that all model weights, configurations, and inference graphs remain self-contained within the repository's `.cache` directory (which is added to `.gitignore`).

### 6.3 30-Second Model Source Check Latency
- **Issue:** By default, every time `PaddleOCR()` is initialized, PaddleX attempts to establish an external HTTPS connection to verify model update manifests. In sandboxed or offline environments without network egress, this caused a 30-second connection timeout before fallback.
- **Resolution:** Exported `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK="True"`. This completely eliminated remote network polling, reducing engine instantiation time from **30.7 seconds** to **2.15 seconds**.

### 6.4 Missing System CCache Warning
- **Issue:** PaddlePaddle's C++ extension loader emitted a warning: `No ccache found. Please be aware that recompiling all source files may be required.`
- **Resolution:** This is purely informational for runtime C++ JIT compilation. Standard inference uses pre-compiled static shared libraries (`libpaddle.so`), so execution proceeds without degradation.

---

## 7. PaddleOCR 3.x Architecture & Model Pipeline

PaddleOCR 3.x utilizes a multi-stage document intelligence pipeline built on PaddleX. When invoking `PaddleOCR(lang="en")`, a graph of 5 specialized neural models is executed:

```
Input Image (Prescription)
        │
        ▼
[ 1. PP-LCNet_x1_0_doc_ori ]  ──> Classifies document orientation (0°, 90°, 180°, 270°)
        │
        ▼
[ 2. UVDoc ]                  ──> Dewarping module: Rectifies curved or folded page paper
        │
        ▼
[ 3. PP-LCNet_x1_0_textline ] ──> Detects inverted textlines (180° rotation correction)
        │
        ▼
[ 4. PP-OCRv6_medium_det ]    ──> DBNet text detection: Generates bounding polygons
        │
        ▼
[ 5. PP-OCRv6_medium_rec ]    ──> SVTR text recognition: Character CTC sequence decoder
        │
        ▼
Structured OCR Result (Polygons, Confidences, Text Lines)
```

### Cached Model Weights Breakdown
All model weights were fetched and cached in `.cache/paddlex/official_models/`:

| Model Pipeline Stage | Model Identifier | Parameter Size | Primary Purpose |
|---|---|---|---|
| Document Orientation | `PP-LCNet_x1_0_doc_ori` | 6.5 MB | Coarse 90-degree document rotation classification |
| Page Dewarping | `UVDoc` | 31.0 MB | Geometry rectification for mobile camera paper curves |
| Textline Orientation | `PP-LCNet_x1_0_textline_ori` | 6.5 MB | Inverted 180-degree line orientation correction |
| Text Detection | `PP-OCRv6_medium_det` | 60.0 MB | Differentiable Binarization (DBNet) text localization |
| Text Recognition | `PP-OCRv6_medium_rec` | 73.0 MB | Vision Transformer (SVTR) text recognition with CTC |
| **Total Model Footprint** | **5 Official Models** | **177.0 MB** | Complete end-to-end OCR pipeline on disk |

---

## 8. Evaluation Dataset & Test Corpus

To rigorously evaluate PaddleOCR under zero data-leakage and zero PHI conditions, an automated generator script ([scripts/generate_dataset.py](file:///home/asta/Projects/medikiosk-ocr/scripts/generate_dataset.py)) was constructed. The dataset contains **30 distinct synthetic prescription images** categorized into three balanced groups of 10 documents each:

### 8.1 Dataset Composition
1. **Category A: Printed Prescriptions (`printed_01` – `printed_10`):**
   - Clean, standardized hospital clinic letterhead prescriptions.
   - Fonts: DejaVu Sans (Modern clean sans-serif) and DejaVu Serif (Formal serif typography).
   - Simulates electronic health record (EHR) printed discharge summaries.
2. **Category B: Handwritten Prescriptions (`handwritten_01` – `handwritten_10`):**
   - Simulates varying doctor handwriting styles, cursiveness, and stroke connections.
   - Fonts licensed under SIL Open Font License (OFL 1.1):
     - *Caveat Regular:* Casual, fluid handwriting with varied stroke weights.
     - *Kalam Regular:* Informal, organic Indian handwriting script.
     - *Dancing Script Regular:* Lively cursive script with connected loops and descenders.
3. **Category C: Degraded Document Prescriptions (`degraded_01` – `degraded_10`):**
   - Tests physical document capture artifacts encountered by patient smartphone cameras:
     - `degraded_01`: Skew rotation (5.5° affine rotation with background padding)
     - `degraded_02`: Low resolution (downsampled 4x then upsampled with bilinear blur)
     - `degraded_03`: Gaussian defocus blur (radius $\sigma = 2.0$)
     - `degraded_04`: Directional motion blur ($15\text{px}$ linear kernel at 45°)
     - `degraded_05`: Uneven non-uniform illumination gradient (diagonal brightness decay)
     - `degraded_06`: Geometric shadow occlusion (elliptical dark vignette cast over text)
     - `degraded_07`: Heavy JPEG compression artifacts (quality setting $Q=15$)
     - `degraded_08`: Low document contrast (dynamic range scaled down 75%)
     - `degraded_09`: Salt-and-pepper speckle noise ($p=0.03$)
     - `degraded_10`: Compound real-world degradation (skew + noise + low contrast combined)

### 8.2 Annotation Manifest
Every image is paired with a ground-truth JSON manifest ([tests/fixtures/ocr/dataset_manifest.json](file:///home/asta/Projects/medikiosk-ocr/tests/fixtures/ocr/dataset_manifest.json)) tracking:
- Full line-by-line ground-truth text
- Clinic and prescriber credentials
- Five annotated **critical healthcare entities**:
  1. `medicine_name`: The active pharmaceutical drug (e.g., *Amoxicillin*, *Metformin HCl*, *Atorvastatin*).
  2. `dosage`: Active strength and unit (e.g., *500mg*, *850mg*, *10mg*).
  3. `frequency`: Dosing regimen acronym (e.g., *OD*, *BD*, *TDS*, *SOS*).
  4. `duration`: Course duration (e.g., *7 days*, *30 days*, *5 days*).
  5. `numeric_instruction`: Unit administration quantity (e.g., *1 capsule*, *1 tablet*, *2 tabs*).

---

## 9. Ground-Truth Evaluation Methodology & Metrics Definitions

### 9.1 Character Error Rate (CER)
CER measures the minimum character-level edit distance between ground truth $T_{\text{gt}}$ and OCR recognized text $T_{\text{ocr}}$, normalized by the character length of the ground truth:

$$\text{CER} = \frac{S_c + D_c + I_c}{N_c}$$

Where:
- $S_c$ = Character substitutions
- $D_c$ = Character deletions
- $I_c$ = Character insertions
- $N_c$ = Total character count in ground truth

Calculated using the Wagner-Fischer dynamic programming algorithm for Levenshtein distance.

### 9.2 Word Error Rate (WER)
WER measures the minimum token-level edit distance between whitespace-tokenized sequences:

$$\text{WER} = \frac{S_w + D_w + I_w}{N_w}$$

Where $N_w$ is the total count of words in the ground truth document.

### 9.3 Text Normalization Rules
To ensure error measurements reflect semantic textual quality rather than irrelevant whitespace variations, both ground-truth and OCR texts were normalized prior to computing Levenshtein distance:
1. Stripping non-printable control characters.
2. Collapsing consecutive whitespace sequences into single spaces (`"  "` $\to$ `" "`).
3. Trimming leading and trailing whitespace.
4. Preserving case-sensitivity to strictly detect capitalization anomalies (e.g., `"OD"` vs `"0D"`).

### 9.4 Critical Healthcare Entity Exact-Match
For clinical utility, general CER is insufficient—a 0.5% CER can represent a fatal clinical error if the altered character changes `"10mg"` to `"70mg"`.
Exact-match accuracy tests whether each of the 5 annotated critical entity strings is found verbatim inside the recognized OCR text:

$$\text{Match}(E_i) = \begin{cases} 1 & \text{if } E_i \subseteq T_{\text{ocr}} \\ 0 & \text{otherwise} \end{cases}$$

Critical Accuracy for an image:

$$\text{Accuracy}_{\text{critical}} = \frac{1}{5} \sum_{i=1}^{5} \text{Match}(E_i)$$

---

## 10. Category A: Printed Prescription Results

The baseline evaluation of 10 printed prescription documents demonstrated high accuracy, proving PaddleOCR 3.x is suitable for printed medical paperwork.

### 10.1 Printed Results Table

| Image ID | Font Style | Latency (s) | CER (%) | WER (%) | Critical Match Acc | Avg Conf | Individual Status |
|---|---|---|---|---|---|---|---|
| `printed_01` | DejaVu Sans | 2.263 | **0.00%** | **0.00%** | **100.0%** (5/5) | 0.9953 | **PASS** |
| `printed_02` | DejaVu Serif | 1.934 | **0.00%** | **0.00%** | **100.0%** (5/5) | 0.9965 | **PASS** |
| `printed_03` | DejaVu Sans | 2.127 | **0.00%** | **0.00%** | **100.0%** (5/5) | 0.9960 | **PASS** |
| `printed_04` | DejaVu Sans | 2.487 | **2.99%** | **3.45%** | **80.0%** (4/5) | 0.9891 | **PARTIAL** |
| `printed_05` | DejaVu Serif | 2.573 | **0.00%** | **0.00%** | **100.0%** (5/5) | 0.9946 | **PASS** |
| `printed_06` | DejaVu Sans | 2.866 | **0.00%** | **0.00%** | **100.0%** (5/5) | 0.9952 | **PASS** |
| `printed_07` | DejaVu Serif | 2.832 | **0.00%** | **0.00%** | **100.0%** (5/5) | 0.9923 | **PASS** |
| `printed_08` | DejaVu Serif | 3.348 | **0.63%** | **3.57%** | **80.0%** (4/5) | 0.9894 | **PARTIAL** |
| `printed_09` | DejaVu Sans | 2.445 | **0.00%** | **0.00%** | **100.0%** (5/5) | 0.9960 | **PASS** |
| `printed_10` | DejaVu Serif | 2.568 | **0.00%** | **0.00%** | **100.0%** (5/5) | 0.9977 | **PASS** |

### 10.2 Category Summary
- **Mean CER:** **0.36%**
- **Median CER:** **0.00%**
- **Mean WER:** **0.70%**
- **Median WER:** **0.00%**
- **Mean Latency:** **2.544 s** (Median: 2.528 s)
- **Critical Entity Accuracy:** **96.00%** (48 / 50 entities matched)

### 10.3 Analysis of Printed Discrepancies
Seven out of ten printed documents achieved **0.00% CER** (perfect 100% transcriptions). Only two documents exhibited minor discrepancies:
1. `printed_04`: Ground truth duration was `"4 days"`. The text detection bounding box cropped the right-hand boundary of the line prematurely, yielding `"for 4 "` (omitting `"days"`).
2. `printed_08`: Ground truth duration was `"3 days"`. The CTC decoder dropped the trailing plural 's', yielding `"3 day"`.
Crucially, all 10 medicine names, 10 dosages, 10 frequencies, and 10 administration instructions were matched with **100.0% accuracy**.

---

## 11. Category B: Handwritten-Style Prescription Results

Category B evaluated 10 prescriptions rendered in handwriting fonts (Caveat, Kalam, and Dancing Script). While general CER remained under 1%, detailed inspection revealed systematic character substitutions.

### 11.1 Handwritten Results Table

| Image ID | Font Style | Latency (s) | CER (%) | WER (%) | Critical Match Acc | Avg Conf | Individual Status |
|---|---|---|---|---|---|---|---|
| `handwritten_01` | Caveat (Neat) | 2.192 | **0.00%** | **0.00%** | **100.0%** (5/5) | 0.9897 | **PASS** |
| `handwritten_02` | Kalam (Moderate) | 2.102 | **0.82%** | **5.00%** | **80.0%** (4/5) | 0.9951 | **PARTIAL** |
| `handwritten_03` | Dancing Script (Cursive) | 2.205 | **0.00%** | **0.00%** | **100.0%** (5/5) | 0.9932 | **PASS** |
| `handwritten_04` | Caveat (Neat) | 2.397 | **0.79%** | **5.00%** | **80.0%** (4/5) | 0.9887 | **PARTIAL** |
| `handwritten_05` | Kalam (Moderate) | 1.938 | **0.00%** | **0.00%** | **100.0%** (5/5) | 0.9920 | **PASS** |
| `handwritten_06` | Dancing Script (Cursive) | 2.130 | **0.00%** | **0.00%** | **100.0%** (5/5) | 0.9929 | **PASS** |
| `handwritten_07` | Caveat (Neat) | 2.009 | **1.63%** | **9.52%** | **60.0%** (3/5) | 0.9893 | **PARTIAL** |
| `handwritten_08` | Kalam (Moderate) | 2.044 | **0.75%** | **4.55%** | **80.0%** (4/5) | 0.9938 | **PARTIAL** |
| `handwritten_09` | Dancing Script (Cursive) | 2.047 | **0.78%** | **4.35%** | **80.0%** (4/5) | 0.9907 | **PARTIAL** |
| `handwritten_10` | Caveat (Neat) | 1.997 | **0.76%** | **4.76%** | **80.0%** (4/5) | 0.9898 | **PARTIAL** |

### 11.2 Category Summary
- **Mean CER:** **0.55%**
- **Median CER:** **0.75%**
- **Mean WER:** **3.32%**
- **Median WER:** **4.45%**
- **Mean Latency:** **2.106 s** (Median: 2.074 s)
- **Critical Entity Accuracy:** **86.00%** (43 / 50 entities matched)

### 11.3 Analysis of Handwritten Discrepancies
While three documents achieved 100% exact matches, seven documents revealed significant OCR substitution patterns:
- **Digit / Capital Letter Confusions:**
  - `handwritten_04`: Ground truth `"OD"` was recognized as `"0D"` (capital *O* recognized as numeral zero *0*).
  - `handwritten_07`: Ground truth `"OD"` was recognized as `"0D"`.
  - `handwritten_09`: Ground truth `"SOS"` was recognized as `"S0S"`.
  - `handwritten_10`: Ground truth `"OD"` was recognized as `"0D"`.
  - **Result:** Prescribed frequency exact-match dropped to **60.0%** across handwritten prescriptions.
- **Numeral / Letter Confusions:**
  - `handwritten_02` & `handwritten_08`: Ground truth instruction `"1 tab"` was recognized as `"I tab"` (numeral *1* recognized as capital *I*).
  - `handwritten_07`: Ground truth dosage `"5mg"` was recognized as `"Smg"` (numeral *5* recognized as capital *S*).
- **Takeaway:** Medicine names were extracted with **100.0% accuracy**, but numerical strengths and dosing abbreviations require post-OCR phonetic/regex normalization.

---

## 12. Category C: Degraded Document Results

Category C evaluated 10 document captures subject to smartphone photography artifacts.

### 12.1 Degraded Results Table

| Image ID | Degradation Type | Latency (s) | CER (%) | WER (%) | Critical Match Acc | Avg Conf | Individual Status |
|---|---|---|---|---|---|---|---|
| `degraded_01` | Skew Rotation (5.5°) | 2.363 | **0.00%** | **0.00%** | **100.0%** (5/5) | 0.9966 | **PASS** |
| `degraded_02` | Low Resolution (4x down) | 2.222 | **1.87%** | **22.22%** | **100.0%** (5/5) | 0.9888 | **PASS** |
| `degraded_03` | Gaussian Defocus ($\sigma=2.0$) | 2.228 | **0.68%** | **8.33%** | **100.0%** (5/5) | 0.9920 | **PASS** |
| `degraded_04` | Motion Blur (15px @ 45°) | 2.476 | **19.87%** | **53.85%** | **60.0%** (3/5) | 0.7879 | **FAIL** |
| `degraded_05` | Uneven Lighting Gradient | 2.828 | **0.66%** | **8.33%** | **100.0%** (5/5) | 0.9976 | **PASS** |
| `degraded_06` | Shadow Occlusion Vignette | 3.369 | **0.00%** | **0.00%** | **100.0%** (5/5) | 0.9948 | **PASS** |
| `degraded_07` | JPEG Compression ($Q=15$) | 3.155 | **0.00%** | **0.00%** | **100.0%** (5/5) | 0.9897 | **PASS** |
| `degraded_08` | Low Contrast (75% clamp) | 2.677 | **0.00%** | **0.00%** | **100.0%** (5/5) | 0.9942 | **PASS** |
| `degraded_09` | Salt-and-Pepper Noise ($p=0.03$) | 2.622 | **0.67%** | **7.69%** | **100.0%** (5/5) | 0.9849 | **PASS** |
| `degraded_10` | Compound Degradation | 3.107 | **0.00%** | **0.00%** | **100.0%** (5/5) | 0.9978 | **PASS** |

### 12.2 Category Summary
- **Mean CER:** **2.38%**
- **Median CER:** **0.33%**
- **Mean WER:** **10.04%**
- **Median WER:** **3.84%**
- **Mean Latency:** **2.705 s** (Median: 2.649 s)
- **Critical Entity Accuracy:** **96.00%** (48 / 50 entities matched)

### 12.3 Robustness & Failure Mode Analysis
1. **Exceptional Skew, Shadow & Contrast Resilience:**
   - PaddleOCR handled document skew (`degraded_01`), cast shadows (`degraded_06`), low contrast (`degraded_08`), and compound degradation (`degraded_10`) with **0.00% CER**, demonstrating the effectiveness of the integrated `PP-LCNet` orientation and `UVDoc` rectifying pipeline.
2. **Whitespace Segmentation Under Blur (`degraded_02`, `degraded_03`, `degraded_05`, `degraded_09`):**
   - Under moderate blur, character glyphs were correctly read, but word spacing was occasionally merged (e.g., `"Rx Metformin"` $\to$ `"RxMetformin"`, `"Date: 15/11/2026"` $\to$ `"Date:15/11/2026"`). This caused minor WER inflation while preserving **100% critical entity extraction**.
3. **Catastrophic Failure on Directional Motion Blur (`degraded_04`):**
   - Under $15\text{px}$ linear motion blur, the engine suffered severe character distortion:
     - Target `"Ciprofloxacin"` recognized as `"Ciprofloracin"` (CER spike).
     - Target `"250mg"` recognized as `"250ng"` (*mg* distorted to *ng*).
     - Hospital header distorted: `"St. Jude Community Health"` $\to$ `"Sit.judle CommumityHealtth"`.
     - Overall confidence plummeted to **0.7879**.
   - **Mitigation:** The application MUST compute image blur metrics (e.g., Laplacian variance threshold $\sigma^2 < 100$) during image upload and prompt the patient to retake shaky photos prior to triggering OCR inference.

---

## 13. Overall Accuracy Metrics & Failure Mode Classification

### 13.1 Comprehensive Metrics Across All 30 Documents

| Evaluation Suite | Sample Count | Mean CER | Median CER | Mean WER | Median WER | Mean Critical Accuracy | Mean Latency (s) |
|---|---|---|---|---|---|---|---|
| **Category A: Printed** | 10 | 0.36% | 0.00% | 0.70% | 0.00% | 96.00% | 2.544 s |
| **Category B: Handwritten** | 10 | 0.55% | 0.75% | 3.32% | 4.45% | 86.00% | 2.106 s |
| **Category C: Degraded** | 10 | 2.38% | 0.33% | 10.04% | 3.84% | 96.00% | 2.705 s |
| **All Test Documents (Baseline)** | **30** | **1.10%** | **0.00%** | **4.69%** | **0.00%** | **92.67%** | **2.452 s** |

### 13.2 Empirical Failure Mode Taxonomy
From the 30 evaluation runs, errors are classified into three distinct categories:

```text
                                 OCR Error Modes
                                        │
        ┌───────────────────────────────┼───────────────────────────────┐
        ▼                               ▼                               ▼
[ Type I: Alphanumeric Substitution ]  [ Type II: Bounding Box Clip ]   [ Type III: Motion Blur Smear ]
- 0 vs O ("0D" instead of "OD")         - Trailing word omission        - Character collapse ("m" -> "n")
- 1 vs I ("I tab" instead of "1 tab")   - Right margin edge clipping    - Confidence drop (< 0.80)
- 5 vs S ("Smg" instead of "5mg")       - Seen on long sentences        - Hallucinated consonants
```

---

## 14. Critical Healthcare Entity Accuracy Breakdown

Critical entity extraction was measured across all 150 individual entities:

| Clinical Field Type | Printed Acc (%) | Handwritten Acc (%) | Degraded Acc (%) | Overall Exact Match Acc (%) | Clinical Severity Risk |
|---|---|---|---|---|---|
| **Medicine Name** | 100.0% (10/10) | 100.0% (10/10) | 90.0% (9/10) | **96.67%** (29/30) | **CRITICAL** (Wrong medication dispensed) |
| **Dosage Strength** | 100.0% (10/10) | 90.0% (9/10) | 90.0% (9/10) | **93.33%** (28/30) | **HIGH** (Sub-therapeutic or overdose) |
| **Dosing Frequency** | 100.0% (10/10) | 60.0% (6/10) | 100.0% (10/10) | **86.67%** (26/30) | **HIGH** (Frequency misinterpretation) |
| **Course Duration** | 80.0% (8/10) | 100.0% (10/10) | 100.0% (10/10) | **93.33%** (28/30) | **MEDIUM** (Treatment prematurely ended) |
| **Numeric Instruction** | 100.0% (10/10) | 80.0% (8/10) | 100.0% (10/10) | **93.33%** (28/30) | **HIGH** (Incorrect pill count taken) |
| **All Critical Fields Combined** | **96.00%** | **86.00%** | **96.00%** | **92.67%** (139/150) | **Systemic Clinical Reliability** |

### Clinical Risk Observations:
- **Medicine Names:** Maintained **96.67%** overall accuracy. The single failure occurred under extreme motion blur (`degraded_04`).
- **Dosing Frequency:** Suffered the lowest accuracy (**86.67%** overall, **60.0%** in handwriting) due to the identical visual morphology of `"O"` and `"0"`. In Phase 2, a regex mapping rule must convert any token matching `^[0-9]D$` to its canonical abbreviation (e.g., `"0D"` $\to$ `"OD"`).

---

## 15. Performance, Latency & Memory Footprint Analysis

Resource utilization was measured across all 30 consecutive inferences using `psutil`:

### 15.1 Latency Benchmarks (CPU Inference)

| Execution State | Latency (seconds) | Notes / Conditions |
|---|---|---|
| **Engine Cold Initialization** | **2.154 s** | Loading 5 ONNX/Paddle models into memory |
| **Cold Inference (Image 1)** | **2.263 s** | JIT graph warm-up and thread allocation |
| **Warm Inference (Mean)** | **2.458 s** | Average across images 2 through 30 |
| **Overall Mean Latency** | **2.452 s** | All 30 images combined |
| **Overall Median Latency** | **2.380 s** | Robust midpoint measurement |
| **Fastest Inference** | **1.934 s** | `printed_02` (simple layout, 6 lines) |
| **Slowest Inference** | **3.370 s** | `degraded_06` (shadow occlusion, large area) |

### 15.2 Memory Footprint (Resident Set Size - RSS)
- **Base Process Memory (Before Engine Init):** **68.29 MB**
- **Peak Process Memory (After 30 Inferences):** **2,132.15 MB** (~2.08 GB)
- **Net Memory Growth:** **2,063.86 MB**
- **Average CPU Core Utilization:** **84.0%** across available physical cores

### 15.3 Deployment Implications for Render
- Free/Starter micro instances with 512 MB RAM **cannot** host PaddleOCR 3.x; attempting to do so will trigger Linux OOM Killer.
- A **Standard Render instance (minimum 2 GB RAM, recommended 4 GB RAM)** is strictly required for reliable CPU microservice deployment.

---

## 16. Image Preprocessing Benchmarks & Comparison

To investigate whether computer vision preprocessing improves OCR accuracy, we benchmarked the complete 30-image corpus across 5 distinct image processing techniques implemented in [app/services/ocr_engine.py](file:///home/asta/Projects/medikiosk-ocr/app/services/ocr_engine.py).

### 16.1 Comparative Preprocessing Results

| Technique | Description | Overall CER (%) | Overall WER (%) | Critical Match Acc (%) | Degraded CER (%) | Mean Time (s) | Verdict |
|---|---|---|---|---|---|---|---|
| **Baseline (None)** | Raw original source image | **1.10%** | **4.69%** | **92.67%** | **2.38%** | **2.452 s** | **Benchmark Standard** |
| **Grayscale** | Grayscale conversion & 3-channel broadcast | **1.07%** | **4.53%** | **93.33%** | **2.37%** | 2.819 s | **Recommended (Marginal Gain)** |
| **CLAHE Contrast** | Adaptive histogram equalization ($2.0$ clip) | **1.11%** | **4.42%** | **93.33%** | **2.51%** | 2.731 s | **Neutral / Optional** |
| **Denoise** | Non-Local Means Denoising filter | **3.67%** | **7.45%** | **94.00%** | **10.19%** | 3.245 s | **Harmful to Degraded Images** |
| **Deskew** | Min-area bounding box affine rotation | **28.18%** | **40.40%** | **70.00%** | **17.07%** | 2.697 s | **FAIL (Severely Breaks OCR)** |
| **Resize 1.5x** | Bicubic upsampling ($1.5\times$ scale) | **1.34%** | **7.21%** | **92.00%** | **2.77%** | 4.151 s | **Unfavorable (69% Slower)** |

### 16.2 Key Preprocessing Insights

#### 1. Why Grayscale Conversion Provided the Best Benefit
Converting images to grayscale and re-broadcasting as 3-channel arrays marginally improved CER from **1.10% down to 1.07%** and elevated critical accuracy from **92.67% to 93.33%**. Grayscale removes chromatic aberration and color casts (e.g., yellowish camera lighting in `degraded_05`), allowing DBNet edge detection to focus strictly on stroke luminance boundaries.

#### 2. Why External Deskewing Caused Catastrophic Failure (28.18% CER)
Manual contour-based deskewing using OpenCV minimum-area bounding boxes was disastrous. PaddleOCR 3.x already integrates an internal **`PP-LCNet_x1_0_doc_ori`** classifier and **`UVDoc`** dewarper. When an image is externally cropped and rotated via affine transforms, rectangular page borders become diagonal wedges, introducing severe black corner triangles. This confuses PaddleOCR's internal textline orientation model, triggering incorrect 90° and 180° orientation inversions. **Conclusion:** Never apply external affine deskewing prior to PaddleOCR 3.x.

#### 3. Why Denoising Degraded Image Accuracy (10.19% Degraded CER)
Non-Local Means Denoising filters out high-frequency noise, but medical prescription text consists of fine lines and narrow character strokes (such as commas, dots over 'i', and thin decimal points). Denoising blurred these thin stroke edges into the background, tripling degraded document error rates.

#### 4. Upsampling 1.5x Is Counterproductive on CPU
Upscaling images by 1.5x increased mean processing latency from **2.45 s to 4.15 s** (+69% latency penalty) while worsening CER to **1.34%**. DBNet is already trained on multi-scale feature pyramids; artificial interpolation introduces interpolation blur along font glyph contours.

---

## 17. Automated Test Suite Results

A dedicated automated test suite was constructed in [tests/test_ocr_engine.py](file:///home/asta/Projects/medikiosk-ocr/tests/test_ocr_engine.py) with configurations registered in [pytest.ini](file:///home/asta/Projects/medikiosk-ocr/pytest.ini).

### 17.1 Test Execution Output
Command executed:
```bash
PYTHONPATH=. .venv-paddle/bin/pytest -v tests/test_ocr_engine.py
```

```text
============================= test session starts ==============================
platform linux -- Python 3.11.10, pytest-9.2.0, pluggy-1.7.0
rootdir: /home/asta/Projects/medikiosk-ocr
configfile: pytest.ini
collected 8 items

tests/test_ocr_engine.py::test_ocr_data_structures PASSED                [ 12%]
tests/test_ocr_engine.py::test_nonexistent_image_handling PASSED         [ 25%]
tests/test_ocr_engine.py::test_empty_image_handling PASSED               [ 37%]
tests/test_ocr_engine.py::test_empty_numpy_array_handling PASSED         [ 50%]
tests/test_ocr_engine.py::test_image_preprocessing_non_mutating PASSED   [ 62%]
tests/test_ocr_engine.py::test_engine_singleton_initialization PASSED    [ 75%]
tests/test_ocr_engine.py::test_inference_valid_fixture PASSED            [ 87%]
tests/test_ocr_engine.py::test_repeated_inference_same_engine PASSED     [100%]

============================== 8 passed in 7.49s ===============================
```

### 17.2 Test Coverage Breakdown
1. `test_ocr_data_structures`: Verifies `OCRTextLine` and `OCRResult` dataclass serialization, dictionary conversion, and schema attributes.
2. `test_nonexistent_image_handling`: Confirms the engine safely returns `success=False` with a descriptive error without raising unhandled exceptions when passed an invalid path.
3. `test_empty_image_handling`: Confirms handling of 0-byte corrupt files.
4. `test_empty_numpy_array_handling`: Confirms safe validation of zero-dimension image matrices.
5. `test_image_preprocessing_non_mutating`: Validates that image preprocessing operations (resize, CLAHE, grayscale) return copies without mutating the input numpy array.
6. `test_engine_singleton_initialization`: Validates that `PaddleOCREngine.get_instance()` returns a true singleton to prevent reloading model weights.
7. `test_inference_valid_fixture`: Runs end-to-end OCR on `printed_01.png`, validating text extraction, line bounding boxes, and confidence score ranges.
8. `test_repeated_inference_same_engine`: Executes sequential inferences, ensuring memory stability and consistent execution without model drift.

---

## 18. Docker Containerization & Deployment Feasibility

### 18.1 Dockerfile Implementation
A specialized container specification was created in [Dockerfile.ocr](file:///home/asta/Projects/medikiosk-ocr/Dockerfile.ocr):

```dockerfile
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK="True"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-ocr.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-ocr.txt

COPY . .

CMD ["python", "-c", "import paddle; print('Paddle ready:', paddle.__version__)"]
```

### 18.2 Docker Build & Image Metrics
- **Build Status:** Successfully built and tagged as `medikiosk-ocr:paddle-test` (Image ID: `6705b11db444`).
- **Uncompressed Virtual Image Size:** **2.3 GB**
- **Compressed Docker Registry Layer Size:** **533 MB**
- **Base OS:** Debian 12 Bookworm Slim with minimal OpenBLAS and OpenGL runtime libraries (`libgl1`, `libgomp1`).

### 18.3 Live Container Inference Verification
Live container inference was verified by mounting the evaluation corpus and running an end-to-end extraction inside the isolated container:

```bash
docker run --rm -v $(pwd):/app -w /app medikiosk-ocr:paddle-test python -c "
import os
os.environ['PADDLE_PDX_CACHE_HOME'] = '/app/.cache/paddlex'
os.environ['PADDLE_HOME'] = '/app/.cache/paddle'
os.environ['PADDLEOCR_HOME'] = '/app/.cache/paddleocr'
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
from app.services.ocr_engine import PaddleOCREngine
engine = PaddleOCREngine.get_instance()
res = engine.run_ocr('tests/fixtures/ocr/images/printed_01.png')
print('Container OCR Success:', res.success, '| Lines:', len(res.lines), '| Time:', round(res.inference_time, 3), 's | Sample:', res.raw_text[:50].replace('\n', ' '))
"
```

**Output from Container:**
```text
Creating model: ('PP-LCNet_x1_0_doc_ori', None, None)
Model files already exist. Using cached files.
Creating model: ('UVDoc', None, None)
Creating model: ('PP-LCNet_x1_0_textline_ori', None, None)
Creating model: ('PP-OCRv6_medium_det', None, None)
Creating model: ('PP-OCRv6_medium_rec', None, None)
Container OCR Success: True | Lines: 6 | Time: 1.779 s | Sample: Metro General Hospital Clinic Dr. Sarah Jenkins, M
```
Container execution ran successfully, returning clean extraction in **1.779 seconds**.

---

## 19. Known Limitations, Clinical Safety Warnings & Phase 2 Recommendations

### 19.1 Known Technical Limitations
1. **Motion Blur Susceptibility:** PaddleOCR's CTC decoder loses alignment when characters exhibit horizontal or vertical smear. Blurry camera snaps will fail entity extraction.
2. **Handwriting Font Ambiguity:** The engine confuses visually identical glyph pairs in handwritten scripts (`0` vs `O`, `1` vs `I`, `5` vs `S`).
3. **Memory Footprint:** Peak RSS exceeds 2.1 GB during active inference batches, requiring appropriate cloud server sizing.

### 19.2 Clinical Safety Mandates (Non-Negotiable)
> [!CAUTION]
> **Clinical Safety Warning:**
> Under no circumstances should unparsed or unverified OCR text be treated as definitive clinical ground truth. A single character substitution (e.g., misreading `"10mg"` as `"70mg"`, or `"OD"` as `"QDS"`) can cause severe medication dosage errors or patient harm.
>
> 1. **Human-in-the-Loop (HITL) Validation:** All extracted prescription entities MUST be presented to a qualified clinician or pharmacist in the Medikiosk user interface for review and explicit approval before insertion into patient medical records.
> 2. **Confidence Thresholding:** Any field extracted with confidence $< 0.90$ must be flagged with a visual warning in the clinician UI.
> 3. **Input Quality Checks:** Prescription images failing blur, glare, or contrast thresholds must be rejected at the client level before OCR processing.

### 19.3 Architectural Recommendations for Phase 2
When advancing to Phase 2 (Microservice Architecture & Backend Integrations):
1. **Prescription Post-Processor Service:** Implement a specialized `PrescriptionParser` service that applies:
   - Regex-based dosing frequency normalization (`0D` $\to$ `OD`, `S0S` $\to$ `SOS`, `I tab` $\to$ `1 tab`).
   - Fuzzy Levenshtein matching against a standardized drug nomenclature database (e.g., FDA National Drug Code / RxNorm or WHO Essential Medicines list).
2. **FastAPI Asynchronous Worker Model:** Because CPU inference takes 2.0–2.7 seconds per image and consumes significant RAM, API endpoints should not run synchronous inline OCR on the main HTTP thread. Requests should be handled asynchronously via background task workers.
3. **Container Pre-Baked Weights:** In the production Docker build, copy `.cache/paddlex/official_models/` directly into the Docker image filesystem during `docker build`, ensuring zero cold-start download latency upon microservice boot.

---

**Report Prepared By:** Medikiosk OCR Microservice Engineering  
**Approved by Decision Gate:** **CONDITIONAL PASS** for Phase 2 Architectural Design.
