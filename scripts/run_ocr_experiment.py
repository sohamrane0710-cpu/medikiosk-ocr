"""Run baseline OCR experiments across the 30-image evaluation dataset.

Calculates:
- Character Error Rate (CER)
- Word Error Rate (WER)
- Critical entity exact-match accuracy (Medicine name, Dosage, Frequency, Duration, Instructions)
- System performance (cold/warm inference times, peak memory, CPU utilization)

Saves detailed JSON and CSV reports in docs/results/.
"""

from __future__ import annotations

import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any

import psutil

from app.services.ocr_engine import PaddleOCREngine

ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_MANIFEST = ROOT_DIR / "tests" / "fixtures" / "ocr" / "dataset_manifest.json"
RESULTS_DIR = ROOT_DIR / "docs" / "results"


def levenshtein_distance(seq1: list[Any] | str, seq2: list[Any] | str) -> int:
    """Compute the Levenshtein distance between two sequences."""
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = [[0] * size_y for _ in range(size_x)]

    for x in range(size_x):
        matrix[x][0] = x
    for y in range(size_y):
        matrix[0][y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x - 1] == seq2[y - 1]:
                matrix[x][y] = matrix[x - 1][y - 1]
            else:
                matrix[x][y] = min(
                    matrix[x - 1][y] + 1,  # deletion
                    matrix[x][y - 1] + 1,  # insertion
                    matrix[x - 1][y - 1] + 1,  # substitution
                )
    return matrix[size_x - 1][size_y - 1]


def normalize_text_for_eval(text: str) -> str:
    """Standardize text for CER/WER comparison.

    Rules:
    - Replace newlines with spaces.
    - Condense multiple whitespace characters to a single space.
    - Convert to lowercase.
    - Keep all medical abbreviations, numbers, and dosage units intact.
    - Strip decorative trailing underscores from signature lines.
    """
    text = text.replace("\n", " ")
    text = text.replace("_", "")
    words = text.split()
    return " ".join(words).strip().lower()


def compute_cer(ref: str, hyp: str) -> float:
    """Compute Character Error Rate (CER = distance / len(ref))."""
    norm_ref = normalize_text_for_eval(ref)
    norm_hyp = normalize_text_for_eval(hyp)
    if not norm_ref:
        return 0.0 if not norm_hyp else 1.0
    dist = levenshtein_distance(norm_ref, norm_hyp)
    return dist / len(norm_ref)


def compute_wer(ref: str, hyp: str) -> float:
    """Compute Word Error Rate (WER = word_distance / len(ref_words))."""
    norm_ref = normalize_text_for_eval(ref)
    norm_hyp = normalize_text_for_eval(hyp)
    ref_words = norm_ref.split()
    hyp_words = norm_hyp.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    dist = levenshtein_distance(ref_words, hyp_words)
    return dist / len(ref_words)


def evaluate_critical_entities(
    critical_spec: dict[str, str], raw_recognized_text: str
) -> dict[str, bool]:
    """Check exact match presence of critical prescription fields in recognized text."""
    norm_text = raw_recognized_text.lower().replace("\n", " ")
    # Also create a collapsed alphanumeric version for concatenated strings (e.g. 500mg vs 500 mg)
    collapsed_text = "".join(ch for ch in norm_text if ch.isalnum())

    results: dict[str, bool] = {}
    for key, expected_val in critical_spec.items():
        val_norm = expected_val.lower().strip()
        val_collapsed = "".join(ch for ch in val_norm if ch.isalnum())

        # Match either exact normalized substring or collapsed alphanumeric substring
        exact_match = (val_norm in norm_text) or (val_collapsed in collapsed_text)
        results[key] = exact_match

    return results


def run_experiment(
    apply_preprocessing: bool = False,
    preprocessing_kwargs: dict[str, Any] | None = None,
    output_prefix: str = "baseline",
) -> dict[str, Any]:
    """Execute complete benchmark suite across the 30 evaluation images."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(DATASET_MANIFEST, "r", encoding="utf-8") as f:
        manifest: list[dict[str, Any]] = json.load(f)

    process = psutil.Process()
    mem_before_mb = process.memory_info().rss / (1024 * 1024)
    cpu_percent_start = psutil.cpu_percent(interval=None)

    print("================================================================================")
    print(f"RUNNING OCR EVALUATION SUITE ({output_prefix.upper()})")
    print(f"Dataset Size: {len(manifest)} images")
    print(f"Preprocessing: {apply_preprocessing} ({preprocessing_kwargs or 'none'})")
    print("================================================================================")

    t_init_start = time.perf_counter()
    engine = PaddleOCREngine.get_instance()
    init_duration = time.perf_counter() - t_init_start

    detailed_results: list[dict[str, Any]] = []
    inference_times: list[float] = []

    for idx, item in enumerate(manifest):
        image_path = ROOT_DIR / item["relative_path"]
        ground_truth = item["ground_truth_text"]

        ocr_res = engine.run_ocr(
            image_path,
            apply_preprocessing=apply_preprocessing,
            preprocessing_kwargs=preprocessing_kwargs,
        )

        inference_times.append(ocr_res.inference_time)

        cer = compute_cer(ground_truth, ocr_res.raw_text)
        wer = compute_wer(ground_truth, ocr_res.raw_text)
        critical_eval = evaluate_critical_entities(item["critical_entities"], ocr_res.raw_text)

        # Average confidence across all detected lines
        confidences = [line.confidence for line in ocr_res.lines]
        avg_confidence = statistics.mean(confidences) if confidences else 0.0

        record = {
            "id": item["id"],
            "category": item["category"],
            "degradation_type": item["degradation_type"],
            "image_filename": item["filename"],
            "success": ocr_res.success,
            "inference_time_sec": round(ocr_res.inference_time, 4),
            "line_count": len(ocr_res.lines),
            "avg_confidence": round(avg_confidence, 4),
            "cer": round(cer, 4),
            "wer": round(wer, 4),
            "critical_matches": critical_eval,
            "critical_accuracy": round(sum(critical_eval.values()) / len(critical_eval), 4),
            "ground_truth_text": ground_truth,
            "raw_ocr_text": ocr_res.raw_text,
            "error_message": ocr_res.error_message,
        }
        detailed_results.append(record)

        print(
            f"[{idx+1:02d}/30] {item['id']:<15} | Cat: {item['category']:<11} | "
            f"CER: {cer:6.2%} | WER: {wer:6.2%} | Time: {ocr_res.inference_time:.3f}s | "
            f"Crit: {sum(critical_eval.values())}/{len(critical_eval)}"
        )

    mem_after_mb = process.memory_info().rss / (1024 * 1024)
    cpu_percent_end = psutil.cpu_percent(interval=None)

    # Aggregate Metrics by Category
    categories = ["printed", "handwritten", "degraded"]
    category_summary: dict[str, Any] = {}

    for cat in categories:
        cat_records = [r for r in detailed_results if r["category"] == cat]
        times = [r["inference_time_sec"] for r in cat_records]
        cers = [r["cer"] for r in cat_records]
        wers = [r["wer"] for r in cat_records]
        crit_accs = [r["critical_accuracy"] for r in cat_records]

        # Specific field accuracy
        field_accuracies: dict[str, float] = {}
        for field_name in ["medicine_name", "dosage", "frequency", "duration", "numeric_instruction"]:
            matches = [r["critical_matches"].get(field_name, False) for r in cat_records]
            field_accuracies[field_name] = round(sum(matches) / len(matches), 4) if matches else 0.0

        category_summary[cat] = {
            "count": len(cat_records),
            "mean_cer": round(statistics.mean(cers), 4),
            "median_cer": round(statistics.median(cers), 4),
            "mean_wer": round(statistics.mean(wers), 4),
            "median_wer": round(statistics.median(wers), 4),
            "mean_inference_time": round(statistics.mean(times), 4),
            "median_inference_time": round(statistics.median(times), 4),
            "mean_critical_accuracy": round(statistics.mean(crit_accs), 4),
            "field_accuracies": field_accuracies,
        }

    all_times = [r["inference_time_sec"] for r in detailed_results]
    all_cers = [r["cer"] for r in detailed_results]
    all_wers = [r["wer"] for r in detailed_results]
    all_crit_accs = [r["critical_accuracy"] for r in detailed_results]

    overall_field_accuracies: dict[str, float] = {}
    for field_name in ["medicine_name", "dosage", "frequency", "duration", "numeric_instruction"]:
        matches = [r["critical_matches"].get(field_name, False) for r in detailed_results]
        overall_field_accuracies[field_name] = round(sum(matches) / len(matches), 4)

    overall_summary = {
        "suite_name": output_prefix,
        "total_images": len(detailed_results),
        "engine_initialization_time_sec": round(init_duration, 4),
        "cold_inference_time_sec": round(all_times[0], 4) if all_times else 0.0,
        "warm_mean_inference_time_sec": round(statistics.mean(all_times[1:]), 4) if len(all_times) > 1 else 0.0,
        "overall_mean_inference_time_sec": round(statistics.mean(all_times), 4),
        "overall_median_inference_time_sec": round(statistics.median(all_times), 4),
        "overall_min_inference_time_sec": round(min(all_times), 4),
        "overall_max_inference_time_sec": round(max(all_times), 4),
        "overall_mean_cer": round(statistics.mean(all_cers), 4),
        "overall_median_cer": round(statistics.median(all_cers), 4),
        "overall_mean_wer": round(statistics.mean(all_wers), 4),
        "overall_median_wer": round(statistics.median(all_wers), 4),
        "overall_mean_critical_accuracy": round(statistics.mean(all_crit_accs), 4),
        "field_accuracies": overall_field_accuracies,
        "memory_rss_before_mb": round(mem_before_mb, 2),
        "memory_rss_after_mb": round(mem_after_mb, 2),
        "memory_rss_growth_mb": round(mem_after_mb - mem_before_mb, 2),
        "cpu_percent_estimate": cpu_percent_end,
        "category_summary": category_summary,
    }

    # Save JSON files
    json_path = RESULTS_DIR / f"{output_prefix}_results.json"
    summary_path = RESULTS_DIR / f"{output_prefix}_summary.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, indent=2)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(overall_summary, f, indent=2)

    # Save CSV report
    csv_path = RESULTS_DIR / f"{output_prefix}_results.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "id",
            "category",
            "degradation_type",
            "inference_time_sec",
            "cer",
            "wer",
            "critical_accuracy",
            "medicine_match",
            "dosage_match",
            "frequency_match",
            "duration_match",
            "instruction_match",
            "avg_confidence",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in detailed_results:
            crit = r["critical_matches"]
            writer.writerow(
                {
                    "id": r["id"],
                    "category": r["category"],
                    "degradation_type": r["degradation_type"],
                    "inference_time_sec": r["inference_time_sec"],
                    "cer": r["cer"],
                    "wer": r["wer"],
                    "critical_accuracy": r["critical_accuracy"],
                    "medicine_match": crit.get("medicine_name", False),
                    "dosage_match": crit.get("dosage", False),
                    "frequency_match": crit.get("frequency", False),
                    "duration_match": crit.get("duration", False),
                    "instruction_match": crit.get("numeric_instruction", False),
                    "avg_confidence": r["avg_confidence"],
                }
            )

    print("\n================================================================================")
    print("EVALUATION SUMMARY")
    print("================================================================================")
    for cat, data in category_summary.items():
        print(
            f"Category: {cat.upper():<12} | Mean CER: {data['mean_cer']:6.2%} | "
            f"Mean WER: {data['mean_wer']:6.2%} | Mean Crit Acc: {data['mean_critical_accuracy']:6.2%} | "
            f"Avg Time: {data['mean_inference_time']:.3f}s"
        )
    print("--------------------------------------------------------------------------------")
    print(
        f"OVERALL METRICS      | Mean CER: {overall_summary['overall_mean_cer']:6.2%} | "
        f"Mean WER: {overall_summary['overall_mean_wer']:6.2%} | "
        f"Critical Exact Match: {overall_summary['overall_mean_critical_accuracy']:6.2%}"
    )
    print(
        f"Inference Latency    | Cold: {overall_summary['cold_inference_time_sec']:.3f}s | "
        f"Warm Mean: {overall_summary['warm_mean_inference_time_sec']:.3f}s | "
        f"Median: {overall_summary['overall_median_inference_time_sec']:.3f}s"
    )
    print(f"Memory RSS           | Growth: {overall_summary['memory_rss_growth_mb']:.2f} MB")
    print(f"Reports saved to {RESULTS_DIR}")
    print("================================================================================")

    return overall_summary


if __name__ == "__main__":
    run_experiment(apply_preprocessing=False, output_prefix="baseline")
