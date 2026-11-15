"""Benchmark and compare preprocessing techniques against baseline OCR performance.

Evaluates:
- Baseline (no preprocessing)
- Grayscale conversion
- Contrast enhancement (CLAHE)
- Denoising (fastNlMeans)
- Deskewing (minAreaRect)
- Resizing (1.5x upscaling)

Saves comparison report to docs/results/preprocessing_comparison.json and .csv.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from scripts.run_ocr_experiment import run_experiment

ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "docs" / "results"

PREPROCESSING_EXPERIMENTS: list[dict[str, Any]] = [
    {
        "name": "grayscale",
        "description": "Convert to grayscale and normalize 3-channel format",
        "kwargs": {"to_grayscale": True},
    },
    {
        "name": "contrast_clahe",
        "description": "Adaptive histogram equalization (CLAHE)",
        "kwargs": {"adjust_contrast": True},
    },
    {
        "name": "denoise",
        "description": "Non-local means denoising filter",
        "kwargs": {"denoise": True},
    },
    {
        "name": "deskew",
        "description": "Contour orientation detection and affine deskew",
        "kwargs": {"deskew": True},
    },
    {
        "name": "resize_1.5x",
        "description": "Bicubic upsampling 1.5x scaling",
        "kwargs": {"resize_scale": 1.5},
    },
]


def run_all_benchmarks():
    """Run all preprocessing benchmarks and compile a comparison table against baseline."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load baseline summary if already executed, or run baseline
    baseline_summary_path = RESULTS_DIR / "baseline_summary.json"
    if baseline_summary_path.exists():
        with open(baseline_summary_path, "r", encoding="utf-8") as f:
            baseline_summary = json.load(f)
    else:
        baseline_summary = run_experiment(apply_preprocessing=False, output_prefix="baseline")

    comparison_records: list[dict[str, Any]] = [
        {
            "technique": "Baseline (None)",
            "description": "Raw original source images without preprocessing",
            "overall_cer": baseline_summary["overall_mean_cer"],
            "overall_wer": baseline_summary["overall_mean_wer"],
            "critical_accuracy": baseline_summary["overall_mean_critical_accuracy"],
            "mean_time_sec": baseline_summary["overall_mean_inference_time_sec"],
            "degraded_cer": baseline_summary["category_summary"]["degraded"]["mean_cer"],
            "degraded_wer": baseline_summary["category_summary"]["degraded"]["mean_wer"],
        }
    ]

    for exp in PREPROCESSING_EXPERIMENTS:
        prefix = f"preproc_{exp['name']}"
        print(f"\nEvaluating Preprocessing: {exp['name']} ({exp['description']})...")
        summary = run_experiment(
            apply_preprocessing=True,
            preprocessing_kwargs=exp["kwargs"],
            output_prefix=prefix,
        )

        comparison_records.append(
            {
                "technique": exp["name"],
                "description": exp["description"],
                "overall_cer": summary["overall_mean_cer"],
                "overall_wer": summary["overall_mean_wer"],
                "critical_accuracy": summary["overall_mean_critical_accuracy"],
                "mean_time_sec": summary["overall_mean_inference_time_sec"],
                "degraded_cer": summary["category_summary"]["degraded"]["mean_cer"],
                "degraded_wer": summary["category_summary"]["degraded"]["mean_wer"],
            }
        )

    # Save comparison files
    comp_json = RESULTS_DIR / "preprocessing_comparison.json"
    with open(comp_json, "w", encoding="utf-8") as f:
        json.dump(comparison_records, f, indent=2)

    comp_csv = RESULTS_DIR / "preprocessing_comparison.csv"
    with open(comp_csv, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "technique",
            "description",
            "overall_cer",
            "overall_wer",
            "critical_accuracy",
            "mean_time_sec",
            "degraded_cer",
            "degraded_wer",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparison_records)

    print("\n================================================================================")
    print("PREPROCESSING COMPARISON TABLE")
    print("================================================================================")
    print(
        f"{'Technique':<18} | {'Overall CER':<11} | {'Overall WER':<11} | "
        f"{'Crit Acc':<10} | {'Degraded CER':<12} | {'Avg Time':<8}"
    )
    print("-" * 80)
    for rec in comparison_records:
        print(
            f"{rec['technique']:<18} | {rec['overall_cer']:11.2%} | {rec['overall_wer']:11.2%} | "
            f"{rec['critical_accuracy']:10.2%} | {rec['degraded_cer']:12.2%} | {rec['mean_time_sec']:.3f}s"
        )
    print("================================================================================")
    print(f"Comparison report saved to {comp_json} and {comp_csv}")


if __name__ == "__main__":
    run_all_benchmarks()
