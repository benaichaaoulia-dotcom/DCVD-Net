"""Per-image vessel metrics and the pre-declared R4 axis calculations."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score
from torch import Tensor

THRESHOLD = 0.6
MIN_THIN_PIXELS = 64
R4_BARS = {
    "F1_mean": 0.68,
    "F1_worst": 0.55,
    "F2_max_sensitivity_spread": 0.09,
    "F3_drive_chase_dsc_gap": 0.01,
    "F4_worst_image_dsc": 0.69,
    "F4_max_sub_0_75_per_dataset": 1,
    "F5_min_dataset_sensitivity": 0.77,
}


def _to_numpy(array: Tensor | np.ndarray) -> np.ndarray:
    if isinstance(array, Tensor):
        array = array.detach().cpu().numpy()
    return np.asarray(array).squeeze()


def image_metrics(
    probability: Tensor | np.ndarray,
    reference: Tensor | np.ndarray,
    threshold: float = THRESHOLD,
) -> dict[str, float | int | None]:
    """Compute AUC, thresholded confusion metrics, DSC, and eroded-thin sensitivity."""
    prob = _to_numpy(probability).astype(np.float64, copy=False)
    target = _to_numpy(reference) > 0.5
    if prob.shape != target.shape:
        raise ValueError(f"probability and reference shapes differ: {prob.shape} != {target.shape}")
    predicted = prob > threshold
    tp = int(np.count_nonzero(predicted & target))
    fp = int(np.count_nonzero(predicted & ~target))
    fn = int(np.count_nonzero(~predicted & target))
    tn = int(np.count_nonzero(~predicted & ~target))
    dsc = 2.0 * tp / (2.0 * tp + fp + fn + 1e-8)
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    auc = 0.5 if np.unique(target).size < 2 else float(roc_auc_score(target.ravel(), prob.ravel()))
    precision = tp / (tp + fp + 1e-8)
    f1 = 2.0 * precision * sensitivity / (precision + sensitivity + 1e-8)

    target_u8 = target.astype(np.uint8)
    eroded = cv2.erode(target_u8, np.ones((3, 3), dtype=np.uint8), iterations=1).astype(bool)
    thin = target & ~eroded
    thin_count = int(np.count_nonzero(thin))
    thin_sensitivity = (
        float(np.count_nonzero(predicted & thin) / thin_count)
        if thin_count >= MIN_THIN_PIXELS
        else None
    )
    return {
        "AUC": auc,
        "DSC": float(dsc),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "F1": float(f1),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "thin_pixels": thin_count,
        "thin_sensitivity": thin_sensitivity,
    }


def summarize_by_dataset(
    records_by_dataset: dict[str, Sequence[dict[str, float | int | None]]],
) -> dict[str, dict[str, float | int]]:
    """Macro-average per-image metrics, retaining the number of evaluated images."""
    summary: dict[str, dict[str, float | int]] = {}
    for dataset, records in records_by_dataset.items():
        if not records:
            raise ValueError(f"no per-image records for {dataset}")
        values: dict[str, float | int] = {"n_images": len(records)}
        for key in ("AUC", "DSC", "sensitivity", "specificity"):
            values[f"macro_{key.lower()}"] = float(
                np.mean([float(record[key]) for record in records])
            )
        thin_values = [
            float(record["thin_sensitivity"])
            for record in records
            if record["thin_sensitivity"] is not None
        ]
        values["thin_mean"] = float(np.mean(thin_values)) if thin_values else float("nan")
        values["thin_worst"] = float(np.min(thin_values)) if thin_values else float("nan")
        values["sub_0_75_count"] = int(sum(float(record["DSC"]) < 0.75 for record in records))
        summary[dataset] = values
    return summary


def r4_axes(
    summary: dict[str, dict[str, float | int]],
    per_image_records: Sequence[dict[str, object]],
) -> dict[str, float | int | bool | dict[str, int]]:
    """Calculate all R4 axes; F1 is decisive and F2–F5 remain secondary diagnostics."""
    required = {"DRIVE", "STARE", "CHASE_DB1"}
    if set(summary) != required:
        raise ValueError(f"R4 evaluation requires datasets {sorted(required)}")
    sensitivities = [float(summary[name]["macro_sensitivity"]) for name in sorted(required)]
    drive_thin_mean = float(summary["DRIVE"]["thin_mean"])
    drive_thin_worst = float(summary["DRIVE"]["thin_worst"])
    f1_pass = drive_thin_mean >= R4_BARS["F1_mean"] and drive_thin_worst >= R4_BARS["F1_worst"]
    sensitivity_spread = max(sensitivities) - min(sensitivities)
    dsc_gap = abs(float(summary["DRIVE"]["macro_dsc"]) - float(summary["CHASE_DB1"]["macro_dsc"]))
    worst_dsc = min(float(record["DSC"]) for record in per_image_records)
    sub_counts = {name: int(summary[name]["sub_0_75_count"]) for name in sorted(required)}
    max_sub_count = max(sub_counts.values())
    min_sensitivity = min(sensitivities)
    return {
        "F1_DRIVE_thin_mean": drive_thin_mean,
        "F1_DRIVE_thin_worst": drive_thin_worst,
        "F1_pass": f1_pass,
        "F2_max_sensitivity_spread": sensitivity_spread,
        "F2_pass": sensitivity_spread <= R4_BARS["F2_max_sensitivity_spread"],
        "F3_abs_DRIVE_CHASE_DB1_DSC_gap": dsc_gap,
        "F3_pass": dsc_gap <= R4_BARS["F3_drive_chase_dsc_gap"],
        "F4_worst_image_DSC": worst_dsc,
        "F4_worst_image_below_0_75_count": sum(
            float(record["DSC"]) < 0.75 for record in per_image_records
        ),
        "F4_total_sub_0_75_count": sum(sub_counts.values()),
        "F4_sub_0_75_by_dataset": sub_counts,
        "F4_max_sub_0_75_per_dataset": max_sub_count,
        "F4_pass": (
            worst_dsc >= R4_BARS["F4_worst_image_dsc"]
            and max_sub_count <= R4_BARS["F4_max_sub_0_75_per_dataset"]
        ),
        "F5_min_dataset_macro_sensitivity": min_sensitivity,
        "F5_pass": min_sensitivity >= R4_BARS["F5_min_dataset_sensitivity"],
        "primary_axis": "F1",
        "candidate_pass": f1_pass,
    }
