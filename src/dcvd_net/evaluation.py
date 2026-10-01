"""Explicit evaluation of locked official test splits; never imported by training."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from dcvd_net.data import JOINT_TEST, make_test_loaders, verify_split_integrity
from dcvd_net.metrics import THRESHOLD, image_metrics, r4_axes, summarize_by_dataset
from dcvd_net.model import PARAMETER_CAP, build_model, compatible_state_dict, count_parameters

PURPOSE = (
    "explicit locked official test evaluation; not for fitting, threshold, "
    "or checkpoint selection"
)


def evaluate(
    checkpoint_path: str | Path,
    arm: str,
    data_root: str | Path,
    output_path: str | Path,
    *,
    allow_test: bool,
    purpose: str = PURPOSE,
    batch_size: int = 4,
    device_name: str = "cuda:0",
) -> dict[str, Any]:
    """Evaluate test sets only after an explicit acknowledgment and record their use."""
    if not allow_test:
        raise PermissionError("test evaluation requires the explicit --allow-test acknowledgment")
    if not purpose.strip():
        raise ValueError("purpose must be a non-empty audit description")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the recorded test-evaluation workflow")
    if device_name.startswith("cuda"):
        index = torch.device(device_name).index or 0
        if index >= torch.cuda.device_count():
            raise ValueError(
                f"requested {device_name}, but only {torch.cuda.device_count()} "
                "CUDA device(s) are visible"
            )
        torch.cuda.set_device(index)
    device = torch.device(device_name)
    if device.type != "cuda":
        raise ValueError("test evaluation requires an explicit CUDA device")

    checkpoint_file = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
    if checkpoint.get("arm") != arm:
        raise ValueError(f"checkpoint arm {checkpoint.get('arm')!r} != requested {arm!r}")
    model = build_model(arm, device)
    if (
        checkpoint.get("params") != count_parameters(model)
        or count_parameters(model) > PARAMETER_CAP
    ):
        raise ValueError("checkpoint/model parameter contract does not match")
    model.load_state_dict(compatible_state_dict(checkpoint["model_state_dict"]), strict=True)
    model.eval()

    split_integrity = verify_split_integrity()
    expected_test_counts = {"DRIVE": 20, "STARE": 9, "CHASE_DB1": 12}
    observed_test_counts = {
        dataset: sum(1 for name, _ in JOINT_TEST if name == dataset)
        for dataset in expected_test_counts
    }
    if observed_test_counts != expected_test_counts or split_integrity["test"] != 41:
        raise RuntimeError(
            f"locked test manifest changed: {observed_test_counts}, {split_integrity}"
        )

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    usage_path = output.parent / "test_usage_log.json"
    if usage_path.exists():
        usage_log = json.loads(usage_path.read_text(encoding="utf-8"))
        if not isinstance(usage_log, list):
            raise TypeError(f"expected list in {usage_path}")
    else:
        usage_log = []
    timestamp_utc = datetime.now(UTC).isoformat()
    split_names = ["DRIVE official test", "STARE official test", "CHASE_DB1 official test"]
    usage_log.append({
        "timestamp_utc": timestamp_utc,
        "arm": arm,
        "seed": checkpoint.get("seed"),
        "checkpoint": str(checkpoint_file),
        "output": str(output),
        "purpose": purpose,
        "split_names": split_names,
        "allow_test_acknowledged": True,
    })
    usage_path.write_text(json.dumps(usage_log, indent=2), encoding="utf-8")

    # This is the first operation that opens any official test image or annotation.
    test_loaders = make_test_loaders(data_root, batch_size=batch_size)
    per_dataset: dict[str, list[dict[str, float | int | None]]] = {}
    per_image: list[dict[str, Any]] = []
    with torch.inference_mode():
        for dataset, (loader, image_ids) in test_loaders.items():
            records: list[dict[str, float | int | None]] = []
            image_offset = 0
            for images, masks, source_indices in loader:
                images = images.to(device)
                masks = masks.to(device)
                source_indices = source_indices.to(device)
                probabilities = model(images, source_indices)
                for batch_index in range(images.shape[0]):
                    metrics = image_metrics(
                        probabilities[batch_index],
                        masks[batch_index],
                        threshold=THRESHOLD,
                    )
                    image_id = image_ids[image_offset]
                    image_offset += 1
                    record = {"dataset": dataset, "image_id": image_id, **metrics}
                    records.append(metrics)
                    per_image.append(record)
            if image_offset != len(image_ids):
                raise RuntimeError(
                    f"evaluated {image_offset} {dataset} images; expected {len(image_ids)}"
                )
            per_dataset[dataset] = records
    summary = summarize_by_dataset(per_dataset)
    axes = r4_axes(summary, per_image)
    result = {
        "schema_version": 1,
        "arm": arm,
        "seed": checkpoint.get("seed"),
        "checkpoint": str(checkpoint_file),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "best_epoch": checkpoint.get("best_epoch"),
        "threshold": THRESHOLD,
        "device": str(device),
        "parameter_count": count_parameters(model),
        "datasets": summary,
        "r4_axes": axes,
        "primary_decision": "PASS" if axes["candidate_pass"] else "FAIL",
        "test_images": len(per_image),
        "per_image": per_image,
        "test_usage": {
            "purpose": purpose,
            "split_names": split_names,
            "allow_test_acknowledged": True,
            "timestamp_utc": timestamp_utc,
        },
    }
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(
        json.dumps(
            {"primary_decision": result["primary_decision"], "r4_axes": axes},
            indent=2,
        ),
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explicitly evaluate DCVD-Net on its locked official test sets."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--arm", required=True, choices=("C", "T", "A1", "A2", "A3", "A4"))
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="acknowledge that locked test data will be evaluated",
    )
    parser.add_argument("--purpose", default=PURPOSE)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    arguments = parser.parse_args()
    evaluate(
        arguments.checkpoint,
        arguments.arm,
        arguments.data_root,
        arguments.output,
        allow_test=arguments.allow_test,
        purpose=arguments.purpose,
        batch_size=arguments.batch_size,
        device_name=arguments.device,
    )


if __name__ == "__main__":
    main()
