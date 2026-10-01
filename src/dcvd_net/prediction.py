"""Single-image deployment inference with explicit source-dataset conditioning."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from dcvd_net.data import preprocess_image_path
from dcvd_net.model import (
    PARAMETER_CAP,
    SOURCE_TO_INDEX,
    build_model,
    compatible_state_dict,
    count_parameters,
)


def predict(
    checkpoint_path: str | Path,
    arm: str,
    image_path: str | Path,
    source: str,
    output_path: str | Path,
    device_name: str = "cuda:0",
    threshold: float = 0.6,
) -> dict[str, Any]:
    """Predict a resized probability/binary mask for one input fundus photograph."""
    if source not in SOURCE_TO_INDEX:
        raise ValueError(f"source must be one of {tuple(SOURCE_TO_INDEX)}, got {source!r}")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    checkpoint_file = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
    if checkpoint.get("arm") != arm:
        raise ValueError(f"checkpoint arm {checkpoint.get('arm')!r} != requested {arm!r}")
    if torch.cuda.is_available() and device_name.startswith("cuda"):
        index = torch.device(device_name).index or 0
        if index >= torch.cuda.device_count():
            raise ValueError(
                f"requested {device_name}, but only {torch.cuda.device_count()} "
                "CUDA device(s) are visible"
            )
        torch.cuda.set_device(index)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model = build_model(arm, device)
    if (
        checkpoint.get("params") != count_parameters(model)
        or count_parameters(model) > PARAMETER_CAP
    ):
        raise ValueError("checkpoint/model parameter contract does not match")
    model.load_state_dict(compatible_state_dict(checkpoint["model_state_dict"]), strict=True)
    model.eval()
    image_tensor = preprocess_image_path(image_path).unsqueeze(0).to(device)
    source_index = torch.tensor([SOURCE_TO_INDEX[source]], dtype=torch.long, device=device)
    with torch.inference_mode():
        probability = model(image_tensor, source_index)[0, 0].cpu().numpy()
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    probability_image = Image.fromarray(
        np.clip(np.rint(probability * 255), 0, 255).astype(np.uint8)
    )
    probability_image.save(output)
    binary_output = output.with_name(f"{output.stem}_threshold-{threshold:g}{output.suffix}")
    binary_image = Image.fromarray((probability >= threshold).astype(np.uint8) * 255)
    binary_image.save(binary_output)
    result = {
        "image": str(Path(image_path).expanduser().resolve()),
        "source": source,
        "source_index": SOURCE_TO_INDEX[source],
        "arm": arm,
        "threshold": threshold,
        "probability_output": str(output),
        "binary_output": str(binary_output),
        "shape": list(probability.shape),
    }
    print(result, flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run single-image DCVD-Net inference.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--arm", default="C", choices=("C", "T", "A1", "A2", "A3", "A4"))
    parser.add_argument("--image", required=True)
    parser.add_argument("--source", required=True, choices=tuple(SOURCE_TO_INDEX))
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threshold", type=float, default=0.6)
    arguments = parser.parse_args()
    predict(
        arguments.checkpoint,
        arguments.arm,
        arguments.image,
        arguments.source,
        arguments.output,
        device_name=arguments.device,
        threshold=arguments.threshold,
    )


if __name__ == "__main__":
    main()
