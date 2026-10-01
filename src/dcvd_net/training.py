"""Deterministic joint-protocol training for DCVD-Net, its twin, and registered ablations."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from dcvd_net.data import make_train_dev_loaders, verify_split_integrity
from dcvd_net.losses import bce_dice_loss
from dcvd_net.metrics import THRESHOLD, image_metrics
from dcvd_net.model import PARAMETER_CAP, build_model, count_parameters


@dataclass(frozen=True)
class TrainConfig:
    """All run-specific training values, loaded from a YAML file."""

    arm: str
    data_root: str
    out_dir: str
    checkpoint_name: str
    last_checkpoint_name: str
    log_name: str
    seed: int = 42
    batch_size: int = 4
    max_epochs: int = 300
    patience: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-3
    optimizer: str = "adam"
    num_workers: int = 0
    gpu: int = 1
    budget_gpu_hours: float = 1.0
    param_cap: int = PARAMETER_CAP
    threshold: float = THRESHOLD
    image_size: int = 512

    def __post_init__(self) -> None:
        if self.arm not in {"C", "T", "A1", "A2", "A3", "A4"}:
            raise ValueError(f"unsupported arm: {self.arm}")
        if self.optimizer.lower() != "adam":
            raise ValueError("the recorded protocol uses Adam")
        if self.num_workers != 0:
            raise ValueError("the deterministic recorded protocol uses num_workers=0")
        if self.threshold != THRESHOLD:
            raise ValueError(f"locked evaluation threshold must remain {THRESHOLD}")
        if not 500_000 <= self.param_cap <= PARAMETER_CAP:
            raise ValueError("param_cap must be in the declared 0.5M–1M band")
        if self.batch_size < 1 or self.max_epochs < 1 or self.patience < 1:
            raise ValueError("batch_size, max_epochs, and patience must be positive")
        if self.lr <= 0 or self.weight_decay < 0 or self.budget_gpu_hours <= 0:
            raise ValueError(
                "learning rate and budget must be positive; "
                "weight decay cannot be negative"
            )
        if self.gpu < 0 or self.image_size < 16 or self.image_size % 16 != 0:
            raise ValueError(
                "gpu must be nonnegative and image_size must be a positive multiple of 16"
            )

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        """Load a strict config; unknown YAML keys are rejected rather than silently ignored."""
        with Path(path).open(encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
        if not isinstance(raw, dict):
            raise TypeError("training YAML must contain a mapping at its root")
        valid_fields = {field.name for field in fields(cls)}
        unknown = set(raw) - valid_fields
        if unknown:
            raise ValueError(f"unknown training config keys: {sorted(unknown)}")
        return cls(**raw)


def set_determinism(seed: int) -> None:
    """Set Python, NumPy, PyTorch, and cuDNN RNG/determinism settings."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def capture_rng_state(train_generator: torch.Generator) -> dict[str, Any]:
    """Capture safe-serializable RNG states for exact epoch-boundary resume."""
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [
            numpy_state[0],
            numpy_state[1].tolist(),
            int(numpy_state[2]),
            int(numpy_state[3]),
            float(numpy_state[4]),
        ],
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "train_generator": train_generator.get_state(),
    }


def restore_rng_state(state: dict[str, Any], train_generator: torch.Generator) -> None:
    """Restore captured RNG states after model/optimizer construction."""
    random.setstate(tuple(state["python"]))
    numpy_state = state["numpy"]
    np.random.set_state(
        (numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), int(numpy_state[2]),
         int(numpy_state[3]), float(numpy_state[4]))
    )
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    train_generator.set_state(state["train_generator"])


def resolve_device(gpu: int) -> torch.device:
    """Select an explicit CUDA device; training deliberately has no silent CPU fallback."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training; CPU fallback is disabled")
    if gpu < 0 or gpu >= torch.cuda.device_count():
        raise ValueError(
            f"requested GPU {gpu}, but only {torch.cuda.device_count()} "
            "CUDA device(s) are visible"
        )
    torch.cuda.set_device(gpu)
    device = torch.device(f"cuda:{gpu}")
    print(f"DEVICE: {device} | GPU: {torch.cuda.get_device_name(gpu)}", flush=True)
    return device


def _checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    epoch: int,
    best_epoch: int,
    best_dev_dsc: float,
    best_dev_loss: float,
    current_dev_loss: float,
    epochs_without_dev_loss_improvement: int,
    train_generator: torch.Generator,
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": None,
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_dev_dsc": best_dev_dsc,
        "best_dev_loss": best_dev_loss,
        "dev_loss": current_dev_loss,
        "epochs_without_dev_loss_improvement": epochs_without_dev_loss_improvement,
        "arm": config.arm,
        "seed": config.seed,
        "params": count_parameters(model),
        "config": asdict(config),
        "rng_state": capture_rng_state(train_generator),
    }


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    path: str | Path,
    expected_arm: str,
    train_generator: torch.Generator,
) -> dict[str, Any]:
    """Strictly restore a checkpoint and reject silent cross-arm loads."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("arm") != expected_arm:
        raise ValueError(f"checkpoint arm {checkpoint.get('arm')!r} != expected {expected_arm!r}")
    if checkpoint.get("params") != count_parameters(model):
        raise ValueError("checkpoint parameter count does not match this model")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "rng_state" in checkpoint:
        restore_rng_state(checkpoint["rng_state"], train_generator)
    return checkpoint


def _evaluate_dev(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    dsc_values: list[float] = []
    loss_sum = 0.0
    image_count = 0
    with torch.no_grad():
        for images, masks, source_indices in loader:
            images = images.to(device)
            masks = masks.to(device)
            source_indices = source_indices.to(device)
            probabilities = model(images, source_indices)
            loss_sum += float(bce_dice_loss(probabilities, masks).item()) * images.shape[0]
            for index in range(images.shape[0]):
                metrics = image_metrics(probabilities[index], masks[index], threshold=THRESHOLD)
                dsc_values.append(float(metrics["DSC"]))
            image_count += images.shape[0]
    if image_count == 0:
        raise RuntimeError("development loader was empty")
    return float(np.mean(dsc_values)), loss_sum / image_count


def _write_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()


def train(
    config: TrainConfig,
    resume: str | Path | None = None,
    gpu_override: int | None = None,
) -> dict[str, Any]:
    """Train one declared arm using train/dev only; this function never creates test loaders."""
    if gpu_override is not None:
        config = replace(config, gpu=gpu_override)
    set_determinism(config.seed)
    device = resolve_device(config.gpu)

    data_root = Path(config.data_root).expanduser().resolve()
    output_dir = Path(config.out_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / config.log_name
    best_checkpoint_path = output_dir / config.checkpoint_name
    last_checkpoint_path = output_dir / config.last_checkpoint_name
    summary_path = output_dir / "summary.json"
    if resume is None:
        existing_outputs = [
            path for path in (log_path, best_checkpoint_path, last_checkpoint_path, summary_path)
            if path.exists()
        ]
        if existing_outputs:
            raise FileExistsError(
                "run outputs already exist; choose a new out_dir or pass an explicit --resume: "
                + ", ".join(str(path) for path in existing_outputs)
            )

    train_generator = torch.Generator().manual_seed(config.seed)
    train_loader, dev_loader = make_train_dev_loaders(
        data_root,
        config.batch_size,
        config.seed,
        image_size=config.image_size,
        generator=train_generator,
    )
    split_report = verify_split_integrity()
    model = build_model(config.arm, device)
    total_parameters = count_parameters(model)
    trainable_parameters = count_parameters(model, trainable_only=True)
    if total_parameters > config.param_cap:
        raise RuntimeError(
            f"model has {total_parameters:,} parameters; cap is {config.param_cap:,}"
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    start_epoch = 0
    best_epoch = 0
    best_dev_dsc = -1.0
    best_dev_loss = float("inf")
    epochs_without_improvement = 0
    if resume:
        checkpoint = load_checkpoint(model, optimizer, resume, config.arm, train_generator)
        saved_config = checkpoint.get("config")
        if not isinstance(saved_config, dict) or "rng_state" not in checkpoint:
            raise ValueError("resume requires a checkpoint created by this package's training loop")
        mutable_resume_fields = {
            "out_dir", "checkpoint_name", "last_checkpoint_name", "log_name", "gpu",
            "budget_gpu_hours", "max_epochs",
        }
        mismatches = {
            field.name: (saved_config.get(field.name), getattr(config, field.name))
            for field in fields(config)
            if field.name not in mutable_resume_fields
            and saved_config.get(field.name) != getattr(config, field.name)
        }
        if mismatches:
            raise ValueError(f"resume config differs from checkpoint: {mismatches}")
        start_epoch = int(checkpoint["epoch"])
        best_epoch = int(checkpoint["best_epoch"])
        best_dev_dsc = float(checkpoint["best_dev_dsc"])
        best_dev_loss = float(checkpoint["best_dev_loss"])
        epochs_without_improvement = int(checkpoint["epochs_without_dev_loss_improvement"])
        print(f"Explicit resume from epoch {start_epoch}: {resume}", flush=True)

    if start_epoch >= config.max_epochs:
        raise ValueError(
            f"checkpoint epoch {start_epoch} is at/above max_epochs={config.max_epochs}"
        )

    if resume is None:
        log_path.write_text("", encoding="utf-8")
    else:
        log_path.touch(exist_ok=True)
    _write_jsonl(log_path, {
        "event": "start",
        "config": asdict(config),
        "device": str(device),
        "parameter_count": total_parameters,
        "trainable_parameters": trainable_parameters,
        "split_integrity": split_report,
        "train_images": len(train_loader.dataset),
        "dev_images": len(dev_loader.dataset),
        "test_data_opened": False,
        "resumed_from_epoch": start_epoch,
    })
    global_update = start_epoch * len(train_loader)
    start_time = time.perf_counter()
    peak_vram_mib = 0.0
    last_epoch = start_epoch

    for epoch_index in range(start_epoch, config.max_epochs):
        model.train()
        weighted_loss = 0.0
        n_seen = 0
        for images, masks, source_indices in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            source_indices = source_indices.to(device)
            optimizer.zero_grad(set_to_none=True)
            probabilities = model(images, source_indices)
            loss = bce_dice_loss(probabilities, masks)
            loss.backward()
            optimizer.step()
            weighted_loss += float(loss.detach().item()) * images.shape[0]
            n_seen += images.shape[0]
            global_update += 1
        if n_seen == 0:
            raise RuntimeError("training loader was empty")
        train_loss = weighted_loss / n_seen
        dev_dsc, dev_loss = _evaluate_dev(model, dev_loader, device)
        epoch = epoch_index + 1
        last_epoch = epoch
        if torch.cuda.is_available():
            peak_vram_mib = max(peak_vram_mib, torch.cuda.memory_allocated(device) / 2**20)

        if dev_loss < best_dev_loss - 1e-5:
            best_dev_loss = dev_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if dev_dsc > best_dev_dsc:
            best_dev_dsc = dev_dsc
            best_epoch = epoch
            payload = _checkpoint_payload(
                model, optimizer, config, epoch, best_epoch, best_dev_dsc, best_dev_loss,
                dev_loss, epochs_without_improvement, train_generator,
            )
            torch.save(payload, best_checkpoint_path)

        payload = _checkpoint_payload(
            model, optimizer, config, epoch, best_epoch, best_dev_dsc, best_dev_loss,
            dev_loss, epochs_without_improvement, train_generator,
        )
        torch.save(payload, last_checkpoint_path)
        record = {
            "epoch": epoch,
            "global_update": global_update,
            "train_loss": train_loss,
            "dev_loss": dev_loss,
            "dev_dsc": dev_dsc,
            "best_dev_dsc": best_dev_dsc,
            "best_epoch": best_epoch,
            "epochs_without_dev_loss_improvement": epochs_without_improvement,
        }
        _write_jsonl(log_path, record)
        if epoch <= 5 or epoch % 25 == 0:
            print(
                f"[{config.arm}] epoch={epoch}/{config.max_epochs} update={global_update} "
                f"train_loss={train_loss:.6f} dev_loss={dev_loss:.6f} dev_dsc={dev_dsc:.6f}",
                flush=True,
            )
        if epochs_without_improvement >= config.patience:
            print(
                f"[{config.arm}] early stop at epoch {epoch}; "
                f"best DSC={best_dev_dsc:.6f} at {best_epoch}",
                flush=True,
            )
            break

    wall_seconds = time.perf_counter() - start_time
    gpu_hours = wall_seconds / 3600.0
    summary: dict[str, Any] = {
        "arm": config.arm,
        "seed": config.seed,
        "device": str(device),
        "params": total_parameters,
        "trainable_params": trainable_parameters,
        "stopped_epoch": last_epoch,
        "final_global_update": global_update,
        "best_dev_dsc": best_dev_dsc,
        "best_epoch": best_epoch,
        "best_dev_loss": best_dev_loss,
        "peak_vram_mib": peak_vram_mib,
        "wall_seconds": wall_seconds,
        "gpu_hours_est": gpu_hours,
        "budget_gpu_hours": config.budget_gpu_hours,
        "within_budget": gpu_hours <= config.budget_gpu_hours,
        "best_checkpoint": str(best_checkpoint_path),
        "last_checkpoint": str(last_checkpoint_path),
        "test_data_opened": False,
    }
    _write_jsonl(log_path, {"event": "summary", "summary": summary})
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[{config.arm}] completed: {json.dumps(summary, sort_keys=True)}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DCVD-Net or a registered comparison arm.")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--gpu", type=int, default=None, help="explicit GPU index; overrides YAML")
    parser.add_argument("--resume", default=None, help="explicit checkpoint path; no auto-resume")
    arguments = parser.parse_args()
    config = TrainConfig.from_yaml(arguments.config)
    train(config, resume=arguments.resume, gpu_override=arguments.gpu)


if __name__ == "__main__":
    main()
