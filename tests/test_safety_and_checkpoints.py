from __future__ import annotations

from pathlib import Path

import pytest
import torch

from dcvd_net.evaluation import evaluate
from dcvd_net.model import build_model
from dcvd_net.training import TrainConfig, _checkpoint_payload, load_checkpoint


def test_test_evaluation_requires_acknowledgment_before_any_other_work(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="--allow-test"):
        evaluate(
            tmp_path / "missing.pth",
            "C",
            tmp_path / "data",
            tmp_path / "result.json",
            allow_test=False,
        )


def test_checkpoint_restore_is_strict_and_arm_checked(tmp_path: Path) -> None:
    source = build_model("C")
    config = TrainConfig(
        arm="C",
        data_root=str(tmp_path / "data"),
        out_dir=str(tmp_path / "run"),
        checkpoint_name="best_C.pth",
        last_checkpoint_name="last_C.pth",
        log_name="train_C.jsonl",
    )
    optimizer = torch.optim.Adam(
        source.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    sampler_generator = torch.Generator().manual_seed(42)
    checkpoint_path = tmp_path / "checkpoint.pth"
    checkpoint = _checkpoint_payload(
        source,
        optimizer,
        config,
        epoch=1,
        best_epoch=1,
        best_dev_dsc=0.7,
        best_dev_loss=1.2,
        current_dev_loss=1.2,
        epochs_without_dev_loss_improvement=0,
        train_generator=sampler_generator,
    )
    torch.save(checkpoint, checkpoint_path)

    restored = build_model("C")
    metadata = load_checkpoint(
        restored,
        optimizer=None,
        path=checkpoint_path,
        expected_arm="C",
        train_generator=torch.Generator().manual_seed(42),
    )
    assert metadata["arm"] == "C"
    assert metadata["config"]["arm"] == "C"
    assert "rng_state" in metadata
    for source_parameter, restored_parameter in zip(
        source.parameters(),
        restored.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(source_parameter, restored_parameter, rtol=0.0, atol=0.0)

    with pytest.raises(ValueError, match="expected 'T'"):
        load_checkpoint(
            build_model("T"),
            optimizer=None,
            path=checkpoint_path,
            expected_arm="T",
            train_generator=torch.Generator().manual_seed(42),
        )
