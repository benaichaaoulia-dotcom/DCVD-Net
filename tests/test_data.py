from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch.utils.data import Dataset

import dcvd_net.data as data


def test_locked_train_dev_test_manifest_is_disjoint_without_opening_files() -> None:
    report = data.verify_split_integrity()
    assert report == {
        "train": 46,
        "dev": 6,
        "train_active": 40,
        "test": 41,
        "train_test_overlap": 0,
        "dev_test_overlap": 0,
        "dev_subset_of_train": True,
    }
    assert not set(data.TRAIN_ACTIVE) & set(data.JOINT_DEV)
    assert not set(data.JOINT_TRAIN) & set(data.JOINT_TEST)


def test_training_loader_constructs_only_train_and_dev(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    created: list[tuple[tuple[str, str], ...]] = []

    class ManifestOnlyDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
        def __init__(
            self,
            data_root: str | Path,
            pairs: Sequence[data.Pair],
            augment: bool = False,
            size: int = data.IMAGE_SIZE,
        ) -> None:
            del data_root, augment, size
            self.pairs = tuple(pairs)
            created.append(self.pairs)

        def __len__(self) -> int:
            return len(self.pairs)

        def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return torch.zeros(1, 16, 16), torch.zeros(1, 16, 16), torch.tensor(0)

    def forbidden_test_loader(*args: object, **kwargs: object) -> None:
        raise AssertionError("training must never construct a test loader")

    monkeypatch.setattr(data, "JointVesselDataset", ManifestOnlyDataset)
    monkeypatch.setattr(data, "make_test_loaders", forbidden_test_loader)

    train_loader, dev_loader = data.make_train_dev_loaders(tmp_path, batch_size=4, seed=42)

    assert len(train_loader.dataset) == 40
    assert len(dev_loader.dataset) == 6
    assert created == [data.TRAIN_ACTIVE, data.JOINT_DEV]
    constructed_pairs = {pair for pairs in created for pair in pairs}
    assert constructed_pairs == set(data.TRAIN_ACTIVE) | set(data.JOINT_DEV)
    assert not constructed_pairs & set(data.JOINT_TEST)


def test_dataset_paths_use_the_declared_official_annotations(tmp_path: Path) -> None:
    images, masks = data.dataset_paths(tmp_path, "DRIVE", ["21_training", "01_test"])
    assert images[0] == tmp_path / "DRIVE_official/training/images/21_training.tif"
    assert masks[0] == tmp_path / "DRIVE_official/training/1st_manual/21_manual1.gif"
    assert images[1] == tmp_path / "DRIVE_official/test/images/01_test.tif"
    assert masks[1] == tmp_path / "DRIVE_official/test/1st_manual/01_manual1.gif"


def test_preprocessing_returns_normalized_single_channel_tensors() -> None:
    image = Image.new("RGB", (64, 64), color=(220, 128, 35))
    mask = Image.new("L", (4, 4), color=0)
    mask.putpixel((1, 1), 255)

    image_tensor = data.preprocess_fundus(image, size=32)
    mask_tensor = data.preprocess_mask(mask, size=32)

    assert image_tensor.shape == (1, 32, 32)
    assert mask_tensor.shape == (1, 32, 32)
    assert image_tensor.dtype == torch.float32
    assert mask_tensor.dtype == torch.float32
    assert torch.isfinite(image_tensor).all()
    assert torch.all((image_tensor >= 0.0) & (image_tensor <= 1.0))
    assert set(mask_tensor.unique().tolist()) == {0.0, 1.0}
