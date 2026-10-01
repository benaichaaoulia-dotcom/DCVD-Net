"""Dataset identities, split manifest, preprocessing, and loaders for the three retinal datasets."""

from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from dcvd_net.model import SOURCE_NAMES, SOURCE_TO_INDEX

type Pair = tuple[str, str]

DRIVE_TRAIN_IDS = tuple(f"{index}_training" for index in range(21, 41))
STARE_TRAIN_IDS = (
    "im0001", "im0002", "im0003", "im0004", "im0044",
    "im0077", "im0139", "im0162", "im0235", "im0239",
)
CHASE_TRAIN_SUBJECTS = tuple(f"{index:02d}" for index in range(1, 9))

JOINT_TRAIN: tuple[Pair, ...] = (
    *(("DRIVE", image_id) for image_id in DRIVE_TRAIN_IDS),
    *(("STARE", image_id) for image_id in STARE_TRAIN_IDS),
    *(("CHASE_DB1", f"Image_{subject}{eye}") for subject in CHASE_TRAIN_SUBJECTS for eye in "LR"),
)
JOINT_DEV: tuple[Pair, ...] = (
    ("DRIVE", "39_training"),
    ("DRIVE", "40_training"),
    ("STARE", "im0235"),
    ("STARE", "im0239"),
    ("CHASE_DB1", "Image_08L"),
    ("CHASE_DB1", "Image_08R"),
)
DRIVE_TEST_IDS = tuple(f"{index:02d}_test" for index in range(1, 21))
STARE_TEST_IDS = (
    "im0081", "im0082", "im0163", "im0236", "im0240",
    "im0255", "im0291", "im0319", "im0324",
)
CHASE_TEST_SUBJECTS = tuple(f"{index:02d}" for index in range(9, 15))
JOINT_TEST: tuple[Pair, ...] = (
    *(("DRIVE", image_id) for image_id in DRIVE_TEST_IDS),
    *(("STARE", image_id) for image_id in STARE_TEST_IDS),
    *(("CHASE_DB1", f"Image_{subject}{eye}") for subject in CHASE_TEST_SUBJECTS for eye in "LR"),
)
_DEV_ID_SET = set(JOINT_DEV)
TRAIN_ACTIVE = tuple(pair for pair in JOINT_TRAIN if pair not in _DEV_ID_SET)
IMAGE_SIZE = 512
ROTATION_ANGLES = (0, 15, 30, 45, 90, 100, 120)
ROTATION_PROBABILITY = 1.0


def verify_split_integrity() -> dict[str, int | bool]:
    """Check the adjudicated train/dev/test image-ID contract without opening image files."""
    train = set(JOINT_TRAIN)
    dev = set(JOINT_DEV)
    test = set(JOINT_TEST)
    if len(train) != len(JOINT_TRAIN) or len(test) != len(JOINT_TEST):
        raise AssertionError("split manifests must not contain duplicate image IDs")
    if not dev <= train:
        raise AssertionError("development IDs must be a subset of the declared training IDs")
    if train & test or dev & test:
        raise AssertionError("train/dev and test IDs must be disjoint")
    if len(JOINT_TRAIN) != 46 or len(JOINT_DEV) != 6 or len(TRAIN_ACTIVE) != 40:
        raise AssertionError("unexpected joint train/dev split cardinality")
    if len(JOINT_TEST) != 41:
        raise AssertionError("unexpected joint test split cardinality")
    train_counts = {
        name: sum(dataset == name for dataset, _ in JOINT_TRAIN)
        for name in SOURCE_NAMES
    }
    test_counts = {
        name: sum(dataset == name for dataset, _ in JOINT_TEST)
        for name in SOURCE_NAMES
    }
    if train_counts != {"DRIVE": 20, "STARE": 10, "CHASE_DB1": 16}:
        raise AssertionError(f"unexpected joint training source counts: {train_counts}")
    if test_counts != {"DRIVE": 20, "STARE": 9, "CHASE_DB1": 12}:
        raise AssertionError(f"unexpected locked test source counts: {test_counts}")
    return {
        "train": len(train),
        "dev": len(dev),
        "train_active": len(TRAIN_ACTIVE),
        "test": len(test),
        "train_test_overlap": len(train & test),
        "dev_test_overlap": len(dev & test),
        "dev_subset_of_train": dev <= train,
    }


def dataset_paths(
    data_root: str | Path,
    dataset: str,
    image_ids: Sequence[str],
) -> tuple[list[Path], list[Path]]:
    """Resolve image and reference-mask paths for the documented local data layout."""
    root = Path(data_root).expanduser().resolve()
    if dataset == "DRIVE":
        base = root / "DRIVE_official"
        image_paths = [
            base / ("training" if "training" in image_id else "test") / "images" / f"{image_id}.tif"
            for image_id in image_ids
        ]
        mask_paths = [
            base / ("training" if "training" in image_id else "test") / "1st_manual"
            / f"{image_id.replace('_training', '').replace('_test', '')}_manual1.gif"
            for image_id in image_ids
        ]
    elif dataset == "STARE":
        base = root / "STARE" / "stare"
        image_paths = [base / "ppmdata" / f"{image_id}.ppm" for image_id in image_ids]
        mask_paths = [base / "labels-ah" / f"{image_id}.ah.ppm" for image_id in image_ids]
    elif dataset == "CHASE_DB1":
        base = root / "CHASE_DB1" / "CHASEDB1"
        image_paths = [base / f"{image_id}.jpg" for image_id in image_ids]
        mask_paths = [base / f"{image_id}_1stHO.png" for image_id in image_ids]
    else:
        raise ValueError(f"unknown dataset: {dataset}")
    return image_paths, mask_paths


def apply_gamma_correction(image: np.ndarray, gamma: float = 1.2) -> np.ndarray:
    """Apply the fixed 8-bit lookup-table gamma transform used by the experiment."""
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    table = np.array(
        [(value / 255.0) ** (1.0 / gamma) * 255 for value in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(image, table)


def prepare_fundus_green(image: Image.Image) -> Image.Image:
    """Extract green, then apply the fixed gamma and CLAHE stages before augmentation."""
    green = np.asarray(image.convert("RGB"), dtype=np.uint8)[:, :, 1]
    corrected = apply_gamma_correction(green, gamma=1.2)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(32, 32))
    equalized = clahe.apply(corrected)
    return Image.fromarray(equalized)


def _image_to_tensor(image: Image.Image, size: int) -> Tensor:
    resized = image.resize((size, size), resample=Image.Resampling.BILINEAR)
    array = np.array(resized, dtype=np.float32, copy=True) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def _mask_to_tensor(mask: Image.Image, size: int) -> Tensor:
    resized = mask.resize((size, size), resample=Image.Resampling.NEAREST)
    array = np.array(resized, dtype=np.float32, copy=True) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def preprocess_fundus(image: Image.Image, size: int = IMAGE_SIZE) -> Tensor:
    """Convert an RGB fundus image to the normalized, resized green-channel tensor."""
    return _image_to_tensor(prepare_fundus_green(image), size)


def _binary_mask(mask: Image.Image) -> Image.Image:
    binary = (np.asarray(mask.convert("L"), dtype=np.uint8) > 127).astype(np.uint8) * 255
    return Image.fromarray(binary)


def preprocess_mask(mask: Image.Image, size: int = IMAGE_SIZE) -> Tensor:
    """Binarize a reference mask and resize it with nearest-neighbor interpolation."""
    return _mask_to_tensor(_binary_mask(mask), size)


def preprocess_image_path(image_path: str | Path, size: int = IMAGE_SIZE) -> Tensor:
    """Preprocess one deployment image without requiring a reference mask."""
    with Image.open(image_path) as image:
        return preprocess_fundus(image, size=size)


class JointVesselDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """A fixed list of source-qualified IDs, with optional paired rotation augmentation."""

    def __init__(
        self,
        data_root: str | Path,
        pairs: Sequence[Pair],
        augment: bool = False,
        size: int = IMAGE_SIZE,
    ) -> None:
        self.pairs = tuple(pairs)
        self.augment = augment
        self.size = size
        self.source_indices = tuple(SOURCE_TO_INDEX[dataset] for dataset, _ in self.pairs)
        self.images: list[Image.Image | None] = [None] * len(self.pairs)
        self.masks: list[Image.Image | None] = [None] * len(self.pairs)
        grouped: dict[str, list[tuple[int, str]]] = {}
        for index, (dataset, image_id) in enumerate(self.pairs):
            grouped.setdefault(dataset, []).append((index, image_id))
        for dataset, indexed_ids in grouped.items():
            image_paths, mask_paths = dataset_paths(
                data_root,
                dataset,
                [image_id for _, image_id in indexed_ids],
            )
            for (index, _), image_path, mask_path in zip(
                indexed_ids,
                image_paths,
                mask_paths,
                strict=True,
            ):
                if not image_path.is_file():
                    raise FileNotFoundError(f"missing {dataset} image: {image_path}")
                if not mask_path.is_file():
                    raise FileNotFoundError(f"missing {dataset} reference mask: {mask_path}")
                with Image.open(image_path) as image:
                    self.images[index] = image.convert("RGB").copy()
                with Image.open(mask_path) as mask:
                    self.masks[index] = mask.convert("L").copy()

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        image = self.images[index]
        mask = self.masks[index]
        if image is None or mask is None:
            raise RuntimeError(f"image pair at index {index} was not initialized")
        image = prepare_fundus_green(image)
        mask = _binary_mask(mask)
        if self.augment and random.random() < ROTATION_PROBABILITY:
            angle = random.choice(ROTATION_ANGLES)
            image = image.rotate(angle, resample=Image.Resampling.BILINEAR)
            mask = mask.rotate(angle, resample=Image.Resampling.NEAREST)
        return (
            _image_to_tensor(image, self.size),
            _mask_to_tensor(mask, self.size),
            torch.tensor(self.source_indices[index], dtype=torch.long),
        )


def make_train_dev_loaders(
    data_root: str | Path,
    batch_size: int = 4,
    seed: int = 42,
    image_size: int = IMAGE_SIZE,
    generator: torch.Generator | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Build only the 40-image train-active and six-image dev loaders; does not open test data."""
    verify_split_integrity()
    if generator is None:
        generator = torch.Generator().manual_seed(seed)
    train_dataset = JointVesselDataset(data_root, TRAIN_ACTIVE, augment=True, size=image_size)
    dev_dataset = JointVesselDataset(data_root, JOINT_DEV, augment=False, size=image_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, dev_loader


def make_test_loaders(
    data_root: str | Path,
    batch_size: int = 4,
) -> dict[str, tuple[DataLoader, tuple[str, ...]]]:
    """Build the three official test loaders. Call only from the explicit evaluation workflow."""
    verify_split_integrity()
    loaders: dict[str, tuple[DataLoader, tuple[str, ...]]] = {}
    for dataset in SOURCE_NAMES:
        pairs = tuple(pair for pair in JOINT_TEST if pair[0] == dataset)
        image_ids = tuple(image_id for _, image_id in pairs)
        dataset_object = JointVesselDataset(data_root, pairs, augment=False)
        loaders[dataset] = (
            DataLoader(dataset_object, batch_size=batch_size, shuffle=False, num_workers=0),
            image_ids,
        )
    return loaders
