from __future__ import annotations

from pathlib import Path

import pytest

from dcvd_net.training import TrainConfig

CONFIG_PATHS = tuple(sorted((Path(__file__).parents[1] / "configs").glob("*.yaml")))


@pytest.mark.parametrize("config_path", CONFIG_PATHS, ids=lambda path: path.stem)
def test_checked_in_training_configs_parse_and_match_the_declared_protocol(
    config_path: Path,
) -> None:
    config = TrainConfig.from_yaml(config_path)

    assert config.arm in {"C", "T", "A1", "A2", "A3", "A4"}
    assert config.batch_size == 4
    assert config.max_epochs == 300
    assert config.patience == 20
    assert config.optimizer == "adam"
    assert config.num_workers == 0
    assert config.threshold == 0.6
    assert config.image_size == 512
    assert config.param_cap == 1_000_000
