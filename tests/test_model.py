from __future__ import annotations

import pytest
import torch

from dcvd_net.model import (
    LEGACY_STATE_PREFIXES,
    MODULATION_PARAMETERS,
    PARAMETER_CAP,
    TRUNK_PARAMETERS,
    AblationDCVDNet,
    ConditioningBlock,
    DCVDNet,
    UNetTrunk,
    build_model,
    compatible_state_dict,
    count_parameters,
)


@pytest.mark.parametrize("arm", ("C", "T", "A1", "A2", "A3", "A4"))
def test_all_arms_match_the_parameter_budget(arm: str) -> None:
    model = build_model(arm)
    assert count_parameters(model) == TRUNK_PARAMETERS + MODULATION_PARAMETERS == 759_911
    assert count_parameters(model) <= PARAMETER_CAP


def test_twin_freezes_only_the_identity_modulations() -> None:
    twin = DCVDNet(pinned=True)
    assert count_parameters(twin) == 759_911
    assert count_parameters(twin, trainable_only=True) == TRUNK_PARAMETERS
    assert all(not parameter.requires_grad for parameter in twin.mod.modulation_parameters())


def test_recorded_run_checkpoint_module_names_translate_strictly() -> None:
    original = build_model("C")
    current_to_legacy = {target: source for source, target in LEGACY_STATE_PREFIXES.items()}
    legacy_state: dict[str, torch.Tensor] = {}
    for key, value in original.state_dict().items():
        prefix, separator, suffix = key.partition(".")
        legacy_prefix = current_to_legacy.get(prefix, prefix)
        legacy_key = f"{legacy_prefix}{separator}{suffix}"
        legacy_state[legacy_key] = value.clone()

    restored = build_model("C")
    result = restored.load_state_dict(compatible_state_dict(legacy_state), strict=True)

    assert not result.missing_keys
    assert not result.unexpected_keys
    for original_parameter, restored_parameter in zip(
        original.parameters(),
        restored.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(original_parameter, restored_parameter, rtol=0.0, atol=0.0)


def test_identity_initialized_twin_matches_unmodulated_trunk_bit_for_bit() -> None:
    torch.manual_seed(7)
    trunk = UNetTrunk().eval()
    twin = DCVDNet(pinned=True).eval()
    result = twin.load_state_dict(trunk.state_dict(), strict=False)
    assert not result.unexpected_keys
    assert set(result.missing_keys) == {f"mod.{name}" for name in twin.mod.state_dict()}

    images = torch.randn(2, 1, 32, 32)
    sources = torch.tensor([0, 2], dtype=torch.long)
    with torch.inference_mode():
        expected = trunk(images)
        actual = twin(images, sources)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_source_conditioning_is_selected_per_sample() -> None:
    block = ConditioningBlock(level_indices=(0,), per_source=True)
    with torch.no_grad():
        block.W_gamma[0][:, :] = torch.tensor([[1.0], [2.0], [3.0]])
        block.W_beta[0][:, :] = torch.tensor([[10.0], [20.0], [30.0]])
    features = torch.ones(3, 40, 2, 2)
    source_indices = torch.tensor([2, 0, 1], dtype=torch.long)

    result = block.modulate(features, level_index=0, source_indices=source_indices)

    torch.testing.assert_close(result[:, 0, 0, 0], torch.tensor([33.0, 11.0, 22.0]))
    assert result.shape == features.shape


def test_ablation_ballast_does_not_participate_in_forward() -> None:
    model = AblationDCVDNet("A2")
    assert model.ballast.numel() == 240
    assert model.ballast.requires_grad
    image = torch.randn(1, 1, 32, 32)
    source = torch.tensor([1], dtype=torch.long)
    model(image, source).mean().backward()
    assert model.ballast.grad is None
