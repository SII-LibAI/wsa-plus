from __future__ import annotations

from pathlib import Path

from torch import nn


def load_shared_safetensors_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
) -> tuple[set[str], list[str], list[str]]:
    """Load a checkpoint while restoring aliases for tied model tensors."""
    from safetensors.torch import load_model

    target_keys = set(model.state_dict())
    missing, unexpected = load_model(
        model,
        checkpoint_path,
        strict=False,
        device="cpu",
    )
    missing_list = list(missing)
    return target_keys - set(missing_list), missing_list, list(unexpected)
