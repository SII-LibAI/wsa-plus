from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


def _load_temporal_module():
    path = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "lerobot"
        / "policies"
        / "WSA_Memory"
        / "temporal_visual_memory.py"
    )
    spec = importlib.util.spec_from_file_location("wsa_memory_temporal_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


temporal = _load_temporal_module()


def test_relative_time_encoding_uses_current_offset_zero() -> None:
    encoding = temporal.fixed_relative_time_encoding(6, 15, 1.0)
    assert encoding.shape == (6, 15)
    # sin(0)=0 and cos(0)=1 for the current timestamp's zero offset.
    torch.testing.assert_close(encoding[-1, :7], torch.zeros(7))
    torch.testing.assert_close(encoding[-1, 7:14], torch.ones(7))
    assert encoding[-1, 14].item() == 0.0


def test_zero_gate_preserves_current_tokens_and_fixed_token_count() -> None:
    module = temporal.TemporalVisualMemory(
        16,
        num_frames=6,
        stride_seconds=1.0,
        history_dropout=0.3,
    )
    frame_tokens = torch.randn(2, 3, 6, 5, 16, dtype=torch.bfloat16)
    valid = torch.tensor(
        [[False, False, True, True, True, True], [True] * 6],
        dtype=torch.bool,
    )
    output = module(frame_tokens, valid)
    assert output.shape == (2, 3, 5, 16)
    assert output.dtype == torch.bfloat16
    torch.testing.assert_close(output, frame_tokens[:, :, -1])


def test_masked_history_cannot_change_temporal_result() -> None:
    torch.manual_seed(0)
    module = temporal.TemporalVisualMemory(
        8,
        num_frames=3,
        stride_seconds=0.5,
        history_dropout=0.0,
    )
    module.residual_gate.data.fill_(1.0)
    valid = torch.tensor([[False, True, True]])
    tokens = torch.randn(1, 1, 3, 2, 8)
    changed = tokens.clone()
    changed[:, :, 0] = 10_000
    torch.testing.assert_close(module(tokens, valid), module(changed, valid))


def test_history_dropout_never_drops_current_or_revives_padding() -> None:
    torch.manual_seed(0)
    module = temporal.TemporalVisualMemory(
        8,
        num_frames=4,
        stride_seconds=1.0,
        history_dropout=0.9,
    )
    original = torch.tensor([[False, True, True, True], [True, True, True, True]])
    dropped = module.apply_history_dropout(original, training=True)
    assert bool(dropped[:, -1].all())
    assert not bool(dropped[0, 0])
    assert bool((~dropped | original).all())
