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


def _module(dim: int = 8, num_frames: int = 3, dropout: float = 0.0):
    return temporal.MEMQwen3VLVisionEncoder(
        dim,
        num_frames=num_frames,
        stride_seconds=1.0,
        history_dropout=dropout,
        temporal_attention_every_n_blocks=4,
    )


class _Attention(torch.nn.Module):
    def __init__(self, dim: int, num_heads: int = 2) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = torch.nn.Linear(dim, dim * 3)
        self.proj = torch.nn.Linear(dim, dim)


class _Block(torch.nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(dim)
        self.attn = _Attention(dim)


def test_current_selection_has_fixed_single_frame_token_count() -> None:
    module = _module(dim=16, num_frames=6, dropout=0.3)
    frame_tokens = torch.randn(2 * 3 * 6 * 5, 16, dtype=torch.bfloat16)
    output = module._current_tokens(
        frame_tokens,
        batch_size=2,
        num_views=3,
        num_frames=6,
        spatial_tokens=5,
    )
    expected = frame_tokens.reshape(2, 3, 6, 5, 16)[:, :, -1].reshape(-1, 16)
    assert output.shape == (2 * 3 * 5, 16)
    torch.testing.assert_close(output, expected)


def test_mem_adapter_has_no_trainable_parameters_and_uses_every_fourth_layer() -> None:
    module = _module()
    assert list(module.parameters()) == []
    assert [index for index in range(12) if module.is_temporal_layer(index)] == [3, 7, 11]


def test_masked_history_cannot_change_temporal_result() -> None:
    torch.manual_seed(0)
    module = _module()
    block = _Block(8)
    valid = torch.tensor([[False, True, True]])
    tokens = torch.randn(1, 1, 3, 2, 8)
    changed = tokens.clone()
    changed[:, :, 0] = 10_000
    output = module._causal_temporal_attention(
        tokens.reshape(-1, 8),
        block,
        valid,
        batch_size=1,
        num_views=1,
        spatial_tokens=2,
    )
    changed_output = module._causal_temporal_attention(
        changed.reshape(-1, 8),
        block,
        valid,
        batch_size=1,
        num_views=1,
        spatial_tokens=2,
    )
    torch.testing.assert_close(output, changed_output)


def test_temporal_attention_is_causal() -> None:
    torch.manual_seed(1)
    module = _module()
    block = _Block(8)
    valid = torch.ones(1, 3, dtype=torch.bool)
    tokens = torch.randn(1, 1, 3, 2, 8)
    changed = tokens.clone()
    changed[:, :, -1] = 10_000

    def run(values: torch.Tensor) -> torch.Tensor:
        return module._causal_temporal_attention(
            values.reshape(-1, 8),
            block,
            valid,
            batch_size=1,
            num_views=1,
            spatial_tokens=2,
        ).reshape(1, 1, 3, 2, 8)

    # A future/current-frame change cannot alter an earlier frame's update.
    torch.testing.assert_close(run(tokens)[:, :, 1], run(changed)[:, :, 1])


def test_one_frame_context_exactly_disables_temporal_residual() -> None:
    module = temporal.MEMQwen3VLVisionEncoder(
        16,
        num_frames=1,
        stride_seconds=1.0,
        history_dropout=0.0,
        temporal_attention_every_n_blocks=4,
    )
    output = module._causal_temporal_attention(
        torch.randn(2 * 3 * 1 * 5, 16),
        _Block(16),
        torch.ones(2, 1, dtype=torch.bool),
        batch_size=2,
        num_views=3,
        spatial_tokens=5,
    )
    torch.testing.assert_close(output, torch.zeros_like(output))


def test_history_dropout_never_drops_current_or_revives_padding() -> None:
    module = _module(num_frames=4, dropout=1.0)
    original = torch.tensor([[False, True, True, True], [True, True, True, True]])
    dropped = module.apply_history_dropout(original, training=True)
    assert bool(dropped[:, -1].all())
    assert not bool(dropped[:, :-1].any())
    assert bool((~dropped | original).all())


def test_real_qwen_vision_interface_and_single_frame_equivalence() -> None:
    from transformers.models.qwen3_vl.configuration_qwen3_vl import (
        Qwen3VLVisionConfig,
    )
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    torch.manual_seed(2)
    config = Qwen3VLVisionConfig(
        depth=4,
        hidden_size=32,
        intermediate_size=64,
        num_heads=4,
        in_channels=3,
        patch_size=2,
        temporal_patch_size=2,
        spatial_merge_size=2,
        out_hidden_size=64,
        num_position_embeddings=64,
        deepstack_visual_indexes=[1, 3],
        _attn_implementation="eager",
    )
    vision = Qwen3VLVisionModel(config).eval()
    module = temporal.MEMQwen3VLVisionEncoder(
        32,
        num_frames=3,
        stride_seconds=1.0,
        history_dropout=0.0,
        temporal_attention_every_n_blocks=4,
    ).eval()
    grid = torch.tensor([[[[1, 4, 4]] * 3]], dtype=torch.long)
    pixels = torch.randn(1, 1, 3, 16, 24)
    current_only = torch.tensor([[False, False, True]])

    with torch.no_grad():
        memory_output, memory_deepstack = module(
            vision,
            pixels.reshape(-1, 24),
            grid,
            current_only,
        )
        base_output, base_deepstack = vision(
            pixels[:, :, -1].reshape(-1, 24),
            grid[:, :, -1].reshape(-1, 3),
        )

    assert memory_output.shape == (4, 64)
    torch.testing.assert_close(memory_output, base_output, rtol=0, atol=0)
    for memory_value, base_value in zip(
        memory_deepstack, base_deepstack, strict=True
    ):
        torch.testing.assert_close(memory_value, base_value, rtol=0, atol=0)
