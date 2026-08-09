from __future__ import annotations

import math

import torch
from torch import nn


def fixed_relative_time_encoding(
    num_frames: int,
    dim: int,
    stride_seconds: float,
) -> torch.Tensor:
    """Sinusoidal encoding whose final (current) timestamp has offset zero."""
    if num_frames <= 0 or dim <= 0:
        raise ValueError("num_frames and dim must be positive")
    offsets = torch.arange(-(num_frames - 1), 1, dtype=torch.float32)
    offsets = offsets * float(stride_seconds)
    half = dim // 2
    if half == 0:
        return offsets[:, None]
    frequencies = torch.exp(
        torch.arange(half, dtype=torch.float32)
        * (-math.log(10_000.0) / max(half - 1, 1))
    )
    angles = offsets[:, None] * frequencies[None, :]
    encoding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if encoding.shape[-1] < dim:
        encoding = torch.nn.functional.pad(encoding, (0, dim - encoding.shape[-1]))
    return encoding[:, :dim]


def _attention_heads(dim: int, maximum: int = 8) -> int:
    for heads in range(min(maximum, dim), 0, -1):
        if dim % heads == 0:
            return heads
    return 1


class TemporalVisualMemory(nn.Module):
    """Stage-A fixed-token temporal attention over spatially encoded frames.

    Qwen processes every frame spatially first.  For each camera and spatial
    position, the current token attends to the aligned tokens over time.  Only
    the current-frame token grid is returned, so the language backbone always
    receives exactly the single-frame placeholder count.
    """

    def __init__(
        self,
        dim: int,
        *,
        num_frames: int,
        stride_seconds: float,
        history_dropout: float,
    ) -> None:
        super().__init__()
        self.num_frames = int(num_frames)
        self.history_dropout = float(history_dropout)
        self.norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            dim,
            _attention_heads(dim),
            batch_first=True,
        )
        self.residual_gate = nn.Parameter(torch.zeros(()))
        self.register_buffer(
            "relative_time_encoding",
            fixed_relative_time_encoding(num_frames, dim, stride_seconds),
            persistent=True,
        )

    def apply_history_dropout(
        self,
        history_valid_mask: torch.Tensor,
        *,
        training: bool,
    ) -> torch.Tensor:
        mask = history_valid_mask.to(dtype=torch.bool)
        if mask.ndim != 2 or mask.shape[1] != self.num_frames:
            raise ValueError(
                f"history_valid_mask must be [B,{self.num_frames}], got {tuple(mask.shape)}"
            )
        if not bool(mask[:, -1].all()):
            raise ValueError("The current visual/state history position must always be valid")
        if training and self.history_dropout > 0 and self.num_frames > 1:
            keep = torch.rand(mask.shape, device=mask.device) >= self.history_dropout
            keep[:, -1] = True
            mask = mask & keep
        return mask

    def forward(
        self,
        frame_tokens: torch.Tensor,
        history_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            frame_tokens: ``[B, V, K, N, D]`` Qwen visual tokens.
            history_valid_mask: ``[B, K]`` after padding and training dropout.

        Returns:
            ``[B, V, N, D]`` tokens, exactly one frame worth per camera.
        """
        if frame_tokens.ndim != 5:
            raise ValueError(
                f"frame_tokens must be [B,V,K,N,D], got {tuple(frame_tokens.shape)}"
            )
        batch, views, frames, spatial_tokens, dim = frame_tokens.shape
        if frames != self.num_frames:
            raise ValueError(f"Expected K={self.num_frames}, got K={frames}")
        mask = history_valid_mask.to(device=frame_tokens.device, dtype=torch.bool)
        if mask.shape != (batch, frames):
            raise ValueError(
                f"history_valid_mask must be {(batch, frames)}, got {tuple(mask.shape)}"
            )

        # The inherited Qwen vision tower commonly emits bf16 while newly
        # constructed PyTorch attention layers start in fp32.  Run the small
        # temporal adapter in its own parameter dtype, then cast the residual
        # back to Qwen's dtype.  This also keeps CPU smoke tests usable.
        input_dtype = frame_tokens.dtype
        compute_dtype = self.attention.in_proj_weight.dtype
        time = self.relative_time_encoding.to(
            device=frame_tokens.device, dtype=compute_dtype
        )
        tokens = frame_tokens.to(dtype=compute_dtype)
        tokens = tokens + time[None, None, :, None, :]
        tokens = tokens.permute(0, 1, 3, 2, 4).reshape(
            batch * views * spatial_tokens, frames, dim
        )
        current = tokens[:, -1:, :]
        expanded_mask = mask[:, None, None, :].expand(
            batch, views, spatial_tokens, frames
        )
        key_padding_mask = ~expanded_mask.reshape(
            batch * views * spatial_tokens, frames
        )
        attended, _ = self.attention(
            self.norm(current),
            self.norm(tokens),
            self.norm(tokens),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        base_current = frame_tokens[:, :, -1]
        attended = attended[:, 0].reshape(batch, views, spatial_tokens, dim)
        attended = attended.to(dtype=input_dtype)
        gate = torch.tanh(self.residual_gate).to(dtype=input_dtype)
        return base_current + gate * attended
