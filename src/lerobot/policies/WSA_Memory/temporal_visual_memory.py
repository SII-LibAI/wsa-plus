from __future__ import annotations

import math
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


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


class MEMQwen3VLVisionEncoder(nn.Module):
    """MEM-style space-time attention inside the pretrained Qwen3-VL ViT.

    Spatial blocks remain identical to the pretrained image encoder. Every
    ``temporal_attention_every_n_blocks``-th block adds causal attention along
    time for each aligned spatial patch, reusing that block's existing QKV and
    output projections. Past-frame tokens are discarded before the patch
    merger, so the language backbone receives exactly one frame of tokens.

    The adapter deliberately owns no trainable parameters. This follows MEM's
    initialization contract: the temporal path reuses the image ViT weights,
    and a one-frame (or all-history-dropped) sample is exactly the image-only
    path because its temporal residual is disabled.
    """

    def __init__(
        self,
        dim: int,
        *,
        num_frames: int,
        stride_seconds: float,
        history_dropout: float,
        temporal_attention_every_n_blocks: int,
    ) -> None:
        super().__init__()
        self.num_frames = int(num_frames)
        self.history_dropout = float(history_dropout)
        self.temporal_attention_every_n_blocks = int(
            temporal_attention_every_n_blocks
        )
        if self.num_frames <= 0:
            raise ValueError("num_frames must be positive")
        if self.temporal_attention_every_n_blocks <= 0:
            raise ValueError("temporal_attention_every_n_blocks must be positive")
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
            # pi0.7/MEM drops the history as one conditioning component rather
            # than sampling an independent dropout decision for every frame.
            keep_history = (
                torch.rand(mask.shape[0], 1, device=mask.device)
                >= self.history_dropout
            )
            mask = torch.cat(
                [mask[:, :-1] & keep_history, mask[:, -1:]], dim=1
            )
        return mask

    def is_temporal_layer(self, layer_index: int) -> bool:
        return (int(layer_index) + 1) % self.temporal_attention_every_n_blocks == 0

    @staticmethod
    def _current_tokens(
        hidden_states: torch.Tensor,
        *,
        batch_size: int,
        num_views: int,
        num_frames: int,
        spatial_tokens: int,
    ) -> torch.Tensor:
        return hidden_states.reshape(
            batch_size, num_views, num_frames, spatial_tokens, -1
        )[:, :, -1].reshape(batch_size * num_views * spatial_tokens, -1)

    def _causal_temporal_attention(
        self,
        hidden_states: torch.Tensor,
        block: nn.Module,
        history_valid_mask: torch.Tensor,
        *,
        batch_size: int,
        num_views: int,
        spatial_tokens: int,
    ) -> torch.Tensor:
        """Apply same-patch causal time attention with the block's own weights."""
        num_frames = self.num_frames
        hidden_dim = hidden_states.shape[-1]
        expected_tokens = batch_size * num_views * num_frames * spatial_tokens
        if hidden_states.shape != (expected_tokens, hidden_dim):
            raise ValueError(
                "Unexpected flattened Qwen visual token layout: "
                f"got={tuple(hidden_states.shape)}, expected_tokens={expected_tokens}"
            )
        mask = history_valid_mask.to(device=hidden_states.device, dtype=torch.bool)
        if mask.shape != (batch_size, num_frames):
            raise ValueError(
                f"history_valid_mask must be {(batch_size, num_frames)}, "
                f"got {tuple(mask.shape)}"
            )

        tokens = hidden_states.reshape(
            batch_size, num_views, num_frames, spatial_tokens, hidden_dim
        ).permute(0, 1, 3, 2, 4)
        time = self.relative_time_encoding.to(
            device=hidden_states.device, dtype=hidden_states.dtype
        )
        temporal_input = block.norm1(tokens + time[None, None, None, :, :])
        temporal_input = temporal_input.reshape(
            batch_size * num_views * spatial_tokens, num_frames, hidden_dim
        )

        attention = block.attn
        qkv = attention.qkv(temporal_input).reshape(
            temporal_input.shape[0],
            num_frames,
            3,
            attention.num_heads,
            attention.head_dim,
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)

        expanded_valid = mask[:, None, None, :].expand(
            batch_size, num_views, spatial_tokens, num_frames
        ).reshape(-1, num_frames)
        causal = torch.ones(
            num_frames, num_frames, dtype=torch.bool, device=hidden_states.device
        ).tril()
        allowed = causal[None, None, :, :] & expanded_valid[:, None, None, :]
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=allowed,
            dropout_p=0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(
            temporal_input.shape[0], num_frames, hidden_dim
        )
        attended = attention.proj(attended)

        # Invalid padded queries never contribute, and a one-frame context must
        # exactly recover the original image-only ViT computation.
        has_history = expanded_valid.sum(dim=-1, keepdim=True) > 1
        active_queries = expanded_valid & has_history
        attended = attended * active_queries[..., None]
        return attended.reshape(
            batch_size, num_views, spatial_tokens, num_frames, hidden_dim
        ).permute(0, 1, 3, 2, 4).reshape(expected_tokens, hidden_dim)

    def _mem_block_forward(
        self,
        hidden_states: torch.Tensor,
        block: nn.Module,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        history_valid_mask: torch.Tensor,
        *,
        batch_size: int,
        num_views: int,
        spatial_tokens: int,
        **kwargs,
    ) -> torch.Tensor:
        normalized = block.norm1(hidden_states)
        spatial_output = block.attn(
            normalized,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        temporal_output = self._causal_temporal_attention(
            hidden_states,
            block,
            history_valid_mask,
            batch_size=batch_size,
            num_views=num_views,
            spatial_tokens=spatial_tokens,
        )
        hidden_states = hidden_states + spatial_output + temporal_output
        return hidden_states + block.mlp(block.norm2(hidden_states))

    def forward(
        self,
        vision_model: nn.Module,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        history_valid_mask: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run the Qwen ViT with interleaved MEM temporal attention.

        ``grid_thw`` is ``[B,V,K,3]`` and each entry represents one processed
        image. All views are resized to the same grid so spatial patch positions
        align across the K history samples.
        """
        if grid_thw.ndim != 4 or grid_thw.shape[-1] != 3:
            raise ValueError(
                f"grid_thw must be [B,V,K,3], got {tuple(grid_thw.shape)}"
            )
        batch_size, num_views, num_frames = grid_thw.shape[:3]
        if num_frames != self.num_frames:
            raise ValueError(f"Expected K={self.num_frames}, got K={num_frames}")
        flat_grid = grid_thw.reshape(-1, 3)
        if not bool((flat_grid[:, 0] == 1).all()):
            raise ValueError("MEM expects each grid entry to represent one image frame")
        if not bool((flat_grid[:, 1:] == flat_grid[0, 1:]).all()):
            raise ValueError("MEM requires equal resized spatial grids across views and history")

        height, width = (int(value.item()) for value in flat_grid[0, 1:])
        spatial_tokens = height * width
        expected_patches = flat_grid.shape[0] * spatial_tokens

        hidden_states = vision_model.patch_embed(pixel_values)
        if hidden_states.shape[0] != expected_patches:
            raise ValueError(
                "Qwen patch/grid mismatch in MEM encoder: "
                f"patches={hidden_states.shape[0]}, expected={expected_patches}"
            )
        hidden_states = hidden_states + vision_model.fast_pos_embed_interpolate(
            flat_grid
        )

        rotary_pos_emb = vision_model.rot_pos_emb(flat_grid).reshape(
            expected_patches, -1
        )
        rotary = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (rotary.cos(), rotary.sin())
        frame_lengths = flat_grid[:, 1] * flat_grid[:, 2]
        cu_seqlens = torch.cumsum(
            frame_lengths,
            dim=0,
            dtype=flat_grid.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        deepstack_features: list[torch.Tensor] = []
        for layer_index, block in enumerate(vision_model.blocks):
            if self.is_temporal_layer(layer_index):
                temporal_block = partial(
                    self._mem_block_forward,
                    block=block,
                    cu_seqlens=cu_seqlens,
                    position_embeddings=position_embeddings,
                    history_valid_mask=history_valid_mask,
                    batch_size=batch_size,
                    num_views=num_views,
                    spatial_tokens=spatial_tokens,
                    **kwargs,
                )

                if getattr(vision_model, "gradient_checkpointing", False) and vision_model.training:
                    hidden_states = checkpoint(
                        temporal_block,
                        hidden_states,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    hidden_states = temporal_block(hidden_states)
            else:
                hidden_states = block(
                    hidden_states,
                    cu_seqlens=cu_seqlens,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )

            if layer_index in vision_model.deepstack_visual_indexes:
                current = self._current_tokens(
                    hidden_states,
                    batch_size=batch_size,
                    num_views=num_views,
                    num_frames=num_frames,
                    spatial_tokens=spatial_tokens,
                )
                merger_index = vision_model.deepstack_visual_indexes.index(layer_index)
                deepstack_features.append(
                    vision_model.deepstack_merger_list[merger_index](current)
                )

        current = self._current_tokens(
            hidden_states,
            batch_size=batch_size,
            num_views=num_views,
            num_frames=num_frames,
            spatial_tokens=spatial_tokens,
        )
        return vision_model.merger(current), deepstack_features
