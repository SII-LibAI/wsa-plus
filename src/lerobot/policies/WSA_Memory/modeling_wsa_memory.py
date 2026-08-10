from __future__ import annotations

import logging
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Hashable, Sequence

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from torch import Tensor, nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.WSA_Base.configuration_wsa_base import WSABaseConfig
from lerobot.policies.WSA_Base.modeling_wsa_base import (
    WSABaseModel,
    WSABasePolicy,
    make_att_2d_masks,
    pad_vector,
)
from lerobot.policies.WSA_Memory.configuration_wsa_memory import (
    WSA_MEMORY,
    WSAMemoryConfig,
)
from lerobot.policies.WSA_Memory.sidecar import (
    build_memory_prompt,
    prompt_memory_from_external,
)
from lerobot.policies.WSA_Memory.temporal_visual_memory import (
    MEMQwen3VLVisionEncoder,
    fixed_relative_time_encoding,
)
from lerobot.policies.WSA_Memory.transform_wsa_memory import (
    MEMORY_ACTION_LOSS_MASK,
    MEMORY_ENV_IDS,
    MEMORY_HISTORY_VALID_MASK,
    MEMORY_PREPARED,
)
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import (
    ACTION,
    OBS_IMAGES,
    OBS_PREFIX,
    OBS_STATE,
    OPENPI_ATTENTION_MASK_VALUE,
    SAMPLE_ACTION_LOSS_MASK,
)


class WSAMemoryModel(WSABaseModel):
    """WSABase action model with fixed-token visual and K-token state memory."""

    def __init__(self, config: WSAMemoryConfig):
        super().__init__(config)
        self._freeze_action_only_output_modules()
        vision_model = self.qwen3_vl_with_expert.und_expert.visual
        vision_hidden_dim = int(vision_model.config.hidden_size)
        state_dim = int(self.state_proj.out_features)
        self.temporal_visual_memory = MEMQwen3VLVisionEncoder(
            vision_hidden_dim,
            num_frames=config.history_num_frames,
            stride_seconds=config.history_stride_seconds,
            history_dropout=config.visual_history_dropout,
            temporal_attention_every_n_blocks=(
                config.temporal_attention_every_n_blocks
            ),
        )
        self.register_buffer(
            "state_relative_time_encoding",
            fixed_relative_time_encoding(
                config.history_num_frames,
                state_dim,
                config.history_stride_seconds,
            ),
            persistent=True,
        )
        self.state_time_gate = nn.Parameter(torch.zeros(()))

    def _freeze_action_only_output_modules(self) -> None:
        """Exclude supervision-only decoders while retaining conditioning paths."""
        output_modules = (
            self.upsample_conv,
            self.cosmos_out_proj,
            self.cosmos_out_layer_norm,
            self.future_3d_messenger_norms,
            self.future_3d_output_decoder,
            self.future_3d_layer_input_norms,
            self.future_3d_shared_refine_trunk,
            self.da3_query_projectors,
        )
        for module in output_modules:
            if module is None:
                continue
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad = False
            self._register_frozen_eval_module(module)
        if self.future_3d_shared_output_queries is not None:
            self.future_3d_shared_output_queries.requires_grad = False

    def embed_prefix(
        self,
        pixel_values: Tensor,
        image_grid_thw: Tensor,
        lang_tokens: Tensor,
        lang_masks: Tensor,
        history_valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch_size = lang_tokens.shape[0]
        grids = image_grid_thw.reshape(batch_size, -1, 3)
        num_views = grids.shape[1]
        num_frames = int(history_valid_mask.shape[1])
        if num_frames != self.config.history_num_frames:
            raise ValueError(
                f"Expected K={self.config.history_num_frames}, got K={num_frames}"
            )

        per_current_token_counts = (
            grids.prod(dim=-1)
            // int(self.qwen3_vl_with_expert.und_expert.visual.spatial_merge_size**2)
        )
        if not bool((per_current_token_counts == per_current_token_counts[0, 0]).all()):
            raise ValueError(
                "MEM temporal memory requires the resized camera grids to have equal token counts"
            )
        spatial_tokens = int(per_current_token_counts[0, 0].item())

        memory_grids = grids[:, :, None, :].expand(
            batch_size, num_views, num_frames, 3
        )
        patch_dim = pixel_values.shape[-1]
        pixels_by_batch = pixel_values.reshape(batch_size, -1, patch_dim)
        expected_raw_patches = int(memory_grids[0].prod(dim=-1).sum().item())
        if pixels_by_batch.shape[1] != expected_raw_patches:
            raise ValueError(
                "Qwen pixel/grid mismatch for WSA-Memory: "
                f"pixels_per_sample={pixels_by_batch.shape[1]}, "
                f"grid_raw_patches={expected_raw_patches}"
            )

        visual_tokens, _ = self.temporal_visual_memory(
            self.qwen3_vl_with_expert.und_expert.visual,
            pixels_by_batch.reshape(-1, patch_dim),
            memory_grids,
            history_valid_mask,
        )
        expected_compressed_tokens = batch_size * num_views * spatial_tokens
        if visual_tokens.shape[0] != expected_compressed_tokens:
            raise ValueError(
                "Qwen visual encoder returned a token count inconsistent with image grids: "
                f"got={visual_tokens.shape[0]}, expected={expected_compressed_tokens}"
            )
        current_tokens = visual_tokens.reshape(
            batch_size, num_views, spatial_tokens, visual_tokens.shape[-1]
        )

        image_token_id = self.qwen3_vl_with_expert.und_expert.config.image_token_id
        embeddings = self.qwen3_vl_with_expert.und_expert.get_input_embeddings()(
            lang_tokens
        )
        placeholder_mask = lang_tokens == image_token_id
        expected_placeholders = num_views * spatial_tokens
        counts = placeholder_mask.sum(dim=1)
        if not bool((counts == expected_placeholders).all()):
            raise ValueError(
                "WSA-Memory current-frame placeholders do not match compressed visual tokens: "
                f"placeholder_counts={counts.tolist()}, expected={expected_placeholders}"
            )
        embeddings[placeholder_mask] = current_tokens.reshape(
            batch_size * expected_placeholders, -1
        ).to(dtype=embeddings.dtype)

        pad_masks = lang_masks.to(torch.bool)
        att_masks = torch.zeros_like(pad_masks, dtype=torch.bool)
        return embeddings, pad_masks, att_masks

    def embed_suffix(
        self,
        state: Tensor,
        history_valid_mask: Tensor,
        noisy_actions: Tensor,
        timestep: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if state.ndim != 3:
            raise ValueError(f"state memory must be [B,K,D], got {tuple(state.shape)}")
        batch_size, num_frames = state.shape[:2]
        if history_valid_mask.shape != (batch_size, num_frames):
            raise ValueError(
                "state and history_valid_mask disagree: "
                f"state={tuple(state.shape)}, mask={tuple(history_valid_mask.shape)}"
            )
        if num_frames > self.state_relative_time_encoding.shape[0]:
            raise ValueError(
                f"Runtime K={num_frames} exceeds configured K={self.config.history_num_frames}"
            )

        # Reuse the complete WSABase action/timestep embedding path, replacing
        # only its single state token with K independently projected tokens.
        base_suffix, _, _ = super().embed_suffix(
            state[:, -1], noisy_actions, timestep
        )
        action_embeddings = base_suffix[:, 1:]
        suffix_dtype = action_embeddings.dtype
        state_embeddings = self.state_proj(
            state.to(dtype=self.state_proj.weight.dtype)
        )
        time_encoding = self.state_relative_time_encoding[-num_frames:].to(
            device=state.device, dtype=state_embeddings.dtype
        )
        state_embeddings = state_embeddings + torch.tanh(self.state_time_gate) * time_encoding
        valid = history_valid_mask.to(device=state.device, dtype=torch.bool)
        if not bool(valid[:, -1].all()):
            raise ValueError("The current state token must always be valid")
        state_embeddings = state_embeddings * valid[..., None]

        embeddings = torch.cat(
            [state_embeddings.to(dtype=suffix_dtype), action_embeddings], dim=1
        )
        action_pad = torch.ones(
            batch_size,
            action_embeddings.shape[1],
            dtype=torch.bool,
            device=state.device,
        )
        pad_masks = torch.cat([valid, action_pad], dim=1)
        att_pattern = [1] + [0] * (num_frames - 1)
        att_pattern += [1] + [0] * (action_embeddings.shape[1] - 1)
        att_masks = torch.tensor(
            att_pattern, dtype=embeddings.dtype, device=embeddings.device
        )[None].expand(batch_size, -1)
        return embeddings, pad_masks, att_masks

    @staticmethod
    def _select_recent_cosmos_frames(
        images: Tensor, history_valid_mask: Tensor, count: int = 2
    ) -> Tensor:
        selected: list[Tensor] = []
        for batch_index in range(images.shape[0]):
            valid_indices = torch.where(history_valid_mask[batch_index])[0]
            if valid_indices.numel() == 0:
                raise ValueError("Cosmos frame selection received an all-invalid history")
            indices = valid_indices[-count:]
            if indices.numel() < count:
                indices = torch.cat(
                    [indices[:1].expand(count - indices.numel()), indices]
                )
            selected.append(images[batch_index, :, indices])
        return torch.stack(selected, dim=0)

    def _prepare_memory_suffix_cache(
        self,
        prefix_pad_masks: Tensor,
        max_prefix_position_ids: Tensor,
        history_valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch_size, num_frames = history_valid_mask.shape
        suffix_len = num_frames + self.config.chunk_size
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d = prefix_pad_masks[:, None, :].expand(
            batch_size, suffix_len, prefix_len
        )
        action_pad = torch.ones(
            batch_size,
            self.config.chunk_size,
            dtype=torch.bool,
            device=prefix_pad_masks.device,
        )
        suffix_pad = torch.cat(
            [history_valid_mask.to(torch.bool), action_pad], dim=1
        )
        att_pattern = [1] + [0] * (num_frames - 1)
        att_pattern += [1] + [0] * (self.config.chunk_size - 1)
        suffix_att = torch.tensor(
            att_pattern, dtype=torch.bool, device=prefix_pad_masks.device
        )[None].expand(batch_size, -1)
        suffix_att_2d = make_att_2d_masks(suffix_pad, suffix_att)
        full_att_2d = torch.cat([prefix_pad_2d, suffix_att_2d], dim=2)
        full_att_4d = torch.where(
            full_att_2d[:, None], 0.0, OPENPI_ATTENTION_MASK_VALUE
        )
        # Compact positions over invalid episode-start history slots, matching
        # the padding-aware training position construction.  Masked slots may
        # share a dummy position; every valid state/action advances by one.
        suffix_offsets = suffix_pad.to(dtype=torch.long).cumsum(dim=-1).clamp_min(1)
        position_ids = (
            suffix_offsets[None]
            .expand(3, -1, -1)
            .to(max_prefix_position_ids)
            + max_prefix_position_ids
        )
        return full_att_4d, position_ids

    def forward(
        self,
        images: Tensor,
        img_masks: Tensor,
        pixel_values: Tensor,
        image_grid_thw: Tensor,
        lang_tokens: Tensor,
        lang_masks: Tensor,
        state: Tensor,
        history_valid_mask: Tensor,
        actions: Tensor,
        noise: Tensor | None = None,
        time: Tensor | None = None,
    ) -> Tensor:
        """Action-only training forward; no future/teacher targets are constructed."""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        target_velocity = noise - actions

        effective_history_mask = self.temporal_visual_memory.apply_history_dropout(
            history_valid_mask, training=self.training
        )
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            pixel_values,
            image_grid_thw,
            lang_tokens,
            lang_masks,
            effective_history_mask,
        )
        cosmos_images = self._select_recent_cosmos_frames(
            images, effective_history_mask
        )
        middle_embs, middle_pad_masks, middle_att_masks = self.embed_middle(
            cosmos_images, img_masks
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
            state, effective_history_mask, x_t, time
        )

        if (
            self.qwen3_vl_with_expert.und_expert.language_model.layers[0]
            .self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            prefix_embs = prefix_embs.to(torch.bfloat16)
            middle_embs = middle_embs.to(torch.bfloat16)
            suffix_embs = suffix_embs.to(torch.bfloat16)

        pad_masks = torch.cat(
            [prefix_pad_masks, middle_pad_masks, suffix_pad_masks], dim=1
        )
        att_masks = torch.cat(
            [prefix_att_masks, middle_att_masks, suffix_att_masks], dim=1
        )
        attention_2d = self.build_training_attention_mask(
            pad_masks, att_masks, prefix_len=prefix_pad_masks.shape[1]
        )
        position_ids, _ = self.get_position_ids(
            lang_tokens, image_grid_thw, pad_masks
        )
        attention_4d = self._prepare_attention_masks_4d(attention_2d)
        outputs, _ = self.qwen3_vl_with_expert.forward(
            attention_mask=attention_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, middle_embs, suffix_embs],
            use_cache=False,
            collect_middle_layers=None,
        )
        suffix_out = outputs[2][:, -self.config.chunk_size :].to(torch.float32)
        velocity = self.action_out_proj(suffix_out)
        return torch.nn.functional.mse_loss(
            target_velocity, velocity, reduction="none"
        )

    @torch.no_grad()
    def sample_actions(
        self,
        images: Tensor,
        img_masks: Tensor,
        pixel_values: Tensor,
        image_grid_thw: Tensor,
        lang_tokens: Tensor,
        lang_masks: Tensor,
        state: Tensor,
        history_valid_mask: Tensor,
        noise: Tensor | None = None,
        num_steps: int | None = None,
    ) -> Tensor:
        if self.config.attention_mask_mode != "default":
            raise NotImplementedError(
                "WSA-Memory causal/RTC inference is not implemented; use default inference"
            )
        if num_steps is None:
            num_steps = self.config.num_inference_steps
        batch_size = state.shape[0]
        device, dtype = state.device, state.dtype
        if noise is None:
            noise = self.sample_noise(
                (batch_size, self.config.chunk_size, self.config.max_action_dim),
                device,
            )
        effective_history_mask = self.temporal_visual_memory.apply_history_dropout(
            history_valid_mask, training=False
        )
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            pixel_values,
            image_grid_thw,
            lang_tokens,
            lang_masks,
            effective_history_mask,
        )
        prefix_attention = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_positions, _ = self.get_position_ids(
            lang_tokens, image_grid_thw, prefix_pad_masks
        )

        with self._temporary_attention_implementations(
            und_expert_impl="eager",
            gen_expert_impl="eager",
            act_expert_impl="eager",
        ):
            _, cache = self.qwen3_vl_with_expert.forward(
                attention_mask=self._prepare_attention_masks_4d(prefix_attention),
                position_ids=prefix_positions,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None, None],
                use_cache=True,
            )
            max_prefix_positions = prefix_positions.max(dim=-1, keepdim=True).values
            cosmos_images = self._select_recent_cosmos_frames(
                images, effective_history_mask
            )
            middle_embs, middle_pad_masks, middle_att_masks = self.embed_middle(
                cosmos_images, img_masks
            )
            middle_len = middle_pad_masks.shape[1]
            prefix_to_middle = prefix_pad_masks[:, None, :].expand(
                batch_size, middle_len, prefix_pad_masks.shape[1]
            )
            middle_attention = make_att_2d_masks(
                middle_pad_masks, middle_att_masks
            )
            middle_attention = self.apply_view_aware_query_attention(
                middle_attention, prefix_len=0
            )
            full_middle_attention = torch.cat(
                [prefix_to_middle, middle_attention], dim=2
            )
            middle_positions = (
                torch.arange(1, middle_len + 1)
                .repeat(3, 1, 1)
                .to(max_prefix_positions)
                + max_prefix_positions
            )
            (_, _, _), cache = self.qwen3_vl_with_expert.forward(
                attention_mask=self._prepare_attention_masks_4d(
                    full_middle_attention
                ),
                position_ids=middle_positions,
                past_key_values=cache,
                inputs_embeds=[None, middle_embs, None],
                use_cache=True,
            )
            max_positions = middle_positions.max(dim=-1, keepdim=True).values
            cached_pad = torch.cat([prefix_pad_masks, middle_pad_masks], dim=1)
            suffix_attention, suffix_positions = self._prepare_memory_suffix_cache(
                cached_pad, max_positions, effective_history_mask
            )

            step = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
            x_t = noise
            current_time = torch.tensor(1.0, dtype=torch.float32, device=device)
            while current_time >= -step / 2:
                expanded_time = current_time.expand(batch_size)
                suffix_embs, _, _ = self.embed_suffix(
                    state,
                    effective_history_mask,
                    x_t.to(dtype),
                    expanded_time.to(dtype),
                )
                outputs, _ = self.qwen3_vl_with_expert.forward(
                    attention_mask=suffix_attention,
                    position_ids=suffix_positions,
                    past_key_values=cache,
                    inputs_embeds=[None, None, suffix_embs],
                    use_cache=False,
                )
                suffix_out = outputs[2][:, -self.config.chunk_size :].to(
                    torch.float32
                )
                velocity = self.action_out_proj(suffix_out)
                x_t = x_t + step * velocity
                current_time += step
        return x_t


@dataclass
class _InferenceHistoryEntry:
    images: tuple[Tensor, ...]
    state: Tensor
    pixel_chunks: tuple[Tensor, ...]
    grids: Tensor


class WSAMemoryPolicy(WSABasePolicy):
    config_class = WSAMemoryConfig
    name = WSA_MEMORY

    def __init__(self, config: WSAMemoryConfig):
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self.model = WSAMemoryModel(config)
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.model.to(config.device)
        self.reset()

    @classmethod
    def from_wsa_base_pretrained(
        cls,
        *,
        config: WSAMemoryConfig,
        pretrained_name_or_path: str | Path,
        local_files_only: bool = False,
    ) -> "WSAMemoryPolicy":
        policy = cls(config)
        policy.initialize_from_wsa_base(
            pretrained_name_or_path, local_files_only=local_files_only
        )
        return policy

    def initialize_from_wsa_base(
        self,
        pretrained_name_or_path: str | Path,
        *,
        local_files_only: bool = False,
    ) -> dict[str, Any]:
        source = str(pretrained_name_or_path)
        source_config = PreTrainedConfig.from_pretrained(
            source, local_files_only=local_files_only
        )
        if isinstance(source_config, WSAMemoryConfig):
            raise ValueError(
                "initialize_from_wsa_base expects a WSA-Base checkpoint, not WSA-Memory"
            )
        if not isinstance(source_config, WSABaseConfig):
            raise ValueError(
                "initialize_from_wsa_base requires a WSA-Base config, "
                f"got {type(source_config).__name__}"
            )
        self._loaded_pretrained_source_config = source_config
        if os.path.isdir(source):
            model_file = os.path.join(source, SAFETENSORS_SINGLE_FILE)
        else:
            model_file = hf_hub_download(
                repo_id=source,
                filename=SAFETENSORS_SINGLE_FILE,
                local_files_only=local_files_only,
            )
        if not os.path.isfile(model_file):
            raise FileNotFoundError(f"WSA-Base weights not found: {model_file}")

        from safetensors.torch import load_file

        source_state = load_file(model_file, device="cpu")
        target_state = self.state_dict()
        loaded_keys = {
            key
            for key, value in source_state.items()
            if key in target_state and target_state[key].shape == value.shape
        }
        incompatible = self.load_state_dict(source_state, strict=False)
        missing, unexpected, expected_base_missing = (
            WSABasePolicy.classify_model_loading_keys(
                self,
                list(incompatible.missing_keys),
                list(incompatible.unexpected_keys),
            )
        )
        memory_prefixes = (
            "model.temporal_visual_memory.",
            "model.state_relative_time_encoding",
            "model.state_time_gate",
        )
        expected_memory_missing = [
            key for key in missing if key.startswith(memory_prefixes)
        ]
        missing = [key for key in missing if key not in expected_memory_missing]
        if missing or unexpected:
            raise RuntimeError(
                "WSA-Base initialization has non-memory incompatibilities: "
                f"missing={missing[:20]}, unexpected={unexpected[:20]}"
            )

        loaded_numel = sum(target_state[key].numel() for key in loaded_keys)
        total_numel = sum(value.numel() for value in target_state.values())
        report = {
            "loaded_keys": len(loaded_keys),
            "loaded_numel": loaded_numel,
            "total_numel": total_numel,
            "loaded_ratio": loaded_numel / max(total_numel, 1),
            "expected_memory_missing": expected_memory_missing,
            "expected_external_missing": expected_base_missing,
            "unexpected_missing": missing,
            "unexpected_keys": unexpected,
        }
        logging.info(
            "Initialized WSA-Memory from %s: keys=%d, parameter_ratio=%.4f, "
            "new_memory_keys=%d, external_keys=%d",
            source,
            len(loaded_keys),
            report["loaded_ratio"],
            len(expected_memory_missing),
            len(expected_base_missing),
        )
        return report

    def classify_model_loading_keys(
        self, missing_keys: list[str], unexpected_keys: list[str]
    ) -> tuple[list[str], list[str], list[str]]:
        missing, unexpected, expected = super().classify_model_loading_keys(
            missing_keys, unexpected_keys
        )
        if missing or unexpected:
            raise RuntimeError(
                "Invalid standalone WSA-Memory checkpoint: "
                f"missing={missing[:20]}, unexpected={unexpected[:20]}"
            )
        return missing, unexpected, expected

    def reset(self, env_ids: Hashable | Sequence[Hashable] | None = None) -> None:
        if not hasattr(self, "_history_by_env") or env_ids is None:
            self._history_by_env: dict[Hashable, deque[_InferenceHistoryEntry]] = {}
            self._action_queues_by_env: dict[Hashable, deque[Tensor]] = {}
            self._text_memory_by_env: dict[Hashable, str] = {}
            self._action_queue = deque(maxlen=self.config.n_action_steps)
            self._queues = {ACTION: self._action_queue}
            return
        if isinstance(env_ids, Tensor):
            ids = env_ids.detach().cpu().flatten().tolist()
        elif isinstance(env_ids, Sequence) and not isinstance(env_ids, (str, bytes)):
            ids = list(env_ids)
        else:
            ids = [env_ids]
        for env_id in ids:
            self._history_by_env.pop(env_id, None)
            self._action_queues_by_env.pop(env_id, None)
            self._text_memory_by_env.pop(env_id, None)

    def set_text_memory(
        self,
        *,
        overall_task: str,
        scene: str | None = None,
        completed_subtasks: Sequence[str] | None = None,
        current_subtask: str | None = None,
        hand_used: int | str | None = None,
        object_rigidity: int | str | None = None,
        env_id: Hashable = 0,
    ) -> str:
        memory = prompt_memory_from_external(
            overall_task=overall_task,
            scene=scene,
            completed_subtasks=completed_subtasks,
            current_subtask=current_subtask,
            hand_used=hand_used,
            object_rigidity=object_rigidity,
            hand_map=self.config.hand_map,
            rigidity_map=self.config.rigidity_map,
            max_completed_subtasks=self.config.max_completed_subtasks,
        )
        prompt = build_memory_prompt(memory)
        self._text_memory_by_env[env_id] = prompt
        self._action_queues_by_env.pop(env_id, None)
        return prompt

    def _preprocess_images(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        images: list[Tensor] = []
        masks: list[Tensor] = []
        for image_index in range(3):
            image = batch[f"{OBS_IMAGES}.image{image_index}"]
            if image.ndim != 5:
                raise ValueError(
                    f"WSA-Memory image{image_index} must be [B,K,C,H,W], "
                    f"got {tuple(image.shape)}"
                )
            images.append(image)
            masks.append(batch[f"{OBS_IMAGES}.image{image_index}_mask"])
        return torch.stack(images, dim=1), torch.stack(masks, dim=1)

    def prepare_state(self, batch: dict[str, Tensor]) -> Tensor:
        state = batch[OBS_STATE]
        if state.ndim != 3:
            raise ValueError(
                f"WSA-Memory state must be [B,K,D], got {tuple(state.shape)}"
            )
        return pad_vector(state, self.config.max_state_dim)

    @staticmethod
    def _normalize_env_ids(
        env_ids: Hashable | Sequence[Hashable] | Tensor | None, batch_size: int
    ) -> list[Hashable]:
        if env_ids is None:
            return list(range(batch_size))
        if isinstance(env_ids, Tensor):
            env_ids = env_ids.detach().cpu().flatten().tolist()
        if isinstance(env_ids, Sequence) and not isinstance(env_ids, (str, bytes)):
            result = list(env_ids)
        else:
            result = [env_ids]
        if len(result) != batch_size:
            raise ValueError(
                f"env_ids length {len(result)} does not match batch size {batch_size}"
            )
        if len(set(result)) != len(result):
            raise ValueError("env_ids must be unique within one vector-environment batch")
        return result

    @staticmethod
    def _split_current_pixels(pixel_values: Tensor, grids: Tensor) -> list[list[Tensor]]:
        chunks_by_batch: list[list[Tensor]] = []
        for batch_index in range(pixel_values.shape[0]):
            counts = grids[batch_index].prod(dim=-1).to(torch.long).tolist()
            flat_pixels = pixel_values[batch_index].reshape(-1, pixel_values.shape[-1])
            if sum(counts) != flat_pixels.shape[0]:
                raise ValueError(
                    "Inference processor pixel/grid mismatch. Use "
                    "Qwen3VLMemoryProcessorTransformFn for deployment."
                )
            chunks_by_batch.append(list(torch.split(flat_pixels, counts, dim=0)))
        return chunks_by_batch

    def _prepare_inference_batch(
        self,
        batch: dict[str, Tensor],
        env_ids: Sequence[Hashable] | Tensor | None = None,
        *,
        update_env_ids: set[Hashable] | None = None,
    ) -> tuple[dict[str, Tensor], list[Hashable]]:
        if MEMORY_HISTORY_VALID_MASK in batch:
            prepared = dict(batch)
            prepared[MEMORY_PREPARED] = True
            batch_size = prepared[MEMORY_HISTORY_VALID_MASK].shape[0]
            return prepared, self._normalize_env_ids(env_ids, batch_size)

        state_input = batch[OBS_STATE]
        batch_size = state_input.shape[0]
        resolved_ids = self._normalize_env_ids(
            env_ids if env_ids is not None else batch.get(MEMORY_ENV_IDS), batch_size
        )
        current_state = state_input[:, -1] if state_input.ndim == 3 else state_input
        grids = batch[f"{OBS_PREFIX}image_grid_thw"].reshape(batch_size, -1, 3)
        pixels = batch[f"{OBS_PREFIX}pixel_values"].reshape(
            batch_size, -1, batch[f"{OBS_PREFIX}pixel_values"].shape[-1]
        )
        pixel_chunks = self._split_current_pixels(pixels, grids)

        current_images: list[Tensor] = []
        for image_index in range(3):
            image = batch[f"{OBS_IMAGES}.image{image_index}"]
            current_images.append(image[:, -1] if image.ndim == 5 else image)

        for batch_index, env_id in enumerate(resolved_ids):
            if update_env_ids is not None and env_id not in update_env_ids:
                continue
            history = self._history_by_env.setdefault(
                env_id, deque(maxlen=self.config.history_num_frames)
            )
            history.append(
                _InferenceHistoryEntry(
                    images=tuple(image[batch_index].detach().clone() for image in current_images),
                    state=current_state[batch_index].detach().clone(),
                    pixel_chunks=tuple(
                        chunk.detach().clone() for chunk in pixel_chunks[batch_index]
                    ),
                    grids=grids[batch_index].detach().clone(),
                )
            )

        prepared = dict(batch)
        state_histories: list[Tensor] = []
        valid_masks: list[Tensor] = []
        pixels_histories: list[Tensor] = []
        image_histories: list[list[Tensor]] = [[], [], []]
        for env_id in resolved_ids:
            actual = list(self._history_by_env[env_id])
            padding = self.config.history_num_frames - len(actual)
            padded = [actual[0]] * padding + actual
            valid_masks.append(
                torch.tensor(
                    [False] * padding + [True] * len(actual),
                    dtype=torch.bool,
                    device=actual[-1].state.device,
                )
            )
            state_histories.append(torch.stack([entry.state for entry in padded]))
            for view_index in range(3):
                image_histories[view_index].append(
                    torch.stack([entry.images[view_index] for entry in padded])
                )
            for entry in padded:
                if not torch.equal(entry.grids, actual[-1].grids):
                    raise ValueError(
                        "Camera token grids changed within an inference history; reset the env history"
                    )
            pixels_histories.append(
                torch.cat(
                    [
                        entry.pixel_chunks[view_index]
                        for view_index in range(3)
                        for entry in padded
                    ],
                    dim=0,
                )
            )

        prepared[OBS_STATE] = torch.stack(state_histories)
        prepared[MEMORY_HISTORY_VALID_MASK] = torch.stack(valid_masks)
        prepared[f"{OBS_PREFIX}pixel_values"] = torch.stack(pixels_histories)
        for view_index in range(3):
            prepared[f"{OBS_IMAGES}.image{view_index}"] = torch.stack(
                image_histories[view_index]
            )
        prepared[MEMORY_PREPARED] = True
        return prepared, resolved_ids

    def _predict_prepared(
        self, batch: dict[str, Tensor], *, decode_image: bool = False
    ) -> tuple[Tensor, None]:
        if decode_image:
            raise NotImplementedError(
                "WSA-Memory is action-only and does not decode future/reconstruction images"
            )
        images, image_masks = self._preprocess_images(batch)
        state = self.prepare_state(batch)
        history_mask = batch[MEMORY_HISTORY_VALID_MASK].to(
            device=state.device, dtype=torch.bool
        )
        actions = self.model.sample_actions(
            images,
            image_masks,
            batch[f"{OBS_PREFIX}pixel_values"],
            batch[f"{OBS_PREFIX}image_grid_thw"],
            batch[f"{OBS_PREFIX}input_ids"],
            batch[f"{OBS_PREFIX}attention_mask"],
            state,
            history_mask,
        )
        output_dim = self.config.output_features[ACTION].shape[0]
        return actions[:, :, :output_dim], None

    @torch.no_grad()
    def predict_action_chunk(
        self,
        batch: dict[str, Tensor],
        decode_image: bool = False,
        *,
        env_ids: Sequence[Hashable] | Tensor | None = None,
        rtc_processor: Any | None = None,
        **rtc_kwargs: Any,
    ) -> tuple[Tensor, None]:
        self.eval()
        if rtc_processor is not None or any(value is not None for value in rtc_kwargs.values()):
            raise NotImplementedError(
                "RTC is not implemented for WSA-Memory's K-state suffix; use synchronous inference"
            )
        prepared, _ = self._prepare_inference_batch(batch, env_ids=env_ids)
        return self._predict_prepared(prepared, decode_image=decode_image)

    @torch.no_grad()
    def select_action(
        self,
        batch: dict[str, Tensor],
        *,
        env_ids: Sequence[Hashable] | Tensor | None = None,
    ) -> Tensor:
        self.eval()
        batch_size = batch[OBS_STATE].shape[0]
        resolved_ids = self._normalize_env_ids(
            env_ids if env_ids is not None else batch.get(MEMORY_ENV_IDS),
            batch_size,
        )
        empty_ids = [
            env_id
            for env_id in resolved_ids
            if not self._action_queues_by_env.get(env_id)
        ]
        if not empty_ids:
            return torch.stack(
                [self._action_queues_by_env[env_id].popleft() for env_id in resolved_ids]
            )

        prepared, _ = self._prepare_inference_batch(
            batch,
            env_ids=resolved_ids,
            update_env_ids=set(empty_ids),
        )
        if empty_ids:
            chunks, _ = self._predict_prepared(prepared)
            for batch_index, env_id in enumerate(resolved_ids):
                if env_id not in empty_ids:
                    continue
                queue: deque[Tensor] = deque(maxlen=self.config.n_action_steps)
                queue.extend(chunks[batch_index, : self.config.n_action_steps])
                self._action_queues_by_env[env_id] = queue
        return torch.stack(
            [self._action_queues_by_env[env_id].popleft() for env_id in resolved_ids]
        )

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        images, image_masks = self._preprocess_images(batch)
        state = self.prepare_state(batch)
        actions = self.prepare_action(batch)
        history_mask = batch[MEMORY_HISTORY_VALID_MASK].to(
            device=state.device, dtype=torch.bool
        )
        losses = self.model.forward(
            images,
            image_masks,
            batch[f"{OBS_PREFIX}pixel_values"],
            batch[f"{OBS_PREFIX}image_grid_thw"],
            batch[f"{OBS_PREFIX}input_ids"],
            batch[f"{OBS_PREFIX}attention_mask"],
            state,
            history_mask,
            actions,
        )
        output_dim = self.config.output_features[ACTION].shape[0]
        losses = losses[:, :, :output_dim]
        valid_dim = output_dim
        if self.config.mask_action_dim_padding_loss:
            valid_dim = min(int(self.config.action_loss_valid_dim), output_dim)
        losses = losses[:, :, :valid_dim]

        temporal_mask = batch.get(MEMORY_ACTION_LOSS_MASK)
        if temporal_mask is None:
            temporal_mask = torch.ones(
                losses.shape[:2], dtype=torch.bool, device=losses.device
            )
        else:
            temporal_mask = temporal_mask.to(losses.device, dtype=torch.bool)
        sample_mask = batch.get(SAMPLE_ACTION_LOSS_MASK)
        if sample_mask is not None:
            sample_mask = sample_mask.to(losses.device)
            sample_mask = sample_mask.reshape(sample_mask.shape[0], -1)[:, 0] > 0.5
            temporal_mask = temporal_mask & sample_mask[:, None]

        expanded_mask = temporal_mask[..., None].expand_as(losses)
        numerator = (losses * expanded_mask).sum()
        denominator = expanded_mask.sum().clamp_min(1)
        loss_action = numerator / denominator

        per_dim_denominator = temporal_mask.sum().clamp_min(1)
        per_dim = (
            (losses * temporal_mask[..., None]).sum(dim=(0, 1))
            / per_dim_denominator
        )
        logs: dict[str, float] = {
            "loss": float(loss_action.detach().item()),
            "loss_action": float(loss_action.detach().item()),
            "valid_action_steps": float(temporal_mask.sum().detach().item()),
        }
        for dim_index in range(output_dim):
            logs[f"loss_action_dim{dim_index}"] = (
                float(per_dim[dim_index].detach().item())
                if dim_index < valid_dim
                else 0.0
            )
        # The total objective is intentionally and strictly action-only.
        return loss_action, logs
