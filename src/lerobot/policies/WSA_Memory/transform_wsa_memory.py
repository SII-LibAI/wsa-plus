from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from lerobot.policies.WSA_Base.transform_wsa_base import (
    TRAINING_INSTRUCTION,
    Qwen3_VLProcessorTransformFn,
)
from lerobot.policies.WSA_Memory.sidecar import (
    PromptMemory,
    SidecarIndex,
    apply_prompt_dropout,
    build_memory_prompt,
    deterministic_prompt_dropout_draws,
    prompt_memory_from_episode,
    prompt_memory_from_external,
)
from lerobot.transforms.core import DataDict, DataTransformFn, DeltaActionTransformFn
from lerobot.transforms.constants import get_feature_mapping
from lerobot.utils.constants import (
    ACTION,
    OBS_IMAGES,
    OBS_PREFIX,
    OBS_STATE,
    OBS_STR,
    SAMPLE_ACTION_LOSS_MASK,
)


MEMORY_HISTORY_VALID_MASK = "memory.history_valid_mask"
MEMORY_FUTURE_VALID_MASK = "memory.future_valid_mask"
MEMORY_ACTION_LOSS_MASK = "memory.action_loss_mask"
MEMORY_PROMPT_COMPONENTS = "_wsa_memory.prompt_components"
MEMORY_PREPARED = "_wsa_memory.prepared"
MEMORY_ENV_IDS = "memory.env_ids"


def _scalar_int(value: Any, field_name: str) -> int:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{field_name} must be scalar, got shape {tuple(value.shape)}")
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer, got {value!r}")
    return int(value)


@DataTransformFn.register_subclass("wsa_memory_context")
@dataclass
class WSAMemoryContextTransformFn(DataTransformFn):
    dataset_root: str | None = None
    text_memory_mode: str = "oracle"
    history_num_frames: int = 6
    max_completed_subtasks: int = 8
    task_instruction_dropout: float = 0.0
    completed_memory_dropout: float = 0.10
    current_subtask_block_dropout: float = 0.20
    # Deprecated decode-only fields kept so existing checkpoints remain loadable.
    scene_dropout: float = 0.05
    allow_overlapping_subtasks: bool = False
    memory_seed: int = 0
    training: bool = True
    hand_map: Mapping[int | str, str] = field(default_factory=dict)
    rigidity_map: Mapping[int | str, str] = field(default_factory=dict)
    action_horizon: int = 0
    history_padding_keys: tuple[str, ...] = ()
    future_padding_keys: tuple[str, ...] = ()
    future_generation_enabled: bool = False
    action_padding_keys: tuple[str, ...] = ()
    # Runtime-only cache.  Keep the annotation as Any so draccus can decode the
    # serialized null in train_config.json; __post_init__ always rebuilds the
    # strongly typed SidecarIndex from dataset_root when one is needed.
    sidecar_index: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.dataset_root is not None and self.text_memory_mode == "oracle":
            self.sidecar_index = SidecarIndex.load(
                self.dataset_root,
                allow_overlaps=self.allow_overlapping_subtasks,
            )
            logging.info(
                "WSA-Memory dataset=%s subtask_sidecar=%s",
                self.dataset_root,
                self.sidecar_index.source_path,
            )

    def _history_valid_mask(self, data: DataDict) -> torch.Tensor:
        masks: list[torch.Tensor] = []
        for key in self.history_padding_keys:
            if key in data:
                candidate = torch.as_tensor(data[key], dtype=torch.bool).flatten()
                masks.append(~candidate)
        if not masks:
            expected = ", ".join(self.history_padding_keys) or "<no keys were bound>"
            raise KeyError(
                "WSA-Memory could not find a LeRobot history padding mask. "
                f"Expected one of: {expected}"
            )
        mask = masks[0]
        for candidate in masks[1:]:
            if not torch.equal(candidate, mask):
                raise ValueError(
                    "Image/state history padding masks are not aligned for this sample"
                )
        if mask.numel() != self.history_num_frames:
            raise ValueError(
                f"Expected {self.history_num_frames} history positions, got {mask.numel()}"
            )
        if not bool(mask[-1]):
            raise ValueError("The current history position must always be valid")
        return mask

    def _future_valid_mask(self, data: DataDict) -> torch.Tensor:
        if not self.future_generation_enabled:
            return torch.tensor(True, dtype=torch.bool)
        masks = [
            torch.as_tensor(data[key], dtype=torch.bool).flatten()
            for key in self.future_padding_keys
            if key in data
        ]
        if not masks:
            expected = ", ".join(self.future_padding_keys) or "<no keys were bound>"
            raise KeyError(
                "WSA-Memory future generation could not find an image padding mask. "
                f"Expected one of: {expected}"
            )
        mask = masks[0]
        for candidate in masks[1:]:
            if not torch.equal(candidate, mask):
                raise ValueError("Future image padding masks are not aligned")
        expected_length = self.history_num_frames + 1
        if mask.numel() != expected_length:
            raise ValueError(
                f"Expected {expected_length} image positions (K history + future), "
                f"got {mask.numel()}"
            )
        return (~mask[-1]).to(dtype=torch.bool)

    def _build_memory(
        self,
        data: DataDict,
        episode_index: int,
        frame_index: int,
    ) -> PromptMemory:
        sample_task = data.get("task")
        if self.text_memory_mode == "oracle":
            if self.sidecar_index is None:
                raise RuntimeError(
                    "WSA-Memory oracle context was not bound to a dataset root. "
                    "Bind the transform after dataset creation before reading samples."
                )
            episode = self.sidecar_index.require(episode_index)
            memory, _ = prompt_memory_from_episode(
                episode,
                frame_index,
                sample_task=sample_task,
                max_completed_subtasks=self.max_completed_subtasks,
            )
            return memory

        if self.text_memory_mode == "external":
            completed = data.get("memory.completed_subtasks", ())
            if isinstance(completed, str):
                completed = (completed,)
            memory = prompt_memory_from_external(
                overall_task=str(data.get("memory.overall_task", sample_task or "unknown")),
                completed_subtasks=completed,
                current_subtask=data.get("memory.current_subtask"),
                max_completed_subtasks=self.max_completed_subtasks,
            )
            return memory

        raise ValueError(
            f"Unsupported WSA-Memory text_memory_mode={self.text_memory_mode!r}"
        )

    def __call__(self, data: DataDict) -> DataDict:
        episode_index = _scalar_int(data["episode_index"], "episode_index")
        frame_index = _scalar_int(data["frame_index"], "frame_index")
        history_valid_mask = self._history_valid_mask(data)
        future_valid_mask = self._future_valid_mask(data)
        # task_only means exactly the original dataset instruction. It must not
        # be expanded into the three-line subtask prompt.
        memory = None
        if self.text_memory_mode != "task_only":
            memory = self._build_memory(data, episode_index, frame_index)

        if self.training and memory is not None:
            task_draw, completed_draw, current_draw = deterministic_prompt_dropout_draws(
                self.memory_seed, episode_index, frame_index
            )
            memory = apply_prompt_dropout(
                memory,
                task_dropped=task_draw < self.task_instruction_dropout,
                completed_dropped=completed_draw < self.completed_memory_dropout,
                current_dropped=current_draw < self.current_subtask_block_dropout,
            )

        if self.action_horizon <= 0:
            raise ValueError("WSA-Memory action horizon was not bound by the dataset")
        action_padding_masks = [
            torch.as_tensor(data[key], dtype=torch.bool).flatten()
            for key in self.action_padding_keys
            if key in data
        ]
        if action_padding_masks:
            action_is_pad = action_padding_masks[0]
            for candidate in action_padding_masks[1:]:
                if not torch.equal(candidate, action_is_pad):
                    raise ValueError("LeRobot action padding masks are not aligned")
        else:
            action_is_pad = torch.zeros(self.action_horizon, dtype=torch.bool)
        if action_is_pad.numel() != self.action_horizon:
            raise ValueError(
                "LeRobot action padding mask does not match the configured action horizon: "
                f"mask={action_is_pad.numel()}, horizon={self.action_horizon}"
            )
        action_loss_mask = ~action_is_pad

        data[MEMORY_HISTORY_VALID_MASK] = history_valid_mask
        data[MEMORY_FUTURE_VALID_MASK] = future_valid_mask
        data[MEMORY_ACTION_LOSS_MASK] = action_loss_mask.to(dtype=torch.bool)
        if memory is None:
            data.pop(MEMORY_PROMPT_COMPONENTS, None)
            data["task"] = str(data.get("task") or "unknown").strip() or "unknown"
        else:
            data[MEMORY_PROMPT_COMPONENTS] = memory
            data["task"] = build_memory_prompt(memory)
        return data


@DataTransformFn.register_subclass("wsa_memory_delta_action")
@dataclass
class CurrentStateDeltaActionTransformFn(DeltaActionTransformFn):
    """WSABase delta action transform using only state at the current history slot."""

    def __call__(self, data: DataDict) -> DataDict:
        state_keys = self.mapping[OBS_STATE]
        state_list, _ = self._align_for_cat([data[key] for key in state_keys])
        state_history = torch.cat(state_list, dim=-1)
        current_state = state_history[-1] if state_history.ndim >= 2 else state_history

        action_keys = self.mapping[ACTION]
        action_list, sizes = self._align_for_cat([data[key] for key in action_keys])
        action = torch.cat(action_list, dim=-1)
        mask = torch.as_tensor(
            self.mask if self.mask is not None else [True] * current_state.shape[-1],
            dtype=torch.bool,
            device=current_state.device,
        )
        action = action - torch.where(mask, current_state, 0)[None]
        start = 0
        for key, size in zip(action_keys, sizes, strict=True):
            data[key] = action[..., start : start + size]
            start += size
        return data


@DataTransformFn.register_subclass("wsa_memory_processor")
@dataclass
class Qwen3VLMemoryProcessorTransformFn(Qwen3_VLProcessorTransformFn):
    max_length: int = 192
    history_num_frames: int = 6
    future_generation_enabled: bool = False

    def _token_ids(self, text: str) -> list[int]:
        encoded = self.processor.tokenizer(text, add_special_tokens=True, truncation=False)
        return list(encoded.input_ids)

    def _fit_memory_prompt(self, memory: PromptMemory) -> str:
        candidate = memory
        prompt = build_memory_prompt(candidate)
        while len(self._token_ids(prompt)) > self.max_length and candidate.completed_subtasks:
            candidate = replace(
                candidate,
                completed_subtasks=candidate.completed_subtasks[1:],
                earlier_completed_omitted=True,
            )
            prompt = build_memory_prompt(candidate)

        # If needed, shorten the task instruction, never the current subtask.
        for field_name in ("overall_task",):
            while len(self._token_ids(prompt)) > self.max_length:
                value = getattr(candidate, field_name)
                value_ids = list(
                    self.processor.tokenizer(
                        value, add_special_tokens=False, truncation=False
                    ).input_ids
                )
                if len(value_ids) <= 1:
                    break
                excess = len(self._token_ids(prompt)) - self.max_length
                keep = max(1, len(value_ids) - max(1, excess))
                shortened = self.processor.tokenizer.decode(
                    value_ids[:keep], skip_special_tokens=True
                ).strip()
                candidate = replace(candidate, **{field_name: shortened or "unknown"})
                prompt = build_memory_prompt(candidate)

        if len(self._token_ids(prompt)) > self.max_length:
            raise ValueError(
                "WSA-Memory prompt cannot fit tokenizer_max_length without truncating "
                "the current-subtask block. Increase tokenizer_max_length."
            )
        return prompt

    def __call__(self, data: DataDict) -> DataDict:
        self._ensure_processor()
        input_ids: list[int] = []
        attention_mask: list[int] = []
        pixel_values: list[torch.Tensor] = []
        current_image_grids: list[torch.Tensor] = []
        has_bound_history = MEMORY_HISTORY_VALID_MASK in data

        for image_index in range(3):
            key = f"{OBS_IMAGES}.image{image_index}"
            image = data[key]
            if image.ndim == 3:
                frames = image.unsqueeze(0)
            elif image.ndim == 4:
                frames = image if has_bound_history else image[-1:]
            else:
                raise ValueError(
                    f"{key} must have shape [C,H,W] or [K,C,H,W], got {tuple(image.shape)}"
                )
            expected_frames = self.history_num_frames + int(self.future_generation_enabled)
            if has_bound_history and frames.shape[0] != expected_frames:
                raise ValueError(
                    f"{key} expected {expected_frames} frames, got {frames.shape[0]}"
                )
            conditioning_frames = frames[: self.history_num_frames] if has_bound_history else frames

            last_grid = None
            # The optional final frame is a Cosmos supervision target. It must
            # never enter the Qwen visual-memory prefix.
            for frame in conditioning_frames:
                image_inputs = self.processor.image_processor(frame, do_rescale=False)
                pixel_values.append(image_inputs.pixel_values)
                last_grid = image_inputs.image_grid_thw
            assert last_grid is not None
            current_image_grids.append(last_grid)
            num_image_tokens = int(
                torch.prod(last_grid).item() // (self.spatial_merge_size**2)
            )
            camera_valid = bool(torch.as_tensor(data[f"{key}_mask"]).item())
            input_ids.extend(
                [self.vision_start_token_id]
                + [self.image_token_id] * num_image_tokens
                + [self.vision_end_token_id]
            )
            attention_mask.extend(
                [1 if camera_valid else 0] * (num_image_tokens + 2)
            )

        data[f"{OBS_STR}.pixel_values"] = torch.cat(pixel_values, dim=0)
        data[f"{OBS_STR}.image_grid_thw"] = torch.cat(current_image_grids, dim=0)

        memory = data.get(MEMORY_PROMPT_COMPONENTS)
        if memory is not None:
            if not isinstance(memory, PromptMemory):
                raise TypeError(f"{MEMORY_PROMPT_COMPONENTS} must be PromptMemory")
            prompt = self._fit_memory_prompt(memory)
            data[self.task_key] = prompt
        else:
            prompt = str(data[self.task_key])
            if len(self._token_ids(prompt)) > self.max_length:
                raise ValueError(
                    "External WSA-Memory prompt exceeds tokenizer_max_length. Build it with "
                    "build_memory_prompt/set_text_memory so the current block is preserved."
                )

        if self.emit_training_instruction:
            data[TRAINING_INSTRUCTION] = prompt

        language_inputs = self.processor.tokenizer(
            prompt,
            max_length=self.max_length,
            padding_side=self.padding_side,
            padding=self.padding,
            truncation=False,
        )
        input_ids.extend(language_inputs.input_ids)
        attention_mask.extend(language_inputs.attention_mask)
        data[f"{OBS_STR}.input_ids"] = torch.tensor(input_ids, dtype=torch.long)
        data[f"{OBS_STR}.attention_mask"] = torch.tensor(
            attention_mask, dtype=torch.long
        )
        return data


@DataTransformFn.register_subclass("unify_wsa_memory_inputs")
@dataclass
class UnifyWSAMemoryInputsTransformFn(DataTransformFn):
    def __call__(self, data: DataDict) -> DataDict:
        default_sample_mask = 0.0 if data.get("robot_type") == "egodex_v" else 1.0
        training_instruction = data.get(TRAINING_INSTRUCTION)
        unified = {
            OBS_STATE: data[OBS_STATE],
            ACTION: data[ACTION],
            SAMPLE_ACTION_LOSS_MASK: data.get(
                SAMPLE_ACTION_LOSS_MASK,
                torch.tensor([default_sample_mask], dtype=torch.float32),
            ),
            MEMORY_HISTORY_VALID_MASK: data[MEMORY_HISTORY_VALID_MASK],
            MEMORY_FUTURE_VALID_MASK: data[MEMORY_FUTURE_VALID_MASK],
            MEMORY_ACTION_LOSS_MASK: data[MEMORY_ACTION_LOSS_MASK],
            f"{OBS_IMAGES}.image0": data[f"{OBS_IMAGES}.image0"],
            f"{OBS_IMAGES}.image1": data[f"{OBS_IMAGES}.image1"],
            f"{OBS_IMAGES}.image2": data[f"{OBS_IMAGES}.image2"],
            f"{OBS_IMAGES}.image0_mask": data[f"{OBS_IMAGES}.image0_mask"],
            f"{OBS_IMAGES}.image1_mask": data[f"{OBS_IMAGES}.image1_mask"],
            f"{OBS_IMAGES}.image2_mask": data[f"{OBS_IMAGES}.image2_mask"],
            f"{OBS_STR}.pixel_values": data[f"{OBS_STR}.pixel_values"],
            f"{OBS_STR}.image_grid_thw": data[f"{OBS_STR}.image_grid_thw"],
            f"{OBS_STR}.input_ids": data[f"{OBS_STR}.input_ids"],
            f"{OBS_STR}.attention_mask": data[f"{OBS_STR}.attention_mask"],
        }
        if training_instruction is not None:
            unified[TRAINING_INSTRUCTION] = training_instruction
        return unified


def bind_wsa_memory_transforms(
    transforms: Sequence[DataTransformFn],
    *,
    dataset: Any,
    policy_config: Any,
) -> list[DataTransformFn]:
    """Bind one transform copy to one repo root and load its sidecar exactly once."""
    if dataset.delta_indices is None:
        raise ValueError("WSA-Memory requires LeRobot delta timestamps for history/action sampling")
    expected_history_offsets = policy_config.history_frame_offsets(dataset.fps)
    expected_image_offsets = policy_config.image_frame_offsets(dataset.fps)
    expected_action_offsets = list(policy_config.action_delta_indices)
    action_feature_keys = set(
        get_feature_mapping(dataset.meta.robot_type, dataset.meta.features)[ACTION]
    )
    action_feature_keys.add(ACTION)
    padding_keys = tuple(
        f"{key}_is_pad"
        for key, offsets in dataset.delta_indices.items()
        if list(offsets) == expected_history_offsets and key not in action_feature_keys
    )
    future_padding_keys = tuple(
        f"{key}_is_pad"
        for key, offsets in dataset.delta_indices.items()
        if list(offsets) == expected_image_offsets and key not in action_feature_keys
    )
    action_padding_keys = tuple(
        f"{key}_is_pad"
        for key, offsets in dataset.delta_indices.items()
        if list(offsets) == expected_action_offsets and key in action_feature_keys
    )
    bound: list[DataTransformFn] = []
    found_context = found_processor = False
    for transform in transforms:
        if isinstance(transform, WSAMemoryContextTransformFn):
            found_context = True
            transform = replace(
                transform,
                dataset_root=str(Path(dataset.root)),
                text_memory_mode=policy_config.text_memory_mode,
                history_num_frames=policy_config.history_num_frames,
                max_completed_subtasks=policy_config.max_completed_subtasks,
                task_instruction_dropout=policy_config.task_instruction_dropout,
                completed_memory_dropout=policy_config.completed_memory_dropout,
                current_subtask_block_dropout=policy_config.current_subtask_block_dropout,
                allow_overlapping_subtasks=policy_config.allow_overlapping_subtasks,
                memory_seed=policy_config.memory_seed,
                action_horizon=policy_config.chunk_size,
                history_padding_keys=padding_keys,
                future_padding_keys=future_padding_keys,
                future_generation_enabled=float(policy_config.lambda_gen) > 0.0,
                action_padding_keys=action_padding_keys,
            )
        elif isinstance(transform, Qwen3VLMemoryProcessorTransformFn):
            found_processor = True
            transform = replace(
                transform,
                max_length=policy_config.tokenizer_max_length,
                history_num_frames=policy_config.history_num_frames,
                future_generation_enabled=float(policy_config.lambda_gen) > 0.0,
            )
        bound.append(transform)
    if not found_context or not found_processor:
        raise ValueError(
            "WSA-Memory dataset pipeline is missing its context or processor transform"
        )
    return bound
