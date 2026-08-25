from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Sequence

from transformers.models.qwen3_vl import Qwen3VLProcessor

import torch

from lerobot.utils.constants import (
    ACTION,
    SAMPLE_ACTION_LOSS_MASK,
    OBS_IMAGE, OBS_IMAGES, OBS_STATE, OBS_STR,
)
from lerobot.transforms.core import DataTransformFn, DataDict
from lerobot.transforms.subtask_context import (
    PromptMemory,
    SidecarIndex,
    apply_prompt_dropout,
    build_memory_prompt,
    deterministic_prompt_dropout_draws,
    prompt_memory_from_episode,
)


SUBTASK_CONTEXT_MODES = ("subtask", "task_only")
SUBTASK_PROMPT_COMPONENTS = "_subtask.prompt_components"
TRAINING_INSTRUCTION = "_training.instruction"


def _scalar_int(value: Any, field_name: str) -> int:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{field_name} must be scalar, got shape {tuple(value.shape)}")
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer, got {value!r}")
    return int(value)


@DataTransformFn.register_subclass("wsa_base_subtask_context")
@dataclass
class WSABaseSubtaskContextTransformFn(DataTransformFn):
    """Optionally replace the raw task with the three-line subtask prompt."""

    dataset_root: str | None = None
    text_context_mode: str = "task_only"
    max_completed_subtasks: int = 8
    task_instruction_dropout: float = 0.0
    completed_memory_dropout: float = 0.1
    current_subtask_block_dropout: float = 0.2
    allow_overlapping_subtasks: bool = False
    memory_seed: int = 0
    training: bool = True
    sidecar_index: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.text_context_mode not in SUBTASK_CONTEXT_MODES:
            raise ValueError(
                f"text_context_mode must be one of {SUBTASK_CONTEXT_MODES}, "
                f"got {self.text_context_mode!r}"
            )
        if self.max_completed_subtasks < 0:
            raise ValueError("max_completed_subtasks cannot be negative")
        for field_name in (
            "task_instruction_dropout",
            "completed_memory_dropout",
            "current_subtask_block_dropout",
        ):
            value = float(getattr(self, field_name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be in [0, 1]")
        if self.dataset_root is not None and self.text_context_mode == "subtask":
            self.sidecar_index = SidecarIndex.load(
                self.dataset_root,
                allow_overlaps=self.allow_overlapping_subtasks,
            )
            logging.info(
                "WSA-Base dataset=%s subtask_sidecar=%s",
                self.dataset_root,
                self.sidecar_index.source_path,
            )

    def __call__(self, data: DataDict) -> DataDict:
        raw_task = str(data.get("task") or "").strip()
        if not raw_task:
            raise ValueError("WSA-Base sample task must be a non-empty string")
        data.pop(SUBTASK_PROMPT_COMPONENTS, None)
        if self.text_context_mode == "task_only":
            data["task"] = raw_task
            return data

        if self.sidecar_index is None:
            raise RuntimeError(
                "WSA-Base subtask context was not bound to a dataset root. "
                "Bind the transform after dataset creation before reading samples."
            )

        episode_index = _scalar_int(data["episode_index"], "episode_index")
        frame_index = _scalar_int(data["frame_index"], "frame_index")
        episode = self.sidecar_index.require(episode_index)

        memory, _ = prompt_memory_from_episode(
            episode,
            frame_index,
            sample_task=raw_task,
            max_completed_subtasks=self.max_completed_subtasks,
        )
        if self.training:
            task_draw, completed_draw, current_draw = deterministic_prompt_dropout_draws(
                self.memory_seed, episode_index, frame_index
            )
            memory = apply_prompt_dropout(
                memory,
                task_dropped=task_draw < self.task_instruction_dropout,
                completed_dropped=completed_draw < self.completed_memory_dropout,
                current_dropped=current_draw < self.current_subtask_block_dropout,
            )

        data[SUBTASK_PROMPT_COMPONENTS] = memory
        data["task"] = build_memory_prompt(memory)
        return data


@DataTransformFn.register_subclass("wsa_base_processor")
@DataTransformFn.register_subclass("tbot_sa1_processor")
@DataTransformFn.register_subclass("magicbot_processor")
@DataTransformFn.register_subclass("cubev2_processor")
@dataclass
class Qwen3_VLProcessorTransformFn(DataTransformFn):
    pretrained_model_name_or_path: str = 'Qwen/Qwen3-VL-2B-Instruct'
    max_length: int = 48
    task_key: str = "task"
    padding_side: str = "right"
    padding: str = "max_length"
    truncation: bool = True
    emit_training_instruction: bool = False

    spatial_merge_size: int = 2

    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    image_token_id: int = 151655

    processor: Any = field(default=None, init=False, repr=False)
    _processor_source: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        # Delay loading so config parsing can override pretrained_model_name_or_path
        # before we touch the filesystem / Hugging Face cache.
        self.processor = None
        self._processor_source = None

    def _ensure_processor(self) -> None:
        if self.processor is not None and self._processor_source == self.pretrained_model_name_or_path:
            return

        self.processor = Qwen3VLProcessor.from_pretrained(self.pretrained_model_name_or_path)
        self._processor_source = self.pretrained_model_name_or_path
        self.vision_start_token_id = self.processor.vision_start_token_id
        self.vision_end_token_id = self.processor.vision_end_token_id
        self.image_token_id = self.processor.image_token_id

    def _token_ids(self, text: str) -> list[int]:
        encoded = self.processor.tokenizer(text, add_special_tokens=True, truncation=False)
        return list(encoded.input_ids)

    def _fit_subtask_prompt(self, memory: PromptMemory) -> str:
        """Fit context without ever truncating the current subtask."""
        candidate = memory
        prompt = build_memory_prompt(candidate)
        while len(self._token_ids(prompt)) > self.max_length and candidate.completed_subtasks:
            candidate = replace(
                candidate,
                completed_subtasks=candidate.completed_subtasks[1:],
                earlier_completed_omitted=True,
            )
            prompt = build_memory_prompt(candidate)

        while len(self._token_ids(prompt)) > self.max_length:
            task_ids = list(
                self.processor.tokenizer(
                    candidate.overall_task,
                    add_special_tokens=False,
                    truncation=False,
                ).input_ids
            )
            if len(task_ids) <= 1:
                break
            excess = len(self._token_ids(prompt)) - self.max_length
            keep = max(1, len(task_ids) - max(1, excess))
            shortened = self.processor.tokenizer.decode(
                task_ids[:keep], skip_special_tokens=True
            ).strip()
            candidate = replace(candidate, overall_task=shortened or "unknown")
            prompt = build_memory_prompt(candidate)

        if len(self._token_ids(prompt)) > self.max_length:
            raise ValueError(
                "WSA-Base subtask prompt cannot fit tokenizer max_length without "
                "truncating the current subtask. Increase dataset.tokenizer_max_length "
                "and policy.tokenizer_max_length."
            )
        return prompt

    def __call__(self, data: DataDict) -> DataDict:
        self._ensure_processor()
        input_ids = []
        attention_mask = []
        pixel_values = []
        image_grid_thw = []
        for i in range(3):
            k = f"{OBS_IMAGES}.image{i}"
            img_inputs = self.processor.image_processor(
                data[k][1],  # we only feed images at current time to vlm
                do_rescale=False,
            )
            num_img_token = torch.prod(img_inputs.image_grid_thw) // self.spatial_merge_size ** 2
            pixel_values.append(img_inputs.pixel_values)
            image_grid_thw.append(img_inputs.image_grid_thw)
            if data[f"{k}_mask"]:
                input_ids += [self.vision_start_token_id] + [self.image_token_id] * num_img_token + [self.vision_end_token_id]
                attention_mask += [1] * (num_img_token + 2)
                # attention_mask += [0] + [1] * num_img_token + [0]
            else:
                input_ids += [self.vision_start_token_id] + [self.image_token_id] * num_img_token + [self.vision_end_token_id]
                attention_mask += [0] * (num_img_token + 2)

        data[f"{OBS_STR}.pixel_values"] = torch.cat(pixel_values)
        data[f"{OBS_STR}.image_grid_thw"] = torch.cat(image_grid_thw)

        memory = data.get(SUBTASK_PROMPT_COMPONENTS)
        if memory is not None:
            if not isinstance(memory, PromptMemory):
                raise TypeError(f"{SUBTASK_PROMPT_COMPONENTS} must be PromptMemory")
            prompt = self._fit_subtask_prompt(memory)
            data[self.task_key] = prompt
        else:
            prompt = str(data[self.task_key])
        if self.emit_training_instruction:
            data[TRAINING_INSTRUCTION] = prompt
        if SUBTASK_PROMPT_COMPONENTS in data:
            unpadded = self.processor.tokenizer(
                prompt,
                add_special_tokens=True,
                padding=False,
                truncation=False,
            )
            if len(unpadded.input_ids) > self.max_length:
                raise ValueError(
                    "WSA-Base subtask prompt exceeds tokenizer max_length "
                    f"({len(unpadded.input_ids)} > {self.max_length}). Increase "
                    "dataset.tokenizer_max_length and policy.tokenizer_max_length; "
                    "the current subtask is never silently truncated."
                )

        lang_inputs = self.processor.tokenizer(
            prompt,
            max_length=self.max_length,
            padding_side=self.padding_side,
            padding=self.padding,
            truncation=self.truncation,
        )

        input_ids += lang_inputs.input_ids
        attention_mask += lang_inputs.attention_mask
        data[f"{OBS_STR}.input_ids"] = torch.tensor(input_ids)
        data[f"{OBS_STR}.attention_mask"] = torch.tensor(attention_mask)

        return data


@DataTransformFn.register_subclass("unify_wsa_base_inputs")
@DataTransformFn.register_subclass("unify_tbot_sa1_inputs")
@DataTransformFn.register_subclass("unify_magicbot_inputs")
@DataTransformFn.register_subclass("unify_cubev2_inputs")
@dataclass
class UnifyWSABaseInputsTransformFn(DataTransformFn):
    def __call__(self, data: DataDict) -> DataDict:
        default_action_loss_mask = 0.0 if data.get("robot_type") == "egodex_v" else 1.0
        training_instruction = data.get(TRAINING_INSTRUCTION)
        unified = {
            OBS_STATE: data[OBS_STATE],
            ACTION: data[ACTION],
            SAMPLE_ACTION_LOSS_MASK: data.get(
                SAMPLE_ACTION_LOSS_MASK,
                torch.tensor([default_action_loss_mask], dtype=torch.float32),
            ),
            f"{OBS_IMAGES}.image0": data[f"{OBS_IMAGES}.image0"],
            f"{OBS_IMAGES}.image1": data[f"{OBS_IMAGES}.image1"],
            f"{OBS_IMAGES}.image2": data[f"{OBS_IMAGES}.image2"],
            f"{OBS_IMAGES}.image0_mask": data[f"{OBS_IMAGES}.image0_mask"],
            f"{OBS_IMAGES}.image1_mask": data[f"{OBS_IMAGES}.image1_mask"],
            f"{OBS_IMAGES}.image2_mask": data[f"{OBS_IMAGES}.image2_mask"],
            f"{OBS_STR}.pixel_values": data[f"{OBS_STR}.pixel_values"],
            f"{OBS_STR}.image_grid_thw": data[f"{OBS_STR}.image_grid_thw"],
            # "task": data["task"],
            f"{OBS_STR}.input_ids": data[f"{OBS_STR}.input_ids"],
            f"{OBS_STR}.attention_mask": data[f"{OBS_STR}.attention_mask"],
        }
        if training_instruction is not None:
            unified[TRAINING_INSTRUCTION] = training_instruction
        return unified


def bind_wsa_base_transforms(
    transforms: Sequence[DataTransformFn],
    *,
    dataset: Any,
    dataset_config: Any,
) -> list[DataTransformFn]:
    """Bind per-repository sidecar state after a LeRobot dataset is created."""
    bound: list[DataTransformFn] = []
    found_context = False
    for transform in transforms:
        if isinstance(transform, WSABaseSubtaskContextTransformFn):
            found_context = True
            transform = replace(transform, dataset_root=str(Path(dataset.root)))
        bound.append(transform)
    if not found_context:
        # Old train_config.json files predate this transform. Insert a no-op
        # task_only instance (or the explicitly configured subtask instance)
        # so resume/evaluation remains backward compatible.
        context = WSABaseSubtaskContextTransformFn(
            dataset_root=str(Path(dataset.root)),
            text_context_mode=dataset_config.text_context_mode,
            max_completed_subtasks=dataset_config.max_completed_subtasks,
            task_instruction_dropout=dataset_config.task_instruction_dropout,
            completed_memory_dropout=dataset_config.completed_memory_dropout,
            current_subtask_block_dropout=dataset_config.current_subtask_block_dropout,
            allow_overlapping_subtasks=dataset_config.allow_overlapping_subtasks,
            memory_seed=dataset_config.memory_seed,
        )
        bound.insert(0, context)
    return bound


if __name__ == "__main__":
    sample = {
        f"{OBS_IMAGES}.image0": torch.rand((224, 224, 3)),
        f"{OBS_IMAGES}.image1": torch.rand((224, 224, 3)),
        f"{OBS_IMAGES}.image2": torch.rand((224, 224, 3)),
        "task": "This is a test sample.",
    }
    processor = Qwen3_VLProcessorTransformFn()
    output = processor(sample)
    print(output)
