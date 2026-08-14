from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.WSA_Base.configuration_wsa_base import (
    WSABaseConfig,
    WSABaseDatasetConfig,
)
from lerobot.policies.WSA_Base.transform_wsa_base import Qwen3_VLProcessorTransformFn
from lerobot.policies.WSA_Memory.sidecar import (
    DEFAULT_RIGIDITY_MAP,
    HAND_MAP,
    normalize_enum_map,
)
from lerobot.policies.WSA_Memory.transform_wsa_memory import (
    CurrentStateDeltaActionTransformFn,
    Qwen3VLMemoryProcessorTransformFn,
    UnifyWSAMemoryInputsTransformFn,
    WSAMemoryContextTransformFn,
)
from lerobot.transforms.core import (
    ComposeFieldsTransform,
    NormalizeTransformFn,
    PadStateAndActionTransformFn,
    RemapImageKeyTransformFn,
    ResizeImagesWithPadFn,
    TransformGroup,
)


WSA_MEMORY = "wsa_memory"
TEXT_MEMORY_MODES = ("oracle", "task_only", "external")


@DatasetConfig.register_subclass(WSA_MEMORY)
@dataclass
class WSAMemoryDatasetConfig(WSABaseDatasetConfig):
    """LeRobot V3 transform pipeline used by :class:`WSAMemoryPolicy`.

    Dataset-specific values (root, fps and the policy's memory configuration)
    are bound by the dataset factory after constructing each individual repo.
    """

    _canonical_type: ClassVar[str] = WSA_MEMORY

    data_transforms: TransformGroup = field(
        default_factory=lambda: TransformGroup(
            inputs=[
                WSAMemoryContextTransformFn(),
                CurrentStateDeltaActionTransformFn(),
                ResizeImagesWithPadFn(
                    height=WSABaseDatasetConfig.height,
                    width=WSABaseDatasetConfig.width,
                ),
                RemapImageKeyTransformFn(),
                NormalizeTransformFn(),
                ComposeFieldsTransform(),
                PadStateAndActionTransformFn(
                    max_state_dim=WSABaseDatasetConfig.max_state_dim,
                    max_action_dim=WSABaseDatasetConfig.max_action_dim,
                ),
                Qwen3VLMemoryProcessorTransformFn(),
                UnifyWSAMemoryInputsTransformFn(),
            ],
            outputs=[],
        )
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        processors = [
            transform
            for transform in self.data_transforms.inputs
            if isinstance(transform, Qwen3_VLProcessorTransformFn)
        ]
        if len(processors) != 1 or not isinstance(
            processors[0], Qwen3VLMemoryProcessorTransformFn
        ):
            raise ValueError(
                "WSAMemoryDatasetConfig requires exactly one memory-aware Qwen3-VL processor"
            )


@PreTrainedConfig.register_subclass(WSA_MEMORY)
@dataclass
class WSAMemoryConfig(WSABaseConfig):
    _canonical_type: ClassVar[str] = WSA_MEMORY

    init_from_wsa_base: str | None = "zaleni/WSA-Base"

    history_num_frames: int = 6
    history_stride_seconds: float = 1.0
    future_generation_offset_seconds: float = 0.5
    visual_history_dropout: float = 0.3
    temporal_attention_every_n_blocks: int = 4

    text_memory_mode: str = "oracle"
    tokenizer_max_length: int = 192
    include_episode_summary: bool = False
    max_completed_subtasks: int = 8
    completed_memory_dropout: float = 0.10
    current_subtask_block_dropout: float = 0.20
    scene_dropout: float = 0.05

    hand_map: dict[int | str, str] = field(default_factory=lambda: dict(HAND_MAP))
    rigidity_map: dict[int | str, str] = field(
        default_factory=lambda: dict(DEFAULT_RIGIDITY_MAP)
    )

    lambda_gen: float = 0.0
    lambda_3d: float = 0.0
    allow_missing_sidecar: bool = False
    allow_overlapping_subtasks: bool = False
    memory_seed: int = 0

    def __post_init__(self) -> None:
        # WSABase validates its historical 3-frame layout.  WSA-Memory replaces
        # that layout with an fps-derived K-frame layout in the dataset factory,
        # while reusing every other WSABase validation.
        self.image_delta_indices = [-1, 0, 1]
        super().__post_init__()
        self.image_delta_indices = list(range(-(int(self.history_num_frames) - 1), 1))

        if self.history_num_frames <= 0:
            raise ValueError("history_num_frames must be positive")
        if self.history_stride_seconds <= 0:
            raise ValueError("history_stride_seconds must be positive")
        if self.future_generation_offset_seconds <= 0:
            raise ValueError("future_generation_offset_seconds must be positive")
        if self.temporal_attention_every_n_blocks <= 0:
            raise ValueError("temporal_attention_every_n_blocks must be positive")
        if self.tokenizer_max_length <= 0:
            raise ValueError("tokenizer_max_length must be positive")
        if self.max_completed_subtasks < 0:
            raise ValueError("max_completed_subtasks cannot be negative")
        if self.text_memory_mode not in TEXT_MEMORY_MODES:
            raise ValueError(
                f"text_memory_mode must be one of {TEXT_MEMORY_MODES}, "
                f"got {self.text_memory_mode!r}"
            )
        for field_name in (
            "visual_history_dropout",
            "completed_memory_dropout",
            "current_subtask_block_dropout",
            "scene_dropout",
        ):
            value = float(getattr(self, field_name))
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{field_name} must be in [0, 1)")
        if self.include_episode_summary:
            raise ValueError(
                "include_episode_summary=true is intentionally unsupported because summaries "
                "can leak future trajectory information"
            )
        if float(self.lambda_gen) < 0.0:
            raise ValueError("lambda_gen cannot be negative")
        if float(self.lambda_3d) != 0.0:
            raise ValueError("WSA-Memory does not support 3D supervision: lambda_3d must be 0")
        if self.attention_mask_mode != "default":
            raise NotImplementedError(
                "WSA-Memory currently supports attention_mask_mode='default' only. "
                "The WSABase causal/RTC paths assume a single state token and are not used silently."
            )

        self.hand_map = normalize_enum_map(self.hand_map, "hand_map")
        self.rigidity_map = normalize_enum_map(self.rigidity_map, "rigidity_map")
        self.memory_seed = int(self.memory_seed)

    def history_frame_offsets(self, fps: float) -> list[int]:
        """Return episode-local frame offsets derived from a dataset's real fps."""
        if fps <= 0:
            raise ValueError(f"Dataset fps must be positive, got {fps}")
        stride_frames = max(1, int(round(float(self.history_stride_seconds) * float(fps))))
        return [
            -(self.history_num_frames - 1 - index) * stride_frames
            for index in range(self.history_num_frames)
        ]

    def history_delta_timestamps(self, fps: float) -> list[float]:
        return [offset / float(fps) for offset in self.history_frame_offsets(fps)]

    def future_generation_frame_offset(self, fps: float) -> int:
        if fps <= 0:
            raise ValueError(f"Dataset fps must be positive, got {fps}")
        return max(1, int(round(float(self.future_generation_offset_seconds) * float(fps))))

    def image_frame_offsets(self, fps: float) -> list[int]:
        offsets = self.history_frame_offsets(fps)
        if float(self.lambda_gen) > 0.0:
            offsets.append(self.future_generation_frame_offset(fps))
        return offsets

    def image_history_delta_timestamps(self, fps: float) -> list[float]:
        """K conditioning frames plus one future target when generation is enabled."""
        return [offset / float(fps) for offset in self.image_frame_offsets(fps)]
