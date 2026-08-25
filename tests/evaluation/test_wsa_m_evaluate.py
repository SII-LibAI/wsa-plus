from __future__ import annotations

import ast
import copy
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[2]
EVALUATE_PATH = ROOT / "evaluation" / "wsa_m" / "evaluate.py"


@dataclass
class FakeBaseConfig:
    type: str = "WSA_Base"
    tokenizer_max_length: int = 128


@dataclass
class FakeMemoryConfig(FakeBaseConfig):
    type: str = "wsa_memory"


@dataclass
class FakeBaseContext:
    text_context_mode: str = "task_only"
    max_completed_subtasks: int = 8
    task_instruction_dropout: float = 0.0
    completed_memory_dropout: float = 0.1
    current_subtask_block_dropout: float = 0.2
    allow_overlapping_subtasks: bool = False
    memory_seed: int = 0
    training: bool = True


@dataclass
class FakeMemoryContext:
    training: bool = True


@dataclass
class FakeProcessor:
    pretrained_model_name_or_path: str = "old-processor"
    max_length: int = 48
    emit_training_instruction: bool = True


@dataclass
class FakeMemoryProcessor(FakeProcessor):
    pass


@dataclass
class FakeImageAugment:
    enabled: bool = True


@dataclass
class FakeTransformGroup:
    inputs: list[Any] = field(default_factory=list)


class FakeBaseDataset:
    def __init__(self, repo_id: str, action_mode: str) -> None:
        self.repo_id = repo_id
        self.action_mode = action_mode
        self.qwen3_vl_processor_path = "base-processor"
        self.tokenizer_max_length = 192
        self.image_transforms = SimpleNamespace(enable=True)
        self.data_transforms = FakeTransformGroup(
            [FakeBaseContext(), FakeImageAugment(), FakeProcessor()]
        )
        self.text_context_mode = "task_only"
        self.max_completed_subtasks = 8
        self.task_instruction_dropout = 0.0
        self.completed_memory_dropout = 0.1
        self.current_subtask_block_dropout = 0.2
        self.allow_overlapping_subtasks = False
        self.memory_seed = 0


class FakeMemoryDataset(FakeBaseDataset):
    def __init__(self, repo_id: str, action_mode: str) -> None:
        super().__init__(repo_id, action_mode)
        self.qwen3_vl_processor_path = "memory-processor"
        self.data_transforms = FakeTransformGroup(
            [FakeMemoryContext(), FakeMemoryProcessor()]
        )


class FakeTrainPipelineConfig:
    def __init__(self, *, dataset: Any, policy: Any) -> None:
        self.dataset = dataset
        self.policy = policy
        self.num_workers = 0


class FakeLogger:
    def info(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def warning(self, *_args: Any, **_kwargs: Any) -> None:
        pass


def _load_functions() -> dict[str, Any]:
    tree = ast.parse(EVALUATE_PATH.read_text(encoding="utf-8"))
    names = {
        "policy_family",
        "policy_display_name",
        "_base_context_transform_for_eval",
        "configure_eval_transforms",
        "build_eval_config",
    }
    future_import = next(
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "__future__"
    )
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    module = ast.Module(body=[future_import, *functions], type_ignores=[])
    namespace = {
        "Any": Any,
        "POLICY_FAMILY_BASE": "wsa_base",
        "POLICY_FAMILY_MEMORY": "wsa_memory",
        "WSA_MEMORY": "wsa_memory",
        "WSABaseConfig": FakeBaseConfig,
        "WSAMemoryConfig": FakeMemoryConfig,
        "WSABaseDatasetConfig": FakeBaseDataset,
        "WSAMemoryDatasetConfig": FakeMemoryDataset,
        "SupportedDatasetConfig": object,
        "SupportedPolicyConfig": object,
        "WSABaseSubtaskContextTransformFn": FakeBaseContext,
        "WSAMemoryContextTransformFn": FakeMemoryContext,
        "Qwen3_VLProcessorTransformFn": FakeProcessor,
        "Pi05ImageAugmentFn": FakeImageAugment,
        "is_wsa_base": lambda value: value in {"WSA_Base", "wsa_base", "TBot_SA1"},
        "replace": replace,
        "copy": copy,
        "TrainPipelineConfig": FakeTrainPipelineConfig,
        "LOGGER": FakeLogger(),
    }
    exec(compile(ast.fix_missing_locations(module), str(EVALUATE_PATH), "exec"), namespace)
    return namespace


FUNCTIONS = _load_functions()


def _base_dataset(transforms: list[Any]) -> SimpleNamespace:
    return SimpleNamespace(
        image_transforms=SimpleNamespace(enable=True),
        data_transforms=FakeTransformGroup(transforms),
        text_context_mode="subtask",
        max_completed_subtasks=5,
        task_instruction_dropout=0.25,
        completed_memory_dropout=0.3,
        current_subtask_block_dropout=0.4,
        allow_overlapping_subtasks=False,
        memory_seed=17,
    )


def test_policy_family_checks_memory_before_base_inheritance() -> None:
    policy_family = FUNCTIONS["policy_family"]

    assert policy_family(FakeMemoryConfig()) == "wsa_memory"
    assert policy_family(FakeBaseConfig()) == "wsa_base"
    assert policy_family(FakeBaseConfig(type="TBot_SA1")) == "wsa_base"

    with pytest.raises(TypeError, match="support only WSA-Base and WSA-Memory"):
        policy_family(SimpleNamespace(type="other"))


@pytest.mark.parametrize(
    ("policy", "expected_dataset", "expected_max_length"),
    [
        (FakeBaseConfig(tokenizer_max_length=256), FakeBaseDataset, 192),
        (FakeMemoryConfig(tokenizer_max_length=384), FakeMemoryDataset, 384),
    ],
)
def test_fallback_eval_config_matches_checkpoint_family(
    policy: FakeBaseConfig,
    expected_dataset: type,
    expected_max_length: int,
) -> None:
    build_eval_config = FUNCTIONS["build_eval_config"]
    args = SimpleNamespace(
        action_mode="delta",
        num_workers=2,
        processor_path=None,
        qwen3_vl_path=None,
    )

    result = build_eval_config(
        args,
        Path("checkpoint"),
        policy,
        Path("repo_ids.txt"),
        SimpleNamespace(path=None, root=None),
        None,
    )

    assert type(result.dataset) is expected_dataset
    assert result.policy is policy
    assert result.dataset.action_mode == "delta"
    processor = next(
        item for item in result.dataset.data_transforms.inputs if isinstance(item, FakeProcessor)
    )
    assert processor.max_length == expected_max_length


def test_base_eval_disables_subtask_dropout_and_image_augmentation() -> None:
    configure = FUNCTIONS["configure_eval_transforms"]
    dataset = _base_dataset(
        [FakeBaseContext(training=True), FakeImageAugment(), FakeProcessor()]
    )

    configure(
        dataset,
        family="wsa_base",
        processor_path="new-processor",
        processor_max_length=256,
    )

    assert dataset.image_transforms.enable is False
    assert not any(isinstance(item, FakeImageAugment) for item in dataset.data_transforms.inputs)
    context = next(item for item in dataset.data_transforms.inputs if isinstance(item, FakeBaseContext))
    processor = next(item for item in dataset.data_transforms.inputs if isinstance(item, FakeProcessor))
    assert context.training is False
    assert processor.pretrained_model_name_or_path == "new-processor"
    assert processor.max_length == 256


def test_old_base_pipeline_gets_deterministic_context_transform() -> None:
    configure = FUNCTIONS["configure_eval_transforms"]
    dataset = _base_dataset([FakeProcessor()])

    configure(
        dataset,
        family="wsa_base",
        processor_path="processor",
        processor_max_length=192,
    )

    context = dataset.data_transforms.inputs[0]
    assert isinstance(context, FakeBaseContext)
    assert context.training is False
    assert context.text_context_mode == "subtask"
    assert context.memory_seed == 17


def test_memory_eval_disables_memory_dropout() -> None:
    configure = FUNCTIONS["configure_eval_transforms"]
    dataset = SimpleNamespace(
        image_transforms=SimpleNamespace(enable=True),
        data_transforms=FakeTransformGroup(
            [FakeMemoryContext(training=True), FakeMemoryProcessor()]
        ),
    )

    configure(
        dataset,
        family="wsa_memory",
        processor_path="memory-processor",
        processor_max_length=384,
    )

    context = next(item for item in dataset.data_transforms.inputs if isinstance(item, FakeMemoryContext))
    processor = next(item for item in dataset.data_transforms.inputs if isinstance(item, FakeMemoryProcessor))
    assert context.training is False
    assert processor.max_length == 384


def test_run_script_keeps_checkpoint_inference_steps_by_default() -> None:
    script = (ROOT / "evaluation" / "wsa_m" / "run.sh").read_text(encoding="utf-8")
    assert 'INFERENCE_STEPS="${INFERENCE_STEPS:-}"' in script
    assert "evaluation/wsa_m/run.sh" in script
