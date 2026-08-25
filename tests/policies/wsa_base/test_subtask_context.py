from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[3]
TRANSFORM_PATH = (
    ROOT / "src" / "lerobot" / "policies" / "WSA_Base" / "transform_wsa_base.py"
)


def _load_shared_context_module():
    path = ROOT / "src" / "lerobot" / "transforms" / "subtask_context.py"
    spec = importlib.util.spec_from_file_location("wsa_shared_subtask_context_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


shared = _load_shared_context_module()


def _load_base_context_call():
    tree = ast.parse(TRANSFORM_PATH.read_text(encoding="utf-8"))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "WSABaseSubtaskContextTransformFn"
    )
    call_node = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "__call__"
    )
    module = ast.Module(body=[call_node], type_ignores=[])
    namespace = {
        "DataDict": dict,
        "SUBTASK_PROMPT_COMPONENTS": "_subtask.prompt_components",
        "_scalar_int": lambda value, _name: int(value),
        "deterministic_prompt_dropout_draws": shared.deterministic_prompt_dropout_draws,
        "apply_prompt_dropout": shared.apply_prompt_dropout,
        "prompt_memory_from_episode": shared.prompt_memory_from_episode,
        "build_memory_prompt": shared.build_memory_prompt,
    }
    exec(compile(ast.fix_missing_locations(module), str(TRANSFORM_PATH), "exec"), namespace)
    return namespace["__call__"]


BASE_CONTEXT_CALL = _load_base_context_call()


def _episode():
    return shared.EpisodeMemory.from_json(
        {
            "episode_id": "episode_000000",
            "episode_index": 0,
            "task_instruction": "sidecar fallback",
            "subtasks": [
                {
                    "subtask_id": 0,
                    "subtask_text": "pick up the towel",
                    "start_frame": 0,
                    "end_frame": 9,
                },
                {
                    "subtask_id": 1,
                    "subtask_text": "wipe the table",
                    "start_frame": 10,
                    "end_frame": 20,
                },
            ],
        },
        allow_overlaps=False,
    )


def test_wsa_base_exposes_exactly_two_text_modes() -> None:
    tree = ast.parse(TRANSFORM_PATH.read_text(encoding="utf-8"))
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "SUBTASK_CONTEXT_MODES"
            for target in node.targets
        )
    )
    assert ast.literal_eval(assignment.value) == ("subtask", "task_only")


def test_task_only_keeps_the_original_dataset_instruction() -> None:
    context = SimpleNamespace(text_context_mode="task_only", sidecar_index=None)
    data = {
        "task": "  wipe the whole table  ",
        "_subtask.prompt_components": object(),
    }
    result = BASE_CONTEXT_CALL(context, data)
    assert result["task"] == "wipe the whole table"
    assert "_subtask.prompt_components" not in result


def test_subtask_mode_builds_only_the_three_requested_components() -> None:
    episode = _episode()
    context = SimpleNamespace(
        text_context_mode="subtask",
        sidecar_index=SimpleNamespace(require=lambda _episode_index: episode),
        max_completed_subtasks=8,
        training=False,
    )
    result = BASE_CONTEXT_CALL(
        context,
        {
            "task": "wipe the whole table",
            "episode_index": 0,
            "frame_index": 12,
        },
    )
    assert result["task"] == "\n".join(
        [
            "Task instruction: wipe the whole table.",
            "Completed subtasks: pick up the towel.",
            "Current subtask: wipe the table.",
        ]
    )


def test_three_components_can_be_dropped_independently() -> None:
    episode = _episode()
    context = SimpleNamespace(
        text_context_mode="subtask",
        sidecar_index=SimpleNamespace(require=lambda _episode_index: episode),
        max_completed_subtasks=8,
        training=True,
        memory_seed=0,
        task_instruction_dropout=1.0,
        completed_memory_dropout=1.0,
        current_subtask_block_dropout=1.0,
    )
    result = BASE_CONTEXT_CALL(
        context,
        {
            "task": "wipe the whole table",
            "episode_index": 0,
            "frame_index": 12,
        },
    )
    assert result["task"] == "\n".join(
        [
            "Task instruction: unknown.",
            "Completed subtasks: none.",
            "Current subtask: unknown.",
        ]
    )
