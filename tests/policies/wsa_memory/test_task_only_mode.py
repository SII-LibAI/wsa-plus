from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[3]


def _load_context_call():
    path = ROOT / "src" / "lerobot" / "policies" / "WSA_Memory" / "transform_wsa_memory.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "WSAMemoryContextTransformFn"
    )
    call_node = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "__call__"
    )
    module = ast.Module(body=[call_node], type_ignores=[])
    namespace = {
        "DataDict": dict,
        "MEMORY_ACTION_LOSS_MASK": "memory.action_loss_mask",
        "MEMORY_FUTURE_VALID_MASK": "memory.future_valid_mask",
        "MEMORY_HISTORY_VALID_MASK": "memory.history_valid_mask",
        "MEMORY_PROMPT_COMPONENTS": "_wsa_memory.prompt_components",
        "_scalar_int": lambda value, _name: int(value),
        "torch": torch,
    }
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["__call__"]


CONTEXT_CALL = _load_context_call()


def test_task_only_bypasses_memory_prompt_template() -> None:
    context = SimpleNamespace(
        text_memory_mode="task_only",
        training=True,
        action_horizon=3,
        action_padding_keys=(),
        _history_valid_mask=lambda _data: torch.tensor([True, True]),
        _future_valid_mask=lambda _data: torch.tensor(True),
    )
    data = {
        "episode_index": 1,
        "frame_index": 2,
        "task": "  fold the shirt  ",
        "_wsa_memory.prompt_components": object(),
    }

    result = CONTEXT_CALL(context, data)

    assert result["task"] == "fold the shirt"
    assert "_wsa_memory.prompt_components" not in result
    assert result["memory.action_loss_mask"].tolist() == [True, True, True]
