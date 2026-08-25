from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[3]


def _load_config_timeline_methods():
    path = ROOT / "src" / "lerobot" / "policies" / "WSA_Memory" / "configuration_wsa_memory.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "WSAMemoryConfig"
    )
    method_names = {
        "history_frame_offsets",
        "history_delta_timestamps",
        "future_generation_frame_offset",
        "image_frame_offsets",
        "image_history_delta_timestamps",
    }
    methods = [
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name in method_names
    ]
    module = ast.Module(body=methods, type_ignores=[])
    namespace: dict[str, object] = {}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return {name: namespace[name] for name in method_names}


METHODS = _load_config_timeline_methods()


def _config(*, lambda_gen: float):
    config = SimpleNamespace(
        history_num_frames=6,
        history_stride_seconds=1.0,
        future_generation_offset_seconds=0.5,
        lambda_gen=lambda_gen,
    )
    for name, method in METHODS.items():
        setattr(config, name, method.__get__(config))
    return config


def test_action_only_timeline_keeps_k_frames() -> None:
    config = _config(lambda_gen=0.0)
    assert config.history_frame_offsets(30) == [-150, -120, -90, -60, -30, 0]
    assert config.image_frame_offsets(30) == [-150, -120, -90, -60, -30, 0]


def test_generation_timeline_appends_future_target() -> None:
    config = _config(lambda_gen=0.01)
    assert config.image_frame_offsets(30) == [-150, -120, -90, -60, -30, 0, 15]
    assert config.image_history_delta_timestamps(30) == [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 0.5]


def test_future_offset_is_at_least_one_frame() -> None:
    config = _config(lambda_gen=0.01)
    config.future_generation_offset_seconds = 0.01
    assert config.future_generation_frame_offset(30) == 1
