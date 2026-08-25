from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_sidecar_module():
    path = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "lerobot"
        / "policies"
        / "WSA_Memory"
        / "sidecar.py"
    )
    spec = importlib.util.spec_from_file_location("wsa_memory_sidecar_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sidecar = _load_sidecar_module()


def _episode() -> dict:
    return {
        "episode_id": "episode_000000",
        "episode_index": 0,
        "task_instruction": (
            "Grab the towel and then wipe the table and put the towel back."
        ),
        "sample_interval_sec": 2.0,
        "episode_summary": "This future-looking value must never enter the prompt.",
        "subtasks": [
            {
                "subtask_id": 0,
                "subtask_text": "move the gripper close to the towel",
                "start_frame": 0,
                "end_frame": 120,
                "confidence": 0.95,
                "hand_used": 1,
                "object_rigidity": 1,
            },
            {
                "subtask_id": 1,
                "subtask_text": "grasp the towel",
                "start_frame": 121,
                "end_frame": 240,
                "confidence": 0.98,
                "hand_used": 1,
                "object_rigidity": 1,
            },
            {
                "subtask_id": 2,
                "subtask_text": "wipe the table",
                "start_frame": 250,
                "end_frame": 300,
                "confidence": None,
                "hand_used": 2,
                "object_rigidity": 1,
            },
        ],
        "scene": "home",
    }


def _write_sidecar(tmp_path: Path, payload: object) -> Path:
    meta = tmp_path / "meta"
    meta.mkdir()
    path = meta / "sidecar.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_episode_index_alignment_and_prompt_are_leak_free(tmp_path: Path) -> None:
    _write_sidecar(tmp_path, [_episode()])
    index = sidecar.SidecarIndex.load(tmp_path)
    episode = index.require(0)

    memory, current = sidecar.prompt_memory_from_episode(
        episode,
        121,
        sample_task=None,
        max_completed_subtasks=8,
    )

    assert current.subtask_id == 1
    assert memory.completed_subtasks == ("move the gripper close to the towel",)
    assert sidecar.build_memory_prompt(memory) == "\n".join(
        [
            "Task instruction: Grab the towel and then wipe the table and put the towel back.",
            "Completed subtasks: move the gripper close to the towel.",
            "Current subtask: grasp the towel.",
        ]
    )
    assert "future-looking" not in sidecar.build_memory_prompt(memory)
    assert "confidence" not in sidecar.build_memory_prompt(memory)


def test_gap_does_not_reuse_the_previous_subtask() -> None:
    episode = sidecar.EpisodeMemory.from_json(_episode(), allow_overlaps=False)
    memory, current = sidecar.prompt_memory_from_episode(
        episode,
        245,
        sample_task="Dataset task has priority.",
        max_completed_subtasks=8,
    )
    assert current is None
    assert memory.overall_task == "Dataset task has priority."
    assert memory.current_subtask == "unknown"


def test_sidecar_validation_fails_fast(tmp_path: Path) -> None:
    overlapping = _episode()
    overlapping["subtasks"][1]["start_frame"] = 120
    _write_sidecar(tmp_path, [overlapping])
    with pytest.raises(ValueError, match="overlapping closed intervals"):
        sidecar.SidecarIndex.load(tmp_path)


def test_missing_sidecar_fails_fast(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Subtask sidecar is missing"):
        sidecar.SidecarIndex.load(tmp_path)


def test_missing_episode_fails_fast(tmp_path: Path) -> None:
    _write_sidecar(tmp_path, [_episode()])
    index = sidecar.SidecarIndex.load(tmp_path)
    with pytest.raises(KeyError, match="episode_index=1"):
        index.require(1)


def test_legacy_fine_grained_fields_are_ignored() -> None:
    episode = sidecar.EpisodeMemory.from_json(_episode(), allow_overlaps=False)
    assert not hasattr(episode, "scene")
    assert not hasattr(episode.subtasks[0], "hand_used")
    assert not hasattr(episode.subtasks[0], "object_rigidity")
    assert not hasattr(episode.subtasks[0], "confidence")


def test_three_prompt_components_have_independent_dropout() -> None:
    memory = sidecar.PromptMemory(
        overall_task="wipe the table",
        completed_subtasks=("pick up the towel",),
        current_subtask="wipe the surface",
    )
    dropped = sidecar.apply_prompt_dropout(
        memory,
        task_dropped=True,
        completed_dropped=True,
        current_dropped=True,
    )
    assert sidecar.build_memory_prompt(dropped) == "\n".join(
        [
            "Task instruction: unknown.",
            "Completed subtasks: none.",
            "Current subtask: unknown.",
        ]
    )
