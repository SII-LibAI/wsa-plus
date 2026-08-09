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
        hand_map=sidecar.HAND_MAP,
        rigidity_map=sidecar.DEFAULT_RIGIDITY_MAP,
        max_completed_subtasks=8,
    )

    assert current.subtask_id == 1
    assert memory.completed_subtasks == ("move the gripper close to the towel",)
    assert sidecar.build_memory_prompt(memory) == "\n".join(
        [
            "Overall task: Grab the towel and then wipe the table and put the towel back.",
            "Scene: home.",
            "Completed subtasks: move the gripper close to the towel.",
            "Current subtask: grasp the towel.",
            "Interaction arm: right.",
            "Object property: flexible.",
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
        hand_map=sidecar.HAND_MAP,
        rigidity_map=sidecar.DEFAULT_RIGIDITY_MAP,
        max_completed_subtasks=8,
    )
    assert current is None
    assert memory.overall_task == "Dataset task has priority."
    assert memory.current_subtask == "unknown"
    assert memory.interaction_arm == "unknown"
    assert memory.object_property == "unknown"


def test_action_mask_combines_padding_and_real_frame_offsets() -> None:
    current = sidecar.Subtask(
        subtask_id=1,
        subtask_text="grasp",
        start_frame=121,
        end_frame=240,
        confidence=None,
        hand_used=1,
        object_rigidity=1,
    )
    assert sidecar.build_action_loss_mask(
        action_is_pad=[False, False, False, True],
        frame_index=239,
        action_frame_offsets=[0, 1, 2, 4],
        current_subtask=current,
        apply_subtask_boundary=True,
    ) == [True, True, False, False]


def test_sidecar_validation_fails_fast(tmp_path: Path) -> None:
    overlapping = _episode()
    overlapping["subtasks"][1]["start_frame"] = 120
    _write_sidecar(tmp_path, [overlapping])
    with pytest.raises(ValueError, match="overlapping closed intervals"):
        sidecar.SidecarIndex.load(tmp_path)


def test_enum_map_rejects_duplicate_normalized_keys() -> None:
    with pytest.raises(ValueError, match="duplicate normalized key"):
        sidecar.normalize_enum_map({1: "right", "1": "also-right"}, "hand_map")

