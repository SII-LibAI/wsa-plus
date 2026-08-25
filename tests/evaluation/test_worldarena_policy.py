from __future__ import annotations

import json
import os
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from evaluation.worldarena.wsa_memory_adapter import (
    MEMORY_HISTORY_VALID_MASK,
    RuntimeSettings,
    VectorScale,
    WSAWorldArenaAdapter,
    WSAMemoryWorldArenaAdapter,
    _build_scale,
    _resolve_checkpoint_dir,
    checkpoint_model_type,
    extract_agilex_state,
    extract_franka_state,
    history_indices,
    normalize_pose_quaternion,
    require_task_only_text_mode,
    select_feature_stats,
    stats_key_candidates,
)


def _clean_worldarena_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(os.environ):
        if name.startswith("WSA_") or name.startswith("WORLDARENA_"):
            monkeypatch.delenv(name, raising=False)


def _rgb() -> np.ndarray:
    return np.zeros((4, 5, 3), dtype=np.uint8)


def _agilex_images() -> dict[str, np.ndarray]:
    return {
        "cam_high": _rgb(),
        "cam_left_wrist": _rgb(),
        "cam_right_wrist": _rgb(),
        "cam_wrist": _rgb(),
    }


def _franka_images() -> dict[str, np.ndarray]:
    return {"cam_high": _rgb(), "cam_left_wrist": _rgb()}


def test_quaternion_normalize_preserves_xyzw_order() -> None:
    xyzw = np.asarray([1, 2, 3, 4, 1, 2, 3, 0.5], dtype=np.float32)
    normalized = normalize_pose_quaternion(xyzw)
    np.testing.assert_array_equal(normalized[:3], xyzw[:3])
    np.testing.assert_array_equal(normalized[7:], xyzw[7:])
    np.testing.assert_allclose(
        normalized[3:7], xyzw[3:7] / np.linalg.norm(xyzw[3:7])
    )
    assert np.linalg.norm(normalized[3:7]) == pytest.approx(1.0)
    with pytest.raises(FloatingPointError, match="near-zero quaternion"):
        normalize_pose_quaternion(np.zeros(8, dtype=np.float32))


def test_agilex_state_strictly_concatenates_official_per_arm_qpos() -> None:
    left = np.arange(7, dtype=np.float32)
    right = np.arange(7, 14, dtype=np.float32)
    obs = {
        "joint_qpos_left": left,
        "joint_qpos_right": right,
        # The official bridge exposes this single-arm alias as well. It must not
        # replace either of the two required per-arm vectors.
        "joint_qpos": right,
    }
    np.testing.assert_array_equal(
        extract_agilex_state(obs), np.concatenate([left, right])
    )

    with pytest.raises(KeyError, match="joint_qpos_left"):
        extract_agilex_state({"joint_qpos": np.concatenate([left, right])})
    with pytest.raises(KeyError, match="joint_qpos_right"):
        extract_agilex_state({"joint_qpos_left": left, "joint_qpos": right})
    with pytest.raises(ValueError, match="exact shape"):
        extract_agilex_state(
            {"joint_qpos_left": left[None, :], "joint_qpos_right": right}
        )


def test_franka_state_only_uses_documented_xyzw_pose_and_joint_gripper() -> None:
    obs = {
        "left_end_pose": np.asarray([1, 2, 3, 0.9, 0.1, 0.2, 0.3], dtype=np.float32),
        "joint_qpos": np.asarray([0, 1, 2, 3, 4, 5, 6, 0.75], dtype=np.float32),
        "state": np.zeros(32, dtype=np.float32),
    }
    state = extract_franka_state(obs)
    np.testing.assert_allclose(state, [1, 2, 3, 0.9, 0.1, 0.2, 0.3, 0.75])

    with pytest.raises(KeyError, match="left_end_pose"):
        extract_franka_state({"end_pose": obs["left_end_pose"], "joint_qpos": obs["joint_qpos"]})
    with pytest.raises(ValueError, match="exact shape"):
        extract_franka_state({"left_end_pose": obs["left_end_pose"], "joint_qpos": np.zeros(7)})


@pytest.mark.parametrize(
    ("length", "frames", "stride", "indices", "valid"),
    [
        (1, 3, 2, [0, 0, 0], [False, False, True]),
        (3, 3, 2, [0, 0, 2], [False, True, True]),
        (5, 3, 2, [0, 2, 4], [True, True, True]),
    ],
)
def test_history_indices(length, frames, stride, indices, valid) -> None:
    assert history_indices(length, frames, stride) == (indices, valid)


def test_nested_stats_selection_and_scale() -> None:
    payload = {
        "cobot_magic_max": {
            "observation.state": {
                "mean": [1.0] * 14,
                "std": [2.0] * 14,
            },
            "action": {"mean": [3.0] * 14, "std": [4.0] * 14},
        },
        "other": {
            "observation.state": {"mean": [0.0], "std": [1.0]},
        },
    }
    selected_key, feature_stats = select_feature_stats(payload, "cobot_magic_max")
    assert selected_key == "cobot_magic_max"
    scale = _build_scale(
        feature_stats,
        explicit_keys=(),
        candidate_groups=(("observation.state",),),
        expected_dim=14,
        mode="mean_std",
        label="state",
    )
    normalized = scale.normalize(np.asarray([3.0] * 14, dtype=np.float32))
    np.testing.assert_allclose(normalized, np.ones(14), atol=1e-6)
    np.testing.assert_allclose(scale.inverse(normalized), [3.0] * 14, atol=1e-6)


def test_franka_does_not_auto_accept_ambiguous_flat_joint_stats() -> None:
    feature_stats = {
        "observation.state": {"mean": [0.0] * 8, "std": [1.0] * 8},
        "action": {"mean": [0.0] * 8, "std": [1.0] * 8},
    }
    state_groups, action_groups = stats_key_candidates("franka")
    with pytest.raises(KeyError, match="WSA_STATE_STATS_KEYS"):
        _build_scale(
            feature_stats,
            explicit_keys=(),
            candidate_groups=state_groups,
            expected_dim=8,
            mode="mean_std",
            label="state",
        )
    with pytest.raises(KeyError, match="WSA_ACTION_STATS_KEYS"):
        _build_scale(
            feature_stats,
            explicit_keys=(),
            candidate_groups=action_groups,
            expected_dim=8,
            mode="mean_std",
            label="action",
        )


def test_agilex_accepts_official_qpos_stats_name() -> None:
    feature_stats = {
        "observations.qpos": {"mean": [0.0] * 14, "std": [1.0] * 14},
    }
    state_groups, _ = stats_key_candidates("agilex")
    scale = _build_scale(
        feature_stats,
        explicit_keys=(),
        candidate_groups=state_groups,
        expected_dim=14,
        mode="mean_std",
        label="state",
    )
    assert scale.keys == ("observations.qpos",)


def test_index_only_sharded_checkpoint_is_rejected(tmp_path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({}), encoding="utf-8")
    (tmp_path / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="model safetensors"):
        _resolve_checkpoint_dir(str(tmp_path))


def test_checkpoint_model_type_auto_dispatches_memory_before_base() -> None:
    assert checkpoint_model_type(SimpleNamespace(type="wsa_memory")) == "wsa_memory"
    assert checkpoint_model_type(SimpleNamespace(type="WSA_Base")) == "wsa_base"
    assert checkpoint_model_type(SimpleNamespace(type="wsa_base")) == "wsa_base"
    with pytest.raises(TypeError, match="only WSA-Base and WSA-Memory"):
        checkpoint_model_type(SimpleNamespace(type="other"))


def test_worldarena_rejects_subtask_text_checkpoints() -> None:
    assert (
        require_task_only_text_mode("wsa_base", base_mode=None) == "task_only"
    )
    assert (
        require_task_only_text_mode("wsa_memory", memory_mode="task_only")
        == "task_only"
    )
    with pytest.raises(ValueError, match="use a task_only checkpoint"):
        require_task_only_text_mode("wsa_base", base_mode="subtask")
    with pytest.raises(ValueError, match="use a task_only checkpoint"):
        require_task_only_text_mode("wsa_memory", memory_mode="oracle")


class _RecordingProcessor:
    def __init__(self) -> None:
        self.samples: list[dict] = []

    def __call__(self, sample: dict) -> dict:
        self.samples.append(dict(sample))
        return sample


def _batch_adapter(
    monkeypatch: pytest.MonkeyPatch, model_type: str, history_num_frames: int
) -> tuple[WSAWorldArenaAdapter, _RecordingProcessor]:
    import torch

    _clean_worldarena_env(monkeypatch)
    monkeypatch.setenv("WORLDARENA_PLATFORM", "agilex")
    monkeypatch.setenv("WSA_DRY_RUN", "1")
    monkeypatch.setenv("WSA_DEVICE", "cpu")
    monkeypatch.setenv("WSA_DTYPE", "float32")
    adapter = WSAWorldArenaAdapter(RuntimeSettings.from_sources())
    adapter.model_type = model_type
    adapter.device = torch.device("cpu")
    adapter.dtype = torch.float32
    adapter.history_num_frames = history_num_frames
    adapter.history_stride_calls = 1
    adapter._history = deque(maxlen=history_num_frames)
    adapter.state_scale = VectorScale(
        mode="mean_std",
        first=np.zeros(14, dtype=np.float32),
        second=np.ones(14, dtype=np.float32),
        keys=("observation.state",),
    )
    processor = _RecordingProcessor()
    adapter.processor_fn = processor
    return adapter, processor


def _tensor_images(value: float):
    import torch

    return tuple(torch.full((3, 4, 5), value) for _ in range(3))


def test_base_batch_uses_past_current_images_and_current_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    adapter, processor = _batch_adapter(monkeypatch, "wsa_base", 2)
    state0 = np.arange(14, dtype=np.float32)
    batch0 = adapter._prepare_model_batch(
        "complete task", state0, _tensor_images(0.0), (True, True, True)
    )
    assert processor.samples[-1]["task"] == "complete task"
    assert batch0["observation.state"].shape == (1, 14)
    assert batch0["observation.images.image0"].shape == (1, 2, 3, 4, 5)
    assert MEMORY_HISTORY_VALID_MASK not in batch0
    torch.testing.assert_close(
        batch0["observation.images.image0"][0, 0],
        batch0["observation.images.image0"][0, 1],
    )

    state1 = state0 + 1
    batch1 = adapter._prepare_model_batch(
        "complete task", state1, _tensor_images(1.0), (True, True, True)
    )
    torch.testing.assert_close(batch1["observation.state"][0], torch.from_numpy(state1))
    assert torch.all(batch1["observation.images.image0"][0, 0] == 0)
    assert torch.all(batch1["observation.images.image0"][0, 1] == 1)


def test_memory_batch_keeps_k_frame_state_and_history_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, _ = _batch_adapter(monkeypatch, "wsa_memory", 3)
    batch = adapter._prepare_model_batch(
        "complete task",
        np.arange(14, dtype=np.float32),
        _tensor_images(0.0),
        (True, True, True),
    )
    assert batch["observation.state"].shape == (1, 3, 14)
    assert batch["observation.images.image0"].shape == (1, 3, 3, 4, 5)
    assert batch[MEMORY_HISTORY_VALID_MASK].tolist() == [[False, False, True]]


def test_agilex_dry_run_returns_hold_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_worldarena_env(monkeypatch)
    monkeypatch.setenv("WORLDARENA_PLATFORM", "agilex")
    monkeypatch.setenv("WSA_DRY_RUN", "1")
    monkeypatch.setenv("WSA_EXECUTE_CHUNK_SIZE", "2")
    adapter = WSAMemoryWorldArenaAdapter(RuntimeSettings.from_sources())
    qpos = np.arange(14, dtype=np.float32)
    result = adapter.infer(
        {
            "prompt": "clean the table",
            "joint_qpos_left": qpos[:7],
            "joint_qpos_right": qpos[7:],
            "joint_qpos": qpos[7:],
            "images": _agilex_images(),
        }
    )
    np.testing.assert_array_equal(result["actions"], np.stack([qpos, qpos]))
    assert result["actions"].dtype == np.float32
    assert result["policy_metadata"]["action_format"] == "joint_absolute"


def test_franka_dry_run_returns_documented_xyzw_hold_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clean_worldarena_env(monkeypatch)
    monkeypatch.setenv("WORLDARENA_PLATFORM", "franka")
    monkeypatch.setenv("WSA_DRY_RUN", "1")
    monkeypatch.setenv("WSA_EXECUTE_CHUNK_SIZE", "2")
    settings = RuntimeSettings.from_sources()
    adapter = WSAMemoryWorldArenaAdapter(settings)
    pose_xyzw = np.asarray([0.4, 0.1, 0.3, 0.9, 0.1, 0.2, 0.3], dtype=np.float32)
    result = adapter.infer(
        {
            "prompt": "wipe the table",
            "left_end_pose": pose_xyzw,
            "joint_qpos": np.asarray([0, 1, 2, 3, 4, 5, 6, 0.25], dtype=np.float32),
            "images": _franka_images(),
        }
    )
    expected = np.concatenate([pose_xyzw, [0.25]]).astype(np.float32)
    expected[3:7] /= np.linalg.norm(expected[3:7])
    np.testing.assert_allclose(result["actions"], np.stack([expected, expected]))
    assert result["policy_metadata"]["action_format"] == "end_pose_base"
    assert result["policy_metadata"]["control_arm"] == "left"


def test_dry_run_rejects_prompt_and_camera_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clean_worldarena_env(monkeypatch)
    monkeypatch.setenv("WORLDARENA_PLATFORM", "agilex")
    monkeypatch.setenv("WSA_DRY_RUN", "1")
    adapter = WSAMemoryWorldArenaAdapter(RuntimeSettings.from_sources())
    qpos = np.zeros(14, dtype=np.float32)
    state = {
        "joint_qpos_left": qpos[:7],
        "joint_qpos_right": qpos[7:],
        "joint_qpos": qpos[7:],
    }

    with pytest.raises(KeyError, match="prompt"):
        adapter.infer({"task": "clean", **state, "images": _agilex_images()})

    wrong_images = {
        "cam_high": _rgb(),
        "cam_wrist_left": _rgb(),
        "cam_wrist_right": _rgb(),
    }
    with pytest.raises(KeyError, match="cam_left_wrist"):
        adapter.infer({"prompt": "clean", **state, "images": wrong_images})


def test_dry_run_rejects_non_uint8_or_chw_images(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clean_worldarena_env(monkeypatch)
    monkeypatch.setenv("WORLDARENA_PLATFORM", "franka")
    monkeypatch.setenv("WSA_DRY_RUN", "1")
    adapter = WSAMemoryWorldArenaAdapter(RuntimeSettings.from_sources())
    obs = {
        "prompt": "wipe",
        "left_end_pose": np.asarray([0, 0, 0, 1, 0, 0, 0], dtype=np.float32),
        "joint_qpos": np.zeros(8, dtype=np.float32),
        "images": _franka_images(),
    }
    obs["images"]["cam_high"] = np.zeros((3, 4, 5), dtype=np.uint8)
    with pytest.raises(ValueError, match="HWC RGB"):
        adapter.infer(obs)

    obs["images"] = _franka_images()
    obs["images"]["cam_high"] = _rgb().astype(np.float32)
    with pytest.raises(TypeError, match="dtype=uint8"):
        adapter.infer(obs)


def test_franka_does_not_fall_back_to_duplicate_cam_wrist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clean_worldarena_env(monkeypatch)
    monkeypatch.setenv("WORLDARENA_PLATFORM", "franka")
    monkeypatch.setenv("WSA_DRY_RUN", "1")
    adapter = WSAMemoryWorldArenaAdapter(RuntimeSettings.from_sources())
    obs = {
        "prompt": "wipe",
        "left_end_pose": np.asarray([0, 0, 0, 1, 0, 0, 0], dtype=np.float32),
        "joint_qpos": np.zeros(8, dtype=np.float32),
        "images": {"cam_high": _rgb(), "cam_wrist": _rgb()},
    }
    with pytest.raises(KeyError, match="cam_left_wrist"):
        adapter.infer(obs)
