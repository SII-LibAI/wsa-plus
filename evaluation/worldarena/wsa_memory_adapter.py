"""WSA-Base/WSA-Memory adapter for the official WorldArena 2.0 Track 3 Policy API.

The official ``serve_policy_worldarena`` process imports :class:`Policy` from
``policy.py`` and passes legacy ``new_obs`` dictionaries to ``infer``.  This
module deliberately implements only the model side of that interface; network
transport, canonical packets, reset dispatch and Hub polling stay in the
official WorldArena package.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import threading
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


LOGGER = logging.getLogger("wsa_worldarena")
SUPPORTED_PLATFORMS = ("agilex", "franka")
PLATFORM_CONTROL_FPS = {"agilex": 30.0, "franka": 15.0}
WSA_BASE_MODEL_TYPE = "wsa_base"
WSA_MEMORY_MODEL_TYPE = "wsa_memory"
MEMORY_HISTORY_VALID_MASK = "memory.history_valid_mask"
WSA_BASE_CHECKPOINT_TYPES = frozenset(
    {"WSA_Base", "wsa_base", "TBot_SA1", "tbot_sa1", "cubev2", "magicbot"}
)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def _optional_env(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    return value.strip()


def _csv(value: str | Sequence[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    return tuple(str(part).strip() for part in value if str(part).strip())


def _load_user_config(config_path: str | None) -> dict[str, Any]:
    selected = config_path or _optional_env("WSA_WORLDARENA_CONFIG")
    if selected is None:
        return {}
    path = Path(selected).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"WSA WorldArena config does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"WSA WorldArena config must contain a JSON object: {path}")
    return payload


def _config_value(
    config: Mapping[str, Any],
    env_name: str,
    key: str,
    default: Any = None,
) -> Any:
    env_value = os.environ.get(env_name)
    if env_value is not None and env_value != "":
        return env_value
    return config.get(key, default)


@dataclass(frozen=True)
class RuntimeSettings:
    platform: str
    checkpoint: str | None
    stats_path: str | None
    stats_key: str | None
    state_stats_keys: tuple[str, ...]
    action_stats_keys: tuple[str, ...]
    robot_type: str
    action_mode: str | None
    normalization_mode: str | None
    delta_mask: tuple[bool, ...]
    device: str
    load_device: str | None
    dtype: str
    qwen3_vl_path: str | None
    processor_path: str | None
    cosmos_tokenizer_path: str | None
    cosmos_device: str | None
    num_inference_steps: int | None
    execute_chunk_size: int
    history_stride_calls: int | None
    allow_franka_delta: bool
    dry_run: bool
    policy_id: str

    @property
    def expected_state_dim(self) -> int:
        return 14 if self.platform == "agilex" else 8

    @property
    def expected_action_dim(self) -> int:
        return 14 if self.platform == "agilex" else 8

    @property
    def action_format(self) -> str:
        return "joint_absolute" if self.platform == "agilex" else "end_pose_base"

    @classmethod
    def from_sources(cls, config_path: str | None = None) -> "RuntimeSettings":
        config = _load_user_config(config_path)
        platform = str(
            _config_value(config, "WORLDARENA_PLATFORM", "platform", "")
        ).strip().lower()
        if platform not in SUPPORTED_PLATFORMS:
            raise ValueError(
                f"WORLDARENA_PLATFORM must be one of {SUPPORTED_PLATFORMS}, got {platform!r}"
            )

        def optional(name: str, key: str) -> str | None:
            value = _config_value(config, name, key)
            if value is None or not str(value).strip():
                return None
            return str(value).strip()

        def integer(name: str, key: str, default: int | None) -> int | None:
            value = _config_value(config, name, key, default)
            if value is None or str(value).strip() == "":
                return None
            return int(value)

        checkpoint = optional("WSA_CHECKPOINT", "checkpoint")
        stats_path = optional("WSA_STATS_PATH", "stats_path")
        execute_chunk_size = int(
            _config_value(config, "WSA_EXECUTE_CHUNK_SIZE", "execute_chunk_size", 1)
        )
        if execute_chunk_size <= 0:
            raise ValueError("WSA_EXECUTE_CHUNK_SIZE must be positive")
        history_stride_calls = integer(
            "WSA_HISTORY_STRIDE_CALLS", "history_stride_calls", None
        )
        if history_stride_calls is not None and history_stride_calls <= 0:
            raise ValueError("WSA_HISTORY_STRIDE_CALLS must be positive when set")

        delta_mask_value = _config_value(config, "WSA_DELTA_MASK", "delta_mask")
        delta_mask: tuple[bool, ...] = ()
        if delta_mask_value:
            raw_values = _csv(delta_mask_value)
            parsed: list[bool] = []
            for raw in raw_values:
                normalized = raw.lower()
                if normalized in {"1", "true", "yes", "on"}:
                    parsed.append(True)
                elif normalized in {"0", "false", "no", "off"}:
                    parsed.append(False)
                else:
                    raise ValueError(f"Invalid WSA_DELTA_MASK element: {raw!r}")
            delta_mask = tuple(parsed)

        default_robot_type = "cobot_magic_max" if platform == "agilex" else "franka"

        settings = cls(
            platform=platform,
            checkpoint=checkpoint,
            stats_path=stats_path,
            stats_key=optional("WSA_STATS_KEY", "stats_key"),
            state_stats_keys=_csv(
                _config_value(config, "WSA_STATE_STATS_KEYS", "state_stats_keys")
            ),
            action_stats_keys=_csv(
                _config_value(config, "WSA_ACTION_STATS_KEYS", "action_stats_keys")
            ),
            robot_type=str(
                _config_value(config, "WSA_ROBOT_TYPE", "robot_type", default_robot_type)
            ).strip(),
            action_mode=optional("WSA_ACTION_MODE", "action_mode"),
            normalization_mode=optional(
                "WSA_NORMALIZATION_MODE", "normalization_mode"
            ),
            delta_mask=delta_mask,
            device=str(_config_value(config, "WSA_DEVICE", "device", "cuda:0")).strip(),
            load_device=optional("WSA_LOAD_DEVICE", "load_device"),
            dtype=str(_config_value(config, "WSA_DTYPE", "dtype", "bfloat16")).strip(),
            qwen3_vl_path=optional("WSA_QWEN3_VL_PATH", "qwen3_vl_path"),
            processor_path=optional("WSA_PROCESSOR_PATH", "processor_path"),
            cosmos_tokenizer_path=optional(
                "WSA_COSMOS_TOKENIZER_PATH", "cosmos_tokenizer_path"
            ),
            cosmos_device=optional("WSA_COSMOS_DEVICE", "cosmos_device"),
            num_inference_steps=integer(
                "WSA_NUM_INFERENCE_STEPS", "num_inference_steps", None
            ),
            execute_chunk_size=execute_chunk_size,
            history_stride_calls=history_stride_calls,
            allow_franka_delta=_env_bool(
                "WSA_ALLOW_FRANKA_DELTA",
                bool(config.get("allow_franka_delta", False)),
            ),
            dry_run=_env_bool("WSA_DRY_RUN", bool(config.get("dry_run", False))),
            policy_id=str(
                _config_value(
                    config,
                    "WSA_POLICY_ID",
                    "policy_id",
                    os.environ.get("POLICY_ID", f"wsa_{platform}"),
                )
            ).strip(),
        )
        if not settings.dry_run and not settings.checkpoint:
            raise ValueError("WSA_CHECKPOINT is required unless WSA_DRY_RUN=1")
        return settings


def _required_vector(
    obs: Mapping[str, Any], key: str, expected_dim: int
) -> np.ndarray:
    """Read one official WorldArena vector without aliases or reshaping."""
    if key not in obs:
        raise KeyError(f"WorldArena observation is missing required `{key}`")
    value = np.asarray(obs[key])
    if value.shape != (expected_dim,):
        raise ValueError(
            f"WorldArena `{key}` must have exact shape ({expected_dim},), "
            f"got {value.shape}"
        )
    if not np.issubdtype(value.dtype, np.number):
        raise TypeError(
            f"WorldArena `{key}` must be numeric, got dtype={value.dtype}"
        )
    value = value.astype(np.float32, copy=False)
    if not np.isfinite(value).all():
        bad = np.argwhere(~np.isfinite(value)).reshape(-1).tolist()
        raise FloatingPointError(
            f"WorldArena `{key}` contains non-finite values at indices {bad[:10]}"
        )
    return np.ascontiguousarray(value)


def _shape_summary(obs: Mapping[str, Any], keys: Iterable[str]) -> str:
    parts: list[str] = []
    for key in keys:
        if key not in obs:
            continue
        try:
            parts.append(f"{key}={tuple(np.asarray(obs[key]).shape)}")
        except Exception:
            parts.append(f"{key}=<{type(obs[key]).__name__}>")
    return ", ".join(parts) or "<none of the expected fields are present>"


def extract_agilex_state(
    obs: Mapping[str, Any],
    expected_dim: int = 14,
) -> np.ndarray:
    """Build left-then-right 14D qpos from the official A-side legacy bridge."""
    if expected_dim != 14:
        raise ValueError(f"AgileX Track 3 state dimension must be 14, got {expected_dim}")
    left = _required_vector(obs, "joint_qpos_left", 7)
    right = _required_vector(obs, "joint_qpos_right", 7)
    return np.ascontiguousarray(np.concatenate([left, right]), dtype=np.float32)


def extract_franka_state(
    obs: Mapping[str, Any],
    expected_dim: int = 8,
) -> np.ndarray:
    """Build official Franka ``xyz+xyzw+gripper`` state with no fallback."""
    if expected_dim != 8:
        raise ValueError(f"Franka Track 3 state dimension must be 8, got {expected_dim}")
    pose = _required_vector(obs, "left_end_pose", 7)
    joint_qpos = _required_vector(obs, "joint_qpos", 8)
    return np.ascontiguousarray(
        np.concatenate([pose, joint_qpos[-1:]]), dtype=np.float32
    )


def history_indices(length: int, frames: int, stride_calls: int) -> tuple[list[int], list[bool]]:
    """Return oldest-to-current indices with LeRobot-style left padding."""
    if length <= 0 or frames <= 0 or stride_calls <= 0:
        raise ValueError("length, frames and stride_calls must be positive")
    indices: list[int] = []
    valid: list[bool] = []
    latest = length - 1
    for slot in range(frames):
        offset = (frames - 1 - slot) * stride_calls
        index = latest - offset
        is_valid = index >= 0
        indices.append(max(0, index))
        valid.append(is_valid)
    valid[-1] = True
    return indices, valid


def _is_stats_leaf(value: Any) -> bool:
    return isinstance(value, dict) and (
        ("mean" in value and "std" in value) or ("min" in value and "max" in value)
    )


def _feature_stats_roots(
    value: Any, path: tuple[str, ...] = ()
) -> list[tuple[tuple[str, ...], dict[str, Any]]]:
    if not isinstance(value, dict):
        return []
    roots: list[tuple[tuple[str, ...], dict[str, Any]]] = []
    if any(_is_stats_leaf(child) for child in value.values()):
        roots.append((path, value))
        return roots
    for key, child in value.items():
        roots.extend(_feature_stats_roots(child, (*path, str(key))))
    return roots


def select_feature_stats(
    payload: Mapping[str, Any], stats_key: str | None
) -> tuple[str, dict[str, Any]]:
    roots = _feature_stats_roots(payload)
    if not roots:
        raise ValueError("Stats JSON contains no feature statistics with mean/std or min/max")
    if stats_key:
        matches = [item for item in roots if item[0] and item[0][-1] == stats_key]
        if not matches and len(roots) == 1 and not roots[0][0]:
            return "<flat>", roots[0][1]
        if len(matches) != 1:
            available = ["/".join(path) or "<flat>" for path, _ in roots]
            raise KeyError(
                f"WSA_STATS_KEY={stats_key!r} did not select exactly one stats block; "
                f"available={available}"
            )
        path, selected = matches[0]
        return "/".join(path), selected
    if len(roots) != 1:
        available = ["/".join(path) or "<flat>" for path, _ in roots]
        raise ValueError(
            "Stats JSON has multiple embodiment blocks; set WSA_STATS_KEY explicitly. "
            f"Available={available}"
        )
    path, selected = roots[0]
    return "/".join(path) or "<flat>", selected


@dataclass(frozen=True)
class VectorScale:
    mode: str
    first: np.ndarray
    second: np.ndarray
    keys: tuple[str, ...]

    @property
    def dim(self) -> int:
        return int(self.first.size)

    def normalize(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float32)
        if value.shape[-1] != self.dim:
            raise ValueError(f"Expected vector dim={self.dim}, got shape={value.shape}")
        if self.mode == "mean_std":
            return (value - self.first) / (self.second + 1e-6)
        if self.mode == "min_max":
            return (value - self.first) / (self.second - self.first + 1e-6)
        raise ValueError(f"Unsupported normalization mode: {self.mode}")

    def inverse(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float32)
        if value.shape[-1] != self.dim:
            raise ValueError(f"Expected vector dim={self.dim}, got shape={value.shape}")
        if self.mode == "mean_std":
            return value * (self.second + 1e-6) + self.first
        if self.mode == "min_max":
            return value * (self.second - self.first + 1e-6) + self.first
        raise ValueError(f"Unsupported normalization mode: {self.mode}")


def _stats_dim(stats: Mapping[str, Any], mode: str) -> int:
    name = "mean" if mode == "mean_std" else "min"
    if name not in stats:
        raise KeyError(f"Stats entry lacks {name!r}: keys={list(stats)}")
    return int(np.asarray(stats[name]).size)


def _build_scale(
    feature_stats: Mapping[str, Any],
    *,
    explicit_keys: Sequence[str],
    candidate_groups: Sequence[Sequence[str]],
    expected_dim: int,
    mode: str,
    label: str,
) -> VectorScale:
    if mode not in {"mean_std", "min_max"}:
        raise ValueError(f"Unsupported normalization mode: {mode}")
    groups = [tuple(explicit_keys)] if explicit_keys else [tuple(g) for g in candidate_groups]
    selected: tuple[str, ...] | None = None
    for group in groups:
        if not group or not all(key in feature_stats and _is_stats_leaf(feature_stats[key]) for key in group):
            continue
        if sum(_stats_dim(feature_stats[key], mode) for key in group) == expected_dim:
            selected = group
            break
    if selected is None:
        available = {
            key: _stats_dim(value, mode)
            for key, value in feature_stats.items()
            if _is_stats_leaf(value)
            and ((mode == "mean_std" and "mean" in value) or (mode == "min_max" and "min" in value))
        }
        requested = list(explicit_keys) if explicit_keys else [list(g) for g in candidate_groups]
        raise KeyError(
            f"Could not resolve {label} stats with total dim={expected_dim}. "
            f"Requested candidates={requested}; available feature dims={available}. "
            f"Set WSA_{label.upper()}_STATS_KEYS explicitly."
        )
    first_name, second_name = ("mean", "std") if mode == "mean_std" else ("min", "max")
    first_parts: list[np.ndarray] = []
    second_parts: list[np.ndarray] = []
    for key in selected:
        stats = feature_stats[key]
        first = np.asarray(stats[first_name], dtype=np.float32).reshape(-1)
        second = np.asarray(stats[second_name], dtype=np.float32).reshape(-1)
        if first.shape != second.shape:
            raise ValueError(f"Stats shape mismatch for {key}: {first.shape} vs {second.shape}")
        first_parts.append(first)
        second_parts.append(second)
    return VectorScale(
        mode=mode,
        first=np.concatenate(first_parts),
        second=np.concatenate(second_parts),
        keys=selected,
    )


def stats_key_candidates(
    platform: str,
) -> tuple[tuple[tuple[str, ...], ...], tuple[tuple[str, ...], ...]]:
    if platform == "agilex":
        state = (
            ("observation.state",),
            ("state",),
            ("observations.qpos",),
            ("observation.qpos",),
            (
                "states.left_joint.position",
                "states.left_gripper.position",
                "states.right_joint.position",
                "states.right_gripper.position",
            ),
        )
        action = (
            ("action",),
            (
                "actions.left_joint.position",
                "actions.left_gripper.position",
                "actions.right_joint.position",
                "actions.right_gripper.position",
            ),
        )
        return state, action

    state = (
        ("states.end_pose",),
        ("observations.end_pose",),
        ("observation.end_pose",),
        ("observation.states.end_pose",),
    )
    action = (
        ("actions.end_pose",),
        ("action.end_pose",),
        ("observations.end_pose",),
    )
    return state, action


def _resolve_checkpoint_dir(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.exists():
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(repo_id=value))
    candidates = (
        path,
        path / "pretrained_model",
        path / "checkpoints" / "last" / "pretrained_model",
        path / "last" / "pretrained_model",
    )
    for candidate in candidates:
        has_config = (candidate / "config.json").is_file()
        # PreTrainedPolicy.from_pretrained currently opens this exact filename;
        # accepting an index-only sharded checkpoint would fail later and hide
        # the real cause.
        has_weights = (candidate / "model.safetensors").is_file()
        if has_config and has_weights:
            return candidate.resolve()
    raise FileNotFoundError(
        f"Could not find config.json and model safetensors under {path}. Pass a "
        "pretrained_model directory, checkpoint step, training run or Hugging Face id."
    )


def checkpoint_model_type(config: Any) -> str:
    """Resolve the supported runtime family from a decoded policy config.

    WSA-Memory inherits the WSA-Base config class, so dispatch must use the
    explicit serialized ``type`` before any Base-class check.
    """
    config_type = getattr(config, "type", None)
    if config_type == WSA_MEMORY_MODEL_TYPE:
        return WSA_MEMORY_MODEL_TYPE

    if config_type in WSA_BASE_CHECKPOINT_TYPES:
        return WSA_BASE_MODEL_TYPE
    raise TypeError(
        "WorldArena supports only WSA-Base and WSA-Memory checkpoints; "
        f"got {type(config).__name__} with type={config_type!r}"
    )


def require_task_only_text_mode(
    model_type: str,
    *,
    memory_mode: str | None = None,
    base_mode: str | None = None,
) -> str:
    """Validate the only text contract available from WorldArena observations."""
    if model_type == WSA_MEMORY_MODEL_TYPE:
        mode = memory_mode
        field_name = "text_memory_mode"
    elif model_type == WSA_BASE_MODEL_TYPE:
        # Checkpoints predating Base subtask support had no saved field and
        # always consumed the original task instruction.
        mode = base_mode or "task_only"
        field_name = "text_context_mode"
    else:
        raise ValueError(f"Unsupported WSA model type: {model_type!r}")
    if mode != "task_only":
        raise ValueError(
            "WorldArena supplies only the complete task instruction and cannot "
            f"serve {field_name}={mode!r}; use a task_only checkpoint."
        )
    return mode


def _find_companion(checkpoint_dir: Path, filename: str) -> Path | None:
    current = checkpoint_dir
    for _ in range(5):
        candidate = current / filename
        if candidate.is_file():
            return candidate
        if current.parent == current:
            break
        current = current.parent
    return None


def _train_hints(checkpoint_dir: Path) -> dict[str, Any]:
    path = _find_companion(checkpoint_dir, "train_config.json")
    if path is None:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        LOGGER.warning("Could not read %s: %s", path, exc)
        return {}
    dataset = payload.get("dataset") if isinstance(payload, dict) else None
    if not isinstance(dataset, dict):
        return {}
    hints = {
        "action_mode": dataset.get("action_mode"),
        "text_context_mode": dataset.get("text_context_mode"),
        "processor_path": dataset.get("qwen3_vl_processor_path"),
        "external_stats_path": dataset.get("external_stats_path"),
        "external_stats_root": dataset.get("external_stats_root"),
        "train_config_path": str(path),
    }

    def find_resize(value: Any) -> tuple[int, int] | None:
        if isinstance(value, dict):
            transform_name = str(
                value.get("type") or value.get("_target_") or value.get("name") or ""
            ).lower()
            if "resize" in transform_name and "height" in value and "width" in value:
                return int(value["height"]), int(value["width"])
            for child in value.values():
                result = find_resize(child)
                if result is not None:
                    return result
        elif isinstance(value, list):
            for child in value:
                result = find_resize(child)
                if result is not None:
                    return result
        return None

    def find_normalization_mode(value: Any) -> str | None:
        if isinstance(value, dict):
            mode = value.get("mode")
            if mode in {"mean_std", "min_max"} and (
                "norm_stats" in value
                or "normalize" in str(value.get("type", "")).lower()
                or "normalize" in str(value.get("_target_", "")).lower()
            ):
                return str(mode)
            for child in value.values():
                result = find_normalization_mode(child)
                if result:
                    return result
        elif isinstance(value, list):
            for child in value:
                result = find_normalization_mode(child)
                if result:
                    return result
        return None

    hints["normalization_mode"] = find_normalization_mode(dataset)
    hints["image_resolution"] = find_resize(dataset)
    return hints


def _resolve_configured_path(value: str | None, checkpoint_dir: Path) -> Path | None:
    if value is None or not str(value).strip():
        return None
    path = Path(str(value)).expanduser()
    candidates = (path,) if path.is_absolute() else (
        checkpoint_dir / path,
        checkpoint_dir.parent / path,
        path,
    )
    return next((candidate.resolve() for candidate in candidates if candidate.exists()), None)


def _resolve_stats_path(
    *,
    explicit_path: str | None,
    checkpoint_dir: Path,
    hints: Mapping[str, Any],
    stats_key: str | None,
    robot_type: str,
    action_mode: str,
) -> Path | None:
    if explicit_path:
        explicit = Path(explicit_path).expanduser()
        if not explicit.is_absolute():
            explicit = checkpoint_dir / explicit
        return explicit.resolve()

    configured = _resolve_configured_path(hints.get("external_stats_path"), checkpoint_dir)
    if configured is not None and configured.is_file():
        return configured

    configured_root = _resolve_configured_path(hints.get("external_stats_root"), checkpoint_dir)
    if configured_root is not None and configured_root.is_dir():
        key = stats_key or robot_type
        candidates = (
            configured_root / f"{key}.json",
            configured_root / key / action_mode / "stats.json",
            configured_root / key / "stats.json",
        )
        selected = next((candidate for candidate in candidates if candidate.is_file()), None)
        if selected is not None:
            return selected.resolve()

    return _find_companion(checkpoint_dir, "stats.json")


def normalize_pose_quaternion(pose: np.ndarray) -> np.ndarray:
    """Normalize the documented xyzw quaternion and reject a zero quaternion."""
    value = np.asarray(pose, dtype=np.float32).copy()
    if value.shape[-1] != 8:
        raise ValueError(f"Franka pose must end in 8 values, got shape={value.shape}")
    quaternion = value[..., 3:7]
    norms = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms < 1e-6):
        raise FloatingPointError("Franka prediction contains an invalid near-zero quaternion")
    value[..., 3:7] = quaternion / norms
    return value


def _required_rgb_image(images: Mapping[str, Any], key: str) -> np.ndarray:
    """Read one official uint8 HWC RGB image without conversion or aliases."""
    if key not in images:
        raise KeyError(
            f"WorldArena images are missing required `{key}`; "
            f"available keys={sorted(str(name) for name in images)}"
        )
    if images[key] is None:
        raise ValueError(f"WorldArena image `{key}` is None")
    image = np.asarray(images[key])
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(
            f"WorldArena image `{key}` must be HWC RGB with shape (H, W, 3), "
            f"got {image.shape}"
        )
    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError(f"WorldArena image `{key}` has an empty spatial dimension")
    if image.dtype != np.uint8:
        raise TypeError(
            f"WorldArena image `{key}` must have dtype=uint8, got {image.dtype}"
        )
    return np.ascontiguousarray(image)


def _to_chw_float(image: np.ndarray) -> Any:
    """Convert an already-validated official RGB image for the WSA processor."""
    import torch

    value = np.asarray(image)
    return torch.from_numpy(value).permute(2, 0, 1).float() / 255.0


@dataclass
class _Snapshot:
    images: tuple[Any, Any, Any]
    camera_masks: tuple[bool, bool, bool]
    normalized_state: Any


class WSAWorldArenaAdapter:
    """Auto-load one WSA-Base or WSA-Memory Track 3 checkpoint."""

    def __init__(self, settings: RuntimeSettings):
        self.settings = settings
        self._lock = threading.RLock()
        self._history: deque[_Snapshot] = deque()
        self._last_prompt: str | None = None
        self._infer_calls = 0
        self._backend_ready = False
        self.model_type = "dry_run" if settings.dry_run else "auto"
        self._metadata: dict[str, Any] = {
            "policy_id": settings.policy_id,
            "model_type": self.model_type,
            "platform": settings.platform,
            "action_format": settings.action_format,
            "action_dim": settings.expected_action_dim,
            "chunk_size": settings.execute_chunk_size,
            "dry_run": settings.dry_run,
        }
        if settings.platform == "franka":
            self._metadata["control_arm"] = "left"
        if not settings.dry_run:
            self._initialize_backend()

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def _initialize_backend(self) -> None:
        import torch

        # Importing both config modules registers their draccus choices before
        # decoding config.json. WSA-Memory must be dispatched before WSA-Base
        # because its config subclasses WSABaseConfig.
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.WSA_Base.configuration_wsa_base import WSABaseConfig
        from lerobot.policies.WSA_Memory.configuration_wsa_memory import (
            WSA_MEMORY,
            WSAMemoryConfig,
        )
        from lerobot.transforms.constants import get_mask_mapping
        from lerobot.transforms.core import ResizeImagesWithPadFn

        assert self.settings.checkpoint is not None
        self.checkpoint_dir = _resolve_checkpoint_dir(self.settings.checkpoint)
        hints = _train_hints(self.checkpoint_dir)
        config = PreTrainedConfig.from_pretrained(self.checkpoint_dir)
        self.model_type = checkpoint_model_type(config)
        if self.model_type == WSA_MEMORY_MODEL_TYPE:
            if not isinstance(config, WSAMemoryConfig) or config.type != WSA_MEMORY:
                raise TypeError(
                    f"Expected a WSA-Memory checkpoint, got {type(config).__name__} "
                    f"with type={getattr(config, 'type', None)!r}"
                )
            self._metadata["text_memory_mode"] = require_task_only_text_mode(
                self.model_type,
                memory_mode=config.text_memory_mode,
            )
            from lerobot.policies.WSA_Memory.modeling_wsa_memory import WSAMemoryPolicy
            from lerobot.policies.WSA_Memory.transform_wsa_memory import (
                Qwen3VLMemoryProcessorTransformFn,
            )

            policy_class = WSAMemoryPolicy
            processor_class = Qwen3VLMemoryProcessorTransformFn
        elif isinstance(config, WSABaseConfig):
            self._metadata["text_context_mode"] = require_task_only_text_mode(
                self.model_type,
                base_mode=hints.get("text_context_mode"),
            )
            from lerobot.policies.WSA_Base.modeling_wsa_base import WSABasePolicy
            from lerobot.policies.WSA_Base.transform_wsa_base import (
                Qwen3_VLProcessorTransformFn,
            )

            policy_class = WSABasePolicy
            processor_class = Qwen3_VLProcessorTransformFn
        else:  # Defensive: checkpoint_model_type already rejects this case.
            raise TypeError(
                f"Expected a WSA-Base checkpoint, got {type(config).__name__} "
                f"with type={getattr(config, 'type', None)!r}"
            )

        self.device = torch.device(self.settings.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {self.device}")
        dtype_map = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        if self.settings.dtype not in dtype_map:
            raise ValueError(f"Unsupported WSA_DTYPE={self.settings.dtype!r}")
        self.dtype = dtype_map[self.settings.dtype]
        if self.device.type == "cpu" and self.dtype != torch.float32:
            LOGGER.warning("Using float32 because CPU inference was requested")
            self.dtype = torch.float32
        load_device = self.settings.load_device or str(self.device)
        config.device = load_device
        # WSABasePolicy.to() uses config.dtype independently for Cosmos.  Keep
        # it synchronized with the rest of the model to avoid mixed conv dtypes.
        config.dtype = "bfloat16" if self.dtype == torch.bfloat16 else "float32"
        config.gradient_checkpointing = False
        # DA3 is a training-only teacher and must not be loaded for Track 3 inference.
        config.lambda_3d = 0.0
        if self.settings.qwen3_vl_path:
            config.qwen3_vl_pretrained_path = self.settings.qwen3_vl_path
        if self.settings.cosmos_tokenizer_path:
            config.cosmos_tokenizer_path_or_name = self.settings.cosmos_tokenizer_path
        if self.settings.num_inference_steps is not None:
            config.num_inference_steps = self.settings.num_inference_steps
        cosmos_device = self.settings.cosmos_device or str(self.device)
        setattr(config, "cosmos_device", cosmos_device)

        self.policy = policy_class.from_pretrained(
            config=config,
            pretrained_name_or_path=self.checkpoint_dir,
        )
        self.policy.config.device = str(self.device)
        setattr(self.policy.config, "cosmos_device", cosmos_device)
        self.policy.to(device=self.device, dtype=self.dtype).eval()
        self.policy.reset()
        self.config = config

        action_mode = self.settings.action_mode or hints.get("action_mode")
        if not action_mode:
            raise ValueError(
                "Could not determine whether the checkpoint was trained with absolute or "
                "delta actions. Set WSA_ACTION_MODE=delta or WSA_ACTION_MODE=abs explicitly."
            )
        if action_mode not in {"abs", "delta"}:
            raise ValueError(f"Unsupported action mode: {action_mode!r}")
        if (
            self.settings.action_mode
            and hints.get("action_mode")
            and self.settings.action_mode != hints["action_mode"]
        ):
            raise ValueError(
                f"WSA_ACTION_MODE={self.settings.action_mode!r} differs from checkpoint "
                f"train_config action_mode={hints['action_mode']!r}"
            )
        if (
            self.settings.platform == "franka"
            and action_mode == "delta"
            and not self.settings.allow_franka_delta
        ):
            raise ValueError(
                "Track 3 Franka actions are base-frame end poses. Use a Franka checkpoint "
                "trained with ACTION_TYPE=abs. Set WSA_ALLOW_FRANKA_DELTA=1 only if state "
                "and action are intentionally the same 8D end-pose representation."
            )
        self.action_mode = action_mode

        normalization_mode = (
            self.settings.normalization_mode
            or hints.get("normalization_mode")
            or "mean_std"
        )
        if (
            self.settings.normalization_mode
            and hints.get("normalization_mode")
            and self.settings.normalization_mode != hints["normalization_mode"]
        ):
            raise ValueError(
                f"WSA_NORMALIZATION_MODE={self.settings.normalization_mode!r} differs from "
                f"checkpoint train_config mode={hints['normalization_mode']!r}"
            )
        stats_path = _resolve_stats_path(
            explicit_path=self.settings.stats_path,
            checkpoint_dir=self.checkpoint_dir,
            hints=hints,
            stats_key=self.settings.stats_key,
            robot_type=self.settings.robot_type,
            action_mode=action_mode,
        )
        if stats_path is None or not stats_path.is_file():
            raise FileNotFoundError(
                "Could not find normalization stats. Set WSA_STATS_PATH to the exact JSON "
                "used for this checkpoint."
            )
        payload = json.loads(stats_path.read_text(encoding="utf-8"))
        selected_stats_key, feature_stats = select_feature_stats(payload, self.settings.stats_key)
        state_groups, action_groups = stats_key_candidates(self.settings.platform)
        self.state_scale = _build_scale(
            feature_stats,
            explicit_keys=self.settings.state_stats_keys,
            candidate_groups=state_groups,
            expected_dim=self.settings.expected_state_dim,
            mode=normalization_mode,
            label="state",
        )
        self.action_scale = _build_scale(
            feature_stats,
            explicit_keys=self.settings.action_stats_keys,
            candidate_groups=action_groups,
            expected_dim=self.settings.expected_action_dim,
            mode=normalization_mode,
            label="action",
        )

        model_action_dim = int(config.output_features["action"].shape[0])
        if model_action_dim < self.settings.expected_action_dim:
            raise ValueError(
                f"Checkpoint action feature has dim={model_action_dim}, but Track 3 "
                f"{self.settings.platform} requires at least "
                f"{self.settings.expected_action_dim}."
            )

        if self.settings.execute_chunk_size > int(config.chunk_size):
            raise ValueError(
                f"WSA_EXECUTE_CHUNK_SIZE={self.settings.execute_chunk_size} exceeds "
                f"checkpoint chunk_size={config.chunk_size}"
            )
        if self.model_type == WSA_MEMORY_MODEL_TYPE:
            self.history_num_frames = int(config.history_num_frames)
            if self.settings.history_stride_calls is not None:
                self.history_stride_calls = self.settings.history_stride_calls
            else:
                # One infer request follows execution of the preceding returned chunk.
                estimated = (
                    PLATFORM_CONTROL_FPS[self.settings.platform]
                    * float(config.history_stride_seconds)
                    / self.settings.execute_chunk_size
                )
                self.history_stride_calls = max(1, int(round(estimated)))
        else:
            # WSA-Base consumes exactly past/current RGB frames. State remains
            # current-only and no WSA-Memory validity mask is supplied.
            self.history_num_frames = 2
            if self.settings.history_stride_calls is not None:
                self.history_stride_calls = self.settings.history_stride_calls
            else:
                image_offsets = [int(value) for value in config.image_delta_indices]
                if len(image_offsets) != 3 or image_offsets[1] != 0:
                    raise ValueError(
                        "WSA-Base WorldArena inference requires image_delta_indices "
                        f"in [past, 0, future] form, got {image_offsets}"
                    )
                # One new observation arrives after executing one returned
                # chunk. Approximate the training past-frame offset in infer
                # calls; an explicit override remains available when the
                # deployment cadence differs from the training frame cadence.
                self.history_stride_calls = max(
                    1,
                    int(round(abs(image_offsets[0]) / self.settings.execute_chunk_size)),
                )
        history_capacity = (self.history_num_frames - 1) * self.history_stride_calls + 1
        self._history = deque(maxlen=max(1, history_capacity))

        image_resolution = hints.get("image_resolution") or getattr(
            config, "image_resolution", (224, 224)
        )
        if not isinstance(image_resolution, (tuple, list)) or len(image_resolution) != 2:
            image_resolution = (224, 224)
        self.resize_fn = ResizeImagesWithPadFn(
            height=int(image_resolution[0]), width=int(image_resolution[1])
        )
        processor_path = (
            self.settings.processor_path
            or self.settings.qwen3_vl_path
            or hints.get("processor_path")
            or getattr(config, "qwen3_vl_pretrained_path", None)
        )
        if not processor_path:
            raise ValueError(
                "Could not resolve Qwen3-VL processor. Set WSA_PROCESSOR_PATH explicitly."
            )
        processor_kwargs: dict[str, Any] = {
            "pretrained_model_name_or_path": str(processor_path),
            "max_length": int(config.tokenizer_max_length),
        }
        if self.model_type == WSA_MEMORY_MODEL_TYPE:
            processor_kwargs.update(
                history_num_frames=self.history_num_frames,
                future_generation_enabled=False,
            )
        self.processor_fn = processor_class(**processor_kwargs)
        # Fail during worker registration instead of on the first real robot
        # observation if the processor path is stale or unavailable.
        self.processor_fn._ensure_processor()

        if self.action_mode == "delta":
            if self.settings.delta_mask:
                mask = np.asarray(self.settings.delta_mask, dtype=bool)
            else:
                mask = (
                    get_mask_mapping(self.settings.robot_type)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(bool)
                )
            if mask.size != self.settings.expected_action_dim:
                raise ValueError(
                    f"Delta mask for robot_type={self.settings.robot_type!r} has dim={mask.size}; "
                    f"Track 3 {self.settings.platform} requires {self.settings.expected_action_dim}. "
                    "Set WSA_DELTA_MASK explicitly if the training schema is custom."
                )
            self.delta_mask = mask
        else:
            self.delta_mask = np.zeros(self.settings.expected_action_dim, dtype=bool)

        self._metadata.update(
            {
                "checkpoint_dir": str(self.checkpoint_dir),
                "model_type": self.model_type,
                "stats_path": str(stats_path),
                "stats_key": selected_stats_key,
                "state_stats_keys": list(self.state_scale.keys),
                "action_stats_keys": list(self.action_scale.keys),
                "action_mode": self.action_mode,
                "model_action_dim": model_action_dim,
                "normalization_mode": normalization_mode,
                "history_num_frames": self.history_num_frames,
                "history_stride_calls": self.history_stride_calls,
                "device": str(self.device),
                "dtype": str(self.dtype),
            }
        )
        self._backend_ready = True
        LOGGER.info("Loaded WorldArena WSA adapter: %s", self._metadata)

    def _reset_unlocked(self) -> None:
        self._history.clear()
        self._last_prompt = None
        self._infer_calls = 0
        if self._backend_ready:
            self.policy.reset()

    def reset(self, reset_info: Mapping[str, Any] | None = None) -> None:
        del reset_info
        with self._lock:
            self._reset_unlocked()

    def _prompt(self, new_obs: Mapping[str, Any]) -> str:
        if "prompt" not in new_obs:
            raise KeyError("WorldArena observation is missing required `prompt`")
        prompt = new_obs["prompt"]
        if not isinstance(prompt, str):
            raise TypeError(
                f"WorldArena `prompt` must be a string, got {type(prompt).__name__}"
            )
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("WorldArena `prompt` must be non-empty")
        return prompt

    def _state(self, new_obs: Mapping[str, Any]) -> np.ndarray:
        if self.settings.platform == "agilex":
            return extract_agilex_state(
                new_obs,
                expected_dim=self.settings.expected_state_dim,
            )
        return extract_franka_state(
            new_obs,
            expected_dim=self.settings.expected_state_dim,
        )

    def _raw_images(
        self, new_obs: Mapping[str, Any]
    ) -> tuple[tuple[np.ndarray | None, ...], tuple[bool, bool, bool]]:
        if "images" not in new_obs:
            raise KeyError("WorldArena observation is missing required `images`")
        images = new_obs["images"]
        if not isinstance(images, Mapping):
            raise TypeError("WorldArena observation `images` must be a mapping")
        head = _required_rgb_image(images, "cam_high")
        if self.settings.platform == "agilex":
            left = _required_rgb_image(images, "cam_left_wrist")
            right = _required_rgb_image(images, "cam_right_wrist")
            return (head, left, right), (True, True, True)

        # policy_guide.md documents cam_left_wrist as the primary Franka key.
        # The duplicated cam_wrist field is intentionally not used as a fallback.
        wrist = _required_rgb_image(images, "cam_left_wrist")
        return (head, wrist, None), (True, True, False)

    def _images(self, new_obs: Mapping[str, Any]) -> tuple[tuple[Any, Any, Any], tuple[bool, bool, bool]]:
        import torch

        required, present = self._raw_images(new_obs)
        raw_tensors: list[Any | None] = [
            None if value is None else _to_chw_float(value) for value in required
        ]
        resize_payload = {
            f"observation.images.image{index}": tensor
            for index, tensor in enumerate(raw_tensors)
            if tensor is not None
        }
        resized = self.resize_fn(resize_payload)
        head_resized = resized["observation.images.image0"]
        final: list[Any] = []
        for index in range(3):
            key = f"observation.images.image{index}"
            final.append(resized[key] if key in resized else torch.ones_like(head_resized))
        return (final[0], final[1], final[2]), present

    def _prepare_model_batch(
        self,
        prompt: str,
        state: np.ndarray,
        images: tuple[Any, Any, Any],
        camera_masks: tuple[bool, bool, bool],
    ) -> dict[str, Any]:
        import torch

        from lerobot.utils.constants import OBS_STATE

        normalized_state = torch.from_numpy(
            np.ascontiguousarray(self.state_scale.normalize(state).astype(np.float32))
        )
        self._history.append(
            _Snapshot(
                images=images,
                camera_masks=camera_masks,
                normalized_state=normalized_state,
            )
        )
        indices, valid = history_indices(
            len(self._history), self.history_num_frames, self.history_stride_calls
        )
        snapshots = list(self._history)
        selected = [snapshots[index] for index in indices]
        sample: dict[str, Any] = {"task": prompt}
        if self.model_type == WSA_MEMORY_MODEL_TYPE:
            sample[OBS_STATE] = torch.stack(
                [snapshot.normalized_state for snapshot in selected]
            )
            sample[MEMORY_HISTORY_VALID_MASK] = torch.tensor(valid, dtype=torch.bool)
        else:
            # Base was trained with a current state vector, not K-frame state
            # memory. At episode start the current RGB frame is duplicated into
            # the past slot by history_indices, yielding [past, current].
            sample[OBS_STATE] = normalized_state
        for image_index in range(3):
            key = f"observation.images.image{image_index}"
            sample[key] = torch.stack(
                [snapshot.images[image_index] for snapshot in selected]
            )
            # WSA exposes a view-level mask rather than a per-frame camera mask.
            # Disable a view for this request if any real selected frame for
            # that view was missing. Left-padding duplicates are ignored.
            view_valid = all(
                snapshot.camera_masks[image_index]
                for snapshot, history_valid in zip(selected, valid, strict=True)
                if history_valid
            )
            sample[f"{key}_mask"] = torch.tensor(view_valid, dtype=torch.bool)
        sample = self.processor_fn(sample)

        batch: dict[str, Any] = {}
        for key, value in sample.items():
            if not isinstance(value, torch.Tensor):
                continue
            value = value.unsqueeze(0)
            if value.is_floating_point():
                value = value.to(device=self.device, dtype=self.dtype, non_blocking=True)
            else:
                value = value.to(device=self.device, non_blocking=True)
            batch[key] = value
        return batch

    def _restore_actions(self, normalized_actions: np.ndarray, state: np.ndarray) -> np.ndarray:
        actions = self.action_scale.inverse(normalized_actions).astype(np.float32, copy=False)
        if self.action_mode == "delta":
            if state.size != actions.shape[-1]:
                raise ValueError(
                    f"Delta restoration requires state/action dims to match, got "
                    f"state={state.size}, action={actions.shape[-1]}"
                )
            actions = actions.copy()
            actions[:, self.delta_mask] += state[None, self.delta_mask]
        if self.settings.platform == "franka":
            actions = normalize_pose_quaternion(actions)
        if actions.ndim != 2 or actions.shape[1] != self.settings.expected_action_dim:
            raise RuntimeError(f"Invalid restored action shape: {actions.shape}")
        if not np.isfinite(actions).all():
            bad = np.argwhere(~np.isfinite(actions))[:10].tolist()
            raise FloatingPointError(f"Model produced non-finite actions at indices {bad}")
        return np.ascontiguousarray(actions, dtype=np.float32)

    def infer(self, new_obs: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(new_obs, Mapping):
            raise TypeError(f"new_obs must be a mapping, got {type(new_obs).__name__}")
        with self._lock:
            started = time.perf_counter()
            prompt = self._prompt(new_obs)
            if self._last_prompt is not None and prompt != self._last_prompt:
                LOGGER.info("Task prompt changed; clearing WSA episode history")
                self._reset_unlocked()
            self._last_prompt = prompt

            if self._infer_calls == 0:
                images = new_obs.get("images")
                image_shapes = (
                    {
                        str(key): tuple(np.asarray(value).shape)
                        for key, value in images.items()
                        if value is not None
                    }
                    if isinstance(images, Mapping)
                    else {}
                )
                LOGGER.info(
                    "First WorldArena observation: platform=%s state_fields=[%s] images=%s",
                    self.settings.platform,
                    _shape_summary(
                        new_obs,
                        (
                            "joint_qpos",
                            "joint_qpos_left",
                            "joint_qpos_right",
                            "left_end_pose",
                            "right_end_pose",
                        ),
                    ),
                    image_shapes,
                )

            state = self._state(new_obs)
            if self.settings.dry_run:
                # Validate the same official image schema even without loading a model.
                self._raw_images(new_obs)
                hold = (
                    state
                    if self.settings.platform == "agilex"
                    else normalize_pose_quaternion(state)
                )
                actions = np.repeat(
                    np.asarray(hold, dtype=np.float32)[None, :],
                    self.settings.execute_chunk_size,
                    axis=0,
                )
            else:
                import torch

                images, camera_masks = self._images(new_obs)
                batch = self._prepare_model_batch(prompt, state, images, camera_masks)
                with torch.inference_mode():
                    prediction, _ = self.policy.predict_action_chunk(
                        batch, decode_image=False
                    )
                if prediction.ndim != 3:
                    raise RuntimeError(
                        f"{self.model_type} returned unexpected shape={tuple(prediction.shape)}"
                    )
                normalized = (
                    prediction[
                        0,
                        : self.settings.execute_chunk_size,
                        : self.settings.expected_action_dim,
                    ]
                    .float()
                    .detach()
                    .cpu()
                    .numpy()
                )
                actions = self._restore_actions(normalized, state)

            self._infer_calls += 1
            infer_ms = (time.perf_counter() - started) * 1000.0
            metadata = self.metadata
            metadata["infer_calls"] = self._infer_calls
            metadata["task_id"] = str(new_obs.get("task_id") or "")
            return {
                "actions": actions,
                "policy_timing": {"infer_ms": float(infer_ms)},
                "policy_metadata": metadata,
            }


# Compatibility for existing imports and deployment code.
WSAMemoryWorldArenaAdapter = WSAWorldArenaAdapter
