#!/usr/bin/env python3
"""Offline action-chunk diagnostics for a trained WSA-Memory checkpoint.

This evaluator intentionally reuses the training dataset factory.  In particular,
it does not emulate WSA-Base's two-frame runtime input: every sample goes through
the WSA-Memory LeRobot v3 history, text-memory, normalization and padding pipeline.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import math
import random
import re
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Importing the WSA-Memory package registers both policy/config choices with
# draccus before train_config.json/config.json are decoded.
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.utils import load_json
from lerobot.policies.WSA_Memory.configuration_wsa_memory import (
    WSA_MEMORY,
    WSAMemoryConfig,
    WSAMemoryDatasetConfig,
)
from lerobot.policies.WSA_Memory.modeling_wsa_memory import WSAMemoryPolicy
from lerobot.policies.WSA_Memory.transform_wsa_memory import (
    MEMORY_ACTION_LOSS_MASK,
    WSAMemoryContextTransformFn,
)
from lerobot.policies.WSA_Base.transform_wsa_base import Qwen3_VLProcessorTransformFn
from lerobot.policies.factory import get_policy_class
from lerobot.transforms.constants import get_feature_mapping, get_mask_mapping
from lerobot.transforms.core import NormalizeTransformFn, Pi05ImageAugmentFn
from lerobot.utils.constants import ACTION, OBS_STATE, SAMPLE_ACTION_LOSS_MASK


LOGGER = logging.getLogger("wsa_memory_diagnostic")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run WSA-Memory action inference on a deterministic 10% sample of a "
            "multi-repository LeRobot v3 training collection."
        )
    )
    parser.add_argument("--checkpoint", required=True, help="Checkpoint/pretrained_model/run directory")
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Parent directory containing multiple local LeRobot v3 datasets",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for metrics and plots")
    parser.add_argument(
        "--stats-path",
        default=None,
        help="Aggregated stats JSON used in training (highest-priority normalization source)",
    )
    parser.add_argument(
        "--stats-root",
        default=None,
        help="Per-embodiment stats root: <robot_type>/<action_mode>/stats.json",
    )
    parser.add_argument("--qwen3-vl-path", default=None, help="Override Qwen3-VL model path")
    parser.add_argument("--processor-path", default=None, help="Override Qwen3-VL processor path")
    parser.add_argument("--cosmos-tokenizer-path", default=None, help="Override Cosmos tokenizer path")
    parser.add_argument("--action-mode", choices=("abs", "delta"), default=None)
    parser.add_argument("--fraction", type=float, default=0.10)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional debug cap applied after the 10%% sample; 0 evaluates all selected samples",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-curve-samples", type=int, default=8)
    parser.add_argument("--inference-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()

    if not 0.0 < args.fraction <= 1.0:
        parser.error("--fraction must be in (0, 1]")
    if args.max_samples < 0:
        parser.error("--max-samples cannot be negative")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if args.num_curve_samples < 0:
        parser.error("--num-curve-samples cannot be negative")
    if args.inference_steps is not None and args.inference_steps <= 0:
        parser.error("--inference-steps must be positive")
    return args


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]
    if device.type == "cpu" and dtype != torch.float32:
        LOGGER.warning("%s is not used on CPU; falling back to float32", name)
        return torch.float32
    return dtype


def resolve_checkpoint_dir(value: str | Path) -> Path:
    """Resolve all checkpoint layouts emitted by lerobot_train.py."""
    path = Path(value).expanduser()
    if not path.exists():
        # Keep parity with the existing RoboTwin evaluator for Hub checkpoint ids.
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(repo_id=str(value))).resolve()

    candidates = [
        path,
        path / "pretrained_model",
        path / "checkpoints" / "last" / "pretrained_model",
        path / "last" / "pretrained_model",
    ]
    for candidate in candidates:
        if (candidate / "config.json").is_file() and (candidate / "model.safetensors").is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Could not find config.json + model.safetensors under {path}. "
        "Pass a pretrained_model directory, a step checkpoint, or a training run directory."
    )


def discover_lerobot_v3_datasets(root: str | Path) -> list[Path]:
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    datasets: list[Path] = []
    for info_path in root.rglob("info.json"):
        if info_path.parent.name != "meta":
            continue
        dataset_dir = info_path.parent.parent
        if (dataset_dir / "data").is_dir() or (dataset_dir / "videos").is_dir():
            datasets.append(dataset_dir.resolve())
    datasets = sorted(set(datasets), key=lambda item: str(item))
    if not datasets:
        raise FileNotFoundError(f"No LeRobot v3 dataset containing meta/info.json found under {root}")
    return datasets


def _looks_like_feature_stats(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    return any(
        isinstance(value, dict) and ("mean" in value or "min" in value)
        for key, value in payload.items()
        if key == ACTION or key.startswith("action") or key.startswith("observation")
    )


@dataclass(frozen=True)
class StatsSource:
    path: str | None
    root: str | None
    description: str


def _materialize_checkpoint_stats(
    stats_file: Path,
    output_dir: Path,
    action_mode: str,
    description: str,
) -> StatsSource | None:
    payload = load_json(stats_file)
    if _looks_like_feature_stats(payload):
        return StatsSource(str(stats_file), None, description)

    nested = {
        str(key): value
        for key, value in payload.items()
        if _looks_like_feature_stats(value)
    }
    if not nested:
        return None
    if len(nested) == 1:
        selected_name, selected = next(iter(nested.items()))
        target = output_dir / "resolved_stats" / f"{selected_name}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(selected, indent=2), encoding="utf-8")
        return StatsSource(str(target), None, f"{description}, key={selected_name}")

    stats_root = output_dir / "resolved_stats"
    for robot_type, robot_stats in nested.items():
        target = stats_root / robot_type / action_mode / "stats.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(robot_stats, indent=2), encoding="utf-8")
    return StatsSource(None, str(stats_root), f"{description}, per-embodiment keys={sorted(nested)}")


def _existing_config_path(value: str | Path | None, checkpoint_dir: Path) -> Path | None:
    if value is None or not str(value).strip():
        return None
    path = Path(value).expanduser()
    candidates = [path] if path.is_absolute() else [checkpoint_dir / path, REPO_ROOT / path, path]
    return next((candidate.resolve() for candidate in candidates if candidate.exists()), None)


def resolve_stats_source(
    args: argparse.Namespace,
    checkpoint_dir: Path,
    train_config: TrainPipelineConfig | None,
    output_dir: Path,
    action_mode: str,
) -> StatsSource:
    if args.stats_path:
        path = Path(args.stats_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"--stats-path does not exist: {path}")
        source = _materialize_checkpoint_stats(path, output_dir, action_mode, "explicit --stats-path")
        if source is None:
            raise ValueError(f"No usable feature statistics found in {path}")
        return source
    if args.stats_root:
        root = Path(args.stats_root).expanduser().resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"--stats-root does not exist: {root}")
        return StatsSource(None, str(root), "explicit --stats-root")

    dataset_cfg = None if train_config is None else train_config.dataset
    if dataset_cfg is not None:
        configured_path = _existing_config_path(dataset_cfg.external_stats_path, checkpoint_dir)
        if configured_path is not None and configured_path.is_file():
            source = _materialize_checkpoint_stats(
                configured_path, output_dir, action_mode, "train_config external_stats_path"
            )
            if source is not None:
                return source
        configured_root = _existing_config_path(dataset_cfg.external_stats_root, checkpoint_dir)
        if configured_root is not None and configured_root.is_dir():
            return StatsSource(None, str(configured_root), "train_config external_stats_root")

    checkpoint_stats = checkpoint_dir / "stats.json"
    if checkpoint_stats.is_file():
        source = _materialize_checkpoint_stats(
            checkpoint_stats, output_dir, action_mode, "checkpoint stats.json"
        )
        if source is not None:
            return source

    if dataset_cfg is not None and not dataset_cfg.use_external_stats:
        return StatsSource(None, None, "per-dataset LeRobot meta/stats.json")
    raise FileNotFoundError(
        "Could not resolve the training normalization statistics. Pass --stats-path with the "
        "aggregated JSON used for training, or --stats-root with the per-embodiment stats root."
    )


def _disable_eval_augmentation_and_memory_dropout(dataset_cfg: WSAMemoryDatasetConfig) -> None:
    dataset_cfg.image_transforms.enable = False
    inputs = []
    found_memory_context = False
    for transform in dataset_cfg.data_transforms.inputs:
        if isinstance(transform, Pi05ImageAugmentFn):
            continue
        if isinstance(transform, WSAMemoryContextTransformFn):
            transform = replace(transform, training=False)
            found_memory_context = True
        inputs.append(transform)
    if not found_memory_context:
        raise ValueError("The checkpoint dataset pipeline has no WSAMemoryContextTransformFn")
    dataset_cfg.data_transforms = replace(dataset_cfg.data_transforms, inputs=inputs)


def _remove_runtime_config_fields(value: Any) -> tuple[Any, bool]:
    """Remove caches that must never participate in dataclass reconstruction."""
    if isinstance(value, dict):
        changed = "sidecar_index" in value
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            if key == "sidecar_index":
                continue
            cleaned_item, item_changed = _remove_runtime_config_fields(item)
            cleaned[key] = cleaned_item
            changed = changed or item_changed
        return cleaned, changed
    if isinstance(value, list):
        cleaned_list = []
        changed = False
        for item in value:
            cleaned_item, item_changed = _remove_runtime_config_fields(item)
            cleaned_list.append(cleaned_item)
            changed = changed or item_changed
        return cleaned_list, changed
    return value, False


def load_train_config_for_eval(checkpoint_dir: Path) -> TrainPipelineConfig:
    """Load train_config.json, tolerating old runtime-only sidecar fields."""
    try:
        return TrainPipelineConfig.from_pretrained(checkpoint_dir)
    except Exception as original_error:
        config_path = checkpoint_dir / "train_config.json"
        if not config_path.is_file():
            raise
        raw_config = json.loads(config_path.read_text(encoding="utf-8"))
        cleaned_config, changed = _remove_runtime_config_fields(raw_config)
        if not changed:
            raise original_error

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".json",
                encoding="utf-8",
                delete=False,
            ) as stream:
                json.dump(cleaned_config, stream)
                temporary_path = Path(stream.name)
            config = TrainPipelineConfig.from_pretrained(temporary_path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        LOGGER.info(
            "Loaded train_config.json after removing legacy runtime-only sidecar_index fields"
        )
        return config


def build_eval_config(
    args: argparse.Namespace,
    checkpoint_dir: Path,
    policy_config: WSAMemoryConfig,
    repo_ids_file: Path,
    stats_source: StatsSource,
    saved_train_config: TrainPipelineConfig | None,
) -> TrainPipelineConfig:
    if saved_train_config is not None:
        train_config = copy.deepcopy(saved_train_config)
        LOGGER.info("Loaded dataset configuration from %s/train_config.json", checkpoint_dir)
    else:
        LOGGER.warning("Constructing a fallback WSA-Memory dataset config")
        dataset_config = WSAMemoryDatasetConfig(
            repo_id="multidata_from_file",
            action_mode=args.action_mode,
        )
        train_config = TrainPipelineConfig(dataset=dataset_config, policy=policy_config)

    if not isinstance(train_config.dataset, WSAMemoryDatasetConfig):
        raise TypeError(
            "Diagnostic evaluation requires a WSA-Memory dataset config, got "
            f"{type(train_config.dataset).__name__}. Use the WSA-Memory checkpoint's train_config.json."
        )
    if args.action_mode is not None and args.action_mode != train_config.dataset.action_mode:
        raise ValueError(
            f"--action-mode={args.action_mode!r} differs from the checkpoint training mode "
            f"{train_config.dataset.action_mode!r}. Offline diagnostics must use the training mode "
            "because delta-action transforms and normalization statistics are mode-specific."
        )
    train_config.policy = policy_config
    train_config.dataset.repo_id = "multidata_from_file"
    train_config.dataset.repo_id_file = str(repo_ids_file)
    train_config.dataset.root = None
    train_config.dataset.episodes = None
    train_config.dataset.streaming = False
    train_config.dataset.dist_loading = False
    train_config.dataset.weight_rules_path = None
    train_config.dataset.action_mode = args.action_mode or train_config.dataset.action_mode
    train_config.dataset.use_external_stats = stats_source.path is not None or stats_source.root is not None
    train_config.dataset.external_stats_path = stats_source.path
    train_config.dataset.external_stats_root = stats_source.root
    train_config.num_workers = args.num_workers

    processor_path = args.processor_path or train_config.dataset.qwen3_vl_processor_path
    if args.qwen3_vl_path and not args.processor_path:
        processor_path = args.qwen3_vl_path
    train_config.dataset.qwen3_vl_processor_path = processor_path
    updated_inputs = []
    for transform in train_config.dataset.data_transforms.inputs:
        if isinstance(transform, Qwen3_VLProcessorTransformFn):
            transform = replace(transform, pretrained_model_name_or_path=processor_path)
        updated_inputs.append(transform)
    train_config.dataset.data_transforms = replace(
        train_config.dataset.data_transforms, inputs=updated_inputs
    )
    _disable_eval_augmentation_and_memory_dropout(train_config.dataset)
    return train_config


def allocate_stratified_counts(lengths: Sequence[int], fraction: float) -> list[int]:
    """Allocate round(total*fraction) samples using largest remainders."""
    target = max(1, min(sum(lengths), int(round(sum(lengths) * fraction))))
    exact = [length * fraction for length in lengths]
    counts = [min(length, int(math.floor(value))) for length, value in zip(lengths, exact, strict=True)]
    remaining = target - sum(counts)
    order = sorted(
        range(len(lengths)),
        key=lambda index: (-(exact[index] - counts[index]), -lengths[index], index),
    )
    for index in order:
        if remaining <= 0:
            break
        if counts[index] < lengths[index]:
            counts[index] += 1
            remaining -= 1
    if remaining:
        for index, length in enumerate(lengths):
            take = min(remaining, length - counts[index])
            counts[index] += take
            remaining -= take
            if remaining == 0:
                break
    return counts


def select_diagnostic_records(
    lengths: Sequence[int], fraction: float, max_samples: int, seed: int
) -> list[tuple[int, int]]:
    rng = np.random.default_rng(seed)
    counts = allocate_stratified_counts(lengths, fraction)
    records: list[tuple[int, int]] = []
    for dataset_index, (length, count) in enumerate(zip(lengths, counts, strict=True)):
        local_indices = rng.choice(length, size=count, replace=False)
        records.extend((dataset_index, int(local_index)) for local_index in local_indices)
    rng.shuffle(records)
    if max_samples > 0 and len(records) > max_samples:
        cap_indices = rng.choice(len(records), size=max_samples, replace=False)
        records = [records[int(index)] for index in cap_indices]
    return records


def _episode_for_local_index(dataset: Any, local_index: int) -> int:
    to_indices = np.asarray(dataset.meta.episodes["dataset_to_index"], dtype=np.int64)
    ordinal = int(np.searchsorted(to_indices, local_index, side="right"))
    try:
        episode = dataset.meta.episodes[ordinal]
        return int(episode.get("episode_index", ordinal))
    except (IndexError, KeyError, TypeError, AttributeError):
        return ordinal


class DiagnosticSubset(Dataset):
    def __init__(self, datasets: Sequence[Any], records: Sequence[tuple[int, int]]) -> None:
        self.datasets = list(datasets)
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        dataset_index, local_index = self.records[index]
        sample = self.datasets[dataset_index][local_index]
        episode_index = _episode_for_local_index(self.datasets[dataset_index], local_index)
        return sample, dataset_index, local_index, episode_index


@dataclass(frozen=True)
class ActionScale:
    mode: str
    first: np.ndarray
    second: np.ndarray
    labels: tuple[str, ...]

    @property
    def dim(self) -> int:
        return int(self.first.size)

    def inverse(self, value: torch.Tensor) -> torch.Tensor:
        first = torch.as_tensor(self.first, dtype=value.dtype, device=value.device)
        second = torch.as_tensor(self.second, dtype=value.dtype, device=value.device)
        if self.mode == "mean_std":
            return value * (second + 1e-6) + first
        if self.mode == "min_max":
            return value * (second - first + 1e-6) + first
        raise ValueError(f"Unsupported normalization mode: {self.mode}")


def action_scale_for_dataset(dataset: Any) -> ActionScale:
    return feature_scale_for_dataset(dataset, ACTION)


def state_scale_for_dataset(dataset: Any) -> ActionScale:
    return feature_scale_for_dataset(dataset, OBS_STATE)


def feature_scale_for_dataset(dataset: Any, feature: str) -> ActionScale:
    normalizer = next(
        (
            transform
            for transform in dataset._transform.transforms
            if isinstance(transform, NormalizeTransformFn)
        ),
        None,
    )
    if normalizer is None:
        raise ValueError(f"Dataset {dataset.repo_id} has no NormalizeTransformFn")
    feature_keys = get_feature_mapping(dataset.meta.robot_type, dataset.meta.features)[feature]
    first_name, second_name = ("mean", "std") if normalizer.mode == "mean_std" else ("min", "max")
    first_parts: list[np.ndarray] = []
    second_parts: list[np.ndarray] = []
    labels: list[str] = []
    for key in feature_keys:
        if key not in normalizer.norm_stats:
            raise KeyError(f"Normalization stats for {key!r} are missing in dataset {dataset.repo_id}")
        stats = normalizer.norm_stats[key]
        first = np.asarray(stats[first_name], dtype=np.float32).reshape(-1)
        second = np.asarray(stats[second_name], dtype=np.float32).reshape(-1)
        if first.shape != second.shape:
            raise ValueError(f"Stats shape mismatch for {key}: {first.shape} vs {second.shape}")
        first_parts.append(first)
        second_parts.append(second)
        labels.extend(f"{key}[{index}]" for index in range(first.size))
    return ActionScale(
        mode=normalizer.mode,
        first=np.concatenate(first_parts),
        second=np.concatenate(second_parts),
        labels=tuple(labels),
    )


def restore_original_action(
    action: torch.Tensor,
    current_state: torch.Tensor,
    delta_mask: torch.Tensor,
    action_mode: str,
) -> torch.Tensor:
    """Restore model-space raw delta actions to the dataset's original action space."""
    if action_mode != "delta":
        return action

    action_dim = int(action.shape[-1])
    mask = torch.as_tensor(delta_mask, dtype=torch.bool, device=action.device).flatten()
    if mask.numel() < action_dim:
        raise ValueError(
            f"Delta-action mask has {mask.numel()} dimensions, but action has {action_dim}"
        )
    mask = mask[:action_dim]
    state = current_state.to(dtype=action.dtype, device=action.device).flatten()
    if bool(mask.any()) and state.numel() < action_dim:
        raise ValueError(
            f"Current state has {state.numel()} dimensions, but delta action has {action_dim}"
        )

    base = torch.zeros(action_dim, dtype=action.dtype, device=action.device)
    base[mask] = state[:action_dim][mask]
    return action + base


def move_batch_to_device(batch: dict[str, Any], device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in batch.items():
        if not isinstance(value, torch.Tensor):
            result[key] = value
        elif value.is_floating_point():
            result[key] = value.to(device=device, dtype=dtype, non_blocking=True)
        else:
            result[key] = value.to(device=device, non_blocking=True)
    return result


class ErrorAccumulator:
    def __init__(self, horizon: int, action_dim: int) -> None:
        self.squared_sum = np.zeros((horizon, action_dim), dtype=np.float64)
        self.count = np.zeros((horizon, action_dim), dtype=np.int64)

    def update(self, squared_error: np.ndarray, valid_horizon: np.ndarray, valid_dim: int) -> None:
        horizon = min(squared_error.shape[0], self.squared_sum.shape[0])
        dim = min(valid_dim, squared_error.shape[1], self.squared_sum.shape[1])
        valid = valid_horizon[:horizon, None].astype(bool)
        self.squared_sum[:horizon, :dim] += squared_error[:horizon, :dim] * valid
        self.count[:horizon, :dim] += np.broadcast_to(valid, (horizon, dim))

    def merge(self, other: "ErrorAccumulator") -> None:
        self.squared_sum += other.squared_sum
        self.count += other.count

    def matrix(self) -> np.ndarray:
        return np.divide(
            self.squared_sum,
            self.count,
            out=np.full_like(self.squared_sum, np.nan),
            where=self.count > 0,
        )

    def mse(self) -> float:
        return float(self.squared_sum.sum() / max(int(self.count.sum()), 1))

    def summary(self) -> dict[str, Any]:
        matrix = self.matrix()
        total_count = int(self.count.sum())
        return {
            "mse": self.mse(),
            "count": total_count,
            "per_horizon_mse": _safe_axis_mean(self.squared_sum, self.count, axis=1),
            "per_dimension_mse": _safe_axis_mean(self.squared_sum, self.count, axis=0),
            "per_horizon_dimension_mse": [
                [None if not np.isfinite(value) else float(value) for value in row]
                for row in matrix
            ],
            "per_horizon_dimension_count": self.count.tolist(),
        }


def _safe_axis_mean(squared_sum: np.ndarray, count: np.ndarray, axis: int) -> list[float | None]:
    numerator = squared_sum.sum(axis=axis)
    denominator = count.sum(axis=axis)
    values = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=np.float64),
        where=denominator > 0,
    )
    return [None if np.isnan(value) else float(value) for value in values]


def sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "dataset"


def save_heatmap(matrix: np.ndarray, output_path: Path, title: str, labels: Sequence[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    visible_dims = np.where(np.isfinite(matrix).any(axis=0))[0]
    if visible_dims.size == 0:
        LOGGER.warning("Skipping empty heatmap %s", output_path)
        return
    matrix = matrix[:, : int(visible_dims[-1]) + 1]
    width = max(9.0, min(24.0, 0.45 * matrix.shape[1] + 5.0))
    height = max(6.0, min(24.0, 0.24 * matrix.shape[0] + 4.0))
    figure, axis = plt.subplots(figsize=(width, height), constrained_layout=True)
    image = axis.imshow(matrix, aspect="auto", origin="upper", interpolation="nearest", cmap="magma")
    axis.set_title(title)
    axis.set_xlabel("Action dimension")
    axis.set_ylabel("Horizon step")
    axis.set_xticks(np.arange(matrix.shape[1]))
    short_labels = [labels[index] if index < len(labels) else f"a{index}" for index in range(matrix.shape[1])]
    axis.set_xticklabels(short_labels, rotation=60, ha="right", fontsize=7)
    if matrix.shape[0] <= 25:
        axis.set_yticks(np.arange(matrix.shape[0]))
    figure.colorbar(image, ax=axis, label="MSE")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_action_curve(payload: dict[str, Any], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prediction = payload["prediction"]
    target = payload["target"]
    valid = payload["valid_horizon"]
    labels = payload["labels"]
    action_dim = prediction.shape[1]
    columns = min(4, action_dim)
    rows = int(math.ceil(action_dim / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 2.7 * rows), squeeze=False)
    horizon = np.arange(prediction.shape[0])
    for dim_index, axis in enumerate(axes.flat):
        if dim_index >= action_dim:
            axis.axis("off")
            continue
        pred_line = np.where(valid, prediction[:, dim_index], np.nan)
        target_line = np.where(valid, target[:, dim_index], np.nan)
        axis.plot(horizon, target_line, label="ground truth", linewidth=1.8)
        axis.plot(horizon, pred_line, label="prediction", linewidth=1.4, alpha=0.9)
        axis.set_title(labels[dim_index] if dim_index < len(labels) else f"action[{dim_index}]", fontsize=8)
        axis.grid(alpha=0.25)
        if dim_index == 0:
            axis.legend(fontsize=8)
    figure.suptitle(
        f"{payload['repo_name']} | episode={payload['episode_index']} frame={payload['local_index']} | "
        f"raw MSE={payload['raw_mse']:.6g}",
        fontsize=11,
    )
    figure.supxlabel("Horizon step")
    figure.supylabel("Original action")
    figure.tight_layout(rect=(0.02, 0.02, 1.0, 0.96))
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def load_policy(
    args: argparse.Namespace,
    checkpoint_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[WSAMemoryPolicy, WSAMemoryConfig]:
    config = PreTrainedConfig.from_pretrained(checkpoint_dir)
    if not isinstance(config, WSAMemoryConfig) or config.type != WSA_MEMORY:
        raise TypeError(
            f"Expected a WSA-Memory checkpoint (type={WSA_MEMORY!r}), got {type(config).__name__} "
            f"with type={getattr(config, 'type', None)!r}"
        )
    config.device = str(device)
    config.gradient_checkpointing = False
    # DA3 is a training-only teacher and is not needed for action inference.
    config.lambda_3d = 0.0
    if args.qwen3_vl_path:
        config.qwen3_vl_pretrained_path = args.qwen3_vl_path
    if args.cosmos_tokenizer_path:
        config.cosmos_tokenizer_path_or_name = args.cosmos_tokenizer_path
    if args.inference_steps is not None:
        config.num_inference_steps = args.inference_steps

    policy_cls = get_policy_class(config.type)
    if policy_cls is not WSAMemoryPolicy:
        raise TypeError(f"Policy registry resolved {policy_cls.__name__}, expected WSAMemoryPolicy")
    policy = policy_cls.from_pretrained(config=config, pretrained_name_or_path=checkpoint_dir)
    policy.to(device=device, dtype=dtype).eval()
    policy.reset()
    return policy, config


def evaluate(args: argparse.Namespace) -> None:
    try:
        import matplotlib  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "WSA-Memory diagnostic plots require matplotlib. Install it with "
            "`python -m pip install matplotlib`."
        ) from exc

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    curves_dir = output_dir / "action_curves"
    curves_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    checkpoint_dir = resolve_checkpoint_dir(args.checkpoint)
    datasets_found = discover_lerobot_v3_datasets(args.dataset_root)
    repo_ids_file = output_dir / "diagnostic_repo_ids.txt"
    repo_ids_file.write_text("".join(f"{path}\n" for path in datasets_found), encoding="utf-8")
    LOGGER.info("Checkpoint: %s", checkpoint_dir)
    LOGGER.info("Discovered %d LeRobot v3 datasets", len(datasets_found))

    # Load the policy config first because it defines the WSA-Memory temporal layout.
    policy_config = PreTrainedConfig.from_pretrained(checkpoint_dir)
    if not isinstance(policy_config, WSAMemoryConfig) or policy_config.type != WSA_MEMORY:
        raise TypeError(f"Checkpoint is not WSA-Memory: type={getattr(policy_config, 'type', None)!r}")
    try:
        saved_train_config = load_train_config_for_eval(checkpoint_dir)
    except Exception as exc:
        LOGGER.warning("train_config.json is unavailable while resolving stats: %s", exc)
        saved_train_config = None
    action_mode = args.action_mode or (
        saved_train_config.dataset.action_mode 
    )
    stats_source = resolve_stats_source(
        args, checkpoint_dir, saved_train_config, output_dir, action_mode
    )
    LOGGER.info("Normalization source: %s", stats_source.description)

    eval_config = build_eval_config(
        args,
        checkpoint_dir,
        policy_config,
        repo_ids_file,
        stats_source,
        saved_train_config,
    )
    dataset, _ = make_dataset(eval_config)
    underlying = list(dataset.datasets) if hasattr(dataset, "datasets") else [dataset]
    lengths = [len(item) for item in underlying]
    records = select_diagnostic_records(lengths, args.fraction, args.max_samples, args.seed)
    if not records:
        raise RuntimeError("The diagnostic split is empty")

    split_manifest = {
        "seed": args.seed,
        "fraction_before_debug_cap": args.fraction,
        "max_samples": args.max_samples,
        "total_training_samples": int(sum(lengths)),
        "selected_samples": len(records),
        "datasets": [
            {
                "dataset_index": index,
                "repo_id": str(item.repo_id),
                "num_frames": lengths[index],
                "selected_samples": sum(record[0] == index for record in records),
            }
            for index, item in enumerate(underlying)
        ],
        "records": [
            {"dataset_index": dataset_index, "local_index": local_index}
            for dataset_index, local_index in records
        ],
    }
    (output_dir / "diagnostic_split.json").write_text(
        json.dumps(split_manifest, indent=2), encoding="utf-8"
    )

    scales = [action_scale_for_dataset(item) for item in underlying]
    state_scales = (
        [state_scale_for_dataset(item) for item in underlying]
        if action_mode == "delta"
        else [None] * len(underlying)
    )
    delta_masks = [
        torch.as_tensor(
            get_mask_mapping(item.meta.robot_type, item.meta.features), dtype=torch.bool
        )
        for item in underlying
    ]
    subset = DiagnosticSubset(underlying, records)
    device = torch.device(
        args.device if not args.device.startswith("cuda") or torch.cuda.is_available() else "cpu"
    )
    dtype = resolve_dtype(args.dtype, device)
    loader_kwargs: dict[str, Any] = {
        "dataset": subset,
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "drop_last": False,
    }
    if args.num_workers > 0:
        loader_kwargs.update(prefetch_factor=2, persistent_workers=True)
    dataloader = DataLoader(**loader_kwargs)

    policy, policy_config = load_policy(args, checkpoint_dir, device, dtype)
    horizon = int(policy_config.chunk_size)
    max_action_dim = int(policy_config.output_features[ACTION].shape[0])
    metric_action_dim = max_action_dim
    if policy_config.mask_action_dim_padding_loss:
        metric_action_dim = min(metric_action_dim, int(policy_config.action_loss_valid_dim))
    normalized = ErrorAccumulator(horizon, max_action_dim)
    raw = ErrorAccumulator(horizon, max_action_dim)
    normalized_by_dataset = [ErrorAccumulator(horizon, max_action_dim) for _ in underlying]
    raw_by_dataset = [ErrorAccumulator(horizon, max_action_dim) for _ in underlying]
    per_sample_rows: list[dict[str, Any]] = []
    curve_payloads: list[dict[str, Any]] = []

    num_batches = int(math.ceil(len(subset) / args.batch_size))
    for batch_index, (batch, dataset_indices, local_indices, episode_indices) in enumerate(dataloader):
        targets = batch[ACTION].detach().to(dtype=torch.float32, device="cpu")
        states = batch[OBS_STATE].detach().to(dtype=torch.float32, device="cpu")
        temporal_mask = batch.get(MEMORY_ACTION_LOSS_MASK)
        if temporal_mask is None:
            temporal_mask = torch.ones(targets.shape[:2], dtype=torch.bool)
        else:
            temporal_mask = temporal_mask.detach().to(dtype=torch.bool, device="cpu")
        sample_mask = batch.get(SAMPLE_ACTION_LOSS_MASK)
        if sample_mask is not None:
            sample_mask = sample_mask.detach().cpu().reshape(sample_mask.shape[0], -1)[:, 0] > 0.5
            temporal_mask &= sample_mask[:, None]

        device_batch = move_batch_to_device(batch, device, dtype)
        with torch.inference_mode():
            predictions, _ = policy.predict_action_chunk(device_batch)
        predictions = predictions.detach().to(dtype=torch.float32, device="cpu")
        common_horizon = min(horizon, targets.shape[1], predictions.shape[1])

        for item_index in range(predictions.shape[0]):
            dataset_index = int(dataset_indices[item_index])
            local_index = int(local_indices[item_index])
            episode_index = int(episode_indices[item_index])
            scale = scales[dataset_index]
            action_dim = min(
                scale.dim,
                metric_action_dim,
                targets.shape[-1],
                predictions.shape[-1],
            )
            valid_horizon = temporal_mask[item_index, :common_horizon].numpy().astype(bool)
            prediction_norm = predictions[item_index, :common_horizon, :action_dim]
            target_norm = targets[item_index, :common_horizon, :action_dim]
            prediction_raw = scale.inverse(prediction_norm)
            target_raw = scale.inverse(target_norm)
            if action_mode == "delta":
                state_scale = state_scales[dataset_index]
                assert state_scale is not None
                state_history = states[item_index]
                current_state_norm = state_history[-1] if state_history.ndim >= 2 else state_history
                current_state_raw = state_scale.inverse(current_state_norm[: state_scale.dim])
                prediction_raw = restore_original_action(
                    prediction_raw,
                    current_state_raw,
                    delta_masks[dataset_index],
                    action_mode,
                )
                target_raw = restore_original_action(
                    target_raw,
                    current_state_raw,
                    delta_masks[dataset_index],
                    action_mode,
                )
            squared_norm = (prediction_norm - target_norm).square().numpy()
            squared_raw = (prediction_raw - target_raw).square().numpy()

            normalized.update(squared_norm, valid_horizon, action_dim)
            raw.update(squared_raw, valid_horizon, action_dim)
            normalized_by_dataset[dataset_index].update(squared_norm, valid_horizon, action_dim)
            raw_by_dataset[dataset_index].update(squared_raw, valid_horizon, action_dim)
            valid_step_count = int(valid_horizon.sum())
            valid_points = valid_step_count * action_dim
            sample_norm_mse = (
                float((squared_norm * valid_horizon[:, None]).sum() / valid_points)
                if valid_points > 0
                else None
            )
            sample_raw_mse = (
                float((squared_raw * valid_horizon[:, None]).sum() / valid_points)
                if valid_points > 0
                else None
            )
            repo_name = Path(str(underlying[dataset_index].repo_id)).name
            per_sample_rows.append(
                {
                    "dataset_index": dataset_index,
                    "repo_id": str(underlying[dataset_index].repo_id),
                    "episode_index": episode_index,
                    "local_index": local_index,
                    "valid_horizon_steps": valid_step_count,
                    "action_dim": action_dim,
                    "normalized_mse": sample_norm_mse,
                    "raw_mse": sample_raw_mse,
                }
            )
            if valid_points > 0 and len(curve_payloads) < args.num_curve_samples:
                curve_payloads.append(
                    {
                        "repo_name": repo_name,
                        "episode_index": episode_index,
                        "local_index": local_index,
                        "prediction": prediction_raw.numpy(),
                        "target": target_raw.numpy(),
                        "valid_horizon": valid_horizon,
                        "labels": scale.labels,
                        "raw_mse": sample_raw_mse,
                        "action_mode": action_mode,
                    }
                )

        if args.log_every > 0 and ((batch_index + 1) % args.log_every == 0 or batch_index + 1 == num_batches):
            LOGGER.info(
                "Evaluated %d/%d batches (%d/%d samples); normalized MSE=%.6g",
                batch_index + 1,
                num_batches,
                min((batch_index + 1) * args.batch_size, len(subset)),
                len(subset),
                normalized.mse(),
            )

    csv_path = output_dir / "per_sample_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_sample_rows[0]))
        writer.writeheader()
        writer.writerows(per_sample_rows)

    # Use labels from the widest embodiment; cells absent for another embodiment remain count-masked.
    widest_scale = max(scales, key=lambda item: item.dim)
    save_heatmap(
        normalized.matrix(),
        output_dir / "horizon_heatmap.png",
        "WSA-Memory normalized action MSE by horizon and dimension",
        widest_scale.labels,
    )
    save_heatmap(
        raw.matrix(),
        output_dir / "horizon_heatmap_raw.png",
        "WSA-Memory denormalized action MSE by horizon and dimension",
        widest_scale.labels,
    )
    for curve_index, payload in enumerate(curve_payloads):
        filename = (
            f"{curve_index:02d}_{sanitize_name(payload['repo_name'])}_"
            f"ep{payload['episode_index']}_frame{payload['local_index']}.png"
        )
        save_action_curve(payload, curves_dir / filename)

    summary = {
        "checkpoint": str(checkpoint_dir),
        "policy_type": policy_config.type,
        "dataset_root": str(Path(args.dataset_root).expanduser().resolve()),
        "action_mode": action_mode,
        "denormalized_action_space": "original_action",
        "normalization_source": {
            "description": stats_source.description,
            "path": stats_source.path,
            "root": stats_source.root,
        },
        "seed": args.seed,
        "fraction_before_debug_cap": args.fraction,
        "max_samples": args.max_samples,
        "total_training_samples": int(sum(lengths)),
        "evaluated_samples": len(per_sample_rows),
        "samples_with_valid_action_targets": sum(
            row["valid_horizon_steps"] > 0 for row in per_sample_rows
        ),
        "inference_steps": int(policy_config.num_inference_steps),
        "chunk_size": horizon,
        "normalized": normalized.summary(),
        "denormalized": raw.summary(),
        "per_dataset": [
            {
                "dataset_index": index,
                "repo_id": str(item.repo_id),
                "evaluated_samples": sum(row["dataset_index"] == index for row in per_sample_rows),
                "action_dim": min(scales[index].dim, metric_action_dim),
                "action_labels": list(scales[index].labels[:metric_action_dim]),
                "normalized": normalized_by_dataset[index].summary(),
                "denormalized": raw_by_dataset[index].summary(),
            }
            for index, item in enumerate(underlying)
        ],
        "artifacts": {
            "per_sample_metrics": str(csv_path),
            "split_manifest": str(output_dir / "diagnostic_split.json"),
            "normalized_horizon_heatmap": str(output_dir / "horizon_heatmap.png"),
            "denormalized_horizon_heatmap": str(output_dir / "horizon_heatmap_raw.png"),
            "action_curves_dir": str(curves_dir),
        },
        "interpretation": (
            "This is an in-training-data fit diagnostic, not a held-out robot success rate. "
            "WSA-Memory flow inference is stochastic, so MSE also depends on the recorded seed."
        ),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info("Done. normalized MSE=%.6g, raw MSE=%.6g", summary["normalized"]["mse"], summary["denormalized"]["mse"])
    LOGGER.info("Summary: %s", summary_path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    evaluate(parse_args())


if __name__ == "__main__":
    main()
