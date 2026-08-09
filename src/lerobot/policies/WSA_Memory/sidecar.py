"""Validated sidecar metadata and pure text/action-memory helpers.

The model never performs sidecar I/O.  A :class:`SidecarIndex` is built once
when an individual LeRobot dataset is constructed and is then reused by all
samples (and copied into DataLoader workers with the dataset object).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence


HAND_MAP = {0: "left", 1: "right", 2: "both"}
DEFAULT_RIGIDITY_MAP = {0: "rigid", 1: "flexible"}


def _require_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer, got {value!r}")
    return value


def _require_text(value: Any, field_name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def normalize_enum_map(mapping: Mapping[int | str, str], field_name: str) -> dict[int, str]:
    """Normalize JSON/YAML string keys to integers at one well-defined boundary."""
    result: dict[int, str] = {}
    for raw_key, raw_value in mapping.items():
        try:
            key = int(raw_key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} key {raw_key!r} is not integer-like") from exc
        if isinstance(raw_key, float) and not raw_key.is_integer():
            raise ValueError(f"{field_name} key {raw_key!r} is not an integer")
        if key in result:
            raise ValueError(
                f"{field_name} contains duplicate normalized key {key}"
            )
        result[key] = _require_text(raw_value, f"{field_name}[{raw_key!r}]")
    return result


@dataclass(frozen=True)
class Subtask:
    subtask_id: int
    subtask_text: str
    start_frame: int
    end_frame: int
    confidence: float | None
    hand_used: int | None
    object_rigidity: int | None

    @classmethod
    def from_json(cls, payload: Any, *, episode_index: int, item_index: int) -> "Subtask":
        if not isinstance(payload, dict):
            raise ValueError(
                f"episode_index={episode_index} subtasks[{item_index}] must be an object"
            )
        prefix = f"episode_index={episode_index} subtasks[{item_index}]"
        start_frame = _require_int(payload.get("start_frame"), f"{prefix}.start_frame")
        end_frame = _require_int(payload.get("end_frame"), f"{prefix}.end_frame")
        if start_frame < 0 or end_frame < start_frame:
            raise ValueError(
                f"{prefix} has invalid closed interval [{start_frame}, {end_frame}]"
            )

        confidence = payload.get("confidence")
        if confidence is not None:
            if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
                raise ValueError(f"{prefix}.confidence must be numeric when present")
            confidence = float(confidence)

        hand_used = payload.get("hand_used")
        if hand_used is not None:
            hand_used = _require_int(hand_used, f"{prefix}.hand_used")
        object_rigidity = payload.get("object_rigidity")
        if object_rigidity is not None:
            object_rigidity = _require_int(
                object_rigidity, f"{prefix}.object_rigidity"
            )

        return cls(
            subtask_id=_require_int(payload.get("subtask_id"), f"{prefix}.subtask_id"),
            subtask_text=_require_text(payload.get("subtask_text"), f"{prefix}.subtask_text"),
            start_frame=start_frame,
            end_frame=end_frame,
            confidence=confidence,
            hand_used=hand_used,
            object_rigidity=object_rigidity,
        )


@dataclass(frozen=True)
class EpisodeMemory:
    episode_id: str
    episode_index: int
    task_instruction: str
    sample_interval_sec: float | None
    episode_summary: str | None
    subtasks: tuple[Subtask, ...]
    scene: str

    @classmethod
    def from_json(cls, payload: Any, *, allow_overlaps: bool) -> "EpisodeMemory":
        if not isinstance(payload, dict):
            raise ValueError("Each sidecar entry must be an episode object")
        episode_index = _require_int(payload.get("episode_index"), "episode_index")
        if episode_index < 0:
            raise ValueError(f"episode_index must be non-negative, got {episode_index}")
        raw_subtasks = payload.get("subtasks")
        if not isinstance(raw_subtasks, list):
            raise ValueError(f"episode_index={episode_index}.subtasks must be an array")
        subtasks = tuple(
            Subtask.from_json(item, episode_index=episode_index, item_index=i)
            for i, item in enumerate(raw_subtasks)
        )

        previous: Subtask | None = None
        seen_subtask_ids: set[int] = set()
        for subtask in subtasks:
            if subtask.subtask_id in seen_subtask_ids:
                raise ValueError(
                    f"episode_index={episode_index} has duplicate subtask_id={subtask.subtask_id}"
                )
            seen_subtask_ids.add(subtask.subtask_id)
            if previous is not None:
                if subtask.start_frame < previous.start_frame:
                    raise ValueError(
                        f"episode_index={episode_index} subtasks are not sorted by start_frame"
                    )
                if not allow_overlaps and subtask.start_frame <= previous.end_frame:
                    raise ValueError(
                        f"episode_index={episode_index} has overlapping closed intervals "
                        f"[{previous.start_frame}, {previous.end_frame}] and "
                        f"[{subtask.start_frame}, {subtask.end_frame}]"
                    )
            previous = subtask

        sample_interval_sec = payload.get("sample_interval_sec")
        if sample_interval_sec is not None:
            if isinstance(sample_interval_sec, bool) or not isinstance(
                sample_interval_sec, (int, float)
            ):
                raise ValueError(
                    f"episode_index={episode_index}.sample_interval_sec must be numeric"
                )
            sample_interval_sec = float(sample_interval_sec)
            if sample_interval_sec <= 0:
                raise ValueError(
                    f"episode_index={episode_index}.sample_interval_sec must be positive"
                )

        summary = payload.get("episode_summary")
        if summary is not None:
            summary = _require_text(
                summary, f"episode_index={episode_index}.episode_summary", allow_empty=True
            )

        return cls(
            episode_id=_require_text(
                payload.get("episode_id"), f"episode_index={episode_index}.episode_id"
            ),
            episode_index=episode_index,
            task_instruction=_require_text(
                payload.get("task_instruction"),
                f"episode_index={episode_index}.task_instruction",
            ),
            sample_interval_sec=sample_interval_sec,
            episode_summary=summary,
            subtasks=subtasks,
            scene=_require_text(
                payload.get("scene", "unknown"),
                f"episode_index={episode_index}.scene",
            ),
        )

    def context_at(self, frame_index: int) -> tuple[tuple[Subtask, ...], Subtask | None]:
        frame_index = _require_int(frame_index, "frame_index")
        completed = tuple(s for s in self.subtasks if s.end_frame < frame_index)
        current = next(
            (
                s
                for s in self.subtasks
                if s.start_frame <= frame_index <= s.end_frame
            ),
            None,
        )
        return completed, current


class SidecarIndex:
    """Immutable, episode-indexed view of one dataset's sidecar."""

    def __init__(self, episodes: Sequence[EpisodeMemory], source_path: Path):
        index: dict[int, EpisodeMemory] = {}
        episode_ids: set[str] = set()
        for episode in episodes:
            if episode.episode_index in index:
                raise ValueError(
                    f"Duplicate episode_index={episode.episode_index} in {source_path}"
                )
            if episode.episode_id in episode_ids:
                raise ValueError(f"Duplicate episode_id={episode.episode_id!r} in {source_path}")
            index[episode.episode_index] = episode
            episode_ids.add(episode.episode_id)
        self._episodes = index
        self.source_path = source_path

    @classmethod
    def load(
        cls,
        dataset_root: str | Path,
        *,
        allow_overlaps: bool = False,
        allow_missing: bool = False,
    ) -> "SidecarIndex | None":
        root = Path(dataset_root)
        sidecar_path = root / "meta" / "sidecar.json"
        if not sidecar_path.is_file():
            if allow_missing:
                return None
            raise FileNotFoundError(
                f"WSA-Memory sidecar is missing for dataset root '{root}'. "
                f"Expected: '{sidecar_path}'. Set allow_missing_sidecar=true only for "
                "an explicit task-only fallback."
            )
        with sidecar_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            raise ValueError(
                f"{sidecar_path} must contain a top-level array of episode objects"
            )
        episodes = [
            EpisodeMemory.from_json(item, allow_overlaps=allow_overlaps)
            for item in payload
        ]
        return cls(episodes, sidecar_path)

    def get(self, episode_index: int) -> EpisodeMemory | None:
        return self._episodes.get(int(episode_index))

    def require(self, episode_index: int) -> EpisodeMemory:
        episode = self.get(episode_index)
        if episode is None:
            raise KeyError(
                f"episode_index={episode_index} is not present in {self.source_path}"
            )
        return episode

    def __len__(self) -> int:
        return len(self._episodes)


@dataclass(frozen=True)
class PromptMemory:
    overall_task: str
    scene: str = "unknown"
    completed_subtasks: tuple[str, ...] = ()
    current_subtask: str = "unknown"
    interaction_arm: str = "unknown"
    object_property: str = "unknown"
    earlier_completed_omitted: bool = False


def _external_enum(value: int | str | None, mapping: Mapping[int, str]) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, bool):
        return "unknown"
    if isinstance(value, int):
        return mapping.get(value, "unknown")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return "unknown"


def prompt_memory_from_episode(
    episode: EpisodeMemory,
    frame_index: int,
    *,
    sample_task: str | None,
    hand_map: Mapping[int, str],
    rigidity_map: Mapping[int, str],
    max_completed_subtasks: int,
) -> tuple[PromptMemory, Subtask | None]:
    completed, current = episode.context_at(frame_index)
    completed_text = tuple(item.subtask_text for item in completed)
    omitted = len(completed_text) > max_completed_subtasks
    if max_completed_subtasks == 0:
        completed_text = ()
    elif omitted:
        completed_text = completed_text[-max_completed_subtasks:]

    overall_task = (
        sample_task.strip()
        if isinstance(sample_task, str) and sample_task.strip()
        else episode.task_instruction
    )
    if current is None:
        current_text = arm = rigidity = "unknown"
    else:
        current_text = current.subtask_text
        arm = _external_enum(current.hand_used, hand_map)
        rigidity = _external_enum(current.object_rigidity, rigidity_map)

    return (
        PromptMemory(
            overall_task=overall_task,
            scene=episode.scene,
            completed_subtasks=completed_text,
            current_subtask=current_text,
            interaction_arm=arm,
            object_property=rigidity,
            earlier_completed_omitted=omitted,
        ),
        current,
    )


def prompt_memory_from_external(
    *,
    overall_task: str,
    scene: str | None = None,
    completed_subtasks: Sequence[str] | None = None,
    current_subtask: str | None = None,
    hand_used: int | str | None = None,
    object_rigidity: int | str | None = None,
    hand_map: Mapping[int, str] = HAND_MAP,
    rigidity_map: Mapping[int, str] = DEFAULT_RIGIDITY_MAP,
    max_completed_subtasks: int = 8,
) -> PromptMemory:
    completed = tuple(
        str(item).strip() for item in (completed_subtasks or ()) if str(item).strip()
    )
    omitted = len(completed) > max_completed_subtasks
    if max_completed_subtasks == 0:
        completed = ()
    elif omitted:
        completed = completed[-max_completed_subtasks:]
    return PromptMemory(
        overall_task=_require_text(overall_task, "overall_task"),
        scene=(scene or "unknown").strip() or "unknown",
        completed_subtasks=completed,
        current_subtask=(current_subtask or "unknown").strip() or "unknown",
        interaction_arm=_external_enum(hand_used, hand_map),
        object_property=_external_enum(object_rigidity, rigidity_map),
        earlier_completed_omitted=omitted,
    )


def apply_prompt_dropout(
    memory: PromptMemory,
    *,
    completed_dropped: bool,
    current_block_dropped: bool,
    scene_dropped: bool,
) -> PromptMemory:
    """Apply already-sampled dropout decisions without hidden global RNG state."""
    updates: dict[str, Any] = {}
    if completed_dropped:
        updates.update(completed_subtasks=(), earlier_completed_omitted=False)
    if current_block_dropped:
        updates.update(
            current_subtask="unknown",
            interaction_arm="unknown",
            object_property="unknown",
        )
    if scene_dropped:
        updates["scene"] = "unknown"
    return replace(memory, **updates)


def build_memory_prompt(memory: PromptMemory) -> str:
    if memory.completed_subtasks:
        recent = "; ".join(memory.completed_subtasks)
        if memory.earlier_completed_omitted:
            completed_line = (
                "Earlier completed subtasks omitted; recent completed subtasks: "
                f"{recent}."
            )
        else:
            completed_line = f"Completed subtasks: {recent}."
    else:
        completed_line = "Completed subtasks: none."
    return "\n".join(
        [
            f"Overall task: {memory.overall_task.rstrip('.')}.",
            f"Scene: {memory.scene.rstrip('.')}.",
            completed_line,
            f"Current subtask: {memory.current_subtask.rstrip('.')}.",
            f"Interaction arm: {memory.interaction_arm.rstrip('.')}.",
            f"Object property: {memory.object_property.rstrip('.')}.",
        ]
    )


def build_action_loss_mask(
    *,
    action_is_pad: Sequence[bool],
    frame_index: int,
    action_frame_offsets: Sequence[int],
    current_subtask: Subtask | None,
    apply_subtask_boundary: bool,
) -> list[bool]:
    """Combine LeRobot action padding and an optional oracle subtask boundary."""
    if len(action_is_pad) != len(action_frame_offsets):
        raise ValueError(
            "action_is_pad and action_frame_offsets must have the same horizon length"
        )
    valid = [not bool(is_pad) for is_pad in action_is_pad]
    if apply_subtask_boundary and current_subtask is not None:
        valid = [
            keep and frame_index + int(offset) <= current_subtask.end_frame
            for keep, offset in zip(valid, action_frame_offsets, strict=True)
        ]
    return valid
