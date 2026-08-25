"""Shared, model-agnostic subtask sidecar and prompt helpers.

Only text that is observable at the current frame is represented here: the
episode instruction, completed subtasks and the current subtask.  Legacy
sidecars may contain extra keys; they are deliberately ignored.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence


def _require_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer, got {value!r}")
    return value


def _require_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class Subtask:
    subtask_id: int
    subtask_text: str
    start_frame: int
    end_frame: int

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
        return cls(
            subtask_id=_require_int(payload.get("subtask_id"), f"{prefix}.subtask_id"),
            subtask_text=_require_text(payload.get("subtask_text"), f"{prefix}.subtask_text"),
            start_frame=start_frame,
            end_frame=end_frame,
        )


@dataclass(frozen=True)
class EpisodeMemory:
    episode_id: str
    episode_index: int
    task_instruction: str
    subtasks: tuple[Subtask, ...]

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
            Subtask.from_json(item, episode_index=episode_index, item_index=index)
            for index, item in enumerate(raw_subtasks)
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

        return cls(
            episode_id=_require_text(
                payload.get("episode_id"), f"episode_index={episode_index}.episode_id"
            ),
            episode_index=episode_index,
            task_instruction=_require_text(
                payload.get("task_instruction"),
                f"episode_index={episode_index}.task_instruction",
            ),
            subtasks=subtasks,
        )

    def context_at(self, frame_index: int) -> tuple[tuple[Subtask, ...], Subtask | None]:
        frame_index = _require_int(frame_index, "frame_index")
        completed = tuple(subtask for subtask in self.subtasks if subtask.end_frame < frame_index)
        current = next(
            (
                subtask
                for subtask in self.subtasks
                if subtask.start_frame <= frame_index <= subtask.end_frame
            ),
            None,
        )
        return completed, current


class SidecarIndex:
    """Immutable, episode-indexed view of one dataset's ``meta/sidecar.json``."""

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
    ) -> "SidecarIndex":
        root = Path(dataset_root)
        sidecar_path = root / "meta" / "sidecar.json"
        if not sidecar_path.is_file():
            raise FileNotFoundError(
                f"Subtask sidecar is missing for dataset root '{root}'. "
                f"Expected: '{sidecar_path}'."
            )
        with sidecar_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            raise ValueError(f"{sidecar_path} must contain a top-level array of episode objects")
        episodes = [
            EpisodeMemory.from_json(item, allow_overlaps=allow_overlaps) for item in payload
        ]
        return cls(episodes, sidecar_path)

    def get(self, episode_index: int) -> EpisodeMemory | None:
        return self._episodes.get(int(episode_index))

    def require(self, episode_index: int) -> EpisodeMemory:
        episode = self.get(episode_index)
        if episode is None:
            raise KeyError(f"episode_index={episode_index} is not present in {self.source_path}")
        return episode

    def __len__(self) -> int:
        return len(self._episodes)


@dataclass(frozen=True)
class PromptMemory:
    overall_task: str
    completed_subtasks: tuple[str, ...] = ()
    current_subtask: str = "unknown"
    earlier_completed_omitted: bool = False


def prompt_memory_from_episode(
    episode: EpisodeMemory,
    frame_index: int,
    *,
    sample_task: str | None,
    max_completed_subtasks: int,
) -> tuple[PromptMemory, Subtask | None]:
    completed, current = episode.context_at(frame_index)
    completed_text = tuple(item.subtask_text for item in completed)
    omitted = len(completed_text) > max_completed_subtasks
    if max_completed_subtasks == 0:
        completed_text = ()
    elif omitted:
        completed_text = completed_text[-max_completed_subtasks:]

    # The LeRobot task is the same instruction used by task_only mode and is
    # therefore authoritative.  The sidecar copy remains a validated fallback.
    overall_task = (
        sample_task.strip()
        if isinstance(sample_task, str) and sample_task.strip()
        else episode.task_instruction
    )
    return (
        PromptMemory(
            overall_task=overall_task,
            completed_subtasks=completed_text,
            current_subtask="unknown" if current is None else current.subtask_text,
            earlier_completed_omitted=omitted,
        ),
        current,
    )


def prompt_memory_from_external(
    *,
    overall_task: str,
    completed_subtasks: Sequence[str] | None = None,
    current_subtask: str | None = None,
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
        completed_subtasks=completed,
        current_subtask=(current_subtask or "unknown").strip() or "unknown",
        earlier_completed_omitted=omitted,
    )


def deterministic_prompt_dropout_draws(
    seed: int, episode_index: int, frame_index: int
) -> tuple[float, float, float]:
    """Draw stable task/completed/current values across workers and processes."""
    mixed = (
        (int(seed) & 0xFFFFFFFFFFFFFFFF)
        ^ ((episode_index + 0x9E3779B9) * 0x85EBCA6B)
        ^ ((frame_index + 0xC2B2AE35) * 0x27D4EB2F)
    ) & 0xFFFFFFFFFFFFFFFF
    rng = random.Random(mixed)
    return rng.random(), rng.random(), rng.random()


def apply_prompt_dropout(
    memory: PromptMemory,
    *,
    task_dropped: bool,
    completed_dropped: bool,
    current_dropped: bool,
) -> PromptMemory:
    """Apply already-sampled dropout decisions without hidden global RNG state."""
    updates: dict[str, Any] = {}
    if task_dropped:
        updates["overall_task"] = "unknown"
    if completed_dropped:
        updates.update(completed_subtasks=(), earlier_completed_omitted=False)
    if current_dropped:
        updates["current_subtask"] = "unknown"
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
            f"Task instruction: {memory.overall_task.rstrip('.')}.",
            completed_line,
            f"Current subtask: {memory.current_subtask.rstrip('.')}.",
        ]
    )
