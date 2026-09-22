"""A transparent current-day route witness for already-existing farm work.

This module deliberately does not choose new investments or emit Kaggle actions.
It takes tasks that are already required by the observed farm and either builds a
concrete same-day walking schedule or says that this greedy construction could
not fit them.  Movement through locked tiles is legal in Kaggriculture, so
Manhattan distance is the exact shortest walk on the board.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any


TURNS_PER_DAY = 24


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _xy(value: Any) -> tuple[int, int]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return _int(value[0]), _int(value[1])
    return 0, 0


def _manhattan(start: tuple[int, int], target: tuple[int, int]) -> int:
    return abs(start[0] - target[0]) + abs(start[1] - target[1])


def route_mandatory_tasks(observation: dict[str, Any], tasks: list[dict[str, Any]]) -> dict[str, Any]:
    """Return a greedy, concrete route witness for current mandatory tasks.

    Tasks at one tile are grouped: one actor walks there once, then does the
    listed one-turn tile operations in priority order. At each placement, the
    highest-priority remaining groups compete globally; ties go to the
    actor--task pair that can finish earliest. A completed witness proves this
    particular task set fits the observed workers and remaining day. A failure
    is intentionally weaker: another ordering may fit, and later market HIRE
    orders are not predicted.
    """

    player = _int(observation.get("player"))
    farms = list(observation.get("farms") or [{}, {}])
    farm = farms[player] if 0 <= player < len(farms) else {}
    starts = [_xy(farm.get("farmer")), *[_xy(pos) for pos in farm.get("hands") or []]]
    remaining = max(0, TURNS_PER_DAY - _int(observation.get("hour")))
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        grouped[_xy(task.get("target"))].append(dict(task))
    groups = [
        {
            "target": target,
            "tasks": sorted(items, key=lambda item: (-_int(item.get("priority")), str(item.get("kind")))),
            "priority": max((_int(item.get("priority")) for item in items), default=0),
        }
        for target, items in grouped.items()
    ]
    actors = [{"actor": index, "position": start, "used": 0, "route": []} for index, start in enumerate(starts)]
    unassigned: list[dict[str, Any]] = []
    while groups:
        priority = max(group["priority"] for group in groups)
        choices = []
        for group_index, group in enumerate(groups):
            if group["priority"] != priority:
                continue
            target = group["target"]
            actions = len(group["tasks"])
            for actor in actors:
                travel = _manhattan(actor["position"], target)
                finish = actor["used"] + travel + actions
                if finish <= remaining:
                    choices.append((finish, travel, actor["actor"], target, group_index, actor))
        if not choices:
            group_index = min((index for index, group in enumerate(groups) if group["priority"] == priority), key=lambda index: groups[index]["target"])
            group = groups.pop(group_index)
            unassigned.append(
                {
                    "target": list(group["target"]),
                    "priority": group["priority"],
                    "task_kinds": [str(task.get("kind")) for task in group["tasks"]],
                    "required_actions_at_tile": len(group["tasks"]),
                    "reason": "no observed actor can fit this highest-priority group after the greedy earlier assignments",
                }
            )
            continue
        finish, travel, _, target, group_index, actor = min(choices, key=lambda item: item[:4])
        group = groups.pop(group_index)
        actions = len(group["tasks"])
        actor["route"].append(
            {
                "start": list(actor["position"]),
                "target": list(target),
                "travel_steps": travel,
                "tile_actions": [str(task.get("kind")) for task in group["tasks"]],
                "task_count": actions,
                "finish_turn_within_day": finish,
            }
        )
        actor["position"] = target
        actor["used"] = finish
    return {
        "method": "dynamic priority-first greedy route witness using exact Manhattan walking distance",
        "remaining_turns_today": remaining,
        "all_mandatory_tasks_witnessed": not unassigned,
        "routes": [
            {
                "actor": actor["actor"],
                "start": list(starts[actor["actor"]]),
                "assigned_groups": actor["route"],
                "used_turns": actor["used"],
                "remaining_turns": remaining - actor["used"],
            }
            for actor in actors
        ],
        "unassigned_groups": unassigned,
        "remaining_worker_turns_after_witness": sum(remaining - actor["used"] for actor in actors),
        "limits": [
            "A success is a concrete schedule for the observed workers and tasks only.",
            "A failure means this greedy priority ordering found no schedule; it is not a mathematical proof that no schedule exists.",
            "No future HIRE, market action, future task, carrying, shed visit, or new investment is predicted here.",
        ],
    }
