"""Daily task inventory and resource reservations for the portfolio agent.

The module intentionally stops before worker routing and action emission.  It
answers the first operational question safely: before buying or planting
anything new today, what existing work and shared resources must be protected?
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .portfolio_model import PRODUCTS, choose_portfolio
from .task_router import route_mandatory_tasks


CROP_FIRST_YIELD = {"WHEAT": 2, "CARROT": 2, "TOMATO": 8, "STRAWBERRY": 10, "MELON": 10}


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Task:
    kind: str
    target: tuple[int, int]
    priority: int
    reason: str
    requires: dict[str, int]

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["target"] = list(self.target)
        return result


def _tasks_for_farm(farm: dict[str, Any], day: int) -> list[Task]:
    tasks: list[Task] = []
    for y, row in enumerate(farm.get("tiles") or []):
        for x, tile in enumerate(row):
            if tile == "LOCKED" or tile is None or not isinstance(tile, dict):
                continue
            target = (x, y)
            if "animal" in tile:
                if not tile.get("fed_today", False):
                    tasks.append(Task("FEED", target, 100, "existing animal needs today's wheat feed", {"WHEAT": 1}))
                if not tile.get("cared_today", False):
                    tasks.append(Task("CARE", target, 55, "existing animal can receive today's care", {}))
                if _int(tile.get("yield_units")) > 0:
                    tasks.append(Task("HARVEST", target, 90, "animal output is ready to collect", {}))
                if tile.get("fertilizer_available", False):
                    tasks.append(Task("COLLECT_FERTILIZER", target, 45, "animal fertilizer is available", {}))
                continue
            if tile.get("kind") == "WEED":
                tasks.append(Task("DIG", target, 70, "weed blocks a usable tile", {}))
                continue
            if tile.get("kind") != "PLANT":
                continue
            crop = str(tile.get("crop") or "")
            age = day - _int(tile.get("planted_day"))
            if _int(tile.get("yield_units")) > 0 and age >= CROP_FIRST_YIELD.get(crop, 99):
                tasks.append(Task("HARVEST", target, 95, "crop output is ready before it decays or blocks replanting", {}))
            if not tile.get("watered_today", False):
                priority = 80 if age >= max(0, CROP_FIRST_YIELD.get(crop, 99) - 1) else 60
                tasks.append(Task("WATER", target, priority, "existing crop can gain or preserve today's production", {}))
    return sorted(tasks, key=lambda task: (-task.priority, task.kind, task.target))


def plan_day(observation: dict[str, Any]) -> dict[str, Any]:
    """Return today's protected work and remaining room for new investment.

    This output remains advisory.  It makes no assumption that a task can be
    routed today; routing feasibility is deliberately a later module.
    """

    player = _int(observation.get("player"))
    farms = list(observation.get("farms") or [{}, {}])
    farm = farms[player]
    private = dict(observation.get("private") or {})
    day = _int(observation.get("day"))
    tasks = _tasks_for_farm(farm, day)
    feed_reserve = sum(task.requires.get("WHEAT", 0) for task in tasks if task.kind == "FEED")
    shed = dict(private.get("shed") or {})
    carried = sum(max(0, _int(amount)) for inventory in private.get("inventories") or [] for amount in dict(inventory or {}).values())
    shed_used = sum(max(0, _int(amount)) for amount in shed.values())
    shed_capacity = _int((observation.get("configuration") or {}).get("shedCapacity"), 100)
    actors = 1 + len(farm.get("hands") or [])
    remaining_steps_today = max(0, 24 - _int(observation.get("hour")))
    mandatory = [task for task in tasks if task.priority >= 80]
    route_witness = route_mandatory_tasks(observation, [task.as_dict() for task in mandatory])
    portfolio = choose_portfolio(observation)
    return {
        "day": day,
        "step": _int(observation.get("step")),
        "mandatory_tasks": [task.as_dict() for task in mandatory],
        "optional_maintenance_tasks": [task.as_dict() for task in tasks if task.priority < 80],
        "reservation": {
            "wheat_for_feed": feed_reserve,
            "wheat_currently_available": _int((private.get("shed") or {}).get("WHEAT")),
            "feed_reserve_satisfied": _int((private.get("shed") or {}).get("WHEAT")) >= feed_reserve,
            "shed_used_including_carried": shed_used + carried,
            "shed_capacity": shed_capacity,
            "shed_free_before_new_work": max(0, shed_capacity - shed_used - carried),
            "actors_currently_available": actors,
            "remaining_action_slots_if_no_travel": actors * remaining_steps_today,
            "mandatory_task_count": len(mandatory),
            "remaining_worker_turns_after_route_witness": route_witness["remaining_worker_turns_after_witness"],
        },
        "mandatory_route_witness": route_witness,
        "portfolio": portfolio,
        "planner_limits": [
            "The route witness only covers currently observed mandatory work and observed workers; it does not predict hires, market actions, carrying, or future work.",
            "Existing care and harvest tasks are listed before discretionary 70/30 investment targets.",
            "The next implementation stage must assign routes, enforce travel-time feasibility, and emit legal actions.",
        ],
    }
