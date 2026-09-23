"""Run a 5×5 region route one real action per hour.

Hour 0 still uses the existing hire and market orders. It does not walk a
guessed field route. From hour 1 the farmer and hands do only the actions
`plan_region_routes` already expanded. This module does not search for a path
of its own, and it does not invent where a new hand will appear.
"""

from __future__ import annotations

from typing import Any

from .region_route import RegionRoutePlan, RegionWorker, plan_region_routes
from .route14_phase1 import _market_orders, choose_day_route
from .route14_state import ANIMAL_NAMES, PENDING, SCHEDULED, TaskAssignment, parse_world, settle_tasks, shed_doors, worker_name
from .task_grid import (
    BUILD_COOP,
    BUILD_PASTURE,
    CARE,
    FEED,
    HARVEST,
    PLACE_ANIMAL,
    PLANT,
    WATER,
    TaskGrid,
    apply_assignments,
    build_task_grid,
)


REGION_SIZE = 5
MAX_REGION_WORKERS = 4
FIELD_OPERATIONS = {WATER, FEED, CARE, HARVEST, PLANT, BUILD_COOP, BUILD_PASTURE, PLACE_ANIMAL}
_MOVES = {
    "NORTH": (0, -1),
    "SOUTH": (0, 1),
    "EAST": (1, 0),
    "WEST": (-1, 0),
}


def make_region_phase1_agent():
    """Hour 0 hires and sells. Hour 1 plans from the people actually on the board."""

    state: dict[str, Any] = {
        "day": None,
        "grid": None,
        "plan": None,
        "routes_by_worker_id": {},
        "expected_positions": {},
        "planned_workers": [],
        "needs_replan": False,
        "market_route": None,
        "market_state": {"wheat_bought": False},
        "tasks": [],
        "world": None,
    }

    def agent(observation: dict[str, Any], configuration: dict[str, Any] | None = None) -> dict[str, Any]:
        del configuration
        day = int(observation.get("day") or 0)
        hour = int(observation.get("hour") or 0)
        if state["day"] != day:
            _reset_day(state, day)
        if hour == 0:
            return _hour_zero(observation, state)
        world = parse_world(observation)
        crew = _crew(world)
        grid = build_task_grid(world, state["grid"], observation)
        state["needs_replan"] = state["plan"] is not None and _diverged(crew, state)
        if state["plan"] is None or state["needs_replan"]:
            _reopen_scheduled(grid)
            origin = _region_origin(grid)
            plan = plan_region_routes(
                grid,
                crew,
                region_size=REGION_SIZE,
                origin=origin,
                start_hour=hour,
                end_hour=23,
                shed_wheat=_shed_wheat(world),
                shed_coords=shed_doors(grid.width),
                shed_animals=_shed_animals(world),
            )
            apply_assignments(grid, world, _assignments(plan))
            state["plan"] = plan
            state["routes_by_worker_id"] = {route.worker_id: route for route in plan.worker_routes}
            state["planned_workers"] = [(worker.id, worker.coord) for worker in crew]
        state["grid"] = grid
        settle_tasks(world.farm, _assignments(state["plan"]), state["tasks"])
        state["tasks"] = list(world.farm.tasks)
        state["world"] = world
        farmer, hands = _commands(world, state, hour)
        _remember_positions(world, state, [farmer, *hands])
        return {
            "farmer": farmer,
            "hands": hands,
            "market": _market(observation, state, [farmer, *hands], hour),
        }

    agent.telemetry = state  # type: ignore[attr-defined]
    agent.label = "region_phase1"  # type: ignore[attr-defined]
    return agent


def _reset_day(state: dict[str, Any], day: int) -> None:
    state["day"] = day
    state["grid"] = None
    state["plan"] = None
    state["routes_by_worker_id"] = {}
    state["expected_positions"] = {}
    state["planned_workers"] = []
    state["needs_replan"] = False
    state["market_route"] = None
    state["market_state"] = {"wheat_bought": False}
    state["tasks"] = []
    state["world"] = None


def _hour_zero(observation: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    """Hire and sell. Nobody walks, and nobody is given a guessed birth tile."""

    if state["market_route"] is None:
        state["market_route"] = choose_day_route(observation)
    world = parse_world(observation)
    hand_count = max(0, len(world.farm.workers) - 1)
    farmer = ["PASS"]
    hands = [["PASS"] for _ in range(hand_count)]
    return {
        "farmer": farmer,
        "hands": hands,
        "market": _market(observation, state, [farmer, *hands], 0),
    }


def _crew(world: Any) -> list[RegionWorker]:
    """The first four people already standing on the board, farmer first."""

    return [
        RegionWorker(
            worker_name(worker.actor),
            worker.coord,
            int(worker.carrying.get("WHEAT", 0) or 0),
            tuple(
                sorted(
                    (name, int(amount))
                    for name, amount in worker.carrying.items()
                    if name != "WHEAT" and int(amount or 0) > 0
                )
            ),
        )
        for worker in world.farm.workers[:MAX_REGION_WORKERS]
    ]


def _diverged(crew: list[RegionWorker], state: dict[str, Any]) -> bool:
    expected: dict[str, tuple[int, int]] = state["expected_positions"]
    for worker in crew:
        if expected.get(worker.id) != worker.coord:
            return True
    return False


def _reopen_scheduled(grid: TaskGrid) -> None:
    """A scheduled job that the board has not confirmed can be planned again."""

    for x in range(grid.width):
        for y in range(grid.height):
            cell = grid[x][y]
            if cell is None:
                continue
            for kind in FIELD_OPERATIONS:
                task = cell.tasks.get(kind)
                if task is not None and task.status == SCHEDULED:
                    task.status = PENDING


def _region_origin(grid: TaskGrid) -> tuple[int, int]:
    coords = _must_coords(grid) or _plan_coords(grid)
    if not coords:
        return (0, 0)
    origin_x = min(coord[0] for coord in coords)
    origin_y = min(coord[1] for coord in coords)
    return (
        min(origin_x, max(0, grid.width - REGION_SIZE)),
        min(origin_y, max(0, grid.height - REGION_SIZE)),
    )


def _plan_coords(grid: TaskGrid) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for x in range(grid.width):
        for y in range(grid.height):
            cell = grid[x][y]
            if cell is None or cell.production_plan is None:
                continue
            coords.append(cell.coord)
    return coords


def _must_coords(grid: TaskGrid) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for x in range(grid.width):
        for y in range(grid.height):
            cell = grid[x][y]
            if cell is None:
                continue
            for kind in FIELD_OPERATIONS:
                task = cell.tasks.get(kind)
                if task is None or task.status != PENDING:
                    continue
                if kind != HARVEST and not getattr(task, "mandatory", False):
                    continue
                coords.append(cell.coord)
                break
    return coords


def _assignments(plan: RegionRoutePlan) -> list[TaskAssignment]:
    """Schedule each field job, and remember which hour a harvest is unloaded."""

    assigned: list[TaskAssignment] = []
    for route in plan.worker_routes:
        place_hour = {
            str(action.args[0]): action.hour
            for action in route.actions_by_hour
            if action.operation == "PLACE" and action.args
        }
        visit_at = {visit.coord: visit for visit in route.visits}
        for action in route.actions_by_hour:
            visit = visit_at.get(action.coord) if action.coord is not None else None
            kind, subject = _assignment_kind(action, visit)
            if kind not in FIELD_OPERATIONS or action.coord is None:
                continue
            drop_hour = None
            if kind == HARVEST:
                product = visit.harvest_product if visit is not None else None
                if product:
                    drop_hour = place_hour.get(product)
            assigned.append(
                TaskAssignment(
                    kind=kind,
                    position=action.coord,
                    assigned_worker=route.worker_id,
                    planned_hour=action.hour,
                    planned_drop_hour=drop_hour,
                    planned_sell_hour=drop_hour,
                    subject=subject,
                )
            )
    return assigned


def _assignment_kind(action: Any, visit: Any) -> tuple[str, str]:
    if (
        action.operation == "PLACE"
        and len(action.args) == 1
        and str(action.args[0]) in ANIMAL_NAMES
    ):
        return PLACE_ANIMAL, str(action.args[0])
    if action.operation == PLANT:
        return PLANT, str(action.args[0]) if action.args else ""
    if action.operation == WATER and visit is not None and visit.production_kind == "crop":
        return WATER, str(visit.production_name or "")
    if action.operation in {BUILD_COOP, BUILD_PASTURE}:
        return action.operation, ""
    return action.operation, ""


def _commands(world: Any, state: dict[str, Any], hour: int) -> tuple[list[Any], list[list[Any]]]:
    hand_count = max(0, len(world.farm.workers) - 1)
    farmer = _action_for(worker_name(0), state, hour)
    hands = [_action_for(worker_name(index), state, hour) for index in range(1, hand_count + 1)]
    return farmer, hands


def _action_for(worker_id: str, state: dict[str, Any], hour: int) -> list[Any]:
    route = state["routes_by_worker_id"].get(worker_id)
    if route is None:
        return ["PASS"]
    for action in route.actions_by_hour:
        if action.hour == hour:
            return [action.operation, *action.args]
    return ["PASS"]


def _remember_positions(world: Any, state: dict[str, Any], commands: list[list[Any]]) -> None:
    expected: dict[str, tuple[int, int]] = {}
    for worker, command in zip(world.farm.workers, commands):
        if worker.actor >= MAX_REGION_WORKERS:
            continue
        expected[worker_name(worker.actor)] = _after(worker.coord, str(command[0]))
    state["expected_positions"] = expected


def _mandatory_feed_reserve(
    grid: TaskGrid,
    origin: tuple[int, int],
    region_size: int = REGION_SIZE,
) -> int:
    """Mandatory feeds in the one square this executor will walk.

    Hour 0 does not know which of those animals a hand hired this hour will
    own, so each feed holds one wheat. Wheat already in someone's hands is
    not subtracted.
    """

    x0, y0 = origin
    count = 0
    for x in range(x0, x0 + region_size):
        for y in range(y0, y0 + region_size):
            if not (0 <= x < grid.width and 0 <= y < grid.height):
                continue
            cell = grid[x][y]
            if cell is None:
                continue
            task = cell.tasks.get(FEED)
            if task is None or task.task_type != FEED:
                continue
            if task.status == PENDING and task.mandatory:
                count += 1
    return count


def _reserved_pickups(plan: RegionRoutePlan | None, hour: int) -> int:
    """Wheat this region's plan still takes from the shed after this hour.

    The sale list is not the old day route. A worker may be sent to an animal
    the old route gave to someone who already had wheat. Until that pickup
    happens, the shed grain has to stay.
    """

    if plan is None:
        return 0
    reserved = 0
    for route in plan.worker_routes:
        for action in route.actions_by_hour:
            if action.hour <= hour or action.operation != "PICKUP" or len(action.args) < 2:
                continue
            if action.args[0] != "WHEAT":
                continue
            reserved += int(action.args[1])
    return reserved


def _shed_animals(world: Any) -> dict[str, int]:
    """Animals sitting in the shed right now. A sheep in someone's hands is not included."""

    inventory = getattr(world.farm, "inventory", None)
    if inventory is None:
        return {}
    return {name: int(inventory.shed.get(name, 0) or 0) for name in ANIMAL_NAMES}


def _shed_wheat(world: Any) -> int:
    """Wheat sitting in the shed right now, not a forecast of later stock."""

    inventory = getattr(world.farm, "inventory", None)
    if inventory is None:
        return 0
    return int(inventory.shed.get("WHEAT", 0) or 0)


def _after(coord: tuple[int, int], operation: str) -> tuple[int, int]:
    step = _MOVES.get(operation)
    if step is None:
        return coord
    return (coord[0] + step[0], coord[1] + step[1])


def _market(
    observation: dict[str, Any],
    state: dict[str, Any],
    commands: list[list[Any]],
    hour: int,
) -> list[list[Any]]:
    route = state["market_route"]
    if route is None:
        return []
    if hour == 0:
        world = parse_world(observation)
        grid = build_task_grid(world, None)
        reserved = _mandatory_feed_reserve(grid, _region_origin(grid))
    else:
        reserved = _reserved_pickups(state.get("plan"), hour)
    orders = _market_orders(observation, route, state["market_state"], commands, hour, reserved)
    if hour != 0:
        orders = [order for order in orders if order[0] != "HIRE"]
    return orders
