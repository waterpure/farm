"""Purchases implied by a route that is already written down.

The field decides what gets planted or housed. This module only buys the
seed or animal that route still lacks, and only in a market hour before the
worker needs it. It does not choose a crop because the price looks good.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from kaggle_environments.envs.kaggriculture.kaggriculture import ANIMALS as ENGINE_ANIMALS
from kaggle_environments.envs.kaggriculture.kaggriculture import CROPS as ENGINE_CROPS

from .region_route import RegionRoutePlan
from .route14_phase1 import MAX_MARKET_ORDERS
from .route14_state import ANIMAL_NAMES
from .task_grid import PLANT


@dataclass(frozen=True)
class SupermarketTask:
    """One market order the day plan has already committed to."""

    operation: str
    item: str = ""
    amount: int = 1
    deadline: int = 0
    source_coords: tuple[tuple[int, int], ...] = ()
    cost: int = 0


def derive_supermarket_tasks(
    plan: RegionRoutePlan,
    seeds: Mapping[str, int],
    shed_animals: Mapping[str, int],
    hire_count: int,
    shed_wheat: int = 0,
    forecast_wheat: int = 0,
    wheat_cost: int = 0,
) -> list[SupermarketTask]:
    """Hires, animal buys, and seed buys required by this one route.

    Seeds already in hand cover the earliest plantings. Animals already in
    the shed or in someone's hands are not bought again. A hire is one order
    each, never a single order with a count.
    """

    tasks = [SupermarketTask("HIRE", deadline=0) for _ in range(max(0, hire_count))]
    tasks.extend(_animal_tasks(plan, shed_animals))
    wheat_short, wheat_coords = _wheat_shortage(plan, shed_wheat, forecast_wheat)
    if wheat_short:
        tasks.append(SupermarketTask("BUY_PRODUCT", "WHEAT", wheat_short, 0, wheat_coords, int(wheat_cost)))
    tasks.extend(_seed_tasks(plan, seeds))
    return tasks


def schedule_market_queue(
    tasks: Sequence[SupermarketTask],
    reserved_slots: Mapping[int, int],
) -> tuple[dict[int, list[SupermarketTask]], list[SupermarketTask]]:
    """Place each purchase as late as it can, without passing its deadline.

    Survival/production inputs keep their slots before discretionary hires:
    WHEAT, animals, and seeds are already part of the route's committed
    chain, while a hand is a one-day resource that can be re-planned tomorrow.
    Hires and animal buys are hour 0 only. A seed walks backward from its
    deadline. An input that never finds a free slot is returned unplaced; a
    hire that does not fit is simply deferred to a smaller crew/day.
    """

    queue: dict[int, list[SupermarketTask]] = {hour: [] for hour in range(24)}
    failed: list[SupermarketTask] = []

    def room(hour: int) -> int:
        return MAX_MARKET_ORDERS - int(reserved_slots.get(hour, 0)) - len(queue[hour])

    hires = [task for task in tasks if task.operation == "HIRE"]
    land = [task for task in tasks if task.operation == "BUY_LAND"]
    products = [task for task in tasks if task.operation == "BUY_PRODUCT"]
    animals = [task for task in tasks if task.operation == "BUY_ANIMAL"]
    seeds = sorted(
        (task for task in tasks if task.operation == "BUY_SEED"),
        key=lambda task: (task.deadline, task.item, task.source_coords),
    )
    for task in land:
        if task.deadline == 0 and room(0) > 0:
            queue[0].append(task)
        else:
            failed.append(task)
    for task in products:
        if task.deadline == 0 and room(0) > 0:
            queue[0].append(task)
        else:
            failed.append(task)
    for task in animals:
        if task.deadline == 0 and room(0) > 0:
            queue[0].append(task)
        else:
            failed.append(task)
    for task in seeds:
        placed = False
        if task.deadline >= 0:
            for hour in range(task.deadline, -1, -1):
                if room(hour) > 0:
                    queue[hour].append(task)
                    placed = True
                    break
        if not placed:
            failed.append(task)
    # Hires consume whatever Hour0 slots remain after the route's required
    # purchases.  They are intentionally not added to ``failed``: omitting a
    # hire does not invalidate the field chain, and the caller can try a
    # smaller crew or re-plan on the next day.
    for task in hires:
        if task.deadline == 0 and room(0) > 0:
            queue[0].append(task)
    return queue, failed


def engine_order(task: SupermarketTask) -> list:
    """The list the engine reads. A hire has no item and no amount."""

    if task.operation == "HIRE":
        return ["HIRE"]
    if task.operation == "BUY_LAND":
        return ["BUY_LAND"]
    return [task.operation, task.item, int(task.amount)]


def purchase_cost(tasks: Sequence[SupermarketTask]) -> int:
    """Cash a queued seed or animal order spends. Hires are paid elsewhere."""

    total = 0
    for task in tasks:
        total += _task_cost(task)
    return total


def tasks_over_budget(tasks: Sequence[SupermarketTask], budget: int) -> list[SupermarketTask]:
    """Animal orders are kept before seed orders when the cash runs out."""

    animals = [task for task in tasks if task.operation == "BUY_ANIMAL"]
    seeds = [task for task in tasks if task.operation == "BUY_SEED"]
    kept = 0
    refused: list[SupermarketTask] = []
    products = [task for task in tasks if task.operation == "BUY_PRODUCT"]
    for task in [*products, *animals, *seeds]:
        price = _task_cost(task)
        if kept + price <= budget:
            kept += price
        else:
            refused.append(task)
    return refused


def _unit_price(task: SupermarketTask) -> int:
    if task.operation == "BUY_PRODUCT" and task.item == "WHEAT":
        # The route caller replaces this with the quoted market cost when it
        # applies its budget.  Keep queue construction price-neutral here.
        return 0
    if task.operation == "BUY_SEED":
        spec = ENGINE_CROPS.get(task.item) or {}
        return int(spec.get("seed", 0) or 0)
    if task.operation == "BUY_ANIMAL":
        spec = ENGINE_ANIMALS.get(task.item) or {}
        return int(spec.get("cost", 0) or 0)
    return 0


def _task_cost(task: SupermarketTask) -> int:
    if task.cost:
        return int(task.cost)
    return _unit_price(task) * max(0, int(task.amount))


def _wheat_shortage(
    plan: RegionRoutePlan, shed_wheat: int, forecast_wheat: int
) -> tuple[int, tuple[tuple[int, int], ...]]:
    """Wheat still missing after real shed stock and already-planned buys.

    Worker-held wheat is excluded from the shed total because the route's
    pickup actions already subtract it from the required draw.
    """

    required = 0
    coords: list[tuple[int, int]] = []
    for route in plan.worker_routes:
        for action in route.actions_by_hour:
            if action.operation == "PICKUP" and len(action.args) >= 2 and action.args[0] == "WHEAT":
                required += int(action.args[1])
        coords.extend(
            visit.coord
            for visit in route.visits
            if visit.production_kind == "animal" and "FEED" in visit.tasks
        )
    # RegionRoute's pickup amount is already net of each worker's carried
    # wheat, so only the real shed and the morning forecast cover it.
    return max(0, required - int(shed_wheat) - int(forecast_wheat)), tuple(sorted(set(coords)))


def _animal_tasks(plan: RegionRoutePlan, shed_animals: Mapping[str, int]) -> list[SupermarketTask]:
    picked: dict[str, int] = {}
    coords: dict[str, list[tuple[int, int]]] = {}
    for route in plan.worker_routes:
        for action in route.actions_by_hour:
            if action.operation != "PICKUP" or len(action.args) < 2:
                continue
            name = str(action.args[0])
            if name not in ANIMAL_NAMES:
                continue
            picked[name] = picked.get(name, 0) + int(action.args[1])
        for visit in route.visits:
            if visit.production_kind != "animal" or not visit.production_name:
                continue
            coords.setdefault(visit.production_name, []).append(visit.coord)
    tasks: list[SupermarketTask] = []
    for name in sorted(picked):
        short = picked[name] - int(shed_animals.get(name, 0) or 0)
        if short <= 0:
            continue
        tasks.append(
            SupermarketTask(
                "BUY_ANIMAL",
                name,
                short,
                0,
                tuple(sorted(coords.get(name, []))),
            )
        )
    return tasks


def _seed_tasks(plan: RegionRoutePlan, seeds: Mapping[str, int]) -> list[SupermarketTask]:
    owned = {str(name): int(amount) for name, amount in seeds.items()}
    plantings: list[tuple[int, str, tuple[int, int]]] = []
    for route in plan.worker_routes:
        for action in route.actions_by_hour:
            if action.operation != PLANT or not action.args or action.coord is None:
                continue
            plantings.append((action.hour, str(action.args[0]), action.coord))
    plantings.sort(key=lambda item: (item[0], item[2][1], item[2][0], item[1]))
    shorts: dict[str, list[tuple[int, tuple[int, int]]]] = {}
    for hour, crop, coord in plantings:
        if owned.get(crop, 0) > 0:
            owned[crop] -= 1
            continue
        shorts.setdefault(crop, []).append((hour, coord))
    tasks: list[SupermarketTask] = []
    for crop in sorted(shorts):
        rows = shorts[crop]
        earliest = min(hour for hour, _ in rows)
        tasks.append(
            SupermarketTask(
                "BUY_SEED",
                crop,
                len(rows),
                earliest - 1,
                tuple(coord for _, coord in rows),
            )
        )
    return tasks
