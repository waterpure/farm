"""One committed production chain per empty tile.

Candidates compare every crop and animal. The tile then keeps a single plan,
written onto the same task grid as the crops and animals already standing there.
This module does not send anyone to plant, build, or pick an animal up.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .route14_economy import (
    ANIMAL_STRUCTURE,
    CROP_OCCUPY_DAYS,
    SHOP_ANIMAL,
    SHOP_CROPS,
    animal_line_money_per_day,
    animal_yield_units,
    crop_finishes_in_season,
    crop_money_per_day,
    line_startup_cost,
    remaining_days,
)
from .route14_state import COMPLETED, PENDING, SCHEDULED, WorldState
from .task_grid import (
    BUILD_COOP,
    BUILD_PASTURE,
    PLACE_ANIMAL,
    PLANT,
    WATER,
    TaskGrid,
    TileTask,
)


_ANIMAL_NAMES = tuple(ANIMAL_STRUCTURE)
_PRODUCT_OF = {animal: product for product, animal in SHOP_ANIMAL.items()}
_PRODUCTION_KINDS = {PLANT, BUILD_COOP, BUILD_PASTURE, PLACE_ANIMAL, WATER}


@dataclass(frozen=True)
class PlannedAction:
    """One step of a production chain. `subject` is the seed or the animal."""

    operation: str
    subject: str = ""


@dataclass(frozen=True)
class ProductionCandidate:
    """One line that could occupy this empty tile. It is not a task yet."""

    position: tuple[int, int]
    kind: str
    name: str
    product: str
    money_per_day: float
    net_value: float
    startup_cash: int
    feasible_before_end: bool
    affordable: bool
    startup_steps: int


@dataclass(frozen=True)
class ProductionPlan:
    """The one chain this tile will follow. Its actions stay in that order."""

    position: tuple[int, int]
    kind: str
    name: str
    product: str
    actions: tuple[PlannedAction, ...]
    money_per_day: float
    net_value: float
    startup_cash: int
    feasible_before_end: bool
    affordable: bool
    startup_steps: int
    atomic_start: bool = True


@dataclass
class _Ledger:
    """Seeds, unplaced animals, and cash still available for a new tile."""

    cash: int
    seeds: dict[str, int]
    animals: dict[str, int]


def production_candidates(
    observation: dict[str, Any],
    position: tuple[int, int],
    ledger: _Ledger | None = None,
) -> tuple[ProductionCandidate, ...]:
    """Every crop and animal this tile could start, priced with the resources still free."""

    books = ledger if ledger is not None else _ledger(observation)
    days_left = remaining_days(observation)
    prices = dict((observation.get("market") or {}).get("prices") or {})
    found: list[ProductionCandidate] = []
    for name in SHOP_CROPS:
        found.append(_crop_candidate(observation, position, name, days_left, prices, books))
    for name in _ANIMAL_NAMES:
        found.append(_animal_candidate(observation, position, name, days_left, prices, books))
    return tuple(found)


def choose_production_plan(
    observation: dict[str, Any],
    position: tuple[int, int],
    ledger: _Ledger | None = None,
) -> ProductionPlan | None:
    """The single line this tile may commit. None when nothing is ready to pay for."""

    books = ledger if ledger is not None else _ledger(observation)
    eligible = [
        candidate
        for candidate in production_candidates(observation, position, books)
        if candidate.feasible_before_end and candidate.net_value > 0 and candidate.affordable
    ]
    if not eligible:
        return None
    eligible.sort(key=lambda item: (-item.money_per_day, -item.net_value, item.startup_steps, item.kind, item.name))
    return _commit(eligible[0])


def apply_empty_production_plans(
    grid: TaskGrid,
    world: WorldState,
    observation: dict[str, Any],
    previous: TaskGrid | None = None,
) -> None:
    """Write at most one production chain onto each empty tile.

    Nearer shed doors are filled first, so the better line takes the shorter
    walk. Each committed line spends its seed, animal, or cash before the next tile.
    A plan already scheduled by a route stays that plan until it is finished.
    """

    books = _ledger(observation)
    _hold_scheduled_resources(previous, books)
    lands = [
        land
        for land in world.farm.lands
        if land.empty and not land.weed and land.plantable
    ]
    lands.sort(key=lambda land: (land.door_distance, land.position[1], land.position[0]))
    frozen: set[tuple[int, int]] = set()
    for land in lands:
        previous_cell = _cell(previous, land.position) if previous is not None else None
        if _scheduled_plan(previous_cell) is None:
            continue
        cell = _cell(grid, land.position)
        if cell is None:
            continue
        _restore_frozen(cell, previous_cell)
        frozen.add(land.position)
    for land in lands:
        if land.position in frozen:
            continue
        cell = _cell(grid, land.position)
        if cell is None:
            continue
        _clear_pending_production(cell)
        plan = choose_production_plan(observation, land.position, books)
        if plan is None:
            cell.production_plan = None
            continue
        _write_plan(cell, plan)
        _reserve(books, plan)


def startup_chain(plan: ProductionPlan) -> tuple[PlannedAction, ...]:
    """The actions that must fit in one day before this plan is allowed to begin.

    A crop is sowing plus the watering that keeps the seedling alive. An animal
    is picking it up, building the shed, and placing it. The pickup is not a
    task on the tile; it is still part of the start.
    """

    if plan.kind == "animal":
        return (PlannedAction("PICKUP", plan.name), *plan.actions)
    return plan.actions


def can_start_production(plan: ProductionPlan, turns_available: int) -> bool:
    """False when the day cannot finish the whole start. A lone sowing is not a start."""

    if turns_available < 0:
        return False
    if plan.atomic_start:
        return turns_available >= len(startup_chain(plan))
    return turns_available >= 1


def seedling_water_is_mandatory(plan: ProductionPlan, plant_status: str) -> bool:
    """Once the seed is in the ground, that same day's watering can no longer wait."""

    if plan.kind != "crop" or plant_status != COMPLETED:
        return False
    return any(action.operation == WATER for action in plan.actions)


def _crop_candidate(
    observation: dict[str, Any],
    position: tuple[int, int],
    name: str,
    days_left: int,
    prices: dict[str, Any],
    ledger: _Ledger,
) -> ProductionCandidate:
    per_day = crop_money_per_day(observation, name, days_left, prices)
    occupy = CROP_OCCUPY_DAYS.get(name, days_left)
    feasible = crop_finishes_in_season(name, days_left) and per_day is not None
    owned_seed = ledger.seeds.get(name, 0) > 0
    startup = 0 if owned_seed else line_startup_cost("crop", name)
    return ProductionCandidate(
        position,
        "crop",
        name,
        name,
        0.0 if per_day is None else float(per_day),
        0.0 if per_day is None else float(per_day) * occupy,
        startup,
        feasible,
        startup <= ledger.cash,
        2,
    )


def _animal_candidate(
    observation: dict[str, Any],
    position: tuple[int, int],
    name: str,
    days_left: int,
    prices: dict[str, Any],
    ledger: _Ledger,
) -> ProductionCandidate:
    per_day = animal_line_money_per_day(observation, name, days_left, prices)
    feasible = animal_yield_units(name, days_left) > 0 and per_day is not None
    owned_animal = ledger.animals.get(name, 0) > 0
    startup = 0 if owned_animal else line_startup_cost("animal", name)
    return ProductionCandidate(
        position,
        "animal",
        name,
        _PRODUCT_OF[name],
        0.0 if per_day is None else float(per_day),
        0.0 if per_day is None else float(per_day) * max(days_left, 1),
        startup,
        feasible,
        startup <= ledger.cash,
        3,
    )


def _commit(candidate: ProductionCandidate) -> ProductionPlan:
    return ProductionPlan(
        candidate.position,
        candidate.kind,
        candidate.name,
        candidate.product,
        _tile_actions(candidate.kind, candidate.name),
        candidate.money_per_day,
        candidate.net_value,
        candidate.startup_cash,
        candidate.feasible_before_end,
        candidate.affordable,
        candidate.startup_steps,
        True,
    )


def _tile_actions(kind: str, name: str) -> tuple[PlannedAction, ...]:
    if kind == "crop":
        return (PlannedAction(PLANT, name), PlannedAction(WATER, name))
    structure = BUILD_COOP if ANIMAL_STRUCTURE[name] == "COOP" else BUILD_PASTURE
    return (PlannedAction(structure), PlannedAction(PLACE_ANIMAL, name))


def _ledger(observation: dict[str, Any]) -> _Ledger:
    private = observation.get("private") if isinstance(observation.get("private"), dict) else {}
    seeds = {
        str(name): int(amount)
        for name, amount in dict(private.get("seeds") or {}).items()
        if int(amount or 0) > 0
    }
    animals = {name: 0 for name in _ANIMAL_NAMES}
    for name, amount in dict(private.get("shed") or {}).items():
        if name in animals:
            animals[name] += int(amount or 0)
    for inventory in private.get("inventories") or []:
        for name, amount in dict(inventory or {}).items():
            if name in animals:
                animals[name] += int(amount or 0)
    player = int(observation.get("player") or 0)
    farms = list(observation.get("farms") or [])
    farm = farms[player] if 0 <= player < len(farms) and isinstance(farms[player], dict) else {}
    return _Ledger(int(farm.get("money") or 0), seeds, animals)


def _reserve(ledger: _Ledger, plan: ProductionPlan) -> None:
    if plan.kind == "crop" and ledger.seeds.get(plan.name, 0) > 0:
        ledger.seeds[plan.name] -= 1
        return
    if plan.kind == "animal" and ledger.animals.get(plan.name, 0) > 0:
        ledger.animals[plan.name] -= 1
        return
    ledger.cash -= plan.startup_cash


def _clear_pending_production(cell: Any) -> None:
    for kind in list(cell.tasks):
        if kind not in _PRODUCTION_KINDS:
            continue
        task = cell.tasks[kind]
        if task.status in {COMPLETED, SCHEDULED}:
            continue
        if kind == WATER and not isinstance(task, TileTask):
            continue
        del cell.tasks[kind]
    cell.production_plan = None


def _scheduled_plan(cell: Any) -> ProductionPlan | None:
    """The plan whose chain a route has already accepted. Pending plans may be replaced."""

    plan = getattr(cell, "production_plan", None) if cell is not None else None
    if plan is None:
        return None
    for action in plan.actions:
        task = cell.tasks.get(action.operation)
        if task is not None and task.status == SCHEDULED:
            return plan
    return None


def _restore_frozen(cell: Any, previous_cell: Any) -> None:
    plan = previous_cell.production_plan
    cell.production_plan = plan
    for action in plan.actions:
        prior = previous_cell.tasks.get(action.operation)
        if prior is not None:
            cell.tasks[action.operation] = replace(prior)


def _hold_scheduled_resources(previous: TaskGrid | None, ledger: _Ledger) -> None:
    """A scheduled sowing or placement already owns its seed or animal."""

    if previous is None:
        return
    for x in range(previous.width):
        for y in range(previous.height):
            cell = previous[x][y]
            if cell is None:
                continue
            plant = cell.tasks.get(PLANT)
            if plant is not None and plant.status == SCHEDULED and plant.subject:
                if ledger.seeds.get(plant.subject, 0) > 0:
                    ledger.seeds[plant.subject] -= 1
            place = cell.tasks.get(PLACE_ANIMAL)
            if place is not None and place.status == SCHEDULED and place.subject:
                if ledger.animals.get(place.subject, 0) > 0:
                    ledger.animals[place.subject] -= 1


def _write_plan(cell: Any, plan: ProductionPlan) -> None:
    cell.production_plan = plan
    for action in plan.actions:
        depends_on = ""
        if action.operation == WATER:
            depends_on = PLANT
        elif action.operation == PLACE_ANIMAL and plan.actions:
            depends_on = plan.actions[0].operation
        cell.tasks[action.operation] = TileTask(
            task_type=action.operation,
            status=PENDING,
            mandatory=False,
            subject=action.subject,
            depends_on=depends_on,
        )


def _cell(grid: TaskGrid, position: tuple[int, int]) -> Any:
    x, y = position
    if not (0 <= x < grid.width and 0 <= y < grid.height):
        return None
    return grid[x][y]
