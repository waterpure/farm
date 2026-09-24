"""One committed production chain per empty tile.

Empty tiles are filled as one portfolio. Each new crop or animal is scored
again against the cash, feed, labor, and supply already promised, then the
tile keeps a single plan on the same task grid. This module does not send
anyone to plant, build, or pick an animal up.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import ceil
from typing import Any

from .route14_economy import (
    ANIMAL_FIRST_YIELD_DAYS,
    ANIMAL_STRUCTURE,
    CASH_BUFFER,
    CROP_OCCUPY_DAYS,
    CROP_YIELD,
    JOBS_PER_WORKER,
    SHOP_ANIMAL,
    SHOP_CROPS,
    animal_labor_turns,
    animal_line_money_per_day,
    animal_yield_units,
    crop_finishes_in_season,
    crop_labor_turns,
    crop_money_per_day,
    fib_hire_cost,
    line_startup_cost,
    own_supply_map,
    remaining_days,
    wheat_buy_price,
)
from .route14_state import COMPLETED, PENDING, SCHEDULED, WorldState, animal_wheat_per_day
from .task_grid import (
    BUILD_COOP,
    BUILD_PASTURE,
    CARE,
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
    feed_reserve_cash: int = 0
    hire_reserve_cash: int = 0
    required_cash: int = 0
    marginal_value: float = 0.0
    feed_units: int = 0
    daily_jobs: int = 0


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
    feed_reserve_cash: int = 0
    hire_reserve_cash: int = 0
    required_cash: int = 0
    feed_units: int = 0
    daily_jobs: int = 0


@dataclass(frozen=True)
class ProductionPortfolio:
    """What the empty tiles committed, and the cash that must stay unspent."""

    crops: tuple[tuple[str, int], ...]
    animals: tuple[tuple[str, int], ...]
    startup_spend: int
    reserved_feed_cash: int
    reserved_hire_cash: int
    cash_buffer: int
    cash_remaining: int
    planned_supply: tuple[tuple[str, int], ...]
    empty_tiles: int


@dataclass
class _Ledger:
    """Cash still in hand, and the feed, labor, and supply already promised."""

    cash: int
    seeds: dict[str, int]
    animals: dict[str, int]
    planned_supply: dict[str, int] = field(default_factory=dict)
    reserved_feed_cash: int = 0
    reserved_hire_cash: int = 0
    reserved_startup_cash: int = 0
    reserved_feed_units: int = 0
    planned_crop_tiles: dict[str, int] = field(default_factory=dict)
    planned_animal_heads: dict[str, int] = field(default_factory=dict)
    planned_daily_jobs: int = 0


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
    eligible.sort(
        key=lambda item: (
            -item.money_per_day,
            -item.net_value,
            item.required_cash,
            item.startup_steps,
            item.kind,
            item.name,
        )
    )
    return _commit(eligible[0])


def apply_empty_production_plans(
    grid: TaskGrid,
    world: WorldState,
    observation: dict[str, Any],
    previous: TaskGrid | None = None,
) -> ProductionPortfolio:
    """Write at most one production chain onto each empty tile.

    Nearer shed doors are filled first, so the better line takes the shorter
    walk. Each committed line spends its seed, animal, or cash before the next tile.
    A plan already scheduled by a route stays that plan until it is finished.
    """

    books = _ledger(observation)
    _hold_scheduled_resources(previous, books, observation)
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
        _account(books, observation, plan, pay=True)
    portfolio = _portfolio(books, empty_tiles=_unplanned(lands, grid))
    grid.portfolio = portfolio
    return portfolio


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


def feed_runway_days(animal: str) -> int:
    """Wheat days from placement through the first sale.

    Goods appear after the official first-yield delay. Harvesting them and
    getting them to market takes the following day, so one extra ration is held.
    """

    return ANIMAL_FIRST_YIELD_DAYS[animal] + 1


def feed_runway_units(animal: str) -> int:
    return feed_runway_days(animal) * animal_wheat_per_day(animal)


def portfolio_summary(portfolio: ProductionPortfolio) -> str:
    """One line for the morning book: what was planted, and what cash is held back."""

    planted = [f"{name} {count}" for name, count in (*portfolio.animals, *portfolio.crops) if count]
    if portfolio.empty_tiles:
        planted.append(f"EMPTY {portfolio.empty_tiles}")
    mix = " ".join(planted) if planted else "EMPTY"
    return (
        f"Day0 portfolio: {mix} "
        f"startup spend = {portfolio.startup_spend} "
        f"feed reserve = {portfolio.reserved_feed_cash} "
        f"hire reserve = {portfolio.reserved_hire_cash} "
        f"cash buffer = {portfolio.cash_buffer} "
        f"cash remaining = {portfolio.cash_remaining}"
    )


def _crop_candidate(
    observation: dict[str, Any],
    position: tuple[int, int],
    name: str,
    days_left: int,
    prices: dict[str, Any],
    ledger: _Ledger,
) -> ProductionCandidate:
    supply = _effective_supply(observation, ledger)
    per_day = crop_money_per_day(observation, name, days_left, prices, supply)
    occupy = CROP_OCCUPY_DAYS.get(name, days_left)
    feasible = crop_finishes_in_season(name, days_left) and per_day is not None
    owned_seed = ledger.seeds.get(name, 0) > 0
    startup = 0 if owned_seed else line_startup_cost("crop", name)
    money = 0.0 if per_day is None else float(per_day)
    return _scored(
        observation,
        position,
        "crop",
        name,
        name,
        money,
        0.0 if per_day is None else money * occupy,
        startup,
        feasible,
        days_left,
        ledger,
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
    supply = _effective_supply(observation, ledger)
    per_day = animal_line_money_per_day(observation, name, days_left, prices, supply)
    feasible = animal_yield_units(name, days_left) > 0 and per_day is not None
    owned_animal = ledger.animals.get(name, 0) > 0
    startup = 0 if owned_animal else line_startup_cost("animal", name)
    money = 0.0 if per_day is None else float(per_day)
    return _scored(
        observation,
        position,
        "animal",
        name,
        _PRODUCT_OF[name],
        money,
        0.0 if per_day is None else money * max(days_left, 1),
        startup,
        feasible,
        days_left,
        ledger,
        3,
    )


def _scored(
    observation: dict[str, Any],
    position: tuple[int, int],
    kind: str,
    name: str,
    product: str,
    money: float,
    net_value: float,
    startup: int,
    feasible: bool,
    days_left: int,
    ledger: _Ledger,
    startup_steps: int,
) -> ProductionCandidate:
    feed_units, feed_cash, hire_cash, required, jobs = _liquidity(observation, ledger, kind, name, startup, days_left)
    return ProductionCandidate(
        position,
        kind,
        name,
        product,
        money,
        net_value,
        startup,
        feasible,
        _can_afford(ledger, startup, feed_cash, hire_cash, required),
        startup_steps,
        feed_cash,
        hire_cash,
        required,
        money,
        feed_units,
        jobs,
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
        candidate.feed_reserve_cash,
        candidate.hire_reserve_cash,
        candidate.required_cash,
        candidate.feed_units,
        candidate.daily_jobs,
    )


def _tile_actions(kind: str, name: str) -> tuple[PlannedAction, ...]:
    if kind == "crop":
        return (PlannedAction(PLANT, name), PlannedAction(WATER, name))
    structure = BUILD_COOP if ANIMAL_STRUCTURE[name] == "COOP" else BUILD_PASTURE
    # A committed animal line owns its first FEED and CARE as part of the
    # same-day startup chain.  These are strategy conditions, not the
    # engine's survival/must_feed flag: the animal does not exist until
    # PLACE_ANIMAL, but the route can still perform both actions after it is
    # placed and before the day ends.
    return (
        PlannedAction(structure),
        PlannedAction(PLACE_ANIMAL, name),
        PlannedAction("FEED", name),
        PlannedAction(CARE, name),
    )


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


def _account(ledger: _Ledger, observation: dict[str, Any], plan: ProductionPlan, *, pay: bool) -> None:
    """Spend a seed, an animal, or the purchase price, then remember the obligation.

    Feed and hire stay as reserves. They are not bought here. A scheduled plan
    already paid, so it only consumes a resource that is still on the books.
    """

    consumed = False
    if plan.kind == "crop" and ledger.seeds.get(plan.name, 0) > 0:
        ledger.seeds[plan.name] -= 1
        consumed = True
    elif plan.kind == "animal" and ledger.animals.get(plan.name, 0) > 0:
        ledger.animals[plan.name] -= 1
        consumed = True
    if pay and not consumed:
        ledger.cash -= plan.startup_cash
        ledger.reserved_startup_cash += plan.startup_cash
    ledger.reserved_feed_cash += plan.feed_reserve_cash
    ledger.reserved_feed_units += plan.feed_units
    ledger.planned_daily_jobs += plan.daily_jobs
    ledger.reserved_hire_cash = _hire_cash(observation, ledger.planned_daily_jobs)
    if plan.kind == "crop":
        ledger.planned_crop_tiles[plan.name] = ledger.planned_crop_tiles.get(plan.name, 0) + 1
        ledger.planned_supply[plan.product] = ledger.planned_supply.get(plan.product, 0) + CROP_YIELD.get(plan.name, 0)
    else:
        days_left = remaining_days(observation)
        ledger.planned_animal_heads[plan.name] = ledger.planned_animal_heads.get(plan.name, 0) + 1
        ledger.planned_supply[plan.product] = ledger.planned_supply.get(plan.product, 0) + animal_yield_units(
            plan.name, days_left
        )


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


def _hold_scheduled_resources(previous: TaskGrid | None, ledger: _Ledger, observation: dict[str, Any]) -> None:
    """A scheduled sowing or placement already owns its seed, animal, and feed.

    A later watering does not count again: that crop is already standing, and
    the observation's own supply already includes it.
    """

    if previous is None:
        return
    for x in range(previous.width):
        for y in range(previous.height):
            cell = previous[x][y]
            plan = _scheduled_plan(cell)
            if plan is None or not _startup_still_scheduled(cell):
                continue
            _account(ledger, observation, plan, pay=False)


def _startup_still_scheduled(cell: Any) -> bool:
    plant = cell.tasks.get(PLANT)
    place = cell.tasks.get(PLACE_ANIMAL)
    if plant is not None and plant.status == SCHEDULED:
        return True
    return place is not None and place.status == SCHEDULED


def _write_plan(cell: Any, plan: ProductionPlan) -> None:
    cell.production_plan = plan
    for action in plan.actions:
        depends_on = ""
        if action.operation == WATER:
            depends_on = PLANT
        elif action.operation == PLACE_ANIMAL and plan.actions:
            depends_on = plan.actions[0].operation
        elif action.operation == "FEED":
            depends_on = PLACE_ANIMAL
        elif action.operation == CARE:
            depends_on = "FEED"
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


def _liquidity(
    observation: dict[str, Any],
    ledger: _Ledger,
    kind: str,
    name: str,
    startup: int,
    days_left: int,
) -> tuple[int, int, int, int, int]:
    """Feed cash, hire cash, the cash this line needs, and the jobs it adds.

    Profitability already charges feed and labor. These numbers only say whether
    the purse can reach the first sale without spending the safety pad.
    """

    feed_units = feed_runway_units(name) if kind == "animal" else 0
    shortage = max(0, feed_units - _free_wheat(observation, ledger))
    feed_cash = shortage * wheat_buy_price(observation) if kind == "animal" else 0
    jobs = _daily_jobs(kind, name, days_left)
    hire_cash = max(0, _hire_cash(observation, ledger.planned_daily_jobs + jobs) - ledger.reserved_hire_cash)
    return feed_units, feed_cash, hire_cash, startup + feed_cash + hire_cash, jobs


def _can_afford(ledger: _Ledger, startup: int, feed_cash: int, hire_cash: int, required: int) -> bool:
    """A free line with no new obligation may start. Anything else must leave the pad."""

    if startup == 0 and feed_cash == 0 and hire_cash == 0:
        return True
    available = ledger.cash - ledger.reserved_feed_cash - ledger.reserved_hire_cash
    return required <= available - CASH_BUFFER


def _daily_jobs(kind: str, name: str, days_left: int) -> int:
    if kind == "crop":
        turns = crop_labor_turns(name, days_left)
        span = max(1, min(CROP_OCCUPY_DAYS.get(name, days_left), max(days_left, 1)))
    else:
        turns = animal_labor_turns(name, days_left)
        span = max(days_left, 1)
    return ceil(turns / span)


def _hire_cash(observation: dict[str, Any], daily_jobs: int) -> int:
    present, already = _crew(observation)
    needed = ceil(daily_jobs / JOBS_PER_WORKER) if daily_jobs > 0 else 0
    extra = max(0, needed - present)
    return sum(fib_hire_cost(already + index) for index in range(extra))


def _crew(observation: dict[str, Any]) -> tuple[int, int]:
    player = int(observation.get("player") or 0)
    farms = list(observation.get("farms") or [])
    farm = farms[player] if 0 <= player < len(farms) and isinstance(farms[player], dict) else {}
    present = 1 + len(farm.get("hands") or [])
    return present, int(farm.get("hires_today") or 0)


def _effective_supply(observation: dict[str, Any], ledger: _Ledger) -> dict[str, int]:
    supply = dict(own_supply_map(observation))
    for item, units in ledger.planned_supply.items():
        supply[item] = supply.get(item, 0) + int(units)
    return supply


def _free_wheat(observation: dict[str, Any], ledger: _Ledger) -> int:
    return max(0, _owned_wheat(observation) - ledger.reserved_feed_units)


def _owned_wheat(observation: dict[str, Any]) -> int:
    private = observation.get("private") if isinstance(observation.get("private"), dict) else {}
    shed = dict(private.get("shed") or {})
    total = int(shed.get("WHEAT") or 0)
    for inventory in private.get("inventories") or []:
        total += int(dict(inventory or {}).get("WHEAT") or 0)
    return total


def _portfolio(ledger: _Ledger, *, empty_tiles: int) -> ProductionPortfolio:
    crops = tuple(sorted((name, count) for name, count in ledger.planned_crop_tiles.items() if count))
    animals = tuple(sorted((name, count) for name, count in ledger.planned_animal_heads.items() if count))
    supply = tuple(sorted((name, count) for name, count in ledger.planned_supply.items() if count))
    return ProductionPortfolio(
        crops,
        animals,
        ledger.reserved_startup_cash,
        ledger.reserved_feed_cash,
        ledger.reserved_hire_cash,
        CASH_BUFFER,
        ledger.cash,
        supply,
        empty_tiles,
    )


def _unplanned(lands: list[Any], grid: TaskGrid) -> int:
    empty = 0
    for land in lands:
        cell = _cell(grid, land.position)
        if cell is None or cell.production_plan is None:
            empty += 1
    return empty
