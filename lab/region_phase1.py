"""Run a 5×5 region route one real action per hour.

Hour 0 writes the whole day: the task grid, the worker route, and the market
queue. Later hours play that plan. A hand's birth tile is the official shed
door, predicted before the hire resolves. The route is replanned only when a
real position does not match that prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .market_queue import (
    derive_supermarket_tasks,
    engine_order,
    schedule_market_queue,
    tasks_over_budget,
)
from .region_route import RegionRoutePlan, RegionWorker, plan_region_routes
from .route14_economy import fib_hire_cost
from .route14_phase1 import (
    MAX_MARKET_ORDERS,
    SALE_RANK,
    _buy_wheat_cost,
    _market_orders,
    _shed,
    _spawn_door,
    choose_day_route,
)
from .route14_state import (
    ANIMAL_NAMES,
    COMPLETED,
    PENDING,
    SCHEDULED,
    SHED_CAPACITY,
    TaskAssignment,
    parse_world,
    settle_tasks,
    shed_doors,
    worker_name,
)
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
    """Hour 0 hires, buys, and writes the day. Later hours play that plan."""

    state: dict[str, Any] = {
        "day": None,
        "grid": None,
        "plan": None,
        "routes_by_worker_id": {},
        "expected_positions": {},
        "planned_workers": [],
        "needs_replan": False,
        "force_replan": False,
        "market_route": None,
        "market_state": {"wheat_bought": False},
        "market_queue": {},
        "planned_new_hires": 0,
        "crew_candidates": (),
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
        forced = bool(state.pop("force_replan", False))
        state["needs_replan"] = forced or (state["plan"] is not None and _diverged(crew, state))
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
                shed_total=_shed_total(world),
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
    state["market_queue"] = {}
    state["planned_new_hires"] = 0
    state["crew_candidates"] = ()
    state["force_replan"] = False
    state["tasks"] = []
    state["world"] = None


def _hour_zero(observation: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    """Pass the morning, and write the worker route and the market queue."""

    if state["market_route"] is None:
        # day_route is still a legacy market helper;
        # its target_hires no longer decides RegionPhase1 crew size.
        state["market_route"] = choose_day_route(observation)
    world = parse_world(observation)
    grid = build_task_grid(world, None, observation)
    best, summary = _choose_crew(observation, world, grid, state["market_route"])
    if best.plan is None:
        raise RuntimeError("the morning did not write a route")
    plan = best.plan
    queue, crew = best.market_queue, best.crew
    state["planned_new_hires"] = best.new_hires
    state["crew_candidates"] = summary
    apply_assignments(grid, world, _assignments(plan))
    state["grid"] = grid
    state["plan"] = plan
    state["routes_by_worker_id"] = {route.worker_id: route for route in plan.worker_routes}
    state["planned_workers"] = [(worker.id, worker.coord) for worker in crew]
    state["expected_positions"] = {worker.id: worker.coord for worker in crew}
    state["market_queue"] = queue
    state["world"] = world
    hand_count = max(0, len(world.farm.workers) - 1)
    farmer = ["PASS"]
    hands = [["PASS"] for _ in range(hand_count)]
    return {
        "farmer": farmer,
        "hands": hands,
        "market": _market(observation, state, [farmer, *hands], 0),
    }


@dataclass
class CrewCandidate:
    """One crew size after its route and its market queue are both feasible."""

    new_hires: int
    crew: list[RegionWorker]
    plan: RegionRoutePlan | None
    market_queue: dict[int, list[Any]]
    mandatory_unfinished: int
    production_value: float
    hire_cost: int
    total_move_count: int
    finish_hour: int | None
    feasible: bool = True

    @property
    def net_production_value(self) -> float:
        return self.production_value - self.hire_cost

    @property
    def key(self) -> tuple:
        """Fewer unfinished jobs, then a higher net, then a smaller crew."""

        return (
            self.mandatory_unfinished,
            -self.net_production_value,
            self.new_hires,
            self.total_move_count,
            0 if self.finish_hour is None else self.finish_hour,
        )


def _commit_day_plan(
    observation: dict[str, Any],
    world: Any,
    grid: TaskGrid,
    day_route: Any,
) -> tuple[RegionRoutePlan, dict[int, list[Any]], list[RegionWorker]]:
    """Try every crew that still fits in four people, and keep the best day.

    day_route is still a legacy market helper;
    its target_hires no longer decides RegionPhase1 crew size.
    """

    best, _summary = _choose_crew(observation, world, grid, day_route)
    if best.plan is None:
        raise RuntimeError("the morning did not write a route")
    return best.plan, best.market_queue, best.crew


def _choose_crew(
    observation: dict[str, Any],
    world: Any,
    grid: TaskGrid,
    day_route: Any,
) -> tuple[CrewCandidate, tuple[tuple[int, int, float, int], ...]]:
    """Rank finished days. A crew that cannot actually be hired is left out."""

    candidates = [
        _plan_for_hires(observation, world, grid, day_route, new_hires)
        for new_hires in _feasible_hire_counts(world)
    ]
    feasible = [candidate for candidate in candidates if candidate.feasible and candidate.plan is not None]
    if not feasible:
        raise RuntimeError("the morning did not write a route")
    best = min(feasible, key=lambda candidate: candidate.key)
    summary = tuple(
        (candidate.new_hires, candidate.mandatory_unfinished, candidate.production_value, candidate.hire_cost)
        for candidate in candidates
        if candidate.feasible
    )
    return best, summary


def _feasible_hire_counts(world: Any) -> list[int]:
    """Zero new hands through as many as still fit beside the people already here."""

    present = min(len(_crew(world)), MAX_REGION_WORKERS)
    return list(range(MAX_REGION_WORKERS - present + 1))


def _plan_for_hires(
    observation: dict[str, Any],
    world: Any,
    grid: TaskGrid,
    day_route: Any,
    new_hires: int,
) -> CrewCandidate:
    """Write one crew's route and purchases. This does not pick the winner.

    The blocked set starts empty. Nothing is removed from the task grid, so
    the next crew size can try the same empty tiles again.
    """

    hire_cost = _hire_cost_for_count(_already_hired(observation), new_hires)
    crew = _predicted_crew(world, new_hires, grid.width)
    hour0_slots, leftover_sales = _opening_market(observation, day_route, grid)
    if new_hires > MAX_MARKET_ORDERS - hour0_slots or hire_cost > _hire_cash(world, day_route):
        return _rejected(new_hires, crew, hire_cost)
    origin = _region_origin(grid)
    seeds = _owned_seeds(world)
    real_animals = _shed_animals(world)
    blocked: set[tuple[int, int]] = set()
    feed_reserve, cash_buffer = _operating_holdback(grid)
    budget = _purchase_budget(world, _wheat_outlay(day_route), hire_cost, feed_reserve, cash_buffer)
    shed_room = _animal_room(observation, day_route, grid, world)
    queue: dict[int, list[Any]] = {hour: [] for hour in range(24)}
    plan: RegionRoutePlan | None = None
    for _ in range(REGION_SIZE * REGION_SIZE + 1):
        offered = _virtual_animals(grid, real_animals, blocked, origin)
        plan = plan_region_routes(
            grid,
            crew,
            region_size=REGION_SIZE,
            origin=origin,
            start_hour=1,
            end_hour=23,
                shed_wheat=(
                    _morning_route_wheat(world, day_route)
                    + _conditional_feed_forecast(grid, origin, blocked)
                ),
            shed_coords=shed_doors(grid.width),
            shed_animals=offered,
            shed_total=_shed_total(world),
            blocked_production=tuple(sorted(blocked)),
            include_unpaid_production=True,
        )
        required_pickup_wheat = sum(
            int(action.args[1])
            for route in plan.worker_routes
            for action in route.actions_by_hour
            if action.operation == "PICKUP"
            and len(action.args) >= 2
            and action.args[0] == "WHEAT"
        )
        extra_wheat = max(
            0,
            required_pickup_wheat
            - _shed_wheat(world)
            - int(getattr(day_route, "wheat_buy", 0) or 0),
        )
        total_wheat_cost = _buy_wheat_cost(observation, int(getattr(day_route, "wheat_buy", 0) or 0) + extra_wheat)
        extra_wheat_cost = max(0, total_wheat_cost - _wheat_outlay(day_route))
        tasks = derive_supermarket_tasks(
            plan,
            seeds,
            real_animals,
            new_hires,
            shed_wheat=_shed_wheat(world),
            forecast_wheat=int(getattr(day_route, "wheat_buy", 0) or 0),
            wheat_cost=extra_wheat_cost,
        )
        occupied = _occupied_slots(hour0_slots, leftover_sales, plan)
        queue, failed = schedule_market_queue(tasks, occupied)
        queued = [task for hour_tasks in queue.values() for task in hour_tasks]
        failed = [*failed, *tasks_over_budget(queued, budget)]
        queued_wheat = sum(
            int(task.amount)
            for task in queued
            if task.operation == "BUY_PRODUCT" and task.item == "WHEAT"
        )
        if sum(task.amount for task in queued if task.operation == "BUY_ANIMAL") > shed_room:
            failed = [*failed, *[task for task in queued if task.operation == "BUY_ANIMAL"]]
        elif queued_wheat and queued_wheat > shed_room:
            failed = [
                *failed,
                *[
                    task
                    for task in queued
                    if task.operation == "BUY_PRODUCT" and task.item == "WHEAT"
                ],
            ]
        if not failed:
            break
        victim = _lowest_failed_plan(grid, failed)
        if victim is None or victim in blocked:
            queue = _without_tasks(queue, failed)
            break
        blocked.add(victim)
    else:
        if plan is None:
            return _rejected(new_hires, crew, hire_cost)
    if plan is None or _hire_orders(queue) != new_hires:
        return _rejected(new_hires, crew, hire_cost)
    return CrewCandidate(
        new_hires,
        crew,
        plan,
        queue,
        len(plan.unfinished_visits),
        _scheduled_production_value(grid, plan),
        hire_cost,
        plan.total_move_count,
        plan.finish_hour,
    )


def _rejected(new_hires: int, crew: list[RegionWorker], hire_cost: int) -> CrewCandidate:
    return CrewCandidate(
        new_hires,
        crew,
        None,
        {hour: [] for hour in range(24)},
        0,
        0.0,
        hire_cost,
        0,
        None,
        False,
    )


def _hire_orders(queue: dict[int, list[Any]]) -> int:
    return sum(1 for task in queue.get(0, []) if task.operation == "HIRE")


def _hire_cost_for_count(already_hired: int, new_hires: int) -> int:
    """Official rising hire price, one new hand at a time."""

    return sum(fib_hire_cost(already_hired + offset) for offset in range(max(0, new_hires)))


def _already_hired(observation: dict[str, Any]) -> int:
    player = int(observation.get("player") or 0)
    farms = observation.get("farms") or []
    farm = farms[player] if 0 <= player < len(farms) else {}
    return int((farm or {}).get("hires_today") or 0)


def _hire_cash(world: Any, day_route: Any) -> int:
    """Cash left for new hands after the wheat this morning still has to buy."""

    return max(0, int(world.money) - _wheat_outlay(day_route))


def _wheat_outlay(day_route: Any) -> int:
    if int(getattr(day_route, "start_hour", 0) or 0) != 0:
        return 0
    return int(getattr(day_route, "wheat_cost", 0) or 0)


def _morning_route_wheat(world: Any, day_route: Any) -> int:
    """Forecast Wheat available to the Hour 1 morning route.

    The planned purchase only enlarges the route's pantry forecast. The route
    still derives its exact pickup from mandatory FEED visits and each worker's
    carried Wheat, so a purchase cannot create extra feed work. Later replans
    use the real observation directly and do not call this helper.
    """

    real = _shed_wheat(world)
    if int(getattr(day_route, "start_hour", 0) or 0) != 0:
        return real
    planned = max(0, int(getattr(day_route, "wheat_buy", 0) or 0))
    return real + planned


def _conditional_feed_forecast(
    grid: TaskGrid,
    origin: tuple[int, int],
    blocked: set[tuple[int, int]],
) -> int:
    """Optimistic first-feed units for still-committable animal lines.

    This is planning-only forecast.  The market queue later converts any
    shortage beyond real shed wheat and the existing morning buy into one
    explicit BUY_PRODUCT WHEAT order; execution still checks real inventory.
    """

    x0, y0 = origin
    total = 0
    for x in range(x0, x0 + REGION_SIZE):
        for y in range(y0, y0 + REGION_SIZE):
            if (x, y) in blocked or not (0 <= x < grid.width and 0 <= y < grid.height):
                continue
            cell = grid[x][y]
            plan = getattr(cell, "production_plan", None) if cell is not None else None
            if plan is None or getattr(plan, "kind", None) != "animal":
                continue
            task = cell.tasks.get(FEED) if cell is not None else None
            if task is not None and getattr(task, "status", None) != PENDING:
                continue
            total += 1
    return total


def _scheduled_production_value(grid: TaskGrid, plan: RegionRoutePlan) -> float:
    """Money per day of empty-tile chains the route actually plants or houses.

    A harvest that was already standing in the field is mandatory work. It is
    not counted again here.
    """

    coords: set[tuple[int, int]] = set()
    for route in plan.worker_routes:
        for action in route.actions_by_hour:
            if action.coord is None:
                continue
            if action.operation == PLANT:
                coords.add(action.coord)
            elif (
                action.operation == "PLACE"
                and len(action.args) == 1
                and str(action.args[0]) in ANIMAL_NAMES
            ):
                coords.add(action.coord)
    total = 0.0
    for coord in sorted(coords):
        item = _plan_at(grid, coord)
        if item is None:
            continue
        total += float(getattr(item, "money_per_day", 0) or 0)
    return total


def _predicted_crew(
    world: Any,
    hire_count: int,
    board_size: int,
    max_workers: int = MAX_REGION_WORKERS,
) -> list[RegionWorker]:
    """People already here, plus each new hire on the official birth tile.

    Each hire stands on the least crowded shed door. A tie goes NW, NE, SW, SE.
    """

    crew = _crew(world)
    positions = [worker.coord for worker in world.farm.workers]
    added = 0
    while added < hire_count and len(crew) < max_workers:
        door = _next_spawn(board_size, positions)
        positions.append(door)
        crew.append(RegionWorker(worker_name(len(positions) - 1), door))
        added += 1
    return crew


def _next_spawn(board_size: int, positions: list[tuple[int, int]]) -> tuple[int, int]:
    """The door the engine's `_spawn_hand` would pick for the next hire."""

    return _spawn_door(positions, shed_doors(board_size))


def _owned_seeds(world: Any) -> dict[str, int]:
    inventory = getattr(world.farm, "inventory", None)
    if inventory is None:
        return {}
    return {str(name): int(amount or 0) for name, amount in inventory.seeds.items()}


def _virtual_animals(
    grid: TaskGrid,
    real: dict[str, int],
    blocked: set[tuple[int, int]],
    origin: tuple[int, int],
) -> dict[str, int]:
    """Shed stock plus one animal for each production plan still allowed.

    The extra animal is not real stock. It only lets the route try a pickup
    that hour 0 can buy. A plan that cannot be bought is not offered again.
    """

    stock = {name: int(real.get(name, 0) or 0) for name in ANIMAL_NAMES}
    x0, y0 = origin
    for x in range(x0, x0 + REGION_SIZE):
        for y in range(y0, y0 + REGION_SIZE):
            if not (0 <= x < grid.width and 0 <= y < grid.height):
                continue
            cell = grid[x][y]
            plan = getattr(cell, "production_plan", None) if cell is not None else None
            if plan is None or cell.coord in blocked or getattr(plan, "kind", None) != "animal":
                continue
            name = str(getattr(plan, "name", "") or "")
            if name:
                stock[name] = stock.get(name, 0) + 1
    return stock


def _opening_market(
    observation: dict[str, Any],
    day_route: Any,
    grid: TaskGrid,
) -> tuple[int, set[str]]:
    """Hour 0 slots already used by wheat and sales, and goods those sales leave behind.

    Hires are not counted here. The queue is the only place a hire is written.
    """

    reserved_wheat = _mandatory_feed_reserve(grid, _region_origin(grid))
    orders = _market_orders(
        observation,
        day_route,
        {"wheat_bought": False},
        [["PASS"]],
        0,
        reserved_wheat,
    )
    sold = {
        str(order[1]): int(order[2])
        for order in orders
        if order and order[0] == "SELL" and len(order) >= 3
    }
    leftover: set[str] = set()
    for item, amount in _shed(observation).items():
        if item not in SALE_RANK:
            continue
        if amount - sold.get(item, 0) > 0:
            leftover.add(item)
    return sum(1 for order in orders if order[0] != "HIRE"), leftover


def _occupied_slots(hour0_slots: int, leftover: set[str], plan: RegionRoutePlan) -> dict[int, int]:
    """Slots already taken before a supermarket task looks for a free one."""

    slots = {hour: len(leftover) for hour in range(1, 24)}
    slots[0] = hour0_slots
    for hour, products in _place_products(plan).items():
        slots[hour] = len(leftover | products)
    return slots


def _place_products(plan: RegionRoutePlan) -> dict[int, set[str]]:
    """Products unloaded after hour 0. Each one is one sale that hour."""

    products: dict[int, set[str]] = {}
    for route in plan.worker_routes:
        for action in route.actions_by_hour:
            if action.operation != "PLACE" or len(action.args) < 2:
                continue
            item = str(action.args[0])
            if item not in SALE_RANK or action.hour <= 0:
                continue
            products.setdefault(action.hour, set()).add(item)
    return products


def _operating_holdback(grid: TaskGrid) -> tuple[int, int]:
    """Feed cash and the safety pad the morning portfolio already promised.

    A hand-built grid has no portfolio, so nothing extra is held back. Once a
    portfolio exists, the real hire cost replaces the portfolio's hire guess:
    that guess is not added again.
    """

    portfolio = getattr(grid, "portfolio", None)
    if portfolio is None:
        return 0, 0
    return int(portfolio.reserved_feed_cash), int(portfolio.cash_buffer)


def _purchase_budget(
    world: Any,
    wheat_cost: int,
    hire_cost: int,
    feed_reserve: int = 0,
    cash_buffer: int = 0,
) -> int:
    """Cash left for seeds and animals after hires, wheat, feed, and the pad."""

    return max(
        0,
        int(world.money) - int(wheat_cost) - int(hire_cost) - int(feed_reserve) - int(cash_buffer),
    )


def _animal_room(observation: dict[str, Any], day_route: Any, grid: TaskGrid, world: Any) -> int:
    """Shed space left after hour 0 sells wheat in and sells goods out.

    An animal bought this hour lands in the shed before anyone can pick it up.
    """

    reserved_wheat = _mandatory_feed_reserve(grid, _region_origin(grid))
    orders = _market_orders(
        observation,
        day_route,
        {"wheat_bought": False},
        [["PASS"]],
        0,
        reserved_wheat,
    )
    sold = sum(int(order[2]) for order in orders if order and order[0] == "SELL" and len(order) >= 3)
    wheat_in = int(day_route.wheat_buy or 0) if int(day_route.start_hour) == 0 else 0
    return SHED_CAPACITY - (_shed_total(world) - sold + wheat_in)


def _lowest_failed_plan(grid: TaskGrid, failed: list[Any]) -> tuple[int, int] | None:
    """The production tile to drop: longer walk first, then less money per day."""

    coords = [coord for task in failed for coord in task.source_coords]
    coords = [coord for coord in coords if _plan_at(grid, coord) is not None]
    if not coords:
        return None
    return min(coords, key=lambda coord: _drop_key(grid, coord))


def _plan_at(grid: TaskGrid, coord: tuple[int, int]) -> Any:
    x, y = coord
    if not (0 <= x < grid.width and 0 <= y < grid.height):
        return None
    cell = grid[x][y]
    return getattr(cell, "production_plan", None) if cell is not None else None


def _drop_key(grid: TaskGrid, coord: tuple[int, int]) -> tuple:
    plan = _plan_at(grid, coord)
    actions = tuple(getattr(plan, "actions", ()) or ())
    extra = len(actions) + (1 if getattr(plan, "kind", None) == "animal" else 0)
    money = float(getattr(plan, "money_per_day", 0) or 0)
    return (-extra, money, coord[1], coord[0])


def _without_tasks(queue: dict[int, list[Any]], failed: list[Any]) -> dict[int, list[Any]]:
    refused = set(failed)
    return {hour: [task for task in tasks if task not in refused] for hour, tasks in queue.items()}


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
    """True when the people on the board are not the people the morning named."""

    expected: dict[str, tuple[int, int]] = state["expected_positions"]
    if not expected:
        return False
    actual = {worker.id: worker.coord for worker in crew}
    if set(actual) != set(expected):
        return True
    return any(actual[worker_id] != coord for worker_id, coord in expected.items())


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
    stock = _spendable(world)
    hand_count = max(0, len(world.farm.workers) - 1)
    farmer = _action_for(worker_name(0), state, hour, stock)
    hands = [_action_for(worker_name(index), state, hour, stock) for index in range(1, hand_count + 1)]
    return farmer, hands


def _spendable(world: Any) -> dict[str, Any]:
    """A copy of real stock this hour's commands may spend.

    Morning route planning may forecast a successful Wheat purchase, but
    execution only spends Wheat that is present in the current observation.
    """

    inventory = getattr(world.farm, "inventory", None)
    if inventory is None:
        return {"seeds": {}, "animals": {}, "wheat": 0}
    return {
        "seeds": {str(name): int(amount or 0) for name, amount in inventory.seeds.items()},
        "animals": {name: int(inventory.shed.get(name, 0) or 0) for name in ANIMAL_NAMES},
        "wheat": int(inventory.shed.get("WHEAT", 0) or 0),
    }


def _action_for(
    worker_id: str,
    state: dict[str, Any],
    hour: int,
    stock: dict[str, Any],
) -> list[Any]:
    route = state["routes_by_worker_id"].get(worker_id)
    if route is None:
        return ["PASS"]
    for action in route.actions_by_hour:
        if action.hour != hour:
            continue
        command = [action.operation, *action.args]
        if not _take_stock(command, stock):
            state["force_replan"] = True
            return ["PASS"]
        return command
    return ["PASS"]


def _take_stock(command: list[Any], stock: dict[str, Any]) -> bool:
    """Validate and reserve one command against this hour's real stock."""

    if command[0] == PLANT:
        seed = str(command[1]) if len(command) > 1 else ""
        if stock["seeds"].get(seed, 0) <= 0:
            return False
        stock["seeds"][seed] -= 1
        return True
    if command[0] == "PICKUP" and len(command) >= 3 and str(command[1]) in ANIMAL_NAMES:
        name = str(command[1])
        amount = int(command[2])
        if stock["animals"].get(name, 0) < amount:
            return False
        stock["animals"][name] -= amount
        return True
    if command[0] == "PICKUP" and len(command) >= 3 and str(command[1]) == "WHEAT":
        amount = int(command[2])
        if stock.get("wheat", 0) < amount:
            return False
        stock["wheat"] -= amount
        return True
    return True


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

    Hour 0 writes the route before it sells, so a feed may already be
    scheduled. It still holds one wheat. Wheat already in someone's hands
    is not subtracted. A finished feed does not hold anything.
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
            if task.status != COMPLETED and task.mandatory:
                count += 1
    return count


def _reserved_pickups(plan: RegionRoutePlan | None, hour: int) -> int:
    """Wheat this region's plan still takes from the shed at/after this hour.

    The sale list is not the old day route. A worker may be sent to an animal
    the old route gave to someone who already had wheat. The engine runs unit
    actions before the market, so a pickup scheduled for this hour must also
    stay out of this hour's SELL order.
    """

    if plan is None:
        return 0
    reserved = 0
    for route in plan.worker_routes:
        for action in route.actions_by_hour:
            if action.hour < hour or action.operation != "PICKUP" or len(action.args) < 2:
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


def _shed_total(world: Any) -> int:
    """Every item in the shed, not only wheat and animals. The cap is on the sum."""

    inventory = getattr(world.farm, "inventory", None)
    if inventory is None:
        return 0
    return sum(int(value or 0) for value in inventory.shed.values())


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
        grid = state.get("grid")
        if grid is None:
            world = parse_world(observation)
            grid = build_task_grid(world, None, observation)
        reserved = max(
            _mandatory_feed_reserve(grid, _region_origin(grid)),
            _reserved_pickups(state.get("plan"), 0),
        )
    else:
        reserved = _reserved_pickups(state.get("plan"), hour)
    orders = _market_orders(observation, route, state["market_state"], commands, hour, reserved)
    if hour != 0 or state.get("market_queue"):
        orders = [order for order in orders if order[0] != "HIRE"]
    for task in state.get("market_queue", {}).get(hour, []):
        if len(orders) >= MAX_MARKET_ORDERS:
            break
        orders.append(engine_order(task))
    return orders[:MAX_MARKET_ORDERS]
