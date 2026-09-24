"""Day routes for one 5×5 region and one to four workers.

A tile's must-do jobs stay together. One worker walks to that tile once and
does every job there before leaving. Jobs are not handed to different people,
and they are not searched as a task-level bit mask. Hour 0 hiring is outside
this module: each worker already has a real coordinate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

from kaggle_environments.envs.kaggriculture.kaggriculture import ANIMALS as ENGINE_ANIMALS
from kaggle_environments.envs.kaggriculture.kaggriculture import CROPS as ENGINE_CROPS

from .route14_state import ANIMAL_NAMES, PENDING, SHED_CAPACITY, shed_doors
from .task_grid import CARE, COLLECT_FERTILIZER, FEED, FERTILIZE, HARVEST, PLACE_ANIMAL, PLANT, WATER, TaskGrid


EXACT_TILES = 10
MUST_KINDS = (WATER, FEED, CARE, COLLECT_FERTILIZER, FERTILIZE, HARVEST)
_KIND_RANK = {FERTILIZE: 0, WATER: 1, FEED: 2, CARE: 3, COLLECT_FERTILIZER: 4, HARVEST: 5}
_MAX_ROUNDS = 24


@dataclass(frozen=True)
class RegionWorker:
    """A person already standing on the board. This planner does not hire."""

    id: str
    coord: tuple[int, int]
    carrying_wheat: int = 0
    carrying_items: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class TileVisit:
    """Every must-do job on one tile, done by one worker in one stop."""

    coord: tuple[int, int]
    tasks: tuple[str, ...]
    action_count: int
    tile_type: str
    harvest_product: str | None = None
    harvest_units: int = 0
    task_args: tuple[tuple, ...] = ()
    production_kind: str | None = None
    production_name: str | None = None


@dataclass(frozen=True)
class RouteAction:
    """One real engine action. A move is one step, never a jump to a tile."""

    hour: int
    operation: str
    coord: tuple[int, int] | None = None
    args: tuple = ()

    def __str__(self) -> str:
        if self.coord is None:
            return f"Hour {self.hour}: {self.operation}"
        x, y = self.coord
        return f"Hour {self.hour}: {self.operation} ({x},{y})"


@dataclass(frozen=True)
class WorkerRoutePlan:
    """One worker's ordered tile visits and the hour-by-hour actions."""

    worker_id: str
    start_coord: tuple[int, int]
    visits: tuple[TileVisit, ...]
    actions_by_hour: tuple[RouteAction, ...]
    move_count: int
    finish_hour: int | None


@dataclass(frozen=True)
class RegionRoutePlan:
    """Whether this fixed crew can finish every must-do visit in the region."""

    feasible: bool
    worker_routes: tuple[WorkerRoutePlan, ...]
    total_move_count: int
    finish_hour: int | None
    unfinished_visits: tuple[TileVisit, ...]


def plan_region_routes(
    task_grid: TaskGrid,
    workers: Sequence[RegionWorker],
    region_size: int = 5,
    start_hour: int = 1,
    end_hour: int = 23,
    origin: tuple[int, int] = (0, 0),
    shed_wheat: int = 0,
    shed_coords: Sequence[tuple[int, int]] | None = None,
    include_optional: bool = True,
    shed_animals: dict[str, int] | None = None,
    shed_fertilizer: int = 0,
    shed_total: int = 0,
    blocked_production: Sequence[tuple[int, int]] | None = None,
    include_unpaid_production: bool = False,
) -> RegionRoutePlan:
    """Assign every must-do tile in one square to the given workers.

    The square starts at `origin` and runs `region_size` cells on both axes.
    The default square is the corner `[0, 5)`. Each tile has one owner.
    Among plans that finish every visit by `end_hour`, fewer steps win, then
    an earlier last hour. A feed is only assigned when that worker can hold
    enough wheat, picking up the shortage once from `shed_wheat` before the
    field walk.
    """

    crew = tuple(workers)
    if not 1 <= len(crew) <= 4:
        raise ValueError("region routes take 1 to 4 workers")
    if region_size < 1:
        raise ValueError("region_size must be positive")
    if start_hour > end_hour:
        raise ValueError("start_hour must be at or before end_hour")
    if shed_wheat < 0:
        raise ValueError("shed_wheat cannot be negative")
    if shed_total < 0:
        raise ValueError("shed_total cannot be negative")
    if shed_fertilizer < 0:
        raise ValueError("shed_fertilizer cannot be negative")
    animal_stock = _animal_stock(shed_animals)
    doors = tuple(shed_coords) if shed_coords is not None else shed_doors(task_grid.width)
    pantry = _Pantry(shed_wheat, doors, animal_stock, shed_total, shed_fertilizer)
    visits = _extract_visits(task_grid, region_size, origin)
    if visits:
        routes: list[list[TileVisit]] = [[] for _ in crew]
        leftover: list[TileVisit] = []
        for visit in sorted(visits, key=_visit_sort):
            placed = _best_insertion(crew, routes, visit, start_hour, end_hour, pantry)
            if placed is None:
                leftover.append(visit)
                continue
            worker_index, ordered = placed
            routes[worker_index] = ordered
        scored = _search(crew, routes, leftover, start_hour, end_hour, pantry)
        if include_optional:
            scored = _add_same_tile_optional_tasks(
                task_grid, crew, scored, start_hour, end_hour, pantry
            )
    else:
        scored = _Score(tuple(tuple() for _ in crew), (), 0, None)
    scored = _add_production_plans(
        task_grid,
        crew,
        scored,
        start_hour,
        end_hour,
        pantry,
        region_size,
        origin,
        blocked_production or (),
        include_unpaid_production,
    )
    return _materialize(crew, scored, start_hour, end_hour, pantry)


def _visit_sort(visit: TileVisit) -> tuple[int, int, tuple[str, ...]]:
    return (visit.coord[1], visit.coord[0], visit.tasks)


def _extract_visits(
    grid: TaskGrid,
    region_size: int,
    origin: tuple[int, int],
) -> tuple[TileVisit, ...]:
    visits: list[TileVisit] = []
    x0, y0 = origin
    for x in range(x0, x0 + region_size):
        for y in range(y0, y0 + region_size):
            if not (0 <= x < grid.width and 0 <= y < grid.height):
                continue
            cell = grid[x][y]
            if cell is None:
                continue
            kinds = [
                kind
                for kind in MUST_KINDS
                if _is_must(cell.tasks.get(kind), kind)
            ]
            if not kinds:
                continue
            tasks = _task_order(cell.tile_type, kinds)
            product, units = _harvest_cargo(cell.tile_type, cell.tasks.get(HARVEST), tasks)
            visits.append(TileVisit(cell.coord, tasks, len(tasks), cell.tile_type, product, units))
    return tuple(visits)


def _is_must(task: object, kind: str) -> bool:
    if task is None or getattr(task, "status", None) != PENDING:
        return False
    if kind in {WATER, FEED, CARE, COLLECT_FERTILIZER, HARVEST} and not getattr(task, "mandatory", False):
        return False
    return True


def _task_order(tile_type: str, kinds: list[str]) -> tuple[str, ...]:
    """Legal order inside one visit. This does not decide whether water is worth doing.

    A one-shot crop disappears when it is harvested, so water has to happen
    first. Ongoing crops stay in the ground; water still comes first.
    """

    ordered = tuple(sorted(kinds, key=lambda kind: _KIND_RANK[kind]))
    spec = ENGINE_CROPS.get(tile_type)
    one_shot = spec is not None and not spec["ongoing"]
    if one_shot and WATER in ordered and HARVEST in ordered:
        return tuple(kind for kind in ordered if kind != HARVEST) + (HARVEST,)
    return ordered


@dataclass(frozen=True)
class _Pantry:
    """Wheat and unplaced animals still in the shed, and the tiles where a worker can take them."""

    wheat: int
    doors: tuple[tuple[int, int], ...]
    animals: tuple[tuple[str, int], ...] = ()
    shed_total: int = 0
    fertilizer: int = 0


@dataclass(frozen=True)
class _Leg:
    """Field walk, plus one shed return when that walk still finishes today."""

    door: tuple[int, int] | None
    pickup: int
    visits: tuple[TileVisit, ...]
    travel: int
    return_door: tuple[int, int] | None = None
    return_travel: int = 0
    deliveries: tuple[tuple[str, int], ...] = ()
    item_pickups: tuple[tuple[str, int], ...] = ()

    @property
    def action_total(self) -> int:
        work = sum(visit.action_count for visit in self.visits)
        return (
            self.travel
            + self.return_travel
            + (1 if self.pickup else 0)
            + len(self.item_pickups)
            + work
            + len(self.deliveries)
        )


def _harvest_cargo(tile_type: str, harvest: object, tasks: tuple[str, ...]) -> tuple[str | None, int]:
    """Goods this visit puts in the worker's hands. Animals yield wool, milk, or eggs."""

    if HARVEST not in tasks or harvest is None:
        return None, 0
    units = int(getattr(harvest, "yield_amount", 0) or 0)
    if units <= 0:
        return None, 0
    animal = ENGINE_ANIMALS.get(tile_type)
    product = str(animal["product"]) if animal is not None else tile_type
    return product, units


def _deliveries(visits: Sequence[TileVisit]) -> tuple[tuple[str, int], ...]:
    """One unload per product, in the order the route first picked that product up."""

    totals: dict[str, int] = {}
    order: list[str] = []
    for visit in visits:
        product = visit.harvest_product
        if not product or visit.harvest_units <= 0:
            continue
        if product not in totals:
            order.append(product)
            totals[product] = 0
        totals[product] += visit.harvest_units
    return tuple((product, totals[product]) for product in order)


def _return_choice(
    last: tuple[int, int],
    doors: Sequence[tuple[int, int]],
) -> tuple[tuple[int, int] | None, int]:
    """Cheapest door from the last field tile. The visit order is already fixed."""

    if not doors:
        return None, 10**9
    door = min(doors, key=lambda item: (_manhattan(last, item), item))
    return door, _manhattan(last, door)


def _with_return(
    pickup_door: tuple[int, int] | None,
    pickup: int,
    ordered: list[TileVisit],
    field_travel: int,
    doors: Sequence[tuple[int, int]],
    item_pickups: tuple[tuple[str, int], ...] = (),
) -> _Leg:
    deliveries = _deliveries(ordered)
    if not deliveries or not ordered:
        return _Leg(pickup_door, pickup, tuple(ordered), field_travel, item_pickups=item_pickups)
    door, steps = _return_choice(ordered[-1].coord, doors)
    return _Leg(
        pickup_door,
        pickup,
        tuple(ordered),
        field_travel,
        door,
        steps,
        deliveries,
        item_pickups,
    )


def _without_return(leg: _Leg) -> _Leg:
    """Same field walk, with no trip back and no unload.

    The harvest is still on the visit. It stays in the worker's hands until
    the day ends and the shed takes whatever still fits.
    """

    return _Leg(
        door=leg.door,
        pickup=leg.pickup,
        visits=leg.visits,
        travel=leg.travel,
        item_pickups=leg.item_pickups,
    )


def _placed_animals(visit: TileVisit) -> tuple[str, ...]:
    subjects: list[str] = []
    for index, kind in enumerate(visit.tasks):
        if kind != PLACE_ANIMAL:
            continue
        args = visit.task_args[index] if index < len(visit.task_args) else ()
        subject = str(args[0]) if args else (visit.production_name or "")
        if subject:
            subjects.append(subject)
    return tuple(subjects)


def _end_inventory(worker: RegionWorker, leg: _Leg) -> dict[str, int]:
    """What this person is still holding after the route, before the day-end drop.

    This is a forecast. It does not change the worker or the shed.
    """

    held: dict[str, int] = {}

    def add(name: str, amount: int) -> None:
        if not name or amount == 0:
            return
        held[name] = held.get(name, 0) + amount

    add("WHEAT", worker.carrying_wheat)
    for name, amount in worker.carrying_items:
        add(name, amount)
    add("WHEAT", leg.pickup)
    for name, amount in leg.item_pickups:
        add(name, amount)
    for visit in leg.visits:
        add("WHEAT", -visit.tasks.count(FEED))
        add("FERTILIZER", visit.tasks.count(COLLECT_FERTILIZER))
        add("FERTILIZER", -visit.tasks.count(FERTILIZE))
        if visit.harvest_product and visit.harvest_units > 0 and HARVEST in visit.tasks:
            add(visit.harvest_product, visit.harvest_units)
        for subject in _placed_animals(visit):
            add(subject, -1)
    for product, units in leg.deliveries:
        add(product, -units)
    return {name: amount for name, amount in held.items() if amount > 0}


def _feed_count(visits: Sequence[TileVisit]) -> int:
    return sum(1 for visit in visits if FEED in visit.tasks)


def _fertilize_count(visits: Sequence[TileVisit]) -> int:
    return sum(1 for visit in visits if FERTILIZE in visit.tasks)


def _carried(worker: RegionWorker, item: str) -> int:
    """How many of `item` this one person is already holding. Another person's hands do not count."""

    if item == "WHEAT":
        return max(0, worker.carrying_wheat)
    return sum(amount for name, amount in worker.carrying_items if name == item and amount > 0)


def _animal_stock(shed_animals: dict[str, int] | None) -> tuple[tuple[str, int], ...]:
    stock: list[tuple[str, int]] = []
    for name, count in sorted((shed_animals or {}).items()):
        amount = int(count)
        if amount < 0:
            raise ValueError("shed animal count cannot be negative")
        if amount > 0:
            stock.append((str(name), amount))
    return tuple(stock)


def _shed_animal_count(pantry: _Pantry, name: str) -> int:
    return sum(amount for item, amount in pantry.animals if item == name)


def _animal_needs(visits: Sequence[TileVisit]) -> dict[str, int]:
    needs: dict[str, int] = {}
    for visit in visits:
        if visit.production_kind != "animal" or not visit.production_name:
            continue
        needs[visit.production_name] = needs.get(visit.production_name, 0) + 1
    return needs


def _item_pickups(worker: RegionWorker, visits: Sequence[TileVisit]) -> tuple[tuple[str, int], ...]:
    """One pickup per animal kind. The count is how many this worker still has to take."""

    needs = _animal_needs(visits)
    pickups: list[tuple[str, int]] = []
    for name in sorted(needs):
        short = needs[name] - _carried(worker, name)
        if short > 0:
            pickups.append((name, short))
    return tuple(pickups)


def _within_animals(
    workers: Sequence[RegionWorker],
    routes: Sequence[Sequence[TileVisit]],
    pantry: _Pantry,
) -> bool:
    """Shed animals are shared. Animals already in a worker's hands are only theirs."""

    drawn = {name: 0 for name in ANIMAL_NAMES}
    for worker, route in zip(workers, routes):
        for name, short in _item_pickups(worker, route):
            drawn[name] = drawn.get(name, 0) + short
    return all(drawn.get(name, 0) <= _shed_animal_count(pantry, name) for name in drawn)


def _pickup_needed(worker: RegionWorker, visits: Sequence[TileVisit]) -> int:
    """Wheat this worker still has to take for the feeds they already own."""

    return max(0, _feed_count(visits) - _carried(worker, "WHEAT"))


def _fertilizer_pickup_needed(worker: RegionWorker, visits: Sequence[TileVisit]) -> int:
    """Fertilizer this worker must take for the crop actions on this route."""

    # A manure pickup on an earlier animal visit can feed a later crop visit
    # in the same route. Only the prefix deficit must come from the shed.
    available = _carried(worker, "FERTILIZER")
    pickup = 0
    for visit in visits:
        need = visit.tasks.count(FERTILIZE)
        if need > available:
            pickup += need - available
            available = need
        available -= need
        available += visit.tasks.count(COLLECT_FERTILIZER)
    return pickup


def _wheat_draw(workers: Sequence[RegionWorker], routes: Sequence[Sequence[TileVisit]]) -> int:
    return sum(_pickup_needed(worker, route) for worker, route in zip(workers, routes))


def _fertilizer_draw(workers: Sequence[RegionWorker], routes: Sequence[Sequence[TileVisit]]) -> int:
    return sum(
        _fertilizer_pickup_needed(worker, route)
        for worker, route in zip(workers, routes)
    )


def _within_wheat(
    workers: Sequence[RegionWorker],
    routes: Sequence[Sequence[TileVisit]],
    pantry: _Pantry,
) -> bool:
    return _wheat_draw(workers, routes) <= pantry.wheat


def _within_fertilizer(
    workers: Sequence[RegionWorker],
    routes: Sequence[Sequence[TileVisit]],
    pantry: _Pantry,
) -> bool:
    return _fertilizer_draw(workers, routes) <= pantry.fertilizer


def _layout(worker: RegionWorker, visits: Sequence[TileVisit], doors: Sequence[tuple[int, int]]) -> _Leg:
    """Cheapest full walk: start, one shed stop if wheat or an animal is short, then the tiles.

    The door is chosen by that whole walk, including every pickup and the trip
    back with a harvest. A nearer door can lose when the field sits closer to another door.
    """

    items = list(visits)
    ordered = _reorder(worker.coord, items)
    pickup = _pickup_needed(worker, ordered)
    animals = _item_pickups(worker, ordered)
    fertilizer = _fertilizer_pickup_needed(worker, ordered)
    item_pickups = animals + (("FERTILIZER", fertilizer),) if fertilizer > 0 else animals
    if not items or (pickup <= 0 and not item_pickups):
        travel = _path_moves(worker.coord, [visit.coord for visit in ordered])
        return _with_return(None, 0, ordered, travel, doors)
    best: tuple[tuple[int, int, tuple[int, int]], _Leg] | None = None
    for door in doors:
        ordered = _reorder(door, items)
        pickup = _pickup_needed(worker, ordered)
        animals = _item_pickups(worker, ordered)
        fertilizer = _fertilizer_pickup_needed(worker, ordered)
        door_pickups = animals + (("FERTILIZER", fertilizer),) if fertilizer > 0 else animals
        if pickup or door_pickups:
            travel = _manhattan(worker.coord, door) + _path_moves(door, [visit.coord for visit in ordered])
            leg = _with_return(door, pickup, ordered, travel, doors, door_pickups)
        else:
            travel = _path_moves(worker.coord, [visit.coord for visit in ordered])
            leg = _with_return(None, 0, ordered, travel, doors)
        key = (leg.action_total, leg.travel + leg.return_travel, door)
        if best is None or key < best[0]:
            best = (key, leg)
    if best is None:
        return _with_return(None, pickup, ordered, 10**9, doors, item_pickups)
    return best[1]


def _leg_fits(leg: _Leg, start_hour: int, end_hour: int) -> bool:
    if (leg.pickup or leg.item_pickups) and leg.door is None:
        return False
    if leg.deliveries and leg.return_door is None:
        return False
    if leg.action_total == 0:
        return True
    return start_hour + leg.action_total - 1 <= end_hour


def _added_cost(old: _Leg | None, new: _Leg) -> int:
    """Extra hours for the route that will actually be walked.

    Dropping a return that no longer fits is not a saving. Otherwise the person
    who cancels a long walk back looks cheaper than the person already standing
    on the tile.
    """

    if old is not None and old.deliveries and not new.deliveries:
        previous = old.action_total - old.return_travel - len(old.deliveries)
        return new.action_total - previous
    return new.action_total - (old.action_total if old is not None else 0)


def _fit_leg(
    worker: RegionWorker,
    visits: Sequence[TileVisit],
    doors: Sequence[tuple[int, int]],
    start_hour: int,
    end_hour: int,
) -> _Leg | None:
    """Return the walk that unloads today, or the same field walk with no unload.

    A return that would pass the last hour is dropped. The field jobs stay.
    If those jobs themselves do not fit, there is no route.
    """

    full = _layout(worker, visits, doors)
    if _leg_fits(full, start_hour, end_hour):
        return full
    field_only = _without_return(full)
    if _leg_fits(field_only, start_hour, end_hour):
        return field_only
    return None


def _shed_delta(worker: RegionWorker, leg: _Leg) -> int:
    """How this route changes what the shed holds after the day-end drop.

    A pickup leaves the shed. An unload enters it. Goods still in the hands
    enter at the end of the day. An unload is not also counted in the hands.
    """

    removed = leg.pickup + sum(amount for _, amount in leg.item_pickups)
    placed = sum(units for _, units in leg.deliveries)
    carried = sum(_end_inventory(worker, leg).values())
    return placed + carried - removed


def _project_end_shed(
    workers: Sequence[RegionWorker],
    routes: Sequence[Sequence[TileVisit]],
    pantry: _Pantry,
    shed_total: int,
    start_hour: int,
    end_hour: int,
) -> int:
    """Shed contents after every planned route and the day-end drop.

    A route that does not fit is treated as over the cap, so it cannot be kept.
    """

    shed = shed_total
    for worker, route in zip(workers, routes):
        leg = _fit_leg(worker, route, pantry.doors, start_hour, end_hour)
        if leg is None:
            return SHED_CAPACITY + 1
        shed += _shed_delta(worker, leg)
    return shed


def _end_day_capacity_safe(
    workers: Sequence[RegionWorker],
    routes: Sequence[Sequence[TileVisit]],
    pantry: _Pantry,
    shed_total: int,
    start_hour: int,
    end_hour: int,
) -> bool:
    """True when the whole crew's day-end drop still fits in the shed."""

    return _project_end_shed(workers, routes, pantry, shed_total, start_hour, end_hour) <= SHED_CAPACITY


def _prepare(
    worker: RegionWorker,
    visits: Sequence[TileVisit],
    doors: Sequence[tuple[int, int]],
    start_hour: int,
    end_hour: int,
) -> tuple[_Leg, tuple[TileVisit, ...]]:
    """Keep every visit that fits, unloading only when the return also fits.

    A harvest is not dropped just because the walk back would pass the last hour.
    Visits are shortened only when the field jobs themselves do not fit.
    """

    items = list(visits)
    if not items:
        return _Leg(None, 0, (), 0), ()
    chosen = _fit_leg(worker, items, doors, start_hour, end_hour)
    if chosen is not None:
        return chosen, ()
    ordered = list(_layout(worker, items, doors).visits)
    for length in range(len(ordered) - 1, -1, -1):
        chosen = _fit_leg(worker, ordered[:length], doors, start_hour, end_hour)
        if chosen is not None:
            kept = {visit.coord for visit in chosen.visits}
            pending = tuple(visit for visit in ordered if visit.coord not in kept)
            return chosen, pending
    return _Leg(None, 0, (), 0), tuple(ordered)


def _best_insertion(
    workers: Sequence[RegionWorker],
    routes: list[list[TileVisit]],
    visit: TileVisit,
    start_hour: int,
    end_hour: int,
    pantry: _Pantry | None = None,
) -> tuple[int, list[TileVisit]] | None:
    """Cheapest worker after the route is reordered, including a wheat stop."""

    if pantry is None:
        pantry = _Pantry(0, shed_doors(10))
    best: tuple[tuple[int, int], int, list[TileVisit]] | None = None
    drawn = _wheat_draw(workers, routes)
    for worker_index, worker in enumerate(workers):
        current = routes[worker_index]
        trial = current + [visit]
        if drawn - _pickup_needed(worker, current) + _pickup_needed(worker, trial) > pantry.wheat:
            continue
        new = _fit_leg(worker, trial, pantry.doors, start_hour, end_hour)
        if new is None:
            continue
        trial_routes = [list(route) for route in routes]
        trial_routes[worker_index] = list(new.visits)
        if not _crew_can_carry(workers, trial_routes, pantry, start_hour, end_hour):
            continue
        old = _fit_leg(worker, current, pantry.doors, start_hour, end_hour)
        extra = _added_cost(old, new)
        key = (extra, worker_index)
        if best is None or key < best[0]:
            best = (key, worker_index, list(new.visits))
    if best is None:
        return None
    return best[1], best[2]


def _insertion_extra(
    start: tuple[int, int],
    route: list[tuple[int, int]],
    index: int,
    coord: tuple[int, int],
) -> int:
    if not route:
        return _manhattan(start, coord)
    if index == 0:
        return _manhattan(start, coord) + _manhattan(coord, route[0]) - _manhattan(start, route[0])
    if index == len(route):
        return _manhattan(route[-1], coord)
    previous = route[index - 1]
    nxt = route[index]
    return _manhattan(previous, coord) + _manhattan(coord, nxt) - _manhattan(previous, nxt)


def _search(
    workers: Sequence[RegionWorker],
    routes: list[list[TileVisit]],
    leftover: list[TileVisit],
    start_hour: int,
    end_hour: int,
    pantry: _Pantry,
) -> "_Score":
    current = _score(workers, routes, leftover, start_hour, end_hour, pantry)
    for _ in range(_MAX_ROUNDS):
        improved = _improve_once(workers, current, start_hour, end_hour, pantry)
        if improved is None:
            break
        current = improved
    return current


def _improve_once(
    workers: Sequence[RegionWorker],
    current: "_Score",
    start_hour: int,
    end_hour: int,
    pantry: _Pantry,
) -> "_Score | None":
    """Put a dropped tile back before moving or swapping work between people.

    A tile that did not fit on the first pass can fit after someone else's plot
    moves. Fewer unfinished tiles beat a shorter walk.
    """

    routes = current.routes
    for visit in current.leftover:
        for target in range(len(workers)):
            trial = _place(routes, target, visit)
            if not _crew_can_carry(workers, trial, pantry, start_hour, end_hour):
                continue
            remaining = [item for item in current.leftover if item.coord != visit.coord]
            scored = _score(workers, trial, remaining, start_hour, end_hour, pantry)
            if not _crew_can_carry(workers, scored.routes, pantry, start_hour, end_hour):
                continue
            if len(scored.leftover) < len(current.leftover):
                return scored
    for source, worker_route in enumerate(routes):
        for visit in sorted(worker_route, key=_visit_sort):
            for target in range(len(workers)):
                if target == source:
                    continue
                trial = _move(routes, source, target, visit)
                if not _crew_can_carry(workers, trial, pantry, start_hour, end_hour):
                    continue
                scored = _score(workers, trial, list(current.leftover), start_hour, end_hour, pantry)
                if _crew_can_carry(workers, scored.routes, pantry, start_hour, end_hour) and scored.key < current.key:
                    return scored
    for left in range(len(workers)):
        for right in range(left + 1, len(workers)):
            for first in routes[left]:
                for second in routes[right]:
                    trial = _swap(routes, left, right, first, second)
                    if not _crew_can_carry(workers, trial, pantry, start_hour, end_hour):
                        continue
                    scored = _score(workers, trial, list(current.leftover), start_hour, end_hour, pantry)
                    if _crew_can_carry(workers, scored.routes, pantry, start_hour, end_hour) and scored.key < current.key:
                        return scored
    return None


def _crew_can_carry(
    workers: Sequence[RegionWorker],
    routes: Sequence[Sequence[TileVisit]],
    pantry: _Pantry,
    start_hour: int,
    end_hour: int,
) -> bool:
    """Check actual ordered walks and pickups against one shared pantry."""

    legs = [
        _fit_leg(worker, route, pantry.doors, start_hour, end_hour)
        for worker, route in zip(workers, routes)
    ]
    if any(leg is None for leg in legs):
        return False
    if sum(leg.pickup for leg in legs if leg is not None) > pantry.wheat:
        return False
    drawn: dict[str, int] = {}
    for leg in legs:
        if leg is None:
            continue
        for name, amount in leg.item_pickups:
            drawn[name] = drawn.get(name, 0) + amount
    if drawn.get("FERTILIZER", 0) > pantry.fertilizer:
        return False
    if any(drawn.get(name, 0) > _shed_animal_count(pantry, name) for name in ANIMAL_NAMES):
        return False
    return _end_day_capacity_safe(workers, routes, pantry, pantry.shed_total, start_hour, end_hour)


def _place(
    routes: Sequence[Sequence[TileVisit]],
    target: int,
    visit: TileVisit,
) -> list[list[TileVisit]]:
    trial = [list(route) for route in routes]
    trial[target] = trial[target] + [visit]
    return trial


def _move(
    routes: list[list[TileVisit]],
    source: int,
    target: int,
    visit: TileVisit,
) -> list[list[TileVisit]]:
    trial = [list(route) for route in routes]
    trial[source] = [item for item in trial[source] if item.coord != visit.coord]
    trial[target] = trial[target] + [visit]
    return trial


def _swap(
    routes: list[list[TileVisit]],
    left: int,
    right: int,
    first: TileVisit,
    second: TileVisit,
) -> list[list[TileVisit]]:
    trial = [list(route) for route in routes]
    trial[left] = [item for item in trial[left] if item.coord != first.coord] + [second]
    trial[right] = [item for item in trial[right] if item.coord != second.coord] + [first]
    return trial


@dataclass(frozen=True)
class _Score:
    routes: tuple[tuple[TileVisit, ...], ...]
    leftover: tuple[TileVisit, ...]
    moves: int
    finish: int | None

    @property
    def key(self) -> tuple[int, int, int]:
        return (len(self.leftover), self.moves, self.finish if self.finish is not None else 0)


def _score(
    workers: Sequence[RegionWorker],
    routes: list[list[TileVisit]],
    leftover: list[TileVisit],
    start_hour: int,
    end_hour: int,
    pantry: _Pantry,
) -> _Score:
    ordered: list[tuple[TileVisit, ...]] = []
    moves = 0
    finish: int | None = None
    still_open = list(leftover)
    for worker, route in zip(workers, routes):
        leg, pending = _prepare(worker, route, pantry.doors, start_hour, end_hour)
        actions, route_moves, route_finish, blocked = _expand(worker.coord, leg, start_hour, end_hour)
        del actions
        if blocked:
            still_open.extend(blocked)
            kept = tuple(visit for visit in leg.visits if visit not in blocked)
        else:
            kept = leg.visits
        still_open.extend(pending)
        ordered.append(kept)
        moves += route_moves
        if route_finish is not None:
            finish = route_finish if finish is None else max(finish, route_finish)
    open_visits = tuple(sorted(still_open, key=_visit_sort))
    return _Score(tuple(ordered), open_visits, moves, finish)


def _reorder(start: tuple[int, int], visits: list[TileVisit]) -> list[TileVisit]:
    if len(visits) <= 1:
        return list(visits)
    by_coord = {visit.coord: visit for visit in visits}
    coords = tuple(sorted(by_coord))
    return [by_coord[coord] for coord in _order_coords(start, coords)]


@lru_cache(maxsize=4096)
def _order_coords(
    start: tuple[int, int],
    coords: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...]:
    points = list(coords)
    if len(points) <= EXACT_TILES:
        return tuple(_held_karp(start, points))
    return tuple(_two_opt(start, _nearest_insertion(start, points)))


def _fits(
    worker: RegionWorker,
    visits: Sequence[TileVisit],
    doors: Sequence[tuple[int, int]],
    start_hour: int,
    end_hour: int,
) -> bool:
    """True when the field jobs finish by end_hour, unloading only if that also fits."""

    return _fit_leg(worker, visits, doors, start_hour, end_hour) is not None


def _add_same_tile_optional_tasks(
    grid: TaskGrid,
    workers: Sequence[RegionWorker],
    scored: _Score,
    start_hour: int,
    end_hour: int,
    pantry: _Pantry,
) -> _Score:
    """Add care or extra-yield water only on tiles this route already visits.

    Each added action costs one hour and is kept only when the field jobs still
    finish by end_hour. The walk back is kept when it fits, and dropped when it
    does not. A tile that is not already on the mandatory route is left alone.
    """

    routes = [list(route) for route in scored.routes]
    candidates = _optional_candidates(grid, routes)
    current = scored
    for candidate in candidates:
        trial = [list(route) for route in routes]
        trial[candidate.worker_index] = [
            _with_added_task(visit, candidate.kind) if visit.coord == candidate.coord else visit
            for visit in trial[candidate.worker_index]
        ]
        if not _fits(
            workers[candidate.worker_index],
            trial[candidate.worker_index],
            pantry.doors,
            start_hour,
            end_hour,
        ):
            continue
        if not _crew_can_carry(workers, trial, pantry, start_hour, end_hour):
            continue
        updated = _score(workers, trial, list(current.leftover), start_hour, end_hour, pantry)
        if {visit.coord for visit in updated.leftover} != {visit.coord for visit in current.leftover}:
            continue
        if not _crew_can_carry(workers, updated.routes, pantry, start_hour, end_hour):
            continue
        routes = [list(route) for route in updated.routes]
        current = updated
    return current


@dataclass(frozen=True)
class _Optional:
    benefit: int
    coord: tuple[int, int]
    kind: str
    worker_index: int


def _optional_candidates(grid: TaskGrid, routes: Sequence[Sequence[TileVisit]]) -> list[_Optional]:
    found: list[_Optional] = []
    for worker_index, route in enumerate(routes):
        for visit in route:
            cell = _cell_at(grid, visit.coord)
            if cell is None:
                continue
            water = cell.tasks.get(WATER)
            if WATER not in visit.tasks and _open_optional_water(water):
                found.append(_Optional(int(water.yield_gain), visit.coord, WATER, worker_index))
            care = cell.tasks.get(CARE)
            if (
                CARE not in visit.tasks
                and _open_care(care)
                and (FEED in visit.tasks or HARVEST in visit.tasks)
            ):
                found.append(_Optional(int(care.bonus_gain), visit.coord, CARE, worker_index))
            collect = cell.tasks.get(COLLECT_FERTILIZER)
            if (
                COLLECT_FERTILIZER not in visit.tasks
                and _open_collect(collect)
                and (FEED in visit.tasks or CARE in visit.tasks or HARVEST in visit.tasks)
            ):
                found.append(_Optional(1, visit.coord, COLLECT_FERTILIZER, worker_index))
    found.sort(key=lambda item: (-item.benefit, item.coord[1], item.coord[0], item.kind))
    return found


def _cell_at(grid: TaskGrid, coord: tuple[int, int]):
    x, y = coord
    if not (0 <= x < grid.width and 0 <= y < grid.height):
        return None
    return grid[x][y]


def _open_optional_water(task: object) -> bool:
    if task is None or getattr(task, "status", None) != PENDING:
        return False
    if getattr(task, "mandatory", False):
        return False
    return int(getattr(task, "yield_gain", 0) or 0) > 0


def _open_care(task: object) -> bool:
    if task is None or getattr(task, "status", None) != PENDING:
        return False
    return int(getattr(task, "bonus_gain", 0) or 0) > 0


def _open_collect(task: object) -> bool:
    return task is not None and getattr(task, "status", None) == PENDING and bool(
        getattr(task, "fertilizer_ready", False)
    )


def _with_added_task(visit: TileVisit, kind: str) -> TileVisit:
    ordered = _task_order(visit.tile_type, [*visit.tasks, kind])
    paired = {task: args for task, args in zip(visit.tasks, visit.task_args)}
    return TileVisit(
        visit.coord,
        ordered,
        len(ordered),
        visit.tile_type,
        visit.harvest_product,
        visit.harvest_units,
        tuple(paired.get(task, ()) for task in ordered),
        visit.production_kind,
        visit.production_name,
    )


def _extract_production_visits(
    grid: TaskGrid,
    region_size: int,
    origin: tuple[int, int],
    blocked: Sequence[tuple[int, int]] = (),
    include_unpaid: bool = False,
) -> tuple[TileVisit, ...]:
    """Empty-tile plans this square may start today.

    A purchase is not a field job. The caller buys a seed or an animal before
    the planned hour, then this visit only plants, builds, and places.
    Without `include_unpaid`, a plan that still needs cash stays off the route.
    """

    refused = set(blocked)
    visits: list[TileVisit] = []
    x0, y0 = origin
    for x in range(x0, x0 + region_size):
        for y in range(y0, y0 + region_size):
            cell = _cell_at(grid, (x, y))
            plan = getattr(cell, "production_plan", None) if cell is not None else None
            if plan is None or cell.coord in refused:
                continue
            if not include_unpaid and int(getattr(plan, "startup_cash", 1) or 0) != 0:
                continue
            if include_unpaid and getattr(plan, "kind", None) not in {"crop", "animal"}:
                continue
            actions = tuple(getattr(plan, "actions", ()) or ())
            if not actions or not _production_pending(cell, actions):
                continue
            tasks = tuple(action.operation for action in actions)
            args = tuple(
                (action.subject,) if action.operation in {PLANT, PLACE_ANIMAL} and action.subject else ()
                for action in actions
            )
            visits.append(
                TileVisit(
                    cell.coord,
                    tasks,
                    len(tasks),
                    cell.tile_type,
                    None,
                    0,
                    args,
                    plan.kind,
                    plan.name,
                )
            )
    return tuple(visits)


def _production_pending(cell: object, actions: Sequence[object]) -> bool:
    tasks = getattr(cell, "tasks", {})
    for action in actions:
        task = tasks.get(action.operation)
        # Older in-memory test grids may predate the conditional first-feed
        # task.  The real TaskGrid always registers it; treating a missing
        # legacy entry as pending keeps those plans routeable while the new
        # chain remains enforced for freshly built grids.
        if action.operation == FEED and task is None:
            continue
        if task is None or getattr(task, "status", None) != PENDING:
            return False
    return True


def _add_production_plans(
    grid: TaskGrid,
    workers: Sequence[RegionWorker],
    scored: _Score,
    start_hour: int,
    end_hour: int,
    pantry: _Pantry,
    region_size: int,
    origin: tuple[int, int],
    blocked: Sequence[tuple[int, int]] = (),
    include_unpaid: bool = False,
) -> _Score:
    """Try each committed plan after the mandatory route is fixed.

    The cheapest extra walk goes first. A plan is kept only when its whole
    chain fits by end_hour and no mandatory visit is pushed out. A skipped
    plan is not unfinished work.
    """

    # Existing survival work is the hard gate for starting a new line.  A
    # production chain may include its own conditional first FEED, but it must
    # never consume time that leaves a real mandatory WATER/FEED/HARVEST visit
    # unfinished.
    if scored.leftover:
        return scored

    routes = [list(route) for route in scored.routes]
    pending = [
        visit
        for visit in _extract_production_visits(grid, region_size, origin, blocked, include_unpaid)
        if all(visit.coord not in {item.coord for item in route} for route in routes)
    ]
    while pending:
        choice: tuple[tuple, int, int, list[TileVisit]] | None = None
        for index, visit in enumerate(pending):
            money = _production_money(grid, visit)
            for worker_index, worker in enumerate(workers):
                trial = [list(route) for route in routes]
                trial[worker_index] = trial[worker_index] + [visit]
                old = _fit_leg(worker, routes[worker_index], pantry.doors, start_hour, end_hour)
                new = _fit_leg(worker, trial[worker_index], pantry.doors, start_hour, end_hour)
                if new is None or not _production_fits(worker, new, start_hour, end_hour):
                    continue
                trial[worker_index] = list(new.visits)
                if not _crew_can_carry(workers, trial, pantry, start_hour, end_hour):
                    continue
                extra = _added_cost(old, new)
                key = (extra, -money, visit.coord[1], visit.coord[0], worker.id)
                if choice is None or key < choice[0]:
                    choice = (key, worker_index, index, list(new.visits))
        if choice is None:
            break
        _, worker_index, index, ordered = choice
        routes[worker_index] = ordered
        pending.pop(index)
    return _Score(tuple(tuple(route) for route in routes), scored.leftover, *_route_totals(workers, routes, start_hour, end_hour, pantry))


def _production_money(grid: TaskGrid, visit: TileVisit) -> float:
    cell = _cell_at(grid, visit.coord)
    plan = getattr(cell, "production_plan", None) if cell is not None else None
    if plan is None:
        return 0.0
    return float(getattr(plan, "money_per_day", 0.0) or 0.0)


def _production_fits(worker: RegionWorker, leg: _Leg, start_hour: int, end_hour: int) -> bool:
    """The chosen leg, return or not, still plays every task in each visit."""

    if not _leg_fits(leg, start_hour, end_hour):
        return False
    _, _, _, blocked = _expand(worker.coord, leg, start_hour, end_hour)
    return not blocked


def _route_totals(
    workers: Sequence[RegionWorker],
    routes: Sequence[Sequence[TileVisit]],
    start_hour: int,
    end_hour: int,
    pantry: _Pantry,
) -> tuple[int, int | None]:
    moves = 0
    finish: int | None = None
    for worker, route in zip(workers, routes):
        leg = _fit_leg(worker, route, pantry.doors, start_hour, end_hour)
        if leg is None:
            raise RuntimeError("a stored route does not fit in the day")
        _, route_moves, route_finish, blocked = _expand(worker.coord, leg, start_hour, end_hour)
        if blocked:
            raise RuntimeError("a stored route does not fit in the day")
        moves += route_moves
        if route_finish is not None:
            finish = route_finish if finish is None else max(finish, route_finish)
    return moves, finish


def _expand(
    start: tuple[int, int],
    leg: _Leg,
    start_hour: int,
    end_hour: int,
) -> tuple[tuple[RouteAction, ...], int, int | None, tuple[TileVisit, ...]]:
    actions: list[RouteAction] = []
    moves = 0
    hour = start_hour
    position = start
    visits = leg.visits
    if leg.pickup or leg.item_pickups:
        if leg.door is None:
            return (), 0, None, visits
        while position != leg.door:
            if hour > end_hour:
                return tuple(actions), moves, _finish(actions), visits
            position, operation = _step_toward(position, leg.door)
            actions.append(RouteAction(hour, operation))
            hour += 1
            moves += 1
        if leg.pickup:
            if hour > end_hour:
                return tuple(actions), moves, _finish(actions), visits
            actions.append(RouteAction(hour, "PICKUP", leg.door, ("WHEAT", leg.pickup)))
            hour += 1
        for item, count in leg.item_pickups:
            if hour > end_hour:
                return tuple(actions), moves, _finish(actions), visits
            actions.append(RouteAction(hour, "PICKUP", leg.door, (item, count)))
            hour += 1
    for index, visit in enumerate(visits):
        cursor = position
        walked: list[RouteAction] = []
        blocked = False
        while cursor != visit.coord:
            if hour > end_hour:
                blocked = True
                break
            cursor, operation = _step_toward(cursor, visit.coord)
            walked.append(RouteAction(hour, operation))
            hour += 1
        if blocked:
            return tuple(actions), moves, _finish(actions), tuple(visits[index:])
        task_actions: list[RouteAction] = []
        task_hour = hour
        for task_index, kind in enumerate(visit.tasks):
            if task_hour > end_hour:
                blocked = True
                break
            args = visit.task_args[task_index] if task_index < len(visit.task_args) else ()
            operation = "PLACE" if kind == PLACE_ANIMAL else kind
            task_actions.append(RouteAction(task_hour, operation, visit.coord, tuple(args)))
            task_hour += 1
        if blocked:
            return tuple(actions), moves, _finish(actions), tuple(visits[index:])
        actions.extend(walked)
        actions.extend(task_actions)
        moves += len(walked)
        hour = task_hour
        position = visit.coord
    if not leg.deliveries:
        return tuple(actions), moves, _finish(actions), ()
    if leg.return_door is None:
        return tuple(actions), moves, _finish(actions), visits
    while position != leg.return_door:
        if hour > end_hour:
            return tuple(actions), moves, _finish(actions), visits
        position, operation = _step_toward(position, leg.return_door)
        actions.append(RouteAction(hour, operation))
        hour += 1
        moves += 1
    for product, units in leg.deliveries:
        if hour > end_hour:
            return tuple(actions), moves, _finish(actions), visits
        actions.append(RouteAction(hour, "PLACE", leg.return_door, (product, units)))
        hour += 1
    return tuple(actions), moves, _finish(actions), ()


def _finish(actions: list[RouteAction]) -> int | None:
    if not actions:
        return None
    return actions[-1].hour


def _materialize(
    workers: Sequence[RegionWorker],
    scored: _Score,
    start_hour: int,
    end_hour: int,
    pantry: _Pantry,
) -> RegionRoutePlan:
    # A partial replan may reach here after the workers' actual inventories or
    # positions changed. Never turn an overdrawn forecast into PICKUP actions:
    # replay the existing visits through the same admission check, preserving
    # as many executable stops as the real pantry permits.
    if not _crew_can_carry(workers, scored.routes, pantry, start_hour, end_hour):
        safe_routes: list[list[TileVisit]] = [[] for _ in workers]
        unassigned = list(scored.leftover)
        for route in scored.routes:
            for visit in route:
                placed = _best_insertion(workers, safe_routes, visit, start_hour, end_hour, pantry)
                if placed is None:
                    unassigned.append(visit)
                else:
                    safe_routes[placed[0]] = placed[1]
        scored = _score(workers, safe_routes, unassigned, start_hour, end_hour, pantry)
    plans: list[WorkerRoutePlan] = []
    unfinished: list[TileVisit] = list(scored.leftover)
    total_moves = 0
    finish_hour: int | None = None
    for worker, visits in zip(workers, scored.routes):
        leg, pending = _prepare(worker, visits, pantry.doors, start_hour, end_hour)
        actions, moves, finish, blocked = _expand(worker.coord, leg, start_hour, end_hour)
        # ``_score`` normally stores only visits that fit.  A replan can still
        # arrive here with a route whose final walk became too long after the
        # worker's real position or carried stock changed.  Materialize the
        # executable prefix and carry the rest forward instead of turning a
        # recoverable partial day into a fatal episode error.
        unfinished.extend(pending)
        unfinished.extend(blocked)
        total_moves += moves
        if finish is not None:
            finish_hour = finish if finish_hour is None else max(finish_hour, finish)
        plans.append(
            WorkerRoutePlan(
                worker.id,
                worker.coord,
                leg.visits,
                actions,
                moves,
                finish,
            )
        )
    if not _end_day_capacity_safe(workers, scored.routes, pantry, pantry.shed_total, start_hour, end_hour):
        # Keep the already materialized work; the next observation will rebuild
        # the route with the updated shed contents.  This mirrors the partial
        # route contract above and avoids rejecting a valid prefix.
        unfinished.extend(
            visit
            for route in scored.routes
            for visit in route
            if visit not in unfinished and not any(
                planned.coord == visit.coord for plan in plans for planned in plan.visits
            )
        )
    seen: set[tuple[int, int]] = set()
    remaining_list: list[TileVisit] = []
    for visit in unfinished:
        if visit.coord in seen:
            continue
        seen.add(visit.coord)
        remaining_list.append(visit)
    remaining = tuple(remaining_list)
    feasible = not remaining
    return RegionRoutePlan(
        feasible,
        tuple(plans),
        total_moves,
        finish_hour,
        remaining,
    )


def _empty_plan(workers: Sequence[RegionWorker]) -> RegionRoutePlan:
    idle = tuple(
        WorkerRoutePlan(worker.id, worker.coord, (), (), 0, None)
        for worker in workers
    )
    return RegionRoutePlan(True, idle, 0, None, ())


def _held_karp(start: tuple[int, int], coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    count = len(coords)
    if count <= 1:
        return list(coords)
    size = 1 << count
    infinity = 10**9
    costs = [[infinity] * count for _ in range(size)]
    parent = [[-1] * count for _ in range(size)]
    for end in range(count):
        costs[1 << end][end] = _manhattan(start, coords[end])
    for mask in range(size):
        for end in range(count):
            if mask & (1 << end) == 0 or costs[mask][end] >= infinity:
                continue
            for nxt in range(count):
                if mask & (1 << nxt):
                    continue
                combined = mask | (1 << nxt)
                cost = costs[mask][end] + _manhattan(coords[end], coords[nxt])
                if cost < costs[combined][nxt]:
                    costs[combined][nxt] = cost
                    parent[combined][nxt] = end
    full = size - 1
    end = min(range(count), key=lambda index: (costs[full][index], coords[index]))
    order: list[tuple[int, int]] = []
    mask = full
    while end >= 0:
        order.append(coords[end])
        previous = parent[mask][end]
        mask &= ~(1 << end)
        end = previous
    order.reverse()
    return order


def _nearest_insertion(start: tuple[int, int], coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    remaining = sorted(coords)
    seed = min(remaining, key=lambda coord: (_manhattan(start, coord), coord))
    remaining.remove(seed)
    route = [seed]
    while remaining:
        choice: tuple[tuple[int, tuple[int, int], int], tuple[int, int], int] | None = None
        for coord in remaining:
            for index in range(len(route) + 1):
                extra = _insertion_extra(start, route, index, coord)
                key = (extra, coord, index)
                if choice is None or key < choice[0]:
                    choice = (key, coord, index)
        assert choice is not None
        route.insert(choice[2], choice[1])
        remaining.remove(choice[1])
    return route


def _two_opt(start: tuple[int, int], route: list[tuple[int, int]]) -> list[tuple[int, int]]:
    route = list(route)
    guard = 0
    improved = True
    while improved and guard < 100:
        guard += 1
        improved = False
        base = _path_moves(start, route)
        for left in range(len(route)):
            for right in range(left + 1, len(route)):
                candidate = route[:left] + list(reversed(route[left : right + 1])) + route[right + 1 :]
                cost = _path_moves(start, candidate)
                if cost < base:
                    route = candidate
                    improved = True
                    break
            if improved:
                break
    return route


def _path_moves(start: tuple[int, int], coords: list[tuple[int, int]]) -> int:
    total = 0
    previous = start
    for coord in coords:
        total += _manhattan(previous, coord)
        previous = coord
    return total


def _manhattan(start: tuple[int, int], end: tuple[int, int]) -> int:
    return abs(start[0] - end[0]) + abs(start[1] - end[1])


def _step_toward(position: tuple[int, int], target: tuple[int, int]) -> tuple[tuple[int, int], str]:
    x, y = position
    if x < target[0]:
        return (x + 1, y), "EAST"
    if x > target[0]:
        return (x - 1, y), "WEST"
    if y < target[1]:
        return (x, y + 1), "SOUTH"
    return (x, y - 1), "NORTH"
