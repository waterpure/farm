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

from .route14_state import PENDING, shed_doors
from .task_grid import FEED, HARVEST, WATER, TaskGrid


EXACT_TILES = 10
MUST_KINDS = (WATER, FEED, HARVEST)
_KIND_RANK = {WATER: 0, FEED: 1, HARVEST: 2}
_MAX_ROUNDS = 24


@dataclass(frozen=True)
class RegionWorker:
    """A person already standing on the board. This planner does not hire."""

    id: str
    coord: tuple[int, int]
    carrying_wheat: int = 0


@dataclass(frozen=True)
class TileVisit:
    """Every must-do job on one tile, done by one worker in one stop."""

    coord: tuple[int, int]
    tasks: tuple[str, ...]
    action_count: int
    tile_type: str
    harvest_product: str | None = None
    harvest_units: int = 0


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
    doors = tuple(shed_coords) if shed_coords is not None else shed_doors(task_grid.width)
    pantry = _Pantry(shed_wheat, doors)
    visits = _extract_visits(task_grid, region_size, origin)
    if not visits:
        return _empty_plan(crew)
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
    if kind in {WATER, FEED} and not getattr(task, "mandatory", False):
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
    """Wheat still in the shed, and the tiles where a worker can take it."""

    wheat: int
    doors: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class _Leg:
    """Field walk, then one return to the shed when the worker is carrying a harvest."""

    door: tuple[int, int] | None
    pickup: int
    visits: tuple[TileVisit, ...]
    travel: int
    return_door: tuple[int, int] | None = None
    return_travel: int = 0
    deliveries: tuple[tuple[str, int], ...] = ()

    @property
    def action_total(self) -> int:
        work = sum(visit.action_count for visit in self.visits)
        return (
            self.travel
            + self.return_travel
            + (1 if self.pickup else 0)
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
) -> _Leg:
    deliveries = _deliveries(ordered)
    if not deliveries or not ordered:
        return _Leg(pickup_door, pickup, tuple(ordered), field_travel)
    door, steps = _return_choice(ordered[-1].coord, doors)
    return _Leg(pickup_door, pickup, tuple(ordered), field_travel, door, steps, deliveries)


def _feed_count(visits: Sequence[TileVisit]) -> int:
    return sum(1 for visit in visits if FEED in visit.tasks)


def _pickup_needed(worker: RegionWorker, visits: Sequence[TileVisit]) -> int:
    """Wheat this worker still has to take for the feeds they already own."""

    return max(0, _feed_count(visits) - max(0, worker.carrying_wheat))


def _wheat_draw(workers: Sequence[RegionWorker], routes: Sequence[Sequence[TileVisit]]) -> int:
    return sum(_pickup_needed(worker, route) for worker, route in zip(workers, routes))


def _within_wheat(
    workers: Sequence[RegionWorker],
    routes: Sequence[Sequence[TileVisit]],
    pantry: _Pantry,
) -> bool:
    return _wheat_draw(workers, routes) <= pantry.wheat


def _layout(worker: RegionWorker, visits: Sequence[TileVisit], doors: Sequence[tuple[int, int]]) -> _Leg:
    """Cheapest full walk: start, optional door and one pickup, then the tiles.

    The door is chosen by that whole walk. A nearer door can lose when the
    animals sit closer to another door.
    """

    items = list(visits)
    pickup = _pickup_needed(worker, items)
    if pickup <= 0 or not items:
        ordered = _reorder(worker.coord, items)
        travel = _path_moves(worker.coord, [visit.coord for visit in ordered])
        return _with_return(None, 0, ordered, travel, doors)
    best: tuple[tuple[int, int, int, tuple[int, int]], _Leg] | None = None
    for door in doors:
        ordered = _reorder(door, items)
        travel = _manhattan(worker.coord, door) + _path_moves(door, [visit.coord for visit in ordered])
        leg = _with_return(door, pickup, ordered, travel, doors)
        key = (leg.action_total, leg.travel + leg.return_travel, door)
        if best is None or key < best[0]:
            best = (key, leg)
    if best is None:
        ordered = _reorder(worker.coord, items)
        return _with_return(None, pickup, ordered, 10**9, doors)
    return best[1]


def _leg_fits(leg: _Leg, start_hour: int, end_hour: int) -> bool:
    if leg.pickup and leg.door is None:
        return False
    if leg.deliveries and leg.return_door is None:
        return False
    if leg.action_total == 0:
        return True
    return start_hour + leg.action_total - 1 <= end_hour


def _prepare(
    worker: RegionWorker,
    visits: Sequence[TileVisit],
    doors: Sequence[tuple[int, int]],
    start_hour: int,
    end_hour: int,
) -> tuple[_Leg, tuple[TileVisit, ...]]:
    """Keep the longest prefix that still fits once the pickup is counted."""

    items = list(visits)
    if not items:
        return _Leg(None, 0, (), 0), ()
    full = _layout(worker, items, doors)
    if _leg_fits(full, start_hour, end_hour):
        return full, ()
    ordered = list(full.visits)
    for length in range(len(ordered) - 1, -1, -1):
        leg = _layout(worker, ordered[:length], doors)
        if _leg_fits(leg, start_hour, end_hour):
            kept = {visit.coord for visit in leg.visits}
            pending = tuple(visit for visit in ordered if visit.coord not in kept)
            return leg, pending
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
        old = _layout(worker, current, pantry.doors)
        new = _layout(worker, trial, pantry.doors)
        if not _fits(worker, trial, pantry.doors, start_hour, end_hour):
            continue
        extra = new.action_total - old.action_total
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
            if not _within_wheat(workers, trial, pantry):
                continue
            remaining = [item for item in current.leftover if item.coord != visit.coord]
            scored = _score(workers, trial, remaining, start_hour, end_hour, pantry)
            if len(scored.leftover) < len(current.leftover):
                return scored
    for source, worker_route in enumerate(routes):
        for visit in sorted(worker_route, key=_visit_sort):
            for target in range(len(workers)):
                if target == source:
                    continue
                trial = _move(routes, source, target, visit)
                if not _within_wheat(workers, trial, pantry):
                    continue
                scored = _score(workers, trial, list(current.leftover), start_hour, end_hour, pantry)
                if scored.key < current.key:
                    return scored
    for left in range(len(workers)):
        for right in range(left + 1, len(workers)):
            for first in routes[left]:
                for second in routes[right]:
                    trial = _swap(routes, left, right, first, second)
                    if not _within_wheat(workers, trial, pantry):
                        continue
                    scored = _score(workers, trial, list(current.leftover), start_hour, end_hour, pantry)
                    if scored.key < current.key:
                        return scored
    return None


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
    """True when the shed stop, the walk, and the jobs all finish by end_hour."""

    return _leg_fits(_layout(worker, visits, doors), start_hour, end_hour)


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
    if leg.pickup:
        if leg.door is None:
            return (), 0, None, visits
        while position != leg.door:
            if hour > end_hour:
                return tuple(actions), moves, _finish(actions), visits
            position, operation = _step_toward(position, leg.door)
            actions.append(RouteAction(hour, operation))
            hour += 1
            moves += 1
        if hour > end_hour:
            return tuple(actions), moves, _finish(actions), visits
        actions.append(RouteAction(hour, "PICKUP", leg.door, ("WHEAT", leg.pickup)))
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
        for kind in visit.tasks:
            if task_hour > end_hour:
                blocked = True
                break
            task_actions.append(RouteAction(task_hour, kind, visit.coord))
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
    plans: list[WorkerRoutePlan] = []
    for worker, visits in zip(workers, scored.routes):
        leg, pending = _prepare(worker, visits, pantry.doors, start_hour, end_hour)
        actions, moves, finish, blocked = _expand(worker.coord, leg, start_hour, end_hour)
        if pending or blocked:
            raise RuntimeError("a stored route does not fit in the day")
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
    feasible = not scored.leftover
    return RegionRoutePlan(
        feasible,
        tuple(plans),
        scored.moves,
        scored.finish,
        scored.leftover,
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
