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

from kaggle_environments.envs.kaggriculture.kaggriculture import CROPS as ENGINE_CROPS

from .route14_state import PENDING
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


@dataclass(frozen=True)
class TileVisit:
    """Every must-do job on one tile, done by one worker in one stop."""

    coord: tuple[int, int]
    tasks: tuple[str, ...]
    action_count: int
    tile_type: str


@dataclass(frozen=True)
class RouteAction:
    """One real engine action. A move is one step, never a jump to a tile."""

    hour: int
    operation: str
    coord: tuple[int, int] | None = None

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
) -> RegionRoutePlan:
    """Assign every must-do tile in one square to the given workers.

    The square starts at `origin` and runs `region_size` cells on both axes.
    The default square is the corner `[0, 5)`. Each tile has one owner.
    Among plans that finish every visit by `end_hour`, fewer steps win, then
    an earlier last hour.
    """

    crew = tuple(workers)
    if not 1 <= len(crew) <= 4:
        raise ValueError("region routes take 1 to 4 workers")
    if region_size < 1:
        raise ValueError("region_size must be positive")
    if start_hour > end_hour:
        raise ValueError("start_hour must be at or before end_hour")
    visits = _extract_visits(task_grid, region_size, origin)
    if not visits:
        return _empty_plan(crew)
    routes: list[list[TileVisit]] = [[] for _ in crew]
    leftover: list[TileVisit] = []
    for visit in sorted(visits, key=_visit_sort):
        placed = _best_insertion(crew, routes, visit, start_hour, end_hour)
        if placed is None:
            leftover.append(visit)
            continue
        worker_index, ordered = placed
        routes[worker_index] = ordered
    scored = _search(crew, routes, leftover, start_hour, end_hour)
    return _materialize(crew, scored, start_hour, end_hour)


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
            visits.append(TileVisit(cell.coord, tasks, len(tasks), cell.tile_type))
    return tuple(visits)


def _is_must(task: object, kind: str) -> bool:
    if task is None or getattr(task, "status", None) != PENDING:
        return False
    if kind in {WATER, FEED} and not getattr(task, "mandatory", False):
        return False
    return True


def _task_order(tile_type: str, kinds: list[str]) -> tuple[str, ...]:
    """Legal order inside one visit. One-shot crops are watered before harvest."""

    ordered = tuple(sorted(kinds, key=lambda kind: _KIND_RANK[kind]))
    spec = ENGINE_CROPS.get(tile_type)
    one_shot = spec is not None and not spec["ongoing"]
    if one_shot and WATER in ordered and HARVEST in ordered:
        watered = tuple(kind for kind in ordered if kind != HARVEST) + (HARVEST,)
        return watered
    return ordered


def _best_insertion(
    workers: Sequence[RegionWorker],
    routes: list[list[TileVisit]],
    visit: TileVisit,
    start_hour: int,
    end_hour: int,
) -> tuple[int, list[TileVisit]] | None:
    """Cheapest feasible slot across workers and positions, not the nearest start."""

    best: tuple[tuple[int, int, int], int, list[TileVisit]] | None = None
    for worker_index, worker in enumerate(workers):
        current = routes[worker_index]
        ordered = _reorder(worker.coord, current + [visit])
        if not _fits(worker.coord, ordered, start_hour, end_hour):
            continue
        coords = [item.coord for item in current]
        for index in range(len(current) + 1):
            extra = _insertion_extra(worker.coord, coords, index, visit.coord)
            key = (extra + visit.action_count, worker_index, index)
            if best is None or key < best[0]:
                best = (key, worker_index, ordered)
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
) -> "_Score":
    current = _score(workers, routes, leftover, start_hour, end_hour)
    for _ in range(_MAX_ROUNDS):
        improved = _improve_once(workers, current, start_hour, end_hour)
        if improved is None:
            break
        current = improved
    return current


def _improve_once(
    workers: Sequence[RegionWorker],
    current: "_Score",
    start_hour: int,
    end_hour: int,
) -> "_Score | None":
    routes = current.routes
    for source, worker_route in enumerate(routes):
        for visit in sorted(worker_route, key=_visit_sort):
            for target in range(len(workers)):
                if target == source:
                    continue
                trial = _move(routes, source, target, visit)
                scored = _score(workers, trial, list(current.leftover), start_hour, end_hour)
                if scored.key < current.key:
                    return scored
    for left in range(len(workers)):
        for right in range(left + 1, len(workers)):
            for first in routes[left]:
                for second in routes[right]:
                    trial = _swap(routes, left, right, first, second)
                    scored = _score(workers, trial, list(current.leftover), start_hour, end_hour)
                    if scored.key < current.key:
                        return scored
    return None


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
) -> _Score:
    ordered: list[tuple[TileVisit, ...]] = []
    moves = 0
    finish: int | None = None
    still_open = list(leftover)
    for worker, route in zip(workers, routes):
        arranged = _reorder(worker.coord, route)
        actions, route_moves, route_finish, pending = _expand(worker.coord, arranged, start_hour, end_hour)
        if pending:
            still_open.extend(pending)
            arranged = tuple(visit for visit in arranged if visit not in pending)
        else:
            arranged = tuple(arranged)
        del actions
        ordered.append(arranged)
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


def _fits(start: tuple[int, int], visits: list[TileVisit], start_hour: int, end_hour: int) -> bool:
    actions = _path_moves(start, [visit.coord for visit in visits]) + sum(visit.action_count for visit in visits)
    if actions == 0:
        return True
    return start_hour + actions - 1 <= end_hour


def _expand(
    start: tuple[int, int],
    visits: list[TileVisit],
    start_hour: int,
    end_hour: int,
) -> tuple[tuple[RouteAction, ...], int, int | None, tuple[TileVisit, ...]]:
    actions: list[RouteAction] = []
    moves = 0
    hour = start_hour
    position = start
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
) -> RegionRoutePlan:
    plans: list[WorkerRoutePlan] = []
    for worker, visits in zip(workers, scored.routes):
        actions, moves, finish, pending = _expand(worker.coord, list(visits), start_hour, end_hour)
        if pending:
            raise RuntimeError("a stored route does not fit in the day")
        plans.append(
            WorkerRoutePlan(
                worker.id,
                worker.coord,
                visits,
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
