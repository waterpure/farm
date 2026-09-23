"""How many hands finish today's must-do work, and the route for that crew.

Hour 0 only hires. A new hand cannot move until hour 1, and one hour has at
most ten market orders, so at most ten hands are hired. The walk does not
assume those hands appear on one shared tile. Each worker starts from the
coordinate on the hour-1 observation (WorkerState.coord).

Every candidate headcount gets its own assignment. The search stops at the
first count that finishes every must job. It does not ask whether one more
hand would sell for more than the wage.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .route14_state import WorkerState, worker_name
from .task_grid import HARVEST, TaskGrid
from .worker_route import LAST_HOUR, PlannedAction, RoutePlan, _must_jobs, plan_worker_route


MAX_HIRE_COUNT = 10
ROUTE_START_HOUR = 1
# w^n assignments above this are not enumerated tile by tile.
_EXACT_ASSIGNMENTS = 20000
_JobKey = tuple[str, tuple[int, int]]


@dataclass(frozen=True)
class RouteWorker:
    """One person entering the day route, already standing on a real tile."""

    worker_id: str
    coord: tuple[int, int]


@dataclass(frozen=True)
class WorkerRoutePlan:
    """One worker's queue from the start hour through the last action."""

    worker_id: str
    start_coord: tuple[int, int]
    actions_by_hour: tuple[PlannedAction, ...]
    visited_tiles: tuple[tuple[int, int], ...]
    assigned_tasks: tuple[_JobKey, ...]
    move_count: int
    finish_hour: int | None


@dataclass(frozen=True)
class TeamRoutePlan:
    """Joint routes for the farmer and the hands given to one headcount."""

    feasible: bool
    workers: tuple[WorkerRoutePlan, ...]
    total_move_count: int
    completed_tasks: tuple[_JobKey, ...]
    unfinished_tasks: tuple[_JobKey, ...]


@dataclass(frozen=True)
class HandCountPlan:
    """Smallest hand count whose fresh team route finishes every must job."""

    feasible: bool
    min_required_hands: int | None
    team: TeamRoutePlan
    tried_hand_counts: tuple[int, ...]


def plan_team_routes(
    workers: Sequence[RouteWorker | WorkerState],
    task_grid: TaskGrid,
    start_hour: int = ROUTE_START_HOUR,
) -> TeamRoutePlan:
    """Assign today's must jobs across these workers and route each one.

    Field work starts at hour 1 even if `start_hour` is 0, because hour 0 is
    the hiring hour. Later hours are kept as given, so a midday replan can
    start at the hour being played. Must jobs are mandatory water, mandatory
    feed, and every pending harvest.
    """

    crew = tuple(_as_worker(worker, f"Hand{index}") for index, worker in enumerate(workers))
    hour = ROUTE_START_HOUR if start_hour < ROUTE_START_HOUR else start_hour
    jobs = _jobs_by_tile(task_grid)
    keys = tuple((job.kind, job.coord) for coord in jobs for job in jobs[coord])
    if not keys:
        return _idle_team(crew, (), True)
    if not crew or _impossible(crew, list(jobs), len(keys), hour):
        return _idle_team(crew, keys, False)
    if len(crew) == 1:
        return _from_single(crew[0], plan_worker_route(crew[0].coord, hour, task_grid), keys)
    tiles = list(jobs)
    if _exact_ok(len(tiles), len(crew)):
        found = _search_assignments(crew, tiles, jobs, task_grid, hour)
        if found is None:
            return _idle_team(crew, keys, False)
        return _compose(crew, tiles, jobs, found, keys)
    return _plan_by_strips(crew, tiles, jobs, task_grid, hour, keys)


def plan_minimum_hands(
    farmer: RouteWorker | WorkerState,
    hand_coords: Sequence[RouteWorker | WorkerState | tuple[int, int]],
    task_grid: TaskGrid,
    start_hour: int = ROUTE_START_HOUR,
) -> HandCountPlan:
    """Try 0, 1, 2, ... hands until the must jobs fit, replanning every time.

    `hand_coords` are the real start tiles of Hand1, Hand2, ... in hire order.
    At most ten are used. The first feasible count is returned; a larger crew
    is not priced against wages or the market.
    """

    farmer_worker = _as_worker(farmer, "Farmer")
    offered = list(hand_coords)[:MAX_HIRE_COUNT]
    tried: list[int] = []
    last: TeamRoutePlan | None = None
    for count in range(len(offered) + 1):
        tried.append(count)
        hands = [_hand_worker(index, offered[index]) for index in range(count)]
        last = plan_team_routes((farmer_worker, *hands), task_grid, start_hour)
        if last.feasible:
            return HandCountPlan(True, count, last, tuple(tried))
    assert last is not None
    return HandCountPlan(False, None, last, tuple(tried))


def _as_worker(worker: RouteWorker | WorkerState, fallback: str) -> RouteWorker:
    if isinstance(worker, RouteWorker):
        return worker
    if isinstance(worker, WorkerState):
        return RouteWorker(worker_name(worker.actor), worker.coord)
    raise TypeError(f"expected a worker, got {type(worker).__name__}")


def _hand_worker(index: int, spec: RouteWorker | WorkerState | tuple[int, int]) -> RouteWorker:
    if isinstance(spec, tuple):
        return RouteWorker(f"Hand{index + 1}", (int(spec[0]), int(spec[1])))
    worker = _as_worker(spec, f"Hand{index + 1}")
    if isinstance(spec, WorkerState):
        return worker
    return RouteWorker(worker.worker_id or f"Hand{index + 1}", worker.coord)


def _jobs_by_tile(grid: TaskGrid) -> dict[tuple[int, int], tuple]:
    grouped: dict[tuple[int, int], list] = {}
    for job in _must_jobs(grid):
        grouped.setdefault(job.coord, []).append(job)
    return {coord: tuple(items) for coord, items in grouped.items()}


def _budget(start_hour: int) -> int:
    if start_hour > LAST_HOUR:
        return 0
    return LAST_HOUR - start_hour + 1


def _mst_edge_weights(points: Sequence[tuple[int, int]]) -> list[int]:
    count = len(points)
    if count <= 1:
        return []
    used = [False] * count
    best = [10**6] * count
    best[0] = 0
    weights: list[int] = []
    for _ in range(count):
        choice = -1
        for index in range(count):
            if not used[index] and (choice < 0 or best[index] < best[choice]):
                choice = index
        used[choice] = True
        if choice != 0:
            weights.append(best[choice])
        cx, cy = points[choice]
        for other in range(count):
            if used[other]:
                continue
            distance = abs(points[other][0] - cx) + abs(points[other][1] - cy)
            if distance < best[other]:
                best[other] = distance
    return weights


def _cover_lower_bound(worker_count: int, tiles: Sequence[tuple[int, int]], job_count: int) -> int:
    """Actions required even before counting the walk out from each worker."""

    if not tiles:
        return 0
    edges = _mst_edge_weights(tiles)
    drop = min(max(0, worker_count - 1), len(edges))
    longest = sorted(edges, reverse=True)[:drop]
    return job_count + sum(edges) - sum(longest)


def _impossible(workers: Sequence[RouteWorker], tiles: Sequence[tuple[int, int]], job_count: int, start_hour: int) -> bool:
    budget = _budget(start_hour)
    if budget <= 0:
        return True
    return _cover_lower_bound(len(workers), tiles, job_count) > budget * len(workers)


def _exact_ok(tile_count: int, worker_count: int) -> bool:
    if tile_count <= 1 or worker_count <= 1:
        return True
    try:
        return worker_count**tile_count <= _EXACT_ASSIGNMENTS
    except OverflowError:
        return False


def _visit_bounds(start: tuple[int, int], coords: Sequence[tuple[int, int]], job_count: int) -> tuple[int, int]:
    if not coords:
        return 0, 0
    travel = sum(_mst_edge_weights((start, *coords)))
    return travel, travel + job_count


def _adjacent_splits(groups: Sequence[Sequence[int]], tiles: Sequence[tuple[int, int]]) -> int:
    owner = {}
    for worker_index, group in enumerate(groups):
        for tile_index in group:
            owner[tiles[tile_index]] = worker_index
    splits = 0
    for x, y in owner:
        for neighbor in ((x + 1, y), (x, y + 1)):
            if neighbor in owner and owner[neighbor] != owner[(x, y)]:
                splits += 1
    return splits


def _route_one(worker: RouteWorker, coords: Sequence[tuple[int, int]], grid: TaskGrid, start_hour: int) -> RoutePlan:
    if not coords:
        return RoutePlan(True, worker.coord, start_hour, (), (), (), 0, None)
    if len(coords) > 16:
        pending = tuple((HARVEST, coord) for coord in coords)
        return RoutePlan(False, worker.coord, start_hour, (), (), pending, 0, None)
    sub = TaskGrid(grid.width, grid.height)
    for x, y in coords:
        cell = grid[x][y]
        if cell is not None:
            sub.put(cell)
    return plan_worker_route(worker.coord, start_hour, sub)


def _search_assignments(workers, tiles, jobs, grid: TaskGrid, start_hour: int):
    budget = _budget(start_hour)
    groups: list[list[int]] = [[] for _ in workers]
    job_counts = [len(jobs[coord]) for coord in tiles]
    cache: dict[tuple, RoutePlan] = {}
    best: dict | None = None

    def bounds(worker_index: int, ids: Sequence[int]) -> tuple[int, int]:
        coords = [tiles[index] for index in ids]
        count = sum(job_counts[index] for index in ids)
        return _visit_bounds(workers[worker_index].coord, coords, count)

    def cached(worker_index: int, ids: Sequence[int]) -> RoutePlan:
        coords = tuple(sorted(tiles[index] for index in ids))
        worker = workers[worker_index]
        key = (worker.worker_id, worker.coord, coords, start_hour)
        plan = cache.get(key)
        if plan is None:
            plan = _route_one(worker, coords, grid, start_hour)
            cache[key] = plan
        return plan

    def rank(worker_index: int, tile: tuple[int, int]) -> tuple[int, int, int]:
        owned = groups[worker_index]
        adjacent = 1
        for index in owned:
            ox, oy = tiles[index]
            if abs(ox - tile[0]) + abs(oy - tile[1]) == 1:
                adjacent = 0
                break
        anchor = tiles[owned[-1]] if owned else workers[worker_index].coord
        return adjacent, abs(anchor[0] - tile[0]) + abs(anchor[1] - tile[1]), worker_index

    def rec(index: int) -> None:
        nonlocal best
        if index == len(tiles):
            travels: list[int] = []
            for worker_index in range(len(workers)):
                travel, actions = bounds(worker_index, groups[worker_index])
                if actions > budget:
                    return
                travels.append(travel)
            if best is not None and sum(travels) > best["moves"]:
                return
            plans = [cached(worker_index, groups[worker_index]) for worker_index in range(len(workers))]
            if not all(plan.feasible for plan in plans):
                return
            moves = sum(plan.move_count for plan in plans)
            key = (
                moves,
                _adjacent_splits(groups, tiles),
                sum(plan.finish_hour or start_hour for plan in plans),
            )
            if best is None or key < best["key"]:
                best = {
                    "key": key,
                    "moves": moves,
                    "plans": plans,
                    "groups": [list(group) for group in groups],
                }
            return
        tile = tiles[index]
        for worker_index in sorted(range(len(workers)), key=lambda item: rank(item, tile)):
            groups[worker_index].append(index)
            _travel, actions = bounds(worker_index, groups[worker_index])
            if actions <= budget:
                rec(index + 1)
            groups[worker_index].pop()

    rec(0)
    return best


def _plan_by_strips(workers, tiles, jobs, grid: TaskGrid, start_hour: int, keys: tuple[_JobKey, ...]) -> TeamRoutePlan:
    """Contiguous slices when the tile count is too large to assign exactly.

    This still gives every tile to one worker and checks the real route. It is
    the fallback for a crowded map that the action bound has not already ruled
    out, and it does not claim the fewest possible steps.
    """

    order = sorted(range(len(tiles)), key=lambda index: (tiles[index][1], tiles[index][0]))
    width = len(workers)
    groups = []
    count = len(order)
    for worker_index in range(width):
        start = worker_index * count // width
        stop = (worker_index + 1) * count // width
        groups.append(order[start:stop])
    plans = []
    for worker_index, group in enumerate(groups):
        coords = [tiles[index] for index in group]
        plans.append(_route_one(workers[worker_index], coords, grid, start_hour))
    found = {"plans": plans, "groups": groups}
    return _compose(workers, tiles, jobs, found, keys)


def _idle_team(workers: Sequence[RouteWorker], keys: tuple[_JobKey, ...], feasible: bool) -> TeamRoutePlan:
    idle = tuple(
        WorkerRoutePlan(worker.worker_id, worker.coord, (), (), (), 0, None)
        for worker in workers
    )
    return TeamRoutePlan(feasible and not keys, idle, 0, (), keys)


def _from_single(worker: RouteWorker, plan: RoutePlan, keys: tuple[_JobKey, ...]) -> TeamRoutePlan:
    visited: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for action in plan.actions:
        if action.coord is not None and action.coord not in seen:
            seen.add(action.coord)
            visited.append(action.coord)
    assigned = tuple(key for key in keys)
    done = set(plan.completed_tasks)
    unfinished = tuple(key for key in keys if key not in done)
    row = WorkerRoutePlan(
        worker.worker_id,
        worker.coord,
        plan.actions,
        tuple(visited),
        assigned,
        plan.move_count,
        plan.finish_hour,
    )
    return TeamRoutePlan(plan.feasible and not unfinished, (row,), plan.move_count, tuple(plan.completed_tasks), unfinished)


def _compose(workers, tiles, jobs, found: dict, keys: tuple[_JobKey, ...]) -> TeamRoutePlan:
    rows: list[WorkerRoutePlan] = []
    completed: list[_JobKey] = []
    claimed: set[_JobKey] = set()
    for worker_index, worker in enumerate(workers):
        plan: RoutePlan = found["plans"][worker_index]
        assigned: list[_JobKey] = []
        for tile_index in found["groups"][worker_index]:
            for job in jobs[tiles[tile_index]]:
                key = (job.kind, job.coord)
                assigned.append(key)
                claimed.add(key)
        visited: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for action in plan.actions:
            if action.coord is not None and action.coord not in seen:
                seen.add(action.coord)
                visited.append(action.coord)
        completed.extend(key for key in plan.completed_tasks if key in claimed)
        rows.append(
            WorkerRoutePlan(
                worker.worker_id,
                worker.coord,
                plan.actions,
                tuple(visited),
                tuple(assigned),
                plan.move_count,
                plan.finish_hour,
            )
        )
    done = set(completed)
    unfinished = tuple(key for key in keys if key not in done)
    feasible = not unfinished and all(plan.feasible for plan in found["plans"])
    moves = sum(row.move_count for row in rows)
    return TeamRoutePlan(feasible, tuple(rows), moves, tuple(completed), unfinished)
