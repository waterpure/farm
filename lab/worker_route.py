"""One worker's route across the must-do jobs on a TaskGrid.

The worker starts at a real coordinate. Each move and each water, feed, or
harvest takes one hour. Hour 1 can act through hour 23, which is 23 actions.
There is no hour 24.

The search is branch-and-bound, not "do the nearest job". A step is either a
legal job on the current tile, or one step toward a job that is still open.
Branches whose earliest finish is after hour 23 are cut. Completing every must
job comes first. Among those routes, fewer steps win. The same step count
finishes earlier only when the last action is earlier; with no idle hour,
fewer steps already finish earlier.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

from kaggle_environments.envs.kaggriculture.kaggriculture import CROPS as ENGINE_CROPS

from .route14_state import PENDING
from .task_grid import FEED, HARVEST, WATER, TaskGrid


LAST_HOUR = 23
_DIRECTIONS = (
    ("NORTH", 0, -1),
    ("SOUTH", 0, 1),
    ("EAST", 1, 0),
    ("WEST", -1, 0),
)


@dataclass(frozen=True)
class PlannedAction:
    """One hour of the route. A move has no coordinate. A job names its tile."""

    hour: int
    operation: str
    coord: tuple[int, int] | None = None

    def __str__(self) -> str:
        if self.coord is None:
            return f"Hour {self.hour}: {self.operation}"
        x, y = self.coord
        return f"Hour {self.hour}: {self.operation} ({x},{y})"


@dataclass(frozen=True)
class RoutePlan:
    """Whether this worker can finish every must job, and the route if so."""

    feasible: bool
    start_coord: tuple[int, int]
    start_hour: int
    actions: tuple[PlannedAction, ...]
    completed_tasks: tuple[tuple[str, tuple[int, int]], ...]
    unfinished_tasks: tuple[tuple[str, tuple[int, int]], ...]
    move_count: int
    finish_hour: int | None


@dataclass(frozen=True)
class _Job:
    coord: tuple[int, int]
    kind: str
    clears_plant: bool


def plan_worker_route(
    coord: tuple[int, int],
    current_hour: int,
    grid: TaskGrid,
) -> RoutePlan:
    """Plan one worker from `coord` at `current_hour` over the open must jobs."""

    jobs = _must_jobs(grid)
    start = (int(coord[0]), int(coord[1]))
    if not jobs:
        return _plan(True, start, current_hour, jobs, 0, 0, ())
    if current_hour > LAST_HOUR:
        return _plan(False, start, current_hour, jobs, (1 << len(jobs)) - 1, 0, ())
    found = _search(jobs, start, current_hour, grid.width, grid.height)
    if found is not None:
        return found
    return _beam_partial(jobs, start, current_hour, grid.width, grid.height)


def _must_jobs(grid: TaskGrid) -> tuple[_Job, ...]:
    """Pending mandatory water, mandatory feed, and every pending harvest."""

    jobs: list[_Job] = []
    for x in range(grid.width):
        for y, cell in enumerate(grid[x]):
            if cell is None:
                continue
            spec = ENGINE_CROPS.get(cell.tile_type)
            one_shot = spec is not None and not spec["ongoing"]
            for kind in (WATER, FEED, HARVEST):
                task = cell.tasks.get(kind)
                if task is None or task.status != PENDING:
                    continue
                if kind in {WATER, FEED} and not task.mandatory:
                    continue
                jobs.append(_Job((x, y), kind, kind == HARVEST and one_shot))
    return tuple(jobs)


def _plan(
    feasible: bool,
    start: tuple[int, int],
    start_hour: int,
    jobs: tuple[_Job, ...],
    open_mask: int,
    moves: int,
    actions: tuple[PlannedAction, ...],
) -> RoutePlan:
    done = tuple((action.operation, action.coord) for action in actions if action.coord is not None)
    unfinished = tuple((job.kind, job.coord) for index, job in enumerate(jobs) if open_mask & (1 << index))
    finish = actions[-1].hour if actions else None
    return RoutePlan(feasible, start, start_hour, actions, done, unfinished, moves, finish)


def _search(
    jobs: tuple[_Job, ...],
    start: tuple[int, int],
    start_hour: int,
    width: int,
    height: int,
) -> RoutePlan | None:
    """Best-first search for a route that finishes every job. None when none fits."""

    full = (1 << len(jobs)) - 1
    targets = _target_table(jobs)
    move_bounds: dict[tuple[int, int, int], int] = {}
    if _too_late(start_hour, full, _move_bound(move_bounds, targets, start[0], start[1], full)):
        return None
    # hour, x, y, open mask, cancelled mask, moves, parent, operation, action x, action y
    nodes: list[tuple] = [(start_hour, start[0], start[1], full, 0, 0, -1, "", -1, -1)]
    best: dict[tuple[int, int, int, int], tuple[int, int]] = {(start[0], start[1], full, 0): (0, start_hour)}
    heap: list[tuple[int, int, int]] = [(0, start_hour, 0)]

    while heap:
        moves, hour, index = heapq.heappop(heap)
        node = nodes[index]
        if (moves, hour) != best.get((node[1], node[2], node[3], node[4])):
            continue
        if node[3] == 0 and node[4] == 0:
            return _assemble(True, start, start_hour, jobs, nodes, index)
        if hour > LAST_HOUR:
            continue
        for child in _children(node, index, jobs, targets, width, height):
            child_moves = child[5]
            child_hour = child[0]
            if child[4]:
                continue
            key = (child[1], child[2], child[3], child[4])
            rank = (child_moves, child_hour)
            old = best.get(key)
            if old is not None and old <= rank:
                continue
            bound = _move_bound(move_bounds, targets, child[1], child[2], child[3])
            if _too_late(child_hour, child[3], bound):
                continue
            best[key] = rank
            nodes.append(child)
            heapq.heappush(heap, (child_moves, child_hour, len(nodes) - 1))
    return None


def _beam_partial(
    jobs: tuple[_Job, ...],
    start: tuple[int, int],
    start_hour: int,
    width: int,
    height: int,
) -> RoutePlan:
    """When the full set does not fit, keep the routes that finish the most jobs."""

    targets = _target_table(jobs)
    full = (1 << len(jobs)) - 1
    nodes: list[tuple] = [(start_hour, start[0], start[1], full, 0, 0, -1, "", -1, -1)]
    best_index = 0
    best_score = (full.bit_count(), 0, start_hour)
    layer = [0]
    while layer:
        chosen: dict[tuple[int, int, int, int], int] = {}
        for index in layer:
            node = nodes[index]
            if node[0] > LAST_HOUR:
                continue
            for child in _children(node, index, jobs, targets, width, height):
                if child[0] - 1 > LAST_HOUR:
                    continue
                key = (child[1], child[2], child[3], child[4])
                previous = chosen.get(key)
                if previous is not None and (nodes[previous][5], nodes[previous][0]) <= (child[5], child[0]):
                    continue
                nodes.append(child)
                child_index = len(nodes) - 1
                chosen[key] = child_index
                score = ((child[3] | child[4]).bit_count(), child[5], child[0])
                if score < best_score:
                    best_score = score
                    best_index = child_index
        ranked = sorted(chosen.values(), key=lambda item: ((nodes[item][3] | nodes[item][4]).bit_count(), nodes[item][5], nodes[item][0]))
        layer = ranked[:48]
    finished = nodes[best_index][3] == 0 and nodes[best_index][4] == 0
    return _assemble(finished, start, start_hour, jobs, nodes, best_index)


def _children(
    node: tuple,
    parent: int,
    jobs: tuple[_Job, ...],
    targets: dict[int, tuple[tuple[int, int], ...]],
    width: int,
    height: int,
) -> list[tuple]:
    hour, x, y, mask, cancelled, moves, _parent, _op, _ax, _ay = node
    if hour > LAST_HOUR:
        return []
    children: list[tuple] = []
    for index, job in enumerate(jobs):
        if not mask & (1 << index) or job.coord != (x, y):
            continue
        children.append(_after_job(node, parent, jobs, index))
    for name, dx, dy in _DIRECTIONS:
        nx = x + dx
        ny = y + dy
        if not (0 <= nx < width and 0 <= ny < height):
            continue
        if not _closer(x, y, nx, ny, targets.get(mask, ())):
            continue
        children.append((hour + 1, nx, ny, mask, cancelled, moves + 1, parent, name, -1, -1))
    return children


def _after_job(node: tuple, parent: int, jobs: tuple[_Job, ...], index: int) -> tuple:
    hour, x, y, mask, cancelled, moves, _parent, _op, _ax, _ay = node
    job = jobs[index]
    mask &= ~(1 << index)
    if job.clears_plant:
        for other_index, other in enumerate(jobs):
            bit = 1 << other_index
            if mask & bit and other.kind == WATER and other.coord == job.coord:
                mask &= ~bit
                cancelled |= bit
    return (hour + 1, x, y, mask, cancelled, moves, parent, job.kind, job.coord[0], job.coord[1])


def _closer(x: int, y: int, nx: int, ny: int, points: tuple[tuple[int, int], ...]) -> bool:
    for tx, ty in points:
        if abs(nx - tx) + abs(ny - ty) < abs(x - tx) + abs(y - ty):
            return True
    return False


def _too_late(hour: int, mask: int, move_bound: int) -> bool:
    """True when the remaining jobs cannot finish by hour 23.

    The next action happens at `hour`. Each remaining job and each step in the
    move bound takes one later hour. The last of those hours has to be 23 or earlier.
    """

    needed = mask.bit_count() + move_bound
    if needed == 0:
        return False
    return hour + needed - 1 > LAST_HOUR


def _target_table(jobs: tuple[_Job, ...]) -> dict[int, tuple[tuple[int, int], ...]]:
    table: dict[int, tuple[tuple[int, int], ...]] = {}
    full = 1 << len(jobs)
    for mask in range(full):
        seen: list[tuple[int, int]] = []
        have: set[tuple[int, int]] = set()
        for index, job in enumerate(jobs):
            if mask & (1 << index) and job.coord not in have:
                have.add(job.coord)
                seen.append(job.coord)
        table[mask] = tuple(seen)
    return table


def _move_bound(
    cache: dict[tuple[int, int, int], int],
    targets: dict[int, tuple[tuple[int, int], ...]],
    x: int,
    y: int,
    mask: int,
) -> int:
    """Manhattan spanning tree of here plus the open job tiles. A path is at least this long."""

    key = (x, y, mask)
    cached = cache.get(key)
    if cached is not None:
        return cached
    points = [(x, y)]
    have = {(x, y)}
    for point in targets[mask]:
        if point not in have:
            have.add(point)
            points.append(point)
    total = _mst(points)
    cache[key] = total
    return total


def _mst(points: list[tuple[int, int]]) -> int:
    count = len(points)
    if count <= 1:
        return 0
    used = [False] * count
    best = [10**6] * count
    best[0] = 0
    total = 0
    for _ in range(count):
        choice = -1
        for index in range(count):
            if not used[index] and (choice < 0 or best[index] < best[choice]):
                choice = index
        used[choice] = True
        total += best[choice]
        cx, cy = points[choice]
        for other in range(count):
            if used[other]:
                continue
            distance = abs(points[other][0] - cx) + abs(points[other][1] - cy)
            if distance < best[other]:
                best[other] = distance
    return total


def _assemble(
    feasible: bool,
    start: tuple[int, int],
    start_hour: int,
    jobs: tuple[_Job, ...],
    nodes: list[tuple],
    index: int,
) -> RoutePlan:
    actions: list[PlannedAction] = []
    cursor = index
    while cursor > 0:
        hour, _x, _y, _mask, _cancelled, _moves, parent, operation, ax, ay = nodes[cursor]
        coord = None if ax < 0 else (ax, ay)
        actions.append(PlannedAction(hour - 1, operation, coord))
        cursor = parent
    actions.reverse()
    node = nodes[index]
    open_mask = node[3] | node[4]
    return _plan(feasible and node[3] == 0 and node[4] == 0, start, start_hour, jobs, open_mask, node[5], tuple(actions))
