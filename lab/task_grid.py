"""A coordinate grid of what each tile needs right now.

This does not plan routes. A later planner can scan the grid. Crop tiles carry
water and harvest. Animal tiles carry feed, care, and harvest. A job is
completed only when a later world shows it happened. Issuing the command does
not complete it. SCHEDULED means a worker and an hour are written.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from kaggle_environments.envs.kaggriculture.kaggriculture import CROPS as ENGINE_CROPS

from .route14_state import COMPLETED, PENDING, SCHEDULED, TURNS_PER_DAY, AnimalState, CropState, WorldState


BOARD_SIZE = 10
WATER = "WATER"
HARVEST = "HARVEST"
FEED = "FEED"
CARE = "CARE"


@dataclass
class TaskState:
    """One need on one tile."""

    task_type: str
    status: str
    mandatory: bool
    assigned_worker: str | None = None
    planned_hour: int | None = None


@dataclass
class WaterTask(TaskState):
    """Watering this crop today, and what that one action changes."""

    needed: bool = False
    turns_until_weed: int = 0
    yield_gain: int = 0


@dataclass
class HarvestTask(TaskState):
    """Goods sitting on this tile right now."""

    yield_amount: int = 0


@dataclass
class FeedTask(TaskState):
    """Feeding the animal on this tile. One feed takes one wheat."""

    fed_today: bool = False
    consecutive_unfed: int = 0
    wheat_cost: int = 1


@dataclass
class CareTask(TaskState):
    """Caring for the animal on this tile. Feed and care together bank one bonus."""

    cared_today: bool = False
    pending_care_bonus: int = 0
    bonus_gain: int = 0


@dataclass
class TaskBucket:
    """Every need currently standing on one coordinate."""

    coord: tuple[int, int]
    tile_type: str
    tasks: dict[str, TaskState] = field(default_factory=dict)


class TaskGrid:
    """Farm map indexed as TaskGrid[x][y]. A missing cell is locked or not owned."""

    def __init__(self, width: int = BOARD_SIZE, height: int = BOARD_SIZE) -> None:
        self.width = width
        self.height = height
        self._cells: list[list[TaskBucket | None]] = [[None for _ in range(height)] for _ in range(width)]

    def __getitem__(self, x: int) -> list[TaskBucket | None]:
        return self._cells[x]

    def put(self, bucket: TaskBucket) -> None:
        x, y = bucket.coord
        self._cells[x][y] = bucket

    def schedule(self, x: int, y: int, task_type: str, worker: str, hour: int) -> None:
        """Write a worker and an hour. This does not mark the job done."""

        cell = self._cells[x][y]
        if cell is None or task_type not in cell.tasks:
            raise KeyError((x, y, task_type))
        task = cell.tasks[task_type]
        task.status = SCHEDULED
        task.assigned_worker = worker
        task.planned_hour = hour


class TaskGridBuilder:
    """Fill a TaskGrid from one parsed world. Pass the previous grid to keep assignments."""

    def build(self, world: WorldState, previous: TaskGrid | None = None) -> TaskGrid:
        grid = TaskGrid()
        farm = world.farm
        occupied: set[tuple[int, int]] = set()
        for crop in farm.crops:
            occupied.add(crop.position)
            grid.put(self._crop_bucket(world, crop, previous))
        animals = {animal.position: animal for animal in farm.animals}
        for land in farm.lands:
            if land.position in occupied:
                continue
            occupied.add(land.position)
            if land.weed:
                tile_type = "WEED"
            elif land.fallow:
                tile_type = "FALLOW"
            else:
                tile_type = "EMPTY"
            bucket = TaskBucket(land.position, tile_type)
            if land.empty and not land.weed:
                done = _completed_harvest(_previous_task(previous, land.position, HARVEST))
                if done is not None:
                    bucket.tasks[HARVEST] = done
            grid.put(bucket)
        for building in farm.buildings:
            if building.position in occupied:
                continue
            bucket = TaskBucket(building.position, building.animal or building.structure)
            animal = animals.get(building.position)
            harvest = _animal_harvest(animal, _previous_task(previous, building.position, HARVEST))
            if harvest is not None:
                bucket.tasks[HARVEST] = harvest
            if animal is not None:
                bucket.tasks[FEED] = _feed_task(animal, _previous_task(previous, building.position, FEED))
                bucket.tasks[CARE] = _care_task(animal, _previous_task(previous, building.position, CARE))
            grid.put(bucket)
        return grid

    def _crop_bucket(self, world: WorldState, crop: CropState, previous: TaskGrid | None) -> TaskBucket:
        bucket = TaskBucket(crop.position, crop.crop)
        water = _water_task(world, crop, _previous_task(previous, crop.position, WATER))
        if water is not None:
            bucket.tasks[WATER] = water
        harvest = _crop_harvest(crop, _previous_task(previous, crop.position, HARVEST))
        if harvest is not None:
            bucket.tasks[HARVEST] = harvest
        return bucket


def _previous_task(previous: TaskGrid | None, position: tuple[int, int], task_type: str) -> TaskState | None:
    if previous is None:
        return None
    x, y = position
    if x >= previous.width or y >= previous.height:
        return None
    cell = previous[x][y]
    if cell is None:
        return None
    return cell.tasks.get(task_type)


def _water_task(world: WorldState, crop: CropState, prior: TaskState | None) -> WaterTask | None:
    gain = water_yield_gain(crop)
    mandatory = crop.weed_countdown_days == 0
    turns = _turns_until_weed(world.hour, crop.weed_countdown_days)
    if crop.watered_today:
        if prior is None:
            return None
        return WaterTask(
            task_type=WATER,
            status=COMPLETED,
            mandatory=False,
            assigned_worker=prior.assigned_worker,
            planned_hour=prior.planned_hour,
            needed=False,
            turns_until_weed=turns,
            yield_gain=0,
        )
    needed = mandatory or gain > 0
    if not needed:
        return None
    status = PENDING
    worker = None
    hour = None
    if prior is not None and prior.status == SCHEDULED:
        status = SCHEDULED
        worker = prior.assigned_worker
        hour = prior.planned_hour
    return WaterTask(
        task_type=WATER,
        status=status,
        mandatory=mandatory,
        assigned_worker=worker,
        planned_hour=hour,
        needed=True,
        turns_until_weed=turns,
        yield_gain=gain,
    )


def _crop_harvest(crop: CropState, prior: TaskState | None) -> HarvestTask | None:
    if crop.must_harvest:
        return _open_harvest(prior, crop.yield_units)
    if crop.yield_units <= 0:
        return _completed_harvest(prior)
    return None


def _animal_harvest(animal: AnimalState | None, prior: TaskState | None) -> HarvestTask | None:
    if animal is None:
        return None
    if animal.must_harvest:
        return _open_harvest(prior, animal.yield_units)
    if animal.yield_units <= 0:
        return _completed_harvest(prior)
    return None


def _feed_task(animal: AnimalState, prior: TaskState | None) -> FeedTask:
    status, worker, hour = _done_or_open(animal.fed_today, prior)
    return FeedTask(
        FEED,
        status,
        animal.must_feed,
        worker,
        hour,
        animal.fed_today,
        animal.consecutive_unfed,
        0 if animal.fed_today else animal.wheat_per_day,
    )


def _care_task(animal: AnimalState, prior: TaskState | None) -> CareTask:
    status, worker, hour = _done_or_open(animal.cared_today, prior)
    return CareTask(
        CARE,
        status,
        False,
        worker,
        hour,
        animal.cared_today,
        animal.care_bonus,
        0 if animal.fed_today and animal.cared_today else 1,
    )


def _done_or_open(done: bool, prior: TaskState | None) -> tuple[str, str | None, int | None]:
    if done:
        worker = prior.assigned_worker if prior is not None else None
        hour = prior.planned_hour if prior is not None else None
        return COMPLETED, worker, hour
    if prior is not None and prior.status == SCHEDULED:
        return SCHEDULED, prior.assigned_worker, prior.planned_hour
    return PENDING, None, None


def _open_harvest(prior: TaskState | None, yield_amount: int) -> HarvestTask:
    status = PENDING
    worker = None
    hour = None
    if prior is not None and prior.status == SCHEDULED:
        status = SCHEDULED
        worker = prior.assigned_worker
        hour = prior.planned_hour
    return HarvestTask(HARVEST, status, True, worker, hour, yield_amount)


def _completed_harvest(prior: TaskState | None) -> HarvestTask | None:
    """The goods are gone. That counts only when this tile already had a harvest job."""

    if prior is None:
        return None
    return HarvestTask(HARVEST, COMPLETED, False, prior.assigned_worker, prior.planned_hour, 0)


def water_yield_gain(crop: CropState) -> int:
    """Extra units this one WATER adds, compared with not watering.

    One-shot crops gain only inside the official age window: 1, or 2 when
    fertilizer is still active, and never past max_yield. Tomato and strawberry
    do not gain from the water action itself. Water changes tonight's fruit
    only when tonight is a production night and fertilizer is active; the
    unwatered plant still gets 1, so the water is worth the difference.
    """

    spec = ENGINE_CROPS[crop.crop]
    cap = int(spec["max_yield"])
    fertilized = crop.fertilizer_days_left > 0
    if spec["ongoing"]:
        return _ongoing_water_gain(crop, spec, cap, fertilized)
    window_start = (int(spec["max_yield_day"]) + 1) // 2
    if not window_start <= crop.age_days <= int(spec["max_yield_day"]):
        return 0
    bonus = 2 if fertilized else 1
    return min(cap, crop.yield_units + bonus) - crop.yield_units


def _ongoing_water_gain(crop: CropState, spec: dict, cap: int, fertilized: bool) -> int:
    first = int(spec["first_yield_day"])
    interval = int(spec["interval"])
    days_since_first = crop.age_days + 1 - first
    if days_since_first < 0 or interval <= 0 or days_since_first % interval != 0:
        return 0
    production_count = days_since_first // interval + 1
    if production_count > cap:
        return 0
    without_water = min(cap, crop.yield_units + 1)
    with_water = min(cap, crop.yield_units + (2 if fertilized else 1))
    return with_water - without_water


def _turns_until_weed(hour: int, weed_countdown_days: int) -> int:
    """Hours from this hour until the refresh that turns the plant into weed, if it is never watered again."""

    return (TURNS_PER_DAY - hour) + weed_countdown_days * TURNS_PER_DAY
