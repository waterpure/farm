"""A coordinate grid of what each tile needs right now.

This does not plan routes. A later planner can scan the grid. Crop tiles carry
water, harvest, and fertilize. Animal tiles carry feed, care, harvest, and
manure collection. A job is completed only when a later world shows it
happened. Issuing the command does not complete it. SCHEDULED means a worker
and an hour are written.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from kaggle_environments.envs.kaggriculture.kaggriculture import CROPS as ENGINE_CROPS

from .route14_state import (
    COMPLETED,
    PENDING,
    SCHEDULED,
    SEASON_DAYS,
    TURNS_PER_DAY,
    AnimalState,
    CropState,
    WorldState,
)


BOARD_SIZE = 10
WATER = "WATER"
HARVEST = "HARVEST"
FEED = "FEED"
CARE = "CARE"
COLLECT_FERTILIZER = "COLLECT_FERTILIZER"
FERTILIZE = "FERTILIZE"
DIG = "DIG"
PLANT = "PLANT"
BUILD_COOP = "BUILD_COOP"
BUILD_PASTURE = "BUILD_PASTURE"
PLACE_ANIMAL = "PLACE_ANIMAL"
TILE_JOBS = (DIG, PLANT, BUILD_COOP, BUILD_PASTURE, PLACE_ANIMAL)


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
class CollectFertilizerTask(TaskState):
    """Manure is ready on this animal."""

    fertilizer_ready: bool = False


@dataclass
class FertilizeTask(TaskState):
    """Fertilizer would raise this crop's later yield, and none is active yet."""

    fertilized_today: bool = False
    fertilizer_days_left: int = 0


@dataclass
class TileTask(TaskState):
    """Dig a weed, plant, build a shed, or put an animal into that shed.

    `depends_on` names the earlier operation on this same tile. A seedling's
    WATER depends on PLANT and is not the same record as a crop already in the ground.
    """

    subject: str = ""
    depends_on: str = ""


@dataclass
class TaskBucket:
    """Every need currently standing on one coordinate."""

    coord: tuple[int, int]
    tile_type: str
    tasks: dict[str, TaskState] = field(default_factory=dict)
    production_plan: Any = None


class TaskGrid:
    """Farm map indexed as TaskGrid[x][y]. A missing cell is locked or not owned."""

    def __init__(self, width: int = BOARD_SIZE, height: int = BOARD_SIZE) -> None:
        self.width = width
        self.height = height
        self.portfolio: Any = None
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
            if land.weed and land.needs_dig:
                bucket.tasks[DIG] = _open_tile(DIG, _previous_task(previous, land.position, DIG))
            _keep_tile_jobs(bucket, world, previous)
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
                collect = _collect_task(animal, _previous_task(previous, building.position, COLLECT_FERTILIZER))
                if collect is not None:
                    bucket.tasks[COLLECT_FERTILIZER] = collect
            _keep_tile_jobs(bucket, world, previous)
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
        fertilize = _fertilize_task(world, crop, _previous_task(previous, crop.position, FERTILIZE))
        if fertilize is not None:
            bucket.tasks[FERTILIZE] = fertilize
        _keep_tile_jobs(bucket, world, previous)
        return bucket


def build_task_grid(
    world: WorldState,
    previous: TaskGrid | None = None,
    observation: dict[str, Any] | None = None,
) -> TaskGrid:
    """Read the parsed farm and return one bucket per owned tile.

    When `observation` is passed, each empty tile also receives at most one
    committed production plan. The route still decides later whether to start it.
    """

    grid = TaskGridBuilder().build(world, previous)
    if observation is not None:
        from .production_plan import apply_empty_production_plans

        apply_empty_production_plans(grid, world, observation, previous)
    return grid


def apply_assignments(grid: TaskGrid, world: WorldState, assignments: list[Any]) -> None:
    """Mark jobs scheduled. This does not mark them done."""

    for item in assignments:
        x, y = item.position
        if x >= grid.width or y >= grid.height:
            continue
        cell = grid[x][y]
        if cell is None:
            continue
        if item.kind in {WATER, FEED, CARE, HARVEST}:
            task = cell.tasks.get(item.kind)
            if task is None or task.status == COMPLETED:
                continue
            task.status = SCHEDULED
            task.assigned_worker = item.assigned_worker
            task.planned_hour = item.planned_hour
            continue
        if item.kind not in TILE_JOBS:
            continue
        prior = cell.tasks.get(item.kind)
        depends_on = prior.depends_on if isinstance(prior, TileTask) else ""
        subject = item.subject or (prior.subject if isinstance(prior, TileTask) else "")
        if _tile_done(world.farm, item.position, item.kind, subject):
            cell.tasks[item.kind] = TileTask(
                item.kind, COMPLETED, False, item.assigned_worker, item.planned_hour, subject, depends_on
            )
        else:
            cell.tasks[item.kind] = TileTask(
                item.kind, SCHEDULED, False, item.assigned_worker, item.planned_hour, subject, depends_on
            )


def _open_tile(kind: str, prior: TaskState | None) -> TileTask:
    status, worker, hour = _carried(prior)
    subject = prior.subject if isinstance(prior, TileTask) else ""
    return TileTask(kind, status, False, worker, hour, subject)


def _keep_tile_jobs(bucket: TaskBucket, world: WorldState, previous: TaskGrid | None) -> None:
    """Keep a scheduled dig, plant, build, or placement until the board shows it."""

    for kind in TILE_JOBS:
        if kind in bucket.tasks:
            continue
        prior = _previous_task(previous, bucket.coord, kind)
        if prior is None or prior.status == COMPLETED:
            continue
        subject = prior.subject if isinstance(prior, TileTask) else ""
        if _tile_done(world.farm, bucket.coord, kind, subject):
            bucket.tasks[kind] = TileTask(kind, COMPLETED, False, prior.assigned_worker, prior.planned_hour, subject)
        else:
            bucket.tasks[kind] = replace(prior)


def _tile_done(farm: Any, position: tuple[int, int], kind: str, subject: str) -> bool:
    crop = next((item for item in farm.crops if item.position == position), None)
    animal = next((item for item in farm.animals if item.position == position), None)
    land = next((item for item in farm.lands if item.position == position), None)
    building = next((item for item in farm.buildings if item.position == position), None)
    if kind == PLANT:
        return crop is not None and (not subject or crop.crop == subject)
    if kind == BUILD_COOP:
        return building is not None and building.structure == "COOP"
    if kind == BUILD_PASTURE:
        return building is not None and building.structure == "PASTURE"
    if kind == PLACE_ANIMAL:
        return animal is not None and (not subject or animal.animal == subject)
    if kind == DIG:
        return crop is None and animal is None and building is None and land is not None and land.empty and not land.weed
    return False


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
    # A one-shot plant disappears after today's harvest.  Once HARVEST is
    # available, finish the crop directly; do not spend a WATER stop on a
    # plant whose lifecycle ends in the same visit, even if watering could
    # have added one more unit.
    if _harvesting_one_shot(crop):
        return None
    # Survival, production, and yield are separate decisions.  A crop can
    # need mandatory WATER even when it is already at its production cap, but
    # only while the plant remains or an ongoing production event is due.
    mandatory = crop.must_water
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


def should_fertilize(crop: CropState, state: WorldState) -> bool:
    """True only when a fertilizer on hand would raise this crop's later yield."""

    if _fertilizer_on_hand(state) <= 0:
        return False
    if crop.remaining_harvests <= 0:
        return False
    if crop.fertilizer_days_left > 0 or crop.fertilized_today:
        return False
    return fertilizer_yield_gain(crop, state) > 0


def fertilizer_yield_gain(crop: CropState, state: WorldState | None = None) -> int:
    """Extra units from fertilizing now, if every useful water still happens.

    Fertilizer covers today and the next two days. One-shot crops gain the
    extra water point only inside the official window, and only up to max
    yield. Tomato and strawberry gain one extra fruit on each production
    night inside those three days; fruit already hanging is unchanged.
    """

    spec = ENGINE_CROPS[crop.crop]
    days_remaining = (
        max(0, int(state.days_remaining))
        if state is not None
        else max(0, SEASON_DAYS - crop.age_days)
    )
    if spec["ongoing"]:
        return _ongoing_fertilizer_gain(crop, spec, days_remaining)
    return _oneshot_fertilizer_gain(crop, spec, days_remaining)


def _fertilizer_on_hand(state: WorldState) -> int:
    inventory = state.farm.inventory
    if inventory is None:
        return 0
    return inventory.fertilizer


def _collect_task(animal: AnimalState, prior: TaskState | None) -> CollectFertilizerTask | None:
    if animal.fertilizer_ready:
        status, worker, hour = _carried(prior)
        return CollectFertilizerTask(COLLECT_FERTILIZER, status, False, worker, hour, True)
    if prior is None or prior.status == COMPLETED:
        return None
    return CollectFertilizerTask(
        COLLECT_FERTILIZER,
        COMPLETED,
        False,
        prior.assigned_worker,
        prior.planned_hour,
        False,
    )


def _fertilize_task(world: WorldState, crop: CropState, prior: TaskState | None) -> FertilizeTask | None:
    active = crop.fertilizer_days_left > 0 or crop.fertilized_today
    if active:
        if prior is None or prior.status == COMPLETED:
            return None
        return FertilizeTask(
            FERTILIZE,
            COMPLETED,
            False,
            prior.assigned_worker,
            prior.planned_hour,
            crop.fertilized_today,
            crop.fertilizer_days_left,
        )
    if not should_fertilize(crop, world):
        return None
    status, worker, hour = _carried(prior)
    return FertilizeTask(FERTILIZE, status, False, worker, hour, False, 0)


def _carried(prior: TaskState | None) -> tuple[str, str | None, int | None]:
    if prior is not None and prior.status == SCHEDULED:
        return SCHEDULED, prior.assigned_worker, prior.planned_hour
    return PENDING, None, None


def _oneshot_fertilizer_gain(crop: CropState, spec: dict, days_remaining: int) -> int:
    cap = int(spec["max_yield"])
    first_yield_day = int(spec["first_yield_day"])
    window_start = (int(spec["max_yield_day"]) + 1) // 2
    window_end = int(spec["max_yield_day"])

    # Extra units that cannot mature before the season ends are not
    # sellable, so they are not a reason to publish FERTILIZE.
    if crop.age_days + max(0, days_remaining - 1) < first_yield_day:
        return 0

    def run(fertilize: bool) -> int:
        units = crop.yield_units
        for ahead in range(max(0, days_remaining)):
            if units >= cap:
                break
            age = crop.age_days + ahead
            if not window_start <= age <= window_end:
                continue
            if ahead == 0 and crop.watered_today:
                continue
            bonus = 2 if fertilize and ahead <= 2 else 1
            units = min(cap, units + bonus)
        return units

    return run(True) - run(False)


def _ongoing_fertilizer_gain(crop: CropState, spec: dict, days_remaining: int) -> int:
    """Additional saleable units from fertilizing this observation.

    The engine's ``max_yield`` is both the number of production events and
    the storage cap.  Simulate the remaining production nights through the
    season, clamping the current stock at that cap.  Fertilizer applied now
    is active for the next three daily refreshes (today plus the next two
    engine refresh windows), so the comparison cannot overstate late-season
    or already-capped crops.
    """

    cap = int(spec["max_yield"])
    first = int(spec["first_yield_day"])
    interval = max(1, int(spec["interval"]))

    def run(fertilize: bool) -> int:
        produced = 0
        sitting = min(cap, max(0, crop.yield_units))
        for ahead in range(1, max(0, days_remaining)):
            future_age = crop.age_days + ahead
            days_since_first = future_age - first
            if days_since_first < 0 or days_since_first % interval != 0:
                continue
            if days_since_first // interval + 1 > cap:
                break
            bonus = 2 if fertilize and ahead <= 3 else 1
            before = sitting
            sitting = min(cap, sitting + bonus)
            produced += sitting - before
        return produced

    return run(True) - run(False)


def _feed_task(animal: AnimalState, prior: TaskState | None) -> FeedTask:
    status, worker, hour = _done_or_open(animal.fed_today, prior)
    # A newly placed animal carries a conditional first-feed commitment from
    # its production chain.  Once the animal exists, keep that commitment as
    # mandatory across replans even though the engine still reports
    # consecutive_unfed == 0.  It disappears after the real observation shows
    # fed_today=True.
    conditional = (
        isinstance(prior, TileTask)
        and prior.depends_on == PLACE_ANIMAL
    ) or (
        prior is not None
        and prior.status == SCHEDULED
        and not animal.fed_today
        and prior.assigned_worker is not None
    )
    return FeedTask(
        FEED,
        status,
        animal.must_feed or conditional,
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


def _harvesting_one_shot(crop: CropState) -> bool:
    spec = ENGINE_CROPS[crop.crop]
    return not spec["ongoing"] and crop.must_harvest


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
    # The engine gives one base unit on every scheduled production. Watering
    # matters here only when fertilizer is active, because then the same
    # production event is doubled to two units.
    without_water = min(cap, crop.yield_units + 1)
    with_water = min(cap, crop.yield_units + (2 if fertilized else 1))
    return with_water - without_water


def _turns_until_weed(hour: int, weed_countdown_days: int) -> int:
    """Hours from this hour until the refresh that turns the plant into weed, if it is never watered again."""

    return (TURNS_PER_DAY - hour) + weed_countdown_days * TURNS_PER_DAY
