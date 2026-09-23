"""Per-tile and per-worker state parsed from one observation.

The route planner reads these objects. It does not scan raw tiles itself.
Countdowns follow the official engine: a plant or animal that has already
missed one day dies or runs tonight; goods in the shed do not rot.

A need has one of three statuses. PENDING means it still has to be done and
nobody has it. SCHEDULED means a worker and an hour are written down. COMPLETED
means a later observation shows the engine actually did it. Writing the plan
does not complete the job, and emitting the command does not either.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from kaggle_environments.envs.kaggriculture.kaggriculture import (
    ANIMALS as ENGINE_ANIMALS,
    CROPS as ENGINE_CROPS,
    MAX_SHOP_INSTANCES,
    SHOPS,
    TOWN_CENTER_PRODUCTS,
    market_price,
)


SEASON_DAYS = 30
TURNS_PER_DAY = 24
SHED_CAPACITY = 100
SHOP_UNLOCK_INTERVAL = 3
SHOP_CONSUME_INTERVAL = 4
CENTER_CONSUME_INTERVAL = 24
MARKET_I0 = 10000
ANIMAL_NAMES = tuple(ENGINE_ANIMALS)
PENDING = "PENDING"
SCHEDULED = "SCHEDULED"
COMPLETED = "COMPLETED"
DIES_TONIGHT = "DIES_TONIGHT"
ESCAPES_TONIGHT = "ESCAPES_TONIGHT"
RIPE = "RIPE"
POOL_KINDS = ("WATER", "FEED", "HARVEST")
FALLOW = "FALLOW"
ESCAPED = "ESCAPED"
TILE_CHANGES = ("PLANT", "BUILD_COOP", "BUILD_PASTURE", "PLACE_ANIMAL", "DIG")


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def shed_doors(board_size: int) -> tuple[tuple[int, int], ...]:
    half = max(1, board_size) // 2
    return ((half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half))


def door_distance(position: tuple[int, int], board_size: int = 10) -> int:
    return min(abs(position[0] - door[0]) + abs(position[1] - door[1]) for door in shed_doors(board_size))


@dataclass
class LandState:
    """One owned tile with nothing growing and no animal on it."""

    position: tuple[int, int]
    door_distance: int
    empty: bool
    weed: bool
    needs_dig: bool
    plantable: bool
    can_build_coop: bool
    can_build_pasture: bool
    fallow: bool = False
    reserved: bool = False
    status: str | None = None
    assigned_worker: str | None = None
    planned_hour: int | None = None
    planned_change: str | None = None


@dataclass
class BuildingState:
    """One coop or pasture, occupied or still empty."""

    structure: str
    position: tuple[int, int]
    door_distance: int
    empty: bool
    animal: str | None
    accepted_animals: tuple[str, ...]
    can_dig: bool
    escaped: bool = False
    reserved: bool = False
    status: str | None = None
    assigned_worker: str | None = None
    planned_animal: str | None = None
    planned_hour: int | None = None


@dataclass
class CropState:
    """One planted tile, including when its next goods can be sold."""

    crop: str
    position: tuple[int, int]
    door_distance: int
    age_days: int
    harvest_countdown_days: int
    rot_countdown_steps: int | None
    weed_countdown_days: int
    yield_units: int
    mature: bool
    must_water: bool
    watered_today: bool
    water_count_today: int
    fertilized_today: bool
    fertilize_count_today: int
    fertilizer_days_left: int
    expected_harvest_units: int
    remaining_harvests: int
    next_price: int
    next_revenue: int
    must_harvest: bool = False
    harvest_completed: bool = False
    status: str | None = None
    assigned_worker: str | None = None
    water_status: str | None = None
    harvest_status: str | None = None
    water_worker: str | None = None
    harvest_worker: str | None = None
    planned_water_hour: int | None = None
    planned_harvest_hour: int | None = None
    planned_drop_hour: int | None = None
    planned_sell_hour: int | None = None


@dataclass
class AnimalState:
    """One animal standing in a coop or pasture."""

    animal: str
    product: str
    position: tuple[int, int]
    door_distance: int
    days_until_next_production: int
    yield_units: int
    max_held: int
    fed_today: bool
    feed_count_today: int
    cared_today: bool
    care_count_today: int
    consecutive_unfed: int
    escape_countdown_days: int
    must_feed: bool
    care_bonus: int
    wheat_per_day: int
    fertilizer_ready: bool
    remaining_productions: int
    next_yield_units: int
    next_price: int
    next_revenue: int
    must_harvest: bool = False
    harvest_completed: bool = False
    status: str | None = None
    assigned_worker: str | None = None
    feed_status: str | None = None
    harvest_status: str | None = None
    feed_worker: str | None = None
    harvest_worker: str | None = None
    planned_feed_hour: int | None = None
    planned_harvest_hour: int | None = None
    planned_drop_hour: int | None = None
    planned_sell_hour: int | None = None


@dataclass
class FieldTaskState:
    """One open job taken from a crop or animal. The pool is the list of these."""

    kind: str
    position: tuple[int, int]
    status: str
    subject: str
    assigned_worker: str | None = None
    planned_hour: int | None = None
    planned_drop_hour: int | None = None
    planned_sell_hour: int | None = None
    units: int = 0
    harvest_completed: bool = False
    reason: str = ""
    urgency: int = 9
    goods: str = ""

    @property
    def line(self) -> str:
        """One row: where, what is standing there, and the job."""

        return f"({self.position[0]}, {self.position[1]}) {self.subject} {self.kind}"


@dataclass
class TaskAssignment:
    """A route step that covers one need. It schedules the need; it does not finish it."""

    kind: str
    position: tuple[int, int]
    assigned_worker: str
    planned_hour: int
    planned_drop_hour: int | None = None
    planned_sell_hour: int | None = None
    subject: str = ""


@dataclass
class InventoryState:
    """Shed, seeds, and goods still in workers' hands."""

    shed: dict[str, int]
    capacity: int
    used: int
    free_space: int
    seeds: dict[str, int]
    fertilizer: int
    unplaced_animals: dict[str, int]
    wheat_reserved: int
    wheat_sellable: int
    carried: dict[str, int]
    expected_intake: int
    overflow_risk: bool


@dataclass
class WorkerStep:
    hour: int
    target: tuple[int, int]
    operation: tuple[Any, ...]


@dataclass
class WorkerState:
    """Farmer or hand. The route is empty until the day planner fills it."""

    actor: int
    role: str
    position: tuple[int, int]
    carrying: dict[str, int]
    current_target: tuple[int, int] | None = None
    next_target: tuple[int, int] | None = None
    route: list[WorkerStep] = field(default_factory=list)
    completed: list[WorkerStep] = field(default_factory=list)
    remaining: list[WorkerStep] = field(default_factory=list)
    planned: bool = False

    @property
    def coord(self) -> tuple[int, int]:
        """Where this worker is standing. A day route starts from here at hour 1."""

        return self.position


@dataclass
class MarketState:
    """The town's shared market, not the goods this farm owns.

    A sale takes a unit out of the shed and, when the price is above 1, puts
    that unit into town_inventory. Town shops and the town center take units
    back out. The price follows town_inventory. price_forecast is the rest of
    today if no farm sells and only the town buys.
    """

    town_inventory: dict[str, int]
    prices: dict[str, int]
    shops: list[str]
    shop_counts: dict[str, int]
    hours_until_shop_consume: int
    hours_until_center_consume: int
    hours_until_next_shop: int | None
    price_forecast: dict[str, dict[int, int]]


@dataclass
class FarmState:
    money: int
    lands: list[LandState]
    buildings: list[BuildingState]
    crops: list[CropState]
    animals: list[AnimalState]
    workers: list[WorkerState]
    inventory: InventoryState | None
    tasks: list[FieldTaskState] = field(default_factory=list)
    task_pool: list[FieldTaskState] = field(default_factory=list)
    task_grid: Any = None
    fallow_positions: list[tuple[int, int]] = field(default_factory=list)
    escaped_positions: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class WorldState:
    money: int
    day: int
    hour: int
    step: int
    hours_remaining: int
    days_remaining: int
    market: MarketState
    farm: FarmState
    opponent: FarmState | None


def _price(market: MarketState, item: str) -> int:
    if item in market.prices:
        return _int(market.prices[item])
    return market_price(item, market.town_inventory.get(item, MARKET_I0))


def _board_size(tiles: list[Any]) -> int:
    return len(tiles) if tiles else 10


def _hours_until_interval(step: int, interval: int) -> int:
    remainder = step % interval
    return 0 if remainder == 0 else interval - remainder


def _consume_town(stock: dict[str, int], shops: list[str], step: int) -> None:
    """Apply one hour of town buying. The sale price of that hour is already fixed."""

    if step % SHOP_CONSUME_INTERVAL == 0:
        for shop_name in shops:
            products = list(SHOPS.get(shop_name, []))
            multiplier = 2 if len(products) == 1 else 1
            for item in products:
                if item in stock:
                    stock[item] -= multiplier
    if step % CENTER_CONSUME_INTERVAL == 0:
        for item in TOWN_CENTER_PRODUCTS:
            if item in stock:
                stock[item] -= 1


def _price_forecast(
    stock: dict[str, int],
    prices: dict[str, int],
    shops: list[str],
    step: int,
    hour: int,
) -> dict[str, dict[int, int]]:
    """Price of each good at each remaining hour if farms sell nothing."""

    running = dict(stock)
    forecast = {item: {} for item in running}
    for ahead in range(TURNS_PER_DAY - hour):
        future_hour = hour + ahead
        for item, amount in running.items():
            if ahead == 0 and item in prices:
                forecast[item][future_hour] = _int(prices[item])
            else:
                forecast[item][future_hour] = market_price(item, amount)
        _consume_town(running, shops, step + ahead)
    return forecast


def _hours_until_next_shop(day: int, hour: int, shop_count: int) -> int | None:
    if shop_count >= MAX_SHOP_INSTANCES:
        return None
    if day % SHOP_UNLOCK_INTERVAL == 0:
        days_ahead = SHOP_UNLOCK_INTERVAL
    else:
        days_ahead = SHOP_UNLOCK_INTERVAL - (day % SHOP_UNLOCK_INTERVAL)
    return days_ahead * TURNS_PER_DAY - hour


def _rot_countdown(tile: dict[str, Any], step: int) -> int | None:
    lifespan = _int(tile.get("max_lifespan_step"), -1)
    if lifespan < 0:
        return None
    return max(0, lifespan - step)


def _weed_countdown(watered: bool, dry_days: int) -> int:
    if watered:
        return 2
    if dry_days >= 1:
        return 0
    return 1


def _escape_countdown(fed: bool, missed_days: int) -> int:
    if fed:
        return 2
    if missed_days >= 1:
        return 0
    return 1


def _remaining_harvests(crop: str, age: int, units: int) -> int:
    spec = ENGINE_CROPS[crop]
    if not spec["ongoing"]:
        return 1
    if age < _int(spec["first_yield_day"]):
        completed = 0
    else:
        interval = max(1, _int(spec["interval"]))
        completed = min(_int(spec["max_yield"]), (age - _int(spec["first_yield_day"])) // interval + 1)
    return max(0, _int(spec["max_yield"]) - completed) + (1 if units > 0 else 0)


def _days_until_next_production(day: int, placed_day: int, first: int, interval: int) -> int:
    interval = max(1, interval)
    for ahead in range(1, SEASON_DAYS + 1):
        since = day + ahead - placed_day - first
        if since >= 0 and since % interval == 0:
            return ahead
    return SEASON_DAYS


def _remaining_productions(day: int, placed_day: int, first: int, interval: int) -> int:
    interval = max(1, interval)
    count = 0
    morning = placed_day + first
    while morning < SEASON_DAYS:
        if morning > day:
            count += 1
        morning += interval
    return count


def _crop_state(tile: dict[str, Any], position: tuple[int, int], day: int, step: int, board: int, market: MarketState) -> CropState:
    crop = str(tile.get("crop") or "")
    spec = ENGINE_CROPS[crop]
    age = day - _int(tile.get("planted_day"))
    units = _int(tile.get("yield_units"))
    watered = bool(tile.get("watered_today", False))
    dry_days = _int(tile.get("consecutive_unwatered"))
    mature = units > 0 and age >= _int(spec["first_yield_day"])
    dies_tonight = not watered and dry_days >= 1
    must_water = dies_tonight and not (mature and not spec["ongoing"])
    fertilized_until = _int(tile.get("fertilized_until_day"), -1)
    fertilized_today = fertilized_until == day + 2
    fertilizer_days = max(0, fertilized_until - day + 1) if fertilized_until >= day else 0
    price = _price(market, crop)
    return CropState(
        crop=crop,
        position=position,
        door_distance=door_distance(position, board),
        age_days=age,
        harvest_countdown_days=max(0, _int(spec["first_yield_day"]) - age),
        rot_countdown_steps=_rot_countdown(tile, step),
        weed_countdown_days=_weed_countdown(watered, dry_days),
        yield_units=units,
        mature=mature,
        must_water=must_water,
        watered_today=watered,
        water_count_today=1 if watered else 0,
        fertilized_today=fertilized_today,
        fertilize_count_today=1 if fertilized_today else 0,
        fertilizer_days_left=fertilizer_days,
        expected_harvest_units=units,
        remaining_harvests=_remaining_harvests(crop, age, units),
        next_price=price,
        next_revenue=units * price,
        must_harvest=mature and units > 0,
    )


def _animal_state(tile: dict[str, Any], position: tuple[int, int], day: int, board: int, market: MarketState) -> AnimalState:
    animal = str(tile.get("animal") or "")
    spec = ENGINE_ANIMALS[animal]
    product = str(spec["product"])
    units = _int(tile.get("yield_units"))
    fed = bool(tile.get("fed_today", False))
    cared = bool(tile.get("cared_today", False))
    missed = _int(tile.get("consecutive_unfed"))
    bonus = _int(tile.get("pending_care_bonus"))
    max_held = _int(spec["max_held"])
    price = _price(market, product)
    next_units = units if units > 0 else min(max_held, 1 + bonus)
    return AnimalState(
        animal=animal,
        product=product,
        position=position,
        door_distance=door_distance(position, board),
        days_until_next_production=_days_until_next_production(day, _int(tile.get("placed_day")), _int(spec["first_yield_day"]), _int(spec["interval"])),
        yield_units=units,
        max_held=max_held,
        fed_today=fed,
        feed_count_today=1 if fed else 0,
        cared_today=cared,
        care_count_today=1 if cared else 0,
        consecutive_unfed=missed,
        escape_countdown_days=_escape_countdown(fed, missed),
        must_feed=not fed and missed >= 1,
        care_bonus=bonus,
        wheat_per_day=1,
        fertilizer_ready=bool(tile.get("fertilizer_available", False)),
        remaining_productions=_remaining_productions(day, _int(tile.get("placed_day")), _int(spec["first_yield_day"]), _int(spec["interval"])),
        next_yield_units=next_units,
        next_price=price,
        next_revenue=next_units * price,
        must_harvest=units > 0,
    )


def _accepted(structure: str) -> tuple[str, ...]:
    return tuple(name for name, spec in ENGINE_ANIMALS.items() if spec["structure"] == structure)


def _parse_tiles(tiles: list[Any], day: int, step: int, market: MarketState) -> tuple[list[LandState], list[BuildingState], list[CropState], list[AnimalState]]:
    board = _board_size(tiles)
    lands: list[LandState] = []
    buildings: list[BuildingState] = []
    crops: list[CropState] = []
    animals: list[AnimalState] = []
    for y, row in enumerate(tiles):
        for x, tile in enumerate(row):
            position = (x, y)
            distance = door_distance(position, board)
            if tile is None:
                lands.append(LandState(position, distance, True, False, False, True, True, True))
                continue
            if not isinstance(tile, dict):
                continue
            kind = str(tile.get("kind") or "")
            if kind == "WEED":
                lands.append(LandState(position, distance, False, True, True, False, False, False))
                continue
            if tile.get("crop"):
                crops.append(_crop_state(tile, position, day, step, board, market))
                continue
            if kind in {"COOP", "PASTURE"}:
                animal = str(tile.get("animal") or "") or None
                buildings.append(
                    BuildingState(
                        structure=kind,
                        position=position,
                        door_distance=distance,
                        empty=animal is None,
                        animal=animal,
                        accepted_animals=_accepted(kind),
                        can_dig=animal is None,
                    )
                )
            if animal_name := str(tile.get("animal") or ""):
                if animal_name in ENGINE_ANIMALS:
                    animals.append(_animal_state(tile, position, day, board, market))
    return lands, buildings, crops, animals


def _carrying(inventories: list[Any]) -> list[dict[str, int]]:
    carried: list[dict[str, int]] = []
    for inventory in inventories:
        carried.append({str(item): amount for item, amount in dict(inventory or {}).items() if _int(amount) > 0})
    return carried


def _workers(farm: dict[str, Any], carried: list[dict[str, int]]) -> list[WorkerState]:
    farmer = farm.get("farmer") or [4, 4]
    positions = [(_int(farmer[0]), _int(farmer[1])), *[(_int(pos[0]), _int(pos[1])) for pos in farm.get("hands") or []]]
    workers: list[WorkerState] = []
    for actor, position in enumerate(positions):
        holding = carried[actor] if actor < len(carried) else {}
        role = "farmer" if actor == 0 else "hand"
        workers.append(WorkerState(actor=actor, role=role, position=position, carrying=dict(holding)))
    return workers


def _inventory(
    private: dict[str, Any] | None,
    animals: list[AnimalState],
    workers: list[WorkerState],
    crops: list[CropState],
) -> InventoryState:
    shed = {str(item): _int(amount) for item, amount in dict((private or {}).get("shed") or {}).items() if _int(amount) > 0}
    seeds = {str(item): _int(amount) for item, amount in dict((private or {}).get("seeds") or {}).items() if _int(amount) > 0}
    carried: dict[str, int] = {}
    for worker in workers:
        for item, amount in worker.carrying.items():
            carried[item] = carried.get(item, 0) + amount
    wheat_in_hands = carried.get("WHEAT", 0)
    must_feed = sum(1 for animal in animals if animal.must_feed)
    wheat_reserved = max(0, must_feed - wheat_in_hands)
    wheat_on_shed = shed.get("WHEAT", 0)
    expected = sum(crop.yield_units for crop in crops if crop.mature) + sum(animal.yield_units for animal in animals if animal.yield_units > 0)
    used = sum(shed.values())
    unplaced = {name: shed.get(name, 0) + carried.get(name, 0) for name in ANIMAL_NAMES}
    unplaced = {name: amount for name, amount in unplaced.items() if amount > 0}
    return InventoryState(
        shed=shed,
        capacity=SHED_CAPACITY,
        used=used,
        free_space=max(0, SHED_CAPACITY - used),
        seeds=seeds,
        fertilizer=shed.get("FERTILIZER", 0) + carried.get("FERTILIZER", 0),
        unplaced_animals=unplaced,
        wheat_reserved=min(wheat_on_shed, wheat_reserved),
        wheat_sellable=max(0, wheat_on_shed - wheat_reserved),
        carried=carried,
        expected_intake=expected,
        overflow_risk=used + sum(carried.values()) + expected > SHED_CAPACITY,
    )


def worker_name(actor: int) -> str:
    """Farmer is actor 0. The first hired hand is Hand1."""

    return "Farmer" if actor == 0 else f"Hand{actor}"


def _occupants(
    farm: FarmState, position: tuple[int, int]
) -> tuple[CropState | None, AnimalState | None, LandState | None]:
    crop = next((item for item in farm.crops if item.position == position), None)
    animal = next((item for item in farm.animals if item.position == position), None)
    land = next((item for item in farm.lands if item.position == position), None)
    return crop, animal, land


def _live_tasks(farm: FarmState) -> list[FieldTaskState]:
    tasks: list[FieldTaskState] = []
    for crop in farm.crops:
        if crop.must_water:
            tasks.append(FieldTaskState("WATER", crop.position, PENDING, crop.crop, reason=DIES_TONIGHT, urgency=1))
        if crop.mature and crop.yield_units > 0:
            tasks.append(FieldTaskState("HARVEST", crop.position, PENDING, crop.crop, units=crop.yield_units, goods=crop.crop, reason=RIPE, urgency=2))
    for animal in farm.animals:
        if animal.must_feed:
            tasks.append(FieldTaskState("FEED", animal.position, PENDING, animal.animal, reason=ESCAPES_TONIGHT, urgency=0))
        if animal.yield_units > 0:
            tasks.append(FieldTaskState("HARVEST", animal.position, PENDING, animal.animal, units=animal.yield_units, goods=animal.product, reason=RIPE, urgency=2))
    return tasks


def _building_at(farm: FarmState, position: tuple[int, int]) -> BuildingState | None:
    return next((item for item in farm.buildings if item.position == position), None)


def _need_confirmed(farm: FarmState, task: FieldTaskState) -> bool:
    """True only when this observation shows the action already happened."""

    if task.status == COMPLETED:
        return True
    crop, animal, land = _occupants(farm, task.position)
    building = _building_at(farm, task.position)
    if task.kind == "PLANT":
        return crop is not None and (not task.subject or crop.crop == task.subject)
    if task.kind == "BUILD_COOP":
        return building is not None and building.structure == "COOP"
    if task.kind == "BUILD_PASTURE":
        return building is not None and building.structure == "PASTURE"
    if task.kind == "PLACE_ANIMAL":
        return animal is not None and (not task.subject or animal.animal == task.subject)
    if task.kind == "DIG":
        return bool(task.assigned_worker) and crop is None and animal is None and building is None and land is not None and land.empty and not land.weed
    if task.kind == "WATER":
        return crop is not None and crop.watered_today
    if task.kind == "FEED":
        return animal is not None and animal.fed_today
    if task.kind == "HARVEST" and task.units > 0:
        if crop is not None:
            return crop.yield_units <= 0
        if animal is not None:
            return animal.yield_units <= 0
        if land is not None and land.empty and not land.weed:
            return True
    return False


def _need_open(farm: FarmState, task: FieldTaskState) -> bool:
    crop, animal, land = _occupants(farm, task.position)
    building = _building_at(farm, task.position)
    if task.kind == "PLANT":
        return crop is None and building is None and land is not None and land.empty and land.plantable
    if task.kind == "BUILD_COOP":
        return crop is None and building is None and land is not None and land.empty and land.can_build_coop
    if task.kind == "BUILD_PASTURE":
        return crop is None and building is None and land is not None and land.empty and land.can_build_pasture
    if task.kind == "PLACE_ANIMAL":
        return animal is None and building is not None and building.empty and (not task.subject or task.subject in building.accepted_animals)
    if task.kind == "DIG":
        if animal is not None:
            return False
        if land is not None and land.weed:
            return True
        if crop is not None:
            return True
        return building is not None and building.empty
    if task.kind == "WATER":
        return crop is not None and crop.must_water
    if task.kind == "FEED":
        return animal is not None and animal.must_feed
    if task.kind == "HARVEST":
        if crop is not None and crop.mature and crop.yield_units > 0:
            return True
        return animal is not None and animal.yield_units > 0
    return False


def _mirror_tasks(farm: FarmState) -> None:
    """Copy each need onto its land, building, crop, or animal."""

    for land in farm.lands:
        land.fallow = False
        land.reserved = False
        land.status = None
        land.assigned_worker = None
        land.planned_hour = None
        land.planned_change = None
    for building in farm.buildings:
        building.escaped = False
        building.reserved = False
        building.status = None
        building.assigned_worker = None
        building.planned_animal = None
        building.planned_hour = None
    for crop in farm.crops:
        crop.must_harvest = False
        crop.harvest_completed = False
        crop.status = None
        crop.assigned_worker = None
        crop.water_status = None
        crop.harvest_status = None
        crop.water_worker = None
        crop.harvest_worker = None
        crop.planned_water_hour = None
        crop.planned_harvest_hour = None
        crop.planned_drop_hour = None
        crop.planned_sell_hour = None
    for animal in farm.animals:
        animal.must_harvest = False
        animal.harvest_completed = False
        animal.status = None
        animal.assigned_worker = None
        animal.feed_status = None
        animal.harvest_status = None
        animal.feed_worker = None
        animal.harvest_worker = None
        animal.planned_feed_hour = None
        animal.planned_harvest_hour = None
        animal.planned_drop_hour = None
        animal.planned_sell_hour = None
    for task in farm.tasks:
        crop, animal, land = _occupants(farm, task.position)
        building = _building_at(farm, task.position)
        if task.kind in TILE_CHANGES and task.status != COMPLETED and land is not None and task.kind != "PLACE_ANIMAL":
            land.reserved = task.status in {PENDING, SCHEDULED}
            land.status = task.status
            land.assigned_worker = task.assigned_worker
            land.planned_hour = task.planned_hour
            land.planned_change = task.subject or task.kind
        elif task.kind == "PLACE_ANIMAL" and building is not None and task.status != COMPLETED:
            building.reserved = task.status in {PENDING, SCHEDULED}
            building.status = task.status
            building.assigned_worker = task.assigned_worker
            building.planned_hour = task.planned_hour
            building.planned_animal = task.subject or None
        elif task.kind == "WATER" and crop is not None:
            crop.water_status = task.status
            crop.water_worker = task.assigned_worker
            crop.planned_water_hour = task.planned_hour
        elif task.kind == "FEED" and animal is not None:
            animal.feed_status = task.status
            animal.feed_worker = task.assigned_worker
            animal.planned_feed_hour = task.planned_hour
        elif task.kind == "HARVEST" and crop is not None:
            crop.harvest_status = task.status
            crop.harvest_worker = task.assigned_worker
            crop.harvest_completed = task.harvest_completed
            crop.must_harvest = task.status != COMPLETED
            crop.planned_harvest_hour = task.planned_hour
            crop.planned_drop_hour = task.planned_drop_hour
            crop.planned_sell_hour = task.planned_sell_hour
        elif task.kind == "HARVEST" and animal is not None:
            animal.harvest_status = task.status
            animal.harvest_worker = task.assigned_worker
            animal.harvest_completed = task.harvest_completed
            animal.must_harvest = task.status != COMPLETED
            animal.planned_harvest_hour = task.planned_hour
            animal.planned_drop_hour = task.planned_drop_hour
            animal.planned_sell_hour = task.planned_sell_hour
    for crop in farm.crops:
        water = next((task for task in farm.tasks if task.position == crop.position and task.kind == "WATER"), None)
        harvest = next((task for task in farm.tasks if task.position == crop.position and task.kind == "HARVEST"), None)
        # A harvest-only tile shows the harvest. A water-only tile shows the watering.
        # While a tile still needs water, that is the status a reader sees first.
        if water is not None and water.status != COMPLETED:
            crop.status = water.status
            crop.assigned_worker = water.assigned_worker
        elif harvest is not None:
            crop.status = harvest.status
            crop.assigned_worker = harvest.assigned_worker
        elif water is not None:
            crop.status = water.status
            crop.assigned_worker = water.assigned_worker
    for animal in farm.animals:
        feed = next((task for task in farm.tasks if task.position == animal.position and task.kind == "FEED"), None)
        harvest = next((task for task in farm.tasks if task.position == animal.position and task.kind == "HARVEST"), None)
        if feed is not None and feed.status != COMPLETED:
            animal.status = feed.status
            animal.assigned_worker = feed.assigned_worker
        elif harvest is not None:
            animal.status = harvest.status
            animal.assigned_worker = harvest.assigned_worker
        elif feed is not None:
            animal.status = feed.status
            animal.assigned_worker = feed.assigned_worker


def _remember_tile_history(
    farm: FarmState,
    fallow: list[tuple[int, int]] | None,
    animals: list[tuple[int, int]] | None,
    escaped: list[tuple[int, int]] | None,
) -> None:
    """Harvest leaves bare ground. An animal that leaves leaves an empty shed."""

    crops = {crop.position for crop in farm.crops}
    standing = {animal.position for animal in farm.animals}
    buildings = {building.position: building for building in farm.buildings}
    lands = {land.position: land for land in farm.lands}
    bare = {position for position in (fallow or []) if position not in crops and position not in buildings}
    for task in farm.tasks:
        if task.kind == "HARVEST" and task.status == COMPLETED:
            land = lands.get(task.position)
            if land is not None and land.empty and not land.weed:
                bare.add(task.position)
    bare = {position for position in bare if (land := lands.get(position)) is not None and land.empty and not land.weed}
    gone = set(escaped or [])
    for position in animals or []:
        building = buildings.get(position)
        if building is not None and building.empty and position not in standing:
            gone.add(position)
    gone = {position for position in gone if (building := buildings.get(position)) is not None and building.empty and position not in standing}
    for land in farm.lands:
        if land.position not in bare:
            continue
        land.fallow = True
        if land.status is None:
            land.status = FALLOW
    for building in farm.buildings:
        if building.position not in gone:
            continue
        building.escaped = True
        if building.status is None:
            building.status = ESCAPED
    farm.fallow_positions = sorted(bare)
    farm.escaped_positions = sorted(gone)


def settle_tasks(
    farm: FarmState,
    assignments: list[TaskAssignment],
    previous: list[FieldTaskState] | None,
    fallow: list[tuple[int, int]] | None = None,
    animals: list[tuple[int, int]] | None = None,
    escaped: list[tuple[int, int]] | None = None,
) -> list[FieldTaskState]:
    """Mark needs scheduled from the route, and completed only from the board.

    A harvested one-shot tile is gone, so the finished harvest stays on the
    task list from the previous hour. A tile that turned to weed is not a
    finished harvest.
    """

    merged: dict[tuple[str, tuple[int, int]], FieldTaskState] = {}
    for task in previous or []:
        merged[(task.kind, task.position)] = replace(task)
    for task in _live_tasks(farm):
        key = (task.kind, task.position)
        current = merged.get(key)
        if current is None:
            merged[key] = task
        else:
            if task.reason:
                current.reason = task.reason
                current.urgency = task.urgency
            current.subject = task.subject or current.subject
            if task.goods:
                current.goods = task.goods
            if current.units <= 0 < task.units:
                current.units = task.units
    assigned: dict[tuple[str, tuple[int, int]], TaskAssignment] = {}
    for item in assignments:
        assigned.setdefault((item.kind, item.position), item)
    for key, item in assigned.items():
        if key[0] in TILE_CHANGES and key not in merged:
            merged[key] = FieldTaskState(item.kind, item.position, PENDING, item.subject)
    result: list[FieldTaskState] = []
    for key, task in merged.items():
        if _need_confirmed(farm, task):
            task.status = COMPLETED
        elif not _need_open(farm, task):
            continue
        elif key in assigned:
            item = assigned[key]
            task.assigned_worker = item.assigned_worker
            task.planned_hour = item.planned_hour
            task.planned_drop_hour = item.planned_drop_hour
            task.planned_sell_hour = item.planned_sell_hour
            if item.subject:
                task.subject = item.subject
            task.status = SCHEDULED
        else:
            task.assigned_worker = None
            task.planned_hour = None
            task.planned_drop_hour = None
            task.planned_sell_hour = None
            task.status = PENDING
        task.harvest_completed = task.kind == "HARVEST" and task.status == COMPLETED
        result.append(task)
    result.sort(key=lambda task: (task.position[1], task.position[0], task.kind))
    farm.tasks = result
    farm.task_pool = sorted(
        (task for task in result if task.kind in POOL_KINDS and task.status != COMPLETED),
        key=lambda task: (task.position[1], task.position[0], task.kind),
    )
    _mirror_tasks(farm)
    _remember_tile_history(farm, fallow, animals, escaped)
    return result


def _farm_state(
    farm: dict[str, Any],
    private: dict[str, Any] | None,
    day: int,
    step: int,
    market: MarketState,
    *,
    known_inventory: bool,
) -> FarmState:
    tiles = list(farm.get("tiles") or [])
    lands, buildings, crops, animals = _parse_tiles(tiles, day, step, market)
    carried = _carrying(list((private or {}).get("inventories") or [])) if known_inventory else []
    workers = _workers(farm, carried)
    inventory = _inventory(private, animals, workers, crops) if known_inventory else None
    state = FarmState(
        money=_int(farm.get("money")),
        lands=lands,
        buildings=buildings,
        crops=crops,
        animals=animals,
        workers=workers,
        inventory=inventory,
    )
    settle_tasks(state, [], None)
    return state


def parse_world(observation: dict[str, Any]) -> WorldState:
    """Build the world, both farms' public tiles, and our private shed."""

    day = _int(observation.get("day"))
    hour = _int(observation.get("hour"))
    step = _int(observation.get("step"), day * TURNS_PER_DAY + hour)
    market_raw = observation.get("market") or {}
    town_inventory = {str(item): _int(amount, MARKET_I0) for item, amount in dict(market_raw.get("inventory") or {}).items()}
    prices = {str(item): _int(amount) for item, amount in dict(market_raw.get("prices") or {}).items()}
    shops = [str(shop) for shop in list((observation.get("town") or {}).get("unlocked_shops") or [])]
    shop_counts: dict[str, int] = {}
    for shop in shops:
        shop_counts[shop] = shop_counts.get(shop, 0) + 1
    market = MarketState(
        town_inventory=town_inventory,
        prices=prices,
        shops=shops,
        shop_counts=shop_counts,
        hours_until_shop_consume=_hours_until_interval(step, SHOP_CONSUME_INTERVAL),
        hours_until_center_consume=_hours_until_interval(step, CENTER_CONSUME_INTERVAL),
        hours_until_next_shop=_hours_until_next_shop(day, hour, len(shops)),
        price_forecast=_price_forecast(town_inventory, prices, shops, step, hour),
    )
    farms = list(observation.get("farms") or [])
    player = _int(observation.get("player"))
    own = farms[player] if 0 <= player < len(farms) and isinstance(farms[player], dict) else {}
    private = observation.get("private") if isinstance(observation.get("private"), dict) else {}
    farm = _farm_state(own, private, day, step, market, known_inventory=True)
    opponent = None
    other = 1 - player
    if 0 <= other < len(farms) and other != player and isinstance(farms[other], dict):
        opponent = _farm_state(farms[other], None, day, step, market, known_inventory=False)
    return WorldState(
        money=farm.money,
        day=day,
        hour=hour,
        step=step,
        hours_remaining=max(0, SEASON_DAYS * TURNS_PER_DAY - step),
        days_remaining=max(0, SEASON_DAYS - day),
        market=market,
        farm=farm,
        opponent=opponent,
    )
