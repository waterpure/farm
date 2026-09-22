"""Conservative same-day labor and cash checks for Route 14 investments."""

from __future__ import annotations

from typing import Any

from .route14_economy import (
    ANIMAL_FIRST_YIELD_DAYS,
    ANIMAL_STRUCTURE,
    CROP_FIRST_YIELD_DAY,
    CROP_ONGOING,
    CROP_YIELD,
    DEFAULT_MAX_HIRES,
    SEASON_DAYS,
    crop_ready_to_harvest,
    empty_slot_plan,
    hire_cost_total,
    place_plan_by_value,
    rank_crop_lines,
    wheat_unit_price,
)
from .task_router import route_mandatory_tasks


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _farm(observation: dict[str, Any]) -> dict[str, Any]:
    farms = list(observation.get("farms") or [])
    player = _int(observation.get("player"))
    return farms[player] if 0 <= player < len(farms) else {}


def _worker_starts(observation: dict[str, Any], hires: int) -> list[list[int]]:
    farm = _farm(observation)
    starts = [list(farm.get("farmer") or [4, 4]), *[list(pos) for pos in farm.get("hands") or []]]
    doors = ((4, 4), (5, 4), (4, 5), (5, 5))
    already = max(_int(farm.get("hires_today")), len(starts) - 1)
    for _ in range(max(0, min(DEFAULT_MAX_HIRES, hires) - already)):
        occupancy = {door: sum(tuple(pos) == door for pos in starts) for door in doors}
        chosen = min(doors, key=lambda door: (occupancy[door], doors.index(door)))
        starts.append(list(chosen))
    return starts


def _today_tasks(observation: dict[str, Any]) -> list[dict[str, Any]]:
    farm = _farm(observation)
    day = _int(observation.get("day"))
    tasks: list[dict[str, Any]] = []
    for y, row in enumerate(farm.get("tiles") or []):
        for x, tile in enumerate(row):
            if not isinstance(tile, dict):
                continue
            target = [x, y]
            crop = str(tile.get("crop") or "")
            if crop:
                age = day - _int(tile.get("planted_day"))
                if crop_ready_to_harvest(crop, age=age, yield_units=_int(tile.get("yield_units"))):
                    tasks.append({"target": target, "kind": "HARVEST", "priority": 95})
                    if not CROP_ONGOING.get(crop, False):
                        # Charge enough turns for the return trip and unload.
                        distance = min(abs(x - 4) + abs(y - 4), abs(x - 5) + abs(y - 4), abs(x - 4) + abs(y - 5), abs(x - 5) + abs(y - 5))
                        tasks.extend({"target": target, "kind": "HAUL_RESERVE", "priority": 95} for _ in range(distance + 1))
                elif not tile.get("watered_today", False):
                    tasks.append({"target": target, "kind": "WATER", "priority": 100})
            if tile.get("animal"):
                if not tile.get("fed_today", False):
                    tasks.append({"target": target, "kind": "FEED", "priority": 105})
                if not tile.get("cared_today", False):
                    tasks.append({"target": target, "kind": "CARE", "priority": 100})
                if _int(tile.get("yield_units")) > 0:
                    tasks.append({"target": target, "kind": "HARVEST", "priority": 95})
                    distance = min(abs(x - 4) + abs(y - 4), abs(x - 5) + abs(y - 4), abs(x - 4) + abs(y - 5), abs(x - 5) + abs(y - 5))
                    tasks.extend({"target": target, "kind": "HAUL_RESERVE", "priority": 95} for _ in range(distance + 1))
            if tile.get("kind") == "WEED":
                tasks.append({"target": target, "kind": "DIG", "priority": 85})
    return tasks


def proposed_crop_slots(
    observation: dict[str, Any], cash: int, animal_budget: int = 0
) -> list[tuple[int, int, str]]:
    farm = _farm(observation)
    empties = [(x, y) for y, row in enumerate(farm.get("tiles") or []) for x, tile in enumerate(row) if tile is None]
    plan = empty_slot_plan(observation, len(empties), cash=cash, max_new_animals=animal_budget)
    placed = place_plan_by_value(observation, empties, plan)
    return [(x, y, name) for (x, y), (kind, name) in placed if kind == "crop"]


def animal_candidate_position(observation: dict[str, Any], animal: str) -> tuple[int, int] | None:
    farm = _farm(observation)
    structure = ANIMAL_STRUCTURE[animal]
    vacant: list[tuple[int, int]] = []
    blank: list[tuple[int, int]] = []
    for y, row in enumerate(farm.get("tiles") or []):
        for x, tile in enumerate(row):
            if isinstance(tile, dict) and tile.get("kind") == structure and not tile.get("animal"):
                vacant.append((x, y))
            elif tile is None:
                blank.append((x, y))
    choices = vacant or blank
    if not choices:
        return None
    return min(choices, key=lambda pos: (min(abs(pos[0] - dx) + abs(pos[1] - dy) for dx, dy in ((4, 4), (5, 4), (4, 5), (5, 5))), pos[1], pos[0]))


def new_land_crop_slots(observation: dict[str, Any], count: int) -> list[tuple[int, int, str]]:
    """The first affordable crop slots in the next engine unlock quadrant."""

    farm = _farm(observation)
    tiles = list(farm.get("tiles") or [])
    if not tiles:
        return []
    extra = max(0, len(farm.get("unlocked_quadrants") or ["NW"]) - 1)
    if extra >= 3:
        return []
    quadrant = ("NE", "SW", "SE")[extra]
    half = len(tiles) // 2
    locked = [
        (x, y)
        for y, row in enumerate(tiles)
        for x, tile in enumerate(row)
        if tile == "LOCKED" and ("N" if y < half else "S") + ("W" if x < half else "E") == quadrant
    ]
    crops = rank_crop_lines(observation)
    if not crops:
        return []
    crop = crops[0][1]
    locked.sort(key=lambda pos: (min(abs(pos[0] - dx) + abs(pos[1] - dy) for dx, dy in ((4, 4), (5, 4), (4, 5), (5, 5))), pos[1], pos[0]))
    return [(x, y, crop) for x, y in locked[:count]]


def day_schedule(
    observation: dict[str, Any],
    hires: int,
    plants: list[tuple[int, int, str]] | None = None,
    extra_tasks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Witness all known care plus PLANT/WATER at each proposed new tile.

    Hires placed this hour start next hour. A witnessed route is a capacity gate,
    not a prediction of the agent's exact task assignments or future market.
    """

    farm = dict(_farm(observation))
    farm["farmer"] = list(farm.get("farmer") or [4, 4])
    farm["hands"] = _worker_starts(observation, hires)[1:]
    farms = list(observation.get("farms") or [])
    player = _int(observation.get("player"))
    farms[player] = farm
    shadow = dict(observation, farms=farms, hour=_int(observation.get("hour")) + 1)
    tasks = _today_tasks(observation)
    for x, y, _crop in plants or []:
        tasks.extend((
            {"target": [x, y], "kind": "PLANT", "priority": 70},
            {"target": [x, y], "kind": "WATER", "priority": 70},
        ))
    tasks.extend(extra_tasks or [])
    witness = route_mandatory_tasks(shadow, tasks)
    return witness


def future_care_schedule(
    observation: dict[str, Any],
    hires: int,
    plants: list[tuple[int, int, str]] | None = None,
    animal: tuple[int, int, str] | None = None,
    days_ahead: int = 1,
) -> dict[str, Any]:
    """Check a future care/harvest day with a fresh crew at the shed."""

    farm = dict(_farm(observation))
    tiles = [[dict(tile) if isinstance(tile, dict) else tile for tile in row] for row in farm.get("tiles") or []]
    future_day = _int(observation.get("day")) + days_ahead
    for row in tiles:
        for index, tile in enumerate(row):
            if not isinstance(tile, dict):
                continue
            if tile.get("crop"):
                crop = str(tile["crop"])
                age = future_day - _int(tile.get("planted_day"))
                first = CROP_FIRST_YIELD_DAY.get(crop, 30)
                if age > first and not CROP_ONGOING.get(crop, False):
                    row[index] = None
                    continue
                tile["watered_today"] = False
                if age >= first:
                    tile["yield_units"] = max(1, _int(tile.get("yield_units")), CROP_YIELD.get(crop, 1))
            if tile.get("animal"):
                tile["fed_today"] = False
                tile["cared_today"] = False
                animal_name = str(tile["animal"])
                if future_day - _int(tile.get("placed_day")) >= ANIMAL_FIRST_YIELD_DAYS.get(animal_name, 30):
                    tile["yield_units"] = max(1, _int(tile.get("yield_units")))
    for x, y, crop in plants or []:
        age = days_ahead
        if age <= CROP_FIRST_YIELD_DAY.get(crop, 30) or CROP_ONGOING.get(crop, False):
            tiles[y][x] = {"kind": "PLANT", "crop": crop, "planted_day": _int(observation.get("day")), "yield_units": CROP_YIELD.get(crop, 1) if age >= CROP_FIRST_YIELD_DAY.get(crop, 30) else 0, "watered_today": False}
    if animal:
        x, y, name = animal
        tiles[y][x] = {"kind": ANIMAL_STRUCTURE[name], "animal": name, "placed_day": _int(observation.get("day")), "fed_today": False, "cared_today": False, "yield_units": int(days_ahead >= ANIMAL_FIRST_YIELD_DAYS[name])}
    farm.update(tiles=tiles, farmer=[4, 4], hands=[], hires_today=0)
    farms = list(observation.get("farms") or [])
    farms[_int(observation.get("player"))] = farm
    shadow = dict(observation, farms=farms, day=future_day, hour=0)
    return day_schedule(shadow, hires)


def future_care_hires(
    observation: dict[str, Any],
    plants: list[tuple[int, int, str]] | None = None,
    animal: tuple[int, int, str] | None = None,
) -> int | None:
    checkpoints = {1}
    for _, _, crop in plants or []:
        checkpoints.add(CROP_FIRST_YIELD_DAY.get(crop, 30))
    if animal:
        checkpoints.add(ANIMAL_FIRST_YIELD_DAYS[animal[2]])
    day = _int(observation.get("day"))
    for row in _farm(observation).get("tiles") or []:
        for tile in row:
            if isinstance(tile, dict) and tile.get("crop"):
                crop = str(tile["crop"])
                gap = CROP_FIRST_YIELD_DAY.get(crop, 30) - (day - _int(tile.get("planted_day")))
                if gap > 0:
                    checkpoints.add(gap)
            if isinstance(tile, dict) and tile.get("animal"):
                animal_name = str(tile["animal"])
                gap = ANIMAL_FIRST_YIELD_DAYS.get(animal_name, 30) - (day - _int(tile.get("placed_day")))
                if gap > 0:
                    checkpoints.add(gap)
    required = 0
    for ahead in sorted(offset for offset in checkpoints if 1 <= offset < SEASON_DAYS - day):
        if not future_care_schedule(observation, DEFAULT_MAX_HIRES, plants, animal, ahead)["all_mandatory_tasks_witnessed"]:
            return None
        low, high = 0, DEFAULT_MAX_HIRES
        while low < high:
            middle = (low + high) // 2
            if future_care_schedule(observation, middle, plants, animal, ahead)["all_mandatory_tasks_witnessed"]:
                high = middle
            else:
                low = middle + 1
        needed = low
        required = max(required, needed)
    return required


def affordable_care_reserve(
    observation: dict[str, Any],
    hires: int,
    plants: list[tuple[int, int, str]] | None = None,
    extra_animals: int = 0,
) -> int:
    """Cash to keep known production alive until its first possible sale.

    Revenue is not credited before sale. The reserve is a liquidity gate; the
    full-season feed bill remains in the animal ROI calculation.
    """

    farm = _farm(observation)
    day = _int(observation.get("day"))
    first = SEASON_DAYS - day
    heads = extra_animals
    live_crops = 0
    for row in farm.get("tiles") or []:
        for tile in row:
            if not isinstance(tile, dict):
                continue
            crop = str(tile.get("crop") or "")
            if crop:
                live_crops += 1
                age = day - _int(tile.get("planted_day"))
                first = min(first, max(1, CROP_FIRST_YIELD_DAY.get(crop, 30) - age))
            animal = str(tile.get("animal") or "")
            if animal:
                heads += 1
                age = day - _int(tile.get("placed_day"))
                first = min(first, max(1, ANIMAL_FIRST_YIELD_DAYS.get(animal, 30) - age))
    private = dict(observation.get("private") or {})
    waiting = dict(private.get("shed") or {})
    for inventory in private.get("inventories") or []:
        for name, qty in dict(inventory or {}).items():
            if name in ANIMAL_FIRST_YIELD_DAYS:
                waiting[name] = _int(waiting.get(name)) + _int(qty)
    for name in ANIMAL_FIRST_YIELD_DAYS:
        amount = _int(waiting.get(name))
        if amount:
            heads += amount
            first = min(first, ANIMAL_FIRST_YIELD_DAYS[name])
    if plants:
        live_crops += len(plants)
        first = min(first, min(CROP_FIRST_YIELD_DAY.get(crop, 30) for _, _, crop in plants))
    days = max(1, min(first, SEASON_DAYS - day))
    if live_crops + heads == 0:
        return 0
    return days * (hire_cost_total(hires) + heads * wheat_unit_price(observation))
