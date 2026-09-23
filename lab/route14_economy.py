"""Pure rules for the 14th route: money-per-day 70/30 and timed selling.

These functions do not emit game actions.  They turn the current observation
into a crop ranking, a next crop, a land-buy decision, and a sell quantity.
"""

from __future__ import annotations

from functools import lru_cache
from math import ceil
from typing import Any

from .portfolio_model import ANIMAL_OUTPUT, PRODUCTS, shop_demand_per_day

SEASON_DAYS = 30
SEED_COST = {"WHEAT": 10, "CARROT": 20, "TOMATO": 50, "STRAWBERRY": 100, "MELON": 80}
LAND_PRICES = (1000, 2000, 4000)
ANIMAL_COST = {"GOOSE": 300, "COW": 400, "SHEEP": 500}
# Official engine CROPS: harvest is legal at first_yield_day. One-shot yield
# grows by watering between (max_yield_day+1)//2 and max_yield_day. Melon hits
# unfertilized cap 6 on day 10, so occupy/money-per-day uses 10 not 12.
CROP_FIRST_YIELD_DAY = {"WHEAT": 2, "CARROT": 2, "TOMATO": 8, "STRAWBERRY": 10, "MELON": 10}
CROP_MAX_YIELD_DAY = {"WHEAT": 4, "CARROT": 3, "TOMATO": 8, "STRAWBERRY": 10, "MELON": 12}
CROP_MAX_YIELD = {"WHEAT": 6, "CARROT": 4, "TOMATO": 4, "STRAWBERRY": 4, "MELON": 6}
CROP_ONGOING = {"WHEAT": False, "CARROT": False, "TOMATO": True, "STRAWBERRY": True, "MELON": False}
CROP_OCCUPY_DAYS = {"WHEAT": 4, "CARROT": 3, "TOMATO": 11, "STRAWBERRY": 16, "MELON": 10}
CROP_YIELD = {"WHEAT": 4, "CARROT": 3, "TOMATO": 4, "STRAWBERRY": 4, "MELON": 6}
SHOP_CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
SHOP_ANIMAL = {"EGG": "GOOSE", "MILK": "COW", "WOOL": "SHEEP"}
ANIMAL_STRUCTURE = {"GOOSE": "COOP", "COW": "PASTURE", "SHEEP": "PASTURE"}
ANIMAL_FIRST_YIELD_DAYS = {"GOOSE": 4, "COW": 8, "SHEEP": 6}
ANIMAL_YIELD_INTERVAL_DAYS = {"GOOSE": 1, "COW": 2, "SHEEP": 3}
ANIMAL_MAX_HELD = {"GOOSE": 4, "COW": 6, "SHEEP": 6}
JOBS_PER_WORKER = 12
MAX_MARKET_ORDERS = 10
NEW_LAND_TILES = 25
DEFAULT_MAX_HIRES = 12
CASH_BUFFER = 200
HOURS_PER_DAY = 24
SHED_DOORS = ((4, 4), (5, 4), (4, 5), (5, 5))
NEAR_DUMP_STEPS = 5
STABLE_CROPS = ("WHEAT", "CARROT")
HOT_CROP_RATIO = 0.70
MIN_LAND_FILL_SEEDS = 4


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def remaining_days(observation: dict[str, Any]) -> int:
    return max(0, SEASON_DAYS - _int(observation.get("day")))


try:  # The official pricing curve, so our own glut estimate cannot drift from it.
    from kaggle_environments.envs.kaggriculture.kaggriculture import market_price as _raw_market_price
except ImportError:  # pragma: no cover - fall back to spot quotes off-environment
    _market_price = None
else:
    # Pure in (item, inventory) and asked the same questions thousands of times
    # per episode, so caching costs nothing in accuracy.
    _market_price = lru_cache(maxsize=None)(_raw_market_price)


def batch_sale_revenue(item: str, inventory: int, units: int) -> float | None:
    """What `units` actually fetch, priced one unit at a time down the glut curve.

    The engine quotes every unit at the inventory standing before it lands, so a
    big harvest sells into the very prices it is pushing down. None when the
    official curve is unavailable.
    """

    if _market_price is None:
        return None
    if units <= 0:
        return 0.0
    return float(sum(_market_price(item, int(inventory) + offset) for offset in range(units)))


def market_inventory(observation: dict[str, Any], item: str) -> int | None:
    stock = dict((observation.get("market") or {}).get("inventory") or {})
    return _int(stock[item]) if item in stock else None


def own_supply_map(observation: dict[str, Any]) -> dict[str, int]:
    """Units of each product we already owe the market, in one pass over the farm.

    Counts the standing field plus anything sitting unsold, because those hit the
    market first and set the price the next tile will actually get.
    """

    player = _int(observation.get("player"))
    farms = list(observation.get("farms") or [])
    farm = farms[player] if 0 <= player < len(farms) else {}
    units: dict[str, int] = {}
    for row in farm.get("tiles") or []:
        for tile in row:
            if not isinstance(tile, dict):
                continue
            crop = tile.get("crop")
            if crop:
                units[str(crop)] = units.get(str(crop), 0) + CROP_YIELD.get(str(crop), 0)
                continue
            animal = tile.get("animal")
            if animal:
                product = next((item for item, kind in SHOP_ANIMAL.items() if kind == animal), None)
                if product:
                    units[product] = units.get(product, 0) + _int(tile.get("yield_units"))
    private = dict(observation.get("private") or {})
    for name, qty in dict(private.get("shed") or {}).items():
        units[str(name)] = units.get(str(name), 0) + _int(qty)
    for inventory in private.get("inventories") or []:
        for name, qty in dict(inventory or {}).items():
            units[str(name)] = units.get(str(name), 0) + _int(qty)
    return units


def own_supply_units(observation: dict[str, Any], item: str) -> int:
    return own_supply_map(observation).get(item, 0)


def crop_finishes_in_season(crop: str, days_left: int) -> bool:
    """Can a seed sown today still reach a harvest before the season ends?

    Harvest is legal at first_yield_day, so a partial yield still counts. The
    last playable day is SEASON_DAYS - 1, hence the one day held back.
    """

    return CROP_FIRST_YIELD_DAY.get(crop, SEASON_DAYS) <= days_left - 1


def owned_tile_count(observation: dict[str, Any]) -> int:
    player = _int(observation.get("player"))
    farms = list(observation.get("farms") or [])
    farm = farms[player] if 0 <= player < len(farms) else {}
    count = 0
    for row in farm.get("tiles") or []:
        for tile in row:
            if tile != "LOCKED":
                count += 1
    return count


def quota_tile_cap(owned: int) -> int:
    return owned * 7 // 10


def empty_unlocked_count(observation: dict[str, Any]) -> int:
    player = _int(observation.get("player"))
    farms = list(observation.get("farms") or [])
    farm = farms[player] if 0 <= player < len(farms) else {}
    return sum(tile is None for row in farm.get("tiles") or [] for tile in row)


def field_job_count(observation: dict[str, Any]) -> int:
    """Water, harvest, weed, feed, and care jobs already on the farm today."""

    player = _int(observation.get("player"))
    farms = list(observation.get("farms") or [])
    farm = farms[player] if 0 <= player < len(farms) else {}
    day = _int(observation.get("day"))
    jobs = 0
    for row in farm.get("tiles") or []:
        for tile in row:
            if not isinstance(tile, dict):
                continue
            if tile.get("kind") == "WEED":
                jobs += 1
                continue
            crop = tile.get("crop")
            if crop:
                age = day - _int(tile.get("planted_day"))
                if crop_ready_to_harvest(str(crop), age=age, yield_units=_int(tile.get("yield_units"))):
                    jobs += 2
                else:
                    jobs += 1
            if tile.get("animal"):
                jobs += 2
    return jobs


def haul_job_count(observation: dict[str, Any]) -> int:
    """Walking time to carry ripe one-shot crops to the shed."""

    player = _int(observation.get("player"))
    farms = list(observation.get("farms") or [])
    farm = farms[player] if 0 <= player < len(farms) else {}
    day = _int(observation.get("day"))
    jobs = 0
    for y, row in enumerate(farm.get("tiles") or []):
        for x, tile in enumerate(row):
            if not isinstance(tile, dict):
                continue
            crop = str(tile.get("crop") or "")
            if not crop or CROP_ONGOING.get(crop, False):
                continue
            if crop_ready_to_harvest(crop, age=day - _int(tile.get("planted_day")), yield_units=_int(tile.get("yield_units"))):
                jobs += door_distance((x, y))
    return jobs


def livestock_heads(observation: dict[str, Any]) -> int:
    player = _int(observation.get("player"))
    farms = list(observation.get("farms") or [])
    farm = farms[player] if 0 <= player < len(farms) else {}
    private = dict(observation.get("private") or {})
    heads = sum(
        int(isinstance(tile, dict) and bool(tile.get("animal")))
        for row in farm.get("tiles") or []
        for tile in row
    )
    shed = dict(private.get("shed") or {})
    for name in ANIMAL_COST:
        heads += _int(shed.get(name))
    for inventory in private.get("inventories") or []:
        held = dict(inventory or {})
        for name in ANIMAL_COST:
            heads += _int(held.get(name))
    return heads


def unfed_heads(observation: dict[str, Any]) -> int:
    player = _int(observation.get("player"))
    farms = list(observation.get("farms") or [])
    farm = farms[player] if 0 <= player < len(farms) else {}
    return sum(
        int(isinstance(tile, dict) and bool(tile.get("animal")) and not tile.get("fed_today", False))
        for row in farm.get("tiles") or []
        for tile in row
    )


def wheat_on_hand(observation: dict[str, Any]) -> int:
    private = dict(observation.get("private") or {})
    total = _int((private.get("shed") or {}).get("WHEAT"))
    for inventory in private.get("inventories") or []:
        total += _int(dict(inventory or {}).get("WHEAT"))
    return total


def liquidation_day(observation: dict[str, Any]) -> bool:
    """The last playable day, when only cash counts.

    Nothing sown, fed or cared for can pay off any more, and shed stock scores
    nothing. So the whole crew exists to bring in every remaining unit and sell
    it, and any spending is pure loss.
    """

    return _int(observation.get("day")) >= SEASON_DAYS - 1


def wheat_feed_order(observation: dict[str, Any], cash: int, extra_heads: int = 0) -> list[Any] | None:
    """Buy today's feed. Feeding may spend the cash buffer; seeds may not."""

    if liquidation_day(observation):
        return None
    need = max(0, livestock_heads(observation) + extra_heads - wheat_on_hand(observation))
    if need <= 0:
        return None
    prices = dict((observation.get("market") or {}).get("prices") or {})
    price = max(1, _int(prices.get("WHEAT"), 25))
    buy = min(need, max(0, cash // price))
    if buy <= 0:
        return None
    return ["BUY_PRODUCT", "WHEAT", buy]


def dump_push(observation: dict[str, Any]) -> bool:
    """True when ripe crops or unsold goods need a sell crew."""

    if oneshot_harvest_pending(observation) or carrying_sale_goods(observation):
        return True
    player = _int(observation.get("player"))
    farms = list(observation.get("farms") or [])
    farm = farms[player] if 0 <= player < len(farms) else {}
    private = dict(observation.get("private") or {})
    animals = sum(int(isinstance(tile, dict) and bool(tile.get("animal"))) for row in farm.get("tiles") or [] for tile in row)
    shed = dict(private.get("shed") or {})
    for product in PRODUCTS:
        amount = _int(shed.get(product))
        if product == "WHEAT":
            amount = max(0, amount - animals)
        if amount > 0 and product != "FERTILIZER":
            return True
    day = _int(observation.get("day"))
    for row in farm.get("tiles") or []:
        for tile in row:
            if not isinstance(tile, dict):
                continue
            crop = tile.get("crop")
            if crop and crop_ready_to_harvest(
                str(crop), age=day - _int(tile.get("planted_day")), yield_units=_int(tile.get("yield_units"))
            ):
                return True
    return False


def oneshot_harvest_pending(observation: dict[str, Any]) -> bool:
    """True while a one-and-done crop (melon/carrot/wheat) is still sitting ripe."""

    player = _int(observation.get("player"))
    farms = list(observation.get("farms") or [])
    farm = farms[player] if 0 <= player < len(farms) else {}
    day = _int(observation.get("day"))
    for row in farm.get("tiles") or []:
        for tile in row:
            if not isinstance(tile, dict):
                continue
            crop = str(tile.get("crop") or "")
            if not crop or CROP_ONGOING.get(crop, False):
                continue
            if crop_ready_to_harvest(crop, age=day - _int(tile.get("planted_day")), yield_units=_int(tile.get("yield_units"))):
                return True
    return False


def carrying_sale_goods(observation: dict[str, Any]) -> bool:
    private = dict(observation.get("private") or {})
    for inventory in private.get("inventories") or []:
        held = dict(inventory or {})
        for product in PRODUCTS:
            if product != "FERTILIZER" and _int(held.get(product)) > 0:
                return True
    return False


def door_distance(pos: tuple[int, int] | list[int]) -> int:
    x, y = _int(pos[0]), _int(pos[1])
    return min(abs(x - door[0]) + abs(y - door[1]) for door in SHED_DOORS)


def shed_place_target(
    position: tuple[int, int] | list[int],
    used_doors: set[tuple[int, int]] | None = None,
) -> tuple[int, int]:
    """Unload on the door we already stand on; otherwise the nearest free door."""

    here = (_int(position[0]), _int(position[1]))
    if here in SHED_DOORS:
        return here
    used = used_doors or set()
    free = [door for door in SHED_DOORS if door not in used] or list(SHED_DOORS)
    return min(free, key=lambda door: (abs(here[0] - door[0]) + abs(here[1] - door[1]), door))


def ripe_melon_distances(observation: dict[str, Any]) -> list[int]:
    player = _int(observation.get("player"))
    farms = list(observation.get("farms") or [])
    farm = farms[player] if 0 <= player < len(farms) else {}
    day = _int(observation.get("day"))
    distances: list[int] = []
    for y, row in enumerate(farm.get("tiles") or []):
        for x, tile in enumerate(row):
            if not isinstance(tile, dict) or tile.get("crop") != "MELON":
                continue
            if crop_ready_to_harvest("MELON", age=day - _int(tile.get("planted_day")), yield_units=_int(tile.get("yield_units"))):
                distances.append(door_distance((x, y)))
    return distances


def melon_harvest_cutoff(observation: dict[str, Any]) -> int | None:
    """Day 10: finish the near ring before corner tiles. Other days unrestricted."""

    if _int(observation.get("day")) != 10:
        return None
    distances = ripe_melon_distances(observation)
    if not distances:
        return None
    if any(dist <= NEAR_DUMP_STEPS for dist in distances):
        return NEAR_DUMP_STEPS
    return min(distances) + 1


def may_harvest_melon_at(observation: dict[str, Any], pos: tuple[int, int]) -> bool:
    cutoff = melon_harvest_cutoff(observation)
    if cutoff is None:
        return True
    return door_distance(pos) <= cutoff


def planting_allowed(observation: dict[str, Any]) -> bool:
    """Buy seed and replant after the dump starts moving.

    Dawn still skips seed buys while one-shot crops are sitting ripe. After
    hour 0, empty tiles can be replanted; workers who are carrying goods still
    only PLACE. A leftover melon or fruiting berries must not freeze the field.
    """

    if _int(observation.get("hour")) == 0 and oneshot_harvest_pending(observation):
        return False
    return True


def plantable_today(observation: dict[str, Any], workers: int, hour: int | None = None) -> int:
    if hour is None:
        hour = _int(observation.get("hour"))
    hours_left = max(1, HOURS_PER_DAY - hour)
    actions = max(0, workers) * min(JOBS_PER_WORKER, hours_left)
    leftover = max(0, actions - field_job_count(observation))
    return min(empty_unlocked_count(observation), leftover // 2)


def hires_for_field_work(observation: dict[str, Any], extra_plant_jobs: int = 0) -> int:
    """Hands needed so farmer plus hired crew can water, harvest, feed, plant."""

    jobs = field_job_count(observation) + max(0, extra_plant_jobs)
    workers = max(1, int(ceil(jobs / JOBS_PER_WORKER))) if jobs else 1
    return min(max_hires_today(observation), max(0, workers - 1))


def max_hires_today(observation: dict[str, Any]) -> int:
    del observation
    return DEFAULT_MAX_HIRES


def hire_target(observation: dict[str, Any], cash: int | None = None) -> int:
    """Hands for today's keep-alive jobs plus whatever the hub still funds."""

    from .route14_hub import hub_plan

    return hub_plan(observation, cash).hires


def remaining_hires(observation: dict[str, Any], cash: int) -> int:
    """How many more HIRE orders fit today, after people already on the payroll."""

    player = _int(observation.get("player"))
    farms = list(observation.get("farms") or [])
    farm = farms[player] if 0 <= player < len(farms) else {}
    already = max(_int(farm.get("hires_today")), len(list(farm.get("hands") or [])))
    want = max(0, hire_target(observation, cash) - already)
    hired = 0
    spent = 0
    for index in range(want):
        cost = fib_hire_cost(already + index)
        if cash < spent + cost + CASH_BUFFER:
            break
        spent += cost
        hired += 1
    return hired


def hire_cost_total(count: int) -> int:
    return sum(fib_hire_cost(index) for index in range(max(0, count)))


def pack_market_orders(
    *,
    sells: list[list[Any]],
    seed: list[Any] | None = None,
    seeds: list[list[Any]] | None = None,
    animal: list[Any] | None,
    wheat: list[Any] | None,
    land: list[Any] | None,
    hires: int,
    limit: int = MAX_MARKET_ORDERS,
    hire_first: bool = False,
) -> list[list[Any]]:
    """Keep seed/animal/feed/land slots; fill leftover slots with hires.

    A haul day puts HIRE ahead of new seeds so the 10-order cap still
    lets the crew grow across hour 0 and hour 1.
    """

    sells = [order for order in sells if order]
    seed_orders = [order for order in (seeds or []) if order]
    if seed:
        seed_orders = [seed, *seed_orders]
    production = seed_orders + [order for order in (animal,) if order]
    land_order = [land] if land else []
    feed = [wheat] if wheat else []
    rest = production + land_order
    if hire_first:
        hire_keep = min(max(0, hires), max(0, limit - len(sells) - len(feed)))
        leftover = max(0, limit - len(sells) - len(feed) - hire_keep)
        return (sells + feed + [["HIRE"] for _ in range(hire_keep)] + rest[:leftover])[:limit]
    reserved = len(sells) + len(feed) + len(rest)
    hire_keep = min(max(0, hires), max(0, limit - reserved))
    return (sells + feed + rest + [["HIRE"] for _ in range(hire_keep)])[:limit]


def shop_target_units(observation: dict[str, Any], product: str) -> float:
    shops = list((observation.get("town") or {}).get("unlocked_shops") or [])
    return shop_demand_per_day(shops, product) * remaining_days(observation) * 0.70


def _held(observation: dict[str, Any], product: str) -> int:
    private = dict(observation.get("private") or {})
    total = _int((private.get("shed") or {}).get(product))
    for inventory in private.get("inventories") or []:
        total += _int(dict(inventory or {}).get(product))
    return total


def _growing_expected(observation: dict[str, Any], product: str) -> int:
    player = _int(observation.get("player"))
    farms = list(observation.get("farms") or [])
    farm = farms[player] if 0 <= player < len(farms) else {}
    day = _int(observation.get("day"))
    total = 0
    for row in farm.get("tiles") or []:
        for tile in row:
            if not isinstance(tile, dict):
                continue
            if tile.get("crop") == product:
                already = _int(tile.get("yield_units"))
                if product in {"TOMATO", "STRAWBERRY"}:
                    total += max(CROP_YIELD[product], already)
                else:
                    total += max(CROP_YIELD.get(product, 0), already)
            animal = tile.get("animal")
            if animal and ANIMAL_OUTPUT.get(animal) == product:
                total += max(1, _int(tile.get("yield_units")))
    del day
    return total


def coverage_gap(observation: dict[str, Any], product: str) -> float:
    return shop_target_units(observation, product) - _held(observation, product) - _growing_expected(observation, product)


def quota_occupancy(observation: dict[str, Any]) -> tuple[int, set[str]]:
    """Tiles already growing/raising something, and which products they make."""

    player = _int(observation.get("player"))
    farms = list(observation.get("farms") or [])
    farm = farms[player] if 0 <= player < len(farms) else {}
    used = 0
    products: set[str] = set()
    for row in farm.get("tiles") or []:
        for tile in row:
            if not isinstance(tile, dict):
                continue
            if tile.get("kind") == "WEED":
                continue
            crop = tile.get("crop")
            animal = tile.get("animal")
            if crop:
                used += 1
                products.add(str(crop))
            elif animal:
                used += 1
                products.add(ANIMAL_OUTPUT[str(animal)])
            elif tile.get("kind") in {"COOP", "PASTURE"}:
                used += 1
    return used, products


def crop_fits(crop: str, days_left: int) -> bool:
    return days_left >= CROP_OCCUPY_DAYS.get(crop, 99)


def crop_ready_to_harvest(crop: str, *, age: int, yield_units: int) -> bool:
    """True when the engine will accept HARVEST. Sell as soon as it is legal;
    do not wait to water up to the yield cap.
    """

    if yield_units <= 0 or age < CROP_FIRST_YIELD_DAY.get(crop, 99):
        return False
    return True


def _seed_count(observation: dict[str, Any], crop: str) -> int:
    private = dict(observation.get("private") or {})
    return _int((private.get("seeds") or {}).get(crop))


def _cash_on_hand(observation: dict[str, Any]) -> int:
    player = _int(observation.get("player"))
    farms = list(observation.get("farms") or [])
    farm = farms[player] if 0 <= player < len(farms) else {}
    return _int(farm.get("money"))


def crop_tile_count(observation: dict[str, Any], crop: str) -> int:
    player = _int(observation.get("player"))
    farms = list(observation.get("farms") or [])
    farm = farms[player] if 0 <= player < len(farms) else {}
    count = 0
    for row in farm.get("tiles") or []:
        for tile in row:
            if isinstance(tile, dict) and tile.get("crop") == crop:
                count += 1
    return count


def animal_tile_count(observation: dict[str, Any], animal: str) -> int:
    player = _int(observation.get("player"))
    farms = list(observation.get("farms") or [])
    farm = farms[player] if 0 <= player < len(farms) else {}
    count = 0
    for row in farm.get("tiles") or []:
        for tile in row:
            if isinstance(tile, dict) and tile.get("animal") == animal:
                count += 1
    return count


def empty_structure_count(observation: dict[str, Any], structure: str) -> int:
    player = _int(observation.get("player"))
    farms = list(observation.get("farms") or [])
    farm = farms[player] if 0 <= player < len(farms) else {}
    return sum(
        1
        for row in farm.get("tiles") or []
        for tile in row
        if isinstance(tile, dict) and tile.get("kind") == structure and not tile.get("animal")
    )


def line_tile_count(observation: dict[str, Any], kind: str, name: str) -> int:
    if kind == "crop":
        return crop_tile_count(observation, name)
    placed = animal_tile_count(observation, name)
    pasture_animals = ("COW", "SHEEP")
    if name == "GOOSE":
        return placed + empty_structure_count(observation, "COOP")
    if name in pasture_animals:
        empty = empty_structure_count(observation, "PASTURE")
        ranked = [item for item in rank_lines(observation) if item[0] == "animal" and item[1] in pasture_animals]
        if ranked and ranked[0][1] == name:
            return placed + empty
        return placed
    return placed


def money_per_day(
    crop: str,
    prices: dict[str, Any],
    days_left: int,
    inventory: int | None = None,
    own_units: int = 0,
    labor_price: float = 0.0,
) -> float | None:
    """(what one more tile really fetches − seed − the care it costs) / occupy days.

    With market inventory the yield is priced behind everything we already owe
    the market, so a crop we have flooded stops looking profitable. Without it
    this falls back to the spot quote. None if the crop cannot finish.
    """

    occupy = CROP_OCCUPY_DAYS.get(crop, 99)
    if occupy > days_left or occupy <= 0:
        return None
    units = CROP_YIELD[crop]
    revenue = None
    if inventory is not None:
        revenue = batch_sale_revenue(crop, inventory + max(0, own_units), units)
    if revenue is None:
        revenue = float(prices.get(crop) or 0) * units
    labor = crop_labor_turns(crop, days_left) * labor_price
    return (revenue - SEED_COST[crop] - labor) / occupy


def labor_price_per_turn(observation: dict[str, Any]) -> float:
    """What one more worker-turn costs right now.

    The next hand of the day costs fib(n) and works 24 turns, so labour gets
    dearer as the crew grows: care that is nearly free at four hands is not free
    at twelve. Walking is not priced here, only the turns spent acting.
    """

    player = _int(observation.get("player"))
    farms = list(observation.get("farms") or [])
    farm = farms[player] if 0 <= player < len(farms) else {}
    crew = max(_int(farm.get("hires_today")), len(farm.get("hands") or []))
    return fib_hire_cost(crew) / float(HOURS_PER_DAY)


def crop_labor_turns(crop: str, days_left: int) -> int:
    """Turns one tile of this crop asks for: sowing, daily watering, harvesting.

    A crop is tended until its tile is cleared, then it is done.
    """

    occupy = min(CROP_OCCUPY_DAYS.get(crop, 99), max(1, days_left))
    harvests = CROP_MAX_YIELD.get(crop, 1) if CROP_ONGOING.get(crop, False) else 1
    return 1 + occupy + harvests


def animal_yield_events(animal: str, days_left: int) -> int:
    first = ANIMAL_FIRST_YIELD_DAYS[animal]
    if days_left < first:
        return 0
    return 1 + max(0, (days_left - first) // ANIMAL_YIELD_INTERVAL_DAYS[animal])


def animal_labor_turns(animal: str, days_left: int) -> int:
    """Turns one head asks for: building and placing it, then feed and care daily.

    Unlike a crop this never ends: every remaining day costs two turns whether it
    produces that day or not, which is what makes livestock labour-heavy.
    """

    days = max(0, days_left)
    return 3 + 2 * days + animal_yield_events(animal, days_left)


def animal_yield_units(animal: str, days_left: int) -> int:
    """Units if placed today, fed and CARE'd every day, harvested before the cap."""

    first = ANIMAL_FIRST_YIELD_DAYS[animal]
    interval = ANIMAL_YIELD_INTERVAL_DAYS[animal]
    if days_left < first:
        return 0
    events = 1 + max(0, (days_left - first) // interval)
    cap = ANIMAL_MAX_HELD[animal]
    first_units = min(cap, first)
    recurring = min(cap, 1 + interval)
    return first_units + max(0, events - 1) * recurring


def animal_money_per_day(
    animal: str,
    prices: dict[str, Any],
    days_left: int,
    inventory: int | None = None,
    own_units: int = 0,
    labor_price: float = 0.0,
) -> float | None:
    """Same yardstick as a crop tile, with livestock's own costs: the head itself,
    a ration every remaining day, and the daily feed-and-care turns it demands.
    """

    units = animal_yield_units(animal, days_left)
    if units <= 0:
        return None
    product = next(name for name, kind in SHOP_ANIMAL.items() if kind == animal)
    feed = days_left * float(prices.get("WHEAT") or 0)
    revenue = None
    if inventory is not None:
        revenue = batch_sale_revenue(product, inventory + max(0, own_units), units)
    if revenue is None:
        revenue = units * float(prices.get(product) or 0)
    labor = animal_labor_turns(animal, days_left) * labor_price
    return (revenue - ANIMAL_COST[animal] - feed - labor) / days_left


def line_startup_cost(kind: str, name: str) -> int:
    return SEED_COST[name] if kind == "crop" else ANIMAL_COST[name]


def crop_money_per_day(
    observation: dict[str, Any],
    crop: str,
    days_left: int,
    prices: dict[str, Any],
    supply: dict[str, int] | None = None,
) -> float | None:
    """Value of one more tile of this crop, priced behind our own standing supply."""

    if supply is None:
        supply = own_supply_map(observation)
    return money_per_day(
        crop,
        prices,
        days_left,
        market_inventory(observation, crop),
        supply.get(crop, 0),
        labor_price_per_turn(observation),
    )


def animal_line_money_per_day(
    observation: dict[str, Any],
    animal: str,
    days_left: int,
    prices: dict[str, Any],
    supply: dict[str, int] | None = None,
) -> float | None:
    if supply is None:
        supply = own_supply_map(observation)
    product = next(name for name, kind in SHOP_ANIMAL.items() if kind == animal)
    return animal_money_per_day(
        animal,
        prices,
        days_left,
        market_inventory(observation, product),
        supply.get(product, 0),
        labor_price_per_turn(observation),
    )


def _line_value(observation: dict[str, Any], kind: str, name: str) -> float:
    days_left = remaining_days(observation)
    prices = dict((observation.get("market") or {}).get("prices") or {})
    value = (
        crop_money_per_day(observation, name, days_left, prices)
        if kind == "crop"
        else animal_line_money_per_day(observation, name, days_left, prices)
    )
    return 0.0 if value is None else value


def rank_lines(observation: dict[str, Any]) -> list[tuple[str, str, float]]:
    """Crops and animals on one ledger: (revenue − cost) / days used."""

    days_left = remaining_days(observation)
    prices = dict((observation.get("market") or {}).get("prices") or {})
    supply = own_supply_map(observation)
    ranked: list[tuple[float, str, str]] = []
    for crop in SHOP_CROPS:
        value = crop_money_per_day(observation, crop, days_left, prices, supply)
        if value is not None and value > 0:
            ranked.append((value, "crop", crop))
    for animal in ANIMAL_COST:
        value = animal_line_money_per_day(observation, animal, days_left, prices, supply)
        if value is not None and value > 0:
            ranked.append((value, "animal", animal))
    ranked.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return [(kind, name, value) for value, kind, name in ranked]


def rank_crops(observation: dict[str, Any]) -> list[str]:
    return [name for kind, name, _value in rank_crop_lines(observation)]


def rank_crop_lines(observation: dict[str, Any]) -> list[tuple[str, str, float]]:
    days_left = remaining_days(observation)
    prices = dict((observation.get("market") or {}).get("prices") or {})
    supply = own_supply_map(observation)
    ranked: list[tuple[float, str, str]] = []
    for crop in SHOP_CROPS:
        value = crop_money_per_day(observation, crop, days_left, prices, supply)
        if value is not None and value > 0:
            ranked.append((value, "crop", crop))
    ranked.sort(key=lambda item: (item[0], item[2]), reverse=True)
    return [(kind, name, value) for value, kind, name in ranked]


def rank_animal_lines(observation: dict[str, Any]) -> list[tuple[str, str, float]]:
    days_left = remaining_days(observation)
    prices = dict((observation.get("market") or {}).get("prices") or {})
    supply = own_supply_map(observation)
    ranked: list[tuple[float, str, str]] = []
    for animal in ANIMAL_COST:
        value = animal_line_money_per_day(observation, animal, days_left, prices, supply)
        if value is not None and value > 0:
            ranked.append((value, "animal", animal))
    ranked.sort(key=lambda item: (item[0], item[2]), reverse=True)
    return [(kind, name, value) for value, kind, name in ranked]


def hot_crop_names(observation: dict[str, Any]) -> list[str]:
    ranked = rank_crop_lines(observation)
    if not ranked:
        return []
    best = ranked[0][2]
    return [name for _kind, name, value in ranked if value + 1e-9 >= HOT_CROP_RATIO * best]


def stable_crop_name(observation: dict[str, Any], hot: list[str] | None = None) -> str | None:
    hot_set = set(hot if hot is not None else hot_crop_names(observation))
    days_left = remaining_days(observation)
    prices = dict((observation.get("market") or {}).get("prices") or {})
    supply = own_supply_map(observation)
    best_name: str | None = None
    best_value = float("-inf")
    for crop in STABLE_CROPS:
        if crop in hot_set:
            continue
        value = crop_money_per_day(observation, crop, days_left, prices, supply)
        if value is None:
            continue
        if value > best_value:
            best_value = value
            best_name = crop
    if best_name:
        return best_name
    for _kind, name, _value in rank_crop_lines(observation):
        if name not in hot_set:
            return name
    return STABLE_CROPS[0]


def livestock_tile_cap(observation: dict[str, Any]) -> int:
    """At most half the owned tiles. Unused livestock slots fall back to crops."""

    return owned_tile_count(observation) // 2


def placed_line_counts(observation: dict[str, Any]) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for crop in SHOP_CROPS:
        counts[("crop", crop)] = crop_tile_count(observation, crop)
    for animal in ANIMAL_COST:
        counts[("animal", animal)] = animal_tile_count(observation, animal)
    return counts


def wheat_unit_price(observation: dict[str, Any]) -> int:
    prices = dict((observation.get("market") or {}).get("prices") or {})
    return max(1, _int(prices.get("WHEAT"), 25))


def wheat_buy_price(observation: dict[str, Any]) -> int:
    """What one BUY_PRODUCT WHEAT costs on the current book.

    The engine quotes a buy at the inventory after this unit is removed.
    When that book is missing, use the spot quote already on the observation.
    """

    inventory = market_inventory(observation, "WHEAT")
    if _market_price is not None and inventory is not None:
        return max(1, int(_market_price("WHEAT", max(0, int(inventory) - 1))))
    return wheat_unit_price(observation)


def feed_runway_cost(observation: dict[str, Any], heads: int) -> int:
    return max(0, heads) * wheat_unit_price(observation) * max(1, remaining_days(observation))


def crop_half_seed_cost(observation: dict[str, Any]) -> int:
    owned = owned_tile_count(observation)
    animal_cap = livestock_tile_cap(observation)
    crop_tiles = max(0, owned - animal_cap)
    hot = hot_crop_names(observation)
    stable = stable_crop_name(observation, hot)
    hot_tiles = crop_tiles * 7 // 10
    stable_tiles = crop_tiles - hot_tiles
    cost = 0
    if hot:
        cost += hot_tiles * SEED_COST[hot[0]]
    if stable:
        cost += stable_tiles * SEED_COST[stable]
    return cost


def can_afford_animal(observation: dict[str, Any], cash: int, name: str, heads_after: int) -> bool:
    """Pay this animal and its own remaining feed. Do not re-lock crop-half seeds."""

    del heads_after
    return cash >= ANIMAL_COST[name] + feed_runway_cost(observation, 1) + CASH_BUFFER


def choose_next_line(
    observation: dict[str, Any],
    *,
    cash: int | None = None,
    counts: dict[tuple[str, str], int] | None = None,
    max_new_animals: int | None = None,
) -> tuple[str | None, str | None, str]:
    """Next empty tile: livestock up to 50% if we can stock it; leftover land is crop 70/30."""

    if cash is None:
        cash = _cash_on_hand(observation)
    owned = owned_tile_count(observation)
    if owned <= 0:
        return None, None, "idle"
    current = dict(counts) if counts is not None else placed_line_counts(observation)
    placed_animals = sum(placed_line_counts(observation).get(("animal", name), 0) for name in ANIMAL_COST)
    structure_land = empty_structure_count(observation, "PASTURE") + empty_structure_count(observation, "COOP")
    planned_extra = max(0, sum(current.get(("animal", name), 0) for name in ANIMAL_COST) - placed_animals)
    occupied_livestock = placed_animals + structure_land + planned_extra
    animal_cap = livestock_tile_cap(observation)

    def crop_count(name: str) -> int:
        return current.get(("crop", name), 0)

    allow_animal = max_new_animals is None or planned_extra < max_new_animals
    animal_pick: tuple[str, float] | None = None
    if occupied_livestock < animal_cap and allow_animal:
        for _kind, name, value in rank_animal_lines(observation):
            if can_afford_animal(observation, cash, name, occupied_livestock + 1):
                animal_pick = (name, value)
                break
        # Cannot stock another animal; leftover empty tiles stay in the crop half.

    crop_target = owned - min(animal_cap, occupied_livestock)

    days_left = remaining_days(observation)
    hot = hot_crop_names(observation)
    hot_cap = crop_target * 7 // 10
    hot_now = sum(crop_count(name) for name in hot)
    crop_pick: tuple[str, float, str] | None = None
    if hot and hot_now < hot_cap:
        for name in hot:
            if crop_finishes_in_season(name, days_left):
                crop_pick = (name, _line_value(observation, "crop", name), "quota")
                break
    if crop_pick is None:
        stable = stable_crop_name(observation, hot)
        if stable and crop_finishes_in_season(stable, days_left):
            crop_pick = (stable, _line_value(observation, "crop", stable), "rotation")
    if crop_pick is None:
        for _kind, name, value in rank_crop_lines(observation):
            if crop_finishes_in_season(name, days_left):
                crop_pick = (name, value, "quota")
                break

    # One yardstick for both, with each side carrying its own costs: a tile pays
    # for seed plus its watering and harvesting; a head pays for itself, a ration
    # every remaining day, and two care turns a day for the rest of the season.
    if animal_pick is not None and (crop_pick is None or animal_pick[1] >= crop_pick[1]):
        return "animal", animal_pick[0], "livestock"
    if crop_pick is not None:
        return "crop", crop_pick[0], crop_pick[2]
    return None, None, "idle"


def empty_slot_plan(
    observation: dict[str, Any],
    slots: int,
    cash: int | None = None,
    max_new_animals: int | None = None,
) -> list[tuple[str, str]]:
    """Assign empty tiles without uprooting. Livestock up to 50% if stockable; leftover is crop 70/30."""

    current = placed_line_counts(observation)
    remaining = _cash_on_hand(observation) if cash is None else cash
    plan: list[tuple[str, str]] = []
    for _ in range(max(0, slots)):
        kind, name, _role = choose_next_line(
            observation, cash=remaining, counts=current, max_new_animals=max_new_animals
        )
        if kind is None or name is None:
            break
        remaining -= line_startup_cost(kind, name)
        if kind == "animal":
            remaining -= wheat_unit_price(observation)
        plan.append((kind, name))
        current[(kind, name)] = current.get((kind, name), 0) + 1
    return plan


def slot_money_per_day(observation: dict[str, Any], kind: str, name: str) -> float:
    """Current money/day for one planned crop or animal line."""

    days_left = remaining_days(observation)
    prices = dict((observation.get("market") or {}).get("prices") or {})
    if kind == "crop":
        value = crop_money_per_day(observation, name, days_left, prices)
    elif kind == "animal":
        value = animal_line_money_per_day(observation, name, days_left, prices)
    else:
        value = None
    return 0.0 if value is None else value


def place_plan_by_value(
    observation: dict[str, Any],
    tiles: list[tuple[int, int]],
    plan: list[tuple[str, str]],
) -> list[tuple[tuple[int, int], tuple[str, str]]]:
    """Higher money/day goes on tiles closer to the shed doors."""

    ordered_tiles = sorted(tiles, key=lambda pos: (door_distance(pos), pos[1], pos[0]))
    ordered_plan = sorted(
        enumerate(plan),
        key=lambda item: (-slot_money_per_day(observation, item[1][0], item[1][1]), item[0]),
    )
    return [(tile, line) for tile, (_index, line) in zip(ordered_tiles, ordered_plan)]


def choose_next_crop(
    observation: dict[str, Any],
    *,
    cash: int | None = None,
    skip: set[str] | None = None,
) -> tuple[str | None, str]:
    del skip
    kind, name, role = choose_next_line(observation, cash=cash)
    if kind == "crop" and name:
        return name, role
    for line_kind, line_name, _value in rank_lines(observation):
        if line_kind == "crop":
            return line_name, "rotation"
    return None, "idle"


def seed_fill_plan(
    observation: dict[str, Any],
    *,
    workers: int,
    cash: int,
    max_new_animals: int | None = None,
) -> list[list[Any]]:
    """Buy seeds only for empty tiles the 70/30 ledger assigned to crops."""

    room = min(empty_unlocked_count(observation), plantable_today(observation, workers))
    if room <= 0 or not planting_allowed(observation):
        return []
    need: dict[str, int] = {}
    for kind, name in empty_slot_plan(observation, room, cash=cash, max_new_animals=max_new_animals):
        if kind == "crop":
            need[name] = need.get(name, 0) + 1
    orders: list[list[Any]] = []
    remaining_cash = cash
    for crop, target in need.items():
        have = _seed_count(observation, crop)
        buy = min(max(0, target - have), max(0, (remaining_cash - CASH_BUFFER) // SEED_COST[crop]))
        if buy:
            orders.append(["BUY_SEED", crop, buy])
            remaining_cash -= buy * SEED_COST[crop]
    return orders


def choose_next_animal(observation: dict[str, Any], cash: int | None = None) -> str | None:
    """Next animal the hub still funds after keep-alive wages and feed."""

    from .route14_hub import hub_plan

    return hub_plan(observation, cash).animal


def next_land_cost(observation: dict[str, Any]) -> int | None:
    player = _int(observation.get("player"))
    farms = list(observation.get("farms") or [])
    farm = farms[player] if 0 <= player < len(farms) else {}
    extra = max(0, len(farm.get("unlocked_quadrants") or ["NW"]) - 1)
    if extra >= len(LAND_PRICES):
        return None
    return LAND_PRICES[extra]


def line_season_value(
    kind: str,
    name: str,
    tiles: int,
    prices: dict[str, Any],
    days_left: int,
    observation: dict[str, Any] | None = None,
) -> float:
    """Season revenue of a block of tiles, each priced behind the ones before it.

    Filling a whole plot with one line floods that product, so the last tiles are
    worth far less than the first. Given an observation this walks the glut curve
    tile by tile instead of paying the first tile's rate for all of them.
    """

    if tiles <= 0:
        return 0.0
    inventory = None
    supply = 0
    product = name
    if observation is not None:
        product = name if kind == "crop" else next(item for item, kind_name in SHOP_ANIMAL.items() if kind_name == name)
        inventory = market_inventory(observation, product)
        supply = own_supply_units(observation, product)
    if kind == "crop":
        occupy = CROP_OCCUPY_DAYS.get(name, 99)
        if occupy > days_left:
            return 0.0
        cycles = max(1, days_left // occupy)
        total = 0.0
        for index in range(tiles):
            value = money_per_day(name, prices, days_left, inventory, supply + index * CROP_YIELD[name] * cycles)
            if value is None:
                return 0.0
            total += cycles * value * occupy
        return total
    total = 0.0
    for index in range(tiles):
        value = animal_money_per_day(name, prices, days_left, inventory, supply + index * animal_yield_units(name, days_left))
        if value is None:
            return 0.0
        total += value * days_left
    return total


def land_buy_value(observation: dict[str, Any]) -> float:
    """Value of one new 5x5 filled at the current 50/50 then crop 70/30."""

    days_left = remaining_days(observation)
    if days_left < 2:
        return 0.0
    prices = dict((observation.get("market") or {}).get("prices") or {})
    animal_tiles = NEW_LAND_TILES // 2
    crop_tiles = NEW_LAND_TILES - animal_tiles
    value = 0.0
    animals = rank_animal_lines(observation)
    if animal_tiles and animals:
        value += line_season_value("animal", animals[0][1], animal_tiles, prices, days_left, observation)
    hot = hot_crop_names(observation)
    stable = stable_crop_name(observation, hot)
    hot_tiles = crop_tiles * 7 // 10
    if hot:
        value += line_season_value("crop", hot[0], hot_tiles, prices, days_left, observation)
    if stable:
        value += line_season_value("crop", stable, crop_tiles - hot_tiles, prices, days_left, observation)
    return value


def staff_tiles_cost(observation: dict[str, Any], tiles: int) -> int:
    """Cash to plant or stock `tiles` at 50/50 land then crop 70/30."""

    if tiles <= 0:
        return 0
    animal_tiles = tiles // 2
    crop_tiles = tiles - animal_tiles
    cost = 0
    animals = rank_animal_lines(observation)
    if animal_tiles and animals:
        cost += animal_tiles * ANIMAL_COST[animals[0][1]]
        cost += feed_runway_cost(observation, animal_tiles)
    hot = hot_crop_names(observation)
    stable = stable_crop_name(observation, hot)
    hot_tiles = crop_tiles * 7 // 10
    if hot:
        cost += hot_tiles * SEED_COST[hot[0]]
    if stable:
        cost += (crop_tiles - hot_tiles) * SEED_COST[stable]
    return cost


def new_land_staff_cost(observation: dict[str, Any]) -> int:
    """Cash to plant or stock one new 5x5 at the current ledger."""

    return staff_tiles_cost(observation, NEW_LAND_TILES)


def planned_new_animals(observation: dict[str, Any]) -> int:
    del observation
    return NEW_LAND_TILES // 2


def land_expansion_plan(observation: dict[str, Any], cash: int) -> tuple[bool, int]:
    """Whether the hub buys land, and how many hands that farm needs today."""

    from .route14_hub import hub_plan

    plan = hub_plan(observation, cash)
    return plan.buy_land, plan.hires


def should_buy_land(observation: dict[str, Any], *, labor_short: bool, workers: int) -> bool:
    del labor_short, workers
    buy, _hires = land_expansion_plan(observation, _cash_on_hand(observation))
    return buy


def unsold_field_units(observation: dict[str, Any], product: str) -> int:
    player = _int(observation.get("player"))
    farms = list(observation.get("farms") or [])
    farm = farms[player] if 0 <= player < len(farms) else {}
    total = 0
    for row in farm.get("tiles") or []:
        for tile in row:
            if not isinstance(tile, dict):
                continue
            if tile.get("crop") == product:
                total += _int(tile.get("yield_units"))
            animal = tile.get("animal")
            if animal and ANIMAL_OUTPUT.get(animal) == product:
                total += _int(tile.get("yield_units"))
    return total


def carried_units(observation: dict[str, Any], product: str) -> int:
    private = dict(observation.get("private") or {})
    return sum(_int(dict(inventory or {}).get(product)) for inventory in private.get("inventories") or [])


def place_arrival_units(commands: list[list[Any]]) -> dict[str, int]:
    """Goods workers PLACE this hour; engine deposits them before market SELL."""

    arrivals: dict[str, int] = {}
    for command in commands:
        if not command or command[0] != "PLACE" or len(command) < 2:
            continue
        item = command[1]
        if item not in PRODUCTS:
            continue
        qty = _int(command[2], 1) if len(command) >= 3 else 1
        if qty > 0:
            arrivals[item] = arrivals.get(item, 0) + qty
    return arrivals


def sell_quantity(
    product: str,
    *,
    day: int,
    days_left: int,
    price: float,
    window: dict[str, Any] | None,
    available: int,
    wheat_reserve: int = 0,
) -> tuple[int, dict[str, Any] | None]:
    """Sell shed stock plus this hour's PLACE arrivals the hour they are there.

    Wheat kept back for feed is not sold. The caller zeroes that reserve on
    the last day. Price and the old watch window are ignored: goods sell now.
    """

    del day, days_left, price, window
    available = max(0, available - (wheat_reserve if product == "WHEAT" else 0))
    if available <= 0:
        return 0, None
    return available, None


def fib_hire_cost(already: int) -> int:
    a, b = 1, 1
    for _ in range(already):
        a, b = b, a + b
    return a


def plan_fill(observation: dict[str, Any], cash: int) -> tuple[int, list[list[Any]]]:
    """Hire and buy seeds from the leftover pool after keep-alive wages and feed."""

    from .route14_hub import hub_plan

    plan = hub_plan(observation, cash)
    return plan.hires, plan.seeds
