"""Executable 14th-route agent: plant by money/day 70/30 and timed sales.

Independent of the frozen V45 tapes.  Field movement is a reserved-target
greedy walker, same safety idea as the archived portfolio executor, with the
new planting, hiring, land and selling rules.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from .portfolio_agent import SHED_ACCESS, Task, _move_toward
from .route14_economy import (
    ANIMAL_COST,
    ANIMAL_STRUCTURE,
    CROP_MAX_YIELD_DAY,
    NEAR_DUMP_STEPS,
    SEASON_DAYS,
    SEED_COST,
    choose_next_line,
    crop_ready_to_harvest,
    door_distance,
    empty_slot_plan,
    empty_structure_count,
    empty_unlocked_count,
    fib_hire_cost,
    place_plan_by_value,
    may_harvest_melon_at,
    liquidation_day,
    melon_harvest_cutoff,
    next_land_cost,
    oneshot_harvest_pending,
    pack_market_orders,
    planting_allowed,
    remaining_days,
    ripe_melon_distances,
    sell_quantity,
    shed_place_target,
    place_arrival_units,
    unfed_heads,
    wheat_feed_order,
)
from .route14_hub import hub_plan

# Unloading produce at a shed door, and the ripe tile a loaded worker
# may collect on the way there.
HAUL_PLACE_PRIORITY = 97

PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")
ANIMALS = {
    "GOOSE": {"structure": "COOP", "cost": 300},
    "COW": {"structure": "PASTURE", "cost": 400},
    "SHEEP": {"structure": "PASTURE", "cost": 500},
}


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _empty_unlocked(observation: dict[str, Any]) -> list[tuple[int, int]]:
    player = _int(observation.get("player"))
    farm = observation["farms"][player]
    return [(x, y) for y, row in enumerate(farm.get("tiles") or []) for x, tile in enumerate(row) if tile is None]


def _empty_structure(observation: dict[str, Any], structure: str) -> list[tuple[int, int]]:
    player = _int(observation["player"])
    farm = observation["farms"][player]
    return [
        (x, y)
        for y, row in enumerate(farm.get("tiles") or [])
        for x, tile in enumerate(row)
        if isinstance(tile, dict) and tile.get("kind") == structure and not tile.get("animal")
    ]


def _tile_tasks(observation: dict[str, Any]) -> list[Task]:
    player = _int(observation.get("player"))
    farm = observation["farms"][player]
    day = _int(observation.get("day"))
    # On the last day nothing tended can pay off again, and an animal that has
    # not been fed would otherwise never reach its harvest branch below, leaving
    # its milk standing in the field. So today the crew only reaps and delivers.
    last_day = liquidation_day(observation)
    tasks: list[Task] = []
    for y, row in enumerate(farm.get("tiles") or []):
        for x, tile in enumerate(row):
            if not isinstance(tile, dict):
                continue
            target = (x, y)
            if tile.get("animal"):
                if last_day:
                    if _int(tile.get("yield_units")) > 0:
                        tasks.append(Task(target, ["HARVEST"], 96))
                elif not tile.get("fed_today", False):
                    tasks.append(Task(target, ["FEED"], 120 if _int(tile.get("consecutive_unfed")) else 100, requires_item="WHEAT"))
                elif _int(tile.get("yield_units")) > 0:
                    tasks.append(Task(target, ["HARVEST"], 96))
                elif not tile.get("cared_today", False):
                    tasks.append(Task(target, ["CARE"], 89))
                continue
            if tile.get("kind") == "WEED":
                if not last_day:
                    tasks.append(Task(target, ["DIG"], 70))
                continue
            if tile.get("kind") != "PLANT":
                continue
            crop = str(tile.get("crop") or "")
            age = day - _int(tile.get("planted_day"))
            if crop_ready_to_harvest(crop, age=age, yield_units=_int(tile.get("yield_units"))):
                if crop == "MELON":
                    if may_harvest_melon_at(observation, target):
                        tasks.append(Task(target, ["HARVEST"], 99))
                elif _int(observation.get("day")) == 10:
                    tasks.append(Task(target, ["HARVEST"], 70))
                else:
                    tasks.append(Task(target, ["HARVEST"], 95))
            elif not last_day and not tile.get("watered_today", False):
                peak = CROP_MAX_YIELD_DAY.get(crop, 99)
                tasks.append(Task(target, ["WATER"], 88 if age >= peak - 1 else 80))
    return tasks


def _side_snapshot(farm: dict[str, Any], private: dict[str, Any] | None = None) -> dict[str, Any]:
    plants = 0
    unwatered = 0
    weeds = 0
    animals = 0
    empty = 0
    owned = 0
    crops: Counter[str] = Counter()
    livestock: Counter[str] = Counter()
    for row in farm.get("tiles") or []:
        for tile in row:
            if tile == "LOCKED":
                continue
            owned += 1
            if tile is None:
                empty += 1
            elif isinstance(tile, dict) and tile.get("kind") == "WEED":
                weeds += 1
            elif isinstance(tile, dict) and tile.get("kind") == "PLANT":
                plants += 1
                crops[str(tile.get("crop"))] += 1
                if not tile.get("watered_today", False):
                    unwatered += 1
            elif isinstance(tile, dict) and tile.get("animal"):
                animals += 1
                livestock[str(tile.get("animal"))] += 1
    shed = dict((private or {}).get("shed") or {})
    return {
        "money": _int(farm.get("money")),
        "hands": len(farm.get("hands") or []),
        "owned": owned,
        "empty": empty,
        "plants": plants,
        "unwatered": unwatered,
        "weeds": weeds,
        "animals": animals,
        "crops": dict(crops),
        "livestock": dict(livestock),
        "shed_units": sum(_int(value) for key, value in shed.items() if key != "FERTILIZER"),
    }


def _farm_snapshot(observation: dict[str, Any]) -> dict[str, Any]:
    player = _int(observation.get("player"))
    farms = list(observation.get("farms") or [])
    ours = _side_snapshot(farms[player] if 0 <= player < len(farms) else {}, dict(observation.get("private") or {}))
    rival = 1 - player if len(farms) == 2 else player
    theirs = _side_snapshot(farms[rival] if 0 <= rival < len(farms) else {}, None)
    return {
        "day": _int(observation.get("day")),
        "hour": _int(observation.get("hour")),
        "money": ours["money"],
        "hands": ours["hands"],
        "owned": ours["owned"],
        "empty": ours["empty"],
        "plants": ours["plants"],
        "unwatered": ours["unwatered"],
        "weeds": ours["weeds"],
        "animals": ours["animals"],
        "crops": ours["crops"],
        "livestock": ours["livestock"],
        "shed_units": ours["shed_units"],
        "opp_money": theirs["money"],
        "opp_hands": theirs["hands"],
        "opp_owned": theirs["owned"],
        "opp_empty": theirs["empty"],
        "opp_plants": theirs["plants"],
        "opp_animals": theirs["animals"],
        "opp_crops": theirs["crops"],
        "opp_livestock": theirs["livestock"],
    }


def _fmt_counts(counts: dict[str, Any]) -> str:
    if not counts:
        return "无"
    return " ".join(f"{name}{int(value)}" for name, value in sorted(counts.items()) if int(value) > 0)


def format_daily_compare(days: list[dict[str, Any]]) -> str:
    """Human-readable us-vs-opponent farm table for one finished game."""

    lines = [
        "天(收工) | 我方钱 | 对方钱 | 我方地(空/已有) | 对方地(空/已有) | 我方种 | 对方种 | 我方养 | 对方养 | 我方雇 | 对方雇"
    ]
    for row in days:
        lines.append(
            f"{row['day']:2d} | {row['money']:6d} | {row['opp_money']:6d} | "
            f"{row['empty']:2d}/{row['owned']:2d} | {row['opp_empty']:2d}/{row['opp_owned']:2d} | "
            f"{_fmt_counts(row.get('crops') or {})} | {_fmt_counts(row.get('opp_crops') or {})} | "
            f"{_fmt_counts(row.get('livestock') or {})} | {_fmt_counts(row.get('opp_livestock') or {})} | "
            f"{row['hands']:2d} | {row['opp_hands']:2d}"
        )
    return "\n".join(lines)


def make_route14_agent():
    state: dict[str, Any] = {
        "day": None,
        "assignments": {},
        "sell_windows": {},
        "labor_short": False,
    }
    telemetry: dict[str, Any] = {}

    def reset_run() -> None:
        telemetry.clear()
        telemetry.update(
            hires_requested=0,
            land_orders=0,
            seed_orders=0,
            sell_orders=0,
            sell_units_requested=0,
            plant_commands=0,
            water_commands=0,
            harvest_commands=0,
            dig_commands=0,
            feed_commands=0,
            care_commands=0,
            pickup_commands=0,
            place_commands=0,
            build_coop_commands=0,
            build_pasture_commands=0,
            animal_orders=0,
            skipped_plants=0,
            skipped_land=0,
            days=[],
            day10_hours=[],
        )
        state["sell_windows"] = {}
        state["labor_short"] = False

    def planned_sales(
        observation: dict[str, Any], incoming: dict[str, int] | None = None
    ) -> tuple[list[list[Any]], int]:
        farm = observation["farms"][_int(observation["player"])]
        private = dict(observation.get("private") or {})
        day = _int(observation.get("day"))
        days_left = remaining_days(observation)
        prices = dict((observation.get("market") or {}).get("prices") or {})
        shed = dict(private.get("shed") or {})
        arrivals = incoming or {}
        animals = sum(
            int(isinstance(tile, dict) and bool(tile.get("animal")))
            for row in farm.get("tiles") or []
            for tile in row
        )
        orders: list[list[Any]] = []
        for item in PRODUCTS:
            available = _int(shed.get(item)) + _int(arrivals.get(item))
            quantity, updated = sell_quantity(
                item,
                day=day,
                days_left=days_left,
                price=float(prices.get(item) or 0),
                window=state["sell_windows"].get(item),
                available=available,
                # Shed stock scores nothing, so the last day keeps no feed back.
                wheat_reserve=0 if liquidation_day(observation) else animals,
            )
            state["sell_windows"][item] = updated
            if quantity > 0:
                orders.append(["SELL", item, quantity])
        sales = sorted(orders, key=lambda order: (-_int(prices.get(order[1])), str(order[1])))
        sale_cash = sum(_int(order[2]) * _int(prices.get(order[1])) for order in sales)
        return sales, sale_cash

    def market_orders(observation: dict[str, Any], plan: Any, incoming: dict[str, int] | None = None) -> list[list[Any]]:
        player = _int(observation["player"])
        farm = observation["farms"][player]
        private = dict(observation.get("private") or {})
        day = _int(observation.get("day"))
        hour = _int(observation.get("hour"))
        sales, _sale_cash = planned_sales(observation, incoming)
        # A quote is not cash until the market actually executes the sale.
        cash = _int(farm.get("money"))
        wheat_order = wheat_feed_order(observation, cash)
        if wheat_order:
            cash -= _int(wheat_order[2]) * max(1, _int(dict((observation.get("market") or {}).get("prices") or {}).get("WHEAT"), 25))
        seed_orders = list(plan.seeds)
        for order in seed_orders:
            cash -= SEED_COST[str(order[1])] * _int(order[2])
        animal_order = None
        land_order = None
        if hour == 0 and plan.buy_land:
            land_order = ["BUY_LAND"]
            cash -= next_land_cost(observation) or 0
        name = plan.animal
        waiting = any(_int(dict(private.get("shed") or {}).get(animal)) > 0 for animal in ANIMALS)
        waiting = waiting or any(
            _int(dict(inventory or {}).get(animal)) > 0
            for inventory in private.get("inventories") or [] for animal in ANIMALS
        )
        heads = sum(
            int(isinstance(tile, dict) and bool(tile.get("animal")))
            for row in farm.get("tiles") or []
            for tile in row
        )
        has_shed = bool(name) and (
            _empty_structure(observation, ANIMALS[name]["structure"]) or empty_unlocked_count(observation) > 0
        )
        if name and has_shed and not _empty_structure(observation, ANIMALS[name]["structure"]):
            # The animal can wait one hour for a planned matching building,
            # but do not buy it when the slot planner will build another type.
            building_plan = empty_slot_plan(
                observation, empty_unlocked_count(observation), cash=cash, max_new_animals=1
            )
            has_shed = any(
                kind == "animal" and ANIMAL_STRUCTURE[animal] == ANIMAL_STRUCTURE[name]
                for kind, animal in building_plan
            )
        if name and not waiting and has_shed and cash >= ANIMAL_COST[name]:
            extra = wheat_feed_order(observation, cash, extra_heads=1)
            if extra:
                wheat_order = extra
                cash -= _int(extra[2]) * max(1, _int(dict((observation.get("market") or {}).get("prices") or {}).get("WHEAT"), 25))
            animal_order = ["BUY_ANIMAL", name, 1]
            cash -= ANIMAL_COST[name]
        more_feed = wheat_feed_order(observation, cash)
        if more_feed:
            wheat_order = more_feed
        already = max(_int(farm.get("hires_today")), len(farm.get("hands") or []))
        hires = max(0, plan.hires - already)
        affordable = 0
        spent = 0
        for index in range(hires):
            cost = fib_hire_cost(already + index)
            if cash < spent + cost:
                break
            spent += cost
            affordable += 1
        hires = affordable
        result = pack_market_orders(
            sells=sales,
            seeds=seed_orders,
            animal=animal_order,
            wheat=wheat_order,
            land=land_order if hour == 0 else None,
            hires=hires,
            hire_first=True,
        )
        telemetry["hires_requested"] += sum(order[0] == "HIRE" for order in result)
        telemetry["land_orders"] += sum(order[0] == "BUY_LAND" for order in result)
        telemetry["skipped_land"] += int(
            state["labor_short"] and not any(order[0] == "BUY_LAND" for order in result)
        )
        telemetry["seed_orders"] += sum(order[0] == "BUY_SEED" for order in result)
        telemetry["animal_orders"] += sum(order[0] == "BUY_ANIMAL" for order in result)
        telemetry["sell_orders"] += sum(order[0] == "SELL" for order in result)
        telemetry["sell_units_requested"] += sum(_int(order[2]) for order in result if order[0] == "SELL")
        return result

    def extra_tasks(observation: dict[str, Any], tasks: list[Task], positions: list[list[int]]) -> list[Task]:
        private = observation.get("private", {})
        inventories = list(private.get("inventories") or [])
        hungry = unfed_heads(observation)
        carriers: list[tuple[int, str, int]] = []
        for actor, inventory in enumerate(inventories[: len(positions)]):
            held = dict(inventory or {})
            if any(_int(held.get(animal)) > 0 for animal in ANIMALS):
                continue
            for item, amount in held.items():
                if item == "WHEAT" and hungry:
                    continue
                if item in PRODUCTS and _int(amount) > 0:
                    carriers.append((actor, item, _int(amount)))
                    break
        used_doors: set[tuple[int, int]] = set()
        for actor, item, amount in sorted(
            carriers,
            key=lambda row: min(
                abs(positions[row[0]][0] - door[0]) + abs(positions[row[0]][1] - door[1]) for door in SHED_ACCESS
            ),
        ):
            target = shed_place_target(positions[actor], used_doors)
            used_doors.add(target)
            tasks.append(Task(target, ["PLACE", item, amount], HAUL_PLACE_PRIORITY, actor=actor))
        for actor, inventory in enumerate(inventories[: len(positions)]):
            held = dict(inventory or {})
            for animal in ANIMALS:
                if _int(held.get(animal)) > 0:
                    matching = _empty_structure(observation, ANIMALS[animal]["structure"])
                    if matching:
                        target = min(matching, key=lambda item: (abs(positions[actor][0] - item[0]) + abs(positions[actor][1] - item[1]), item))
                        tasks.append(Task(target, ["PLACE", animal], 99, actor=actor))
                    break
        carried_wheat = sum(_int(dict(inv or {}).get("WHEAT")) for inv in inventories)
        feed_to_pick = min(max(0, hungry - carried_wheat), _int(private.get("shed", {}).get("WHEAT")))
        # Use all four doors so several carers can feed in parallel. One
        # worker carrying an entire herd's ration is too slow on a large farm.
        if feed_to_pick:
            doors = sorted(SHED_ACCESS, key=lambda item: (min(abs(pos[0] - item[0]) + abs(pos[1] - item[1]) for pos in positions), item))
            batches = min(len(doors), feed_to_pick)
            herd = observation["farms"][_int(observation["player"])].get("tiles") or []
            feed_priority = 121 if any(
                isinstance(tile, dict) and tile.get("animal") and _int(tile.get("consecutive_unfed"))
                for row in herd for tile in row
            ) else 101
            for index in range(batches):
                qty = feed_to_pick // batches + int(index < feed_to_pick % batches)
                tasks.append(Task(doors[index], ["PICKUP", "WHEAT", qty], feed_priority))
        for animal, info in ANIMALS.items():
            if _int(private.get("shed", {}).get(animal)) > 0 and _empty_structure(observation, info["structure"]):
                target = min(SHED_ACCESS, key=lambda item: (min(abs(pos[0] - item[0]) + abs(pos[1] - item[1]) for pos in positions), item))
                tasks.append(Task(target, ["PICKUP", animal, 1], 98))
                break
        desired_kind, desired_name, _role = choose_next_line(observation)
        seeds_map = dict(private.get("seeds") or {})
        empties = _empty_unlocked(observation)
        plan = state["hour_plan"]
        build_left = 0
        if plan.animal:
            structure = ANIMAL_STRUCTURE[plan.animal]
            if empty_structure_count(observation, structure) <= 0:
                build_left = 1
        slots = empty_slot_plan(observation, len(empties), max_new_animals=build_left)
        # Water a planted crop before starting another seed; otherwise the
        # last planted tiles can die while the crew fills the rest of the map.
        plant_priority = 75
        busy = {tuple(reserved["target"]) for reserved in state["assignments"].values()}
        dump_melon = oneshot_harvest_pending(observation)
        approved_plants = plan.plant_slots
        if planting_allowed(observation) and not dump_melon:
            for target, (kind, name) in place_plan_by_value(observation, empties, slots):
                if target in busy:
                    continue
                if kind == "animal":
                    tasks.append(Task(target, [f"BUILD_{ANIMAL_STRUCTURE[name]}"], 93))
                    busy.add(target)
                    continue
                if approved_plants > 0 and _int(seeds_map.get(name)) > 0:
                    tasks.append(Task(target, ["PLANT", name], plant_priority))
                    busy.add(target)
                    seeds_map[name] = _int(seeds_map.get(name)) - 1
                    approved_plants -= 1
        elif desired_kind == "crop" and _int(observation.get("hour")) == 0:
            telemetry["skipped_plants"] += 1
        return tasks

    def assignment_is_current(observation: dict[str, Any], actor: int, assignment: dict[str, Any]) -> bool:
        player = _int(observation["player"])
        x, y = assignment["target"]
        tile = observation["farms"][player]["tiles"][y][x]
        operation = assignment["operation"]
        op = operation[0]
        if op == "PLANT":
            if not planting_allowed(observation):
                return False
            inventory = list(observation.get("private", {}).get("inventories") or [])
            held = dict(inventory[actor] or {}) if actor < len(inventory) else {}
            if any(item not in {"FERTILIZER", "WHEAT"} and _int(held.get(item)) > 0 for item in PRODUCTS):
                return False
            return tile is None and _int(observation["private"].get("seeds", {}).get(operation[1])) > 0
        if op == "DIG":
            return isinstance(tile, dict) and tile.get("kind") == "WEED"
        if op == "WATER":
            return isinstance(tile, dict) and tile.get("kind") == "PLANT" and not tile.get("watered_today", False)
        if op == "FEED":
            inventory = list(observation.get("private", {}).get("inventories") or [])
            has_wheat = actor < len(inventory) and _int(dict(inventory[actor] or {}).get("WHEAT")) > 0
            return isinstance(tile, dict) and tile.get("animal") and not tile.get("fed_today", False) and has_wheat
        if op == "HARVEST":
            if not isinstance(tile, dict) or _int(tile.get("yield_units")) <= 0:
                return False
            if tile.get("animal"):
                return True
            crop = str(tile.get("crop") or "")
            age = _int(observation.get("day")) - _int(tile.get("planted_day"))
            return crop_ready_to_harvest(crop, age=age, yield_units=_int(tile.get("yield_units")))
        if op == "CARE":
            return isinstance(tile, dict) and tile.get("animal") and tile.get("fed_today", False) and not tile.get("cared_today", False)
        if op == "PICKUP":
            return tuple(assignment["target"]) in SHED_ACCESS and _int(observation["private"].get("shed", {}).get(operation[1])) > 0
        if op == "PLACE" and operation[1] in ANIMALS:
            return isinstance(tile, dict) and tile.get("kind") == ANIMALS[operation[1]]["structure"] and not tile.get("animal")
        if op == "PLACE":
            inventory = list(observation.get("private", {}).get("inventories") or [])
            if actor >= len(inventory) or _int(dict(inventory[actor] or {}).get(operation[1])) <= 0:
                return False
            farm = observation["farms"][player]
            if actor == 0:
                here = tuple(farm["farmer"])
            else:
                hands = list(farm.get("hands") or [])
                if actor - 1 >= len(hands):
                    return False
                here = tuple(hands[actor - 1])
            if here in SHED_ACCESS:
                return here == tuple(assignment["target"])
            return tuple(assignment["target"]) in SHED_ACCESS
        if op in {"BUILD_COOP", "BUILD_PASTURE"}:
            return tile is None
        return False

    def _carrying_goods(observation: dict[str, Any], actor: int) -> bool:
        """True while this worker still has produce to unload at the shed."""

        inventories = list(observation.get("private", {}).get("inventories") or [])
        held = dict(inventories[actor] or {}) if actor < len(inventories) else {}
        skip = {"FERTILIZER", "WHEAT"} if unfed_heads(observation) else {"FERTILIZER"}
        return any(item not in skip and _int(held.get(item)) > 0 for item in PRODUCTS)

    def commands(observation: dict[str, Any]) -> list[list[str]]:
        player = _int(observation["player"])
        farm = observation["farms"][player]
        positions = [list(farm["farmer"]), *[list(pos) for pos in farm.get("hands") or []]]
        active: dict[int, dict[str, Any]] = {}
        for actor, assignment in list(state["assignments"].items()):
            if actor >= len(positions) or not assignment_is_current(observation, actor, assignment):
                state["assignments"].pop(actor, None)
            else:
                active[actor] = assignment
        tasks = extra_tasks(observation, _tile_tasks(observation), positions)
        # A job under a worker's feet costs one turn; walking past it to a job
        # across the farm costs several. So a worker crossing the field is handed
        # back whenever it is standing on work of its own, and the assignment
        # loop below settles every zero-distance job before any distant one.
        underfoot: dict[tuple[int, int], int] = {}
        for task in tasks:
            if task.actor is None:
                underfoot[task.target] = max(underfoot.get(task.target, 0), task.priority)
        for actor, assignment in list(active.items()):
            here = (positions[actor][0], positions[actor][1])
            if here == tuple(assignment["target"]) or here not in underfoot:
                continue
            if underfoot[here] < _int(assignment.get("priority")):
                continue
            if _carrying_goods(observation, actor):
                continue
            state["assignments"].pop(actor, None)
            active.pop(actor, None)
        blocked = {tuple(assignment["target"]) for assignment in active.values()}
        available_actors = [actor for actor in range(len(positions)) if actor not in active]
        unassigned_mandatory = 0
        while available_actors:
            choices: list[tuple[int, int, int, tuple[int, int], Task]] = []
            for task in tasks:
                shed_place = task.operation[0] == "PLACE" and task.target in SHED_ACCESS
                if task.target in blocked and not shed_place:
                    continue
                for actor in available_actors:
                    if task.actor is not None and task.actor != actor:
                        continue
                    if task.requires_item:
                        inventories = list(observation.get("private", {}).get("inventories") or [])
                        if actor >= len(inventories) or _int(dict(inventories[actor] or {}).get(task.requires_item)) <= 0:
                            continue
                    inventories = list(observation.get("private", {}).get("inventories") or [])
                    held = dict(inventories[actor] or {}) if actor < len(inventories) else {}
                    holding_wheat = _int(held.get("WHEAT")) > 0
                    hungry = unfed_heads(observation)
                    carrying = any(item != "FERTILIZER" and _int(held.get(item)) > 0 for item in PRODUCTS)
                    if holding_wheat and hungry:
                        carrying = any(
                            item not in {"FERTILIZER", "WHEAT"} and _int(held.get(item)) > 0 for item in PRODUCTS
                        )
                    distance = abs(positions[actor][0] - task.target[0]) + abs(positions[actor][1] - task.target[1])
                    # A worker already owes a walk home, and the engine puts no
                    # limit on what one pair of hands can hold. Picking up another
                    # ripe tile on the way therefore costs nothing extra while the
                    # detour is no longer than that walk home; without this every
                    # single tile pays for its own round trip. Such a pickup also
                    # outranks the delivery it is riding along with.
                    priority = task.priority
                    if carrying:
                        if task.operation[0] == "HARVEST" and distance <= door_distance(positions[actor]):
                            priority = HAUL_PLACE_PRIORITY + 1
                        elif task.operation[0] != "PLACE":
                            continue
                    if holding_wheat and hungry and task.operation[0] not in {"FEED", "PLACE"}:
                        continue
                    if task.operation[0] == "PLANT" and (carrying or not planting_allowed(observation)):
                        continue
                    choices.append((min(distance, 1), -priority, distance, actor, task.target, task))
            if not choices:
                unassigned_mandatory = sum(1 for task in tasks if task.priority >= 80 and task.target not in blocked)
                break
            *_, actor, _, task = min(choices, key=lambda item: item[:5])
            state["assignments"][actor] = {
                "target": list(task.target),
                "operation": list(task.operation),
                "priority": task.priority,
            }
            active[actor] = state["assignments"][actor]
            if not (task.operation[0] == "PLACE" and task.target in SHED_ACCESS):
                blocked.add(task.target)
            available_actors.remove(actor)
        if _int(observation.get("hour")) != 0:
            state["labor_short"] = unassigned_mandatory > 0
        result: list[list[str]] = []
        inventories = list(observation.get("private", {}).get("inventories") or [])
        hungry = unfed_heads(observation)
        for actor, position in enumerate(positions):
            held = dict(inventories[actor] or {}) if actor < len(inventories) else {}
            sale_item = None
            sale_amount = 0
            if not any(_int(held.get(animal)) > 0 for animal in ANIMALS):
                for item, amount in held.items():
                    if item == "WHEAT" and hungry:
                        continue
                    if item in PRODUCTS and _int(amount) > 0:
                        sale_item, sale_amount = item, _int(amount)
                        break
            if sale_item and tuple(position) in SHED_ACCESS:
                command = ["PLACE", sale_item, sale_amount]
                key = "place_commands"
                if key in telemetry:
                    telemetry[key] += 1
                result.append(command)
                continue
            assignment = active.get(actor)
            if not assignment:
                result.append(["PASS"])
                continue
            command = list(assignment["operation"]) if tuple(position) == tuple(assignment["target"]) else _move_toward(position, tuple(assignment["target"]))
            key = f"{command[0].lower()}_commands"
            if key in telemetry:
                telemetry[key] += 1
            result.append(command)
        return result

    def agent(observation: dict[str, Any], configuration: dict[str, Any] | None = None) -> dict[str, Any]:
        del configuration
        day = _int(observation.get("day"))
        if _int(observation.get("step")) == 0:
            reset_run()
        if state["day"] != day:
            state["day"] = day
            state["assignments"] = {}
        hour = _int(observation.get("hour"))
        farm = observation["farms"][_int(observation["player"])]
        actual_cash = _int(farm.get("money"))
        feed = wheat_feed_order(observation, actual_cash)
        feed_cost = 0 if not feed else _int(feed[2]) * max(1, _int((observation.get("market") or {}).get("prices", {}).get("WHEAT"), 25))
        plan = hub_plan(observation, actual_cash - feed_cost)
        state["hour_plan"] = plan
        if hour == 23:
            days = telemetry.setdefault("days", [])
            if not days or days[-1]["day"] != day:
                snap = _farm_snapshot(observation)
                snap["labor_short"] = bool(state["labor_short"])
                days.append(snap)
        action_commands = commands(observation)
        incoming = place_arrival_units(action_commands)
        market = market_orders(observation, plan, incoming)
        if day == 10:
            farms = list(observation.get("farms") or [])
            player = _int(observation.get("player"))
            rival = 1 - player if len(farms) == 2 else player
            hours = telemetry.setdefault("day10_hours", [])
            if not hours or hours[-1]["hour"] != hour:
                farm = farms[player] if 0 <= player < len(farms) else {}
                positions = [list(farm.get("farmer") or [0, 0]), *[list(pos) for pos in farm.get("hands") or []]]
                inventories = list(observation.get("private", {}).get("inventories") or [])
                walk_off = 0
                on_door_place = 0
                for actor, position in enumerate(positions):
                    held = dict(inventories[actor] or {}) if actor < len(inventories) else {}
                    goods = any(item != "FERTILIZER" and item not in ANIMALS and _int(held.get(item)) > 0 for item in PRODUCTS)
                    if not goods or tuple(position) not in SHED_ACCESS:
                        continue
                    command = action_commands[actor] if actor < len(action_commands) else []
                    if command and command[0] == "PLACE":
                        on_door_place += 1
                    elif command and command[0] in {"NORTH", "SOUTH", "EAST", "WEST"}:
                        walk_off += 1
                ripe = ripe_melon_distances(observation)
                hours.append(
                    {
                        "hour": hour,
                        "money": _int(farms[player].get("money")) if 0 <= player < len(farms) else 0,
                        "opp_money": _int(farms[rival].get("money")) if 0 <= rival < len(farms) else 0,
                        "sell_melon": sum(_int(order[2]) for order in market if order and order[0] == "SELL" and order[1] == "MELON"),
                        "place_melon": _int(incoming.get("MELON")),
                        "on_door_place": on_door_place,
                        "walk_off_door": walk_off,
                        "harvest_cutoff": melon_harvest_cutoff(observation),
                        "ripe_near": sum(1 for dist in ripe if dist <= NEAR_DUMP_STEPS),
                        "ripe_far": sum(1 for dist in ripe if dist > NEAR_DUMP_STEPS),
                    }
                )
        return {
            "farmer": action_commands[0] if action_commands else ["PASS"],
            "hands": action_commands[1:],
            "market": market,
        }

    agent.telemetry = telemetry  # type: ignore[attr-defined]
    agent.label = "route14"  # type: ignore[attr-defined]
    return agent
