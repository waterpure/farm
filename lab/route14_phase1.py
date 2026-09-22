"""Phase-1 day routes: keep the current farm alive, and time each sale.

This does not plant, buy animals, buy land, care, or fertilize. Those are later
phases. A route is a sequence of tile actions for the farmer and each hand.
Harvest, the walk to a shed door, and the drop are separate hours. The sale is
the same hour as the drop, because the engine unloads workers before the market.

The hand count is the one that finishes today's survival work and leaves the
most projected cash. Projected cash uses the official price curve at the sale
hour, after town buying and after both farms' planned drops. Unsold goods are
not cash.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kaggle_environments.envs.kaggriculture.kaggriculture import (
    ANIMALS as ENGINE_ANIMALS,
    SHOPS,
    TOWN_CENTER_PRODUCTS,
    market_price,
)

from .route14_economy import fib_hire_cost
from .route14_state import (
    PENDING,
    SHED_CAPACITY,
    TaskAssignment,
    WorkerState,
    WorkerStep,
    parse_world,
    settle_tasks,
    worker_name,
)


TURNS_PER_DAY = 24
MAX_MARKET_ORDERS = 10
MAX_NEW_HIRES = 12
MARKET_I0 = 10000
# Sell the crop whose sticker price is highest first, matching a same-hour race.
SALE_RANK = {
    "MELON": 8,
    "WOOL": 7,
    "MILK": 6,
    "STRAWBERRY": 5,
    "FERTILIZER": 4,
    "TOMATO": 3,
    "EGG": 2,
    "CARROT": 1,
    "WHEAT": 0,
}


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _farm(observation: dict[str, Any]) -> dict[str, Any]:
    farms = list(observation.get("farms") or [])
    player = _int(observation.get("player"))
    if 0 <= player < len(farms) and isinstance(farms[player], dict):
        return farms[player]
    return {}


def _board_size(farm: dict[str, Any]) -> int:
    tiles = farm.get("tiles") or []
    return len(tiles) if tiles else 10


def shed_doors(board_size: int) -> tuple[tuple[int, int], ...]:
    half = board_size // 2
    return ((half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half))


def _manhattan(start: tuple[int, int], target: tuple[int, int]) -> int:
    return abs(start[0] - target[0]) + abs(start[1] - target[1])


def _move_toward(position: list[int] | tuple[int, int], target: tuple[int, int]) -> list[str]:
    x, y = _int(position[0]), _int(position[1])
    if x < target[0]:
        return ["EAST"]
    if x > target[0]:
        return ["WEST"]
    if y < target[1]:
        return ["SOUTH"]
    if y > target[1]:
        return ["NORTH"]
    return ["PASS"]


def _nearest_door(position: tuple[int, int], doors: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return min(doors, key=lambda door: (_manhattan(position, door), doors.index(door)))


def _spawn_door(positions: list[tuple[int, int]], doors: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    occupancy = {door: 0 for door in doors}
    for position in positions:
        if position in occupancy:
            occupancy[position] += 1
    return min(doors, key=lambda door: (occupancy[door], doors.index(door)))


@dataclass
class Step:
    target: tuple[int, int]
    operation: tuple[Any, ...]
    not_before: int


@dataclass
class Batch:
    """One tile's goods, from the harvest hour through the hour they can sell."""

    product: str
    units: int
    target: tuple[int, int]
    actor: int
    harvest_hour: int
    drop_hour: int
    sell_hour: int


@dataclass
class ActorRoute:
    ready_hour: int
    start: tuple[int, int]
    steps: list[Step] = field(default_factory=list)
    hour: int = 0
    pos: tuple[int, int] = (0, 0)

    def __post_init__(self) -> None:
        self.hour = self.ready_hour
        self.pos = self.start


@dataclass(frozen=True)
class FieldTask:
    kind: str
    target: tuple[int, int]
    product: str = ""
    units: int = 0
    name: str = ""


@dataclass
class DayRoute:
    extra_hires: int
    target_hires: int
    start_hour: int
    wheat_buy: int
    wages: int
    wheat_cost: int
    actors: list[ActorRoute]
    batches: list[Batch]
    survival_total: int
    survival_done: int
    harvests_total: int
    harvests_delivered: int
    revenue: int
    net: int

    @property
    def survival_complete(self) -> bool:
        return self.survival_done >= self.survival_total


def field_tasks(observation: dict[str, Any]) -> list[FieldTask]:
    """Take today's open jobs from the task pool, one job per tile and kind."""

    pool = parse_world(observation).farm.task_pool
    return [
        FieldTask(task.kind, task.position, task.goods if task.kind == "HARVEST" else "", task.units, task.subject)
        for task in pool
        if task.status == PENDING
    ]


def _positions(farm: dict[str, Any]) -> list[tuple[int, int]]:
    farmer = farm.get("farmer") or [4, 4]
    hands = [(_int(pos[0]), _int(pos[1])) for pos in farm.get("hands") or []]
    return [(_int(farmer[0]), _int(farmer[1])), *hands]


def _wheat_held(observation: dict[str, Any]) -> list[int]:
    private = observation.get("private") or {}
    return [_int(dict(inv or {}).get("WHEAT")) for inv in list(private.get("inventories") or [])]


def _shed(observation: dict[str, Any]) -> dict[str, int]:
    private = observation.get("private") or {}
    return {str(item): _int(amount) for item, amount in dict(private.get("shed") or {}).items() if _int(amount) > 0}


def _commit(actor: ActorRoute, target: tuple[int, int], operation: tuple[Any, ...], not_before: int | None = None) -> int | None:
    arrive = actor.hour + _manhattan(actor.pos, target)
    hour = arrive if not_before is None else max(arrive, not_before)
    if hour >= TURNS_PER_DAY:
        return None
    actor.steps.append(Step(target, operation, hour))
    actor.hour = hour + 1
    actor.pos = target
    return hour


def _make_actors(observation: dict[str, Any], extra_hires: int) -> list[ActorRoute]:
    farm = _farm(observation)
    hour = _int(observation.get("hour"))
    doors = shed_doors(_board_size(farm))
    positions = _positions(farm)
    actors = [ActorRoute(ready_hour=hour, start=position) for position in positions]
    occupied = list(positions)
    for _ in range(max(0, extra_hires)):
        door = _spawn_door(occupied, doors)
        occupied.append(door)
        # A hand hired during this hour can move on the next hour.
        actors.append(ActorRoute(ready_hour=hour + 1, start=door))
    return actors


def _assign_water(actors: list[ActorRoute], tasks: list[FieldTask]) -> int:
    pending = [task.target for task in tasks]
    done = 0
    while pending:
        choice: tuple[int, int, int, tuple[int, int]] | None = None
        for actor_index, actor in enumerate(actors):
            for target in pending:
                hour = actor.hour + _manhattan(actor.pos, target)
                if hour >= TURNS_PER_DAY:
                    continue
                candidate = (hour, actor_index, target[1], target)
                if choice is None or candidate < choice:
                    choice = candidate
        if choice is None:
            break
        hour, actor_index, _, target = choice
        if _commit(actors[actor_index], target, ("WATER",), hour) is None:
            break
        pending.remove(target)
        done += 1
    return done


def _feed_finish(actor: ActorRoute, animals: list[tuple[int, int]], wheat_ready: int, doors: tuple[tuple[int, int], ...]) -> int | None:
    """Hour after this worker has fed every animal, without changing the route."""

    hour = actor.hour
    pos = actor.pos
    if animals:
        door = _nearest_door(pos, doors)
        pickup = max(hour + _manhattan(pos, door), wheat_ready)
        if pickup >= TURNS_PER_DAY:
            return None
        hour = pickup + 1
        pos = door
    remaining = list(animals)
    while remaining:
        target = min(remaining, key=lambda item: (_manhattan(pos, item), item[1], item[0]))
        feed = hour + _manhattan(pos, target)
        if feed >= TURNS_PER_DAY:
            return None
        hour = feed + 1
        pos = target
        remaining.remove(target)
    return hour


def _assign_feed(
    actors: list[ActorRoute],
    tasks: list[FieldTask],
    held: list[int],
    wheat_ready: int,
    doors: tuple[tuple[int, int], ...],
) -> int:
    pending = [task.target for task in tasks]
    done = 0
    for actor_index, actor in enumerate(actors):
        carry = held[actor_index] if actor_index < len(held) else 0
        while carry > 0 and pending:
            target = min(pending, key=lambda item: (_manhattan(actor.pos, item) + actor.hour, item[1], item[0]))
            if _commit(actor, target, ("FEED",)) is None:
                break
            pending.remove(target)
            carry -= 1
            done += 1
    if not pending:
        return done
    choice: tuple[int, int] | None = None
    for actor_index, actor in enumerate(actors):
        finish = _feed_finish(actor, pending, wheat_ready, doors)
        if finish is None:
            continue
        candidate = (finish, actor_index)
        if choice is None or candidate < choice:
            choice = candidate
    if choice is None:
        return done
    actor = actors[choice[1]]
    door = _nearest_door(actor.pos, doors)
    if _commit(actor, door, ("PICKUP", "WHEAT", len(pending)), wheat_ready) is None:
        return done
    while pending:
        target = min(pending, key=lambda item: (_manhattan(actor.pos, item), item[1], item[0]))
        if _commit(actor, target, ("FEED",)) is None:
            break
        pending.remove(target)
        done += 1
    return done


def _harvest_option(
    actor: ActorRoute,
    actor_index: int,
    task: FieldTask,
    task_index: int,
    doors: tuple[tuple[int, int], ...],
) -> tuple[int, int, int, int, tuple[int, int]] | None:
    harvest_hour = actor.hour + _manhattan(actor.pos, task.target)
    if harvest_hour >= TURNS_PER_DAY:
        return None
    door = _nearest_door(task.target, doors)
    drop_hour = harvest_hour + 1 + _manhattan(task.target, door)
    if drop_hour >= TURNS_PER_DAY:
        return None
    return drop_hour, harvest_hour, actor_index, task_index, door


def _assign_harvests(
    actors: list[ActorRoute],
    tasks: list[FieldTask],
    doors: tuple[tuple[int, int], ...],
) -> list[Batch]:
    """Give each worker at most one harvest per wave, hardest tile first.

    A fast worker otherwise grabs two nearby tiles and leaves the far tile for
    a hand that reaches it hours later. The far goods then miss the price.
    """

    pending = list(tasks)
    batches: list[Batch] = []
    while pending:
        options: list[tuple[int, int, int, int, tuple[int, int]]] = []
        best_drop: dict[int, int] = {}
        for actor_index, actor in enumerate(actors):
            for task_index, task in enumerate(pending):
                option = _harvest_option(actor, actor_index, task, task_index, doors)
                if option is None:
                    continue
                options.append(option)
                drop_hour = option[0]
                if task_index not in best_drop or drop_hour < best_drop[task_index]:
                    best_drop[task_index] = drop_hour
        if not options:
            break
        used_actors: set[int] = set()
        assigned: list[tuple[int, int, int, int, tuple[int, int]]] = []
        for task_index in sorted(best_drop, key=lambda index: (-best_drop[index], pending[index].target[1], pending[index].target[0])):
            open_options = [option for option in options if option[3] == task_index and option[2] not in used_actors]
            if not open_options:
                continue
            pick = min(open_options)
            used_actors.add(pick[2])
            assigned.append(pick)
        if not assigned:
            break
        for drop_hour, harvest_hour, actor_index, task_index, door in assigned:
            task = pending[task_index]
            actor = actors[actor_index]
            if _commit(actor, task.target, ("HARVEST",), harvest_hour) is None:
                continue
            if _commit(actor, door, ("PLACE", task.product, task.units), drop_hour) is None:
                continue
            batches.append(
                Batch(
                    product=task.product,
                    units=task.units,
                    target=task.target,
                    actor=actor_index,
                    harvest_hour=harvest_hour,
                    drop_hour=drop_hour,
                    sell_hour=drop_hour,
                )
            )
        for task_index in sorted({option[3] for option in assigned}, reverse=True):
            pending.pop(task_index)
    return batches


def _inventory(observation: dict[str, Any]) -> dict[str, int]:
    market = observation.get("market") or {}
    raw = dict(market.get("inventory") or {})
    return {item: _int(raw.get(item), MARKET_I0) for item in SALE_RANK}


def _town_consume(inventory: dict[str, int], observation: dict[str, Any], step: int) -> None:
    if step % 4 == 0:
        shops = list((observation.get("town") or {}).get("unlocked_shops") or [])
        for shop_name in shops:
            products = list(SHOPS.get(str(shop_name), []))
            multiplier = 2 if len(products) == 1 else 1
            for item in products:
                if item in inventory:
                    inventory[item] -= multiplier
    if step % 24 == 0:
        for item in TOWN_CENTER_PRODUCTS:
            if item in inventory:
                inventory[item] -= 1


def _step(observation: dict[str, Any], hour: int) -> int:
    if "step" in (observation or {}):
        return _int(observation.get("step")) + (hour - _int(observation.get("hour")))
    return _int(observation.get("day")) * TURNS_PER_DAY + hour


def project_sale_revenue(
    observation: dict[str, Any],
    batches: list[Batch],
    opponent_batches: list[Batch] | None = None,
    shed_sales: dict[str, int] | None = None,
    batch_prices: dict[int, int] | None = None,
) -> int:
    """Cash from our sales if both farms drop on this schedule.

    Inside one hour the engine prices both players against the same inventory,
    then adds both units. Town buying happens after that hour's sales.
    """

    inventory = _inventory(observation)
    start = _int(observation.get("hour"))
    ours: dict[int, dict[str, int]] = {}
    theirs: dict[int, dict[str, int]] = {}

    def add(bucket: dict[int, dict[str, int]], hour: int, product: str, units: int) -> None:
        if units <= 0 or product not in SALE_RANK:
            return
        hour_sales = bucket.setdefault(hour, {})
        hour_sales[product] = hour_sales.get(product, 0) + units

    for product, units in (shed_sales or {}).items():
        add(ours, start, product, units)
    batch_queues: dict[tuple[int, str], list[list[Any]]] = {}
    for batch in batches:
        add(ours, batch.sell_hour, batch.product, batch.units)
        if batch_prices is not None:
            batch_queues.setdefault((batch.sell_hour, batch.product), []).append([batch, batch.units])
    for batch in opponent_batches or []:
        add(theirs, batch.sell_hour, batch.product, batch.units)
    shed_left = {product: units for product, units in (shed_sales or {}).items()}

    revenue = 0
    for hour in range(start, TURNS_PER_DAY):
        items = set(ours.get(hour, {})) | set(theirs.get(hour, {}))
        for item in sorted(items, key=lambda name: (-SALE_RANK.get(name, -1), name)):
            left = ours.get(hour, {}).get(item, 0)
            right = theirs.get(hour, {}).get(item, 0)
            while left or right:
                price = market_price(item, inventory[item])
                sold = 0
                if left:
                    revenue += price
                    if batch_prices is not None and hour == start and shed_left.get(item, 0) > 0:
                        shed_left[item] -= 1
                    elif batch_prices is not None:
                        queue = batch_queues.get((hour, item)) or []
                        if queue:
                            queued, remaining = queue[0]
                            batch_prices.setdefault(id(queued), price)
                            remaining -= 1
                            if remaining <= 0:
                                queue.pop(0)
                            else:
                                queue[0][1] = remaining
                    left -= 1
                    sold += int(price > 1)
                if right:
                    right -= 1
                    sold += int(price > 1)
                inventory[item] += sold
        _town_consume(inventory, observation, _step(observation, hour))
    return revenue


def _buy_wheat_cost(observation: dict[str, Any], units: int) -> int:
    if units <= 0:
        return 0
    inventory = _inventory(observation)["WHEAT"]
    spent = 0
    for _ in range(units):
        # A buy is quoted against the inventory after this unit is removed.
        spent += market_price("WHEAT", inventory - 1)
        inventory -= 1
    return spent


def _shed_sales(observation: dict[str, Any], wheat_keep: int) -> dict[str, int]:
    sales: dict[str, int] = {}
    for item, amount in _shed(observation).items():
        if item not in SALE_RANK or item in ENGINE_ANIMALS:
            continue
        if item == "WHEAT":
            amount -= max(0, wheat_keep)
        if amount > 0:
            sales[item] = amount
    return sales


def schedule_day(
    observation: dict[str, Any],
    extra_hires: int,
    opponent_batches: list[Batch] | None = None,
    *,
    include_survival: bool = True,
    include_harvest: bool = True,
) -> DayRoute:
    """Build one concrete route for a fixed number of new hands."""

    farm = _farm(observation)
    hour = _int(observation.get("hour"))
    doors = shed_doors(_board_size(farm))
    tasks = field_tasks(observation)
    waters = [task for task in tasks if task.kind == "WATER"] if include_survival else []
    feeds = [task for task in tasks if task.kind == "FEED"] if include_survival else []
    harvests = [task for task in tasks if task.kind == "HARVEST"] if include_harvest else []
    held = _wheat_held(observation)
    shed_wheat = _shed(observation).get("WHEAT", 0)
    carried = sum(held)
    wheat_buy = max(0, len(feeds) - carried - shed_wheat) if include_survival else 0
    wheat_ready = hour if wheat_buy <= 0 else hour + 1
    actors = _make_actors(observation, extra_hires)
    survival_done = _assign_water(actors, waters)
    survival_done += _assign_feed(actors, feeds, held, wheat_ready, doors)
    batches = _assign_harvests(actors, harvests, doors) if include_harvest else []
    already = _int(farm.get("hires_today"))
    wages = sum(fib_hire_cost(already + index) for index in range(max(0, extra_hires)))
    wheat_keep = max(0, len(feeds) - carried)
    sales = _shed_sales(observation, wheat_keep) if include_survival or include_harvest else {}
    revenue = project_sale_revenue(observation, batches, opponent_batches, sales)
    wheat_cost = _buy_wheat_cost(observation, wheat_buy)
    return DayRoute(
        extra_hires=max(0, extra_hires),
        target_hires=already + max(0, extra_hires),
        start_hour=hour,
        wheat_buy=wheat_buy,
        wages=wages,
        wheat_cost=wheat_cost,
        actors=actors,
        batches=batches,
        survival_total=len(waters) + len(feeds),
        survival_done=survival_done,
        harvests_total=len(harvests),
        harvests_delivered=len(batches),
        revenue=revenue,
        net=revenue - wages - wheat_cost,
    )


def _opponent_batches(observation: dict[str, Any]) -> list[Batch]:
    farms = list(observation.get("farms") or [])
    player = _int(observation.get("player"))
    other = 1 - player
    if other < 0 or other >= len(farms) or other == player:
        return []
    shadow = dict(observation)
    shadow["player"] = other
    shadow["private"] = {"shed": {}, "inventories": [{}], "seeds": {}}
    route = schedule_day(shadow, 0, include_survival=False, include_harvest=True)
    return route.batches


def _hire_limit(observation: dict[str, Any], wheat_buy: int, shed_kinds: int) -> int:
    farm = _farm(observation)
    cash = _int(farm.get("money")) - _buy_wheat_cost(observation, wheat_buy)
    already = _int(farm.get("hires_today"))
    slots = MAX_MARKET_ORDERS - int(wheat_buy > 0) - shed_kinds
    hired = 0
    spent = 0
    while hired < MAX_NEW_HIRES - already and hired < max(0, slots):
        cost = fib_hire_cost(already + hired)
        if cash < spent + cost:
            break
        spent += cost
        hired += 1
    return hired


def choose_day_route(observation: dict[str, Any]) -> DayRoute:
    """Fewest new hands that finish survival, then the schedule with the most cash.

    Every candidate pays for today's water and feed before it takes a harvest.
    A larger crew is used only when the earlier sale hour raises projected cash
    by more than the wages.
    """

    opponent = _opponent_batches(observation)
    probe = schedule_day(observation, 0, opponent)
    # The hire hour shares its ten market slots with feed and with anything sold then.
    kept = max(0, sum(1 for task in field_tasks(observation) if task.kind == "FEED") - sum(_wheat_held(observation)))
    sale_kinds = set(_shed_sales(observation, kept))
    sale_kinds.update(batch.product for batch in probe.batches if batch.sell_hour == probe.start_hour)
    limit = _hire_limit(observation, probe.wheat_buy, len(sale_kinds))
    best: DayRoute | None = None
    for extra in range(limit + 1):
        route = schedule_day(observation, extra, opponent)
        if best is None:
            best = route
            continue
        if route.survival_complete and not best.survival_complete:
            best = route
            continue
        if route.survival_complete != best.survival_complete:
            continue
        if (route.net, route.harvests_delivered, -route.extra_hires) > (
            best.net,
            best.harvests_delivered,
            -best.extra_hires,
        ):
            best = route
    return best if best is not None else probe


def _task_assignments(route: DayRoute) -> list[TaskAssignment]:
    """The route covers a need. Coverage is not the same as the engine doing it."""

    assignments: list[TaskAssignment] = []
    for actor_index, actor in enumerate(route.actors):
        name = worker_name(actor_index)
        steps = actor.steps
        for index, step in enumerate(steps):
            if not step.operation:
                continue
            kind = str(step.operation[0])
            subject = ""
            if kind == "PLACE" and len(step.operation) > 1 and str(step.operation[1]) in {"GOOSE", "COW", "SHEEP"}:
                kind = "PLACE_ANIMAL"
                subject = str(step.operation[1])
            elif kind == "PLANT" and len(step.operation) > 1:
                subject = str(step.operation[1])
            elif kind == "BUILD_COOP":
                subject = "COOP"
            elif kind == "BUILD_PASTURE":
                subject = "PASTURE"
            if kind not in {"WATER", "FEED", "HARVEST", "PLANT", "BUILD_COOP", "BUILD_PASTURE", "PLACE_ANIMAL", "DIG"}:
                continue
            drop_hour = None
            if kind == "HARVEST":
                place = next(
                    (item for item in steps[index + 1 :] if item.operation and item.operation[0] == "PLACE"),
                    None,
                )
                if place is not None:
                    drop_hour = place.not_before
            assignments.append(
                TaskAssignment(
                    kind=kind,
                    position=step.target,
                    assigned_worker=name,
                    planned_hour=step.not_before,
                    planned_drop_hour=drop_hour,
                    planned_sell_hour=drop_hour,
                    subject=subject,
                )
            )
    return assignments


def attach_route(
    world: Any,
    route: DayRoute,
    observation: dict[str, Any],
    cursors: list[int],
    previous_tasks: list[Any] | None = None,
    fallow: list[tuple[int, int]] | None = None,
    animals: list[tuple[int, int]] | None = None,
    escaped: list[tuple[int, int]] | None = None,
) -> None:
    """Write the route onto workers and mark those needs scheduled.

    Prices are estimates for the planned sale hour. A need stays scheduled
    until a later observation shows the water, the feed, or the harvest.
    """

    prices: dict[int, int] = {}
    project_sale_revenue(observation, route.batches, _opponent_batches(observation), batch_prices=prices)
    farm = world.farm
    for actor_index, actor in enumerate(route.actors):
        steps = [WorkerStep(step.not_before, step.target, tuple(step.operation)) for step in actor.steps]
        cursor = cursors[actor_index] if actor_index < len(cursors) else 0
        if actor_index < len(farm.workers):
            worker = farm.workers[actor_index]
        else:
            worker = WorkerState(actor=actor_index, role="hand", position=actor.start, carrying={}, planned=True)
            farm.workers.append(worker)
        worker.route = steps
        worker.completed = steps[:cursor]
        worker.remaining = steps[cursor:]
        worker.current_target = worker.remaining[0].target if worker.remaining else None
        worker.next_target = worker.remaining[1].target if len(worker.remaining) > 1 else None
    for batch in route.batches:
        price = prices.get(id(batch), 0)
        for holder in (*farm.crops, *farm.animals):
            if holder.position != batch.target:
                continue
            holder.next_price = price
            holder.next_revenue = price * batch.units
    settle_tasks(farm, _task_assignments(route), previous_tasks, fallow, animals, escaped)
    if farm.inventory is not None:
        intake = sum(batch.units for batch in route.batches)
        carried = sum(farm.inventory.carried.values())
        farm.inventory.expected_intake = intake
        farm.inventory.overflow_risk = farm.inventory.used + carried + intake > SHED_CAPACITY


def _rival_hand_count(observation: dict[str, Any]) -> int:
    farms = list(observation.get("farms") or [])
    other = 1 - _int(observation.get("player"))
    if other < 0 or other >= len(farms) or not isinstance(farms[other], dict):
        return 0
    return len(farms[other].get("hands") or [])


def make_route14_phase1_agent():
    """Follow the morning route. Do not open a new planting or livestock plan.

    The other farm's hands are cleared overnight and hired again in the morning,
    so they are invisible at hour 0. Their appearance is the one rebuild: the
    sale hours are wrong until the route can see that crew. A price move does
    not rebuild the route.
    """

    state: dict[str, Any] = {
        "day": None,
        "route": None,
        "cursors": [],
        "wheat_bought": False,
        "rival_hands": None,
        "tasks": [],
        "fallow": [],
        "animals": [],
        "escaped": [],
    }

    def agent(observation: dict[str, Any], configuration: dict[str, Any] | None = None) -> dict[str, Any]:
        del configuration
        day = _int(observation.get("day"))
        hour = _int(observation.get("hour"))
        rival_hands = _rival_hand_count(observation)
        day_changed = state["day"] != day or state["route"] is None
        if day_changed or rival_hands != state["rival_hands"]:
            if day_changed:
                state["wheat_bought"] = False
                state["tasks"] = []
            state["day"] = day
            state["rival_hands"] = rival_hands
            state["route"] = choose_day_route(observation)
            state["cursors"] = [0] * len(state["route"].actors)
        route: DayRoute = state["route"]
        world = parse_world(observation)
        attach_route(
            world,
            route,
            observation,
            state["cursors"],
            state["tasks"],
            state["fallow"],
            state["animals"],
            state["escaped"],
        )
        state["tasks"] = list(world.farm.tasks)
        state["fallow"] = list(world.farm.fallow_positions)
        state["escaped"] = list(world.farm.escaped_positions)
        state["animals"] = [animal.position for animal in world.farm.animals]
        state["world"] = world
        farm = _farm(observation)
        positions = _positions(farm)
        reserved = _reserved_wheat(route, state["cursors"])
        commands: list[list[Any]] = []
        for actor, position in enumerate(positions):
            commands.append(_actor_command(route, state, actor, position, hour))
        market = _market_orders(observation, route, state, commands, hour, reserved)
        return {
            "farmer": commands[0] if commands else ["PASS"],
            "hands": commands[1:],
            "market": market,
        }

    agent.telemetry = state  # type: ignore[attr-defined]
    agent.label = "route14_phase1"  # type: ignore[attr-defined]
    return agent


def _actor_command(route: DayRoute, state: dict[str, Any], actor: int, position: tuple[int, int], hour: int) -> list[Any]:
    if actor >= len(route.actors):
        return ["PASS"]
    steps = route.actors[actor].steps
    cursors: list[int] = state["cursors"]
    while actor >= len(cursors):
        cursors.append(0)
    cursor = cursors[actor]
    if cursor >= len(steps):
        return ["PASS"]
    step = steps[cursor]
    if position != step.target:
        return _move_toward(position, step.target)
    if hour < step.not_before:
        return ["PASS"]
    cursors[actor] = cursor + 1
    return list(step.operation)


def _reserved_wheat(route: DayRoute, cursors: list[int]) -> int:
    reserved = 0
    for actor, actor_route in enumerate(route.actors):
        cursor = cursors[actor] if actor < len(cursors) else 0
        for step in actor_route.steps[cursor:]:
            if step.operation and step.operation[0] == "PICKUP" and step.operation[1] == "WHEAT":
                reserved += _int(step.operation[2])
    return reserved


def _incoming_drops(commands: list[list[Any]]) -> dict[str, int]:
    incoming: dict[str, int] = {}
    for command in commands:
        if len(command) >= 3 and command[0] in {"PLACE", "DROP"} and command[1] in SALE_RANK:
            incoming[str(command[1])] = incoming.get(str(command[1]), 0) + _int(command[2])
    return incoming


def _market_orders(
    observation: dict[str, Any],
    route: DayRoute,
    state: dict[str, Any],
    commands: list[list[Any]],
    hour: int,
    reserved_wheat: int,
) -> list[list[Any]]:
    """Buy feed and sell this hour's drops before spending the remaining slots on hires.

    The engine unloads workers before it runs the market, so a drop and its sale
    share an hour. A hire issued in a later slot can wait; a missed sale cannot.
    """

    incoming = _incoming_drops(commands)
    prices = dict((observation.get("market") or {}).get("prices") or {})
    sells: list[list[Any]] = []
    seen: set[str] = set()
    for item, amount in _shed(observation).items():
        if item in ENGINE_ANIMALS or item not in SALE_RANK:
            continue
        available = amount + incoming.get(item, 0)
        if item == "WHEAT":
            available -= reserved_wheat
        if available > 0:
            sells.append(["SELL", item, available])
            seen.add(item)
    for item, amount in incoming.items():
        if item in seen or item not in SALE_RANK:
            continue
        available = amount - reserved_wheat if item == "WHEAT" else amount
        if available > 0:
            sells.append(["SELL", item, available])
    sells.sort(key=lambda order: (-_int(prices.get(order[1])), -SALE_RANK.get(str(order[1]), 0), str(order[1])))

    orders: list[list[Any]] = []
    if route.wheat_buy > 0 and hour == route.start_hour and not state["wheat_bought"]:
        orders.append(["BUY_PRODUCT", "WHEAT", route.wheat_buy])
        state["wheat_bought"] = True
    for order in sells:
        if len(orders) >= MAX_MARKET_ORDERS:
            break
        orders.append(order)
    farm = _farm(observation)
    hires_left = max(0, route.target_hires - _int(farm.get("hires_today")))
    while hires_left and len(orders) < MAX_MARKET_ORDERS:
        orders.append(["HIRE"])
        hires_left -= 1
    return orders
