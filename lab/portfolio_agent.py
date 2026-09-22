"""First executable, independently authored 70/30 portfolio agent.

This is intentionally a small safety-first executor, not a V45 wrapper. It
maintains observed crops/animals before discretionary work, hires a bounded
daily crew, sells safely stored output, and grows crop/animal lines through
explicit legal task packages.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any

from .portfolio_model import choose_portfolio


SEED_COST = {"WHEAT": 10, "CARROT": 20, "TOMATO": 50, "STRAWBERRY": 100, "MELON": 80}
CROP_PEAK_DAY = {"WHEAT": 4, "CARROT": 3, "TOMATO": 11, "STRAWBERRY": 16, "MELON": 10}
PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")
ANIMALS = {
    "GOOSE": {"structure": "COOP", "cost": 300},
    "COW": {"structure": "PASTURE", "cost": 400},
    "SHEEP": {"structure": "PASTURE", "cost": 500},
}
SHED_ACCESS = ((4, 4), (5, 4), (4, 5), (5, 5))


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _move_toward(position: list[int], target: tuple[int, int]) -> list[str]:
    x, y = _int(position[0]), _int(position[1])
    tx, ty = target
    if x < tx:
        return ["EAST"]
    if x > tx:
        return ["WEST"]
    if y < ty:
        return ["SOUTH"]
    if y > ty:
        return ["NORTH"]
    return ["PASS"]


@dataclass(frozen=True)
class Task:
    target: tuple[int, int]
    operation: list[str]
    priority: int
    actor: int | None = None
    requires_item: str | None = None


@dataclass(frozen=True)
class PortfolioAgentConfig:
    max_daily_hires: int = 4
    target_crop_tiles: int = 12
    target_animals: int = 2
    cash_reserve: int = 500
    enable_care: bool = True

    def label(self) -> str:
        return "portfolio-executor-v0-crops"


def _tile_tasks(observation: dict[str, Any], *, enable_care: bool) -> list[Task]:
    """Return only work that is already visible and legal at its target tile."""

    player = _int(observation.get("player"))
    farm = list(observation.get("farms") or [{}, {}])[player]
    day = _int(observation.get("day"))
    tasks: list[Task] = []
    for y, row in enumerate(farm.get("tiles") or []):
        for x, tile in enumerate(row):
            if not isinstance(tile, dict):
                continue
            target = (x, y)
            if tile.get("animal"):
                if not tile.get("fed_today", False):
                    tasks.append(Task(target, ["FEED"], 100, requires_item="WHEAT"))
                elif _int(tile.get("yield_units")) > 0:
                    tasks.append(Task(target, ["HARVEST"], 95))
                elif enable_care and not tile.get("cared_today", False):
                    tasks.append(Task(target, ["CARE"], 85))
                continue
            if tile.get("kind") == "WEED":
                tasks.append(Task(target, ["DIG"], 70))
                continue
            if tile.get("kind") != "PLANT":
                continue
            crop = str(tile.get("crop") or "")
            age = day - _int(tile.get("planted_day"))
            if _int(tile.get("yield_units")) > 0 and age >= CROP_PEAK_DAY.get(crop, 99):
                tasks.append(Task(target, ["HARVEST"], 95))
            elif not tile.get("watered_today", False):
                # All plants need water to survive, with a small lift near the
                # crop's payoff date. Either way it outranks new planting.
                tasks.append(Task(target, ["WATER"], 88 if age >= CROP_PEAK_DAY.get(crop, 99) - 1 else 80))
    return tasks


def _best_crop(observation: dict[str, Any]) -> str | None:
    portfolio = choose_portfolio(observation)
    for line in portfolio["ranked_lines"]:
        if line["kind"] == "crop" and line["output"] in SEED_COST and line["feasible_before_terminal"]:
            return str(line["output"])
    return None


def _best_animal(observation: dict[str, Any]) -> str | None:
    """Map the transparent portfolio's current animal choice to a legal item."""

    for line in choose_portfolio(observation)["ranked_lines"]:
        if line["kind"] != "animal" or not line["feasible_before_terminal"]:
            continue
        name = str(line["line"])
        if name.startswith("goose_"):
            return "GOOSE"
        if name.startswith("cow_"):
            return "COW"
        if name.startswith("sheep_"):
            return "SHEEP"
    return None


def _empty_unlocked_tiles(observation: dict[str, Any]) -> list[tuple[int, int]]:
    player = _int(observation.get("player"))
    farm = list(observation.get("farms") or [{}, {}])[player]
    return [(x, y) for y, row in enumerate(farm.get("tiles") or []) for x, tile in enumerate(row) if tile is None]


def _animal_counts(observation: dict[str, Any]) -> tuple[int, int]:
    player = _int(observation["player"])
    farm = observation["farms"][player]
    placed = sum(
        int(isinstance(tile, dict) and bool(tile.get("animal")))
        for row in farm.get("tiles") or []
        for tile in row
    )
    shed = observation.get("private", {}).get("shed", {})
    stored = sum(_int(shed.get(animal)) for animal in ANIMALS)
    stored += sum(
        _int(dict(inventory or {}).get(animal))
        for inventory in observation.get("private", {}).get("inventories", [])
        for animal in ANIMALS
    )
    return placed, stored


def _empty_structure_tiles(observation: dict[str, Any], structure: str) -> list[tuple[int, int]]:
    player = _int(observation["player"])
    farm = observation["farms"][player]
    return [
        (x, y)
        for y, row in enumerate(farm.get("tiles") or [])
        for x, tile in enumerate(row)
        if isinstance(tile, dict) and tile.get("kind") == structure and not tile.get("animal")
    ]


def make_portfolio_agent(config: PortfolioAgentConfig = PortfolioAgentConfig()):
    """Return a fresh agent with day-local target reservations.

    Reservations make a worker continue walking to its assigned field instead
    of re-deciding every turn.  They are revalidated against the next public
    observation, so an executed action or a changed tile releases the target.
    """

    state: dict[str, Any] = {"day": None, "assignments": {}}
    telemetry: dict[str, int] = {}

    def reset_run() -> None:
        telemetry.clear()
        telemetry.update(
            hires_requested=0,
            seed_orders=0,
            seed_units_requested=0,
            sell_orders=0,
            sell_units_requested=0,
            feed_commands=0,
            water_commands=0,
            harvest_commands=0,
            dig_commands=0,
            plant_commands=0,
            invalidated_assignments=0,
            animal_orders=0,
            care_commands=0,
            pickup_commands=0,
            place_commands=0,
            build_coop_commands=0,
            build_pasture_commands=0,
        )

    def reset_day(day: int) -> None:
        state["day"] = day
        state["assignments"] = {}

    def reserve_crop_tasks(observation: dict[str, Any], tasks: list[Task], actor_count: int) -> list[Task]:
        player = _int(observation["player"])
        private = dict(observation.get("private") or {})
        crop = _best_crop(observation)
        if crop is None or _int(private.get("seeds", {}).get(crop)) <= 0:
            return tasks
        existing_crop_tiles = sum(
            int(isinstance(tile, dict) and tile.get("kind") == "PLANT" and tile.get("crop") == crop)
            for row in observation["farms"][player].get("tiles") or []
            for tile in row
        )
        available = max(0, min(_int(private.get("seeds", {}).get(crop)), config.target_crop_tiles - existing_crop_tiles))
        already_reserved = sum(
            1
            for reserved in state["assignments"].values()
            if reserved.get("operation") == ["PLANT", crop]
        )
        available = max(0, available - already_reserved)
        if not available:
            return tasks
        busy = {tuple(reserved["target"]) for reserved in state["assignments"].values()}
        for target in _empty_unlocked_tiles(observation):
            if target in busy:
                continue
            tasks.append(Task(target, ["PLANT", crop], 50))
            available -= 1
            if available <= 0 or len(tasks) >= actor_count + len(state["assignments"]):
                break
        return tasks

    def reserve_animal_tasks(observation: dict[str, Any], tasks: list[Task], positions: list[list[int]]) -> list[Task]:
        """Append one safe next step of a BUILD -> BUY -> PICKUP -> PLACE chain.

        BUY itself is handled in the hour-zero market list. This function only
        emits field tasks that are legal from currently visible state; it never
        assumes an animal purchase succeeded before it appears in the shed.
        """

        private = observation.get("private", {})
        inventories = list(private.get("inventories") or [])
        # A worker already carrying an animal must place it before any second
        # worker is allowed to pick up another one.
        for actor, inventory in enumerate(inventories[:len(positions)]):
            for animal, info in ANIMALS.items():
                if _int(dict(inventory or {}).get(animal)) > 0:
                    matching = _empty_structure_tiles(observation, info["structure"])
                    if matching:
                        target = min(matching, key=lambda item: (abs(positions[actor][0] - item[0]) + abs(positions[actor][1] - item[1]), item))
                        tasks.append(Task(target, ["PLACE", animal], 78, actor=actor))
                    return tasks
        # Animals already bought are commitments, not optional inventory. Place
        # any one that has a matching empty structure before applying the target
        # animal cap or consulting a newly changed market recommendation.
        stored_choices = [
            animal
            for animal, info in ANIMALS.items()
            if _int(private.get("shed", {}).get(animal)) > 0 and _empty_structure_tiles(observation, info["structure"])
        ]
        if stored_choices:
            desired = _best_animal(observation)
            animal = desired if desired in stored_choices else sorted(stored_choices)[0]
            target = min(SHED_ACCESS, key=lambda item: (min(abs(pos[0] - item[0]) + abs(pos[1] - item[1]) for pos in positions), item))
            tasks.append(Task(target, ["PICKUP", animal, 1], 76))
            return tasks
        desired = _best_animal(observation)
        if desired is None:
            return tasks
        placed, stored = _animal_counts(observation)
        if placed + stored >= config.target_animals:
            return tasks
        structure = ANIMALS[desired]["structure"]
        structures = _empty_structure_tiles(observation, structure)
        busy = {tuple(reserved["target"]) for reserved in state["assignments"].values()}
        if not structures:
            empties = [target for target in _empty_unlocked_tiles(observation) if target not in busy]
            if empties:
                target = min(empties, key=lambda item: (min(abs(pos[0] - item[0]) + abs(pos[1] - item[1]) for pos in positions), item))
                tasks.append(Task(target, [f"BUILD_{structure}"], 60))
        return tasks

    def reserve_feed_pickup_task(observation: dict[str, Any], tasks: list[Task], positions: list[list[int]]) -> list[Task]:
        """Put wheat in a worker's hands before any FEED task can be assigned."""

        player = _int(observation["player"])
        farm = observation["farms"][player]
        unfed = sum(
            int(isinstance(tile, dict) and bool(tile.get("animal")) and not tile.get("fed_today", False))
            for row in farm.get("tiles") or []
            for tile in row
        )
        if not unfed:
            return tasks
        inventories = list(observation.get("private", {}).get("inventories") or [])
        carried = sum(_int(dict(inventory or {}).get("WHEAT")) for inventory in inventories)
        if carried:
            return tasks
        shed_wheat = _int(observation.get("private", {}).get("shed", {}).get("WHEAT"))
        if not shed_wheat:
            return tasks
        target = min(SHED_ACCESS, key=lambda item: (min(abs(pos[0] - item[0]) + abs(pos[1] - item[1]) for pos in positions), item))
        tasks.append(Task(target, ["PICKUP", "WHEAT", min(unfed, shed_wheat)], 101))
        return tasks

    def market_orders(observation: dict[str, Any], mandatory_count: int) -> list[list[Any]]:
        player = _int(observation["player"])
        farm = observation["farms"][player]
        private = dict(observation.get("private") or {})
        day, hour = _int(observation.get("day")), _int(observation.get("hour"))
        if hour != 0:
            return []
        orders: list[list[Any]] = []
        animals, stored_animals = _animal_counts(observation)
        shed = dict(private.get("shed") or {})
        for item in PRODUCTS:
            quantity = _int(shed.get(item))
            if item == "WHEAT":
                quantity = max(0, quantity - animals)
            if quantity > 0:
                orders.append(["SELL", item, quantity])
        if day < 29:
            desired_hires = min(config.max_daily_hires, max(2, ceil(mandatory_count / 8)))
            # Do not spend the last working capital on temporary workers.
            affordable_hires = 0
            simulated_money = _int(farm.get("money")) - config.cash_reserve
            fib_costs = (1, 1, 2, 3, 5, 8, 13, 21)
            for cost in fib_costs[:desired_hires]:
                if simulated_money < cost:
                    break
                simulated_money -= cost
                affordable_hires += 1
            orders.extend([["HIRE"] for _ in range(affordable_hires)])
            telemetry["hires_requested"] += affordable_hires
        if animals and _int(shed.get("WHEAT")) < animals:
            orders.append(["BUY_PRODUCT", "WHEAT", animals - _int(shed.get("WHEAT"))])
        crop = _best_crop(observation)
        if crop and day <= CROP_PEAK_DAY[crop]:
            seed_count = _int(private.get("seeds", {}).get(crop))
            empty = len(_empty_unlocked_tiles(observation))
            wanted = max(0, min(config.target_crop_tiles, empty) - seed_count)
            cost = wanted * SEED_COST[crop]
            if wanted and _int(farm.get("money")) - config.cash_reserve >= cost:
                orders.append(["BUY_SEED", crop, wanted])
                telemetry["seed_orders"] += 1
                telemetry["seed_units_requested"] += wanted
        desired_animal = _best_animal(observation)
        if desired_animal and animals + stored_animals < config.target_animals:
            structure = ANIMALS[desired_animal]["structure"]
            if _empty_structure_tiles(observation, structure):
                cost = ANIMALS[desired_animal]["cost"]
                if _int(farm.get("money")) - config.cash_reserve >= cost:
                    orders.append(["BUY_ANIMAL", desired_animal, 1])
                    telemetry["animal_orders"] += 1
        # Sale orders come first deliberately. The environment caps every turn
        # at ten orders, so preserve the most valuable current products first.
        prices = dict(observation.get("market", {}).get("prices", {}) or {})
        sales = [order for order in orders if order[0] == "SELL"]
        other = [order for order in orders if order[0] != "SELL"]
        sales.sort(key=lambda order: (-_int(prices.get(order[1])), str(order[1])))
        result = (sales + other)[:10]
        telemetry["sell_orders"] += sum(order[0] == "SELL" for order in result)
        telemetry["sell_units_requested"] += sum(_int(order[2]) for order in result if order[0] == "SELL")
        return result

    def assignment_is_current(observation: dict[str, Any], actor: int, assignment: dict[str, Any]) -> bool:
        player = _int(observation["player"])
        x, y = assignment["target"]
        tile = observation["farms"][player]["tiles"][y][x]
        operation = assignment["operation"]
        if operation[0] == "PLANT":
            return tile is None and _int(observation["private"].get("seeds", {}).get(operation[1])) > 0
        if operation[0] == "DIG":
            return isinstance(tile, dict) and tile.get("kind") == "WEED"
        if operation[0] == "WATER":
            return isinstance(tile, dict) and tile.get("kind") == "PLANT" and not tile.get("watered_today", False)
        if operation[0] == "FEED":
            inventory = list(observation.get("private", {}).get("inventories") or [])
            has_wheat = actor < len(inventory) and _int(dict(inventory[actor] or {}).get("WHEAT")) > 0
            return isinstance(tile, dict) and tile.get("animal") and not tile.get("fed_today", False) and has_wheat
        if operation[0] == "HARVEST":
            return isinstance(tile, dict) and _int(tile.get("yield_units")) > 0
        if operation[0] == "CARE":
            return isinstance(tile, dict) and tile.get("animal") and tile.get("fed_today", False) and not tile.get("cared_today", False)
        if operation[0] == "PICKUP":
            return tuple(assignment["target"]) in SHED_ACCESS and _int(observation["private"].get("shed", {}).get(operation[1])) > 0
        if operation[0] == "PLACE":
            return isinstance(tile, dict) and tile.get("kind") == ANIMALS[operation[1]]["structure"] and not tile.get("animal")
        if operation[0] in {"BUILD_COOP", "BUILD_PASTURE"}:
            return tile is None
        return False

    def commands(observation: dict[str, Any]) -> list[list[str]]:
        player = _int(observation["player"])
        farm = observation["farms"][player]
        positions = [list(farm["farmer"]), *[list(pos) for pos in farm.get("hands") or []]]
        active: dict[int, dict[str, Any]] = {}
        for actor, assignment in list(state["assignments"].items()):
            if actor >= len(positions) or not assignment_is_current(observation, actor, assignment):
                state["assignments"].pop(actor, None)
                telemetry["invalidated_assignments"] += 1
            else:
                active[actor] = assignment
        tasks = reserve_crop_tasks(observation, _tile_tasks(observation, enable_care=config.enable_care), len(positions))
        tasks = reserve_feed_pickup_task(observation, tasks, positions)
        tasks = reserve_animal_tasks(observation, tasks, positions)
        blocked = {tuple(assignment["target"]) for assignment in active.values()}
        available_actors = [actor for actor in range(len(positions)) if actor not in active]
        while available_actors:
            choices: list[tuple[int, int, int, tuple[int, int], Task]] = []
            for task in tasks:
                if task.target in blocked:
                    continue
                for actor in available_actors:
                    if task.actor is not None and task.actor != actor:
                        continue
                    if task.requires_item:
                        inventories = list(observation.get("private", {}).get("inventories") or [])
                        if actor >= len(inventories) or _int(dict(inventories[actor] or {}).get(task.requires_item)) <= 0:
                            continue
                    distance = abs(positions[actor][0] - task.target[0]) + abs(positions[actor][1] - task.target[1])
                    choices.append((-task.priority, distance, actor, task.target, task))
            if not choices:
                break
            _, _, actor, _, task = min(choices, key=lambda item: item[:4])
            state["assignments"][actor] = {"target": list(task.target), "operation": list(task.operation)}
            active[actor] = state["assignments"][actor]
            blocked.add(task.target)
            available_actors.remove(actor)
        result: list[list[str]] = []
        for actor, position in enumerate(positions):
            assignment = active.get(actor)
            if not assignment:
                result.append(["PASS"])
                continue
            target = tuple(assignment["target"])
            command = list(assignment["operation"]) if tuple(position) == target else _move_toward(position, target)
            if command[0] in {"FEED", "WATER", "HARVEST", "DIG", "PLANT", "CARE", "PICKUP", "PLACE", "BUILD_COOP", "BUILD_PASTURE"}:
                telemetry[f"{command[0].lower()}_commands"] += 1
            result.append(command)
        return result

    def agent(observation: dict[str, Any], configuration: dict[str, Any] | None = None) -> dict[str, Any]:
        del configuration
        day = _int(observation.get("day"))
        if _int(observation.get("step")) == 0:
            reset_run()
        if state["day"] != day:
            reset_day(day)
        mandatory_count = len(_tile_tasks(observation, enable_care=config.enable_care))
        action_commands = commands(observation)
        return {
            "farmer": action_commands[0] if action_commands else ["PASS"],
            "hands": action_commands[1:],
            "market": market_orders(observation, mandatory_count),
        }

    agent.telemetry = telemetry  # type: ignore[attr-defined]
    agent.label = config.label()  # type: ignore[attr-defined]
    return agent
