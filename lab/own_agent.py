"""An independently authored, inspectable Kaggriculture production baseline.

It deliberately uses a small 15-tile melon field rather than copying any public
route tape.  Two daily hands plus the main farmer each own a row, which makes the
daily watering/harvest schedule explainable and keeps one full harvest below the
100-unit shed cap.  Its sale policy is parameterized for local search.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


FIELD_ROWS = (2, 3, 4)
FIELD_COLUMNS = tuple(range(5))
MELON_SEED_COST = 80
MELON_FIRST_YIELD_DAY = 10
MELON_BASE_PRICE = 250


@dataclass(frozen=True)
class MelonFarmConfig:
    """Small, bounded parameter space for the first policy-search experiments."""

    sell_batch: int = 90
    min_price_ratio: float = 0.70
    emergency_shed_units: int = 80
    daily_hands: int = 2
    initial_seed_units: int = 15
    restock_days: tuple[int, ...] = (0, 12)

    def label(self) -> str:
        return f"own-melon-b{self.sell_batch}-p{self.min_price_ratio:.2f}-e{self.emergency_shed_units}"


def _move_toward(position: list[int], target: tuple[int, int]) -> list[str]:
    x, y = position
    target_x, target_y = target
    if x < target_x:
        return ["EAST"]
    if x > target_x:
        return ["WEST"]
    if y < target_y:
        return ["SOUTH"]
    if y > target_y:
        return ["NORTH"]
    return ["PASS"]


def _tile_at(farm: dict[str, Any], target: tuple[int, int]) -> Any:
    x, y = target
    return farm["tiles"][y][x]


def _row_command(
    obs: dict[str, Any],
    position: list[int],
    row: int,
) -> list[str]:
    """Choose one legal, local task for an actor responsible for exactly one row."""

    farm = obs["farms"][obs["player"]]
    private = obs["private"]
    day = int(obs["day"])
    seeds = int(private["seeds"].get("MELON", 0))
    candidates: list[tuple[int, int, int, tuple[int, int], list[str]]] = []

    for x in FIELD_COLUMNS:
        target = (x, row)
        tile = _tile_at(farm, target)
        distance = abs(position[0] - x) + abs(position[1] - row)
        command: list[str] | None = None
        priority = 99

        if isinstance(tile, dict) and tile.get("kind") == "PLANT" and tile.get("crop") == "MELON":
            age = day - int(tile["planted_day"])
            if age >= MELON_FIRST_YIELD_DAY and int(tile.get("yield_units", 0)) > 0:
                priority, command = 0, ["HARVEST"]
            elif not tile.get("watered_today", False):
                priority, command = 1, ["WATER"]
        elif isinstance(tile, dict) and tile.get("kind") == "WEED":
            priority, command = 2, ["DIG"]
        elif tile is None and seeds > 0:
            priority, command = 3, ["PLANT", "MELON"]

        if command is not None:
            candidates.append((priority, distance, x, target, command))

    if not candidates:
        return ["PASS"]

    _, _, _, target, command = min(candidates)
    return command if tuple(position) == target else _move_toward(position, target)


def make_melon_farm_agent(config: MelonFarmConfig = MelonFarmConfig()):
    """Create a fresh stateful agent function for one independent local episode."""

    telemetry: dict[str, int] = {}

    def reset_telemetry() -> None:
        telemetry.clear()
        telemetry.update(
            hires_requested=0,
            seed_orders=0,
            seed_units_requested=0,
            sell_orders=0,
            sell_units_requested=0,
            emergency_sales=0,
            harvest_commands=0,
            water_commands=0,
            plant_commands=0,
            weed_dig_commands=0,
        )

    def note(command: list[str]) -> None:
        op = command[0]
        if op == "HARVEST":
            telemetry["harvest_commands"] += 1
        elif op == "WATER":
            telemetry["water_commands"] += 1
        elif op == "PLANT":
            telemetry["plant_commands"] += 1
        elif op == "DIG":
            telemetry["weed_dig_commands"] += 1

    def agent(obs: dict[str, Any], configuration: dict[str, Any] | None = None) -> dict[str, Any]:
        del configuration  # Standard configuration is read from the observation.
        if int(obs.get("step", 0)) == 0:
            reset_telemetry()

        player = int(obs["player"])
        farm = obs["farms"][player]
        private = obs["private"]
        day = int(obs["day"])
        hour = int(obs["hour"])
        market: list[list[Any]] = []
        shed_melons = int(private["shed"].get("MELON", 0))
        current_price = int(obs["market"]["prices"].get("MELON", MELON_BASE_PRICE))

        # Sell before purchases: it protects shed capacity and gives the first market
        # order position to our only sale. Terminal liquidation always overrides price.
        days_remaining = 30 - day
        emergency = shed_melons >= config.emergency_shed_units
        price_ok = current_price >= MELON_BASE_PRICE * config.min_price_ratio
        if shed_melons and (emergency or price_ok or days_remaining <= 2):
            quantity = shed_melons if emergency or days_remaining <= 2 else min(shed_melons, config.sell_batch)
            market.append(["SELL", "MELON", quantity])
            telemetry["sell_orders"] += 1
            telemetry["sell_units_requested"] += quantity
            telemetry["emergency_sales"] += int(emergency)

        # Hands disappear at day end. Hire only at hour zero; they are available from
        # the next hour and each owns one field row for the rest of the day.
        if hour == 0 and day < 29:
            for _ in range(config.daily_hands):
                market.append(["HIRE"])
                telemetry["hires_requested"] += 1

        # A 15-unit seed order fills the three owned rows. Day 12 replenishes before
        # the first harvest/replant cycle; no late-season restock is needed.
        if day in config.restock_days and hour == 0 and private["seeds"].get("MELON", 0) == 0:
            affordable = int(farm["money"]) >= config.initial_seed_units * MELON_SEED_COST
            if affordable:
                market.append(["BUY_SEED", "MELON", config.initial_seed_units])
                telemetry["seed_orders"] += 1
                telemetry["seed_units_requested"] += config.initial_seed_units

        actor_positions = [farm["farmer"], *farm.get("hands", [])]
        commands: list[list[str]] = []
        for actor_index, position in enumerate(actor_positions):
            row = FIELD_ROWS[actor_index] if actor_index < len(FIELD_ROWS) else FIELD_ROWS[-1]
            command = _row_command(obs, position, row)
            note(command)
            commands.append(command)

        return {"farmer": commands[0], "hands": commands[1:], "market": market[:10]}

    agent.telemetry = telemetry  # type: ignore[attr-defined]
    agent.label = config.label()  # type: ignore[attr-defined]
    return agent


def own_melon_agent(obs: dict[str, Any], configuration: dict[str, Any] | None = None) -> dict[str, Any]:
    """Default parameterization used by the command-line local league."""

    return _DEFAULT_AGENT(obs, configuration)


_DEFAULT_AGENT = make_melon_farm_agent()
own_melon_agent.telemetry = _DEFAULT_AGENT.telemetry  # type: ignore[attr-defined]
own_melon_agent.label = _DEFAULT_AGENT.label  # type: ignore[attr-defined]
