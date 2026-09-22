"""Optimistic raw-route animal output upper bound from a day-6 observation.

This is a forecast, not a replay: it never reads future shops, prices, cash or
V45 overlays.  It assumes every existing animal and every later PLACE written
in the candidate tape is fed, cared and harvested on time.  Wool especially
can be under-counted when the later sheep overlay actually fires.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from .portfolio_model import ANIMAL_OUTPUT

# first_yield_days, interval, max_held, product
ANIMAL_YIELD = {
    "GOOSE": (4, 1, 4, "EGG"),
    "COW": (8, 2, 6, "MILK"),
    "SHEEP": (6, 3, 6, "WOOL"),
}


def future_units(animal: str, placed_day: int, *, day: int, held: int = 0, pending_care: int = 0) -> int:
    """Upper-bound product units if the animal survives and is fully tended."""

    first, interval, cap, _ = ANIMAL_YIELD[animal]
    total = max(0, held)
    pending = max(0, pending_care)
    for current_day in range(day, 29):
        if current_day < placed_day:
            continue
        pending += 1
        next_day = current_day + 1
        if next_day - placed_day >= first and (next_day - placed_day - first) % interval == 0:
            total += min(cap, 1 + pending)
            pending = 0
    return total


def placed_animals_in_route(module: Any, route: int) -> list[tuple[str, int]]:
    placements: list[tuple[str, int]] = []
    for step, action in enumerate(module._IMPL.chassis.routes[route][144:648], start=144):
        for command in [action.get("farmer") or ["PASS"], *(action.get("hands") or [])]:
            if len(command) >= 2 and command[0] == "PLACE" and command[1] in ANIMAL_YIELD:
                placements.append((command[1], step // 24))
    return placements


def forecast_animal_units(observation: dict[str, Any], module: Any, route: int) -> Counter[str]:
    """Existing animals plus raw-route PLACE commands; overlays are excluded."""

    player = int(observation.get("player", 0))
    day = int(observation.get("day", 6))
    outputs: Counter[str] = Counter()
    farms = list(observation.get("farms") or [])
    farm = farms[player] if 0 <= player < len(farms) else {}
    for row in farm.get("tiles") or []:
        for tile in row:
            if not isinstance(tile, dict) or tile.get("animal") not in ANIMAL_YIELD:
                continue
            animal = str(tile["animal"])
            outputs[ANIMAL_OUTPUT[animal]] += future_units(
                animal,
                int(tile.get("placed_day", day)),
                day=day,
                held=int(tile.get("yield_units", 0)),
                pending_care=int(tile.get("pending_care_bonus", 0)),
            )
    for animal, placed_day in placed_animals_in_route(module, route):
        outputs[ANIMAL_OUTPUT[animal]] += future_units(animal, placed_day, day=day)
    return outputs
