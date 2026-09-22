"""Conservative day-6 animal-output forecast audit; never selects a route."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from kaggle_environments import make

from .baselines import load_external_agent, load_v45_base_agent, load_v45_base_module
from .portfolio_model import ANIMAL_OUTPUT


CASES = ((1, 108, 123), (2, 105, 123), (3, 7, 123))
ANIMAL = {
    "GOOSE": (4, 1, 4, "EGG"),
    "COW": (8, 2, 6, "MILK"),
    "SHEEP": (6, 3, 6, "WOOL"),
}


def _day6_observation(seed: int) -> dict[str, Any]:
    environment = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed}, debug=True)
    environment.run([load_v45_base_agent(), load_external_agent("boatlee_v16")])
    return environment.steps[144][0].observation


def _placed_animals_in_route(module: Any, route: int) -> list[tuple[str, int]]:
    """Raw tape's written PLACE commands, without claiming they will succeed."""

    placements: list[tuple[str, int]] = []
    for step, action in enumerate(module._IMPL.chassis.routes[route][144:648], start=144):
        for command in [action.get("farmer") or ["PASS"], *(action.get("hands") or [])]:
            if len(command) >= 2 and command[0] == "PLACE" and command[1] in ANIMAL:
                placements.append((command[1], step // 24))
    return placements


def _future_units(animal: str, placed_day: int, *, day: int, held: int = 0, pending_care: int = 0) -> int:
    """Upper-bound product units if it survives, is fed/cared and promptly harvested.

    It intentionally ignores walking, money, shed space, conflicting workers and
    V45 overlays.  Those omissions are the error sources this audit measures.
    """

    first, interval, cap, _ = ANIMAL[animal]
    total = max(0, held)
    pending = max(0, pending_care)
    # At day-6 hour 0 through the last harvestable day-29 boundary.  Add one
    # CARE to every day as an optimistic upper bound, then immediately harvest
    # each produced batch so `max_held` does not hide output.
    for current_day in range(day, 29):
        if current_day < placed_day:
            continue
        pending += 1
        next_day = current_day + 1
        if next_day - placed_day >= first and (next_day - placed_day - first) % interval == 0:
            total += min(cap, 1 + pending)
            pending = 0
    return total


def _forecast(observation: dict[str, Any], module: Any, route: int) -> dict[str, Any]:
    player = int(observation["player"])
    day = int(observation["day"])
    outputs: Counter[str] = Counter()
    existing: Counter[str] = Counter()
    for row in observation["farms"][player]["tiles"]:
        for tile in row:
            if isinstance(tile, dict) and tile.get("animal") in ANIMAL:
                animal = tile["animal"]
                product = ANIMAL[animal][3]
                existing[product] += 1
                outputs[product] += _future_units(
                    animal,
                    int(tile["placed_day"]),
                    day=day,
                    held=int(tile.get("yield_units", 0)),
                    pending_care=int(tile.get("pending_care_bonus", 0)),
                )
    planned = _placed_animals_in_route(module, route)
    for animal, placed_day in planned:
        outputs[ANIMAL[animal][3]] += _future_units(animal, placed_day, day=day)
    return {"existing_animals_by_output": dict(existing), "raw_route_places": planned, "forecast_units": dict(outputs)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("experiments/v45_day6_animal_forecast_audit_v1.json"))
    args = parser.parse_args()
    module = load_v45_base_module()
    realized = json.loads(Path("experiments/v45_route_supply_demand_audit_v3.json").read_text(encoding="utf-8"))
    actual_by_seed = {int(case["seed"]): case for case in realized["cases"]}
    cases = []
    for seed, original, candidate in CASES:
        observation = _day6_observation(seed)
        raw = actual_by_seed[seed]
        row = {"seed": seed, "shops": observation["town"]["unlocked_shops"], "original_route": original, "candidate_route": candidate}
        for name, route, key in (("baseline", original, "baseline_harvested"), ("candidate", candidate, "candidate_harvested")):
            forecast = _forecast(observation, module, route)
            actual = {item["product"]: item[key] for item in raw["products"] if item["product"] in ANIMAL_OUTPUT.values()}
            forecast["actual_harvested"] = actual
            forecast["error"] = {product: forecast["forecast_units"].get(product, 0) - actual.get(product, 0) for product in ANIMAL_OUTPUT.values()}
            row[name] = forecast
        cases.append(row)
    report = {
        "protocol": "day6-animal-forecast-audit-v1-optimistic-raw-place",
        "limitations": [
            "Forecasts only animals, not crops or fertilizer.",
            "Raw PLACE commands may fail; V45 overlays may add animals not in raw routes.",
            "Assumes daily feed/care and prompt harvest, so it is an upper-bound rather than live policy output.",
        ],
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "cases": len(cases)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
