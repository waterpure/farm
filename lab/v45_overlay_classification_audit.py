"""Classify V45's late sheep/tomato overlays from the day-6 decision point.

This is a read-only audit.  It does not alter a route or propose one: its
purpose is to prevent a day-6 forecast from treating later conditional V45
expansions as guaranteed output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from kaggle_environments import make

from .baselines import load_external_agent, load_v45_base_agent, load_v45_base_module
from .v45_variants import make_animal_route_selector_agent


CASES = ((1, 108, 123), (2, 105, 123), (3, 7, 123))


def _day6_observation(seed: int) -> dict[str, Any]:
    environment = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed}, debug=True)
    environment.run([load_v45_base_agent(), load_external_agent("boatlee_v16")])
    return environment.steps[144][0].observation


def _v233_static_route_block(module: Any, route: int) -> list[str]:
    """Return route-tape facts that make V233 impossible, regardless of day-12 state."""

    blocked: list[str] = []
    native = {"route": route}
    for day in range(12, 30):
        for action in module._v219_native_day(native, day):
            if any(order and order[0] == "BUY_LAND" for order in action.get("market", [])):
                blocked.append(f"day {day}: raw route itself buys land")
                return blocked
            if any(order and order[:2] == ["BUY_ANIMAL", "SHEEP"] for order in action.get("market", [])):
                blocked.append(f"day {day}: raw route itself buys sheep")
                return blocked
            commands = [action.get("farmer"), *(action.get("hands") or [])]
            if any(command and command[0] in ("PICKUP", "PLACE") and len(command) > 1 and command[1] == "SHEEP" for command in commands):
                blocked.append(f"day {day}: raw route itself moves sheep")
                return blocked
    return blocked


def _v219_static_schedule_block(module: Any) -> list[str]:
    """Return global schedule facts which make V219 impossible for every route."""

    for route, tape in module._IMPL.chassis.routes.items():
        for step, action in enumerate(tape[432:719], start=432):
            if any(order and order[0] == "BUY_LAND" for order in action.get("market", [])):
                return [f"route {route}, day {step // 24}: a V45 route buys land after day 18"]
            commands = [action.get("farmer"), *(action.get("hands") or [])]
            if any(command == ["PLANT", "TOMATO"] for command in commands):
                return [f"route {route}, day {step // 24}: a V45 route plants tomato after day 18"]
    return []


def _day6_snapshot(obs: dict[str, Any]) -> dict[str, Any]:
    player = int(obs["player"])
    farm = obs["farms"][player]
    return {
        "day": int(obs["day"]),
        "shops_now": list(obs["town"]["unlocked_shops"]),
        "prices_now": {key: int(obs["market"]["prices"][key]) for key in ("WOOL", "WHEAT", "TOMATO")},
        "money_now": int(farm["money"]),
        "unlocked_quadrants_now": sorted(farm["unlocked_quadrants"]),
        "tile_count_now": len(farm["tiles"]),
    }


def _classification(module: Any, obs: dict[str, Any], route: int, overlay: str) -> dict[str, Any]:
    """Classify without pretending the day-6 snapshot is a later-day observation."""

    if overlay == "v233_sheep":
        static = _v233_static_route_block(module, route)
        if static:
            return {
                "status": "certainly_impossible",
                "decision_day": 12,
                "reason": static,
                "forecast_treatment": "count_zero_extra_output",
            }
        return {
            "status": "future_uncertain",
            "decision_day": 12,
            "known_at_day6": _day6_snapshot(obs),
            "unknown_until_day12": [
                "whether exactly NW, NE, SW are unlocked and exactly 10 tiles exist",
                "whether at least two Yarn shops have opened",
                "then-current wool price is at least 220 and wheat price is at most 45",
                "cash can cover land, six sheep, wheat, two hires, plus the safety reserve",
                "shed capacity, market-order slots and native hire timing remain available",
            ],
            "forecast_treatment": "do_not_count_extra_six_sheep_as_guaranteed_output",
        }
    static = _v219_static_schedule_block(module)
    if static:
        return {
            "status": "certainly_impossible",
            "decision_day": 18,
            "reason": static,
            "forecast_treatment": "count_zero_extra_output",
        }
    return {
        "status": "future_uncertain",
        "decision_day": 18,
        "known_at_day6": _day6_snapshot(obs),
        "unknown_until_day18": [
            "whether exactly NW, NE, SW are unlocked and exactly 10 tiles exist",
            "cash is at least 12000 and tomato price is at least 70",
            "at least three Pizza/Farmers Market shops are then open",
            "the SE target cells are still empty and no tomato asset exists",
            "the dedicated hires, land, seeds, water, fertilizer and harvest actions can all execute",
        ],
        "forecast_treatment": "do_not_count_extra_tomatoes_as_guaranteed_output",
    }


def _actual_overlay_report(seed: int, policy: str) -> dict[str, int]:
    agent = load_v45_base_agent() if policy == "baseline" else make_animal_route_selector_agent()
    environment = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed}, debug=True)
    environment.run([agent, load_external_agent("boatlee_v16")])
    telemetry = getattr(agent, "telemetry", {})
    if "parent_telemetry" in telemetry:
        telemetry = telemetry["parent_telemetry"]
    keys = (
        "sheep_commit_requests", "sheep_committed", "sheep_wool_harvested",
        "commitments", "confirmed_plants", "confirmed_harvest_units",
    )
    return {key: int(telemetry.get(key, 0)) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("experiments/v45_overlay_classification_audit_v1.json"))
    args = parser.parse_args()
    module = load_v45_base_module()
    cases: list[dict[str, Any]] = []
    for seed, baseline_route, candidate_route in CASES:
        observation = _day6_observation(seed)
        row: dict[str, Any] = {"seed": seed, "day6": _day6_snapshot(observation), "policies": {}}
        for label, route in (("baseline", baseline_route), ("historical_wrong_selector", candidate_route)):
            row["policies"][label] = {
                "raw_route": route,
                "v233_sheep": _classification(module, observation, route, "v233_sheep"),
                "v219_tomato": _classification(module, observation, route, "v219_tomato"),
                "actual_final_telemetry": _actual_overlay_report(seed, "baseline" if label == "baseline" else "candidate"),
            }
        cases.append(row)
    report = {
        "protocol": "v45-overlay-classification-audit-v1-day6-known-vs-later-conditions",
        "scope": "Three historical route-selector switch cases against boatlee_v16; audit only.",
        "rule": "Only certainly-impossible overlays count as zero. Future-uncertain overlays must not be added to a day-6 point forecast as guaranteed supply.",
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "cases": len(cases)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
