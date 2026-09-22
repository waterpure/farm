"""Read-only diagnosis of why selected V45 animal routes differ in cash.

This is an audit tool, not a strategy variant.  It compares raw frozen route
tapes and replays only already-observed losing development conditions.  No
agent action is changed and no Kaggle submission is involved.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from kaggle_environments import make

from .baselines import load_external_agent, load_v45_base_agent, load_v45_base_module
from .portfolio_model import ANIMAL_OUTPUT
from .v45_variants import make_animal_route_selector_agent


ROUTES = (7, 105, 108, 123)
# One seat is sufficient for the diagnosis: the paired ledger confirmed these
# same seed outcomes on both seats.  Keep the expected cash here so a future
# environment/version drift fails loudly rather than becoming a new fact.
CASES = (
    {"seed": 1, "original_route": 108, "candidate_reward": 159_325.0, "baseline_reward": 158_719.0},
    {"seed": 2, "original_route": 105, "candidate_reward": 123_169.0, "baseline_reward": 127_741.0},
    {"seed": 3, "original_route": 7, "candidate_reward": 105_973.0, "baseline_reward": 128_752.0},
)


def _commands(action: dict[str, Any]) -> list[list[Any]]:
    return [list(action.get("farmer") or ["PASS"]), *(list(command) for command in action.get("hands") or [])]


def _route_day_profile(tape: list[dict[str, Any]], day: int) -> dict[str, Any]:
    market: Counter[str] = Counter()
    units: Counter[str] = Counter()
    commands: Counter[str] = Counter()
    details: list[list[Any]] = []
    for action in tape[day * 24 : (day + 1) * 24]:
        for order in action.get("market") or []:
            if not order:
                continue
            op = str(order[0])
            item = str(order[1]) if len(order) > 1 else ""
            quantity = int(order[2]) if len(order) > 2 else 1
            market[f"{op}:{item}"] += 1
            units[f"{op}:{item}"] += quantity
        for command in _commands(action):
            op = str(command[0]) if command else "PASS"
            commands[op] += 1
            if op in {"PLANT", "BUILD_COOP", "BUILD_PASTURE", "PLACE", "PICKUP", "FEED", "CARE", "HARVEST", "COLLECT_FERTILIZER"}:
                details.append(command)
    return {
        "market_orders": dict(sorted(market.items())),
        "market_units": dict(sorted(units.items())),
        "commands": dict(sorted(commands.items())),
        "field_details": details,
    }


def static_route_differences() -> dict[str, Any]:
    module = load_v45_base_module()
    tapes = module._IMPL.chassis.routes
    profiles = {route: [_route_day_profile(tapes[route], day) for day in range(6, 27)] for route in ROUTES}
    differences: dict[str, list[dict[str, Any]]] = {}
    for original in (7, 105, 108):
        rows: list[dict[str, Any]] = []
        for day, (left, right) in enumerate(zip(profiles[original], profiles[123]), start=6):
            if left != right:
                rows.append({"day": day, f"route_{original}": left, "route_123": right})
        differences[f"{original}_vs_123"] = rows
    return {"routes": list(ROUTES), "day_profiles": {str(route): profiles[route] for route in ROUTES}, "differences": differences}


def _tile_counts(farm: dict[str, Any]) -> dict[str, dict[str, int]]:
    animals: Counter[str] = Counter()
    crops: Counter[str] = Counter()
    structures: Counter[str] = Counter()
    for row in farm.get("tiles") or []:
        for tile in row:
            if not isinstance(tile, dict):
                continue
            if tile.get("animal") in ANIMAL_OUTPUT:
                animals[ANIMAL_OUTPUT[tile["animal"]]] += 1
            if tile.get("crop"):
                crops[str(tile["crop"])] += 1
            if tile.get("kind"):
                structures[str(tile["kind"])] += 1
    return {"animals": dict(sorted(animals.items())), "crops": dict(sorted(crops.items())), "structures": dict(sorted(structures.items()))}


def _daily_trace(environment: Any, player: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for day in range(30):
        end_step = min(719, day * 24 + 23)
        state = environment.steps[end_step][player]
        observation = state.observation
        farm = observation["farms"][player]
        private = observation["private"]
        orders: Counter[str] = Counter()
        order_units: Counter[str] = Counter()
        for offset in range(day * 24, min(720, (day + 1) * 24)):
            for order in environment.steps[offset][player].action.get("market") or []:
                if not order:
                    continue
                key = f"{order[0]}:{order[1] if len(order) > 1 else ''}"
                orders[key] += 1
                order_units[key] += int(order[2]) if len(order) > 2 else 1
        rows.append(
            {
                "day": day,
                "money": float(farm.get("money", 0)),
                "hands": len(farm.get("hands") or []),
                "unlocked_quadrants": list(farm.get("unlocked_quadrants") or []),
                **_tile_counts(farm),
                "shed": dict(sorted(private.get("shed", {}).items())),
                "seeds": dict(sorted(private.get("seeds", {}).items())),
                "market_orders": dict(sorted(orders.items())),
                "market_units": dict(sorted(order_units.items())),
                "prices": dict(sorted(observation.get("market", {}).get("prices", {}).items())),
            }
        )
    return rows


def _run(policy: str, seed: int) -> tuple[Any, float, dict[str, Any]]:
    left = make_animal_route_selector_agent() if policy == "candidate" else load_v45_base_agent()
    environment = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed}, debug=True)
    environment.run([left, load_external_agent("boatlee_v16")])
    final = environment.steps[-1][0]
    return environment, float(final.reward), dict(getattr(left, "telemetry", {}))


def replay_cases() -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for case in CASES:
        candidate_env, candidate_reward, telemetry = _run("candidate", case["seed"])
        baseline_env, baseline_reward, _ = _run("baseline", case["seed"])
        if candidate_reward != case["candidate_reward"] or baseline_reward != case["baseline_reward"]:
            raise RuntimeError(
                f"seed {case['seed']} replay drift: candidate {candidate_reward} / baseline {baseline_reward}; "
                f"expected {case['candidate_reward']} / {case['baseline_reward']}"
            )
        candidate_days = _daily_trace(candidate_env, 0)
        baseline_days = _daily_trace(baseline_env, 0)
        changed_days = [
            {
                "day": candidate["day"],
                "candidate": candidate,
                "baseline": baseline,
                "money_delta": round(candidate["money"] - baseline["money"], 3),
            }
            for candidate, baseline in zip(candidate_days, baseline_days)
            if candidate != baseline
        ]
        reports.append(
            {
                **case,
                "delta_vs_v45": candidate_reward - baseline_reward,
                "selector_event": telemetry.get("selector_events", [{}])[0],
                "changed_days": changed_days,
            }
        )
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("experiments/v45_animal_route_diff_audit_v1.json"))
    args = parser.parse_args()
    report = {"static": static_route_differences(), "replays": replay_cases()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "replays": len(report["replays"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
