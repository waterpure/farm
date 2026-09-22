"""Read-only known-shop demand audit for the failed V45 animal-route switches.

For a replay, count products actually harvested by our farm after the day-6
route choice, then compare that with the amount *currently unlocked* shops
would buy during the rest of the season.  Future shops are intentionally not
included: they are unknown at the moment a route must be chosen.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from kaggle_environments import make

from .baselines import load_external_agent, load_v45_base_agent
from .portfolio_model import PRODUCTS, shop_demand_per_day
from .v45_variants import make_animal_route_selector_agent


CASES = (
    {"seed": 1, "original_route": 108, "candidate_reward": 159_325.0, "baseline_reward": 158_719.0},
    {"seed": 2, "original_route": 105, "candidate_reward": 123_169.0, "baseline_reward": 127_741.0},
    {"seed": 3, "original_route": 7, "candidate_reward": 105_973.0, "baseline_reward": 128_752.0},
)
DAY6_STEP = 144
TERMINAL_STEP = 720


def _run(policy: str, seed: int) -> tuple[Any, float, dict[str, Any]]:
    agent = make_animal_route_selector_agent() if policy == "candidate" else load_v45_base_agent()
    environment = make("kaggriculture", configuration={"episodeSteps": TERMINAL_STEP, "seed": seed}, debug=True)
    environment.run([agent, load_external_agent("boatlee_v16")])
    return environment, float(environment.steps[-1][0].reward), dict(getattr(agent, "telemetry", {}))


def _harvested_after_day6(environment: Any, player: int = 0) -> dict[str, int]:
    """Count positive inventory deltas caused by actual HARVEST commands.

    A unit is counted only if the acting farmer/hand issued HARVEST and their
    own next observation shows that product increasing.  Market buys/sells and
    PICKUP/DROP therefore cannot be mistaken for farm production.
    """

    harvests: Counter[str] = Counter()
    for step in range(DAY6_STEP, TERMINAL_STEP - 1):
        state = environment.steps[step][player]
        following = environment.steps[step + 1][player]
        before = state.observation["private"]["inventories"]
        after = following.observation["private"]["inventories"]
        # kaggle-environments stores the action that advanced state ``step``
        # to ``step + 1`` beside the following state, not beside the preceding
        # observation.  Using ``state.action`` here silently observes the
        # previous command and records zero real harvests.
        commands = [following.action.get("farmer") or ["PASS"], *(following.action.get("hands") or [])]
        for actor, command in enumerate(commands):
            if not command or command[0] != "HARVEST" or actor >= len(before) or actor >= len(after):
                continue
            for product in PRODUCTS:
                delta = int(after[actor].get(product, 0)) - int(before[actor].get(product, 0))
                if delta > 0:
                    harvests[product] += delta
    return dict(sorted(harvests.items()))


def _known_shop_capacity(observation: dict[str, Any]) -> dict[str, Any]:
    day = int(observation["day"])
    # The decision is at the first callback of day 6, so days 6..29 inclusive
    # remain.  Already-open shops remain open; future shops are omitted.
    remaining_days = 30 - day
    shops = list(observation["town"]["unlocked_shops"])
    per_day = {product: shop_demand_per_day(shops, product) for product in PRODUCTS}
    return {
        "shops": shops,
        "decision_day": day,
        "remaining_days": remaining_days,
        "known_shop_buy_per_day": per_day,
        "known_shop_buy_remaining": {product: int(per_day[product] * remaining_days) for product in PRODUCTS},
    }


def _audit_case(case: dict[str, Any]) -> dict[str, Any]:
    candidate_env, candidate_reward, telemetry = _run("candidate", case["seed"])
    baseline_env, baseline_reward, _ = _run("baseline", case["seed"])
    if candidate_reward != case["candidate_reward"] or baseline_reward != case["baseline_reward"]:
        raise RuntimeError(f"Replay drift for seed {case['seed']}")
    capacity = _known_shop_capacity(baseline_env.steps[DAY6_STEP][0].observation)
    candidate_harvests = _harvested_after_day6(candidate_env)
    baseline_harvests = _harvested_after_day6(baseline_env)
    rows = []
    for product in PRODUCTS:
        limit = capacity["known_shop_buy_remaining"][product]
        rows.append(
            {
                "product": product,
                "known_shop_capacity": limit,
                "candidate_harvested": candidate_harvests.get(product, 0),
                "baseline_harvested": baseline_harvests.get(product, 0),
                "candidate_over_known_capacity": candidate_harvests.get(product, 0) - limit,
                "baseline_over_known_capacity": baseline_harvests.get(product, 0) - limit,
            }
        )
    return {
        **case,
        "delta_vs_v45": candidate_reward - baseline_reward,
        "selector_event": telemetry.get("selector_events", [{}])[0],
        **capacity,
        "products": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("experiments/v45_route_supply_demand_audit_v1.json"))
    args = parser.parse_args()
    report = {"protocol": "known-shop-supply-demand-audit-v3-transition-aligned", "cases": [_audit_case(case) for case in CASES]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "cases": len(report["cases"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
