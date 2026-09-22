"""Observe V45's current-day residual resources without changing its actions."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from kaggle_environments import make

from .baselines import load_external_agent, load_v45_base_module
from .daily_planner import plan_day
from .diagnostics import reference_rewards
from .league import DEVELOPMENT_SEEDS, OPPONENTS, SEATS


PROTOCOL = "v45-portfolio-capacity-audit-v3-multiday-tiles"
DEFAULT_OUTPUT = Path("experiments/v45_portfolio_capacity_audit_v3.jsonl")
SEED_COST = {"WHEAT": 10, "CARROT": 20, "TOMATO": 50, "STRAWBERRY": 100, "MELON": 80}
ANIMAL_COST = {"GOOSE": 300, "COW": 400, "SHEEP": 500}


def _fib(n: int) -> int:
    first, second = 1, 1
    for _ in range(max(0, n)):
        first, second = second, first + second
    return first


def _current_order_cost(observation: dict[str, Any], action: dict[str, Any]) -> int:
    """Conservative cash committed by this callback's parent market orders."""

    farm = observation["farms"][int(observation["player"])]
    prices = observation["market"]["prices"]
    hires = int(farm.get("hires_today", 0))
    cost = 0
    for order in action.get("market", []):
        if not order:
            continue
        op = order[0]
        if op == "HIRE":
            # Official escalating hand price, matching V45's own helper.
            cost += _fib(hires)
            hires += 1
        elif op == "BUY_LAND":
            cost += 4000
        elif len(order) >= 3:
            item, quantity = str(order[1]), max(0, int(order[2]))
            if op == "BUY_SEED":
                cost += SEED_COST.get(item, 0) * quantity
            elif op == "BUY_ANIMAL":
                cost += ANIMAL_COST.get(item, 0) * quantity
            elif op == "BUY_PRODUCT":
                cost += int(prices.get(item, 0)) * quantity
    return cost


def _empty_owned_tiles(farm: dict[str, Any]) -> int:
    return sum(tile is None for row in farm.get("tiles") or [] for tile in row)


def _empty_owned_positions(farm: dict[str, Any]) -> list[list[int]]:
    return [[x, y] for y, row in enumerate(farm.get("tiles") or []) for x, tile in enumerate(row) if tile is None]


class ObservedV45:
    def __init__(self) -> None:
        self.module = load_v45_base_module()
        self.policy = self.module.agent
        self.daily: list[dict[str, Any]] = []

    def __call__(self, observation: dict[str, Any], configuration: dict[str, Any] | None = None) -> dict[str, Any]:
        action = self.policy(observation, configuration)
        # Hour 1 sees hands hired at hour 0. It is the earliest non-speculative
        # point for today's observed worker capacity.
        if int(observation.get("hour", 0)) == 1:
            player = int(observation["player"])
            farm = observation["farms"][player]
            private = observation["private"]
            day_plan = plan_day(observation)
            reserve = day_plan["reservation"]
            carried_wheat = sum(int(inv.get("WHEAT", 0)) for inv in private.get("inventories", []))
            self.daily.append(
                {
                    "day": int(observation["day"]),
                    "route": self.module._IMPL.chassis.players.get(player, {}).get("route"),
                    "empty_owned_tiles_before_new_work": _empty_owned_tiles(farm),
                    "empty_owned_positions_before_new_work": _empty_owned_positions(farm),
                    "money_before_market": int(farm.get("money", 0)),
                    "parent_current_market_cost": _current_order_cost(observation, action),
                    "cash_after_current_parent_orders_lower_bound": max(0, int(farm.get("money", 0)) - _current_order_cost(observation, action)),
                    "wheat_in_shed_and_hands": int(private.get("shed", {}).get("WHEAT", 0)) + carried_wheat,
                    "wheat_reserved_for_existing_feed": reserve["wheat_for_feed"],
                    "shed_free_before_new_work": reserve["shed_free_before_new_work"],
                    "mandatory_route_witnessed": day_plan["mandatory_route_witness"]["all_mandatory_tasks_witnessed"],
                    "current_day_worker_turns_after_mandatory_witness": reserve["remaining_worker_turns_after_route_witness"],
                }
            )
        return action


def _conditions() -> list[dict[str, Any]]:
    return [{"opponent": o, "seed": s, "candidate_seat": seat} for o in OPPONENTS for s in DEVELOPMENT_SEEDS for seat in SEATS]


def _key(row: dict[str, Any]) -> tuple[str, int, str]:
    return str(row["opponent"]), int(row["seed"]), str(row["candidate_seat"])


def collect(output: Path, max_games: int) -> dict[str, Any]:
    existing = {_key(json.loads(line)) for line in output.read_text(encoding="utf-8").splitlines()} if output.exists() else set()
    expected = reference_rewards()
    rows: list[dict[str, Any]] = []
    for condition in _conditions():
        if _key(condition) in existing:
            continue
        observed, opponent = ObservedV45(), load_external_agent(condition["opponent"])
        left, right = (observed, opponent) if condition["candidate_seat"] == "left" else (opponent, observed)
        environment = make("kaggriculture", configuration={"episodeSteps": 720, "seed": condition["seed"]}, debug=True)
        started = time.perf_counter(); environment.run([left, right]); terminal = environment.steps[-1]
        rewards = tuple(float(state.reward) for state in terminal); index = 0 if condition["candidate_seat"] == "left" else 1
        actual = (rewards[index], rewards[1 - index])
        row = {"protocol": PROTOCOL, **condition, "elapsed_seconds": round(time.perf_counter() - started, 3), "statuses": [s.status for s in terminal], "terminal_matches_frozen_league": expected.get(_key(condition)) == actual, "daily_residuals": observed.daily}
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        rows.append(row)
        if len(rows) >= max_games:
            break
    return {"protocol": PROTOCOL, "new_games": len(rows), "remaining_games": len(_conditions()) - len(existing) - len(rows), "all_new_games_match_reference": all(row["terminal_matches_frozen_league"] for row in rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-games", type=int, default=1)
    args = parser.parse_args(); args.output.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps(collect(args.output, args.max_games), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
