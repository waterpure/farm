"""Observe V45 investment decisions beside the 70/30 model without changing V45.

This is deliberately an observer, not a candidate policy.  It returns the
frozen parent's exact action and records only the final market investment
orders V45 chose to issue.  The record is the evidence needed before claiming
that a V45 action is a discretionary slot our portfolio layer may touch.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from kaggle_environments import make

from .baselines import load_external_agent, load_v45_base_module
from .diagnostics import reference_rewards
from .league import DEVELOPMENT_SEEDS, OPPONENTS, SEATS
from .portfolio_model import choose_portfolio


PROTOCOL = "v45-portfolio-opportunity-audit-v1"
DEFAULT_OUTPUT = Path("experiments/v45_portfolio_opportunity_audit_v1.jsonl")
INVESTMENT_OPS = frozenset({"BUY_LAND", "BUY_SEED", "BUY_ANIMAL", "HIRE"})


class ObservedV45:
    """Fresh V45 parent with immutable-action investment telemetry."""

    def __init__(self) -> None:
        self.module = load_v45_base_module()
        self.policy = self.module.agent
        self.investment_events: list[dict[str, Any]] = []

    def __call__(self, observation: dict[str, Any], configuration: dict[str, Any] | None = None) -> dict[str, Any]:
        action = self.policy(observation, configuration)
        orders = [list(order) for order in action.get("market", []) if order and order[0] in INVESTMENT_OPS]
        if orders:
            player = int(observation["player"])
            route_state = self.module._IMPL.chassis.players.get(player, {})
            recommendation = choose_portfolio(observation)
            self.investment_events.append(
                {
                    "step": int(observation["step"]),
                    "day": int(observation["day"]),
                    "hour": int(observation["hour"]),
                    "route": route_state.get("route"),
                    "shops": list((observation.get("town") or {}).get("unlocked_shops") or []),
                    "money_before_market": float(observation["farms"][player].get("money", 0)),
                    "final_market_investment_orders": orders,
                    "portfolio_allocation": recommendation["allocation"],
                    "portfolio_primary": recommendation["primary"],
                    "portfolio_secondary": recommendation["secondary"],
                }
            )
        return action


def planned_conditions() -> list[dict[str, Any]]:
    return [
        {"opponent": opponent, "seed": seed, "candidate_seat": seat}
        for opponent in OPPONENTS
        for seed in DEVELOPMENT_SEEDS
        for seat in SEATS
    ]


def _key(row: dict[str, Any]) -> tuple[str, int, str]:
    return str(row["opponent"]), int(row["seed"]), str(row["candidate_seat"])


def _completed(path: Path) -> set[tuple[str, int, str]]:
    if not path.exists():
        return set()
    result: set[tuple[str, int, str]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("protocol") == PROTOCOL:
            result.add(_key(row))
    return result


def _run(condition: dict[str, Any], expected: tuple[float, float] | None) -> dict[str, Any]:
    observed = ObservedV45()
    opponent = load_external_agent(str(condition["opponent"]))
    left, right = (observed, opponent) if condition["candidate_seat"] == "left" else (opponent, observed)
    environment = make("kaggriculture", configuration={"episodeSteps": 720, "seed": condition["seed"]}, debug=True)
    started = time.perf_counter()
    environment.run([left, right])
    terminal = environment.steps[-1]
    rewards = tuple(float(state.reward) for state in terminal)
    candidate_index = 0 if condition["candidate_seat"] == "left" else 1
    actual = (rewards[candidate_index], rewards[1 - candidate_index])
    return {
        "protocol": PROTOCOL,
        **condition,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "statuses": [state.status for state in terminal],
        "terminal_matches_frozen_league": expected is not None and actual == expected,
        "investment_events": observed.investment_events,
    }


def collect(output: Path, max_games: int) -> dict[str, Any]:
    expected = reference_rewards()
    completed = _completed(output)
    new_rows: list[dict[str, Any]] = []
    for condition in planned_conditions():
        if _key(condition) in completed:
            continue
        row = _run(condition, expected.get(_key(condition)))
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        new_rows.append(row)
        if len(new_rows) >= max_games:
            break
    return {
        "protocol": PROTOCOL,
        "new_games": len(new_rows),
        "remaining_games": len(planned_conditions()) - len(completed) - len(new_rows),
        "all_new_games_match_reference": all(row["terminal_matches_frozen_league"] for row in new_rows),
    }


def summarize(output: Path) -> dict[str, Any]:
    """Aggregate observed parent orders without treating recommendations as actions."""

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in rows if row.get("protocol") == PROTOCOL]
    valid = [row for row in rows if row.get("terminal_matches_frozen_league")]
    orders, by_day, by_route, primary_lines = Counter(), Counter(), Counter(), Counter()
    matching_primary_orders = total_product_orders = 0
    animal_output = {"GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL"}
    for row in valid:
        for event in row["investment_events"]:
            route = str(event.get("route"))
            by_route[route] += 1
            primary = event.get("portfolio_primary") or {}
            primary_output = primary.get("output")
            if primary.get("line"):
                primary_lines[str(primary["line"])] += 1
            for order in event["final_market_investment_orders"]:
                op = str(order[0])
                item = str(order[1]) if len(order) > 1 else ""
                quantity = int(order[2]) if len(order) > 2 else 1
                label = f"{op} {item}".rstrip()
                orders[label] += quantity
                by_day[f"day {event['day']}: {label}"] += quantity
                if op == "BUY_SEED":
                    actual_output = item
                elif op == "BUY_ANIMAL":
                    actual_output = animal_output.get(item)
                else:
                    actual_output = None
                if actual_output is not None:
                    total_product_orders += quantity
                    matching_primary_orders += quantity * int(actual_output == primary_output)
    return {
        "protocol": PROTOCOL,
        "source": str(output),
        "conditions_recorded": len(rows),
        "conditions_strictly_reproduced": len(valid),
        "parent_investment_event_callbacks": sum(by_route.values()),
        "order_units": dict(orders),
        "order_units_by_day": dict(sorted(by_day.items())),
        "event_callbacks_by_route": dict(sorted(by_route.items())),
        "portfolio_primary_at_investment_events": dict(primary_lines),
        "product_order_units_matching_current_primary": matching_primary_orders,
        "product_order_units_observed": total_product_orders,
        "limits": [
            "This reports the final V45 action after all V45 wrappers; it does not prove which internal wrapper produced an order.",
            "A match or mismatch with the portfolio heuristic is an observation, not evidence that replacing that order helps.",
            "No parent action was changed during this audit.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-games", type=int, default=1)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--summary-output", type=Path, default=Path("experiments/v45_portfolio_opportunity_audit_v1_summary.json"))
    args = parser.parse_args()
    if args.summarize:
        summary = summarize(args.output)
        args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        print(json.dumps(collect(args.output, args.max_games), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
