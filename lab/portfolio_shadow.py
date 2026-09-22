"""Run the 70/30 portfolio and daily-planner proposal beside frozen V45.

The observer records one advisory plan each game day while returning V45's exact
action unchanged.  It is an integration/legality check for our own planning
modules, not an agent benchmark and not a V45 variant.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from kaggle_environments import make

from .baselines import load_external_agent, load_v45_base_agent
from .daily_planner import plan_day
from .diagnostics import reference_rewards
from .league import DEVELOPMENT_SEEDS, OPPONENTS, SEATS


# v2.4 reads actual post-HIRE workers. v2.5 keeps its evidence immutable while
# fixing same-priority task placement to select the globally earliest finish.
PROTOCOL = "portfolio-daily-shadow-v2.5-dynamic-route-witness"
DEFAULT_OUTPUT = Path("experiments/portfolio_daily_shadow_v2_5.jsonl")


class _ObservedV45:
    def __init__(self) -> None:
        self.policy = load_v45_base_agent()
        self.daily_plans: list[dict[str, Any]] = []

    def __call__(self, observation: dict[str, Any], configuration: dict[str, Any] | None = None) -> dict[str, Any]:
        # The prior callback's market HIRE orders have already resolved by hour
        # one, so this uses actual hands rather than forecasting an order's
        # affordability or spawn position.
        if int(observation.get("hour", 0)) == 1:
            self.daily_plans.append(plan_day(observation))
        return self.policy(observation, configuration)


def planned_conditions() -> list[dict[str, Any]]:
    return [{"opponent": opponent, "seed": seed, "candidate_seat": seat} for opponent in OPPONENTS for seed in DEVELOPMENT_SEEDS for seat in SEATS]


def _key(row: dict[str, Any]) -> tuple[str, int, str]:
    return str(row["opponent"]), int(row["seed"]), str(row["candidate_seat"])


def _completed(path: Path) -> set[tuple[str, int, str]]:
    if not path.exists():
        return set()
    return {_key(row) for line in path.read_text(encoding="utf-8").splitlines() if (row := json.loads(line)).get("protocol") == PROTOCOL}


def _run(condition: dict[str, Any], expected: tuple[float, float] | None) -> dict[str, Any]:
    candidate = _ObservedV45()
    opponent = load_external_agent(str(condition["opponent"]))
    left, right = (candidate, opponent) if condition["candidate_seat"] == "left" else (opponent, candidate)
    environment = make("kaggriculture", configuration={"episodeSteps": 720, "seed": condition["seed"]}, debug=True)
    started = time.perf_counter()
    environment.run([left, right])
    terminal = environment.steps[-1]
    rewards = tuple(float(state.reward) for state in terminal)
    index = 0 if condition["candidate_seat"] == "left" else 1
    observed = (rewards[index], rewards[1 - index])
    return {
        "protocol": PROTOCOL,
        **condition,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "statuses": [state.status for state in terminal],
        "terminal_matches_frozen_league": expected is not None and observed == expected,
        "daily_plans": candidate.daily_plans,
    }


def collect(output: Path, max_games: int) -> dict[str, Any]:
    expected = reference_rewards()
    done = _completed(output)
    new: list[dict[str, Any]] = []
    for condition in planned_conditions():
        if _key(condition) in done:
            continue
        row = _run(condition, expected.get(_key(condition)))
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        new.append(row)
        if len(new) >= max_games:
            break
    return {
        "protocol": PROTOCOL,
        "new_games": len(new),
        "remaining_games": len(planned_conditions()) - len(done) - len(new),
        "all_new_games_match_reference": all(row["terminal_matches_frozen_league"] for row in new),
    }


def summarize(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    valid = [row for row in rows if row.get("protocol") == PROTOCOL and row.get("terminal_matches_frozen_league")]
    primary, secondary, first_two_signatures, first_two_primary = Counter(), Counter(), Counter(), {}
    route_witnessed = route_unwitnessed = route_unassigned_groups = route_remaining_turns = 0
    plan_count = 0
    for row in valid:
        for daily in row["daily_plans"]:
            plan_count += 1
            portfolio = daily["portfolio"]
            if portfolio["primary"]:
                primary[portfolio["primary"]["line"]] += 1
            if portfolio["secondary"]:
                secondary[portfolio["secondary"]["line"]] += 1
            witness = daily["mandatory_route_witness"]
            route_witnessed += int(witness["all_mandatory_tasks_witnessed"])
            route_unwitnessed += int(not witness["all_mandatory_tasks_witnessed"])
            route_unassigned_groups += len(witness["unassigned_groups"])
            route_remaining_turns += int(witness["remaining_worker_turns_after_witness"])
            signature = tuple(portfolio["shops"][:2])
            first_two_signatures[signature] += 1
            bucket = first_two_primary.setdefault(" | ".join(signature) or "no shops", Counter())
            bucket[portfolio["primary"]["line"] if portfolio["primary"] else "none"] += 1
    return {
        "protocol": PROTOCOL,
        "source": str(path),
        "accepted_conditions": len(valid),
        "planned_conditions": len(planned_conditions()),
        "daily_plans_recorded": plan_count,
        "primary_line_counts": dict(primary),
        "secondary_line_counts": dict(secondary),
        "mandatory_route_witness": {
            "witnessed_daily_plans": route_witnessed,
            "greedy_unwitnessed_daily_plans": route_unwitnessed,
            "unassigned_task_groups": route_unassigned_groups,
            "mean_remaining_worker_turns_after_witness": round(route_remaining_turns / plan_count, 2) if plan_count else None,
        },
        "first_two_shop_signature_counts": {" | ".join(signature) or "no shops": count for signature, count in first_two_signatures.items()},
        "primary_by_first_two_shops": {signature: dict(counts) for signature, counts in first_two_primary.items()},
        "limits": [
            "The observer returns V45's action unchanged; no portfolio target becomes a game action.",
            "Scores are transparent priority heuristics, not a claim of predicted cash or a league improvement.",
            "Routing, action selection, and 70/30 execution require a separately reviewed next stage.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-games", type=int, default=1)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--summary-output", type=Path, default=Path("experiments/portfolio_daily_shadow_v2_5_summary.json"))
    args = parser.parse_args()
    if args.summarize:
        result = summarize(args.output)
        args.summary_output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result = collect(args.output, args.max_games)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
