"""Audit which V45 raw-route crop contracts survive its outer safety layers.

The observer never alters V45 actions.  At a raw-tape PLANT it projects only
the public V45 tapes: before day 6 the current tape is fixed; at day 6 it
checks every route that the router itself could select; at day 27 it uses the
fixed terminal route.  A contract is valid only if every such tape branch
keeps the actor on the same tile for the crop's useful WATER/HARVEST work and
contains a sufficiently large SELL order.  The actual match then tells us
whether V45's outer layers emitted that planned PLANT or removed it.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from kaggle_environments import make

from .baselines import load_external_agent, load_v45_base_module
from .diagnostics import reference_rewards
from .latent_slot_audit import CROPS
from .league import DEVELOPMENT_SEEDS, OPPONENTS, SEATS


PROTOCOL = "v45-route-commitment-audit-v4-suppression-focus"
DEFAULT_OUTPUT = Path("experiments/v45_route_commitment_audit_v4.jsonl")
DAY6_STEP = 144
DAY27_STEP = 648
LAST_STEP = 718
MOVES = {"NORTH": (0, -1), "SOUTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0)}


def _commands(action: dict[str, Any], count: int) -> list[list[Any]]:
    commands = [list(action.get("farmer") or ["PASS"])] + [list(command or ["PASS"]) for command in action.get("hands") or []]
    return commands[:count] + [["PASS"] for _ in range(max(0, count - len(commands)))]


def _future_day6_routes(module: Any) -> tuple[int, ...]:
    # Both branches of V45's own router plus their documented fallbacks.
    choices = set(module._R108_SHOP_ROUTES.values()) | set(module._R110_OLD_SHOPS.values()) | {0, 100}
    return tuple(sorted(route for route in choices if route in module._ROUTES))


def _route_at(step: int, current_route: int, day6_route: int) -> int:
    if step < DAY6_STEP:
        return current_route
    if step < DAY27_STEP:
        return day6_route
    return 2


def _branch_contract(
    module: Any,
    *,
    step: int,
    current_route: int,
    day6_route: int,
    plant_actor: int,
    start_positions: list[tuple[int, int]],
    coordinate: tuple[int, int],
    crop: str,
) -> dict[str, Any]:
    """Project all existing workers through static V45 tapes; no future observation."""
    meta = CROPS[crop]
    planted_day = step // 24
    positions = [list(position) for position in start_positions]
    waters: list[dict[str, int]] = []
    harvest: int | None = None
    harvest_actor: int | None = None
    sell: int | None = None
    sell_quantity = 0
    for future_step in range(step + 1, LAST_STEP + 1):
        route = _route_at(future_step, current_route, day6_route)
        action = module._ROUTES[route][future_step]
        commands = _commands(action, len(positions))
        for actor, command in enumerate(commands):
            if tuple(positions[actor]) == coordinate:
                if command and command[0] == "WATER":
                    waters.append({"step": future_step, "actor": actor})
                elif command and command[0] == "HARVEST" and harvest is None:
                    age = future_step // 24 - planted_day
                    if meta["first"] <= age <= meta["last"]:
                        harvest, harvest_actor = future_step, actor
        if harvest is not None and sell is None:
            quantity = sum(
                max(0, int(order[2]))
                for order in action.get("market") or []
                if len(order) >= 3 and order[0] == "SELL" and order[1] == crop
            )
            if quantity:
                sell, sell_quantity = future_step, quantity
        for actor, command in enumerate(commands):
            if command and command[0] in MOVES:
                dx, dy = MOVES[command[0]]
                positions[actor][0] += dx
                positions[actor][1] += dy
        if harvest is not None and sell is not None:
            break
    qualifying = sum(
        1
        for water in waters
        if planted_day + (meta["last"] + 1) // 2 <= water["step"] // 24 <= planted_day + meta["last"]
        and harvest is not None
        and water["step"] < harvest
    )
    expected_yield = min(meta["cap"], 1 + qualifying)
    immediate_water = any(water["step"] <= step + 2 and water["actor"] == plant_actor for water in waters)
    valid = immediate_water and harvest is not None and sell is not None and sell_quantity >= expected_yield
    return {
        "day6_route": day6_route,
        "valid": valid,
        "immediate_water": immediate_water,
        "water_steps": waters,
        "qualifying_water_count": qualifying,
        "expected_yield_units": expected_yield,
        "harvest_step": harvest,
        "harvest_actor": harvest_actor,
        "sell_step": sell,
        "sell_quantity": sell_quantity,
    }


def static_contract(module: Any, *, step: int, current_route: int, actor: int, positions: list[tuple[int, int]], crop: str) -> dict[str, Any]:
    """Return only a contract supported in every V45 route branch still possible now."""
    if step < DAY6_STEP:
        branches = _future_day6_routes(module)
    elif step < DAY27_STEP:
        branches = (current_route,)
    else:
        branches = (2,)
    reports = [
        _branch_contract(
            module,
            step=step,
            current_route=current_route,
            day6_route=branch,
            plant_actor=actor,
            start_positions=positions,
            coordinate=positions[actor],
            crop=crop,
        )
        for branch in branches
    ]
    return {
        "all_possible_day6_routes": list(branches),
        "all_branches_supported": bool(reports) and all(report["valid"] for report in reports),
        "branch_reports": reports,
    }


class _ObservedV45:
    def __init__(self) -> None:
        self.module = load_v45_base_module()
        self.policy = self.module.agent
        self.commitments: list[dict[str, Any]] = []

    def __call__(self, observation: dict[str, Any], configuration: dict[str, Any] | None = None) -> dict[str, Any]:
        player = int(observation["player"])
        step = int(observation["step"])
        before_r124 = copy.deepcopy(getattr(self.module, "_R124_REPORT", {}))
        before_r148 = copy.deepcopy(getattr(self.module, "_R148_REPORT", {}))
        action = self.policy(observation, configuration)
        after_r124 = getattr(self.module, "_R124_REPORT", {})
        after_r148 = getattr(self.module, "_R148_REPORT", {})
        farm = observation["farms"][player]
        positions = [tuple(farm["farmer"]), *[tuple(position) for position in farm["hands"]]]
        state = self.module._IMPL.chassis.players.get(player, {})
        route = int(state.get("route", 0))
        raw = self.module._IMPL.chassis.routes[route][step]
        raw_commands = _commands(raw, len(positions))
        emitted_commands = _commands(action, len(positions))
        r124_dropped = int(after_r124.get("opening_atomic_dropped", 0)) - int(before_r124.get("opening_atomic_dropped", 0))
        r148_removed = int(after_r148.get("atomic_removed_requests", 0)) - int(before_r148.get("atomic_removed_requests", 0))
        for actor, raw_command in enumerate(raw_commands):
            if len(raw_command) < 2 or raw_command[0] != "PLANT" or raw_command[1] not in CROPS:
                continue
            crop = raw_command[1]
            x, y = positions[actor]
            if farm["tiles"][y][x] is not None or int(observation["private"]["seeds"].get(crop, 0)) <= 0:
                continue
            emitted_raw_plant = emitted_commands[actor] == raw_command
            # Full cross-route projection is useful only when an existing raw
            # request was removed; emitted requests require no explanation here.
            contract = None if emitted_raw_plant else static_contract(
                self.module,
                step=step,
                current_route=route,
                actor=actor,
                positions=positions,
                crop=crop,
            )
            self.commitments.append(
                {
                    "step": step,
                    "day": step // 24,
                    "route_at_decision": route,
                    "actor": actor,
                    "coordinate": [x, y],
                    "crop": crop,
                    "seed_available": int(observation["private"]["seeds"].get(crop, 0)),
                    "raw_command": raw_command,
                    "emitted_command": emitted_commands[actor],
                    "emitted_raw_plant": emitted_raw_plant,
                    "r124_opening_atomic_dropped_delta": r124_dropped,
                    "r148_atomic_removed_delta": r148_removed,
                    "static_contract": contract,
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
    return {
        _key(row)
        for line in path.read_text(encoding="utf-8").splitlines()
        if (row := json.loads(line)).get("protocol") == PROTOCOL and row.get("record_type") == "game"
    }


def _run(condition: dict[str, Any], expected: tuple[float, float] | None) -> dict[str, Any]:
    v45 = _ObservedV45()
    opponent = load_external_agent(str(condition["opponent"]))
    left, right = (v45, opponent) if condition["candidate_seat"] == "left" else (opponent, v45)
    environment = make("kaggriculture", configuration={"episodeSteps": 720, "seed": condition["seed"]}, debug=True)
    started = time.perf_counter()
    environment.run([left, right])
    terminal = environment.steps[-1]
    rewards = tuple(float(state.reward) for state in terminal)
    index = 0 if condition["candidate_seat"] == "left" else 1
    observed = (rewards[index], rewards[1 - index])
    return {
        "record_type": "game",
        "protocol": PROTOCOL,
        **condition,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "statuses": [state.status for state in terminal],
        "v45_reward": observed[0],
        "opponent_reward": observed[1],
        "expected_v45_reward": expected[0] if expected else None,
        "expected_opponent_reward": expected[1] if expected else None,
        "terminal_matches_frozen_league": expected is not None and observed == expected,
        "static_commitments": v45.commitments,
    }


def collect(output: Path, max_games: int) -> dict[str, Any]:
    expected = reference_rewards()
    done = _completed(output)
    completed = []
    for condition in planned_conditions():
        if _key(condition) in done:
            continue
        row = _run(condition, expected.get(_key(condition)))
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        completed.append(row)
        if len(completed) >= max_games:
            break
    return {"protocol": PROTOCOL, "new_games": len(completed), "remaining_games": len(planned_conditions()) - len(done) - len(completed), "all_new_games_match_reference": all(row["terminal_matches_frozen_league"] for row in completed)}


def summarize(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    valid = [row for row in rows if row.get("protocol") == PROTOCOL and row.get("terminal_matches_frozen_league")]
    outcomes = Counter()
    contract_status = Counter()
    suppressed_with_r124_drop = 0
    by_opponent: dict[str, Counter[str]] = defaultdict(Counter)
    sdy_target = []
    total = 0
    for row in valid:
        for item in row["static_commitments"]:
            total += 1
            outcome = "emitted" if item["emitted_raw_plant"] else "suppressed"
            outcomes[outcome] += 1
            if outcome == "suppressed" and item["r124_opening_atomic_dropped_delta"]:
                suppressed_with_r124_drop += 1
            by_opponent[row["opponent"]][outcome] += 1
            if item["static_contract"] is not None:
                contract_status["all_branches_supported" if item["static_contract"]["all_branches_supported"] else "branch_dependent"] += 1
            if row["opponent"] == "sdy2842" and item["step"] == 18 and item["coordinate"] == [0, 4] and item["crop"] == "MELON":
                sdy_target.append({"seed": row["seed"], "candidate_seat": row["candidate_seat"], "commitment": item})
    return {
        "protocol": PROTOCOL,
        "source": str(path),
        "accepted_conditions": len(valid),
        "planned_conditions": len(planned_conditions()),
        "raw_route_plant_events_with_current_seed": total,
        "contract_status": dict(contract_status),
        "outcomes": dict(outcomes),
        "suppressed_with_r124_opening_atomic_drop": suppressed_with_r124_drop,
        "outcomes_by_opponent": {opponent: dict(counts) for opponent, counts in sorted(by_opponent.items())},
        "sdy_step18_coordinate_0_4": sdy_target,
        "limits": [
            "Contracts are guaranteed only across V45 raw route tapes, not across every reactive outer-layer change.",
            "The audit observes whether outer layers emitted a raw plant; it does not change that decision or prove profit.",
            "A planned SELL order is a route commitment, not a guarantee of shared-market fill or price.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-games", type=int, default=1)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--summary-output", type=Path, default=Path("experiments/v45_route_commitment_audit_v4_summary.json"))
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
