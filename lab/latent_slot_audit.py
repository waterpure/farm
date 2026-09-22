"""Observe possible crop investments inside V45's existing future work slots.

This is a shadow analysis only: the wrapped V45 policy returns its unmodified
action.  After a match, the observer asks whether an idle actor on an empty
tile already has later V45 WATER, HARVEST, and SELL actions that could support
an owned seed.  It does not claim that the counterfactual plant would preserve
the same future trajectory or make money.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from kaggle_environments import make

from .baselines import load_external_agent, load_v45_base_module
from .diagnostics import reference_rewards
from .league import DEVELOPMENT_SEEDS, OPPONENTS, SEATS


PROTOCOL = "v45-latent-slot-dryrun-v3-immediate-service"
DEFAULT_OUTPUT = Path("experiments/v45_latent_slot_dryrun_v3.jsonl")
CROPS = {
    "WHEAT": {"seed": 10, "first": 2, "last": 4, "cap": 6, "base": 25},
    "CARROT": {"seed": 20, "first": 2, "last": 3, "cap": 4, "base": 35},
    "TOMATO": {"seed": 50, "first": 8, "last": 999, "cap": 4, "base": 60},
    "STRAWBERRY": {"seed": 100, "first": 10, "last": 999, "cap": 4, "base": 120},
    "MELON": {"seed": 80, "first": 10, "last": 12, "cap": 6, "base": 250},
}


def _commands(action: dict[str, Any], count: int) -> list[list[Any]]:
    commands = [list(action.get("farmer") or ["PASS"])] + [list(command or ["PASS"]) for command in action.get("hands") or []]
    return commands[:count] + [["PASS"] for _ in range(max(0, count - len(commands)))]


class _ObservedV45:
    """Fresh V45 module plus a compact observation/action journal."""

    def __init__(self) -> None:
        self.module = load_v45_base_module()
        self.policy = self.module.agent
        self.rows: list[dict[str, Any]] = []

    def __call__(self, observation: dict[str, Any], configuration: dict[str, Any] | None = None) -> dict[str, Any]:
        action = self.policy(observation, configuration)
        player = int(observation["player"])
        farm = observation["farms"][player]
        positions = [list(farm["farmer"]), *[list(position) for position in farm["hands"]]]
        self.rows.append(
            {
                "step": int(observation["step"]),
                "day": int(observation["step"]) // 24,
                "positions": positions,
                "commands": _commands(action, len(positions)),
                "market": copy.deepcopy(action.get("market") or []),
                "tiles": copy.deepcopy(farm["tiles"]),
                "seeds": {crop: int(observation["private"]["seeds"].get(crop, 0)) for crop in CROPS},
                "prices": {crop: int(observation["market"]["prices"].get(crop, 0)) for crop in CROPS},
            }
        )
        return action


def _future_window(rows: list[dict[str, Any]], start_index: int, coordinate: tuple[int, int], crop: str) -> dict[str, Any] | None:
    start = rows[start_index]
    planted_day = start["day"]
    meta = CROPS[crop]
    waters: list[int] = []
    harvest: int | None = None
    sell: int | None = None
    for future in rows[start_index + 1:]:
        for position, command in zip(future["positions"], future["commands"]):
            if tuple(position) != coordinate or not command:
                continue
            if command[0] == "WATER":
                waters.append(future["step"])
            elif command[0] == "HARVEST" and harvest is None:
                age = future["day"] - planted_day
                if age >= meta["first"] and age <= meta["last"]:
                    harvest = future["step"]
        if harvest is not None and sell is None:
            for command in future["market"]:
                if len(command) >= 2 and command[0] == "SELL" and command[1] == crop:
                    sell = future["step"]
        if harvest is not None and sell is not None:
            break
    if harvest is None or sell is None:
        return None
    qualifying_waters = sum(
        1 for step in waters
        if planted_day + (meta["last"] + 1) // 2 <= step // 24 <= planted_day + meta["last"] and step < harvest
    )
    expected_yield = min(meta["cap"], 1 + qualifying_waters)
    conservative_price = min(start["prices"].get(crop, meta["base"]), meta["base"])
    return {
        "water_steps": waters,
        "harvest_step": harvest,
        "sell_step": sell,
        "qualifying_water_count": qualifying_waters,
        "estimated_yield_units": expected_yield,
        "conservative_unit_price": conservative_price,
        "estimated_net_before_labor": expected_yield * conservative_price - meta["seed"],
    }


def latent_slots(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    slots = []
    for row_index, row in enumerate(rows):
        for actor, (position, command) in enumerate(zip(row["positions"], row["commands"])):
            x, y = position
            if command != ["PASS"] or row["tiles"][y][x] is not None:
                continue
            immediate_water_steps = [
                future["step"]
                for future in rows[row_index + 1:row_index + 3]
                if actor < len(future["positions"])
                and tuple(future["positions"][actor]) == (x, y)
                and future["commands"][actor]
                and future["commands"][actor][0] == "WATER"
            ]
            if not immediate_water_steps:
                continue
            supported = []
            for crop, quantity in row["seeds"].items():
                if quantity <= 0:
                    continue
                window = _future_window(rows, row_index, (x, y), crop)
                if window is None or window["estimated_net_before_labor"] <= 0:
                    continue
                supported.append({"crop": crop, "seed_available": quantity, **window})
            if supported:
                slots.append(
                    {
                        "step": row["step"],
                        "day": row["day"],
                        "actor": actor,
                        "coordinate": [x, y],
                        "same_actor_immediate_water_steps": immediate_water_steps,
                        "supported_crops": supported,
                    }
                )
    return slots


def planned_conditions() -> list[dict[str, Any]]:
    return [
        {"opponent": opponent, "seed": seed, "candidate_seat": seat}
        for opponent in OPPONENTS
        for seed in DEVELOPMENT_SEEDS
        for seat in SEATS
    ]


def _key(row: dict[str, Any]) -> tuple[str, int, str]:
    return str(row["opponent"]), int(row["seed"]), str(row["candidate_seat"])


def completed_conditions(path: Path) -> set[tuple[str, int, str]]:
    if not path.exists():
        return set()
    return {
        _key(row)
        for line in path.read_text(encoding="utf-8").splitlines()
        if (row := json.loads(line)).get("protocol") == PROTOCOL and row.get("record_type") == "game"
    }


def run_trace(condition: dict[str, Any], expected: tuple[float, float] | None) -> dict[str, Any]:
    seat = str(condition["candidate_seat"])
    v45 = _ObservedV45()
    opponent = load_external_agent(str(condition["opponent"]))
    left, right = (v45, opponent) if seat == "left" else (opponent, v45)
    environment = make("kaggriculture", configuration={"episodeSteps": 720, "seed": condition["seed"]}, debug=True)
    started = time.perf_counter()
    environment.run([left, right])
    terminal = environment.steps[-1]
    rewards = tuple(float(state.reward) for state in terminal)
    v45_index = 0 if seat == "left" else 1
    observed = (rewards[v45_index], rewards[1 - v45_index])
    slots = latent_slots(v45.rows)
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
        "observed_steps": len(v45.rows),
        "latent_slots": slots,
    }


def _append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def collect(output: Path, max_games: int) -> dict[str, Any]:
    expected = reference_rewards()
    done = completed_conditions(output)
    completed = []
    for condition in planned_conditions():
        if _key(condition) in done:
            continue
        row = run_trace(condition, expected.get(_key(condition)))
        _append(output, row)
        completed.append(row)
        if len(completed) >= max_games:
            break
    return {
        "protocol": PROTOCOL,
        "output": str(output),
        "new_games": len(completed),
        "remaining_games": len(planned_conditions()) - len(done) - len(completed),
        "all_new_games_match_reference": all(row["terminal_matches_frozen_league"] for row in completed),
    }


def summarize(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    accepted = [row for row in rows if row.get("protocol") == PROTOCOL and row.get("terminal_matches_frozen_league")]
    crop_counts: dict[str, int] = defaultdict(int)
    per_opponent: dict[str, list[int]] = defaultdict(list)
    sdy_reference = []
    for row in accepted:
        per_opponent[row["opponent"]].append(len(row["latent_slots"]))
        for slot in row["latent_slots"]:
            for candidate in slot["supported_crops"]:
                crop_counts[candidate["crop"]] += 1
            if row["opponent"] == "sdy2842" and slot["step"] == 18 and slot["coordinate"] == [0, 4]:
                sdy_reference.append({
                    "seed": row["seed"], "candidate_seat": row["candidate_seat"], "slot": slot,
                })
    return {
        "protocol": PROTOCOL,
        "source": str(path),
        "accepted_conditions": len(accepted),
        "planned_conditions": len(planned_conditions()),
        "slots_per_opponent": {opponent: {"conditions": len(values), "mean": round(mean(values), 3), "values": values} for opponent, values in sorted(per_opponent.items())},
        "supported_crop_counts": dict(sorted(crop_counts.items())),
        "sdy_step18_coordinate_0_4": sdy_reference,
        "limits": [
            "This replays V45's unmodified future actions; adding a crop could change later observations and actions.",
            "Estimated net uses current-or-base price and seed opportunity cost, not a counterfactual market simulation.",
            "A slot is a feasibility signal, not evidence that planting improves a match.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-games", type=int, default=1)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--summary-output", type=Path, default=Path("experiments/v45_latent_slot_dryrun_v3_summary.json"))
    args = parser.parse_args()
    if args.summarize:
        result = summarize(args.output)
        args.summary_output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        result = collect(args.output, args.max_games)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
