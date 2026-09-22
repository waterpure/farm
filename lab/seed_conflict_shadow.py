"""Shadow-test a real-time, crop-agnostic allocator for scarce route seeds.

This observer never changes V45.  When V45's raw tape asks multiple workers to
plant the same crop but fewer seeds exist, it scores each currently valid raw
request only from information available now: whether that worker is scheduled
to WATER the same square in the next two steps and how many same-square WATER
tasks the current tape has before today's end.  A tie deliberately abstains;
coordinates, crop names, opponents, and later trace events never break it.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from kaggle_environments import make

from .baselines import load_external_agent, load_v45_base_module
from .diagnostics import reference_rewards
from .league import DEVELOPMENT_SEEDS, OPPONENTS, SEATS


PROTOCOL = "v45-seed-conflict-shadow-v1"
DEFAULT_OUTPUT = Path("experiments/v45_seed_conflict_shadow_v1.jsonl")
MOVES = {"NORTH": (0, -1), "SOUTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0)}


def _commands(action: dict[str, Any], count: int) -> list[list[Any]]:
    commands = [list(action.get("farmer") or ["PASS"])] + [list(command or ["PASS"]) for command in action.get("hands") or []]
    return commands[:count] + [["PASS"] for _ in range(max(0, count - len(commands)))]


def _same_day_service(module: Any, route: int, step: int, actor: int, start: tuple[int, int]) -> dict[str, Any]:
    """Known current-tape evidence only; never crosses today's route boundary."""
    position = list(start)
    water_steps = []
    end = min((step // 24 + 1) * 24 - 1, 718)
    for future_step in range(step + 1, end + 1):
        command = _commands(module._IMPL.chassis.routes[route][future_step], actor + 1)[actor]
        if tuple(position) == start and command and command[0] == "WATER":
            water_steps.append(future_step)
        if command and command[0] in MOVES:
            dx, dy = MOVES[command[0]]
            position[0] += dx
            position[1] += dy
    return {
        "immediate_same_actor_water": any(water_step <= step + 2 for water_step in water_steps),
        "same_day_water_steps": water_steps,
        "score": [int(any(water_step <= step + 2 for water_step in water_steps)), len(water_steps)],
    }


class _ObservedV45:
    def __init__(self) -> None:
        self.module = load_v45_base_module()
        self.policy = self.module.agent
        self.conflicts: list[dict[str, Any]] = []

    def __call__(self, observation: dict[str, Any], configuration: dict[str, Any] | None = None) -> dict[str, Any]:
        player = int(observation["player"])
        step = int(observation["step"])
        action = self.policy(observation, configuration)
        farm = observation["farms"][player]
        positions = [tuple(farm["farmer"]), *[tuple(position) for position in farm["hands"]]]
        route = int(self.module._IMPL.chassis.players.get(player, {}).get("route", 0))
        raw = self.module._IMPL.chassis.routes[route][step]
        raw_commands = _commands(raw, len(positions))
        emitted_commands = _commands(action, len(positions))
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for actor, command in enumerate(raw_commands):
            if len(command) < 2 or command[0] != "PLANT":
                continue
            crop = command[1]
            x, y = positions[actor]
            if farm["tiles"][y][x] is not None:
                continue
            grouped[crop].append({"actor": actor, "coordinate": [x, y], "raw_command": command})
        for crop, candidates in grouped.items():
            available = int(observation["private"]["seeds"].get(crop, 0))
            if available <= 0 or len(candidates) <= available:
                continue
            scored = []
            for candidate in candidates:
                service = _same_day_service(self.module, route, step, candidate["actor"], tuple(candidate["coordinate"]))
                scored.append({
                    **candidate,
                    "emitted_command": emitted_commands[candidate["actor"]],
                    "emitted_raw_plant": emitted_commands[candidate["actor"]] == candidate["raw_command"],
                    "service": service,
                })
            maximum = max(tuple(candidate["service"]["score"]) for candidate in scored)
            winners = [candidate for candidate in scored if tuple(candidate["service"]["score"]) == maximum]
            self.conflicts.append({
                "step": step,
                "day": step // 24,
                "route_at_decision": route,
                "crop": crop,
                "seed_available": available,
                "candidate_count": len(scored),
                "candidates": scored,
                "recommendation": (
                    {"kind": "unique", "actors": [candidate["actor"] for candidate in winners]}
                    if len(winners) == 1
                    else {"kind": "abstain_tie", "actors": [candidate["actor"] for candidate in winners]}
                ),
            })
        return action


def planned_conditions() -> list[dict[str, Any]]:
    return [{"opponent": opponent, "seed": seed, "candidate_seat": seat} for opponent in OPPONENTS for seed in DEVELOPMENT_SEEDS for seat in SEATS]


def _key(row: dict[str, Any]) -> tuple[str, int, str]:
    return str(row["opponent"]), int(row["seed"]), str(row["candidate_seat"])


def _completed(path: Path) -> set[tuple[str, int, str]]:
    if not path.exists():
        return set()
    return {_key(row) for line in path.read_text(encoding="utf-8").splitlines() if (row := json.loads(line)).get("protocol") == PROTOCOL}


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
        "protocol": PROTOCOL,
        **condition,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "statuses": [state.status for state in terminal],
        "terminal_matches_frozen_league": expected is not None and observed == expected,
        "conflicts": v45.conflicts,
    }


def collect(output: Path, max_games: int) -> dict[str, Any]:
    expected = reference_rewards()
    done = _completed(output)
    new = []
    for condition in planned_conditions():
        if _key(condition) in done:
            continue
        row = _run(condition, expected.get(_key(condition)))
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        new.append(row)
        if len(new) >= max_games:
            break
    return {"protocol": PROTOCOL, "new_games": len(new), "remaining_games": len(planned_conditions()) - len(done) - len(new), "all_new_games_match_reference": all(row["terminal_matches_frozen_league"] for row in new)}


def summarize(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    valid = [row for row in rows if row.get("protocol") == PROTOCOL and row.get("terminal_matches_frozen_league")]
    recommendation_counts = Counter()
    sdy = []
    for row in valid:
        for conflict in row["conflicts"]:
            recommendation_counts[conflict["recommendation"]["kind"]] += 1
            if row["opponent"] == "sdy2842" and conflict["step"] == 18 and conflict["crop"] == "MELON":
                sdy.append({"seed": row["seed"], "candidate_seat": row["candidate_seat"], "conflict": conflict})
    return {
        "protocol": PROTOCOL,
        "source": str(path),
        "accepted_conditions": len(valid),
        "planned_conditions": len(planned_conditions()),
        "recommendation_counts": dict(recommendation_counts),
        "sdy_step18_melon_conflicts": sdy,
        "limits": [
            "Scores use only current route commands through the current day; they intentionally do not predict later coordinates, market fills, or opponent actions.",
            "A tie abstains, so this observer cannot create a coordinate-specific choice by construction.",
            "This never modifies V45 actions and does not establish a score improvement.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-games", type=int, default=1)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--summary-output", type=Path, default=Path("experiments/v45_seed_conflict_shadow_v1_summary.json"))
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
