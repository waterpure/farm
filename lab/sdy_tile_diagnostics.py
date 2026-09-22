"""Capture the exact tiles behind sdy's repeated WATER/HARVEST no-ops.

This is deliberately narrower than the worker trace: it instruments only
``hand_0`` at action steps 245 and 246 in the eight frozen sdy conditions.
No policy code or returned action is changed.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture as game

from .diagnostics import reference_rewards
from .league import DEVELOPMENT_SEEDS, SEATS
from .runner import AgentSpec, _agent_label, _resolve_agent


PROTOCOL = "v45-sdy-tile-diagnosis-v1"
OPPONENT = "sdy2842"
TARGET_HAND_INDEX = 1  # farmer is 0; hand_0 is 1.
TARGET_STEPS = (245, 246)
DEFAULT_OUTPUT = Path("experiments/v45_sdy_tile_diagnosis_v1.jsonl")


def _plain(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _position(farm: Any, actor_index: int) -> list[int] | None:
    if actor_index == 0:
        return list(farm.get("farmer") or [])
    hands = list(farm.get("hands") or [])
    hand_index = actor_index - 1
    return list(hands[hand_index]) if hand_index < len(hands) else None


def _tile(farm: Any, position: list[int] | None) -> Any:
    if not position or len(position) != 2:
        return None
    x, y = position
    tiles = farm.get("tiles") or []
    if y < 0 or y >= len(tiles) or x < 0 or x >= len(tiles[y]):
        return None
    return _plain(tiles[y][x])


def planned_conditions() -> list[dict[str, Any]]:
    return [{"opponent": OPPONENT, "seed": seed, "candidate_seat": seat} for seed in DEVELOPMENT_SEEDS for seat in SEATS]


def _key(row: dict[str, Any]) -> tuple[str, int, str]:
    return str(row["opponent"]), int(row["seed"]), str(row["candidate_seat"])


def _completed(path: Path) -> set[tuple[str, int, str]]:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("protocol") == PROTOCOL and row.get("record_type") == "game":
            done.add(_key(row))
    return done


def run_trace(condition: dict[str, Any], expected: tuple[float, float] | None) -> dict[str, Any]:
    seat = str(condition["candidate_seat"])
    left: AgentSpec = "v45_base" if seat == "left" else OPPONENT
    right: AgentSpec = OPPONENT if seat == "left" else "v45_base"
    left_agent, right_agent = _resolve_agent(left), _resolve_agent(right)
    environment = make("kaggriculture", configuration={"episodeSteps": 720, "seed": condition["seed"]}, debug=True)
    original_interpreter = environment.interpreter
    original_apply = game._apply_unit_action
    current_step: int | None = None
    player_by_private: dict[int, int] = {}
    events: list[dict[str, Any]] = []

    def observed_interpreter(state: list[Any], env: Any) -> Any:
        nonlocal current_step, player_by_private
        current_step = int(state[0].observation.step)
        player_by_private = {id(entry.observation.private): player for player, entry in enumerate(state)}
        return original_interpreter(state, env)

    def observed_apply(farm: Any, private: Any, actor_index: int, action: Any, *args: Any, **kwargs: Any) -> Any:
        player = player_by_private.get(id(private), -1)
        capture = current_step in TARGET_STEPS and actor_index == TARGET_HAND_INDEX
        position_before = _position(farm, actor_index) if capture else None
        tile_before = _tile(farm, position_before) if capture else None
        result = original_apply(farm, private, actor_index, action, *args, **kwargs)
        if capture:
            position_after = _position(farm, actor_index)
            tile_after = _tile(farm, position_after)
            events.append(
                {
                    "action_step": current_step,
                    "player": player,
                    "actor": "hand_0",
                    "action": _plain(action),
                    "position_before": position_before,
                    "position_after": position_after,
                    "tile_before": tile_before,
                    "tile_after": tile_after,
                    "tile_changed": tile_before != tile_after,
                }
            )
        return result

    environment.interpreter = observed_interpreter
    game._apply_unit_action = observed_apply
    started = time.perf_counter()
    try:
        environment.run([left_agent, right_agent])
    finally:
        environment.interpreter = original_interpreter
        game._apply_unit_action = original_apply
    terminal = environment.steps[-1]
    rewards = tuple(float(state.reward) for state in terminal)
    v45_player = 0 if seat == "left" else 1
    observed = (rewards[v45_player], rewards[1 - v45_player])
    return {
        "record_type": "game",
        "protocol": PROTOCOL,
        **condition,
        "left": _agent_label(left),
        "right": _agent_label(right),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "statuses": [state.status for state in terminal],
        "v45_player": v45_player,
        "v45_reward": observed[0],
        "opponent_reward": observed[1],
        "expected_v45_reward": expected[0] if expected else None,
        "expected_opponent_reward": expected[1] if expected else None,
        "terminal_matches_frozen_league": expected is not None and observed == expected,
        "tile_events": events,
    }


def _append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def collect(output: Path, max_games: int) -> dict[str, Any]:
    expected = reference_rewards()
    done = _completed(output)
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-games", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(collect(args.output, args.max_games), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
