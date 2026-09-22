"""Trace the full early history of the disputed sdy tile at coordinate (0, 4)."""

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


PROTOCOL = "v45-sdy-tile-history-v1"
OPPONENT = "sdy2842"
COORDINATE = (0, 4)
FINAL_ACTION_STEP = 246
DEFAULT_OUTPUT = Path("experiments/v45_sdy_tile_history_v1.jsonl")


def _plain(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _tile(farm: Any) -> Any:
    x, y = COORDINATE
    return _plain(farm.get("tiles", [])[y][x])


def _position(farm: Any, actor_index: int) -> tuple[int, int] | None:
    if actor_index == 0:
        pos = farm.get("farmer") or []
    else:
        hands = farm.get("hands") or []
        pos = hands[actor_index - 1] if actor_index - 1 < len(hands) else []
    return (int(pos[0]), int(pos[1])) if len(pos) == 2 else None


def _key(row: dict[str, Any]) -> tuple[str, int, str]:
    return str(row["opponent"]), int(row["seed"]), str(row["candidate_seat"])


def planned_conditions() -> list[dict[str, Any]]:
    return [{"opponent": OPPONENT, "seed": seed, "candidate_seat": seat} for seed in DEVELOPMENT_SEEDS for seat in SEATS]


def _completed(path: Path) -> set[tuple[str, int, str]]:
    if not path.exists():
        return set()
    return {
        _key(row)
        for line in path.read_text(encoding="utf-8").splitlines()
        if (row := json.loads(line)).get("protocol") == PROTOCOL and row.get("record_type") == "game"
    }


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
    timeline: list[dict[str, Any]] = []
    tile_actions: list[dict[str, Any]] = []

    def observed_interpreter(state: list[Any], env: Any) -> Any:
        nonlocal current_step, player_by_private
        observation = state[0].observation
        if not hasattr(observation, "farms") or len(observation.farms) < 2:
            return original_interpreter(state, env)
        current_step = int(observation.step)
        player_by_private = {id(entry.observation.private): player for player, entry in enumerate(state)}
        if current_step > FINAL_ACTION_STEP:
            return original_interpreter(state, env)
        before = [_tile(entry.observation.farms[player]) for player, entry in enumerate(state)]
        result = original_interpreter(state, env)
        after = [_tile(result[0].observation.farms[player]) for player in range(2)]
        timeline.append({"action_step": current_step, "tile_before": before, "tile_after": after})
        return result

    def observed_apply(farm: Any, private: Any, actor_index: int, action: Any, *args: Any, **kwargs: Any) -> Any:
        player = player_by_private.get(id(private), -1)
        on_target = current_step is not None and current_step <= FINAL_ACTION_STEP and _position(farm, actor_index) == COORDINATE
        tile_before = _tile(farm) if on_target else None
        result = original_apply(farm, private, actor_index, action, *args, **kwargs)
        if on_target:
            tile_actions.append(
                {
                    "action_step": current_step,
                    "player": player,
                    "actor": "farmer" if actor_index == 0 else f"hand_{actor_index - 1}",
                    "action": _plain(action),
                    "tile_before": tile_before,
                    "tile_after": _tile(farm),
                    "tile_changed": tile_before != _tile(farm),
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
        "coordinate": list(COORDINATE),
        "timeline": timeline,
        "tile_actions": tile_actions,
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
