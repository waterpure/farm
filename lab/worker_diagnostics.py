"""Record worker-level action outcomes for the fixed near-peer audit matches.

The module wraps only the local Kaggriculture interpreter instance during a
match.  It never changes an agent action.  The original interpreter, unit
action implementation, and market commit implementation are restored before
the next operation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture as game

from .diagnostics import PRODUCTS, reference_rewards
from .league import DEVELOPMENT_SEEDS, SEATS
from .runner import AgentSpec, _agent_label, _resolve_agent


PROTOCOL = "v45-worker-diagnosis-v1"
TARGETS = ("sdy2842", "v46", "nathan_pipe7", "kaito_v58")
REFERENCE = Path("experiments/seven_opponent_league_v1.jsonl")
DEFAULT_OUTPUT = Path("experiments/v45_worker_diagnosis_v1.jsonl")
DEFAULT_SUMMARY_OUTPUT = Path("experiments/v45_worker_diagnosis_v1_summary.json")
FOCUS = {
    "sdy2842": ((251, "MELON"),),
    "nathan_pipe7": ((80, "WHEAT"), (100, "WHEAT")),
    "kaito_v58": ((251, "MELON"),),
    "v46": ((2, "WHEAT"),),
}


def _plain(value: Any) -> Any:
    """Copy Struct/dict/list values into JSON-safe regular Python objects."""

    return json.loads(json.dumps(value))


def _items(inventories: list[dict[str, Any]], actor_index: int) -> dict[str, int]:
    if actor_index >= len(inventories):
        return {}
    return {str(item): int(quantity) for item, quantity in dict(inventories[actor_index] or {}).items()}


def _count(inventories: list[dict[str, Any]], actor_index: int, item: str) -> int:
    return int(_items(inventories, actor_index).get(item, 0))


class _WorkerRecorder:
    def __init__(self) -> None:
        self.step: int | None = None
        self.player_by_private: dict[int, int] = {}
        self.unit_events: list[dict[str, Any]] = []
        self.market_commits: list[dict[str, Any]] = []

    def begin_step(self, state: list[Any]) -> None:
        observation = state[0].observation
        self.step = int(observation.step)
        self.player_by_private = {id(entry.observation.private): player for player, entry in enumerate(state)}

    def player(self, private: Any) -> int:
        return self.player_by_private.get(id(private), -1)

    def record_unit(
        self,
        *,
        player: int,
        actor_index: int,
        action: Any,
        farm_before: dict[str, Any],
        farm_after: dict[str, Any],
        private_before: dict[str, Any],
        private_after: dict[str, Any],
    ) -> None:
        action_value = _plain(action)
        op = action_value[0] if isinstance(action_value, list) and action_value else None
        item = action_value[1] if isinstance(action_value, list) and len(action_value) > 1 else None
        amount = int(action_value[2]) if isinstance(action_value, list) and len(action_value) > 2 and isinstance(action_value[2], int) else None
        inventories_before = list(private_before.get("inventories") or [])
        inventories_after = list(private_after.get("inventories") or [])
        shed_before = dict(private_before.get("shed") or {})
        shed_after = dict(private_after.get("shed") or {})
        actor_item_before = _count(inventories_before, actor_index, str(item)) if item else 0
        actor_item_after = _count(inventories_after, actor_index, str(item)) if item else 0
        shed_before_count = int(shed_before.get(item, 0)) if item else 0
        shed_after_count = int(shed_after.get(item, 0)) if item else 0
        shed_delta = shed_after_count - shed_before_count
        carried_delta = actor_item_after - actor_item_before
        changed = farm_before != farm_after or private_before != private_after
        if op == "PLACE" and item in PRODUCTS:
            outcome = "success" if shed_delta > 0 and carried_delta < 0 else "no_op"
        elif op == "PLACE":
            # Animal placement changes a farm tile rather than the shed.
            outcome = "state_changed" if changed else "no_op"
        elif op == "PICKUP":
            outcome = "success" if carried_delta > 0 and shed_delta < 0 else "no_op"
        else:
            outcome = "state_changed" if changed else "no_op"
        self.unit_events.append(
            {
                "action_step": self.step,
                "player": player,
                "actor": "farmer" if actor_index == 0 else f"hand_{actor_index - 1}",
                "actor_index": actor_index,
                "action": action_value,
                "operation": op,
                "item": item,
                "requested_amount": amount,
                "outcome": outcome,
                "actor_item_before": actor_item_before,
                "actor_item_after": actor_item_after,
                "actor_item_delta": carried_delta,
                "shed_item_before": shed_before_count,
                "shed_item_after": shed_after_count,
                "shed_item_delta": shed_delta,
                "actor_inventory_before": _items(inventories_before, actor_index),
                "actor_inventory_after": _items(inventories_after, actor_index),
            }
        )

    def record_market(
        self, *, player: int, op: str, item: str, price: int, ok: bool, requested_remaining: int
    ) -> None:
        self.market_commits.append(
            {
                "action_step": self.step,
                "player": player,
                "operation": op,
                "item": item,
                "quoted_price": price,
                "success": ok,
                "remaining_before_commit": requested_remaining,
            }
        )


def _instrument(environment: Any, recorder: _WorkerRecorder) -> tuple[Any, Any, Any]:
    """Install wrappers and return originals for explicit restoration."""

    original_interpreter = environment.interpreter
    original_apply = game._apply_unit_action
    original_commit = game._commit_unit

    def observed_interpreter(state: list[Any], env: Any) -> Any:
        recorder.begin_step(state)
        return original_interpreter(state, env)

    def observed_apply(farm: Any, private: Any, actor_index: int, action: Any, *args: Any, **kwargs: Any) -> Any:
        player = recorder.player(private)
        farm_before = _plain(farm)
        private_before = _plain(private)
        result = original_apply(farm, private, actor_index, action, *args, **kwargs)
        recorder.record_unit(
            player=player,
            actor_index=actor_index,
            action=action,
            farm_before=farm_before,
            farm_after=_plain(farm),
            private_before=private_before,
            private_after=_plain(private),
        )
        return result

    def observed_commit(op: str, item: str, price: int, farm: Any, private: Any, market: Any, shed_capacity: int = 100) -> bool:
        player = recorder.player(private)
        private_before = _plain(private)
        result = original_commit(op, item, price, farm, private, market, shed_capacity)
        recorder.record_market(
            player=player,
            op=op,
            item=item,
            price=int(price),
            ok=bool(result),
            requested_remaining=0,
        )
        # Keep this local snapshot check intentionally cheap: it catches an
        # accidental wrapper mutation without prescribing any game behavior.
        if _plain(private) != private_before and not result and op in {"SELL", "BUY_PRODUCT", "BUY_SEED", "BUY_ANIMAL"}:
            raise RuntimeError("A failed market commit mutated private state")
        return result

    environment.interpreter = observed_interpreter
    game._apply_unit_action = observed_apply
    game._commit_unit = observed_commit
    return original_interpreter, original_apply, original_commit


def _restore(environment: Any, originals: tuple[Any, Any, Any]) -> None:
    environment.interpreter, game._apply_unit_action, game._commit_unit = originals


def _market_summary(commits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for commit in commits:
        grouped[(commit["action_step"], commit["player"], commit["operation"], commit["item"])].append(commit)
    return [
        {
            "action_step": step,
            "player": player,
            "operation": operation,
            "item": item,
            "attempts": len(entries),
            "filled": sum(1 for entry in entries if entry["success"]),
            "failed_attempts": sum(1 for entry in entries if not entry["success"]),
            "revenue_or_cost": sum(entry["quoted_price"] for entry in entries if entry["success"]),
        }
        for (step, player, operation, item), entries in sorted(grouped.items())
    ]


def _condition(opponent: str, seed: int, seat: str) -> dict[str, Any]:
    return {"opponent": opponent, "seed": seed, "candidate_seat": seat}


def planned_conditions() -> list[dict[str, Any]]:
    return [_condition(opponent, seed, seat) for opponent in TARGETS for seed in DEVELOPMENT_SEEDS for seat in SEATS]


def _key(condition: dict[str, Any]) -> tuple[str, int, str]:
    return str(condition["opponent"]), int(condition["seed"]), str(condition["candidate_seat"])


def completed_conditions(path: Path) -> set[tuple[str, int, str]]:
    if not path.exists():
        return set()
    completed = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("protocol") == PROTOCOL and row.get("record_type") == "game":
            completed.add(_key(row))
    return completed


def run_trace(condition: dict[str, Any], expected: tuple[float, float] | None) -> dict[str, Any]:
    opponent = str(condition["opponent"])
    seat = str(condition["candidate_seat"])
    left: AgentSpec = "v45_base" if seat == "left" else opponent
    right: AgentSpec = opponent if seat == "left" else "v45_base"
    left_agent, right_agent = _resolve_agent(left), _resolve_agent(right)
    environment = make("kaggriculture", configuration={"episodeSteps": 720, "seed": condition["seed"]}, debug=True)
    recorder = _WorkerRecorder()
    originals = _instrument(environment, recorder)
    started = time.perf_counter()
    try:
        environment.run([left_agent, right_agent])
    finally:
        _restore(environment, originals)
    terminal = environment.steps[-1]
    rewards = tuple(float(state.reward) for state in terminal)
    v45_index = 0 if seat == "left" else 1
    observed = (rewards[v45_index], rewards[1 - v45_index])
    return {
        "record_type": "game",
        "protocol": PROTOCOL,
        **condition,
        "left": _agent_label(left),
        "right": _agent_label(right),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "statuses": [state.status for state in terminal],
        "v45_reward": observed[0],
        "opponent_reward": observed[1],
        "expected_v45_reward": expected[0] if expected else None,
        "expected_opponent_reward": expected[1] if expected else None,
        "terminal_matches_frozen_league": expected is not None and observed == expected,
        "v45_player": v45_index,
        "unit_events": recorder.unit_events,
        "market_commits": recorder.market_commits,
        "market_summary": _market_summary(recorder.market_commits),
    }


def _append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def collect(output: Path, max_games: int) -> dict[str, Any]:
    expected = reference_rewards(REFERENCE)
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


def summarize_focus(path: Path) -> dict[str, Any]:
    """Extract the pre-registered worker evidence without rereading agents."""

    raw_records = 0
    accepted: dict[tuple[str, int, str], str] = {}
    duplicate_keys: set[tuple[str, int, str]] = set()
    conflicting_duplicates: set[tuple[str, int, str]] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("protocol") != PROTOCOL or row.get("record_type") != "game":
                continue
            raw_records += 1
            key = _key(row)
            semantic = dict(row)
            semantic.pop("elapsed_seconds", None)
            digest = hashlib.sha256(json.dumps(semantic, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
            if key in accepted:
                duplicate_keys.add(key)
                if accepted[key] != digest:
                    conflicting_duplicates.add(key)
                continue
            accepted[key] = digest

    # Re-read one record at a time.  The raw action trace is intentionally
    # large; keeping only the pre-registered focus events avoids a multi-GB
    # in-memory aggregate.
    details = []
    aggregates: dict[tuple[str, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    seen_focus: set[tuple[str, int, str]] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("protocol") != PROTOCOL or row.get("record_type") != "game" or not row.get("terminal_matches_frozen_league"):
                continue
            key = _key(row)
            if key in seen_focus:
                continue
            seen_focus.add(key)
            v45_player = int(row["v45_player"])
            for action_step, item in FOCUS.get(row["opponent"], ()):
                place_events = [
                    event for event in row["unit_events"]
                    if event["action_step"] == action_step and event["operation"] == "PLACE" and event["item"] == item
                ]
                market = [
                    event for event in row["market_summary"]
                    if event["action_step"] == action_step and event["operation"] == "SELL" and event["item"] == item
                ]
                detail = {
                "opponent": row["opponent"],
                "seed": row["seed"],
                "candidate_seat": row["candidate_seat"],
                "v45_player": v45_player,
                "action_step": action_step,
                "state_step_after": action_step + 1,
                "item": item,
                "place_events": place_events,
                "market_sales": market,
            }
                details.append(detail)
                aggregates[(row["opponent"], action_step, item, "v45")].append(
                    {
                        "place_successes": sum(event["outcome"] == "success" for event in place_events if event["player"] == v45_player),
                        "place_noops": sum(event["outcome"] == "no_op" for event in place_events if event["player"] == v45_player),
                        "filled": sum(event["filled"] for event in market if event["player"] == v45_player),
                    }
                )
                rival = 1 - v45_player
                aggregates[(row["opponent"], action_step, item, "rival")].append(
                    {
                        "place_successes": sum(event["outcome"] == "success" for event in place_events if event["player"] == rival),
                        "place_noops": sum(event["outcome"] == "no_op" for event in place_events if event["player"] == rival),
                        "filled": sum(event["filled"] for event in market if event["player"] == rival),
                    }
                )
    aggregate_rows = []
    for (opponent, action_step, item, actor), values in sorted(aggregates.items()):
        aggregate_rows.append(
            {
                "opponent": opponent,
                "action_step": action_step,
                "state_step_after": action_step + 1,
                "item": item,
                "actor": actor,
                "conditions": len(values),
                "place_successes_per_condition": sorted(value["place_successes"] for value in values),
                "place_noops_per_condition": sorted(value["place_noops"] for value in values),
                "filled_per_condition": sorted(value["filled"] for value in values),
            }
        )
    planned = len(planned_conditions())
    return {
        "protocol": PROTOCOL,
        "source": str(path),
        "audit": {
            "raw_game_records": raw_records,
            "unique_condition_records": len(accepted),
            "duplicate_records": raw_records - len(accepted),
            "duplicate_keys": [list(key) for key in sorted(duplicate_keys)],
            "conflicting_duplicate_keys": [list(key) for key in sorted(conflicting_duplicates)],
            "ledger_clean": not duplicate_keys,
        },
        "accepted_conditions": len(seen_focus),
        "planned_conditions": planned,
        "rejected_or_missing_conditions": planned - len(seen_focus),
        "focus_aggregate": aggregate_rows,
        "focus_condition_details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-games", type=int, default=1)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    args = parser.parse_args()
    if args.summarize:
        result = summarize_focus(args.output)
        args.summary_output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        result = collect(args.output, args.max_games)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
