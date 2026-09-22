"""Trace and aggregate frozen V45 matches without changing either policy.

The trace samples the environment's post-turn observation and the action that
produced it.  It intentionally records only local experimental evidence and
never creates a Kaggle replay or changes an agent implementation.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from kaggle_environments import make

from .baselines import EXTERNAL_BASELINES
from .league import DEVELOPMENT_SEEDS, OPPONENTS, SEATS, read_records
from .runner import AgentSpec, _agent_label, _resolve_agent


TRACE_PROTOCOL = "v45-opponent-diagnosis-v1"
REFERENCE_LEDGER = Path("experiments/seven_opponent_league_v1.jsonl")
PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")
PHASES = (("opening", 0, 72), ("development", 73, 287), ("market", 288, 647), ("terminal", 648, 719))


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _item_counts(items: dict[str, Any] | None) -> dict[str, int]:
    return {item: _int((items or {}).get(item, 0)) for item in PRODUCTS}


def _tiles_summary(tiles: list[list[Any]]) -> dict[str, Any]:
    kinds: Counter[str] = Counter()
    crops: Counter[str] = Counter()
    animals: Counter[str] = Counter()
    yield_units = 0
    yield_ready_plants = 0
    unfed_animals = 0
    for row in tiles:
        for tile in row:
            if tile is None:
                kinds["EMPTY"] += 1
            elif tile == "LOCKED":
                kinds["LOCKED"] += 1
            elif isinstance(tile, dict):
                kinds[str(tile.get("kind", "UNKNOWN"))] += 1
                if tile.get("crop"):
                    crops[str(tile["crop"])] += 1
                    yield_ready_plants += int(_int(tile.get("yield_units")) > 0)
                if tile.get("animal"):
                    animals[str(tile["animal"])] += 1
                    unfed_animals += int(not tile.get("fed_today", False))
                yield_units += max(0, _int(tile.get("yield_units")))
            else:
                kinds["OTHER"] += 1
    return {
        "kinds": dict(sorted(kinds.items())),
        "crops": dict(sorted(crops.items())),
        "animals": dict(sorted(animals.items())),
        "yield_units": yield_units,
        # This is intentionally about currently available yield, not biological
        # maturity: some crop records carry a future yield counter before it is
        # harvestable.
        "yield_ready_plants": yield_ready_plants,
        "unfed_animals": unfed_animals,
    }


def _farm_snapshot(farm: dict[str, Any]) -> dict[str, Any]:
    return {
        "money": _int(farm.get("money")),
        "farmer": list(farm.get("farmer") or []),
        "hands": [list(hand) for hand in (farm.get("hands") or [])],
        "hand_count": len(farm.get("hands") or []),
        "unlocked_quadrants": list(farm.get("unlocked_quadrants") or []),
        "hires_today": _int(farm.get("hires_today")),
        "tiles": _tiles_summary(farm.get("tiles") or []),
    }


def _private_snapshot(private: dict[str, Any]) -> dict[str, Any]:
    inventories = list(private.get("inventories") or [])
    carried: Counter[str] = Counter()
    for inventory in inventories:
        carried.update({item: _int(value) for item, value in dict(inventory or {}).items()})
    return {
        "shed": _item_counts(private.get("shed")),
        "seeds": {item: _int(value) for item, value in dict(private.get("seeds") or {}).items()},
        "carried": {item: _int(carried.get(item, 0)) for item in PRODUCTS},
    }


def _turn_record(
    *,
    condition: dict[str, Any],
    state_step: int,
    states: list[Any],
) -> dict[str, Any]:
    public = states[0].observation
    players = []
    for player, state in enumerate(states):
        observation = state.observation
        players.append(
            {
                "farm": _farm_snapshot(public["farms"][player]),
                "private": _private_snapshot(observation["private"]),
                # At state_step N (> 0), Kaggle exposes the action executed at N-1.
                "previous_action": state.action if state_step > 0 else None,
            }
        )
    return {
        "record_type": "turn",
        "protocol": TRACE_PROTOCOL,
        **condition,
        "state_step": state_step,
        "action_step": state_step - 1 if state_step > 0 else None,
        "market": {
            "prices": _item_counts(public.get("market", {}).get("prices")),
            "inventory": _item_counts(public.get("market", {}).get("inventory")),
        },
        "shops": list(public.get("town", {}).get("unlocked_shops") or []),
        "players": players,
    }


def reference_rewards(path: Path = REFERENCE_LEDGER) -> dict[tuple[str, int, str], tuple[float, float]]:
    """Return the frozen V45 direct-game rewards and reject conflicting evidence."""

    values: dict[tuple[str, int, str], set[tuple[float, float]]] = defaultdict(set)
    for row in read_records(path):
        if not (
            row.get("protocol") == "seven-opponent-league-v1"
            and row.get("candidate") == "v45_base"
            and row.get("split") == "development"
            and row.get("evaluated_policy") == "candidate"
        ):
            continue
        key = (str(row["opponent"]), _int(row["seed"]), str(row["candidate_seat"]))
        values[key].add((float(row["candidate_reward"]), float(row["opponent_reward"])))
    conflict = {key: item for key, item in values.items() if len(item) != 1}
    if conflict:
        raise RuntimeError(f"Conflicting frozen reference rewards: {sorted(conflict)}")
    return {key: next(iter(value)) for key, value in values.items()}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def run_trace(condition: dict[str, Any], expected: tuple[float, float] | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    opponent = str(condition["opponent"])
    seat = str(condition["candidate_seat"])
    left: AgentSpec = "v45_base" if seat == "left" else opponent
    right: AgentSpec = opponent if seat == "left" else "v45_base"
    left_agent, right_agent = _resolve_agent(left), _resolve_agent(right)
    environment = make("kaggriculture", configuration={"episodeSteps": 720, "seed": condition["seed"]}, debug=True)
    started = time.perf_counter()
    environment.run([left_agent, right_agent])
    elapsed = round(time.perf_counter() - started, 3)
    turns: list[dict[str, Any]] = []
    for state_step, states in enumerate(environment.steps):
        turns.append(_turn_record(condition=condition, state_step=state_step, states=states))
    terminal = environment.steps[-1]
    rewards = tuple(float(state.reward) for state in terminal)
    v45_index = 0 if seat == "left" else 1
    observed = (rewards[v45_index], rewards[1 - v45_index])
    validation = expected is not None and observed == expected
    summary = {
        "record_type": "game_end",
        "protocol": TRACE_PROTOCOL,
        **condition,
        "left": _agent_label(left),
        "right": _agent_label(right),
        "elapsed_seconds": elapsed,
        "statuses": [state.status for state in terminal],
        "v45_reward": observed[0],
        "opponent_reward": observed[1],
        "expected_v45_reward": expected[0] if expected else None,
        "expected_opponent_reward": expected[1] if expected else None,
        "terminal_matches_frozen_league": validation,
        "v45_telemetry": dict(getattr(left_agent if v45_index == 0 else right_agent, "telemetry", {})),
    }
    return turns, summary


def planned_conditions() -> list[dict[str, Any]]:
    return [
        {"opponent": opponent, "seed": seed, "candidate_seat": seat}
        for opponent in OPPONENTS
        for seed in DEVELOPMENT_SEEDS
        for seat in SEATS
    ]


def completed_conditions(path: Path) -> set[tuple[str, int, str]]:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("protocol") == TRACE_PROTOCOL and row.get("record_type") == "game_end":
            done.add((str(row["opponent"]), _int(row["seed"]), str(row["candidate_seat"])))
    return done


def collect(output: Path, max_games: int) -> dict[str, Any]:
    if max_games < 1:
        raise ValueError("max_games must be positive")
    expected = reference_rewards()
    planned = planned_conditions()
    done = completed_conditions(output)
    completed: list[dict[str, Any]] = []
    for condition in planned:
        key = (condition["opponent"], condition["seed"], condition["candidate_seat"])
        if key in done:
            continue
        turns, summary = run_trace(condition, expected.get(key))
        _write_jsonl(output, [*turns, summary])
        completed.append(summary)
        if len(completed) >= max_games:
            break
    return {
        "protocol": TRACE_PROTOCOL,
        "output": str(output),
        "new_games": len(completed),
        "remaining_games": len(planned) - len(done) - len(completed),
        "all_new_games_match_reference": all(row["terminal_matches_frozen_league"] for row in completed),
        "new_game_summaries": completed,
    }


def _mean(values: list[float | int]) -> float:
    return round(mean(values), 3) if values else 0.0


def summarize_trace(path: Path) -> dict[str, Any]:
    """Aggregate only fully reproduced games into auditable phase statistics.

    Market order quantities are *requested* quantities, not confirmed fills;
    phase cash movements are the primary realised outcome measure.
    """

    games: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    ends: dict[tuple[str, int, str], dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("protocol") != TRACE_PROTOCOL:
            continue
        key = (str(row["opponent"]), _int(row["seed"]), str(row["candidate_seat"]))
        if row.get("record_type") == "turn":
            games[key].append(row)
        elif row.get("record_type") == "game_end":
            ends[key] = row

    accepted = {key for key, end in ends.items() if end.get("terminal_matches_frozen_league")}
    malformed = [key for key in accepted if len(games.get(key, [])) != 720]
    accepted.difference_update(malformed)
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    telemetry_keys = (
        "race_clone_turns", "race_horizon_turns", "race_escalations", "sale_reserved_units",
        "sale_reservations", "replenishment_trim_units", "feed_skips", "supply_budget_declines",
        "opening_day1_cash", "warehouse_extra_sales", "quote_reordered_turns",
    )

    for key in sorted(accepted):
        opponent, _, seat = key
        v45_index = 0 if seat == "left" else 1
        opponent_index = 1 - v45_index
        rows = {row["state_step"]: row for row in games[key]}
        end = ends[key]
        grouped[opponent]["terminal_margin"].append(end["v45_reward"] - end["opponent_reward"])
        for phase, start, finish in PHASES:
            before = rows[0 if start == 0 else start - 1]
            after = rows[finish]
            v45_before = before["players"][v45_index]["farm"]["money"]
            opp_before = before["players"][opponent_index]["farm"]["money"]
            v45_after = after["players"][v45_index]["farm"]["money"]
            opp_after = after["players"][opponent_index]["farm"]["money"]
            grouped[opponent][f"cash_delta_margin.{phase}"].append(
                (v45_after - v45_before) - (opp_after - opp_before)
            )
            grouped[opponent][f"cash_margin_end.{phase}"].append(v45_after - opp_after)
            for actor, index in (("v45", v45_index), ("opponent", opponent_index)):
                snapshot = after["players"][index]
                tiles = snapshot["farm"]["tiles"]
                assets = {
                    "hands": snapshot["farm"]["hand_count"],
                    "crops": sum(tiles["crops"].values()),
                    "animals": sum(tiles["animals"].values()),
                    "yield_units": tiles["yield_units"],
                    "shed_units": sum(snapshot["private"]["shed"].values()),
                }
                for metric, value in assets.items():
                    grouped[opponent][f"asset.{phase}.{actor}.{metric}"].append(value)
            for row in rows.values():
                action_step = row["action_step"]
                if action_step is None or not start <= action_step <= finish:
                    continue
                for actor, index in (("v45", v45_index), ("opponent", opponent_index)):
                    for order in row["players"][index]["previous_action"].get("market", []):
                        if len(order) >= 3 and order[0] == "SELL":
                            grouped[opponent][f"requested_sell.{phase}.{actor}.{order[1]}"].append(_int(order[2]))
            # The opening-round order executes between state 0 and state 1.
            opening_action = rows[1]["players"]
            for actor, index in (("v45", v45_index), ("opponent", opponent_index)):
                for order in opening_action[index]["previous_action"].get("market", []):
                    if len(order) >= 3 and order[1] == "WHEAT" and order[0] in {"BUY_PRODUCT", "SELL"}:
                        grouped[opponent][f"opening_wheat.{actor}.{order[0]}"].append(_int(order[2]))
        for metric in telemetry_keys:
            grouped[opponent][f"telemetry.{metric}"].append(_int(end["v45_telemetry"].get(metric)))

    opponents: dict[str, Any] = {}
    for opponent, measures in grouped.items():
        phase_data: dict[str, Any] = {}
        for phase, _, _ in PHASES:
            assets = {}
            for metric in ("hands", "crops", "animals", "yield_units", "shed_units"):
                v45 = _mean(measures[f"asset.{phase}.v45.{metric}"])
                rival = _mean(measures[f"asset.{phase}.opponent.{metric}"])
                assets[metric] = {"v45": v45, "opponent": rival, "difference": round(v45 - rival, 3)}
            requested = {
                item: round(
                    sum(measures[f"requested_sell.{phase}.v45.{item}"])
                    - sum(measures[f"requested_sell.{phase}.opponent.{item}"]),
                    3,
                )
                for item in PRODUCTS
            }
            phase_data[phase] = {
                "cash_delta_margin": _mean(measures[f"cash_delta_margin.{phase}"]),
                "cash_margin_end": _mean(measures[f"cash_margin_end.{phase}"]),
                "assets_at_phase_end": assets,
                "requested_sell_unit_difference_total": requested,
            }
        opponents[opponent] = {
            "accepted_conditions": len(measures["terminal_margin"]),
            "mean_terminal_cash_margin": _mean(measures["terminal_margin"]),
            "opening_wheat_order_units": {
                "v45_buy": _mean(measures["opening_wheat.v45.BUY_PRODUCT"]),
                "v45_sell": _mean(measures["opening_wheat.v45.SELL"]),
                "opponent_buy": _mean(measures["opening_wheat.opponent.BUY_PRODUCT"]),
                "opponent_sell": _mean(measures["opening_wheat.opponent.SELL"]),
            },
            "phases": phase_data,
            "v45_telemetry_mean": {
                metric: _mean(measures[f"telemetry.{metric}"])
                for metric in telemetry_keys
            },
        }
    return {
        "protocol": TRACE_PROTOCOL,
        "source": str(path),
        "planned_conditions": len(planned_conditions()),
        "accepted_conditions": len(accepted),
        "rejected_or_incomplete_conditions": len(planned_conditions()) - len(accepted),
        "malformed_accepted_conditions": [list(key) for key in malformed],
        "order_quantity_note": "requested SELL quantities are agent requests, not confirmed fills; interpret phase cash deltas as realised evidence.",
        "opponents": opponents,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("experiments/v45_opponent_diagnosis_v1.jsonl"))
    parser.add_argument("--max-games", type=int, default=1)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--summary-output", type=Path, default=Path("experiments/v45_opponent_diagnosis_v1_summary.json"))
    args = parser.parse_args()
    if args.summarize:
        summary = summarize_trace(args.output)
        args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(collect(args.output, args.max_games), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
