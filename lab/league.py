"""Resumable paired league for V45 variants against seven frozen opponents.

The unit of evidence is a *paired condition*: candidate and frozen V45 play
the same opponent, seed, and seat.  Their terminal-cash difference removes
much of the variation caused by a particular public market stream.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from statistics import mean
from typing import Any

from .baselines import EXTERNAL_BASELINES
from .runner import AGENTS, AgentSpec, run_match
from .v45_variants import make_animal_route_selector_agent, make_feed_multiplier_agent


PROTOCOL = "seven-opponent-league-v1"
OPPONENTS = tuple(EXTERNAL_BASELINES)
DEVELOPMENT_SEEDS = tuple(range(4))
HOLDOUT_SEEDS = tuple(range(1000, 1008))
SEATS = ("left", "right")
BASELINE = "v45_base"

PolicyFactory = Callable[[], AgentSpec]


def candidate_factory(name: str) -> PolicyFactory:
    """Return a fresh factory for an explicitly supported local candidate."""

    if name in AGENTS:
        return lambda: name
    if name.startswith("v45-feed-m"):
        try:
            multiplier = float(name.removeprefix("v45-feed-m"))
        except ValueError as error:
            raise ValueError(f"Invalid feed candidate name: {name}") from error
        return lambda: make_feed_multiplier_agent(multiplier)
    if name == "v45-animal-route-identity":
        return lambda: make_animal_route_selector_agent(identity_only=True)
    if name == "v45-animal-route-v1":
        return make_animal_route_selector_agent
    if name == "v45-animal-route-v2":
        return lambda: make_animal_route_selector_agent(version=2)
    choices = ", ".join(sorted(AGENTS)) + ", v45-feed-m<multiplier>, v45-animal-route-identity, v45-animal-route-v1, v45-animal-route-v2"
    raise ValueError(f"Unknown candidate {name!r}; choose an AGENTS name or {choices}")


def _conditions(split: str) -> Iterable[dict[str, Any]]:
    seeds = DEVELOPMENT_SEEDS if split == "development" else HOLDOUT_SEEDS
    for opponent in OPPONENTS:
        for seed in seeds:
            for seat in SEATS:
                yield {"opponent": opponent, "seed": seed, "candidate_seat": seat}


def _record_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("protocol"),
        row.get("split"),
        row.get("candidate"),
        row.get("evaluated_policy"),
        row.get("opponent"),
        row.get("seed"),
        row.get("candidate_seat"),
    )


def read_records(path: Path) -> list[dict[str, Any]]:
    """Read valid historical JSONL rows without rewriting prior evidence."""

    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error
        if isinstance(row, dict):
            records.append(row)
    return records


def audit_records(records: list[dict[str, Any]], candidate: str, split: str) -> dict[str, Any]:
    """Report duplicate condition keys without silently discarding evidence."""

    selected = [
        row
        for row in records
        if row.get("protocol") == PROTOCOL and row.get("candidate") == candidate and row.get("split") == split
    ]
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        groups[_record_key(row)].append(row)
    duplicates = {key: rows for key, rows in groups.items() if len(rows) > 1}

    def semantic_value(row: dict[str, Any]) -> tuple[Any, ...]:
        return (row["candidate_reward"], row["opponent_reward"], row["margin"], tuple(row["statuses"]))

    conflicts = sum(
        1 for rows in duplicates.values() if len({semantic_value(row) for row in rows}) > 1
    )
    return {
        "raw_records": len(selected),
        "unique_condition_records": len(groups),
        "duplicate_records": sum(len(rows) - 1 for rows in duplicates.values()),
        "duplicate_keys": len(duplicates),
        "conflicting_duplicate_keys": conflicts,
        "ledger_clean": not duplicates,
    }


def _run_one(
    *,
    candidate: str,
    evaluated_policy: str,
    factory: PolicyFactory,
    opponent: str,
    seed: int,
    candidate_seat: str,
) -> dict[str, Any]:
    candidate_agent = factory()
    left: AgentSpec
    right: AgentSpec
    if candidate_seat == "left":
        left, right = candidate_agent, opponent
        candidate_index = 0
    else:
        left, right = opponent, candidate_agent
        candidate_index = 1
    match = run_match(left, right, steps=720, seed=seed)
    opponent_index = 1 - candidate_index
    return {
        "protocol": PROTOCOL,
        "split": None,
        "candidate": candidate,
        "evaluated_policy": evaluated_policy,
        "opponent": opponent,
        "seed": seed,
        "candidate_seat": candidate_seat,
        "candidate_reward": float(match["rewards"][candidate_index]),
        "opponent_reward": float(match["rewards"][opponent_index]),
        "margin": float(match["rewards"][candidate_index]) - float(match["rewards"][opponent_index]),
        "elapsed_seconds": match["elapsed_seconds"],
        "statuses": match["statuses"],
        "telemetry": match["telemetry"][candidate_index],
    }


def _planned_keys(candidate: str, split: str) -> list[tuple[Any, ...]]:
    return [
        (PROTOCOL, split, candidate, evaluated_policy, condition["opponent"], condition["seed"], condition["candidate_seat"])
        for condition in _conditions(split)
        for evaluated_policy in ("candidate", "v45_base")
    ]


def _append(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def summarize(records: list[dict[str, Any]], candidate: str, split: str) -> dict[str, Any]:
    """Summarize complete candidate/V45 pairs without filling missing data."""

    selected = [
        row
        for row in records
        if row.get("protocol") == PROTOCOL and row.get("candidate") == candidate and row.get("split") == split
    ]
    paired: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in selected:
        paired[(row["opponent"], row["seed"], row["candidate_seat"])][row["evaluated_policy"]] = row

    deltas: list[dict[str, Any]] = []
    for (opponent, seed, seat), pair in sorted(paired.items()):
        candidate_row = pair.get("candidate")
        baseline_row = pair.get("v45_base")
        if candidate_row is None or baseline_row is None:
            continue
        deltas.append(
            {
                "opponent": opponent,
                "seed": seed,
                "candidate_seat": seat,
                "delta_vs_v45": round(candidate_row["candidate_reward"] - baseline_row["candidate_reward"], 3),
                "candidate_done": candidate_row["statuses"] == ["DONE", "DONE"],
                "baseline_done": baseline_row["statuses"] == ["DONE", "DONE"],
            }
        )

    per_opponent = {
        opponent: round(mean(row["delta_vs_v45"] for row in deltas if row["opponent"] == opponent), 3)
        for opponent in OPPONENTS
        if any(row["opponent"] == opponent for row in deltas)
    }
    self_opponent = candidate if candidate in OPPONENTS else None
    nonself_deltas = [row for row in deltas if row["opponent"] != self_opponent]
    nonself_per_opponent = {
        opponent: round(mean(row["delta_vs_v45"] for row in nonself_deltas if row["opponent"] == opponent), 3)
        for opponent in OPPONENTS
        if opponent != self_opponent and any(row["opponent"] == opponent for row in nonself_deltas)
    }
    expected_pairs = len(OPPONENTS) * (len(DEVELOPMENT_SEEDS) if split == "development" else len(HOLDOUT_SEEDS)) * len(SEATS)
    all_done = len(deltas) == expected_pairs and all(
        row["candidate_done"] and row["baseline_done"] for row in deltas
    )
    mean_delta = round(mean(row["delta_vs_v45"] for row in deltas), 3) if deltas else None
    nonnegative_opponents = sum(value >= 0 for value in per_opponent.values())
    baseline_gate = bool(all_done and mean_delta is not None and mean_delta > 0 and nonnegative_opponents >= 4)
    nonself_mean = round(mean(row["delta_vs_v45"] for row in nonself_deltas), 3) if nonself_deltas else None
    return {
        "candidate": candidate,
        "split": split,
        "paired_conditions_complete": len(deltas),
        "paired_conditions_expected": expected_pairs,
        "all_done": all_done,
        "mean_delta_vs_v45": mean_delta,
        "per_opponent_mean_delta": per_opponent,
        "nonnegative_opponents": nonnegative_opponents,
        "minimum_gate_met": baseline_gate,
        "self_opponent": self_opponent,
        "nonself_mean_delta_vs_v45": nonself_mean,
        "nonself_per_opponent_mean_delta": nonself_per_opponent,
        "promotion_gate_met": bool(baseline_gate and (self_opponent is None or nonself_mean is not None and nonself_mean > 0)),
        "ledger_audit": audit_records(records, candidate, split),
    }


def run_batch(candidate: str, split: str, output: Path, max_games: int) -> dict[str, Any]:
    """Append at most ``max_games`` missing runs, then report paired progress."""

    if split not in {"development", "holdout"}:
        raise ValueError("split must be development or holdout")
    if max_games < 1:
        raise ValueError("max_games must be positive")
    factory = candidate_factory(candidate)
    records = read_records(output)
    audit = audit_records(records, candidate, split)
    if audit["duplicate_keys"]:
        raise RuntimeError(
            "Refusing to append to a ledger with duplicate condition keys for this candidate/split; "
            "preserve it for audit and use a new output path."
        )
    existing = {_record_key(row) for row in records}
    new_rows: list[dict[str, Any]] = []
    for condition in _conditions(split):
        for evaluated_policy, policy_factory in (("candidate", factory), ("v45_base", lambda: BASELINE)):
            key = (PROTOCOL, split, candidate, evaluated_policy, condition["opponent"], condition["seed"], condition["candidate_seat"])
            if key in existing:
                continue
            row = _run_one(
                candidate=candidate,
                evaluated_policy=evaluated_policy,
                factory=policy_factory,
                opponent=condition["opponent"],
                seed=condition["seed"],
                candidate_seat=condition["candidate_seat"],
            )
            row["split"] = split
            new_rows.append(row)
            # Persist every completed episode before starting another one.
            # Some public opponents are slow enough to outlive a short shell
            # window; losing an entire otherwise-complete batch would make the
            # advertised resume behaviour false.
            _append(output, [row])
            existing.add(key)
            if len(new_rows) >= max_games:
                break
        if len(new_rows) >= max_games:
            break
    updated = records + new_rows
    summary = summarize(updated, candidate, split)
    return {
        "protocol": PROTOCOL,
        "output": str(output),
        "new_games": len(new_rows),
        "remaining_games": len([key for key in _planned_keys(candidate, split) if key not in existing]),
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", default="v45_base")
    parser.add_argument("--split", choices=("development", "holdout"), required=True)
    parser.add_argument("--output", type=Path, default=Path("experiments/seven_opponent_league_v1.jsonl"))
    parser.add_argument("--max-games", type=int, default=8, help="Append this many missing games (default: 8).")
    args = parser.parse_args()
    print(json.dumps(run_batch(**vars(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
