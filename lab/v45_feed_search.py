"""Two-stage local evaluation for the V45 R85 feed-multiplier variants."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .runner import run_match
from .v45_variants import make_feed_multiplier_agent


CHALLENGERS = (1.15, 1.35)
OPPONENTS = ("v45_base", "starter", "melon")
DEV_SEEDS = (0, 1, 2)
HOLDOUT_SEEDS = (1000, 1001, 1002)


def _label(multiplier: float) -> str:
    return f"v45-feed-m{multiplier:.2f}"


def _record(multiplier: float, split: str, opponent: str, seed: int) -> dict[str, Any]:
    candidate = make_feed_multiplier_agent(multiplier)
    match = run_match(candidate, opponent, steps=720, seed=seed)
    return {
        "candidate": _label(multiplier),
        "feed_multiplier": multiplier,
        "split": split,
        "opponent": opponent,
        "seed": seed,
        "candidate_reward": float(match["rewards"][0]),
        "opponent_reward": float(match["rewards"][1]),
        "margin": float(match["rewards"][0]) - float(match["rewards"][1]),
        "elapsed_seconds": match["elapsed_seconds"],
        "statuses": match["statuses"],
        "telemetry": match["telemetry"][0],
    }


def _summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[row["candidate"]].append(row)
    rows: list[dict[str, Any]] = []
    for candidate, matches in grouped.items():
        margins_by_opponent = {
            opponent: round(mean(row["margin"] for row in matches if row["opponent"] == opponent), 3)
            for opponent in OPPONENTS
        }
        rows.append(
            {
                "candidate": candidate,
                "games": len(matches),
                "mean_reward": round(mean(row["candidate_reward"] for row in matches), 3),
                "mean_margin_vs_v45_base": margins_by_opponent["v45_base"],
                "mean_margin_by_opponent": margins_by_opponent,
                "eligible": margins_by_opponent["starter"] >= 0 and margins_by_opponent["melon"] >= 0,
            }
        )
    return sorted(rows, key=lambda row: (row["eligible"], row["mean_margin_vs_v45_base"], row["mean_reward"]), reverse=True)


def _write(output: Path, rows: list[dict[str, Any]], append: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a" if append else "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def run_development(output: Path) -> dict[str, Any]:
    records = [
        _record(multiplier, "development", opponent, seed)
        for multiplier in CHALLENGERS
        for opponent in OPPONENTS
        for seed in DEV_SEEDS
    ]
    _write(output, records, append=False)
    ranking = _summary(records)
    selected = next((row["candidate"] for row in ranking if row["eligible"]), None)
    return {"output": str(output), "development_ranking": ranking, "selected_for_holdout": selected, "game_count": len(records)}


def run_holdout(output: Path, selected: str) -> dict[str, Any]:
    lookup = {_label(multiplier): multiplier for multiplier in CHALLENGERS}
    if selected not in lookup:
        raise ValueError(f"Unknown feed-multiplier candidate: {selected}")
    multiplier = lookup[selected]
    records = [
        _record(multiplier, "holdout", opponent, seed)
        for opponent in OPPONENTS
        for seed in HOLDOUT_SEEDS
    ]
    _write(output, records, append=True)
    return {"output": str(output), "holdout_ranking": _summary(records), "game_count": len(records)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("experiments/v45_feed_multiplier_v1.jsonl"))
    parser.add_argument("--stage", choices=("development", "holdout"), required=True)
    parser.add_argument("--selected", help="Candidate name fixed by the development result")
    args = parser.parse_args()
    if args.stage == "development":
        result = run_development(args.output)
    else:
        if not args.selected:
            parser.error("--selected is required for holdout")
        result = run_holdout(args.output, args.selected)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
