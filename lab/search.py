"""Small, reproducible parameter search for the independently authored melon policy."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any

from .own_agent import MelonFarmConfig, make_melon_farm_agent
from .runner import run_match


DEFAULT_CANDIDATES = (
    MelonFarmConfig(sell_batch=30, min_price_ratio=0.50, emergency_shed_units=70),
    MelonFarmConfig(sell_batch=45, min_price_ratio=0.70, emergency_shed_units=70),
    MelonFarmConfig(sell_batch=60, min_price_ratio=0.70, emergency_shed_units=80),
    MelonFarmConfig(sell_batch=90, min_price_ratio=0.70, emergency_shed_units=80),
)
OPPONENTS = ("starter", "melon", "v45")
DEV_SEEDS = (0, 1)
HOLDOUT_SEEDS = (1000, 1001, 1002)


def _record(
    config: MelonFarmConfig,
    split: str,
    opponent: str,
    seed: int,
) -> dict[str, Any]:
    candidate = make_melon_farm_agent(config)
    match = run_match(candidate, opponent, steps=720, seed=seed)
    return {
        "candidate": config.label(),
        "config": asdict(config),
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


def _summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["candidate"]].append(record)
    return [
        {
            "candidate": candidate,
            "games": len(rows),
            "mean_reward": round(mean(row["candidate_reward"] for row in rows), 3),
            "mean_margin": round(mean(row["margin"] for row in rows), 3),
            "wins": sum(row["margin"] > 0 for row in rows),
            "losses": sum(row["margin"] < 0 for row in rows),
        }
        for candidate, rows in grouped.items()
    ]


def _write_records(output: Path, records: list[dict[str, Any]], append: bool = False) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a" if append else "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def run_development(output: Path) -> dict[str, Any]:
    """Evaluate all candidates only on the development split and write immediately."""

    development = [
        _record(config, "development", opponent, seed)
        for config in DEFAULT_CANDIDATES
        for opponent in OPPONENTS
        for seed in DEV_SEEDS
    ]
    _write_records(output, development)
    development_summary = sorted(_summarize(development), key=lambda row: row["mean_reward"], reverse=True)
    return {
        "output": str(output),
        "development_ranking": development_summary,
        "selected_for_holdout": [row["candidate"] for row in development_summary[:2]],
        "game_count": len(development),
    }


def run_holdout(output: Path, selected_names: tuple[str, ...]) -> dict[str, Any]:
    """Evaluate pre-selected candidates only on seeds never used for selection."""

    selected_configs = [config for config in DEFAULT_CANDIDATES if config.label() in selected_names]
    if not selected_configs:
        raise ValueError("No recognized candidates were selected for holdout")
    holdout = [
        _record(config, "holdout", opponent, seed)
        for config in selected_configs
        for opponent in OPPONENTS
        for seed in HOLDOUT_SEEDS
    ]
    _write_records(output, holdout, append=True)
    return {
        "output": str(output),
        "holdout_ranking": sorted(_summarize(holdout), key=lambda row: row["mean_reward"], reverse=True),
        "game_count": len(holdout),
    }


def run_search(output: Path, top_k: int = 2) -> dict[str, Any]:
    """Convenience entry point for environments that permit the full two-stage run."""

    development_result = run_development(output)
    selected_names = tuple(development_result["selected_for_holdout"][:top_k])
    holdout_result = run_holdout(output, selected_names)
    return {
        "output": str(output),
        "development_ranking": development_result["development_ranking"],
        "selected_for_holdout": list(selected_names),
        "holdout_ranking": holdout_result["holdout_ranking"],
        "game_count": development_result["game_count"] + holdout_result["game_count"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("experiments/melon_search_v1.jsonl"))
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--stage", choices=("all", "development", "holdout"), default="all")
    parser.add_argument("--selected", nargs="*", default=[])
    args = parser.parse_args()
    if args.stage == "development":
        result = run_development(args.output)
    elif args.stage == "holdout":
        result = run_holdout(args.output, tuple(args.selected))
    else:
        result = run_search(args.output, args.top_k)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
