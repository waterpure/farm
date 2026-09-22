"""Deterministic local batch evaluation for candidate Kaggriculture agents."""

from __future__ import annotations

import argparse
import json
from statistics import mean, pstdev
from typing import Any

from .runner import AGENTS, run_match


def evaluate_pair(
    left: str,
    right: str,
    *,
    seed_start: int,
    count: int,
    steps: int = 720,
) -> dict[str, Any]:
    """Evaluate fixed agents on consecutive seeds and summarize terminal rewards."""

    if count < 1:
        raise ValueError("count must be positive")

    matches = [
        run_match(left, right, steps=steps, seed=seed)
        for seed in range(seed_start, seed_start + count)
    ]
    left_scores = [float(match["rewards"][0]) for match in matches]
    right_scores = [float(match["rewards"][1]) for match in matches]
    margins = [left_score - right_score for left_score, right_score in zip(left_scores, right_scores)]

    return {
        "left": left,
        "right": right,
        "steps": steps,
        "seed_start": seed_start,
        "count": count,
        "left_mean": round(mean(left_scores), 3),
        "right_mean": round(mean(right_scores), 3),
        "margin_mean": round(mean(margins), 3),
        "margin_stddev": round(pstdev(margins), 3),
        "left_wins": sum(margin > 0 for margin in margins),
        "ties": sum(margin == 0 for margin in margins),
        "right_wins": sum(margin < 0 for margin in margins),
        "elapsed_seconds_total": round(sum(float(match["elapsed_seconds"]) for match in matches), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", choices=sorted(AGENTS), default="v45_base")
    parser.add_argument("--right", choices=sorted(AGENTS), default="starter")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--steps", type=int, default=720)
    args = parser.parse_args()
    print(json.dumps(evaluate_pair(**vars(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
