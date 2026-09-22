"""Rank v3 latent-slot opportunities without changing or replaying V45.

This is deliberately a selector over completed shadow traces, not an agent.
Each game may nominate at most one crop/slot.  Eligibility came from the v3
immediate-service observer; ranking is crop- and opponent-agnostic:
conservative expected sale value minus the seed's opportunity cost, then the
earlier scheduled sale as a deterministic tie-break.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SOURCE_PROTOCOL = "v45-latent-slot-dryrun-v3-immediate-service"
PROTOCOL = "v45-latent-slot-top1-v1"
DEFAULT_INPUT = Path("experiments/v45_latent_slot_dryrun_v3.jsonl")
DEFAULT_OUTPUT = Path("experiments/v45_latent_slot_top1_v1.jsonl")
DEFAULT_SUMMARY = Path("experiments/v45_latent_slot_top1_v1_summary.json")


def _key(row: dict[str, Any]) -> tuple[str, int, str]:
    return str(row["opponent"]), int(row["seed"]), str(row["candidate_seat"])


def candidates(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten and rank V3-eligible crop choices; no crop-specific bonus."""
    found = []
    for slot in row["latent_slots"]:
        for crop in slot["supported_crops"]:
            found.append(
                {
                    "step": slot["step"],
                    "day": slot["day"],
                    "actor": slot["actor"],
                    "coordinate": slot["coordinate"],
                    "same_actor_immediate_water_steps": slot["same_actor_immediate_water_steps"],
                    "crop": crop["crop"],
                    "seed_available": crop["seed_available"],
                    "estimated_yield_units": crop["estimated_yield_units"],
                    "conservative_unit_price": crop["conservative_unit_price"],
                    "estimated_net_before_labor": crop["estimated_net_before_labor"],
                    "harvest_step": crop["harvest_step"],
                    "sell_step": crop["sell_step"],
                    "qualifying_water_count": crop["qualifying_water_count"],
                }
            )
    return sorted(
        found,
        key=lambda item: (
            -item["estimated_net_before_labor"],
            item["sell_step"],
            item["step"],
            item["actor"],
            item["coordinate"],
            item["crop"],
        ),
    )


def select(source: Path, output: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    valid = [
        row
        for row in rows
        if row.get("protocol") == SOURCE_PROTOCOL and row.get("terminal_matches_frozen_league")
    ]
    selected = []
    for row in valid:
        ranked = candidates(row)
        selected.append(
            {
                "record_type": "selection",
                "protocol": PROTOCOL,
                "source_protocol": SOURCE_PROTOCOL,
                "opponent": row["opponent"],
                "seed": row["seed"],
                "candidate_seat": row["candidate_seat"],
                "source_terminal_matches_frozen_league": True,
                "eligible_candidate_count": len(ranked),
                "selection": ranked[0] if ranked else None,
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in selected),
        encoding="utf-8",
    )
    return selected


def summarize(rows: list[dict[str, Any]], source: Path) -> dict[str, Any]:
    crops = Counter()
    by_opponent: dict[str, Counter[str]] = defaultdict(Counter)
    sdy_target = []
    no_selection = 0
    for row in rows:
        chosen = row["selection"]
        if chosen is None:
            no_selection += 1
            continue
        crops[chosen["crop"]] += 1
        by_opponent[row["opponent"]][chosen["crop"]] += 1
        if (
            row["opponent"] == "sdy2842"
            and chosen["step"] == 18
            and chosen["coordinate"] == [0, 4]
            and chosen["crop"] == "MELON"
        ):
            sdy_target.append({
                "seed": row["seed"],
                "candidate_seat": row["candidate_seat"],
                "selection": chosen,
            })
    return {
        "protocol": PROTOCOL,
        "source": str(source),
        "source_conditions": len(rows),
        "selected_conditions": len(rows) - no_selection,
        "no_eligible_candidate_conditions": no_selection,
        "selection_rule": {
            "eligibility": "V3 immediate-service candidate only",
            "primary_rank": "highest conservative expected sale value minus seed opportunity cost",
            "tie_break": "earlier scheduled sell step, then stable position ordering",
            "crop_or_opponent_specific_bonus": False,
        },
        "selected_crop_counts": dict(sorted(crops.items())),
        "selected_crop_counts_by_opponent": {
            opponent: dict(sorted(counts.items()))
            for opponent, counts in sorted(by_opponent.items())
        },
        "sdy_target_selected": sdy_target,
        "limits": [
            "This ranks the v3 shadow candidates; it does not replay an action or prove counterfactual profit.",
            "The score has no future market forecast, labor cost, or reinvestment model.",
            "A future V45 sell request is evidence of a planned sale window, not a guarantee that added output fills at that price.",
        ],
    }


def main() -> None:
    rows = select(DEFAULT_INPUT, DEFAULT_OUTPUT)
    summary = summarize(rows, DEFAULT_INPUT)
    DEFAULT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
