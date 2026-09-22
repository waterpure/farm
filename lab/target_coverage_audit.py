"""Offline check of the user's per-product 70%-of-known-demand target.

This consumes the read-only replay harvest ledger.  It deliberately does not
choose a route in a live game: actual future harvest is available only after a
replay.  The audit asks whether the *shape* of a future live selector is
sound before anyone attempts to predict those harvests at day 6.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TARGET_SHARE = 0.70


def _decision(case: dict[str, Any]) -> dict[str, Any]:
    remaining_days = int(case["remaining_days"])
    products: list[dict[str, Any]] = []
    coverage_gain = 0.0
    worsened_excess: list[str] = []
    for row in case["products"]:
        product = str(row["product"])
        capacity = float(row["known_shop_capacity"])
        per_day = capacity / remaining_days
        # Town Center's one unit/day is a universal floor, not evidence that a
        # product is currently a scarce specialist line.  A product becomes
        # "tight" only when an opened specialist/multi-product shop adds more.
        tight = per_day > 1.0
        target = capacity * TARGET_SHARE
        candidate = float(row["candidate_harvested"])
        baseline = float(row["baseline_harvested"])
        candidate_covered = min(candidate, target) if tight else 0.0
        baseline_covered = min(baseline, target) if tight else 0.0
        candidate_excess = max(0.0, candidate - target) if tight else 0.0
        baseline_excess = max(0.0, baseline - target) if tight else 0.0
        gain = candidate_covered - baseline_covered
        excess_change = candidate_excess - baseline_excess
        if tight:
            coverage_gain += gain
            if excess_change > 1e-9:
                worsened_excess.append(product)
        products.append(
            {
                "product": product,
                "tight_from_open_shops": tight,
                "known_buy_per_day": per_day,
                "target_70_percent": target,
                "candidate_harvested": candidate,
                "baseline_harvested": baseline,
                "candidate_covered_toward_target": candidate_covered,
                "baseline_covered_toward_target": baseline_covered,
                "coverage_gain_vs_baseline": gain,
                "candidate_excess_over_target": candidate_excess,
                "baseline_excess_over_target": baseline_excess,
                "excess_change_vs_baseline": excess_change,
            }
        )
    if coverage_gain > 1e-9 and not worsened_excess:
        decision = "would_pass_target_screen"
        reason = "fills an open-shop 70% target without worsening another tight product's excess"
    elif worsened_excess:
        decision = "reject_candidate"
        reason = "adds excess to already-covered tight product(s): " + ", ".join(worsened_excess)
    else:
        decision = "abstain_keep_v45"
        reason = "does not fill any additional open-shop 70% target"
    return {
        "seed": case["seed"],
        "shops": case["shops"],
        "original_route": case["original_route"],
        "candidate_route": case["selector_event"]["selected_route"],
        "actual_cash_delta_vs_v45": case["delta_vs_v45"],
        "coverage_gain_total": coverage_gain,
        "worsened_tight_product_excess": worsened_excess,
        "screen_decision": decision,
        "reason": reason,
        "products": products,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("experiments/v45_route_supply_demand_audit_v3.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/v45_target_coverage_audit_v1.json"),
    )
    args = parser.parse_args()
    ledger = json.loads(args.input.read_text(encoding="utf-8"))
    if ledger.get("protocol") != "known-shop-supply-demand-audit-v3-transition-aligned":
        raise RuntimeError("Input must be the transition-aligned v3 supply-demand audit")
    cases = [_decision(case) for case in ledger["cases"]]
    # All three routes already cover every currently tight product to its 70%
    # target.  The +606 seed-1 outcome is not a reason to add extra strawberry
    # production after that target is filled, so the user's rule rejects it too.
    expected = {1: "reject_candidate", 2: "reject_candidate", 3: "reject_candidate"}
    actual = {case["seed"]: case["screen_decision"] for case in cases}
    if actual != expected:
        raise AssertionError(f"70% target screen drift: expected {expected}, got {actual}")
    report = {
        "protocol": "v45-target-coverage-audit-v1-retrospective",
        "target_share": TARGET_SHARE,
        "limitations": [
            "Uses realized replay harvest after day 6; it is an offline rule check, not live route selection.",
            "Future shop openings and opponent private state are intentionally excluded.",
            "A live selector would need a separately validated day-6 production forecast before using this screen.",
        ],
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "decisions": actual}, ensure_ascii=False))


if __name__ == "__main__":
    main()
