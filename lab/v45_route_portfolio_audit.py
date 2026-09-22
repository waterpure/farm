"""Profile V45's day-6 route tapes and check their prefix compatibility.

Changing a V45 route at step 144 is safe only if the candidate route was
written for the same first-six-day state.  This static audit never runs an
environment and never changes an agent; it compares the immutable route tapes
loaded from the frozen source.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .baselines import load_v45_base_module


PROTOCOL = "v45-portfolio-route-audit-v1"
ROUTE_SWITCH_STEP = 144
TERMINAL_ROUTE_STEP = 648
ANIMAL_OUTPUT = {"GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL"}


def _signature(actions: list[dict[str, Any]]) -> str:
    return json.dumps(actions, sort_keys=True, separators=(",", ":"))


def _route_profile(tape: list[dict[str, Any]]) -> dict[str, Any]:
    plants, placed_animals, purchases = Counter(), Counter(), Counter()
    for action in tape[ROUTE_SWITCH_STEP:TERMINAL_ROUTE_STEP]:
        for command in [action.get("farmer") or ["PASS"], *(action.get("hands") or [])]:
            if len(command) > 1 and command[0] == "PLANT":
                plants[str(command[1])] += 1
            elif len(command) > 1 and command[0] == "PLACE" and command[1] in ANIMAL_OUTPUT:
                placed_animals[ANIMAL_OUTPUT[command[1]]] += 1
        for order in action.get("market") or []:
            if not order or order[0] not in {"BUY_SEED", "BUY_ANIMAL", "BUY_LAND", "HIRE"}:
                continue
            label = str(order[0]) if len(order) == 1 else f"{order[0]} {order[1]}"
            purchases[label] += int(order[2]) if len(order) > 2 else 1
    production = Counter(plants)
    production.update(placed_animals)
    return {
        "planned_plants": dict(sorted(plants.items())),
        "planned_animal_outputs": dict(sorted(placed_animals.items())),
        "planned_output_births": dict(sorted(production.items())),
        "planned_investment_order_units": dict(sorted(purchases.items())),
    }


def audit() -> dict[str, Any]:
    module = load_v45_base_module()
    routes = {int(route_id): list(tape) for route_id, tape in module._ROUTES.items()}
    actual_prefix = _signature(routes[0][:ROUTE_SWITCH_STEP])
    compatible = [route_id for route_id, tape in routes.items() if _signature(tape[:ROUTE_SWITCH_STEP]) == actual_prefix]
    shop_routes = {" | ".join(key): int(value) for key, value in module._R108_SHOP_ROUTES.items()}
    old_shop_routes = {" | ".join(key): int(value) for key, value in module._R110_OLD_SHOPS.items()}
    return {
        "protocol": PROTOCOL,
        "route_switch_step": ROUTE_SWITCH_STEP,
        "terminal_route_step": TERMINAL_ROUTE_STEP,
        "all_route_ids": sorted(routes),
        "actual_first_144_route": 0,
        "prefix_compatible_route_ids": sorted(compatible),
        "prefix_compatible_alternatives": [route_id for route_id in sorted(compatible) if route_id != 0],
        "shop_router_new_map": shop_routes,
        "shop_router_old_map": old_shop_routes,
        "profiles": {str(route_id): _route_profile(tape) for route_id, tape in sorted(routes.items())},
        "limits": [
            "Exact prefix equality is only a necessary state-compatibility condition; it is not an economic proof.",
            "The profile counts planned PLANT/PLACE births, not actual successful output or terminal cash.",
            "No route selection or agent action is changed by this audit.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("experiments/v45_portfolio_route_audit_v1.json"))
    args = parser.parse_args()
    result = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
