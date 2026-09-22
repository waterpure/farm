"""Read-only audit of V45's day-6 shop-to-route choices.

It deliberately ignores future shops, future prices and conditional overlays.
For every currently observable two-shop state, record the route V45 actually
selects and the route tape's own crop/animal commitments.  These commitments
are not a production forecast: they only show what the fixed route is trying
to build before later reactive layers intervene.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .baselines import load_v45_base_module
from .portfolio_model import ANIMAL_OUTPUT, PRODUCTS, shop_demand_per_day


ANIMALS = frozenset(ANIMAL_OUTPUT)


def _route_commitments(module: Any, route: int) -> dict[str, dict[str, int]]:
    plants: Counter[str] = Counter()
    animal_places: Counter[str] = Counter()
    animal_buys: Counter[str] = Counter()
    for action in module._IMPL.chassis.routes[route][144:648]:
        for command in [action.get("farmer"), *(action.get("hands") or [])]:
            if not command:
                continue
            if command[0] == "PLANT" and len(command) > 1 and command[1] in PRODUCTS:
                plants[command[1]] += 1
            if command[0] == "PLACE" and len(command) > 1 and command[1] in ANIMALS:
                animal_places[ANIMAL_OUTPUT[command[1]]] += 1
        for order in action.get("market", []):
            if order and order[0] == "BUY_ANIMAL" and len(order) > 2 and order[1] in ANIMALS:
                animal_buys[ANIMAL_OUTPUT[order[1]]] += int(order[2])
    return {
        "raw_plant_commands": dict(sorted(plants.items())),
        "raw_animal_place_commands": dict(sorted(animal_places.items())),
        "raw_animal_buy_units": dict(sorted(animal_buys.items())),
    }


def _selected_route(module: Any, shops: tuple[str, str]) -> tuple[int, str]:
    # This is V45's exact router rule at step144.  The old V39 table is used
    # whenever a Yarn Store is already visible; otherwise EXP240 is used.
    if shops.count("YARN_STORE"):
        return int(module._R110_OLD_SHOPS.get(shops, 0)), "V39/Yarn branch"
    return int(module._R108_SHOP_ROUTES.get(shops, 100)), "EXP240/non-Yarn branch"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("experiments/v45_day6_known_demand_route_audit_v1.json"))
    args = parser.parse_args()
    module = load_v45_base_module()
    shop_pairs = sorted({tuple(pair) for pair in module._R108_SHOP_ROUTES} | {tuple(pair) for pair in module._R110_OLD_SHOPS})
    cache: dict[int, dict[str, dict[str, int]]] = {}
    rows = []
    for shops in shop_pairs:
        route, branch = _selected_route(module, shops)
        cache.setdefault(route, _route_commitments(module, route))
        demand = {product: shop_demand_per_day(list(shops), product) for product in PRODUCTS}
        rows.append(
            {
                "shops": list(shops),
                "selected_route": route,
                "router_branch": branch,
                "known_demand_per_day": demand,
                "tight_products": [product for product in PRODUCTS if demand[product] > 1],
                "route_raw_commitments": cache[route],
            }
        )
    report = {
        "protocol": "day6-known-demand-route-audit-v1-current-shops-only",
        "limitations": [
            "A raw PLANT/PLACE command is a route commitment, not a guarantee it will execute or be harvested.",
            "No future shop, future price, conditional V45 overlay, rival private state or final cash is used.",
            "This audit describes V45 before proposing any replacement route rule.",
        ],
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "shop_pairs": len(rows), "routes_used": sorted(cache)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
