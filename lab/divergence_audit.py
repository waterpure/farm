"""Audit persistent cash divergences in the frozen V45 diagnostic trace.

This is deliberately an analysis-only consumer of
``v45_opponent_diagnosis_v1.jsonl``.  It does not invoke an agent or create a
new game.  An event is eligible when its one-turn change in V45-minus-rival
cash is at least 25 and the net difference 24 states later remains in the
same direction by at least 25.  Market quantities remain *requests* unless a
strict single-item, sell-only transition can be reconstructed exactly.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from kaggle_environments.envs.kaggriculture.kaggriculture import SHOPS, TOWN_CENTER_PRODUCTS, market_price

from .diagnostics import PRODUCTS, TRACE_PROTOCOL, _int


AUDIT_PROTOCOL = "v45-near-peer-divergence-v1"
SOURCE = Path("experiments/v45_opponent_diagnosis_v1.jsonl")
TARGETS = ("sdy2842", "v46", "nathan_pipe7")
CONTROL = "kaito_v58"
SHOCK_THRESHOLD = 25
PERSISTENCE_STATES = 24
CONTEXT_RADIUS = 4
STABLE_OCCURRENCES = 6


def _condition_key(row: dict[str, Any]) -> tuple[str, int, str]:
    return (str(row["opponent"]), _int(row["seed"]), str(row["candidate_seat"]))


def _market_orders(player: dict[str, Any]) -> list[list[Any]]:
    action = player.get("previous_action") or {}
    market = action.get("market") if isinstance(action, dict) else []
    return [list(order) for order in market if isinstance(order, list)] if isinstance(market, list) else []


def _order_signature(orders: list[list[Any]]) -> tuple[str, ...]:
    """Preserve order position but remove quantity for repeatability grouping."""

    result = []
    for order in orders:
        if not order:
            continue
        if len(order) >= 2:
            result.append(f"{order[0]}:{order[1]}")
        else:
            result.append(str(order[0]))
    return tuple(result)


def _phase(step: int) -> str:
    if step <= 72:
        return "opening"
    if step <= 287:
        return "development"
    if step <= 647:
        return "market"
    return "terminal"


def _player_view(row: dict[str, Any], index: int) -> dict[str, Any]:
    player = row["players"][index]
    return {
        "money": _int(player["farm"]["money"]),
        "shed": dict(player["private"]["shed"]),
        "carried": dict(player["private"]["carried"]),
        "market_orders": _market_orders(player),
        "market_order_signature": list(_order_signature(_market_orders(player))),
    }


def _market_change(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    changed = {}
    for item in PRODUCTS:
        inventory_before = _int(before["market"]["inventory"].get(item))
        inventory_after = _int(after["market"]["inventory"].get(item))
        price_before = _int(before["market"]["prices"].get(item))
        price_after = _int(after["market"]["prices"].get(item))
        if inventory_before != inventory_after or price_before != price_after:
            changed[item] = {
                "inventory_before": inventory_before,
                "inventory_after": inventory_after,
                "inventory_delta": inventory_after - inventory_before,
                "price_before": price_before,
                "price_after": price_after,
            }
    return changed


def _single_sell(orders: list[list[Any]]) -> tuple[str, int] | None:
    if len(orders) != 1:
        return None
    order = orders[0]
    if len(order) < 3 or order[0] != "SELL" or order[1] not in PRODUCTS:
        return None
    quantity = _int(order[2])
    return (str(order[1]), quantity) if quantity > 0 else None


def _sale_proceeds(item: str, inventory: int, v45_fills: int, rival_fills: int) -> tuple[int, int, int]:
    """Replay the market's lockstep pricing for one pair of sell-only orders."""

    v45_money = 0
    rival_money = 0
    current = inventory
    shared = min(v45_fills, rival_fills)
    for _ in range(shared):
        price = market_price(item, current)
        v45_money += price
        rival_money += price
        current += 2
    for actor, remaining in (("v45", v45_fills - shared), ("rival", rival_fills - shared)):
        for _ in range(remaining):
            price = market_price(item, current)
            if actor == "v45":
                v45_money += price
            else:
                rival_money += price
            current += 1
    return v45_money, rival_money, current


def _town_inventory_delta(before: dict[str, Any], action_step: int, item: str) -> int:
    """Return deterministic town consumption after a transition's market.

    The diagnostic games use Kaggriculture's default shop/center intervals
    (4/24). Town consumption changes public inventory after cash settlement,
    so it must be removed before inventory can validate a fill reconstruction.
    """

    consumed = 0
    if action_step % 4 == 0:
        for shop_name in before.get("shops", []):
            products = SHOPS.get(shop_name, [])
            if item in products:
                consumed += 2 if len(products) == 1 else 1
    if action_step % 24 == 0 and item in TOWN_CENTER_PRODUCTS:
        consumed += 1
    return -consumed


def _infer_sell_fills(before: dict[str, Any], after: dict[str, Any], v45_index: int) -> dict[str, Any] | None:
    """Infer fills only for a transition with exactly one sell-only order each.

    Unit actions can put items into the shed before market settlement.  The
    trace does not preserve the per-worker carrying layout, so we never infer
    availability from aggregate inventory.  Instead, enumerate possible fill
    counts and accept an answer only when market inventory (after deterministic
    town consumption) and each player's cash delta identify one exact outcome
    under the official lockstep pricing.
    """

    rival_index = 1 - v45_index
    v_order = _single_sell(_market_orders(after["players"][v45_index]))
    r_order = _single_sell(_market_orders(after["players"][rival_index]))
    if v_order is None or r_order is None or v_order[0] != r_order[0]:
        return None
    item, v_requested = v_order
    _, r_requested = r_order
    inventory_before = _int(before["market"]["inventory"].get(item))
    inventory_after = _int(after["market"]["inventory"].get(item))
    observed_v45 = _int(after["players"][v45_index]["farm"]["money"]) - _int(before["players"][v45_index]["farm"]["money"])
    observed_rival = _int(after["players"][rival_index]["farm"]["money"]) - _int(before["players"][rival_index]["farm"]["money"])
    town_delta = _town_inventory_delta(before, _int(after["action_step"]), item)
    matches = []
    for v_fills in range(v_requested + 1):
        for r_fills in range(r_requested + 1):
            v_money, r_money, final_inventory = _sale_proceeds(item, inventory_before, v_fills, r_fills)
            if (
                final_inventory + town_delta == inventory_after
                and v_money == observed_v45
                and r_money == observed_rival
            ):
                matches.append((v_fills, r_fills))
    if len(matches) != 1:
        return None
    v_fills, r_fills = matches[0]
    return {
        "method": "exact_single_sell_lockstep_reconstruction",
        "item": item,
        "v45_requested": v_requested,
        "rival_requested": r_requested,
        "v45_filled": v_fills,
        "rival_filled": r_fills,
        "v45_unfilled": v_requested - v_fills,
        "rival_unfilled": r_requested - r_fills,
        "cash_delta_v45": observed_v45,
        "cash_delta_rival": observed_rival,
        "post_market_town_inventory_delta": town_delta,
    }


def _context(rows: dict[int, dict[str, Any]], step: int, v45_index: int) -> list[dict[str, Any]]:
    rival_index = 1 - v45_index
    result = []
    for context_step in range(max(0, step - CONTEXT_RADIUS), min(max(rows), step + CONTEXT_RADIUS) + 1):
        row = rows[context_step]
        result.append(
            {
                "state_step": context_step,
                "cash_gap_v45_minus_rival": _int(row["players"][v45_index]["farm"]["money"]) - _int(row["players"][rival_index]["farm"]["money"]),
                "market_orders_v45": _market_orders(row["players"][v45_index]),
                "market_orders_rival": _market_orders(row["players"][rival_index]),
                "market": row["market"],
            }
        )
    return result


def _event(before: dict[str, Any], after: dict[str, Any], later: dict[str, Any], v45_index: int) -> dict[str, Any]:
    rival_index = 1 - v45_index
    gap_before = _int(before["players"][v45_index]["farm"]["money"]) - _int(before["players"][rival_index]["farm"]["money"])
    gap_after = _int(after["players"][v45_index]["farm"]["money"]) - _int(after["players"][rival_index]["farm"]["money"])
    gap_later = _int(later["players"][v45_index]["farm"]["money"]) - _int(later["players"][rival_index]["farm"]["money"])
    v_orders = _market_orders(after["players"][v45_index])
    r_orders = _market_orders(after["players"][rival_index])
    return {
        "state_step": after["state_step"],
        "action_step": after["action_step"],
        "phase": _phase(after["state_step"]),
        "cash_gap_before": gap_before,
        "cash_gap_after": gap_after,
        "one_turn_shock": gap_after - gap_before,
        "cash_gap_after_24": gap_later,
        "net_change_after_24": gap_later - gap_before,
        "market_orders": {"v45": v_orders, "rival": r_orders},
        "market_order_signature": {"v45": list(_order_signature(v_orders)), "rival": list(_order_signature(r_orders))},
        "state_before": {
            "v45": _player_view(before, v45_index),
            "rival": _player_view(before, rival_index),
            "market": before["market"],
            "shops": before["shops"],
        },
        "state_after": {
            "v45": _player_view(after, v45_index),
            "rival": _player_view(after, rival_index),
            "market": after["market"],
            "shops": after["shops"],
        },
        "market_change": _market_change(before, after),
        "fill_inference": _infer_sell_fills(before, after, v45_index),
    }


def _accepted_games(path: Path) -> dict[tuple[str, int, str], dict[int, dict[str, Any]]]:
    turns: dict[tuple[str, int, str], dict[int, dict[str, Any]]] = defaultdict(dict)
    accepted: set[tuple[str, int, str]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("protocol") != TRACE_PROTOCOL:
            continue
        key = _condition_key(row)
        if row.get("record_type") == "turn":
            turns[key][_int(row["state_step"])] = row
        elif row.get("record_type") == "game_end" and row.get("terminal_matches_frozen_league"):
            accepted.add(key)
    return {key: turns[key] for key in accepted if len(turns[key]) == 720}


def audit(path: Path = SOURCE) -> dict[str, Any]:
    games = _accepted_games(path)
    candidates: list[dict[str, Any]] = []
    condition_counts: dict[str, int] = defaultdict(int)
    for key, rows in sorted(games.items()):
        opponent, seed, seat = key
        if opponent not in (*TARGETS, CONTROL):
            continue
        v45_index = 0 if seat == "left" else 1
        for step in range(1, 720 - PERSISTENCE_STATES):
            before, after, later = rows[step - 1], rows[step], rows[step + PERSISTENCE_STATES]
            item = _event(before, after, later, v45_index)
            shock = item["one_turn_shock"]
            persistence = item["net_change_after_24"]
            if abs(shock) < SHOCK_THRESHOLD or abs(persistence) < SHOCK_THRESHOLD or shock * persistence <= 0:
                continue
            item.update({"opponent": opponent, "seed": seed, "candidate_seat": seat})
            item["context"] = _context(rows, step, v45_index)
            candidates.append(item)
            condition_counts[opponent] += 1

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        signature = (
            item["opponent"], item["state_step"], item["phase"],
            tuple(item["market_order_signature"]["v45"]),
            tuple(item["market_order_signature"]["rival"]),
            1 if item["one_turn_shock"] > 0 else -1,
        )
        groups[signature].append(item)

    stable = []
    for signature, items in groups.items():
        if len(items) < STABLE_OCCURRENCES:
            continue
        stable.append(
            {
                "opponent": signature[0],
                "state_step": signature[1],
                "phase": signature[2],
                "v45_order_signature": list(signature[3]),
                "rival_order_signature": list(signature[4]),
                "shock_direction": "v45_gain" if signature[5] > 0 else "v45_loss",
                "occurrences": len(items),
                "mean_one_turn_shock": round(mean(item["one_turn_shock"] for item in items), 3),
                "mean_net_change_after_24": round(mean(item["net_change_after_24"] for item in items), 3),
                "mean_cash_gap_after": round(mean(item["cash_gap_after"] for item in items), 3),
                "exact_fill_reconstructions": [item["fill_inference"] for item in items if item["fill_inference"]],
                "conditions": [
                    {"seed": item["seed"], "candidate_seat": item["candidate_seat"], "event": item}
                    for item in items
                ],
            }
        )
    stable.sort(key=lambda item: (item["opponent"], item["state_step"], item["shock_direction"]))
    return {
        "protocol": AUDIT_PROTOCOL,
        "source_trace": str(path),
        "scope": {
            "targets": list(TARGETS),
            "control": CONTROL,
            "accepted_games_in_scope": {opponent: sum(1 for key in games if key[0] == opponent) for opponent in (*TARGETS, CONTROL)},
            "shock_threshold": SHOCK_THRESHOLD,
            "persistence_states": PERSISTENCE_STATES,
            "stable_occurrences_required": STABLE_OCCURRENCES,
        },
        "interpretation_limits": [
            "Most market quantities are requested orders, not confirmed fills.",
            "Exact fills are reported only when one sell-only order per player, market inventory, and both cash deltas yield exactly one official lockstep replay.",
            "A persistent cash shock is evidence of a repeatable transition, not proof that the displayed order is a globally optimal policy change.",
        ],
        "candidate_event_counts": dict(sorted(condition_counts.items())),
        "stable_event_groups": stable,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=Path("experiments/v45_near_peer_divergence_v1.json"))
    args = parser.parse_args()
    result = audit(args.input)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "candidate_event_counts": result["candidate_event_counts"],
        "stable_event_groups": len(result["stable_event_groups"]),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
