"""Small, named V45-derived variants for local A/B evaluation.

Each factory loads an isolated copy of the frozen third-party source, then
overrides exactly one documented helper.  The frozen file is never modified.
The variant implementation remains a derivative: notices and provenance live
beside the parent source and must accompany any future packaging review.
"""

from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Callable
from types import ModuleType
from typing import Any

from .animal_forecast import forecast_animal_units
from .baselines import AgentFunction, load_v45_base_module
from .portfolio_model import ANIMAL_OUTPUT, evaluate_lines, shop_demand_per_day


def _install_feed_multiplier(module: ModuleType, multiplier: float) -> None:
    """Replace only V45 R85's economic threshold, preserving all other logic."""

    def feed(obs: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
        step = int(obs["step"])
        day = step // 24
        if not 10 <= day <= 28 or step % 24 > 21:
            return action
        player = int(obs["player"])
        native = module._IMPL.chassis.players[player]
        tape = module._v219_native_day(native, day)
        expected = max(len(item.get("hands", [])) for item in tape)
        farm = obs["farms"][player]
        positions = [farm["farmer"], *farm["hands"]]
        commands = [action.get("farmer") or ["PASS"], *(action.get("hands") or [])]
        prices = obs["market"]["prices"]
        changed = False

        for actor, command in enumerate(commands[: expected + 1]):
            if command != ["FEED"] or actor >= len(positions):
                continue
            tile = module._tile_at(farm["tiles"], positions[actor])
            if not isinstance(tile, dict) or tile.get("animal") not in ("GOOSE", "COW", "SHEEP"):
                continue
            if tile.get("fed_today") or int(tile.get("consecutive_unfed", 0)) != 0:
                continue
            if int(obs["private"]["inventories"][actor].get("WHEAT", 0)) <= 0:
                continue
            item = {"GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL"}[tile["animal"]]
            bonus = module._r88_feed_bonus_cost(tile, day)
            if bonus * (float(prices[item]) + 5) * multiplier >= float(prices["WHEAT"]):
                continue
            if not module._r86_next_feed(obs, positions[actor]):
                continue
            commands[actor] = ["PASS"]
            changed = True
            module._R85_REPORT["feed_skips"] += 1

        if not changed:
            return action
        result = copy.deepcopy(action)
        result["farmer"], result["hands"] = commands[0], commands[1:]
        return result

    module._r85_feed = feed


def make_feed_multiplier_agent(multiplier: float) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Make a V45 variant with one changed R85 feed-value multiplier.

    The parent uses 1.25.  A lower number skips feed more readily; a higher
    number is more conservative and keeps feed more often.
    """

    if not 0.5 <= multiplier <= 2.0:
        raise ValueError("feed multiplier must be between 0.5 and 2.0")
    module = load_v45_base_module()
    _install_feed_multiplier(module, multiplier)
    agent = getattr(module, "agent", None)
    if not callable(agent):
        raise RuntimeError("V45 base did not expose a callable agent")
    agent.label = f"v45-feed-m{multiplier:.2f}"  # type: ignore[attr-defined]
    return agent


# A route is eligible only if it has exactly the same opening tape as route 0.
# This is deliberately computed from the frozen source at install time rather
# than copied from a historical audit artifact: a variant must prove the
# compatibility condition against the parent it is actually wrapping.
_ANIMAL_PRODUCTS = frozenset(ANIMAL_OUTPUT.values())
_TARGET_SHARE = 0.70
_SEASON_DAYS = 30


def _compatible_animal_routes(module: ModuleType) -> tuple[int, ...]:
    routes = module._IMPL.chassis.routes
    opening = routes[0][:144]
    return tuple(route for route in sorted(routes) if route != 1 and routes[route][:144] == opening)


def _route_animal_buys(module: ModuleType, route: int) -> Counter[str]:
    """Count the full post-day-6 animal commitment written in one route tape.

    This is not an estimate of animals that will certainly be placed.  It is a
    comparable description of the *complete route* which will also supply the
    workers, feed, structures and product-delivery actions for those purchases.
    """

    counts: Counter[str] = Counter()
    for action in module._IMPL.chassis.routes[route][144:648]:
        for order in action.get("market", []):
            if len(order) >= 3 and order[0] == "BUY_ANIMAL" and order[1] in ANIMAL_OUTPUT:
                counts[ANIMAL_OUTPUT[order[1]]] += int(order[2])
    return counts


def _owned_animal_outputs(observation: dict[str, Any]) -> Counter[str]:
    player = int(observation.get("player", 0))
    farms = list(observation.get("farms") or [])
    farm = farms[player] if 0 <= player < len(farms) else {}
    counts: Counter[str] = Counter()
    for row in farm.get("tiles") or []:
        for tile in row:
            if isinstance(tile, dict) and tile.get("animal") in ANIMAL_OUTPUT:
                counts[ANIMAL_OUTPUT[tile["animal"]]] += 1
    return counts


def _animal_targets(observation: dict[str, Any]) -> tuple[str, str] | None:
    """Return the two best distinct animal products using only this observation."""

    by_output: dict[str, float] = {}
    for line in evaluate_lines(observation):
        if line.kind != "animal" or not line.feasible_before_terminal or line.score <= 0:
            continue
        by_output[line.output] = max(by_output.get(line.output, float("-inf")), line.score)
    ranked = sorted(by_output, key=lambda output: (-by_output[output], output))
    return (ranked[0], ranked[1]) if len(ranked) >= 2 else None


def _route_composition_loss(
    existing: Counter[str],
    planned: Counter[str],
    primary: str,
    secondary: str,
) -> float:
    """Distance from the user's 70/30 target across all actual animal tiles.

    Any third animal line is counted explicitly as off-target capacity.  This
    makes a 70/30 wool/milk request prefer a route that really concentrates on
    those two lines over a superficially similar route padded with geese.
    """

    combined = existing + planned
    total = sum(combined.values())
    if total <= 0:
        return float("inf")
    return (
        abs(combined[primary] / total - 0.70)
        + abs(combined[secondary] / total - 0.30)
        + sum(amount for output, amount in combined.items() if output not in (primary, secondary)) / total
    )


def _route_family(module: ModuleType, route: int) -> str:
    """Yarn-table routes and EXP240 routes are not interchangeable shop plans."""

    yarn = {int(value) for value in module._R110_OLD_SHOPS.values()}
    exp = {int(value) for value in module._R108_SHOP_ROUTES.values()}
    if route in yarn:
        return "yarn"
    if route in exp or route in {2, 100}:
        return "exp"
    return "other"


def _family_routes(module: ModuleType, original_route: int) -> tuple[int, ...]:
    family = _route_family(module, original_route)
    return tuple(
        route
        for route in _compatible_animal_routes(module)
        if _route_family(module, route) == family and (route != 0 or route == original_route)
    )


def _tight_animal_targets(observation: dict[str, Any]) -> dict[str, float]:
    shops = list((observation.get("town") or {}).get("unlocked_shops") or [])
    remaining_days = max(0, _SEASON_DAYS - int(observation.get("day", 0)))
    targets: dict[str, float] = {}
    for product in _ANIMAL_PRODUCTS:
        per_day = shop_demand_per_day(shops, product)
        if per_day > 1.0:
            targets[product] = per_day * remaining_days * _TARGET_SHARE
    return targets


def _coverage_delta(candidate: Counter[str], baseline: Counter[str], targets: dict[str, float]) -> tuple[float, list[str]]:
    gain = 0.0
    worsened: list[str] = []
    for product, target in targets.items():
        cand = float(candidate.get(product, 0))
        base = float(baseline.get(product, 0))
        gain += min(cand, target) - min(base, target)
        if max(0.0, cand - target) > max(0.0, base - target) + 1e-9:
            worsened.append(product)
    return gain, worsened


def _select_coverage_route(module: ModuleType, observation: dict[str, Any], original_route: int) -> dict[str, Any]:
    """Keep V45's shop family; switch only to uniquely better 70% coverage."""

    targets = _tight_animal_targets(observation)
    family = _route_family(module, original_route)
    candidates = _family_routes(module, original_route)
    if original_route not in candidates:
        return {"route": original_route, "reason": "original route is outside the selectable family", "family": family}
    if not targets:
        return {"route": original_route, "reason": "no tight animal products from open shops", "family": family}
    baseline = forecast_animal_units(observation, module, original_route)
    scored: dict[int, dict[str, Any]] = {}
    for route in candidates:
        forecast = forecast_animal_units(observation, module, route)
        gain, worsened = _coverage_delta(forecast, baseline, targets)
        scored[route] = {
            "forecast": dict(forecast),
            "coverage_gain": round(gain, 6),
            "worsened_excess": worsened,
        }
    improvers = [
        route
        for route, row in scored.items()
        if route != original_route and row["coverage_gain"] > 1e-9 and not row["worsened_excess"]
    ]
    if not improvers:
        return {
            "route": original_route,
            "reason": "no same-family route fills extra 70% demand without extra excess",
            "family": family,
            "targets": {key: round(value, 4) for key, value in targets.items()},
            "original_forecast": dict(baseline),
        }
    best_gain = max(scored[route]["coverage_gain"] for route in improvers)
    winners = [route for route in improvers if abs(scored[route]["coverage_gain"] - best_gain) < 1e-12]
    if len(winners) != 1:
        return {
            "route": original_route,
            "reason": "best same-family coverage routes tied",
            "family": family,
            "tied_routes": winners,
            "best_gain": best_gain,
        }
    selected = winners[0]
    return {
        "route": selected,
        "reason": "unique same-family 70% coverage improvement",
        "family": family,
        "targets": {key: round(value, 4) for key, value in targets.items()},
        "original_forecast": dict(baseline),
        "selected_forecast": scored[selected]["forecast"],
        "coverage_gain": scored[selected]["coverage_gain"],
    }


def _select_animal_route(module: ModuleType, observation: dict[str, Any], original_route: int) -> dict[str, Any]:
    """Choose one fully prepared V45 route, or explicitly abstain.

    The selector is intentionally narrow.  It runs only at V45's native day-6
    switch point, never makes a market order itself, and accepts a replacement
    only when exactly one opening-compatible route has a strictly lower 70/30
    composition loss than V45's original route.  Ties are abstentions, not an
    excuse to smuggle a fixed route preference into a supposedly dynamic rule.
    """

    targets = _animal_targets(observation)
    if targets is None:
        return {"route": original_route, "reason": "fewer than two viable animal products"}
    primary, secondary = targets
    existing = _owned_animal_outputs(observation)
    candidates = _compatible_animal_routes(module)
    profiles = {route: _route_animal_buys(module, route) for route in candidates}
    original_loss = _route_composition_loss(existing, profiles[original_route], primary, secondary)
    scored = {
        route: _route_composition_loss(existing, profile, primary, secondary)
        for route, profile in profiles.items()
    }
    best_loss = min(scored.values())
    winners = [route for route, loss in scored.items() if abs(loss - best_loss) < 1e-12]
    if len(winners) != 1:
        return {
            "route": original_route,
            "reason": "best composition route tied",
            "primary": primary,
            "secondary": secondary,
            "original_loss": round(original_loss, 6),
            "best_loss": round(best_loss, 6),
            "tied_routes": winners,
        }
    selected = winners[0]
    if selected == original_route or best_loss >= original_loss - 1e-12:
        return {
            "route": original_route,
            "reason": "original route is not strictly worse",
            "primary": primary,
            "secondary": secondary,
            "original_loss": round(original_loss, 6),
            "best_loss": round(best_loss, 6),
        }
    return {
        "route": selected,
        "reason": "unique lower 70/30 animal-composition loss",
        "primary": primary,
        "secondary": secondary,
        "original_loss": round(original_loss, 6),
        "selected_loss": round(best_loss, 6),
        "original_profile": dict(profiles[original_route]),
        "selected_profile": dict(profiles[selected]),
        "existing_animals": dict(existing),
    }


def _install_animal_route_selector(
    module: ModuleType,
    *,
    identity_only: bool = False,
    version: int = 1,
) -> dict[str, Any]:
    """Install the day-6 selector inside V45's router, before V45's own layers."""

    if version not in {1, 2}:
        raise ValueError("animal route selector version must be 1 or 2")
    parent_router = module._IMPL.chassis.router
    report: dict[str, Any] = {
        "selector_calls": 0,
        "selector_switches": 0,
        "selector_abstentions": 0,
        "selector_errors": 0,
        "selector_version": version,
        "selector_events": [],
        "compatible_routes": list(_compatible_animal_routes(module)),
    }

    def selector_router(observation: dict[str, Any], step: int, state: dict[str, Any]) -> int:
        original_route = parent_router(observation, step, state)
        if step != 144 or state.get("portfolio_animal_selector_done"):
            return original_route
        state["portfolio_animal_selector_done"] = True
        report["selector_calls"] += 1
        try:
            if identity_only:
                decision = {"route": original_route, "reason": "identity smoke mode"}
            elif version == 2:
                decision = _select_coverage_route(module, observation, original_route)
            else:
                decision = _select_animal_route(module, observation, original_route)
            selected = int(decision["route"])
            if selected not in module._IMPL.chassis.routes:
                raise ValueError(f"selected unknown route {selected}")
            state["route"] = selected
            event = {"original_route": original_route, "selected_route": selected, **decision}
            report["selector_events"].append(event)
            if selected != original_route:
                report["selector_switches"] += 1
            else:
                report["selector_abstentions"] += 1
        except Exception as error:
            report["selector_errors"] += 1
            report["selector_events"].append({"original_route": original_route, "selected_route": original_route, "reason": f"selector error: {error}"})
        return state.get("route", original_route)

    module._IMPL.chassis.router = selector_router
    return report


def make_animal_route_selector_agent(*, identity_only: bool = False, version: int = 1) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Return a V45-derived day-6 animal-route selector for local A/B tests.

    ``identity_only`` is a required smoke mode: the selector is installed but
    always returns V45's original route, proving the hook itself is inert before
    any route may be changed.  Version 1 is the failed 70/30 mix selector.
    Version 2 stays inside V45's current shop family and uses known-shop 70%
    coverage of an optimistic animal forecast.
    """

    module = load_v45_base_module()
    report = _install_animal_route_selector(module, identity_only=identity_only, version=version)
    parent = getattr(module, "agent", None)
    if not callable(parent):
        raise RuntimeError("V45 base did not expose a callable agent")

    def agent(observation: dict[str, Any], configuration: dict[str, Any] | None = None) -> dict[str, Any]:
        result = parent(observation, configuration)
        parent_report = getattr(parent, "telemetry", {})
        if isinstance(parent_report, dict):
            report["parent_telemetry"] = dict(parent_report)
        return result

    if identity_only:
        label = "v45-animal-route-identity"
    elif version == 2:
        label = "v45-animal-route-v2"
    else:
        label = "v45-animal-route-v1"
    agent.label = label  # type: ignore[attr-defined]
    agent.telemetry = report  # type: ignore[attr-defined]
    return agent
