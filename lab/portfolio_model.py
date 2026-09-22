"""Transparent, observation-only production-line ranking for the new agent.

This is deliberately not a profit oracle and does not issue game actions.  It
turns public shop demand, the visible market, visible rival production, and the
remaining season into an explainable ranking of *new* production lines.  The
daily planner uses the ranking to express the approved 70/30 portfolio idea.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL")
MARKET_EQUILIBRIUM = 10_000
TURNS_PER_DAY = 24
SHOP_TICKS_PER_DAY = 6

# A line means a complete new investment choice, not just a good whose current
# price happens to be high. `units_per_yield_event` is deliberately separate
# from `max_yield_events`: tomato's "4" means four scheduled one-unit events,
# not four units at every event.  Animals explicitly have a low-work FEED-only
# option and a FEED+CARE option; neither hides CARE's extra unit actions.
#
# The resulting action count is a *tile-action lower bound*: it counts work on
# the field, but not walking between the shed and a future chosen tile.  A
# routing module must add that missing travel cost before any action can be
# emitted.
@dataclass(frozen=True)
class ProductionLine:
    name: str
    output: str
    startup_cost: int
    first_yield_days: int
    units_per_yield_event: int
    yield_interval_days: int | None
    max_yield_events: int | None
    harvest_day: int | None
    daily_feed_units: int
    care_daily: bool
    max_held_units: int | None
    setup_unit_actions: int
    kind: str


LINES = (
    # One-time crop `harvest_day` is the age at which its listed unfertilized
    # yield is achieved.  It therefore sets both its final water day and its
    # one harvest, rather than mistaking first possible yield for peak yield.
    ProductionLine("wheat", "WHEAT", 10, 2, 4, None, 1, 4, 0, False, None, 1, "crop"),
    ProductionLine("carrot", "CARROT", 20, 2, 3, None, 1, 3, 0, False, None, 1, "crop"),
    ProductionLine("tomato", "TOMATO", 50, 8, 1, 1, 4, None, 0, False, None, 1, "crop"),
    ProductionLine("strawberry", "STRAWBERRY", 100, 10, 1, 2, 4, None, 0, False, None, 1, "crop"),
    ProductionLine("melon", "MELON", 80, 10, 6, None, 1, 10, 0, False, None, 1, "crop"),
    # Animal setup lower bound = BUILD structure, PICKUP animal, PLACE animal,
    # PICKUP wheat.  CARE choices are separate lines, so added yield always
    # travels with added daily CARE work.
    ProductionLine("goose_eggs_basic", "EGG", 300, 4, 1, 1, None, None, 1, False, 4, 4, "animal"),
    ProductionLine("goose_eggs_cared", "EGG", 300, 4, 1, 1, None, None, 1, True, 4, 4, "animal"),
    ProductionLine("cow_milk_basic", "MILK", 400, 8, 1, 2, None, None, 1, False, 6, 4, "animal"),
    ProductionLine("cow_milk_cared", "MILK", 400, 8, 1, 2, None, None, 1, True, 6, 4, "animal"),
    ProductionLine("sheep_wool_basic", "WOOL", 500, 6, 1, 3, None, None, 1, False, 6, 4, "animal"),
    ProductionLine("sheep_wool_cared", "WOOL", 500, 6, 1, 3, None, None, 1, True, 6, 4, "animal"),
)

BASE_PRICES = {
    "WHEAT": 25,
    "CARROT": 35,
    "TOMATO": 60,
    "STRAWBERRY": 120,
    "MELON": 250,
    "EGG": 50,
    "MILK": 160,
    "WOOL": 200,
}

SHOP_PRODUCTS = {
    "BAKERY": ("EGG", "WHEAT"),
    "PIZZA_SHOP": ("MILK", "TOMATO", "WHEAT"),
    "BRUNCH_SPOT": ("EGG", "WHEAT", "STRAWBERRY"),
    "YARN_STORE": ("WOOL",),
    "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"),
    "PET_CAFE": ("CARROT",),
    "SMOOTHIE_SHOP": ("STRAWBERRY", "MILK"),
    "FARMERS_MARKET": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY"),
}

ANIMAL_OUTPUT = {"GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL"}


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def shop_demand_per_day(shops: list[Any], product: str) -> float:
    """Current engine demand, not the removed old Town Center multiplier.

    The Town Center buys one of each product (other than fertilizer) per day.
    Every shop ticks six times per day; a single-product shop consumes two units
    per tick, otherwise one unit of each listed product.
    """

    demand = 1.0 if product in PRODUCTS else 0.0
    for raw_shop in shops:
        offered = SHOP_PRODUCTS.get(str(raw_shop), ())
        if product in offered:
            units_per_tick = 2 if len(offered) == 1 else 1
            demand += SHOP_TICKS_PER_DAY * units_per_tick
    return demand


def visible_rival_output_counts(observation: dict[str, Any]) -> dict[str, int]:
    """Count only publicly visible rival tiles; never read rival private state."""

    player = _int(observation.get("player"))
    farms = list(observation.get("farms") or [])
    if len(farms) != 2 or player not in (0, 1):
        return {product: 0 for product in PRODUCTS}
    counts = {product: 0 for product in PRODUCTS}
    for row in farms[1 - player].get("tiles") or []:
        for tile in row:
            if not isinstance(tile, dict):
                continue
            crop = str(tile.get("crop") or "")
            animal = str(tile.get("animal") or "")
            if crop in counts:
                counts[crop] += 1
            output = ANIMAL_OUTPUT.get(animal)
            if output:
                counts[output] += 1
    return counts


@dataclass(frozen=True)
class LineEvaluation:
    line: str
    output: str
    kind: str
    score: float
    current_price: int
    market_inventory: int
    shop_demand_per_day: float
    rival_visible_units: int
    startup_cost: int
    first_yield_days: int
    days_remaining: int
    yield_events: int
    expected_output_units: int
    field_action_lower_bound: int
    care_mode: str
    estimated_feed_cost: int
    currently_affordable: bool
    feasible_before_terminal: bool
    components: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _yield_event_count(line: ProductionLine, days_remaining: int) -> int:
    """Count scheduled yields that can land before the terminal day."""

    if days_remaining <= line.first_yield_days:
        return 0
    events = 1
    if line.yield_interval_days is not None:
        events += max(0, (days_remaining - line.first_yield_days) // line.yield_interval_days)
    if line.max_yield_events is not None:
        events = min(events, line.max_yield_events)
    return events


def _animal_event_units(line: ProductionLine, event_count: int) -> list[int]:
    """Return exact base/CARE production units before each timely harvest.

    A cared animal placed and first cared on day zero pays its saved CARE on the
    first production: base 1 plus `first_yield_days - 1` prior care days. Later
    production receives base 1 plus one saved CARE for each interval day. The
    engine caps unharvested output; the cap applies even to this first payout.
    """

    if event_count <= 0:
        return []
    if not line.care_daily:
        return [line.units_per_yield_event] * event_count
    cap = line.max_held_units or 1
    first = min(cap, line.units_per_yield_event + line.first_yield_days - 1)
    recurring = min(cap, line.units_per_yield_event + (line.yield_interval_days or 0))
    return [first, *([recurring] * (event_count - 1))]


def _animal_harvest_actions(event_units: list[int], max_held: int) -> int:
    """Minimum harvest count that avoids a known tile-cap loss and sells all output."""

    held = harvests = 0
    for units in event_units:
        if held and held + units > max_held:
            harvests += 1
            held = 0
        held = min(max_held, held + units)
    return harvests + int(held > 0)


def _field_action_lower_bound(line: ProductionLine, event_count: int, event_units: list[int], days_remaining: int) -> int:
    """Count indispensable tile actions, intentionally excluding unknown walking."""

    if event_count <= 0:
        return 0
    if line.kind == "animal":
        feed_days = days_remaining
        harvests = _animal_harvest_actions(event_units, line.max_held_units or 1)
        return line.setup_unit_actions + feed_days + (feed_days if line.care_daily else 0) + harvests
    if line.harvest_day is not None:
        water_days = min(days_remaining, line.harvest_day + 1)
        return line.setup_unit_actions + water_days + 1  # PLANT + WATERs + HARVEST
    # Ongoing crops need watering through the final scheduled production, one
    # harvest to realize accumulated output, then a DIG after decay to free the
    # tile for the next investment.
    final_production_day = line.first_yield_days + (event_count - 1) * (line.yield_interval_days or 0)
    water_days = min(days_remaining, final_production_day + 1)
    return line.setup_unit_actions + water_days + 1 + 1


def evaluate_lines(observation: dict[str, Any]) -> list[LineEvaluation]:
    """Rank production lines using facts available in the current observation.

    Score is an *investment-priority heuristic*, not an asserted future cash
    forecast. It combines expected margin per indispensable field action with
    public demand, price, rival crowding, and terminal feasibility. Walking is
    intentionally not invented here; routing must price it later.
    """

    player = _int(observation.get("player"))
    farm = list(observation.get("farms") or [{}, {}])[player]
    day = _int(observation.get("day"))
    days_remaining = max(0, 30 - day)
    market = dict(observation.get("market") or {})
    prices = dict(market.get("prices") or {})
    inventory = dict(market.get("inventory") or {})
    shops = list((observation.get("town") or {}).get("unlocked_shops") or [])
    rival_counts = visible_rival_output_counts(observation)
    money = _int(farm.get("money"))

    evaluations: list[LineEvaluation] = []
    for line in LINES:
        price = max(1, _int(prices.get(line.output), BASE_PRICES[line.output]))
        stock = max(0, _int(inventory.get(line.output), MARKET_EQUILIBRIUM))
        demand = shop_demand_per_day(shops, line.output)
        rival = rival_counts[line.output]
        yield_events = _yield_event_count(line, days_remaining)
        feasible = yield_events > 0
        event_units = _animal_event_units(line, yield_events) if line.kind == "animal" else [line.units_per_yield_event] * yield_events
        expected_units = sum(event_units)
        field_actions = _field_action_lower_bound(line, yield_events, event_units, days_remaining)
        wheat_price = max(1, _int(prices.get("WHEAT"), BASE_PRICES["WHEAT"]))
        feed_cost = line.daily_feed_units * days_remaining * wheat_price
        margin = max(0.0, expected_units * price - line.startup_cost - feed_cost)
        margin_per_field_action = margin / max(1, field_actions)
        demand_factor = 1.0 + min(1.5, demand / 24.0)
        price_factor = _clamp(price / BASE_PRICES[line.output], 0.45, 1.60)
        scarcity_factor = _clamp(1.0 + (MARKET_EQUILIBRIUM - stock) / MARKET_EQUILIBRIUM, 0.50, 1.50)
        rival_factor = 1.0 / (1.0 + 0.08 * rival)
        horizon_factor = 1.0 if feasible else 0.0
        score = margin_per_field_action * demand_factor * price_factor * scarcity_factor * rival_factor * horizon_factor
        evaluations.append(
            LineEvaluation(
                line=line.name,
                output=line.output,
                kind=line.kind,
                score=round(score, 4),
                current_price=price,
                market_inventory=stock,
                shop_demand_per_day=round(demand, 2),
                rival_visible_units=rival,
                startup_cost=line.startup_cost,
                first_yield_days=line.first_yield_days,
                days_remaining=days_remaining,
                yield_events=yield_events,
                expected_output_units=expected_units,
                field_action_lower_bound=field_actions,
                care_mode="daily_care" if line.care_daily else "feed_only" if line.kind == "animal" else "not_applicable",
                estimated_feed_cost=feed_cost,
                currently_affordable=money >= line.startup_cost,
                feasible_before_terminal=feasible,
                components={
                    "margin_per_field_action": round(margin_per_field_action, 4),
                    "estimated_margin_before_labor": round(margin, 4),
                    "demand_factor": round(demand_factor, 4),
                    "price_factor": round(price_factor, 4),
                    "scarcity_factor": round(scarcity_factor, 4),
                    "rival_factor": round(rival_factor, 4),
                    "horizon_factor": horizon_factor,
                },
            )
        )
    return sorted(evaluations, key=lambda item: (-item.score, item.line))


def choose_portfolio(observation: dict[str, Any]) -> dict[str, Any]:
    """Express the approved 70/30 allocation without issuing an investment.

    A line must be able to yield before the end to receive a share.  Lack of
    immediate cash is shown explicitly rather than silently treating a future
    investment as possible.  When only one line is viable, the other 30% remains
    cash/work-capacity reserve rather than becoming forced overinvestment.
    """

    ranked = evaluate_lines(observation)
    viable = [line for line in ranked if line.feasible_before_terminal and line.score > 0]
    primary = viable[0] if viable else None
    # A cared and a feed-only version of the same animal are two execution
    # modes, not diversification.  The secondary line must sell a different
    # product, otherwise the written 70/30 rule would secretly become 100% of
    # one shared market.
    secondary = next((line for line in viable[1:] if primary and line.output != primary.output), None)
    allocations: list[dict[str, Any]] = []
    if primary:
        allocations.append({"line": primary.line, "output": primary.output, "share": 0.70, "reason": "highest current investment-priority score"})
    if secondary:
        allocations.append({"line": secondary.line, "output": secondary.output, "share": 0.30, "reason": "second line diversifies market and production risk"})
    elif primary:
        allocations.append({"line": "reserve", "output": None, "share": 0.30, "reason": "no second viable line; do not force concentration"})
    else:
        allocations.append({"line": "reserve", "output": None, "share": 1.0, "reason": "no line can repay before terminal"})
    return {
        "shops": list((observation.get("town") or {}).get("unlocked_shops") or []),
        "ranked_lines": [line.as_dict() for line in ranked],
        "allocation": allocations,
        "primary": primary.as_dict() if primary else None,
        "secondary": secondary.as_dict() if secondary else None,
        "policy_limits": [
            "Shares describe only future discretionary production capacity, never existing crops, animals, or mandatory care.",
            "This selection does not issue BUY, PLANT, PLACE, HIRE, or market actions.",
            "A line without enough cash is a future target, not permission to overspend today.",
        ],
    }


def allocate_ranked_capacity(
    ranked_outputs: Sequence[str],
    available_slots: int,
    safe_slot_cap_by_output: Mapping[str, int],
) -> dict[str, Any]:
    """Allocate new, competing production slots under the 70/30 rule.

    ``safe_slot_cap_by_output`` is deliberately supplied by the caller.  This
    pure function does not pretend that it can infer V45's future labour,
    feed, shed, cash, and route commitments from a single abstract slot count.
    The first distinct product gets at most 70% of the slots; any shortfall is
    immediately offered to the next distinct product, then the next, so a
    resource-limited primary never leaves productive capacity stranded.
    """

    slots = max(0, int(available_slots))
    outputs: list[str] = []
    for output in ranked_outputs:
        product = str(output)
        if product and product not in outputs:
            outputs.append(product)
    primary_target = min(slots, int(slots * 0.70 + 0.5))
    remaining = slots
    allocations: list[dict[str, Any]] = []
    for index, output in enumerate(outputs):
        cap = max(0, int(safe_slot_cap_by_output.get(output, 0)))
        target = min(remaining, primary_target) if index == 0 else remaining
        assigned = min(target, cap)
        allocations.append(
            {
                "output": output,
                "rank": index + 1,
                "ideal_slots": primary_target if index == 0 else target,
                "safe_slot_cap": cap,
                "assigned_slots": assigned,
                "reason": "primary 70% target" if index == 0 else "fills capacity released by earlier constraints",
            }
        )
        remaining -= assigned
        if remaining <= 0:
            break
    return {
        "available_slots": slots,
        "primary_target_slots": primary_target,
        "allocations": allocations,
        "unallocated_slots": remaining,
        "limits": [
            "Only new, uncommitted slots belong here; existing crops, animals, and mandatory care do not.",
            "Safe caps must already include land, cash, feed, labour, shed, and remaining-time limits.",
        ],
    }


def allocate_new_capacity(
    observation: dict[str, Any],
    available_slots: int,
    safe_slot_cap_by_output: Mapping[str, int],
) -> dict[str, Any]:
    """Apply the current observation ranking to the pure slot allocator.

    This is still advisory: it returns a proposed split and never creates a
    BUY, BUILD, PLANT, PLACE, HIRE, or SELL action.
    """

    ranked = evaluate_lines(observation)
    viable_outputs: list[str] = []
    for line in ranked:
        if line.feasible_before_terminal and line.score > 0 and line.output not in viable_outputs:
            viable_outputs.append(line.output)
    allocation = allocate_ranked_capacity(viable_outputs, available_slots, safe_slot_cap_by_output)
    allocation["ranked_outputs"] = viable_outputs
    return allocation
