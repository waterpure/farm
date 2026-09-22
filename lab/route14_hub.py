"""One ledger for hires, land, animals, and seeds.

Existing crops and animals are paid first. Whatever cash and worker-turns remain
is spent on the next investment that still pays back before the season ends.
Nothing here keys off a fixed day or a fixed headcount.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from typing import Any

from .route14_capacity import (
    affordable_care_reserve,
    animal_candidate_position,
    day_schedule,
    future_care_hires,
    new_land_crop_slots,
    proposed_crop_slots,
)
from .route14_economy import (
    ANIMAL_COST,
    ANIMAL_FIRST_YIELD_DAYS,
    ANIMAL_STRUCTURE,
    CASH_BUFFER,
    DEFAULT_MAX_HIRES,
    JOBS_PER_WORKER,
    MIN_LAND_FILL_SEEDS,
    NEW_LAND_TILES,
    SEASON_DAYS,
    SEED_COST,
    animal_line_money_per_day,
    animal_tile_count,
    empty_structure_count,
    empty_unlocked_count,
    field_job_count,
    fib_hire_cost,
    haul_job_count,
    hire_cost_total,
    line_season_value,
    livestock_tile_cap,
    next_land_cost,
    planting_allowed,
    rank_animal_lines,
    remaining_days,
    wheat_unit_price,
)


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass
class HubPlan:
    hires: int
    buy_land: bool
    animal: str | None
    seeds: list[list[Any]]
    plant_slots: int = 0
    hire_first: bool = False
    reasons: list[str] = field(default_factory=list)


def _heads_on_field(observation: dict[str, Any]) -> int:
    return sum(animal_tile_count(observation, name) for name in ANIMAL_COST)


def _jobs_today(observation: dict[str, Any]) -> int:
    return field_job_count(observation) + haul_job_count(observation)


def _hires_for_jobs(jobs: int) -> int:
    workers = max(1, int(ceil(max(0, jobs) / JOBS_PER_WORKER))) if jobs else 1
    return min(DEFAULT_MAX_HIRES, max(0, workers - 1))


def _max_affordable_hires(cash: int) -> int:
    hired = 0
    spent = 0
    for index in range(DEFAULT_MAX_HIRES):
        cost = fib_hire_cost(index)
        if cash < spent + cost + CASH_BUFFER:
            break
        spent += cost
        hired += 1
    return hired


def _daily_keep(hires: int, heads: int, wheat: int) -> int:
    return hire_cost_total(hires) + max(0, heads) * max(1, wheat)


def _runway_days(cash: int, daily: int) -> int:
    if daily <= 0:
        return SEASON_DAYS
    return max(0, cash // daily)


def _seed_cost(orders: list[list[Any]]) -> int:
    return sum(_int(order[2]) * SEED_COST[str(order[1])] for order in orders)


def _animal_option(
    observation: dict[str, Any],
    *,
    leftover: int,
    wheat: int,
    days_left: int,
    placed: int,
    jobs: int,
    extra_jobs: int,
    hires: int,
    empty: int,
    empty_sheds: int,
    existing_reserve: int,
) -> dict[str, Any] | None:
    if placed >= livestock_tile_cap(observation):
        return None
    if empty_sheds <= 0 and empty <= 0:
        return None
    prices = dict((observation.get("market") or {}).get("prices") or {})
    for _kind, name, value in rank_animal_lines(observation):
        first = ANIMAL_FIRST_YIELD_DAYS[name]
        if first > days_left or leftover < ANIMAL_COST[name] + CASH_BUFFER:
            continue
        position = animal_candidate_position(observation, name)
        if position is None:
            continue
        x, y = position
        build = empty_structure_count(observation, ANIMAL_STRUCTURE[name]) <= 0
        animal_tasks = [
            {"target": [x, y], "kind": "PLACE_ANIMAL", "priority": 90},
            {"target": [x, y], "kind": "FEED", "priority": 90},
            {"target": [x, y], "kind": "CARE", "priority": 90},
        ]
        if build:
            animal_tasks.insert(0, {"target": [x, y], "kind": "BUILD", "priority": 90})
        future_hires = future_care_hires(observation, animal=(x, y, name))
        if future_hires is None:
            continue
        need_hires = next(
            (
                count
                for count in range(max(hires, future_hires), DEFAULT_MAX_HIRES + 1)
                if day_schedule(observation, count, extra_tasks=animal_tasks)["all_mandatory_tasks_witnessed"]
            ),
            None,
        )
        if need_hires is None:
            continue
        extra_wage = max(0, hire_cost_total(need_hires) - hire_cost_total(hires))
        new_reserve = affordable_care_reserve(observation, need_hires, extra_animals=1)
        cost = ANIMAL_COST[name] + extra_wage + max(0, new_reserve - existing_reserve)
        after = leftover - cost
        if after < CASH_BUFFER:
            continue
        per_day = animal_line_money_per_day(observation, name, days_left, prices) or value
        return {
            "name": name,
            "cost": cost,
            "hires": need_hires,
            "jobs": len(animal_tasks),
            "score": per_day / max(1.0, float(cost)),
            "first": first,
        }
    return None


def _land_option(
    observation: dict[str, Any],
    *,
    leftover: int,
    wheat: int,
    days_left: int,
    placed: int,
    jobs: int,
    extra_jobs: int,
    hires: int,
    empty: int,
    hour: int,
    existing_reserve: int,
) -> dict[str, Any] | None:
    if empty > 0 or hour != 0 or days_left < 3:
        return None
    land_cost = next_land_cost(observation)
    if land_cost is None or leftover < land_cost + CASH_BUFFER:
        return None
    slots = new_land_crop_slots(observation, MIN_LAND_FILL_SEEDS)
    if len(slots) < MIN_LAND_FILL_SEEDS:
        return None
    future_hires = future_care_hires(observation, plants=slots)
    if future_hires is None:
        return None
    need_hires = next(
        (count for count in range(max(hires, future_hires), DEFAULT_MAX_HIRES + 1) if day_schedule(observation, count, slots)["all_mandatory_tasks_witnessed"]),
        None,
    )
    if need_hires is None:
        return None
    extra_wage = max(0, hire_cost_total(need_hires) - hire_cost_total(hires))
    fill_cash = sum(SEED_COST[crop] for _, _, crop in slots)
    incremental_reserve = max(0, affordable_care_reserve(observation, max(need_hires, future_hires), slots) - existing_reserve)
    cost = land_cost + extra_wage + fill_cash + incremental_reserve
    after = leftover - cost
    prices = dict((observation.get("market") or {}).get("prices") or {})
    worth = line_season_value("crop", slots[0][2], len(slots), prices, days_left, observation)
    season_labor = max(0, hire_cost_total(future_hires) - hire_cost_total(hires)) * days_left
    if after < CASH_BUFFER or worth <= float(land_cost + season_labor):
        return None
    return {
        "cost": cost,
        "hires": need_hires,
        "score": (worth - land_cost - season_labor) / max(1.0, float(cost)),
    }


def _fill_plan(
    observation: dict[str, Any],
    *,
    leftover: int,
    jobs: int,
    keep_hires: int,
    keep_wage: int,
    animal_budget: int,
    existing_reserve: int,
) -> tuple[int, list[list[Any]], int, int]:
    if leftover <= 0 or not planting_allowed(observation) or empty_unlocked_count(observation) <= 0:
        return keep_hires, [], 0, 0
    slots = proposed_crop_slots(observation, leftover, animal_budget)
    if not slots:
        return keep_hires, [], 0, 0
    existing_seeds = dict((observation.get("private") or {}).get("seeds") or {})
    best_hires = keep_hires
    best_seeds: list[list[Any]] = []
    best_filled = 0
    for hired in range(keep_hires, DEFAULT_MAX_HIRES + 1):
        extra_wage = max(0, hire_cost_total(hired) - keep_wage)
        if leftover < extra_wage + CASH_BUFFER:
            break
        if hired < best_hires and best_filled == len(slots):
            break
        # A larger planting prefix contains every task in a smaller prefix.
        # Find the route capacity first, then examine cash and future care.
        lo, hi = best_filled + 1, len(slots)
        route_cap = best_filled
        while lo <= hi:
            mid = (lo + hi) // 2
            if day_schedule(observation, hired, slots[:mid])["all_mandatory_tasks_witnessed"]:
                route_cap = mid
                lo = mid + 1
            else:
                hi = mid - 1
        for count in range(route_cap, best_filled, -1):
            trial = slots[:count]
            future_hires = future_care_hires(observation, plants=trial)
            if future_hires is None:
                continue
            need: dict[str, int] = {}
            for _, _, crop in trial:
                need[crop] = need.get(crop, 0) + 1
            seeds = [["BUY_SEED", crop, max(0, qty - _int(existing_seeds.get(crop)))] for crop, qty in need.items()]
            seeds = [order for order in seeds if order[2] > 0]
            incremental_reserve = max(0, affordable_care_reserve(observation, max(hired, future_hires), trial) - existing_reserve)
            if leftover < extra_wage + _seed_cost(seeds) + incremental_reserve + CASH_BUFFER:
                continue
            best_filled, best_hires, best_seeds = count, hired, seeds
            break
        if best_filled == len(slots):
            break
    return best_hires, best_seeds, best_filled * 2, best_filled


def hub_plan(observation: dict[str, Any], cash: int | None = None) -> HubPlan:
    """Reserve upkeep, then spend leftover cash/labor on fill, animals, or land."""

    player = _int(observation.get("player"))
    farms = list(observation.get("farms") or [])
    farm = farms[player] if 0 <= player < len(farms) else {}
    if cash is None:
        cash = _int(farm.get("money"))
    days_left = remaining_days(observation)
    hour = _int(observation.get("hour"))
    wheat = wheat_unit_price(observation)
    placed = _heads_on_field(observation)
    empty = empty_unlocked_count(observation)
    jobs = _jobs_today(observation)
    empty_sheds = empty_structure_count(observation, "PASTURE") + empty_structure_count(observation, "COOP")
    reasons: list[str] = []

    already = max(_int(farm.get("hires_today")), len(farm.get("hands") or []))
    keep_hires = already
    best_missing = len(day_schedule(observation, already)["unassigned_groups"])
    for trial_hires in range(already + 1, DEFAULT_MAX_HIRES + 1):
        missing = len(day_schedule(observation, trial_hires)["unassigned_groups"])
        if missing < best_missing:
            keep_hires, best_missing = trial_hires, missing
        if missing == 0:
            break
    hire_first = haul_job_count(observation) > 0 or keep_hires >= 6
    feed_today = placed * wheat
    leftover = cash - feed_today
    keep_wage = hire_cost_total(keep_hires)
    existing_reserve = affordable_care_reserve(observation, keep_hires)

    if leftover < keep_wage + existing_reserve + CASH_BUFFER:
        hires = min(keep_hires, _max_affordable_hires(max(0, leftover)))
        reasons.append("keep_only")
        return HubPlan(hires=hires, buy_land=False, animal=None, seeds=[], hire_first=hire_first, reasons=reasons)

    leftover -= keep_wage + existing_reserve
    hires = keep_hires
    extra_jobs = 0
    peek = _animal_option(
        observation,
        leftover=leftover,
        wheat=wheat,
        days_left=days_left,
        placed=placed,
        jobs=jobs,
        extra_jobs=0,
        hires=hires,
        empty=empty,
        empty_sheds=empty_sheds,
        existing_reserve=existing_reserve,
    )
    animal_budget = 0
    if peek is not None:
        structure = ANIMAL_STRUCTURE[str(peek["name"])]
        if empty_structure_count(observation, structure) <= 0:
            animal_budget = 1

    hires, seeds, plant_jobs, plant_slots = _fill_plan(
        observation,
        leftover=leftover,
        jobs=jobs,
        keep_hires=keep_hires,
        keep_wage=keep_wage,
        animal_budget=animal_budget,
        existing_reserve=existing_reserve,
    )
    if seeds:
        leftover -= _seed_cost(seeds) + max(0, hire_cost_total(hires) - keep_wage)
        extra_jobs += plant_jobs
        reasons.append("fill")

    animal: str | None = None
    buy_land = False
    animal_opt = _animal_option(
        observation,
        leftover=leftover,
        wheat=wheat,
        days_left=days_left,
        placed=placed,
        jobs=jobs,
        extra_jobs=extra_jobs,
        hires=hires,
        empty=empty,
        empty_sheds=empty_sheds,
        existing_reserve=existing_reserve,
    )
    land_opt = _land_option(
        observation,
        leftover=leftover,
        wheat=wheat,
        days_left=days_left,
        placed=placed + int(animal_opt is not None),
        jobs=jobs,
        extra_jobs=extra_jobs,
        hires=hires,
        empty=empty,
        hour=hour,
        existing_reserve=existing_reserve,
    )
    pick_land = bool(land_opt) and (animal_opt is None or float(land_opt["score"]) > float(animal_opt["score"]))
    if pick_land and land_opt is not None:
        buy_land = True
        leftover -= int(land_opt["cost"])
        hires = int(land_opt["hires"])
        reasons.append("land")
    elif animal_opt is not None:
        animal = str(animal_opt["name"])
        leftover -= int(animal_opt["cost"])
        hires = int(animal_opt["hires"])
        extra_jobs += int(animal_opt["jobs"])
        reasons.append("animal")
        if empty == 0:
            reasons.append("skip_land_for_animal")
    elif empty > 0:
        reasons.append("land_empty")
    elif land_opt is None:
        reasons.append("skip_land")

    # The last day still needs hands: the crop standing in the field is worth
    # far more than a day of wages. Hiring in the final hour is useless, though,
    # because a new hand only appears on the next turn.
    if hour >= 23:
        hires = 0
    hires = min(hires, _max_affordable_hires(cash - feed_today))
    return HubPlan(
        hires=hires,
        buy_land=buy_land,
        animal=animal,
        seeds=seeds,
        plant_slots=plant_slots,
        hire_first=hire_first,
        reasons=reasons,
    )
