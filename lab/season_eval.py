"""Compare two agents across full seasons and keep the official cash ledger.

The episode itself is ``runner.run_match``.  This module only records what the
installed Kaggriculture engine actually committed.  Wrappers call the original
functions and put the original functions back before returning.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean, median
from typing import Any

from kaggle_environments.envs.kaggriculture import kaggriculture as game

from .runner import AgentSpec, _agent_label, _resolve_agent, run_match


SALE_ITEMS = tuple(game.PRODUCTS)
MOVES = frozenset(game.FARMER_MOVES)
DEFAULT_JSONL = Path("experiments/region_phase1_season_v1.jsonl")
DEFAULT_SUMMARY = Path("experiments/region_phase1_season_v1_summary.json")
RUN_TIMEOUT_SECONDS = 3600


@dataclass(frozen=True)
class DayPoint:
    day: int
    money: int
    shed_units: int
    crop_count: int
    animal_count: int
    owned_tiles: int


@dataclass(frozen=True)
class SeasonResult:
    seed: int
    agent: str
    opponent: str
    seat: str
    steps: int = 0
    terminal_reward: float | None = None
    final_money: int = 0
    initial_money: int = 0
    sell_revenue: int = 0
    total_spend: int = 0
    net_cash_change: int = 0
    seed_spend: int = 0
    animal_spend: int = 0
    product_spend: int = 0
    hire_spend: int = 0
    land_spend: int = 0
    other_income: int = 0
    other_spend: int = 0
    cash_reconciliation_error: int = 0
    sell_revenue_by_item: dict[str, int] = field(default_factory=dict)
    successful_hires: int = 0
    seed_units_bought: int = 0
    animal_units_bought: int = 0
    plant_successes: int = 0
    harvest_successes: int = 0
    sell_units: int = 0
    failed_market_orders: int = 0
    failed_unit_actions: int = 0
    pass_hours: int = 0
    move_hours: int = 0
    work_hours: int = 0
    replans: int = 0
    daily_money: tuple[int, ...] = ()
    daily: tuple[DayPoint, ...] = ()
    status: str = "DONE"
    elapsed_seconds: float = 0.0
    reward_equals_final_money: bool = False
    final_day: int | None = None
    final_hour: int | None = None
    exception: str | None = None
    last_action: Any = None
    last_day: int | None = None
    last_hour: int | None = None

    @property
    def failed_actions(self) -> int:
        return self.failed_unit_actions + self.failed_market_orders


def _field(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _money(farm: Any) -> int:
    return int(round(float(_field(farm, "money", 0) or 0)))


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((str(key), _freeze(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _position(farm: Any, actor_index: int) -> tuple[int, int] | None:
    if actor_index == 0:
        raw = _field(farm, "farmer")
    else:
        hands = _field(farm, "hands") or []
        raw = hands[actor_index - 1] if actor_index - 1 < len(hands) else None
    if raw is None:
        return None
    return int(raw[0]), int(raw[1])


def _signature(farm: Any, private: Any, actor_index: int) -> tuple[Any, ...]:
    pos = _position(farm, actor_index)
    tile = None
    if pos is not None:
        tiles = _field(farm, "tiles") or []
        tile = tiles[pos[1]][pos[0]]
    inventories = _field(private, "inventories") or []
    carried = inventories[actor_index] if actor_index < len(inventories) else None
    return (
        pos,
        _freeze(tile),
        _freeze(carried),
        _freeze(_field(private, "shed")),
        _freeze(_field(private, "seeds")),
    )


def _operation(action: Any) -> str | None:
    if isinstance(action, (list, tuple)) and action:
        return str(action[0])
    return None


class _Books:
    """Per-player totals taken only from successful engine commits."""

    def __init__(self) -> None:
        self.private_ids: dict[int, int] = {}
        self.farm_ids: dict[int, int] = {}
        self.sell: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.seed_spend: dict[int, int] = defaultdict(int)
        self.animal_spend: dict[int, int] = defaultdict(int)
        self.product_spend: dict[int, int] = defaultdict(int)
        self.hire_spend: dict[int, int] = defaultdict(int)
        self.land_spend: dict[int, int] = defaultdict(int)
        self.other_income: dict[int, int] = defaultdict(int)
        self.other_spend: dict[int, int] = defaultdict(int)
        self.seed_units: dict[int, int] = defaultdict(int)
        self.animal_units: dict[int, int] = defaultdict(int)
        self.sell_units: dict[int, int] = defaultdict(int)
        self.hires: dict[int, int] = defaultdict(int)
        self.plants: dict[int, int] = defaultdict(int)
        self.harvests: dict[int, int] = defaultdict(int)
        self.failed_market: dict[int, int] = defaultdict(int)
        self.failed_unit: dict[int, int] = defaultdict(int)
        self.pass_hours: dict[int, int] = defaultdict(int)
        self.move_hours: dict[int, int] = defaultdict(int)
        self.work_hours: dict[int, int] = defaultdict(int)

    def see(self, state: list[Any]) -> None:
        farms = _field(_field(state[0], "observation"), "farms")
        if not farms:
            return
        self.farm_ids = {id(farm): index for index, farm in enumerate(farms)}
        self.private_ids = {}
        for index, entry in enumerate(state):
            private = _field(_field(entry, "observation"), "private")
            if private is not None:
                self.private_ids[id(private)] = index

    def player(self, private: Any = None, farm: Any = None) -> int:
        if private is not None and id(private) in self.private_ids:
            return self.private_ids[id(private)]
        if farm is not None and id(farm) in self.farm_ids:
            return self.farm_ids[id(farm)]
        return -1

    def note_commit(self, player: int, op: str, item: str, price: int, ok: bool) -> None:
        if player < 0:
            return
        if not ok:
            self.failed_market[player] += 1
            return
        amount = int(price)
        if op == "SELL":
            self.sell[player][str(item)] += amount
            self.sell_units[player] += 1
        elif op == "BUY_SEED":
            self.seed_spend[player] += amount
            self.seed_units[player] += 1
        elif op == "BUY_ANIMAL":
            self.animal_spend[player] += amount
            self.animal_units[player] += 1
        elif op == "BUY_PRODUCT":
            self.product_spend[player] += amount
        else:
            self.other_spend[player] += amount

    def note_hire(self, player: int, before: int, after: int, hired: bool) -> None:
        if player < 0:
            return
        if hired:
            self.hire_spend[player] += before - after
            self.hires[player] += 1
        else:
            self.failed_market[player] += 1

    def note_land(self, player: int, before: int, after: int, bought: bool) -> None:
        if player < 0:
            return
        if bought:
            self.land_spend[player] += before - after
        else:
            self.failed_market[player] += 1

    def note_unit(self, player: int, op: str | None, changed: bool, stood: bool) -> None:
        if player < 0 or not stood:
            return
        if op == "PASS":
            self.pass_hours[player] += 1
        elif op in MOVES:
            if changed:
                self.move_hours[player] += 1
            else:
                self.failed_unit[player] += 1
        elif changed:
            self.work_hours[player] += 1
            if op == "PLANT":
                self.plants[player] += 1
            elif op == "HARVEST":
                self.harvests[player] += 1
        else:
            self.failed_unit[player] += 1


def _install(environment: Any, books: _Books) -> Callable[[], None]:
    """Record engine commits.  Actions and return values stay the originals."""

    original_interpreter = environment.interpreter
    original_apply = game._apply_unit_action
    original_commit = game._commit_unit
    original_hire = game._do_hire
    original_land = game._do_buy_land

    def observed_interpreter(state: list[Any], env: Any) -> Any:
        books.see(state)
        return original_interpreter(state, env)

    def observed_apply(farm: Any, private: Any, actor_index: int, action: Any, *args: Any, **kwargs: Any) -> Any:
        player = books.player(private, farm)
        before = _signature(farm, private, actor_index)
        result = original_apply(farm, private, actor_index, action, *args, **kwargs)
        after = _signature(farm, private, actor_index)
        books.note_unit(player, _operation(action), before != after, before[0] is not None)
        return result

    def observed_commit(
        op: str, item: str, price: int, farm: Any, private: Any, market: Any, shed_capacity: int = 100
    ) -> bool:
        player = books.player(private, farm)
        result = original_commit(op, item, price, farm, private, market, shed_capacity)
        books.note_commit(player, str(op), str(item), int(price), bool(result))
        return result

    def observed_hire(farm: Any, private: Any, board_size: int, mult: int = game.FARM_HAND_COST_MULT) -> Any:
        player = books.player(private, farm)
        before_money = _money(farm)
        before_hires = int(_field(farm, "hires_today", 0) or 0)
        result = original_hire(farm, private, board_size, mult)
        books.note_hire(
            player,
            before_money,
            _money(farm),
            int(_field(farm, "hires_today", 0) or 0) > before_hires,
        )
        return result

    def observed_land(farm: Any, board_size: int) -> Any:
        player = books.player(farm=farm)
        before_money = _money(farm)
        before_quads = len(_field(farm, "unlocked_quadrants") or [])
        result = original_land(farm, board_size)
        books.note_land(
            player,
            before_money,
            _money(farm),
            len(_field(farm, "unlocked_quadrants") or []) > before_quads,
        )
        return result

    environment.interpreter = observed_interpreter
    game._apply_unit_action = observed_apply
    game._commit_unit = observed_commit
    game._do_hire = observed_hire
    game._do_buy_land = observed_land

    def restore() -> None:
        environment.interpreter = original_interpreter
        game._apply_unit_action = original_apply
        game._commit_unit = original_commit
        game._do_hire = original_hire
        game._do_buy_land = original_land

    return restore


def _watch(agent: Callable[..., Any]) -> tuple[Callable[..., Any], dict[str, Any]]:
    trace: dict[str, Any] = {"replans": 0, "last_action": None, "day": None, "hour": None, "error": None}

    def wrapped(observation: dict[str, Any], configuration: dict[str, Any] | None = None) -> Any:
        trace["day"] = _field(observation, "day")
        trace["hour"] = _field(observation, "hour")
        try:
            action = agent(observation, configuration)
        except Exception as exc:
            trace["error"] = f"{type(exc).__name__}: {exc}"
            raise
        trace["last_action"] = action
        telemetry = getattr(agent, "telemetry", None)
        if isinstance(telemetry, dict) and telemetry.get("needs_replan"):
            trace["replans"] += 1
        return action

    wrapped.telemetry = getattr(agent, "telemetry", {})  # type: ignore[attr-defined]
    wrapped.label = getattr(agent, "label", None)  # type: ignore[attr-defined]
    return wrapped, trace


def _observation(step: Any, player: int) -> Any:
    entry = step[player]
    if isinstance(entry, dict):
        return entry.get("observation")
    return entry.observation


def _tile_counts(tiles: Any) -> tuple[int, int, int]:
    owned = crops = animals = 0
    for row in tiles or []:
        for tile in row:
            if tile != "LOCKED":
                owned += 1
            if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                crops += 1
            if isinstance(tile, dict) and "animal" in tile:
                animals += 1
    return owned, crops, animals


def _snapshot(step: Any, player: int, day: int) -> DayPoint:
    shared = _observation(step, 0)
    farms = _field(shared, "farms") or []
    farm = farms[player]
    private = _field(_observation(step, player), "private") or {}
    shed = _field(private, "shed") or {}
    owned, crops, animals = _tile_counts(_field(farm, "tiles"))
    return DayPoint(
        day=day,
        money=_money(farm),
        shed_units=sum(int(amount) for amount in shed.values()),
        crop_count=crops,
        animal_count=animals,
        owned_tiles=owned,
    )


def _daily(steps: list[Any], player: int) -> tuple[DayPoint, ...]:
    """Money after each finished day, plus the terminal hour if the episode stops there.

    A midnight observation (next day, hour 0) is the end of the previous day.
    A 720-step episode stops at day 29 hour 23, so that last observation is
    kept as day 29.  It is the official ending cash.
    """

    points: list[DayPoint] = []
    seen: set[int] = set()
    for step in steps:
        shared = _observation(step, 0)
        day = int(_field(shared, "day", 0) or 0)
        hour = int(_field(shared, "hour", 0) or 0)
        if hour != 0 or day <= 0 or (day - 1) in seen:
            continue
        seen.add(day - 1)
        points.append(_snapshot(step, player, day - 1))
    if not steps:
        return tuple(points)
    last = _observation(steps[-1], 0)
    final_day = int(_field(last, "day", 0) or 0)
    final_hour = int(_field(last, "hour", 0) or 0)
    if final_day in seen:
        return tuple(points)
    if final_hour == 0 and (final_day - 1) in seen:
        return tuple(points)
    points.append(_snapshot(steps[-1], player, final_day))
    return tuple(points)


def _sell_map(raw: dict[str, int]) -> dict[str, int]:
    ordered = {item: int(raw.get(item, 0)) for item in SALE_ITEMS}
    for item, amount in raw.items():
        if item not in ordered:
            ordered[str(item)] = int(amount)
    return ordered


def _result_from_books(
    *,
    books: _Books,
    player: int,
    seed: int,
    agent: str,
    opponent: str,
    seat: str,
    steps: int,
    elapsed_seconds: float,
    environment: Any | None,
    trace: dict[str, Any],
    exception: str | None,
) -> SeasonResult:
    episode_steps = list(getattr(environment, "steps", []) or [])
    initial = final = 0
    reward: float | None = None
    status = "ERROR" if exception else "DONE"
    final_day = final_hour = None
    if episode_steps:
        shared_first = _observation(episode_steps[0], 0)
        shared_last = _observation(episode_steps[-1], 0)
        farms_first = _field(shared_first, "farms") or []
        farms_last = _field(shared_last, "farms") or []
        if player < len(farms_first):
            initial = _money(farms_first[player])
        if player < len(farms_last):
            final = _money(farms_last[player])
        final_day = int(_field(shared_last, "day", 0) or 0)
        final_hour = int(_field(shared_last, "hour", 0) or 0)
        terminal = episode_steps[-1][player]
        reward_value = _field(terminal, "reward")
        status = str(_field(terminal, "status") or status)
        if reward_value is not None:
            reward = float(reward_value)
    if exception and status not in {"ERROR", "INVALID", "TIMEOUT"}:
        status = "ERROR"

    sell_by_item = _sell_map(books.sell[player])
    sell_revenue = sum(sell_by_item.values())
    seed_spend = int(books.seed_spend[player])
    animal_spend = int(books.animal_spend[player])
    product_spend = int(books.product_spend[player])
    hire_spend = int(books.hire_spend[player])
    land_spend = int(books.land_spend[player])
    other_income = int(books.other_income[player])
    other_spend = int(books.other_spend[player])
    total_spend = seed_spend + animal_spend + product_spend + hire_spend + land_spend + other_spend
    daily = _daily(episode_steps, player) if episode_steps else ()
    raw_final = final
    return SeasonResult(
        seed=seed,
        agent=agent,
        opponent=opponent,
        seat=seat,
        steps=steps,
        terminal_reward=reward,
        final_money=raw_final,
        initial_money=initial,
        sell_revenue=sell_revenue,
        total_spend=total_spend,
        net_cash_change=raw_final - initial,
        seed_spend=seed_spend,
        animal_spend=animal_spend,
        product_spend=product_spend,
        hire_spend=hire_spend,
        land_spend=land_spend,
        other_income=other_income,
        other_spend=other_spend,
        cash_reconciliation_error=raw_final - (initial + sell_revenue + other_income - total_spend),
        sell_revenue_by_item=sell_by_item,
        successful_hires=int(books.hires[player]),
        seed_units_bought=int(books.seed_units[player]),
        animal_units_bought=int(books.animal_units[player]),
        plant_successes=int(books.plants[player]),
        harvest_successes=int(books.harvests[player]),
        sell_units=int(books.sell_units[player]),
        failed_market_orders=int(books.failed_market[player]),
        failed_unit_actions=int(books.failed_unit[player]),
        pass_hours=int(books.pass_hours[player]),
        move_hours=int(books.move_hours[player]),
        work_hours=int(books.work_hours[player]),
        replans=int(trace.get("replans") or 0),
        daily_money=tuple(point.money for point in daily),
        daily=daily,
        status=status,
        elapsed_seconds=round(elapsed_seconds, 3),
        reward_equals_final_money=reward is not None and abs(reward - raw_final) < 1e-6,
        final_day=final_day,
        final_hour=final_hour,
        exception=exception,
        last_action=trace.get("last_action"),
        last_day=None if trace.get("day") is None else int(trace["day"]),
        last_hour=None if trace.get("hour") is None else int(trace["hour"]),
    )


def run_season(
    agent: AgentSpec,
    opponent: AgentSpec,
    seed: int,
    steps: int = 720,
    seat: str = "left",
) -> SeasonResult:
    """Play one full episode and return the seated agent's ledger."""

    if seat not in {"left", "right"}:
        raise ValueError("seat must be left or right")
    player = 0 if seat == "left" else 1
    agent_name = agent if isinstance(agent, str) else _agent_label(agent)
    opponent_name = opponent if isinstance(opponent, str) else _agent_label(opponent)
    resolved = _resolve_agent(agent)
    watched, trace = _watch(resolved)
    watched.label = agent_name  # type: ignore[attr-defined]
    left: AgentSpec = watched if seat == "left" else opponent
    right: AgentSpec = opponent if seat == "left" else watched
    books = _Books()
    held: dict[str, Any] = {}

    def prepare(environment: Any) -> Callable[[], None]:
        held["environment"] = environment
        return _install(environment, books)

    started = time.perf_counter()
    exception: str | None = None
    try:
        run_match(
            left,
            right,
            steps,
            seed,
            prepare=prepare,
            configuration={"runTimeout": RUN_TIMEOUT_SECONDS},
        )
    except Exception as exc:
        exception = trace.get("error") or f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - started
    return _result_from_books(
        books=books,
        player=player,
        seed=seed,
        agent=agent_name,
        opponent=opponent_name,
        seat=seat,
        steps=steps,
        elapsed_seconds=elapsed,
        environment=held.get("environment"),
        trace=trace,
        exception=exception,
    )


def _pair(region: SeasonResult, baseline: SeasonResult) -> dict[str, Any]:
    return {
        "seed": region.seed,
        "region_final_money": region.final_money,
        "baseline_final_money": baseline.final_money,
        "delta_money": region.final_money - baseline.final_money,
        "region_sell_revenue": region.sell_revenue,
        "baseline_sell_revenue": baseline.sell_revenue,
        "region_total_spend": region.total_spend,
        "baseline_total_spend": baseline.total_spend,
        "region_hires": region.successful_hires,
        "baseline_hires": baseline.successful_hires,
        "region_failed_actions": region.failed_actions,
        "baseline_failed_actions": baseline.failed_actions,
        "region_status": region.status,
        "baseline_status": baseline.status,
    }


def _mean(values: Sequence[float]) -> float:
    return round(mean(values), 3) if values else 0.0


def _mean_attr(games: Sequence[SeasonResult], attr: str) -> float:
    return _mean([float(getattr(game, attr)) for game in games])


def _sell_means(games: Sequence[SeasonResult]) -> dict[str, float]:
    items = list(SALE_ITEMS)
    for game in games:
        for item in game.sell_revenue_by_item:
            if item not in items:
                items.append(item)
    return {item: _mean([game.sell_revenue_by_item.get(item, 0) for game in games]) for item in items}


def summarize(pairs: Sequence[dict[str, Any]], games: Sequence[SeasonResult]) -> dict[str, Any]:
    """Aggregate a paired season run.  Wins compare final money, not reward."""

    deltas = [int(pair["delta_money"]) for pair in pairs]
    region_money = [int(pair["region_final_money"]) for pair in pairs]
    baseline_money = [int(pair["baseline_final_money"]) for pair in pairs]
    best = max(pairs, key=lambda pair: (int(pair["delta_money"]), -int(pair["seed"]))) if pairs else None
    worst = min(pairs, key=lambda pair: (int(pair["delta_money"]), int(pair["seed"]))) if pairs else None
    if games and len({game.agent for game in games}) >= 2:
        candidate_name = games[0].agent
        baseline_name = next(game.agent for game in games if game.agent != candidate_name)
        region_games = [game for game in games if game.agent == candidate_name]
        baseline_games = [game for game in games if game.agent == baseline_name]
    else:
        region_games = list(games)
        baseline_games = []
    summary = {
        "count": len(pairs),
        "region_mean_final_money": _mean(region_money),
        "baseline_mean_final_money": _mean(baseline_money),
        "mean_delta": _mean(deltas),
        "median_delta": round(median(deltas), 3) if deltas else 0.0,
        "region_wins": sum(delta > 0 for delta in deltas),
        "ties": sum(delta == 0 for delta in deltas),
        "baseline_wins": sum(delta < 0 for delta in deltas),
        "best_delta": None if best is None else int(best["delta_money"]),
        "worst_delta": None if worst is None else int(worst["delta_money"]),
        "best_seed": None if best is None else int(best["seed"]),
        "worst_seed": None if worst is None else int(worst["seed"]),
        "region_mean_sell_revenue": _mean([int(pair["region_sell_revenue"]) for pair in pairs]),
        "baseline_mean_sell_revenue": _mean([int(pair["baseline_sell_revenue"]) for pair in pairs]),
        "region_mean_spend": _mean([int(pair["region_total_spend"]) for pair in pairs]),
        "baseline_mean_spend": _mean([int(pair["baseline_total_spend"]) for pair in pairs]),
        "region_mean_hires": _mean([int(pair["region_hires"]) for pair in pairs]),
        "baseline_mean_hires": _mean([int(pair["baseline_hires"]) for pair in pairs]),
        "region_total_failed_actions": sum(int(pair["region_failed_actions"]) for pair in pairs),
        "baseline_total_failed_actions": sum(int(pair["baseline_failed_actions"]) for pair in pairs),
        "region_min_final_money": min(region_money) if region_money else None,
        "region_max_final_money": max(region_money) if region_money else None,
        "all_done": bool(pairs)
        and all(pair["region_status"] == "DONE" and pair["baseline_status"] == "DONE" for pair in pairs),
        "terminal_reward_equals_final_money": bool(games) and all(game.reward_equals_final_money for game in games),
        "max_abs_cash_reconciliation_error": max((abs(game.cash_reconciliation_error) for game in games), default=0),
    }
    for label, group in (("region", region_games), ("baseline", baseline_games)):
        summary[f"{label}_mean_pass_hours"] = _mean_attr(group, "pass_hours")
        summary[f"{label}_mean_move_hours"] = _mean_attr(group, "move_hours")
        summary[f"{label}_mean_work_hours"] = _mean_attr(group, "work_hours")
        summary[f"{label}_mean_plant_successes"] = _mean_attr(group, "plant_successes")
        summary[f"{label}_mean_harvest_successes"] = _mean_attr(group, "harvest_successes")
        summary[f"{label}_mean_sell_units"] = _mean_attr(group, "sell_units")
        summary[f"{label}_mean_replans"] = _mean_attr(group, "replans")
        summary[f"{label}_mean_seed_spend"] = _mean_attr(group, "seed_spend")
        summary[f"{label}_mean_animal_spend"] = _mean_attr(group, "animal_spend")
        summary[f"{label}_mean_product_spend"] = _mean_attr(group, "product_spend")
        summary[f"{label}_mean_hire_spend"] = _mean_attr(group, "hire_spend")
        summary[f"{label}_mean_land_spend"] = _mean_attr(group, "land_spend")
        summary[f"{label}_mean_failed_unit_actions"] = _mean_attr(group, "failed_unit_actions")
        summary[f"{label}_mean_failed_market_orders"] = _mean_attr(group, "failed_market_orders")
        summary[f"{label}_mean_sell_by_item"] = _sell_means(group)
    return summary


def _run_season_job(payload: tuple[str, str, int, int, str]) -> SeasonResult:
    agent, opponent, seed, steps, seat = payload
    return run_season(agent, opponent, seed, steps, seat)


def _append_jsonl(path: Path, game: SeasonResult) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(game), ensure_ascii=False, default=str))
        handle.write("\n")
        handle.flush()


def evaluate_seasons(
    candidate: AgentSpec = "region_phase1",
    baseline: AgentSpec = "route14_phase1",
    opponent: AgentSpec = "starter",
    seeds: Sequence[int] = (),
    steps: int = 720,
    seat: str = "left",
    *,
    workers: int = 1,
    jsonl_path: Path | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run candidate and baseline on the same seeds, seat, opponent, and length."""

    if not seeds:
        raise ValueError("seeds must not be empty")
    if workers < 1:
        raise ValueError("workers must be positive")
    seed_list = [int(seed) for seed in seeds]
    if jsonl_path is not None:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_path.write_text("", encoding="utf-8")

    games: list[SeasonResult] = []
    if workers == 1:
        for seed in seed_list:
            for agent in (candidate, baseline):
                game = run_season(agent, opponent, seed, steps, seat)
                games.append(game)
                if jsonl_path is not None:
                    _append_jsonl(jsonl_path, game)
                if verbose:
                    print(
                        f"seed={game.seed} agent={game.agent} status={game.status} "
                        f"money={game.final_money} reward={game.terminal_reward} "
                        f"elapsed={game.elapsed_seconds}s"
                        + (f" error={game.exception}" if game.exception else ""),
                        file=sys.stderr,
                    )
    else:
        if not all(isinstance(agent, str) for agent in (candidate, baseline, opponent)):
            raise ValueError("parallel seasons require named agents")
        jobs = [
            (str(agent), str(opponent), seed, steps, seat)
            for seed in seed_list
            for agent in (str(candidate), str(baseline))
        ]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for game in pool.map(_run_season_job, jobs):
                games.append(game)
                if jsonl_path is not None:
                    _append_jsonl(jsonl_path, game)
                if verbose:
                    print(
                        f"seed={game.seed} agent={game.agent} status={game.status} "
                        f"money={game.final_money} reward={game.terminal_reward} "
                        f"elapsed={game.elapsed_seconds}s"
                        + (f" error={game.exception}" if game.exception else ""),
                        file=sys.stderr,
                    )
        order = {str(candidate): 0, str(baseline): 1}
        games.sort(key=lambda game: (game.seed, order.get(game.agent, 2)))
        if jsonl_path is not None:
            jsonl_path.write_text("", encoding="utf-8")
            for game in games:
                _append_jsonl(jsonl_path, game)

    by_seed: dict[int, dict[str, SeasonResult]] = {}
    for game in games:
        by_seed.setdefault(game.seed, {})[game.agent] = game
    candidate_name = games[0].agent
    baseline_name = next(game.agent for game in games if game.agent != candidate_name) if len(games) > 1 else candidate_name
    # Named runs keep the requested labels.  Two different callables can share
    # a label; the sequential loop then stored them in candidate-then-baseline
    # order, two per seed.
    if all(isinstance(agent, str) for agent in (candidate, baseline)):
        candidate_name = str(candidate)
        baseline_name = str(baseline)
    pairs = []
    ordered: list[SeasonResult] = []
    for seed in seed_list:
        found = by_seed.get(seed, {})
        if candidate_name == baseline_name:
            seeded = [game for game in games if game.seed == seed]
            region_game, baseline_game = seeded[0], seeded[1]
        else:
            region_game = found[candidate_name]
            baseline_game = found[baseline_name]
        ordered.extend([region_game, baseline_game])
        pairs.append(_pair(region_game, baseline_game))
    if jsonl_path is not None and ordered != games:
        jsonl_path.write_text("", encoding="utf-8")
        for game in ordered:
            _append_jsonl(jsonl_path, game)
    summary = summarize(pairs, ordered)
    summary["candidate"] = candidate_name
    summary["baseline"] = baseline_name
    summary["opponent"] = opponent if isinstance(opponent, str) else _agent_label(opponent)
    summary["seat"] = seat
    summary["steps"] = steps
    summary["seeds"] = seed_list
    return {"games": ordered, "pairs": pairs, "summary": summary}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", default="region_phase1")
    parser.add_argument("--baseline", default="route14_phase1")
    parser.add_argument("--opponent", default="starter")
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--steps", type=int, default=720)
    parser.add_argument("--seat", choices=("left", "right"), default="left")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args(argv)
    seeds = range(args.seed_start, args.seed_start + args.count)
    report = evaluate_seasons(
        args.candidate,
        args.baseline,
        args.opponent,
        seeds,
        steps=args.steps,
        seat=args.seat,
        workers=args.workers,
        jsonl_path=args.jsonl,
        verbose=True,
    )
    payload = dict(report["summary"])
    payload["pairs"] = report["pairs"]
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
