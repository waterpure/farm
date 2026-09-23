"""Hour 0 picks the crew by comparing finished days, not the old hire target."""

from __future__ import annotations

import unittest

from lab.region_phase1 import (
    CrewCandidate,
    _choose_crew,
    _feasible_hire_counts,
    _hire_cost_for_count,
    _morning_route_wheat,
    _plan_for_hires,
    _spendable,
    _take_stock,
    make_region_phase1_agent,
)
from lab.region_route import RegionRoutePlan
from lab.route14_phase1 import MAX_MARKET_ORDERS, SALE_RANK, DayRoute
from lab.route14_state import parse_world
from lab.task_grid import FEED, WATER, TaskBucket, TaskGrid, WaterTask, build_task_grid
from lab.test_route14_phase1 import _animal
from lab.test_region_production import _crop_plan
from lab.test_route14_phase1 import _observation, _plant, _tiles


def _legacy(**overrides: object) -> DayRoute:
    values: dict[str, object] = dict(
        extra_hires=9,
        target_hires=9,
        start_hour=0,
        wheat_buy=0,
        wages=500,
        wheat_cost=0,
        actors=[],
        batches=[],
        survival_total=0,
        survival_done=0,
        harvests_total=0,
        harvests_delivered=0,
        revenue=0,
        net=0,
    )
    values.update(overrides)
    return DayRoute(**values)  # type: ignore[arg-type]


def _waters(count: int) -> list[TaskBucket]:
    coords = [(x, y) for y in range(5) for x in range(5)]
    buckets = []
    for x, y in coords[:count]:
        bucket = TaskBucket((x, y), "WHEAT")
        bucket.tasks[WATER] = WaterTask(WATER, "PENDING", True)
        buckets.append(bucket)
    return buckets


def _grid(*buckets: TaskBucket) -> TaskGrid:
    grid = TaskGrid()
    for bucket in buckets:
        grid.put(bucket)
    return grid


def _dry_square() -> list[list]:
    tiles = _tiles()
    for x in range(5):
        for y in range(5):
            tiles[y][x] = _plant("WHEAT", dry=1)
    return tiles


class CrewSizeTests(unittest.TestCase):
    def test_morning_forecast_wheat_feeds_animals_without_overallocating(self) -> None:
        tiles = _tiles()
        tiles[4][2] = _animal("SHEEP", unfed=1)
        tiles[3][2] = _animal("SHEEP", unfed=1)
        observation = _observation(tiles, day=1, money=5000, farmer=(4, 4))
        world = parse_world(observation)
        grid = build_task_grid(world)
        route = _legacy(wheat_buy=2, wheat_cost=50)

        self.assertEqual(_morning_route_wheat(world, route), 2)
        chosen = _plan_for_hires(observation, world, grid, route, 0)
        self.assertTrue(chosen.feasible)
        assert chosen.plan is not None
        actions = [action for worker in chosen.plan.worker_routes for action in worker.actions_by_hour]
        pickups = [action for action in actions if action.operation == "PICKUP" and action.args[:1] == ("WHEAT",)]
        feeds = [action for action in actions if action.operation == FEED]

        self.assertEqual(sum(int(action.args[1]) for action in pickups), 2)
        self.assertEqual(len(feeds), 2)
        self.assertEqual(sum(int(action.args[1]) for action in pickups), len(feeds))

    def test_morning_route_does_not_invent_wheat_without_a_forecast(self) -> None:
        tiles = _tiles()
        tiles[4][2] = _animal("SHEEP", unfed=1)
        observation = _observation(tiles, day=1, money=5000, farmer=(4, 4))
        world = parse_world(observation)
        grid = build_task_grid(world)
        chosen = _plan_for_hires(observation, world, grid, _legacy(wheat_buy=0), 0)

        self.assertTrue(chosen.feasible)
        assert chosen.plan is not None
        actions = [action for worker in chosen.plan.worker_routes for action in worker.actions_by_hour]
        self.assertFalse(any(action.operation == "PICKUP" and action.args[:1] == ("WHEAT",) for action in actions))
        self.assertFalse(any(action.operation == FEED for action in actions))

    def test_execution_stock_only_allows_real_wheat_pickups(self) -> None:
        tiles = _tiles()
        observation = _observation(tiles, shed={}, farmer=(4, 4))
        world = parse_world(observation)
        stock = _spendable(world)

        self.assertFalse(_take_stock(["PICKUP", "WHEAT", 1], stock))
        self.assertEqual(stock["wheat"], 0)

        observation = _observation(tiles, shed={"WHEAT": 1}, farmer=(4, 4))
        stock = _spendable(parse_world(observation))
        self.assertTrue(_take_stock(["PICKUP", "WHEAT", 1], stock))
        self.assertEqual(stock["wheat"], 0)
        self.assertFalse(_take_stock(["PICKUP", "WHEAT", 1], stock))

    def test_hire_prices_rise_one_hand_at_a_time(self) -> None:
        self.assertEqual(_hire_cost_for_count(0, 0), 0)
        self.assertEqual(_hire_cost_for_count(0, 1), 1)
        self.assertEqual(_hire_cost_for_count(0, 2), 2)
        self.assertEqual(_hire_cost_for_count(0, 3), 4)
        self.assertEqual(_hire_cost_for_count(2, 1), 2)

    def test_an_existing_hand_shortens_the_list_of_crews(self) -> None:
        alone = parse_world(_observation(_tiles(), farmer=(4, 4)))
        paired = parse_world(_observation(_tiles(), farmer=(4, 4), hands=[(5, 4)]))
        self.assertEqual(_feasible_hire_counts(alone), [0, 1, 2, 3])
        self.assertEqual(_feasible_hire_counts(paired), [0, 1, 2])

    def test_nobody_is_hired_when_the_farmer_finishes_and_nothing_is_planted(self) -> None:
        observation = _observation(_tiles(), money=5000, farmer=(4, 4))
        best, summary = _choose_crew(observation, parse_world(observation), _grid(), _legacy())

        self.assertEqual(best.new_hires, 0)
        self.assertEqual(_hires(best), 0)
        self.assertEqual(summary[0][0], 0)
        self.assertTrue(all(item[0] == 0 or item[2] == 0 for item in summary))

    def test_one_hand_is_hired_when_the_farmer_cannot_finish_the_watering(self) -> None:
        observation = _observation(_tiles(), money=5000, farmer=(0, 0))
        grid = _grid(*_waters(13))
        world = parse_world(observation)
        alone = _plan_for_hires(observation, world, grid, _legacy(), 0)
        best, _summary = _choose_crew(observation, world, grid, _legacy())

        self.assertGreater(alone.mandatory_unfinished, 0)
        self.assertEqual(best.new_hires, 1)
        self.assertEqual(best.mandatory_unfinished, 0)
        self.assertEqual(_hires(best), 1)

    def test_a_second_hand_is_not_hired_when_the_day_is_already_finished(self) -> None:
        observation = _observation(_tiles(), money=5000, farmer=(0, 0))
        grid = _grid(*_waters(13))
        world = parse_world(observation)
        one = _plan_for_hires(observation, world, grid, _legacy(), 1)
        two = _plan_for_hires(observation, world, grid, _legacy(), 2)
        best, _summary = _choose_crew(observation, world, grid, _legacy())

        self.assertEqual(one.mandatory_unfinished, 0)
        self.assertEqual(two.mandatory_unfinished, 0)
        self.assertEqual(one.production_value, two.production_value)
        self.assertEqual(best.new_hires, 1)

    def test_a_hand_is_hired_when_that_hand_plants_more_than_the_wage(self) -> None:
        observation = _observation(_tiles(), money=5000, farmer=(0, 0))
        grid = _grid(*_waters(20), *[_crop_plan(x, 4, "MELON", cash=80, money=10) for x in range(5)])
        world = parse_world(observation)
        one = _plan_for_hires(observation, world, grid, _legacy(), 1)
        two = _plan_for_hires(observation, world, grid, _legacy(), 2)
        best, _summary = _choose_crew(observation, world, grid, _legacy())

        self.assertEqual(one.mandatory_unfinished, 0)
        self.assertEqual(two.mandatory_unfinished, 0)
        self.assertGreater(two.production_value - one.production_value, two.hire_cost - one.hire_cost)
        self.assertEqual(best.new_hires, 2)
        self.assertEqual(best.production_value, 50)

    def test_an_idle_extra_hand_loses_to_the_smaller_crew(self) -> None:
        observation = _observation(_tiles(), money=5000, farmer=(0, 0))
        grid = _grid(*_waters(12), _crop_plan(4, 4, "MELON", cash=80, money=100))
        world = parse_world(observation)
        one = _plan_for_hires(observation, world, grid, _legacy(), 1)
        two = _plan_for_hires(observation, world, grid, _legacy(), 2)
        best, _summary = _choose_crew(observation, world, grid, _legacy())

        self.assertEqual(one.production_value, two.production_value)
        self.assertGreater(one.net_production_value, two.net_production_value)
        self.assertEqual(best.new_hires, 1)

    def test_a_crew_that_cannot_fit_its_hires_into_the_morning_market_is_dropped(self) -> None:
        observation = _observation(
            _tiles(),
            money=5000,
            farmer=(4, 4),
            shed={item: 1 for item in list(SALE_RANK)[:8]},
        )
        world = parse_world(observation)
        grid = _grid(*_waters(25))
        rejected = _plan_for_hires(observation, world, grid, _legacy(), 3)
        best, _summary = _choose_crew(observation, world, grid, _legacy())

        self.assertFalse(rejected.feasible)
        self.assertIsNone(rejected.plan)
        self.assertEqual(_hires(rejected), 0)
        self.assertLessEqual(best.new_hires, 2)
        self.assertEqual(len(best.crew), 1 + best.new_hires)
        self.assertEqual(_hires(best), best.new_hires)

    def test_a_crew_that_cannot_pay_its_hires_is_dropped(self) -> None:
        observation = _observation(_tiles(), money=2, farmer=(4, 4))
        world = parse_world(observation)
        grid = _grid(*_waters(25))
        rejected = _plan_for_hires(observation, world, grid, _legacy(), 3)
        affordable = _plan_for_hires(observation, world, grid, _legacy(), 2)
        best, _summary = _choose_crew(observation, world, grid, _legacy())

        self.assertFalse(rejected.feasible)
        self.assertEqual(rejected.hire_cost, 4)
        self.assertTrue(affordable.feasible)
        self.assertNotEqual(best.new_hires, 3)
        self.assertLessEqual(best.hire_cost, 2)

    def test_a_crop_one_crew_cannot_reach_is_still_open_for_the_next_crew(self) -> None:
        observation = _observation(_tiles(), money=5000, farmer=(0, 0))
        grid = _grid(*_waters(12), _crop_plan(4, 4, "MELON", cash=80, money=100))
        world = parse_world(observation)
        short = _plan_for_hires(observation, world, grid, _legacy(), 0)
        longer = _plan_for_hires(observation, world, grid, _legacy(), 1)

        self.assertEqual(short.production_value, 0)
        self.assertEqual(longer.production_value, 100)
        self.assertEqual(_plants(longer), [(4, 4)])
        self.assertIsNotNone(grid[4][4].production_plan)
        self.assertEqual(grid[4][4].production_plan.name, "MELON")

    def test_the_old_hire_target_does_not_choose_the_crew(self) -> None:
        observation = _observation(_tiles(), money=5000, farmer=(4, 4))
        best, _summary = _choose_crew(observation, parse_world(observation), _grid(), _legacy(target_hires=3))
        agent = make_region_phase1_agent()
        morning = agent(observation)

        self.assertEqual(best.new_hires, 0)
        self.assertEqual(agent.telemetry["planned_new_hires"], 0)
        self.assertNotIn(["HIRE"], morning["market"])
        self.assertEqual(_hires_in(agent.telemetry["market_queue"]), 0)

    def test_three_new_hands_stand_on_the_official_doors(self) -> None:
        observation = _observation(_tiles(), money=5000, farmer=(4, 4))
        chosen = _plan_for_hires(observation, parse_world(observation), _grid(*_waters(25)), _legacy(), 3)

        self.assertTrue(chosen.feasible)
        self.assertEqual(
            [worker.coord for worker in chosen.crew],
            [(4, 4), (5, 4), (4, 5), (5, 5)],
        )
        self.assertEqual(_hires(chosen), 3)

    def test_the_saved_hire_count_matches_the_morning_queue(self) -> None:
        agent = make_region_phase1_agent()
        morning = agent(_observation(_dry_square(), money=5000, farmer=(4, 4)))
        queued = _hires_in(agent.telemetry["market_queue"])

        self.assertEqual(agent.telemetry["planned_new_hires"], queued)
        self.assertEqual(morning["market"].count(["HIRE"]), queued)
        self.assertEqual(queued, 2)
        self.assertLessEqual(len(morning["market"]), MAX_MARKET_ORDERS)

    def test_hour_one_keeps_the_crew_the_morning_already_chose(self) -> None:
        tiles = _dry_square()
        agent = make_region_phase1_agent()
        agent(_observation(tiles, day=3, hour=0, money=5000, farmer=(4, 4)))
        plan = agent.telemetry["plan"]
        hands = [coord for name, coord in agent.telemetry["planned_workers"] if name != "Farmer"]

        self.assertEqual(agent.telemetry["planned_new_hires"], 2)
        self.assertEqual(hands, [(5, 4), (4, 5)])
        agent(_observation(tiles, day=3, hour=1, money=5000, farmer=(4, 4), hands=hands))
        self.assertIs(agent.telemetry["plan"], plan)
        self.assertFalse(agent.telemetry["needs_replan"])

    def test_unfinished_mandatory_work_beats_a_richer_planting(self) -> None:
        unfinished = CrewCandidate(0, [], _idle(), {}, 1, 1000, 0, 0, 1)
        finished = CrewCandidate(1, [], _idle(), {}, 0, 1, 1, 5, 4)

        self.assertLess(finished.key, unfinished.key)
        observation = _observation(_tiles(), money=5000, farmer=(0, 0))
        grid = _grid(*_waters(13), _crop_plan(4, 4, "MELON", cash=80, money=100))
        best, _summary = _choose_crew(observation, parse_world(observation), grid, _legacy())
        self.assertEqual(best.mandatory_unfinished, 0)
        self.assertEqual(best.new_hires, 1)

    def test_the_chosen_market_never_passes_ten_orders(self) -> None:
        observation = _observation(
            _tiles(),
            money=5000,
            farmer=(4, 4),
            shed={item: 1 for item in list(SALE_RANK)[:4]},
        )
        best, _summary = _choose_crew(observation, parse_world(observation), _grid(*_waters(25)), _legacy())

        self.assertTrue(all(len(tasks) <= MAX_MARKET_ORDERS for tasks in best.market_queue.values()))
        self.assertLessEqual(4 + _hires(best), MAX_MARKET_ORDERS)


def _hires(candidate: CrewCandidate) -> int:
    return sum(1 for task in candidate.market_queue.get(0, []) if task.operation == "HIRE")


def _hires_in(queue: dict) -> int:
    return sum(1 for task in queue.get(0, []) if task.operation == "HIRE")


def _plants(candidate: CrewCandidate) -> list[tuple[int, int]]:
    if candidate.plan is None:
        return []
    return [
        action.coord
        for route in candidate.plan.worker_routes
        for action in route.actions_by_hour
        if action.operation == "PLANT" and action.coord is not None
    ]


def _idle() -> RegionRoutePlan:
    return RegionRoutePlan(True, (), 0, None, ())
