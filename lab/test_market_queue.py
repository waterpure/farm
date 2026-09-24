"""Hour 0 writes the route, then buys only what that route still lacks."""

from __future__ import annotations

import unittest

from lab.market_queue import (
    SupermarketTask,
    derive_supermarket_tasks,
    engine_order,
    schedule_market_queue,
)
from lab.production_plan import PlannedAction, ProductionPlan
from lab.region_phase1 import (
    _action_for,
    _commit_day_plan,
    _plan_for_hires,
    _predicted_crew,
    make_region_phase1_agent,
)
from lab.region_route import (
    RegionRoutePlan,
    RegionWorker,
    RouteAction,
    TileVisit,
    WorkerRoutePlan,
    plan_region_routes,
)
from lab.route14_phase1 import MAX_MARKET_ORDERS, SALE_RANK, DayRoute
from lab.route14_state import parse_world
from lab.task_grid import BUILD_PASTURE, PLACE_ANIMAL, PLANT, WATER, TaskBucket, TaskGrid, TileTask
from lab.test_region_production import _crop_plan, _water
from lab.test_route14_phase1 import _observation, _plant, _tiles


def _route(**overrides: object) -> DayRoute:
    values: dict[str, object] = dict(
        extra_hires=0,
        target_hires=0,
        start_hour=0,
        wheat_buy=0,
        wages=0,
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


def _animal_plan(x: int, y: int, name: str = "SHEEP", money: float = 10, cash: int = 500) -> TaskBucket:
    plan = ProductionPlan(
        (x, y),
        "animal",
        name,
        "WOOL" if name == "SHEEP" else name,
        (PlannedAction(BUILD_PASTURE), PlannedAction(PLACE_ANIMAL, name)),
        money,
        money,
        cash,
        True,
        True,
        3,
    )
    bucket = TaskBucket((x, y), "EMPTY", production_plan=plan)
    bucket.tasks[BUILD_PASTURE] = TileTask(BUILD_PASTURE, "PENDING", False)
    bucket.tasks[PLACE_ANIMAL] = TileTask(PLACE_ANIMAL, "PENDING", False, subject=name, depends_on=BUILD_PASTURE)
    return bucket


def _grid(*buckets: TaskBucket) -> TaskGrid:
    grid = TaskGrid()
    for bucket in buckets:
        grid.put(bucket)
    return grid


def _visit(coord: tuple[int, int], kind: str, name: str) -> TileVisit:
    return TileVisit(coord, (PLANT,), 1, "EMPTY", production_kind=kind, production_name=name)


def _plan_with(*routes: WorkerRoutePlan) -> RegionRoutePlan:
    return RegionRoutePlan(True, routes, 0, None, ())


def _worker(actions: list[RouteAction], visits: list[TileVisit] | None = None) -> WorkerRoutePlan:
    return WorkerRoutePlan("Farmer", (4, 4), tuple(visits or []), tuple(actions), 0, None)


class MarketQueueTests(unittest.TestCase):
    def test_buy_land_is_an_atomic_hour_zero_order(self) -> None:
        land = SupermarketTask("BUY_LAND", deadline=0)
        queue, failed = schedule_market_queue([land], {0: 0})
        self.assertEqual(failed, [])
        self.assertEqual(queue[0], [land])
        self.assertEqual(engine_order(land), ["BUY_LAND"])

    def test_hires_fill_the_least_crowded_door_in_nwse_order(self) -> None:
        world = parse_world(_observation(_tiles(), farmer=(4, 4)))
        crew = _predicted_crew(world, 3, 10)
        self.assertEqual([worker.coord for worker in crew], [(4, 4), (5, 4), (4, 5), (5, 5)])

        crowded = parse_world(_observation(_tiles(), farmer=(4, 4), hands=[(5, 4)]))
        nxt = _predicted_crew(crowded, 1, 10)
        self.assertEqual(nxt[-1].coord, (4, 5))

        empty = parse_world(_observation(_tiles(), farmer=(0, 0)))
        self.assertEqual(_predicted_crew(empty, 1, 10)[-1].coord, (4, 4))
        self.assertEqual(len(_predicted_crew(world, 10, 10)), 4)

    def test_an_unpaid_crop_can_enter_the_morning_route_and_not_the_old_one(self) -> None:
        grid = _grid(_crop_plan(1, 1, "STRAWBERRY", cash=100))
        farmer = [RegionWorker("Farmer", (1, 1))]
        unpaid = plan_region_routes(grid, farmer)
        paid = plan_region_routes(grid, farmer, include_unpaid_production=True)
        blocked = plan_region_routes(
            grid,
            farmer,
            include_unpaid_production=True,
            blocked_production=[(1, 1)],
        )

        self.assertFalse(_plants(unpaid))
        self.assertEqual(_plants(paid), [(1, "STRAWBERRY")])
        self.assertFalse(_plants(blocked))

    def test_owned_seeds_cover_the_earliest_planting_and_one_order_covers_the_rest(self) -> None:
        route = _plan_with(
            _worker(
                [
                    RouteAction(4, PLANT, (2, 3), ("STRAWBERRY",)),
                    RouteAction(8, PLANT, (2, 1), ("STRAWBERRY",)),
                    RouteAction(12, PLANT, (3, 1), ("MELON",)),
                    RouteAction(16, PLANT, (3, 2), ("MELON",)),
                    RouteAction(18, PLANT, (3, 3), ("MELON",)),
                ],
                [
                    _visit((2, 3), "crop", "STRAWBERRY"),
                    _visit((2, 1), "crop", "STRAWBERRY"),
                    _visit((3, 1), "crop", "MELON"),
                    _visit((3, 2), "crop", "MELON"),
                    _visit((3, 3), "crop", "MELON"),
                ],
            )
        )
        tasks = derive_supermarket_tasks(route, {"STRAWBERRY": 1, "MELON": 3}, {}, 0)
        seeds = {task.item: task for task in tasks if task.operation == "BUY_SEED"}

        self.assertEqual(seeds["STRAWBERRY"].amount, 1)
        self.assertEqual(seeds["STRAWBERRY"].deadline, 7)
        self.assertEqual(seeds["STRAWBERRY"].source_coords, ((2, 1),))
        self.assertNotIn("MELON", seeds)

    def test_three_missing_melons_are_one_order_at_the_earliest_shortage(self) -> None:
        route = _plan_with(
            _worker(
                [
                    RouteAction(4, PLANT, (1, 1), ("MELON",)),
                    RouteAction(8, PLANT, (1, 2), ("MELON",)),
                    RouteAction(12, PLANT, (1, 3), ("MELON",)),
                ]
            )
        )
        tasks = [task for task in derive_supermarket_tasks(route, {"MELON": 0}, {}, 0) if task.operation == "BUY_SEED"]

        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].item, "MELON")
        self.assertEqual(tasks[0].amount, 3)
        self.assertEqual(tasks[0].deadline, 3)

    def test_animals_already_in_the_shed_or_in_hand_are_not_bought(self) -> None:
        route = _plan_with(
            _worker(
                [RouteAction(1, "PICKUP", (4, 4), ("SHEEP", 1))],
                [_visit((3, 4), "animal", "SHEEP"), _visit((3, 3), "animal", "SHEEP")],
            )
        )
        bought = [
            task
            for task in derive_supermarket_tasks(route, {}, {"SHEEP": 1}, 0)
            if task.operation == "BUY_ANIMAL"
        ]
        self.assertEqual(bought, [])

        short = _plan_with(
            _worker(
                [RouteAction(1, "PICKUP", (4, 4), ("SHEEP", 2)), RouteAction(1, "PICKUP", (4, 4), ("COW", 1))],
                [_visit((3, 4), "animal", "SHEEP"), _visit((3, 3), "animal", "SHEEP"), _visit((2, 4), "animal", "COW")],
            )
        )
        orders = {
            task.item: task
            for task in derive_supermarket_tasks(short, {}, {"SHEEP": 1}, 0)
            if task.operation == "BUY_ANIMAL"
        }
        self.assertEqual(orders["SHEEP"].amount, 1)
        self.assertEqual(orders["SHEEP"].deadline, 0)
        self.assertEqual(orders["COW"].amount, 1)
        self.assertEqual(orders["COW"].deadline, 0)

    def test_committed_animal_feed_adds_one_hour_zero_wheat_order(self) -> None:
        route = _plan_with(
            _worker(
                [
                    RouteAction(1, "PICKUP", (4, 4), ("WHEAT", 2)),
                    RouteAction(2, "PICKUP", (4, 4), ("SHEEP", 1)),
                    RouteAction(5, "PLACE", (3, 4), ("SHEEP",)),
                    RouteAction(6, "FEED", (3, 4)),
                ],
                [_visit((3, 4), "animal", "SHEEP")],
            )
        )
        tasks = derive_supermarket_tasks(
            route,
            {},
            {"SHEEP": 1},
            0,
            shed_wheat=0,
            forecast_wheat=0,
            wheat_cost=26,
        )
        wheat = [task for task in tasks if task.operation == "BUY_PRODUCT"]
        self.assertEqual([(task.item, task.amount, task.deadline, task.cost) for task in wheat], [("WHEAT", 2, 0, 26)])
        queue, failed = schedule_market_queue(tasks, {0: 0})
        self.assertEqual(failed, [])
        self.assertEqual(queue[0], wheat)

    def test_three_hires_are_three_orders_and_seeds_do_not_take_their_slots(self) -> None:
        hires = [SupermarketTask("HIRE", deadline=0) for _ in range(4)]
        seeds = [SupermarketTask("BUY_SEED", f"SEED{index}", 1, 0, ((index, 0),)) for index in range(5)]
        animal = SupermarketTask("BUY_ANIMAL", "SHEEP", 1, 0, ((3, 4),))
        queue, failed = schedule_market_queue([*seeds, animal, *hires], {0: 6})

        placed = queue[0]
        self.assertEqual([task.operation for task in placed], ["HIRE", "HIRE", "HIRE", "HIRE"])
        self.assertEqual(len(failed), 6)
        self.assertTrue(all(task.operation != "HIRE" for task in failed))
        self.assertLessEqual(len(placed), MAX_MARKET_ORDERS)

    def test_a_seed_waits_until_its_deadline_and_stops_at_ten_orders(self) -> None:
        seed = SupermarketTask("BUY_SEED", "STRAWBERRY", 1, 7, ((2, 3),))
        queue, failed = schedule_market_queue([seed], {7: 10, 6: 10, 5: 2})
        self.assertEqual(failed, [])
        self.assertEqual(queue[5], [seed])
        self.assertEqual(queue[7], [])

        packed = [SupermarketTask("BUY_SEED", "MELON", 1, 3, ((index, 0),)) for index in range(8)]
        reserved = {0: MAX_MARKET_ORDERS, 1: MAX_MARKET_ORDERS, 2: MAX_MARKET_ORDERS, 3: 3}
        fitted, left = schedule_market_queue(packed, reserved)
        self.assertEqual(len(fitted[3]), 7)
        self.assertEqual(len(left), 1)
        self.assertTrue(all(len(hour_tasks) <= MAX_MARKET_ORDERS for hour_tasks in fitted.values()))

    def test_a_seed_that_cannot_be_bought_before_planting_drops_the_whole_chain(self) -> None:
        grid = _grid(_water(0, 0), _crop_plan(2, 3, "MELON", cash=80, money=4))
        observation = _observation(
            _tiles(),
            money=5000,
            farmer=(2, 3),
            shed={item: 1 for item in SALE_RANK},
        )
        plan, queue, _crew = _commit_day_plan(observation, parse_world(observation), grid, _route(wheat_buy=1))
        actions = [action for route in plan.worker_routes for action in route.actions_by_hour]

        self.assertFalse(any(action.operation == PLANT for action in actions))
        self.assertFalse(any(action.coord == (2, 3) and action.operation == WATER for action in actions))
        self.assertTrue(any(action.coord == (0, 0) and action.operation == WATER for action in actions))
        self.assertFalse(any(task.operation == "BUY_SEED" for tasks in queue.values() for task in tasks))

    def test_the_morning_buys_a_seed_the_hour_before_it_is_planted(self) -> None:
        grid = _grid(_water(2, 1), _crop_plan(2, 3, "STRAWBERRY", cash=100))
        observation = _observation(_tiles(), money=5000, farmer=(4, 4))
        plan, queue, crew = _commit_day_plan(observation, parse_world(observation), grid, _route())
        plants = [
            action
            for route in plan.worker_routes
            for action in route.actions_by_hour
            if action.operation == PLANT and action.args == ("STRAWBERRY",)
        ]
        buys = [task for tasks in queue.values() for task in tasks if task.operation == "BUY_SEED"]

        self.assertEqual([worker.coord for worker in crew], [(4, 4)])
        self.assertEqual(len(plants), 1)
        self.assertEqual(len(buys), 1)
        self.assertEqual(buys[0].amount, 1)
        self.assertEqual(buys[0].deadline, plants[0].hour - 1)
        self.assertIn(buys[0], queue[plants[0].hour - 1])
        self.assertNotIn(buys[0], queue[0] if plants[0].hour > 1 else [])

    def test_hour_zero_hires_before_it_buys_the_animal_the_route_will_pick_up(self) -> None:
        grid = _grid(_water(2, 1), _crop_plan(2, 3, "STRAWBERRY", cash=100), _animal_plan(3, 4))
        observation = _observation(_tiles(), money=5000, farmer=(4, 4))
        chosen = _plan_for_hires(
            observation,
            parse_world(observation),
            grid,
            _route(target_hires=3, extra_hires=3, wages=99),
            2,
        )
        self.assertTrue(chosen.feasible)
        self.assertIsNotNone(chosen.plan)
        morning = chosen.market_queue[0]
        assert chosen.plan is not None
        plants = [action for route in chosen.plan.worker_routes for action in route.actions_by_hour if action.operation == PLANT]
        pickups = [
            action
            for route in chosen.plan.worker_routes
            for action in route.actions_by_hour
            if action.operation == "PICKUP" and action.args and action.args[0] == "SHEEP"
        ]

        self.assertEqual([worker.coord for worker in chosen.crew], [(4, 4), (5, 4), (4, 5)])
        self.assertEqual(sum(1 for task in morning if task.operation == "HIRE"), 2)
        self.assertTrue(plants)
        self.assertTrue(pickups)
        self.assertEqual(pickups[0].hour, 1)
        animals = [task for task in morning if task.operation == "BUY_ANIMAL"]
        self.assertEqual([(task.item, task.amount, task.deadline) for task in animals], [("SHEEP", 1, 0)])
        seed_hours = [
            hour for hour, tasks in chosen.market_queue.items() if any(task.operation == "BUY_SEED" for task in tasks)
        ]
        self.assertEqual(seed_hours, [plants[0].hour - 1])

    def test_a_sheep_that_does_not_fit_is_dropped_and_the_hires_stay(self) -> None:
        grid = _grid(_water(0, 0), _animal_plan(3, 4, money=9), _animal_plan(3, 3, money=1))
        observation = _observation(
            _tiles(),
            money=5000,
            farmer=(4, 4),
            shed={item: 1 for item in list(SALE_RANK)[:7]},
        )
        chosen = _plan_for_hires(
            observation,
            parse_world(observation),
            grid,
            _route(target_hires=4, extra_hires=4, wages=99),
            3,
        )
        self.assertTrue(chosen.feasible)
        self.assertIsNotNone(chosen.plan)
        assert chosen.plan is not None
        self.assertEqual(sum(1 for task in chosen.market_queue[0] if task.operation == "HIRE"), 3)
        bought = [task for tasks in chosen.market_queue.values() for task in tasks if task.operation == "BUY_ANIMAL"]
        self.assertEqual(bought, [])
        placed = [
            action.coord
            for route in chosen.plan.worker_routes
            for action in route.actions_by_hour
            if action.operation == "PLACE"
        ]
        self.assertEqual(placed, [])
        self.assertTrue(
            any(action.operation == WATER for route in chosen.plan.worker_routes for action in route.actions_by_hour)
        )

    def test_cash_for_one_sheep_keeps_the_richer_pasture_only(self) -> None:
        grid = _grid(_animal_plan(1, 1, money=9), _animal_plan(1, 2, money=1))
        observation = _observation(_tiles(), money=500, farmer=(4, 4))
        plan, queue, _crew = _commit_day_plan(observation, parse_world(observation), grid, _route())
        bought = [task for tasks in queue.values() for task in tasks if task.operation == "BUY_ANIMAL"]
        placed = [
            action.coord
            for route in plan.worker_routes
            for action in route.actions_by_hour
            if action.operation == "PLACE"
        ]

        self.assertEqual([(task.item, task.amount) for task in bought], [("SHEEP", 1)])
        self.assertIn((1, 1), placed)
        self.assertNotIn((1, 2), placed)

    def test_a_plan_outside_the_route_is_not_bought(self) -> None:
        grid = _grid(_water(0, 0), _crop_plan(0, 1, "MELON", cash=80), _crop_plan(9, 9, "MELON", cash=80))
        observation = _observation(_tiles(), money=5000, farmer=(0, 0))
        plan, queue, _crew = _commit_day_plan(observation, parse_world(observation), grid, _route())
        buys = [task for tasks in queue.values() for task in tasks if task.operation == "BUY_SEED"]
        planted = [action.coord for route in plan.worker_routes for action in route.actions_by_hour if action.operation == PLANT]

        self.assertEqual(planted, [(0, 1)])
        self.assertEqual([(task.item, task.amount) for task in buys], [("MELON", 1)])
        self.assertNotIn((9, 9), buys[0].source_coords)

    def test_hour_one_plays_the_morning_plan_when_the_birth_tile_matches(self) -> None:
        tiles = _tiles()
        for x in range(5):
            for y in range(5):
                tiles[y][x] = _plant("WHEAT", dry=1)
        agent = make_region_phase1_agent()
        observation = _observation(tiles, day=1, hour=0, farmer=(4, 4))
        observation["private"]["seeds"] = {"MELON": 1}
        observation["private"]["shed"] = {"SHEEP": 1}
        morning = agent(observation)
        plan = agent.telemetry["plan"]
        hands = [coord for name, coord in agent.telemetry["planned_workers"] if name != "Farmer"]
        queued_hires = [task for task in agent.telemetry["market_queue"][0] if task.operation == "HIRE"]

        self.assertEqual(observation["private"]["seeds"], {"MELON": 1})
        self.assertEqual(observation["private"]["shed"], {"SHEEP": 1})
        self.assertEqual([order for order in morning["market"] if order == ["HIRE"]], [["HIRE"] for _ in queued_hires])
        self.assertEqual(hands, [(5, 4), (4, 5)])
        self.assertEqual(agent.telemetry["planned_new_hires"], 2)

        agent(_observation(tiles, day=1, hour=1, farmer=(4, 4), hands=hands))
        self.assertIs(agent.telemetry["plan"], plan)
        self.assertFalse(agent.telemetry["needs_replan"])

    def test_a_plant_without_a_seed_passes_and_asks_for_a_new_plan(self) -> None:
        route = _worker([RouteAction(4, PLANT, (2, 3), ("STRAWBERRY",))])
        state = {"routes_by_worker_id": {"Farmer": route}, "force_replan": False}
        stock = {"seeds": {}, "animals": {"SHEEP": 0}}

        self.assertEqual(_action_for("Farmer", state, 4, stock), ["PASS"])
        self.assertTrue(state["force_replan"])
        self.assertEqual(stock["seeds"], {})


def _plants(plan: RegionRoutePlan) -> list[tuple[int, str]]:
    return [
        (action.hour, str(action.args[0]))
        for route in plan.worker_routes
        for action in route.actions_by_hour
        if action.operation == PLANT and action.args
    ]
