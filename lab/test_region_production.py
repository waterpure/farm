"""A committed empty-tile plan rides along only when the whole chain fits today."""

from __future__ import annotations

import unittest

from kaggle_environments import make

from lab.production_plan import PlannedAction, ProductionPlan
from lab.region_phase1 import _assignments, _crew, _shed_animals, make_region_phase1_agent
from lab.region_route import RegionWorker, plan_region_routes
from lab.route14_state import COMPLETED, PENDING, SCHEDULED, parse_world
from lab.task_grid import (
    BUILD_PASTURE,
    PLACE_ANIMAL,
    PLANT,
    WATER,
    FeedTask,
    TaskBucket,
    TaskGrid,
    TileTask,
    WaterTask,
    apply_assignments,
    build_task_grid,
)
from lab.test_region_phase1 import _animal
from lab.test_route14_phase1 import _observation, _plant, _tiles


def _crop_plan(x: int, y: int, name: str = "MELON", cash: int = 0, money: float = 10) -> TaskBucket:
    plan = ProductionPlan(
        (x, y),
        "crop",
        name,
        name,
        (PlannedAction(PLANT, name), PlannedAction(WATER, name)),
        money,
        money,
        cash,
        True,
        True,
        2,
    )
    bucket = TaskBucket((x, y), "EMPTY", production_plan=plan)
    bucket.tasks[PLANT] = TileTask(PLANT, PENDING, False, subject=name)
    bucket.tasks[WATER] = TileTask(WATER, PENDING, False, subject=name, depends_on=PLANT)
    return bucket


def _animal_plan(x: int, y: int, name: str = "SHEEP", structure: str = BUILD_PASTURE, money: float = 10) -> TaskBucket:
    plan = ProductionPlan(
        (x, y),
        "animal",
        name,
        "WOOL" if name == "SHEEP" else name,
        (PlannedAction(structure), PlannedAction(PLACE_ANIMAL, name)),
        money,
        money,
        0,
        True,
        True,
        3,
    )
    bucket = TaskBucket((x, y), "EMPTY", production_plan=plan)
    bucket.tasks[structure] = TileTask(structure, PENDING, False, depends_on="")
    bucket.tasks[PLACE_ANIMAL] = TileTask(PLACE_ANIMAL, PENDING, False, subject=name, depends_on=structure)
    return bucket


def _water(x: int, y: int) -> TaskBucket:
    bucket = TaskBucket((x, y), "WHEAT")
    bucket.tasks[WATER] = WaterTask(WATER, PENDING, True)
    return bucket


def _grid(*buckets: TaskBucket) -> TaskGrid:
    grid = TaskGrid()
    for bucket in buckets:
        grid.put(bucket)
    return grid


def _ops(plan, worker_id: str | None = None) -> list[tuple]:
    routes = plan.worker_routes
    if worker_id is not None:
        routes = [route for route in routes if route.worker_id == worker_id]
    return [(action.hour, action.operation, action.args) for route in routes for action in route.actions_by_hour]


def _played(action: dict) -> list:
    return list(action["farmer"])


class RegionProductionTests(unittest.TestCase):
    def test_a_seedling_on_the_way_adds_work_but_not_walking(self) -> None:
        grid = _grid(_water(0, 2), _crop_plan(0, 1))
        plan = plan_region_routes(grid, [RegionWorker("Farmer", (0, 0))])

        self.assertEqual(
            _ops(plan),
            [
                (1, "SOUTH", ()),
                (2, PLANT, ("MELON",)),
                (3, WATER, ()),
                (4, "SOUTH", ()),
                (5, WATER, ()),
            ],
        )
        self.assertTrue(plan.feasible)
        self.assertEqual(plan.unfinished_visits, ())

    def test_a_crop_chain_is_left_out_when_only_the_sowing_fits(self) -> None:
        grid = _grid(_crop_plan(0, 0))
        plan = plan_region_routes(
            grid,
            [RegionWorker("Farmer", (0, 0))],
            start_hour=23,
            end_hour=23,
        )

        self.assertEqual(_ops(plan), [])
        self.assertTrue(plan.feasible)
        self.assertNotIn(PLANT, [operation for _, operation, _ in _ops(plan)])

    def test_plant_keeps_the_crop_name(self) -> None:
        grid = _grid(_crop_plan(0, 0))
        plan = plan_region_routes(grid, [RegionWorker("Farmer", (0, 0))])
        commands = [
            [action.operation, *action.args]
            for action in plan.worker_routes[0].actions_by_hour
        ]

        self.assertEqual(commands, [[PLANT, "MELON"], [WATER]])

    def test_a_crop_that_still_needs_buying_is_not_started(self) -> None:
        grid = _grid(_water(0, 0), _crop_plan(0, 1, cash=80))
        plan = plan_region_routes(grid, [RegionWorker("Farmer", (0, 0))])

        self.assertEqual([operation for _, operation, _ in _ops(plan)], [WATER])
        self.assertTrue(plan.feasible)
        self.assertEqual(plan.unfinished_visits, ())

    def test_a_shed_sheep_is_picked_up_then_housed_on_the_hour(self) -> None:
        grid = _grid(_animal_plan(4, 2))
        plan = plan_region_routes(
            grid,
            [RegionWorker("Farmer", (4, 3))],
            shed_animals={"SHEEP": 1},
        )

        self.assertEqual(
            _ops(plan),
            [
                (1, "SOUTH", ()),
                (2, "PICKUP", ("SHEEP", 1)),
                (3, "NORTH", ()),
                (4, "NORTH", ()),
                (5, BUILD_PASTURE, ()),
                (6, "PLACE", ("SHEEP",)),
            ],
        )
        hours = [hour for hour, _, _ in _ops(plan)]
        self.assertEqual(hours, list(range(1, 7)))

    def test_a_sheep_already_in_hand_skips_the_shed(self) -> None:
        grid = _grid(_animal_plan(4, 3))
        plan = plan_region_routes(
            grid,
            [RegionWorker("Farmer", (4, 3), carrying_items=(("SHEEP", 1),))],
            shed_animals={"SHEEP": 0},
        )

        self.assertEqual(_ops(plan), [(1, BUILD_PASTURE, ()), (2, "PLACE", ("SHEEP",))])
        self.assertNotIn("PICKUP", [operation for _, operation, _ in _ops(plan)])

    def test_a_carried_sheep_stays_with_the_person_holding_it(self) -> None:
        grid = _grid(_animal_plan(0, 0))
        plan = plan_region_routes(
            grid,
            [
                RegionWorker("Farmer", (4, 4), carrying_items=(("SHEEP", 1),)),
                RegionWorker("Hand1", (0, 0)),
            ],
            shed_animals={"SHEEP": 0},
        )
        owners = {visit.coord: route.worker_id for route in plan.worker_routes for visit in route.visits}

        self.assertEqual(owners[(0, 0)], "Farmer")
        self.assertTrue(plan.feasible)

    def test_one_shed_sheep_starts_only_one_pasture(self) -> None:
        grid = _grid(_animal_plan(4, 3), _animal_plan(4, 2))
        plan = plan_region_routes(
            grid,
            [RegionWorker("Farmer", (4, 4))],
            shed_animals={"SHEEP": 1},
        )
        placed = [args for _, operation, args in _ops(plan) if operation == "PLACE"]

        self.assertEqual(placed, [("SHEEP",)])
        self.assertTrue(plan.feasible)
        self.assertEqual(plan.unfinished_visits, ())

    def test_two_sheep_are_one_pickup(self) -> None:
        grid = _grid(_animal_plan(4, 3), _animal_plan(4, 2))
        plan = plan_region_routes(
            grid,
            [RegionWorker("Farmer", (4, 4))],
            shed_animals={"SHEEP": 2},
        )
        pickups = [(hour, args) for hour, operation, args in _ops(plan) if operation == "PICKUP"]

        self.assertEqual(pickups, [(1, ("SHEEP", 2))])

    def test_a_cow_and_a_sheep_are_separate_pickups(self) -> None:
        grid = _grid(_animal_plan(4, 3, "SHEEP"), _animal_plan(4, 2, "COW"))
        plan = plan_region_routes(
            grid,
            [RegionWorker("Farmer", (4, 4))],
            shed_animals={"SHEEP": 1, "COW": 1},
        )
        pickups = [(hour, args) for hour, operation, args in _ops(plan) if operation == "PICKUP"]

        self.assertEqual(pickups, [(1, ("COW", 1)), (2, ("SHEEP", 1))])

    def test_wheat_and_a_sheep_share_one_shed_stop(self) -> None:
        first = TaskBucket((3, 4), "SHEEP")
        first.tasks["FEED"] = FeedTask("FEED", PENDING, True)
        second = TaskBucket((2, 4), "SHEEP")
        second.tasks["FEED"] = FeedTask("FEED", PENDING, True)
        grid = _grid(first, second, _animal_plan(4, 3))
        plan = plan_region_routes(
            grid,
            [RegionWorker("Farmer", (0, 0))],
            shed_wheat=2,
            shed_animals={"SHEEP": 1},
        )
        steps = plan.worker_routes[0].actions_by_hour
        taken = [(action.hour, action.args, action.coord) for action in steps if action.operation == "PICKUP"]
        field = [action for action in steps if action.operation in {"FEED", BUILD_PASTURE, "PLACE"}]

        self.assertEqual([item[1] for item in taken], [("WHEAT", 2), ("SHEEP", 1)])
        self.assertEqual(taken[0][2], taken[1][2])
        self.assertEqual(taken[1][0], taken[0][0] + 1)
        self.assertLess(taken[-1][0], field[0].hour)
        self.assertEqual(sum(1 for action in steps if action.operation == "PICKUP"), 2)

    def test_a_chain_that_would_pass_hour_23_stays_off_the_route(self) -> None:
        grid = _grid(_water(0, 0), _crop_plan(0, 1))
        plan = plan_region_routes(
            grid,
            [RegionWorker("Farmer", (0, 0))],
            start_hour=21,
            end_hour=23,
        )

        self.assertEqual(_ops(plan), [(21, WATER, ())])
        self.assertTrue(plan.feasible)
        self.assertEqual(plan.unfinished_visits, ())

    def test_the_stop_already_on_the_path_is_taken_before_a_richer_detour(self) -> None:
        grid = _grid(
            _water(0, 2),
            _crop_plan(0, 1, money=10),
            _crop_plan(3, 0, money=500),
        )
        plan = plan_region_routes(
            grid,
            [RegionWorker("Farmer", (0, 0))],
            end_hour=10,
        )
        planted = [action.coord for action in plan.worker_routes[0].actions_by_hour if action.operation == PLANT]

        self.assertEqual(planted, [(0, 1)])
        self.assertTrue(plan.feasible)
        self.assertEqual(plan.unfinished_visits, ())

    def test_skipped_empty_tiles_are_not_unfinished_work(self) -> None:
        waters = [_water(0, 0), _water(2, 0), _water(4, 0), _water(4, 2), _water(4, 4)]
        on_the_path = [_crop_plan(1, 0), _crop_plan(3, 0), _crop_plan(4, 1), _crop_plan(4, 3)]
        off_the_path = [_crop_plan(x, y) for x, y in ((0, 1), (0, 2), (0, 3), (1, 1), (2, 1), (2, 2))]
        grid = _grid(*waters, *on_the_path, *off_the_path)
        plan = plan_region_routes(grid, [RegionWorker("Farmer", (0, 0))], end_hour=17)
        planted = [action.coord for action in plan.worker_routes[0].actions_by_hour if action.operation == PLANT]

        self.assertEqual(planted, [(1, 0), (3, 0)])
        self.assertTrue(plan.feasible)
        self.assertEqual(plan.unfinished_visits, ())

    def test_a_scheduled_plan_is_not_replaced_when_prices_change(self) -> None:
        tiles = _tiles()
        tiles[0][0] = None
        first = _observation(tiles, day=0, hour=1, money=0, farmer=(0, 0))
        first["market"]["inventory"] = {}
        first["market"]["prices"] = {"MELON": 500, "WHEAT": 1}
        first["private"]["seeds"] = {"MELON": 1}
        world = parse_world(first)
        grid = build_task_grid(world, None, first)
        plan = plan_region_routes(grid, _crew(world))
        apply_assignments(grid, world, _assignments(plan))
        self.assertEqual(grid[0][0].production_plan.name, "MELON")
        self.assertEqual(grid[0][0].tasks[PLANT].status, SCHEDULED)

        second = _observation(tiles, day=0, hour=2, money=0, farmer=(0, 0))
        second["market"]["inventory"] = {}
        second["market"]["prices"] = {"WHEAT": 500, "MELON": 1}
        second["private"]["seeds"] = {"MELON": 1, "WHEAT": 1}
        rebuilt = build_task_grid(parse_world(second), grid, second)

        self.assertEqual(rebuilt[0][0].production_plan.name, "MELON")
        self.assertEqual(rebuilt[0][0].tasks[PLANT].subject, "MELON")
        self.assertNotIn("WHEAT", [rebuilt[0][0].tasks[PLANT].subject])

    def test_the_engine_waters_the_seedling_after_it_is_planted(self) -> None:
        environment = make("kaggriculture", configuration={"episodeSteps": 8, "seed": 1}, debug=True)
        environment.reset()
        observation = environment.steps[0][0].observation
        farm = observation.farms[0]
        farm["tiles"] = [["LOCKED"] * 10 for _ in range(10)]
        farm["tiles"][0][0] = None
        farm["farmer"] = [0, 0]
        farm["money"] = 0
        observation.private["seeds"]["MELON"] = 1
        agent = make_region_phase1_agent()
        opponent = {"farmer": ["PASS"], "hands": [], "market": []}
        played: list[list] = []
        for _ in range(3):
            current = environment.steps[-1][0].observation
            action = agent(current)
            played.append(_played(action))
            environment.step([action, opponent])
        tile = environment.steps[-1][0].observation.farms[0]["tiles"][0][0]

        self.assertEqual(played[0], ["PASS"])
        self.assertEqual(played[1], [PLANT, "MELON"])
        self.assertEqual(played[2], [WATER])
        self.assertEqual(tile["crop"], "MELON")
        self.assertTrue(tile["watered_today"])

    def test_the_engine_puts_the_sheep_in_the_pasture_it_just_built(self) -> None:
        environment = make("kaggriculture", configuration={"episodeSteps": 10, "seed": 1}, debug=True)
        environment.reset()
        observation = environment.steps[0][0].observation
        farm = observation.farms[0]
        farm["tiles"] = [["LOCKED"] * 10 for _ in range(10)]
        farm["tiles"][3][4] = None
        farm["farmer"] = [4, 4]
        farm["money"] = 0
        observation.private["shed"]["SHEEP"] = 1
        observation.private["shed"]["WHEAT"] = 7
        agent = make_region_phase1_agent()
        opponent = {"farmer": ["PASS"], "hands": [], "market": []}
        played: list[list] = []
        for _ in range(5):
            current = environment.steps[-1][0].observation
            action = agent(current)
            played.append(_played(action))
            environment.step([action, opponent])
        tile = environment.steps[-1][0].observation.farms[0]["tiles"][3][4]

        self.assertEqual(played[0], ["PASS"])
        self.assertEqual(played[1], ["PICKUP", "SHEEP", 1])
        self.assertEqual(played[2], ["NORTH"])
        self.assertEqual(played[3], [BUILD_PASTURE])
        self.assertEqual(played[4], ["PLACE", "SHEEP"])
        self.assertEqual(tile["animal"], "SHEEP")
        self.assertEqual(tile["kind"], "PASTURE")

    def test_assignments_keep_the_crop_and_the_animal_and_then_complete(self) -> None:
        tiles = _tiles()
        tiles[0][0] = None
        opening = _observation(tiles, day=0, hour=1, money=0, farmer=(0, 0))
        opening["market"]["inventory"] = {}
        opening["market"]["prices"] = {"MELON": 500}
        opening["private"]["seeds"] = {"MELON": 1}
        world = parse_world(opening)
        grid = build_task_grid(world, None, opening)
        self.assertEqual(grid[0][0].tasks[PLANT].status, PENDING)
        plan = plan_region_routes(grid, _crew(world))
        assignments = _assignments(plan)
        apply_assignments(grid, world, assignments)
        plant = next(item for item in assignments if item.kind == PLANT)
        self.assertEqual(plant.subject, "MELON")
        self.assertEqual(grid[0][0].tasks[PLANT].status, SCHEDULED)
        self.assertEqual(grid[0][0].tasks[WATER].subject, "MELON")

        planted = _tiles()
        planted[0][0] = _plant("MELON", dry=1)
        watered_board = _observation(planted, day=0, hour=2, money=0, farmer=(0, 0))
        watered_board["market"]["inventory"] = {}
        watered_board["market"]["prices"] = {"WHEAT": 500}
        after_plant = build_task_grid(parse_world(watered_board), grid, watered_board)
        self.assertEqual(after_plant[0][0].tasks[PLANT].status, COMPLETED)
        self.assertEqual(after_plant[0][0].tasks[WATER].status, SCHEDULED)

        planted[0][0] = _plant("MELON", dry=1, watered=True)
        done_board = _observation(planted, day=0, hour=3, money=0, farmer=(0, 0))
        done_board["market"]["inventory"] = {}
        finished = build_task_grid(parse_world(done_board), after_plant, done_board)
        self.assertEqual(finished[0][0].tasks[WATER].status, COMPLETED)

        pasture = _tiles()
        pasture[1][1] = None
        sheep_board = _observation(pasture, day=0, hour=1, money=0, farmer=(4, 4), shed={"SHEEP": 1, "WHEAT": 7})
        sheep_board["market"]["inventory"] = {}
        sheep_board["market"]["prices"] = {"WOOL": 500, "WHEAT": 0}
        sheep_world = parse_world(sheep_board)
        sheep_grid = build_task_grid(sheep_world, None, sheep_board)
        sheep_plan = plan_region_routes(
            sheep_grid,
            _crew(sheep_world),
            origin=(0, 0),
            shed_animals=_shed_animals(sheep_world),
        )
        sheep_assignments = _assignments(sheep_plan)
        placed = next(item for item in sheep_assignments if item.kind == PLACE_ANIMAL)
        self.assertEqual(placed.subject, "SHEEP")
        apply_assignments(sheep_grid, sheep_world, sheep_assignments)
        self.assertEqual(sheep_grid[1][1].tasks[PLACE_ANIMAL].status, SCHEDULED)

        housed = _tiles()
        housed[1][1] = _animal("SHEEP")
        housed_board = _observation(housed, day=0, hour=4, money=0, farmer=(1, 1))
        housed_board["market"]["inventory"] = {}
        housed_grid = build_task_grid(parse_world(housed_board), sheep_grid, housed_board)
        self.assertEqual(housed_grid[1][1].tasks[PLACE_ANIMAL].status, COMPLETED)


if __name__ == "__main__":
    unittest.main()
