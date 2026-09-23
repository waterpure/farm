"""A 5×5 day is planned one tile at a time, for one to four workers."""

from __future__ import annotations

import unittest

from lab.region_phase1 import _assignments, make_region_phase1_agent
from lab.region_route import (
    RegionWorker,
    TileVisit,
    _Pantry,
    _best_insertion,
    _Leg,
    _end_day_capacity_safe,
    _end_inventory,
    _fit_leg,
    _extract_visits,
    _insertion_extra,
    _path_moves,
    _project_end_shed,
    _reorder,
    _shed_delta,
    _task_order,
    _visit_sort,
    plan_region_routes,
)
from lab.test_region_production import _animal_plan, _crop_plan, _ops
from lab.test_route14_phase1 import _observation, _plant, _tiles
from lab.route14_state import PENDING, SCHEDULED, shed_doors
from lab.task_grid import (
    TaskGridBuilder,
    BUILD_PASTURE,
    CARE,
    COLLECT_FERTILIZER,
    FEED,
    FERTILIZE,
    HARVEST,
    PLACE_ANIMAL,
    WATER,
    CareTask,
    CollectFertilizerTask,
    FeedTask,
    FertilizeTask,
    HarvestTask,
    TaskBucket,
    TaskGrid,
    WaterTask,
)
from lab.test_task_grid import _world


def _harvest(x: int, y: int, tile: str = "WHEAT", units: int = 1) -> TaskBucket:
    bucket = TaskBucket((x, y), tile)
    bucket.tasks[HARVEST] = HarvestTask(HARVEST, PENDING, True, yield_amount=units)
    return bucket


def _feed(x: int, y: int) -> TaskBucket:
    bucket = TaskBucket((x, y), "SHEEP")
    bucket.tasks[FEED] = FeedTask(FEED, PENDING, True)
    return bucket


def _feed_and_harvest(x: int, y: int) -> TaskBucket:
    bucket = _feed(x, y)
    bucket.tasks[HARVEST] = HarvestTask(HARVEST, PENDING, True, yield_amount=1)
    return bucket


def _actions(plan, worker_id: str | None = None):
    routes = plan.worker_routes
    if worker_id is not None:
        routes = [route for route in routes if route.worker_id == worker_id]
    return [action for route in routes for action in route.actions_by_hour]


def _pickups(plan):
    return [action for action in _actions(plan) if action.operation == "PICKUP"]


def _feeds(plan):
    return [action for action in _actions(plan) if action.operation == "FEED"]


def _grid(*buckets: TaskBucket) -> TaskGrid:
    grid = TaskGrid()
    for bucket in buckets:
        grid.put(bucket)
    return grid


def _owners(plan) -> dict[tuple[int, int], str]:
    found: dict[tuple[int, int], str] = {}
    for route in plan.worker_routes:
        for visit in route.visits:
            if visit.coord in found:
                raise AssertionError(f"{visit.coord} has two owners")
            found[visit.coord] = route.worker_id
    return found


class RegionRouteTests(unittest.TestCase):
    def test_a_line_is_walked_one_step_at_a_time(self) -> None:
        grid = _grid(_harvest(0, 0), _harvest(1, 0), _harvest(2, 0))
        plan = plan_region_routes(grid, [RegionWorker("Farmer", (0, 0))])

        self.assertTrue(plan.feasible)
        self.assertEqual(plan.total_move_count, 8)
        self.assertEqual(plan.unfinished_visits, ())
        route = plan.worker_routes[0]
        self.assertEqual(
            [(action.hour, action.operation, action.coord, action.args) for action in route.actions_by_hour],
            [
                (1, "HARVEST", (0, 0), ()),
                (2, "EAST", None, ()),
                (3, "HARVEST", (1, 0), ()),
                (4, "EAST", None, ()),
                (5, "HARVEST", (2, 0), ()),
                (6, "EAST", None, ()),
                (7, "EAST", None, ()),
                (8, "SOUTH", None, ()),
                (9, "SOUTH", None, ()),
                (10, "SOUTH", None, ()),
                (11, "SOUTH", None, ()),
                (12, "PLACE", (4, 4), ("WHEAT", 3)),
            ],
        )
        self.assertEqual(route.finish_hour, 12)

    def test_ripe_wheat_below_the_cap_is_harvested_while_the_water_stays_optional(self) -> None:
        world = _world("WHEAT", day=2, units=4, dry=1)
        grid = TaskGridBuilder().build(world)
        water = grid[2][3].tasks[WATER]
        self.assertEqual(water.yield_gain, 1)
        self.assertFalse(water.mandatory)
        self.assertIn(HARVEST, grid[2][3].tasks)

        plan = plan_region_routes(grid, [RegionWorker("Farmer", (2, 3))])

        self.assertEqual(plan.worker_routes[0].visits[0].tasks, (WATER, HARVEST))
        self.assertEqual(plan.worker_routes[0].visits[0].action_count, 2)
        operations = [action.operation for action in plan.worker_routes[0].actions_by_hour]
        self.assertEqual(operations[:2], [WATER, HARVEST])
        self.assertEqual(operations[-1], "PLACE")
        self.assertEqual(grid[2][3].tasks[WATER].yield_gain, 1)
        self.assertFalse(grid[2][3].tasks[WATER].mandatory)

    def test_wheat_at_max_yield_is_harvested_without_a_water_stop(self) -> None:
        world = _world("WHEAT", day=4, units=6, dry=1, fertilized_until=4)
        grid = TaskGridBuilder().build(world)
        self.assertNotIn(WATER, grid[2][3].tasks)
        self.assertEqual(grid[2][3].tasks[HARVEST].yield_amount, 6)
        plan = plan_region_routes(grid, [RegionWorker("Farmer", (2, 3))])
        operations = [action.operation for action in plan.worker_routes[0].actions_by_hour]
        self.assertEqual(operations[0], HARVEST)
        self.assertNotIn(WATER, operations)
        self.assertEqual(plan.worker_routes[0].actions_by_hour[-1].args, ("WHEAT", 6))

    def test_one_shot_visit_waters_before_it_harvests(self) -> None:
        self.assertEqual(_task_order("WHEAT", [HARVEST, WATER]), (WATER, HARVEST))
        self.assertEqual(_task_order("CARROT", [HARVEST, WATER]), (WATER, HARVEST))
        self.assertEqual(_task_order("MELON", [HARVEST, WATER]), (WATER, HARVEST))
        bucket = TaskBucket((2, 3), "WHEAT")
        bucket.tasks[WATER] = WaterTask(WATER, PENDING, True, needed=True, turns_until_weed=0, yield_gain=1)
        bucket.tasks[HARVEST] = HarvestTask(HARVEST, PENDING, True, yield_amount=5)
        plan = plan_region_routes(_grid(bucket), [RegionWorker("Farmer", (2, 3))])

        self.assertEqual(plan.worker_routes[0].visits[0].tasks, (WATER, HARVEST))
        self.assertEqual(plan.worker_routes[0].visits[0].action_count, 2)
        self.assertEqual(plan.worker_routes[0].visits[0].harvest_product, "WHEAT")
        self.assertEqual(plan.worker_routes[0].visits[0].harvest_units, 5)
        operations = [action.operation for action in plan.worker_routes[0].actions_by_hour]
        self.assertEqual(operations[:2], [WATER, HARVEST])
        self.assertEqual(plan.worker_routes[0].actions_by_hour[-1].args, ("WHEAT", 5))

    def test_one_tile_has_one_owner(self) -> None:
        bucket = TaskBucket((2, 3), "WHEAT")
        bucket.tasks[WATER] = WaterTask(WATER, PENDING, True, needed=True, turns_until_weed=0, yield_gain=1)
        bucket.tasks[HARVEST] = HarvestTask(HARVEST, PENDING, True, yield_amount=2)
        workers = [RegionWorker("Farmer", (2, 3)), RegionWorker("Hand1", (0, 0))]
        plan = plan_region_routes(_grid(bucket), workers)

        self.assertEqual(_owners(plan), {(2, 3): "Farmer"})
        self.assertEqual(plan.worker_routes[1].visits, ())
        self.assertEqual(plan.worker_routes[1].move_count, 0)
        self.assertEqual(plan.total_move_count, 3)

    def test_an_ongoing_crop_keeps_water_before_harvest_in_the_same_visit(self) -> None:
        bucket = TaskBucket((1, 1), "TOMATO")
        bucket.tasks[WATER] = WaterTask(WATER, PENDING, True, needed=True, turns_until_weed=0, yield_gain=0)
        bucket.tasks[HARVEST] = HarvestTask(HARVEST, PENDING, True, yield_amount=1)
        bucket.tasks[FERTILIZE] = FertilizeTask(FERTILIZE, PENDING, False)
        plan = plan_region_routes(_grid(bucket), [RegionWorker("Farmer", (1, 1))])

        visit = plan.worker_routes[0].visits[0]
        self.assertEqual(visit.tasks, (WATER, HARVEST))
        self.assertEqual(visit.action_count, 2)
        self.assertEqual(visit.harvest_product, "TOMATO")
        operations = [action.operation for action in plan.worker_routes[0].actions_by_hour]
        self.assertEqual(operations[:2], [WATER, HARVEST])
        self.assertEqual(plan.worker_routes[0].actions_by_hour[-1].args, ("TOMATO", 1))

    def test_feed_and_harvest_on_one_animal_stay_together(self) -> None:
        bucket = TaskBucket((3, 4), "SHEEP")
        bucket.tasks[FEED] = FeedTask(FEED, PENDING, True)
        bucket.tasks[HARVEST] = HarvestTask(HARVEST, PENDING, True, yield_amount=1)
        bucket.tasks[CARE] = CareTask(CARE, PENDING, False)
        bucket.tasks[COLLECT_FERTILIZER] = CollectFertilizerTask(COLLECT_FERTILIZER, PENDING, False, fertilizer_ready=True)
        plan = plan_region_routes(_grid(bucket), [RegionWorker("Farmer", (3, 4), carrying_wheat=1)])

        self.assertEqual(plan.worker_routes[0].visits[0].tasks, (FEED, HARVEST))
        self.assertEqual(plan.worker_routes[0].visits[0].harvest_product, "WOOL")
        operations = [action.operation for action in plan.worker_routes[0].actions_by_hour]
        self.assertEqual(operations[:2], [FEED, HARVEST])
        places = [action for action in plan.worker_routes[0].actions_by_hour if action.operation == "PLACE"]
        self.assertEqual([(action.args) for action in places], [("WOOL", 1)])

    def test_optional_water_and_already_assigned_work_are_left_out(self) -> None:
        optional = TaskBucket((0, 1), "WHEAT")
        optional.tasks[WATER] = WaterTask(WATER, PENDING, False, needed=True, turns_until_weed=20, yield_gain=1)
        optional.tasks[HARVEST] = HarvestTask(HARVEST, PENDING, True, yield_amount=1)
        assigned = TaskBucket((0, 2), "WHEAT")
        assigned.tasks[WATER] = WaterTask(
            WATER, SCHEDULED, True, assigned_worker="Hand2", planned_hour=6, needed=True
        )
        outside = _harvest(5, 0)
        plan = plan_region_routes(
            _grid(optional, assigned, outside, _harvest(0, 0)),
            [RegionWorker("Farmer", (0, 0))],
        )

        self.assertEqual(set(_owners(plan)), {(0, 0), (0, 1)})
        tasks = {visit.coord: visit.tasks for visit in plan.worker_routes[0].visits}
        self.assertEqual(tasks[(0, 0)], (HARVEST,))
        self.assertEqual(tasks[(0, 1)], (WATER, HARVEST))

    def test_insertion_follows_the_route_not_each_workers_start(self) -> None:
        grid = _grid(_harvest(0, 0), _harvest(4, 0), _harvest(4, 1))
        workers = [RegionWorker("A", (0, 0)), RegionWorker("B", (0, 1))]
        plan = plan_region_routes(grid, workers)

        self.assertTrue(plan.feasible)
        self.assertEqual(plan.total_move_count, 8)
        self.assertEqual(_owners(plan)[(4, 0)], _owners(plan)[(4, 1)])
        self.assertEqual(plan.worker_routes[1].move_count, 0)

    def test_four_workers_keep_the_corner_they_are_standing_on(self) -> None:
        tiles = [_harvest(0, 0), _harvest(0, 4), _harvest(4, 0), _harvest(4, 4), _harvest(2, 2)]
        workers = [
            RegionWorker("A", (0, 0)),
            RegionWorker("B", (0, 4)),
            RegionWorker("C", (4, 0)),
            RegionWorker("D", (4, 4)),
        ]
        plan = plan_region_routes(_grid(*tiles), workers)
        alone = plan_region_routes(_grid(*tiles), [workers[0]])

        self.assertTrue(plan.feasible)
        self.assertEqual(plan.total_move_count, 16)
        self.assertTrue(alone.feasible)
        self.assertNotIn("PLACE", [action.operation for action in alone.worker_routes[0].actions_by_hour])
        owners = _owners(plan)
        self.assertEqual(owners[(0, 0)], "A")
        self.assertEqual(owners[(0, 4)], "A")
        self.assertEqual(owners[(4, 4)], "D")
        self.assertEqual(plan.worker_routes[1].visits, ())
        self.assertEqual(len(owners), 5)

    def test_two_workers_finish_the_corners_in_fewer_steps_than_one(self) -> None:
        tiles = [_harvest(0, 0), _harvest(0, 4), _harvest(4, 0), _harvest(4, 4), _harvest(2, 2)]
        one = plan_region_routes(_grid(*tiles), [RegionWorker("A", (0, 0))])
        two = plan_region_routes(
            _grid(*tiles),
            [RegionWorker("A", (0, 0)), RegionWorker("B", (4, 4))],
        )

        self.assertTrue(one.feasible)
        self.assertTrue(two.feasible)
        self.assertLessEqual(two.total_move_count, one.total_move_count)
        owners = _owners(two)
        self.assertEqual(owners[(0, 0)], "A")
        self.assertEqual(owners[(4, 4)], "B")

    def test_past_hour_23_leaves_the_day_unfinished(self) -> None:
        tiles = [_harvest(x, y) for y in range(3) for x in range(5)]
        tiles.append(_harvest(0, 3))
        tiles.append(_harvest(1, 3))
        tiles.append(_harvest(2, 3))
        plan = plan_region_routes(_grid(*tiles), [RegionWorker("Farmer", (0, 0))])

        self.assertEqual(len(tiles), 18)
        self.assertFalse(plan.feasible)
        self.assertGreater(len(plan.unfinished_visits), 0)
        done = sum(len(route.visits) for route in plan.worker_routes)
        self.assertEqual(done + len(plan.unfinished_visits), len(tiles))
        for route in plan.worker_routes:
            for action in route.actions_by_hour:
                self.assertLessEqual(action.hour, 23)
                self.assertGreaterEqual(action.hour, 1)

    def test_a_short_day_cannot_pretend_a_walk_is_one_action(self) -> None:
        plan = plan_region_routes(
            _grid(_harvest(0, 0), _harvest(1, 0)),
            [RegionWorker("Farmer", (0, 0))],
            end_hour=1,
        )

        self.assertFalse(plan.feasible)
        self.assertEqual([visit.coord for visit in plan.unfinished_visits], [(1, 0)])
        self.assertEqual(
            [(action.hour, action.operation) for action in plan.worker_routes[0].actions_by_hour],
            [(1, "HARVEST")],
        )

    def test_more_than_ten_tiles_still_returns_a_route(self) -> None:
        tiles = [_harvest(index % 5, index // 5) for index in range(11)]
        plan = plan_region_routes(_grid(*tiles), [RegionWorker("Farmer", (0, 0))])

        carried = plan.worker_routes[0].visits
        self.assertEqual(len(carried), 11)
        self.assertEqual(plan.unfinished_visits, ())
        self.assertTrue(plan.feasible)
        self.assertLessEqual(plan.finish_hour or 0, 23)
        self.assertNotIn(
            "PLACE",
            [action.operation for action in plan.worker_routes[0].actions_by_hour],
        )

    def test_a_full_five_by_five_with_four_workers_stays_feasible(self) -> None:
        tiles = [_harvest(x, y) for y in range(5) for x in range(5)]
        workers = [
            RegionWorker("A", (0, 0)),
            RegionWorker("B", (0, 4)),
            RegionWorker("C", (4, 0)),
            RegionWorker("D", (4, 4)),
        ]
        plan = plan_region_routes(_grid(*tiles), workers)
        again = plan_region_routes(_grid(*tiles), workers)

        self.assertEqual(plan, again)
        self.assertTrue(plan.feasible)
        self.assertEqual(len(_owners(plan)), 25)
        self.assertLessEqual(plan.finish_hour, 23)
        for route in plan.worker_routes:
            self.assertLessEqual(route.finish_hour or 0, 23)
            self.assertNotIn("GO_TO", [action.operation for action in route.actions_by_hour])

    def test_another_five_by_five_starts_at_the_given_origin(self) -> None:
        plan = plan_region_routes(
            _grid(_harvest(5, 0), _harvest(0, 0)),
            [RegionWorker("Farmer", (5, 0))],
            origin=(5, 0),
        )

        self.assertEqual(set(_owners(plan)), {(5, 0)})
        self.assertEqual(plan.total_move_count, 4)
        self.assertEqual(plan.worker_routes[0].actions_by_hour[-1].args, ("WHEAT", 1))

    def test_a_dropped_tile_is_picked_up_after_another_plot_moves(self) -> None:
        coords = [(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (0, 2)]
        workers = [RegionWorker("A", (0, 0)), RegionWorker("B", (4, 0))]
        grid = _grid(*[_harvest(x, y) for x, y in coords])
        routes: list[list] = [[], []]
        dropped: list[tuple[int, int]] = []
        for visit in sorted(_extract_visits(grid, 5, (0, 0)), key=_visit_sort):
            placed = _best_insertion(workers, routes, visit, 1, 6)
            if placed is None:
                dropped.append(visit.coord)
            else:
                routes[placed[0]] = placed[1]

        self.assertIn((0, 1), dropped)
        self.assertIn((0, 2), dropped)
        plan = plan_region_routes(grid, workers, end_hour=6)
        self.assertTrue(plan.feasible)
        self.assertEqual(plan.unfinished_visits, ())
        self.assertEqual(set(_owners(plan)), set(coords))

    def test_insertion_score_follows_the_reordered_route(self) -> None:
        def standing(coord: tuple[int, int]) -> TileVisit:
            return TileVisit(coord, (HARVEST,), 1, "WHEAT")

        workers = [RegionWorker("A", (0, 0)), RegionWorker("B", (0, 4))]
        routes = [
            [standing((2, 0)), standing((3, 0))],
            [standing((0, 0)), standing((1, 0))],
        ]
        added = standing((2, 3))
        splice_winner: tuple[tuple[int, int, int], int] | None = None
        for worker_index, worker in enumerate(workers):
            coords = [visit.coord for visit in routes[worker_index]]
            for index in range(len(coords) + 1):
                extra = _insertion_extra(worker.coord, coords, index, added.coord)
                key = (extra + added.action_count, worker_index, index)
                if splice_winner is None or key < splice_winner[0]:
                    splice_winner = (key, worker_index)
        assert splice_winner is not None
        true_extra: list[int] = []
        for worker, route in zip(workers, routes):
            old_order = _reorder(worker.coord, route)
            new_order = _reorder(worker.coord, route + [added])
            old_cost = _path_moves(worker.coord, [visit.coord for visit in old_order])
            new_cost = _path_moves(worker.coord, [visit.coord for visit in new_order])
            true_extra.append(new_cost - old_cost)

        placed = _best_insertion(workers, routes, added, 1, 23)
        assert placed is not None
        self.assertEqual(splice_winner[1], 0)
        self.assertLess(true_extra[1], true_extra[0])
        self.assertEqual(placed[0], 1)
        self.assertEqual([visit.coord for visit in placed[1]], [visit.coord for visit in _reorder(workers[1].coord, routes[1] + [added])])

    def test_worker_count_outside_one_to_four_is_rejected(self) -> None:
        grid = _grid(_harvest(0, 0))
        with self.assertRaises(ValueError):
            plan_region_routes(grid, [])
        with self.assertRaises(ValueError):
            plan_region_routes(
                grid,
                [RegionWorker(str(index), (0, 0)) for index in range(5)],
            )

    def test_one_feed_picks_up_wheat_before_walking_to_the_animal(self) -> None:
        plan = plan_region_routes(
            _grid(_feed(2, 4)),
            [RegionWorker("Farmer", (4, 4))],
            shed_wheat=1,
            shed_coords=shed_doors(10),
        )

        self.assertTrue(plan.feasible)
        pickups = _pickups(plan)
        self.assertEqual(len(pickups), 1)
        self.assertEqual(pickups[0].args, ("WHEAT", 1))
        self.assertEqual(pickups[0].coord, (4, 4))
        self.assertLess(pickups[0].hour, _feeds(plan)[0].hour)

    def test_three_feeds_are_one_pickup_of_three(self) -> None:
        plan = plan_region_routes(
            _grid(_feed(2, 4), _feed(2, 3), _feed(2, 2)),
            [RegionWorker("Farmer", (4, 4))],
            shed_wheat=3,
            shed_coords=shed_doors(10),
        )

        pickups = _pickups(plan)
        self.assertEqual(len(pickups), 1)
        self.assertEqual(pickups[0].args, ("WHEAT", 3))
        self.assertEqual(len(_feeds(plan)), 3)
        self.assertTrue(plan.feasible)

    def test_wheat_already_in_hand_skips_the_shed(self) -> None:
        plan = plan_region_routes(
            _grid(_feed(2, 4), _feed(2, 3)),
            [RegionWorker("Farmer", (4, 4), carrying_wheat=2)],
            shed_wheat=0,
            shed_coords=shed_doors(10),
        )

        self.assertEqual(_pickups(plan), [])
        self.assertEqual(len(_feeds(plan)), 2)
        self.assertTrue(plan.feasible)
        self.assertNotIn("PICKUP", [action.operation for action in _actions(plan)])

    def test_a_partial_armful_picks_up_only_the_shortage(self) -> None:
        plan = plan_region_routes(
            _grid(_feed(2, 4), _feed(2, 3), _feed(2, 2)),
            [RegionWorker("Farmer", (4, 4), carrying_wheat=1)],
            shed_wheat=2,
            shed_coords=shed_doors(10),
        )

        pickups = _pickups(plan)
        self.assertEqual(len(pickups), 1)
        self.assertEqual(pickups[0].args, ("WHEAT", 2))
        self.assertEqual(len(_feeds(plan)), 3)

    def test_two_feeds_from_the_shed_door_take_six_hours(self) -> None:
        plan = plan_region_routes(
            _grid(_feed(2, 4), _feed(2, 3)),
            [RegionWorker("Hand1", (4, 4))],
            shed_wheat=2,
            shed_coords=shed_doors(10),
        )
        route = plan.worker_routes[0]

        self.assertEqual(
            [(action.hour, action.operation, action.coord, action.args) for action in route.actions_by_hour],
            [
                (1, "PICKUP", (4, 4), ("WHEAT", 2)),
                (2, "WEST", None, ()),
                (3, "WEST", None, ()),
                (4, "FEED", (2, 4), ()),
                (5, "NORTH", None, ()),
                (6, "FEED", (2, 3), ()),
            ],
        )
        self.assertEqual(route.move_count, 3)
        self.assertNotIn("PLACE", [action.operation for action in route.actions_by_hour])

    def test_counting_the_pickup_pushes_a_full_day_past_hour_23(self) -> None:
        buckets = [_feed(x, 0) for x in range(5)] + [_feed(x, 4) for x in range(5)]
        grid = _grid(*buckets)
        short = plan_region_routes(
            grid,
            [RegionWorker("Farmer", (4, 3))],
            shed_wheat=10,
            shed_coords=shed_doors(10),
        )
        loaded = plan_region_routes(
            grid,
            [RegionWorker("Farmer", (4, 3), carrying_wheat=10)],
            shed_wheat=0,
            shed_coords=shed_doors(10),
        )

        self.assertFalse(short.feasible)
        self.assertTrue(short.unfinished_visits)
        self.assertTrue(loaded.feasible)
        self.assertEqual(loaded.finish_hour, 23)
        self.assertEqual(_pickups(loaded), [])

    def test_shared_shed_wheat_is_not_promised_twice(self) -> None:
        animals = _grid(_feed(0, 0), _feed(1, 0), _feed(2, 0))
        workers = [RegionWorker("A", (4, 4)), RegionWorker("B", (5, 4))]
        doors = shed_doors(10)
        enough = plan_region_routes(animals, workers, shed_wheat=3, shed_coords=doors)
        short = plan_region_routes(animals, workers, shed_wheat=2, shed_coords=doors)

        self.assertTrue(enough.feasible)
        self.assertEqual(len(_feeds(enough)), 3)
        self.assertEqual(sum(action.args[1] for action in _pickups(enough)), 3)
        self.assertFalse(short.feasible)
        self.assertLess(len(_feeds(short)), 3)
        self.assertLessEqual(sum(action.args[1] for action in _pickups(short)), 2)
        self.assertEqual(sum(action.args[1] for action in _pickups(short)), len(_feeds(short)))

    def test_the_worker_who_already_holds_wheat_gets_the_animal(self) -> None:
        plan = plan_region_routes(
            _grid(_feed(0, 0)),
            [
                RegionWorker("A", (0, 1)),
                RegionWorker("B", (3, 0), carrying_wheat=1),
            ],
            shed_wheat=5,
            shed_coords=shed_doors(10),
        )

        self.assertEqual(_owners(plan)[(0, 0)], "B")
        self.assertEqual(_pickups(plan), [])
        self.assertTrue(plan.feasible)

    def test_same_tile_optional_water_comes_before_harvest(self) -> None:
        bucket = TaskBucket((4, 4), "WHEAT")
        bucket.tasks[HARVEST] = HarvestTask(HARVEST, PENDING, True, yield_amount=4)
        bucket.tasks[WATER] = WaterTask(WATER, PENDING, False, needed=True, yield_gain=1)
        plan = plan_region_routes(_grid(bucket), [RegionWorker("Farmer", (4, 4))])
        visit = plan.worker_routes[0].visits[0]

        self.assertEqual(visit.tasks, (WATER, HARVEST))
        self.assertEqual(visit.action_count, len(visit.tasks))
        self.assertEqual(
            [action.operation for action in plan.worker_routes[0].actions_by_hour[:2]],
            [WATER, HARVEST],
        )

    def test_optional_water_stays_when_only_the_shed_trip_does_not_fit(self) -> None:
        bucket = TaskBucket((4, 4), "WHEAT")
        bucket.tasks[HARVEST] = HarvestTask(HARVEST, PENDING, True, yield_amount=1)
        bucket.tasks[WATER] = WaterTask(WATER, PENDING, False, needed=True, yield_gain=1)
        plan = plan_region_routes(
            _grid(bucket),
            [RegionWorker("Farmer", (4, 4))],
            start_hour=22,
            end_hour=23,
        )

        self.assertTrue(plan.feasible)
        self.assertEqual(plan.unfinished_visits, ())
        self.assertEqual(plan.worker_routes[0].visits[0].tasks, (WATER, HARVEST))
        self.assertEqual(
            [action.operation for action in plan.worker_routes[0].actions_by_hour],
            ["WATER", "HARVEST"],
        )
        self.assertEqual(plan.worker_routes[0].actions_by_hour[-1].hour, 23)
        self.assertNotIn("PLACE", [action.operation for action in plan.worker_routes[0].actions_by_hour])

    def test_care_follows_the_feed_on_the_same_animal(self) -> None:
        bucket = TaskBucket((4, 4), "SHEEP")
        bucket.tasks[FEED] = FeedTask(FEED, PENDING, True)
        bucket.tasks[CARE] = CareTask(CARE, PENDING, False, bonus_gain=1)
        plan = plan_region_routes(
            _grid(bucket),
            [RegionWorker("Farmer", (4, 4), carrying_wheat=1)],
        )
        visit = plan.worker_routes[0].visits[0]
        operations = [action.operation for action in plan.worker_routes[0].actions_by_hour]

        self.assertEqual(visit.tasks, (FEED, CARE))
        self.assertEqual(visit.action_count, 2)
        self.assertEqual(operations, [FEED, CARE])

    def test_feed_care_and_harvest_take_three_hours_for_one_worker(self) -> None:
        bucket = TaskBucket((4, 4), "SHEEP")
        bucket.tasks[FEED] = FeedTask(FEED, PENDING, True)
        bucket.tasks[CARE] = CareTask(CARE, PENDING, False, bonus_gain=1)
        bucket.tasks[HARVEST] = HarvestTask(HARVEST, PENDING, True, yield_amount=1)
        plan = plan_region_routes(
            _grid(bucket),
            [RegionWorker("Farmer", (4, 4), carrying_wheat=1), RegionWorker("Hand1", (0, 0))],
        )
        visit = plan.worker_routes[0].visits[0]

        self.assertEqual(_owners(plan), {(4, 4): "Farmer"})
        self.assertEqual(visit.tasks, (FEED, CARE, HARVEST))
        self.assertEqual(visit.action_count, 3)
        self.assertEqual(plan.worker_routes[1].visits, ())

    def test_a_care_only_tile_is_not_a_detour(self) -> None:
        care = TaskBucket((1, 1), "SHEEP")
        care.tasks[CARE] = CareTask(CARE, PENDING, False, bonus_gain=1)
        plan = plan_region_routes(
            _grid(care, _harvest(0, 0)),
            [RegionWorker("Farmer", (0, 0))],
        )

        self.assertEqual(set(_owners(plan)), {(0, 0)})
        self.assertNotIn(CARE, [action.operation for action in plan.worker_routes[0].actions_by_hour])

    def test_care_moves_the_unload_one_hour_later(self) -> None:
        bare = TaskBucket((4, 3), "SHEEP")
        bare.tasks[HARVEST] = HarvestTask(HARVEST, PENDING, True, yield_amount=1)
        cared = TaskBucket((4, 3), "SHEEP")
        cared.tasks[HARVEST] = HarvestTask(HARVEST, PENDING, True, yield_amount=1)
        cared.tasks[CARE] = CareTask(CARE, PENDING, False, bonus_gain=1)
        worker = [RegionWorker("Farmer", (4, 4))]
        before = plan_region_routes(_grid(bare), worker)
        after = plan_region_routes(_grid(cared), worker)

        def hours(plan, operation: str) -> list[int]:
            return [action.hour for action in plan.worker_routes[0].actions_by_hour if action.operation == operation]

        self.assertEqual(hours(after, "CARE"), [hours(before, "HARVEST")[0]])
        self.assertEqual(hours(after, "HARVEST"), [hours(before, "HARVEST")[0] + 1])
        self.assertEqual(hours(after, "PLACE"), [hours(before, "PLACE")[0] + 1])

    def test_the_larger_same_tile_gain_is_kept_when_only_one_hour_is_left(self) -> None:
        wheat = TaskBucket((4, 4), "WHEAT")
        wheat.tasks[HARVEST] = HarvestTask(HARVEST, PENDING, True, yield_amount=1)
        wheat.tasks[WATER] = WaterTask(WATER, PENDING, False, needed=True, yield_gain=2)
        sheep = TaskBucket((3, 4), "SHEEP")
        sheep.tasks[FEED] = FeedTask(FEED, PENDING, True)
        sheep.tasks[CARE] = CareTask(CARE, PENDING, False, bonus_gain=1)
        workers = [RegionWorker("Farmer", (4, 4), carrying_wheat=1)]
        plan = plan_region_routes(_grid(wheat, sheep), workers, end_hour=4)
        tasks = {visit.coord: visit.tasks for visit in plan.worker_routes[0].visits}

        self.assertTrue(plan.feasible)
        self.assertEqual(plan.unfinished_visits, ())
        self.assertEqual(tasks[(4, 4)], (WATER, HARVEST))
        self.assertEqual(tasks[(3, 4)], (FEED,))
        self.assertLessEqual(plan.finish_hour or 0, 4)
        self.assertNotIn("PLACE", [action.operation for action in plan.worker_routes[0].actions_by_hour])

    def test_optional_work_does_not_increase_unfinished_mandatory_visits(self) -> None:
        wheat = TaskBucket((4, 4), "WHEAT")
        wheat.tasks[HARVEST] = HarvestTask(HARVEST, PENDING, True, yield_amount=1)
        wheat.tasks[WATER] = WaterTask(WATER, PENDING, False, needed=True, yield_gain=2)
        sheep = TaskBucket((3, 4), "SHEEP")
        sheep.tasks[FEED] = FeedTask(FEED, PENDING, True)
        sheep.tasks[CARE] = CareTask(CARE, PENDING, False, bonus_gain=1)
        workers = [RegionWorker("Farmer", (4, 4), carrying_wheat=1)]
        grid = _grid(wheat, sheep)
        plain = plan_region_routes(grid, workers, end_hour=4, include_optional=False)
        full = plan_region_routes(grid, workers, end_hour=4, include_optional=True)

        self.assertEqual(
            sorted(visit.coord for visit in full.unfinished_visits),
            sorted(visit.coord for visit in plain.unfinished_visits),
        )
        self.assertLessEqual(len(full.unfinished_visits), len(plain.unfinished_visits))

    def test_three_harvests_come_home_once(self) -> None:
        plan = plan_region_routes(
            _grid(_harvest(1, 1), _harvest(1, 2), _harvest(2, 2)),
            [RegionWorker("Farmer", (4, 4))],
        )
        actions = plan.worker_routes[0].actions_by_hour
        harvests = [index for index, action in enumerate(actions) if action.operation == "HARVEST"]
        places = [action for action in actions if action.operation == "PLACE"]

        self.assertTrue(plan.feasible)
        self.assertEqual(len(harvests), 3)
        self.assertEqual(places[-1].args, ("WHEAT", 3))
        self.assertEqual(len(places), 1)
        self.assertGreater(actions.index(places[0]), harvests[-1])
        self.assertNotIn("DROP", [action.operation for action in actions])

    def test_the_same_product_is_unloaded_once(self) -> None:
        plan = plan_region_routes(
            _grid(_harvest(1, 1, "WHEAT", 3), _harvest(1, 2, "WHEAT", 2), _harvest(2, 2, "COW", 4)),
            [RegionWorker("Farmer", (4, 4))],
        )
        actions = plan.worker_routes[0].actions_by_hour
        places = [action.args for action in actions if action.operation == "PLACE"]
        last_harvest = max(index for index, action in enumerate(actions) if action.operation == "HARVEST")

        self.assertTrue(plan.feasible)
        self.assertCountEqual(places, [("WHEAT", 5), ("MILK", 4)])
        self.assertGreater(min(index for index, action in enumerate(actions) if action.operation == "PLACE"), last_harvest)

    def test_animals_unload_wool_milk_and_eggs(self) -> None:
        plan = plan_region_routes(
            _grid(_harvest(1, 1, "SHEEP", 1), _harvest(1, 2, "COW", 4), _harvest(2, 2, "GOOSE", 2)),
            [RegionWorker("Farmer", (4, 4))],
        )
        places = [action.args for action in plan.worker_routes[0].actions_by_hour if action.operation == "PLACE"]

        self.assertCountEqual(places, [("WOOL", 1), ("MILK", 4), ("EGG", 2)])

    def test_a_harvest_that_cannot_be_unloaded_today_stays_in_the_field(self) -> None:
        late = plan_region_routes(
            _grid(_harvest(2, 4)),
            [RegionWorker("Farmer", (2, 4))],
            start_hour=21,
            end_hour=23,
        )
        home = plan_region_routes(
            _grid(_harvest(2, 4)),
            [RegionWorker("Farmer", (2, 4))],
            end_hour=4,
        )

        self.assertTrue(late.feasible)
        self.assertEqual(late.unfinished_visits, ())
        self.assertEqual(
            [(action.hour, action.operation) for action in late.worker_routes[0].actions_by_hour],
            [(21, "HARVEST")],
        )
        self.assertEqual(
            [(action.hour, action.operation, action.args) for action in home.worker_routes[0].actions_by_hour],
            [
                (1, "HARVEST", ()),
                (2, "EAST", ()),
                (3, "EAST", ()),
                (4, "PLACE", ("WHEAT", 1)),
            ],
        )

    def test_the_person_already_hauling_keeps_the_next_field(self) -> None:
        plan = plan_region_routes(
            _grid(_harvest(0, 0), _harvest(0, 2)),
            [RegionWorker("A", (1, 2)), RegionWorker("B", (0, 0))],
        )

        self.assertEqual(_owners(plan)[(0, 0)], "B")
        self.assertEqual(_owners(plan)[(0, 2)], "B")
        self.assertEqual(plan.worker_routes[0].visits, ())

    def test_feed_wheat_is_not_unloaded_with_the_harvest(self) -> None:
        plan = plan_region_routes(
            _grid(_feed_and_harvest(2, 4)),
            [RegionWorker("Farmer", (4, 4), carrying_wheat=3)],
            shed_wheat=0,
            shed_coords=shed_doors(10),
        )
        places = [action.args for action in plan.worker_routes[0].actions_by_hour if action.operation == "PLACE"]

        self.assertEqual(places, [("WOOL", 1)])
        self.assertEqual(_pickups(plan), [])


class HarvestMayStayInTheFieldTests(unittest.TestCase):
    def test_a_harvest_still_returns_when_the_shed_trip_fits(self) -> None:
        plan = plan_region_routes(
            _grid(_harvest(4, 3)),
            [RegionWorker("Farmer", (4, 3))],
            start_hour=18,
            end_hour=23,
        )

        self.assertTrue(plan.feasible)
        self.assertEqual(
            [(action.hour, action.operation, action.args) for action in plan.worker_routes[0].actions_by_hour],
            [
                (18, "HARVEST", ()),
                (19, "SOUTH", ()),
                (20, "PLACE", ("WHEAT", 1)),
            ],
        )

    def test_hour_23_harvests_in_place_without_an_unload(self) -> None:
        plan = plan_region_routes(
            _grid(_harvest(2, 4, "MELON", 6)),
            [RegionWorker("Farmer", (2, 4))],
            start_hour=23,
            end_hour=23,
            shed_total=80,
        )
        actions = plan.worker_routes[0].actions_by_hour

        self.assertTrue(plan.feasible)
        self.assertEqual([(action.hour, action.operation) for action in actions], [(23, "HARVEST")])
        self.assertNotIn("PLACE", [action.operation for action in actions])

    def test_two_late_harvests_stay_when_only_the_return_does_not_fit(self) -> None:
        plan = plan_region_routes(
            _grid(_harvest(0, 0, "MELON", 1), _harvest(0, 1, "WHEAT", 1)),
            [RegionWorker("Farmer", (0, 0))],
            start_hour=21,
            end_hour=23,
        )
        actions = plan.worker_routes[0].actions_by_hour

        self.assertTrue(plan.feasible)
        self.assertEqual(
            [(action.hour, action.operation) for action in actions],
            [(21, "HARVEST"), (22, "SOUTH"), (23, "HARVEST")],
        )
        self.assertNotIn("PLACE", [action.operation for action in actions])

    def test_carried_goods_under_the_cap_are_kept(self) -> None:
        plan = self._late_melon(15, shed_total=80)

        self.assertTrue(plan.feasible)
        self.assertEqual(_project_end_shed(*self._projection(plan, 80)), 95)

    def test_a_full_shed_after_the_drop_is_still_legal(self) -> None:
        plan = self._late_melon(15, shed_total=85)

        self.assertTrue(plan.feasible)
        self.assertEqual(_project_end_shed(*self._projection(plan, 85)), 100)

    def test_one_over_the_cap_is_not_harvested(self) -> None:
        plan = self._late_melon(15, shed_total=86)

        self.assertFalse(plan.feasible)
        self.assertEqual(plan.worker_routes[0].actions_by_hour, ())
        self.assertEqual([visit.coord for visit in plan.unfinished_visits], [(2, 4)])

    def test_three_workers_share_one_shed_limit(self) -> None:
        workers = [
            RegionWorker("Hand1", (0, 0)),
            RegionWorker("Hand2", (0, 1)),
            RegionWorker("Farmer", (0, 2)),
        ]
        legal = plan_region_routes(
            _grid(_harvest(0, 0, "MELON", 10), _harvest(0, 1, "MELON", 15), _harvest(0, 2, "MELON", 5)),
            workers,
            start_hour=23,
            end_hour=23,
            shed_total=70,
        )
        illegal = plan_region_routes(
            _grid(_harvest(0, 0, "MELON", 10), _harvest(0, 1, "MELON", 16), _harvest(0, 2, "MELON", 5)),
            workers,
            start_hour=23,
            end_hour=23,
            shed_total=70,
        )

        self.assertTrue(legal.feasible)
        self.assertEqual(sum(visit.harvest_units for route in legal.worker_routes for visit in route.visits), 30)
        self.assertFalse(illegal.feasible)
        self.assertLess(
            sum(visit.harvest_units for route in illegal.worker_routes for visit in route.visits),
            31,
        )
        pantry = _Pantry(0, shed_doors(10), shed_total=70)
        self.assertTrue(
            _end_day_capacity_safe(workers, self._cargo_routes(10, 15, 5), pantry, 70, 23, 23)
        )
        self.assertFalse(
            _end_day_capacity_safe(workers, self._cargo_routes(10, 16, 5), pantry, 70, 23, 23)
        )

    def test_picking_up_animals_frees_shed_space(self) -> None:
        worker = RegionWorker("Farmer", (4, 4))
        visits = [
            self._sheep_visit(4, 3),
            self._sheep_visit(4, 2),
        ]
        pantry = _Pantry(0, shed_doors(10), (("SHEEP", 2),), 100)

        self.assertEqual(_project_end_shed([worker], [visits], pantry, 100, 1, 23), 98)

    def test_an_unplaced_animal_returns_to_the_shed_at_day_end(self) -> None:
        worker = RegionWorker("Farmer", (4, 4))
        leg = _Leg((4, 4), 0, (self._sheep_visit(4, 3),), 1, item_pickups=(("SHEEP", 2),))

        self.assertEqual(_end_inventory(worker, leg).get("SHEEP"), 1)
        self.assertEqual(_shed_delta(worker, leg), -1)

    def test_unused_wheat_returns_to_the_shed_at_day_end(self) -> None:
        worker = RegionWorker("Farmer", (4, 4))
        leg = _Leg(
            (4, 4),
            3,
            (TileVisit((2, 4), (FEED,), 1, "SHEEP"), TileVisit((2, 3), (FEED,), 1, "SHEEP")),
            4,
        )

        self.assertEqual(_end_inventory(worker, leg).get("WHEAT"), 1)
        self.assertEqual(_shed_delta(worker, leg), -2)

    def test_an_unloaded_harvest_is_counted_once(self) -> None:
        worker = RegionWorker("Farmer", (4, 4))
        plan = plan_region_routes(
            _grid(_harvest(4, 4, "MELON", 6)),
            [worker],
            shed_total=0,
        )
        actions = plan.worker_routes[0].actions_by_hour

        self.assertEqual([action.operation for action in actions], ["HARVEST", "PLACE"])
        self.assertEqual(
            _project_end_shed(
                [worker],
                [list(plan.worker_routes[0].visits)],
                _Pantry(0, shed_doors(10)),
                0,
                1,
                23,
            ),
            6,
        )
        leg = _fit_leg(worker, plan.worker_routes[0].visits, shed_doors(10), 1, 23)
        assert leg is not None
        self.assertEqual(_end_inventory(worker, leg).get("MELON", 0), 0)
        self.assertEqual(sum(units for _, units in leg.deliveries), 6)

    def test_a_harvest_left_in_the_field_has_no_drop_hour(self) -> None:
        tiles = _tiles()
        tiles[0][0] = _plant("MELON", planted_day=-8, units=6)
        agent = make_region_phase1_agent()
        agent(_observation(tiles, day=12, hour=23, farmer=(0, 0)))
        crop = next(item for item in agent.telemetry["world"].farm.crops if item.crop == "MELON")
        harvest = next(item for item in _assignments(agent.telemetry["plan"]) if item.kind == HARVEST)

        self.assertEqual(crop.planned_harvest_hour, 23)
        self.assertEqual(harvest.planned_hour, 23)
        self.assertIsNone(crop.planned_drop_hour)
        self.assertIsNone(crop.planned_sell_hour)
        self.assertIsNone(harvest.planned_drop_hour)
        self.assertIsNone(harvest.planned_sell_hour)

    def test_optional_overflow_does_not_spoil_the_mandatory_route(self) -> None:
        worker = RegionWorker("Farmer", (0, 0))
        water = TaskBucket((0, 0), "WHEAT")
        water.tasks[WATER] = WaterTask(WATER, PENDING, True)
        extra = TaskBucket((0, 1), "MELON")
        extra.tasks[HARVEST] = HarvestTask(HARVEST, PENDING, False, yield_amount=5)
        plan = plan_region_routes(
            _grid(water, extra),
            [worker],
            shed_total=98,
        )
        visit = TileVisit((0, 1), (HARVEST,), 1, "MELON", "MELON", 5)
        pantry = _Pantry(0, shed_doors(10), shed_total=98)

        self.assertTrue(plan.feasible)
        self.assertEqual(plan.unfinished_visits, ())
        self.assertNotIn("HARVEST", [action.operation for action in plan.worker_routes[0].actions_by_hour])
        self.assertIsNone(
            _best_insertion([worker], [[TileVisit((0, 0), (WATER,), 1, "WHEAT")]], visit, 1, 23, pantry)
        )

    def test_planting_and_housing_chains_stay_whole(self) -> None:
        sown = plan_region_routes(
            _grid(_crop_plan(0, 0)),
            [RegionWorker("Farmer", (0, 0))],
            start_hour=23,
            end_hour=23,
        )
        housed = plan_region_routes(
            _grid(_animal_plan(4, 3)),
            [RegionWorker("Farmer", (4, 4))],
            shed_animals={"SHEEP": 1},
            shed_total=100,
        )

        self.assertEqual(_ops(sown), [])
        self.assertEqual(
            [operation for _, operation, _ in _ops(housed)],
            ["PICKUP", "NORTH", BUILD_PASTURE, "PLACE"],
        )

    def _late_melon(self, units: int, shed_total: int):
        return plan_region_routes(
            _grid(_harvest(2, 4, "MELON", units)),
            [RegionWorker("Farmer", (2, 4))],
            start_hour=23,
            end_hour=23,
            shed_total=shed_total,
        )

    def _projection(self, plan, shed_total: int):
        workers = [RegionWorker(route.worker_id, route.start_coord) for route in plan.worker_routes]
        routes = [list(route.visits) for route in plan.worker_routes]
        return workers, routes, _Pantry(0, shed_doors(10), shed_total=shed_total), shed_total, 23, 23

    def _cargo_routes(self, hand1: int, hand2: int, farmer: int) -> list[list[TileVisit]]:
        return [
            [TileVisit((0, 0), (HARVEST,), 1, "MELON", "MELON", hand1)],
            [TileVisit((0, 1), (HARVEST,), 1, "MELON", "MELON", hand2)],
            [TileVisit((0, 2), (HARVEST,), 1, "MELON", "MELON", farmer)],
        ]

    def _sheep_visit(self, x: int, y: int) -> TileVisit:
        return TileVisit(
            (x, y),
            (BUILD_PASTURE, PLACE_ANIMAL),
            2,
            "EMPTY",
            None,
            0,
            ((), ("SHEEP",)),
            "animal",
            "SHEEP",
        )
