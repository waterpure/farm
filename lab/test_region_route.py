"""A 5×5 day is planned one tile at a time, for one to four workers."""

from __future__ import annotations

import unittest

from lab.region_route import (
    RegionWorker,
    TileVisit,
    _best_insertion,
    _extract_visits,
    _insertion_extra,
    _path_moves,
    _reorder,
    _visit_sort,
    plan_region_routes,
)
from lab.route14_state import PENDING, SCHEDULED
from lab.task_grid import (
    CARE,
    COLLECT_FERTILIZER,
    FEED,
    FERTILIZE,
    HARVEST,
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


def _harvest(x: int, y: int, tile: str = "WHEAT") -> TaskBucket:
    bucket = TaskBucket((x, y), tile)
    bucket.tasks[HARVEST] = HarvestTask(HARVEST, PENDING, True, yield_amount=1)
    return bucket


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
        self.assertEqual(plan.total_move_count, 2)
        self.assertEqual(plan.unfinished_visits, ())
        route = plan.worker_routes[0]
        self.assertEqual(
            [(action.hour, action.operation, action.coord) for action in route.actions_by_hour],
            [
                (1, "HARVEST", (0, 0)),
                (2, "EAST", None),
                (3, "HARVEST", (1, 0)),
                (4, "EAST", None),
                (5, "HARVEST", (2, 0)),
            ],
        )
        self.assertNotIn("GO_TO", [action.operation for action in route.actions_by_hour])
        self.assertEqual(route.finish_hour, 5)

    def test_ripe_wheat_is_harvested_without_watering(self) -> None:
        bucket = TaskBucket((2, 3), "WHEAT")
        bucket.tasks[WATER] = WaterTask(WATER, PENDING, True, needed=True, turns_until_weed=0, yield_gain=1)
        bucket.tasks[HARVEST] = HarvestTask(HARVEST, PENDING, True, yield_amount=2)
        thirsty = TaskBucket((0, 0), "WHEAT")
        thirsty.tasks[WATER] = WaterTask(WATER, PENDING, True, needed=True, turns_until_weed=0, yield_gain=1)
        plan = plan_region_routes(_grid(bucket, thirsty), [RegionWorker("Farmer", (2, 2))])

        ripe = next(visit for visit in plan.worker_routes[0].visits if visit.coord == (2, 3))
        young = next(visit for visit in plan.worker_routes[0].visits if visit.coord == (0, 0))
        self.assertEqual(ripe.tasks, (HARVEST,))
        self.assertEqual(ripe.action_count, 1)
        self.assertEqual(young.tasks, (WATER,))
        ripe_actions = [
            (action.operation, action.coord)
            for action in plan.worker_routes[0].actions_by_hour
            if action.coord == (2, 3)
        ]
        self.assertEqual(ripe_actions, [("HARVEST", (2, 3))])
        self.assertNotIn(
            ("WATER", (2, 3)),
            [(action.operation, action.coord) for action in plan.worker_routes[0].actions_by_hour],
        )

    def test_one_tile_has_one_owner(self) -> None:
        bucket = TaskBucket((2, 3), "WHEAT")
        bucket.tasks[WATER] = WaterTask(WATER, PENDING, True, needed=True, turns_until_weed=0, yield_gain=1)
        bucket.tasks[HARVEST] = HarvestTask(HARVEST, PENDING, True, yield_amount=2)
        workers = [RegionWorker("Farmer", (2, 3)), RegionWorker("Hand1", (0, 0))]
        plan = plan_region_routes(_grid(bucket), workers)

        self.assertEqual(_owners(plan), {(2, 3): "Farmer"})
        self.assertEqual(plan.worker_routes[1].visits, ())
        self.assertEqual(plan.worker_routes[1].move_count, 0)
        self.assertEqual(plan.total_move_count, 0)

    def test_an_ongoing_crop_keeps_water_before_harvest_in_the_same_visit(self) -> None:
        bucket = TaskBucket((1, 1), "TOMATO")
        bucket.tasks[WATER] = WaterTask(WATER, PENDING, True, needed=True, turns_until_weed=0, yield_gain=0)
        bucket.tasks[HARVEST] = HarvestTask(HARVEST, PENDING, True, yield_amount=1)
        bucket.tasks[FERTILIZE] = FertilizeTask(FERTILIZE, PENDING, False)
        plan = plan_region_routes(_grid(bucket), [RegionWorker("Farmer", (1, 1))])

        visit = plan.worker_routes[0].visits[0]
        self.assertEqual(visit.tasks, (WATER, HARVEST))
        self.assertEqual(visit.action_count, 2)
        self.assertEqual(
            [action.operation for action in plan.worker_routes[0].actions_by_hour],
            [WATER, HARVEST],
        )

    def test_feed_and_harvest_on_one_animal_stay_together(self) -> None:
        bucket = TaskBucket((3, 4), "SHEEP")
        bucket.tasks[FEED] = FeedTask(FEED, PENDING, True)
        bucket.tasks[HARVEST] = HarvestTask(HARVEST, PENDING, True, yield_amount=1)
        bucket.tasks[CARE] = CareTask(CARE, PENDING, False)
        bucket.tasks[COLLECT_FERTILIZER] = CollectFertilizerTask(COLLECT_FERTILIZER, PENDING, False, fertilizer_ready=True)
        plan = plan_region_routes(_grid(bucket), [RegionWorker("Farmer", (3, 4))])

        self.assertEqual(plan.worker_routes[0].visits[0].tasks, (FEED, HARVEST))
        self.assertEqual(
            [action.operation for action in plan.worker_routes[0].actions_by_hour],
            [FEED, HARVEST],
        )

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
        self.assertTrue(all(visit.tasks == (HARVEST,) for visit in plan.worker_routes[0].visits))

    def test_insertion_follows_the_route_not_each_workers_start(self) -> None:
        grid = _grid(_harvest(0, 0), _harvest(4, 0), _harvest(4, 1))
        workers = [RegionWorker("A", (0, 0)), RegionWorker("B", (0, 1))]
        plan = plan_region_routes(grid, workers)

        self.assertTrue(plan.feasible)
        self.assertEqual(plan.total_move_count, 5)
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
        self.assertEqual(plan.total_move_count, 4)
        self.assertLess(plan.total_move_count, alone.total_move_count)
        owners = _owners(plan)
        self.assertEqual(owners[(0, 0)], "A")
        self.assertEqual(owners[(0, 4)], "B")
        self.assertEqual(owners[(4, 0)], "C")
        self.assertEqual(owners[(4, 4)], "D")
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
        self.assertEqual(one.total_move_count, 16)
        self.assertLess(two.total_move_count, one.total_move_count)
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

        self.assertEqual(len(plan.worker_routes[0].visits), 11)
        self.assertTrue(plan.feasible)
        self.assertLessEqual(plan.total_move_count, 12)
        self.assertLessEqual(plan.finish_hour, 23)
        self.assertEqual(plan.unfinished_visits, ())

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
        self.assertEqual(plan.total_move_count, 0)

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
