"""One worker's day route reads the task grid and does not walk back to the shed."""

from __future__ import annotations

import unittest

from lab.route14_state import PENDING, SCHEDULED
from lab.task_grid import (
    DIG,
    FEED,
    HARVEST,
    WATER,
    FeedTask,
    HarvestTask,
    TaskBucket,
    TaskGrid,
    TileTask,
    WaterTask,
)
from lab.worker_route import LAST_HOUR, _children, _must_jobs, _target_table, plan_worker_route


def _harvest(x: int, y: int, tile: str = "WHEAT") -> TaskBucket:
    bucket = TaskBucket((x, y), tile)
    bucket.tasks[HARVEST] = HarvestTask(HARVEST, PENDING, True, yield_amount=1)
    return bucket


def _grid(*buckets: TaskBucket) -> TaskGrid:
    grid = TaskGrid()
    for bucket in buckets:
        grid.put(bucket)
    return grid


class WorkerRouteTests(unittest.TestCase):
    def test_a_straight_line_is_not_walked_back_to_the_shed(self) -> None:
        start = TaskBucket((0, 0), "WHEAT")
        start.tasks[WATER] = WaterTask(WATER, PENDING, False, needed=True, turns_until_weed=20, yield_gain=1)
        start.tasks[DIG] = TileTask(DIG, PENDING, False)
        sheep = TaskBucket((0, 1), "SHEEP")
        sheep.tasks[FEED] = FeedTask(FEED, PENDING, False)
        later = TaskBucket((8, 8), "WHEAT")
        later.tasks[WATER] = WaterTask(WATER, SCHEDULED, True, assigned_worker="Hand2", planned_hour=6, needed=True)
        fed = TaskBucket((3, 0), "SHEEP")
        fed.tasks[FEED] = FeedTask(FEED, PENDING, True)
        grid = _grid(start, sheep, later, _harvest(1, 0), _harvest(2, 0), fed, _harvest(4, 0))

        plan = plan_worker_route((0, 0), 1, grid)

        self.assertTrue(plan.feasible)
        self.assertEqual(plan.move_count, 4)
        self.assertEqual(
            [(action.operation, action.coord) for action in plan.actions],
            [
                ("EAST", None),
                ("HARVEST", (1, 0)),
                ("EAST", None),
                ("HARVEST", (2, 0)),
                ("EAST", None),
                ("FEED", (3, 0)),
                ("EAST", None),
                ("HARVEST", (4, 0)),
            ],
        )
        self.assertNotIn("WATER", [action.operation for action in plan.actions])
        self.assertNotIn("DIG", [action.operation for action in plan.actions])

    def test_adjacent_fields_are_visited_one_after_another(self) -> None:
        grid = _grid(_harvest(2, 3), _harvest(2, 4))
        plan = plan_worker_route((2, 3), 1, grid)
        self.assertTrue(plan.feasible)
        self.assertEqual(plan.move_count, 1)
        self.assertEqual(
            [(action.hour, action.operation, action.coord) for action in plan.actions],
            [(1, "HARVEST", (2, 3)), (2, "SOUTH", None), (3, "HARVEST", (2, 4))],
        )

    def test_water_and_harvest_on_one_tile_are_done_together(self) -> None:
        bucket = TaskBucket((2, 3), "TOMATO")
        bucket.tasks[WATER] = WaterTask(WATER, PENDING, True, needed=True, turns_until_weed=0, yield_gain=1)
        bucket.tasks[HARVEST] = HarvestTask(HARVEST, PENDING, True, yield_amount=1)
        plan = plan_worker_route((2, 3), 1, _grid(bucket))
        self.assertTrue(plan.feasible)
        self.assertEqual(plan.move_count, 0)
        self.assertEqual([action.operation for action in plan.actions], ["WATER", "HARVEST"])
        self.assertEqual([action.hour for action in plan.actions], [1, 2])
        self.assertEqual(plan.completed_tasks, (("WATER", (2, 3)), ("HARVEST", (2, 3))))

    def test_one_shot_harvest_does_not_leave_a_water_behind(self) -> None:
        bucket = TaskBucket((2, 3), "WHEAT")
        bucket.tasks[WATER] = WaterTask(WATER, PENDING, True, needed=True, turns_until_weed=4, yield_gain=1)
        bucket.tasks[HARVEST] = HarvestTask(HARVEST, PENDING, True, yield_amount=4)
        grid = _grid(bucket)
        jobs = _must_jobs(grid)
        targets = _target_table(jobs)
        root = (1, 2, 3, (1 << len(jobs)) - 1, 0, 0, -1, "", -1, -1)
        openings = [child[7] for child in _children(root, 0, jobs, targets, 10, 10)]
        self.assertIn("WATER", openings)
        self.assertIn("HARVEST", openings)
        harvested = next(child for child in _children(root, 0, jobs, targets, 10, 10) if child[7] == "HARVEST")
        self.assertNotIn("WATER", [child[7] for child in _children(harvested, 1, jobs, targets, 10, 10)])

        plan = plan_worker_route((2, 3), 1, grid)
        self.assertTrue(plan.feasible)
        self.assertEqual([action.operation for action in plan.actions], ["WATER", "HARVEST"])
        self.assertEqual(plan.unfinished_tasks, ())

    def test_a_far_job_is_not_lost_to_the_nearest_one(self) -> None:
        grid = _grid(_harvest(0, 3), _harvest(1, 1), _harvest(2, 0), _harvest(7, 0))
        plan = plan_worker_route((0, 0), 7, grid)
        self.assertTrue(plan.feasible)
        self.assertEqual(plan.move_count, 13)
        self.assertEqual(plan.finish_hour, LAST_HOUR)
        self.assertEqual(
            plan.completed_tasks,
            (("HARVEST", (0, 3)), ("HARVEST", (1, 1)), ("HARVEST", (2, 0)), ("HARVEST", (7, 0))),
        )

    def test_twenty_three_actions_still_fit(self) -> None:
        buckets = [_harvest(x, 0) for x in range(2, 10)]
        buckets.extend(_harvest(9, y) for y in range(1, 4))
        plan = plan_worker_route((0, 0), 1, _grid(*buckets))
        self.assertTrue(plan.feasible)
        self.assertEqual(len(plan.actions), 23)
        self.assertEqual(plan.actions[0].hour, 1)
        self.assertEqual(plan.actions[-1].hour, 23)
        self.assertEqual(plan.move_count, 12)
        self.assertEqual(plan.finish_hour, 23)
        self.assertEqual(plan.unfinished_tasks, ())
        self.assertTrue(all(action.hour <= LAST_HOUR for action in plan.actions))

    def test_twenty_four_actions_do_not_fit(self) -> None:
        buckets = [_harvest(x, 0) for x in range(1, 10)]
        buckets.extend(_harvest(9, y) for y in range(1, 4))
        plan = plan_worker_route((0, 0), 1, _grid(*buckets))
        self.assertFalse(plan.feasible)
        self.assertGreater(len(plan.unfinished_tasks), 0)
        self.assertLessEqual(len(plan.actions), 23)
        self.assertTrue(all(1 <= action.hour <= LAST_HOUR for action in plan.actions))
