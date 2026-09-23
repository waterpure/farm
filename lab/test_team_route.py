"""Headcount is the smallest crew whose fresh joint route finishes every must job."""

from __future__ import annotations

import unittest

from lab.route14_state import PENDING, WorkerState
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
from lab.team_route import MAX_HIRE_COUNT, plan_minimum_hands, plan_team_routes, RouteWorker


def _harvest(x: int, y: int, tile: str = "WHEAT") -> TaskBucket:
    bucket = TaskBucket((x, y), tile)
    bucket.tasks[HARVEST] = HarvestTask(HARVEST, PENDING, True, yield_amount=1)
    return bucket


def _grid(*buckets: TaskBucket) -> TaskGrid:
    grid = TaskGrid()
    for bucket in buckets:
        grid.put(bucket)
    return grid


def _claimed(plan) -> list[tuple[str, tuple[int, int]]]:
    return [task for worker in plan.workers for task in worker.assigned_tasks]


class TeamRouteTests(unittest.TestCase):
    def test_zero_hands_finish_the_must_work(self) -> None:
        farmer = RouteWorker("Farmer", (2, 3))
        grid = _grid(_harvest(2, 3), _harvest(2, 4))

        plan = plan_minimum_hands(farmer, [(0, 0), (9, 9)], grid)

        self.assertTrue(plan.feasible)
        self.assertEqual(plan.min_required_hands, 0)
        self.assertEqual(plan.tried_hand_counts, (0,))
        self.assertEqual([worker.worker_id for worker in plan.team.workers], ["Farmer"])
        self.assertEqual(plan.team.unfinished_tasks, ())
        self.assertEqual(plan.team.workers[0].start_coord, (2, 3))
        self.assertLessEqual(plan.team.workers[0].actions_by_hour[-1].hour, 23)

    def test_one_hand_finishes_what_the_farmer_cannot(self) -> None:
        farmer = WorkerState(actor=0, role="farmer", position=(0, 0), carrying={})
        self.assertEqual(farmer.coord, (0, 0))
        grid = _grid(_harvest(0, 9), _harvest(9, 9), _harvest(9, 0))

        plan = plan_minimum_hands(farmer, [(9, 0), (0, 9)], grid)

        self.assertTrue(plan.feasible)
        self.assertEqual(plan.min_required_hands, 1)
        self.assertEqual(plan.tried_hand_counts, (0, 1))
        self.assertEqual([worker.worker_id for worker in plan.team.workers], ["Farmer", "Hand1"])
        self.assertEqual(plan.team.workers[1].start_coord, (9, 0))
        self.assertEqual(plan.team.unfinished_tasks, ())
        self.assertEqual(len(_claimed(plan.team)), len(set(_claimed(plan.team))))
        self.assertEqual(
            set(plan.team.completed_tasks),
            {("HARVEST", (0, 9)), ("HARVEST", (9, 9)), ("HARVEST", (9, 0))},
        )
        hand = plan.team.workers[1]
        self.assertIn(("HARVEST", (9, 0)), hand.assigned_tasks)
        self.assertNotIn((0, 0), [action.coord for action in hand.actions_by_hour])

    def test_two_hands_finish_what_one_hand_cannot(self) -> None:
        farmer = RouteWorker("Farmer", (0, 0))
        grid = _grid(_harvest(0, 0), _harvest(5, 0), _harvest(0, 5))

        plan = plan_minimum_hands(farmer, [(5, 0), (0, 5), (9, 9)], grid, start_hour=21)

        self.assertTrue(plan.feasible)
        self.assertEqual(plan.min_required_hands, 2)
        self.assertEqual(plan.tried_hand_counts, (0, 1, 2))
        self.assertEqual(
            [worker.start_coord for worker in plan.team.workers],
            [(0, 0), (5, 0), (0, 5)],
        )
        claimed = _claimed(plan.team)
        self.assertEqual(len(claimed), len(set(claimed)))
        self.assertEqual(set(plan.team.completed_tasks), set(claimed))
        self.assertEqual(plan.team.unfinished_tasks, ())
        for worker in plan.team.workers:
            self.assertTrue(all(1 <= action.hour <= 23 for action in worker.actions_by_hour))

    def test_ten_hands_still_cannot_finish(self) -> None:
        grid = TaskGrid()
        for x in range(10):
            for y in range(10):
                bucket = TaskBucket((x, y), "WHEAT")
                bucket.tasks[WATER] = WaterTask(WATER, PENDING, True, needed=True, turns_until_weed=0, yield_gain=1)
                bucket.tasks[HARVEST] = HarvestTask(HARVEST, PENDING, True, yield_amount=1)
                grid.put(bucket)
        coords = [(index % 10, index // 10) for index in range(MAX_HIRE_COUNT + 2)]

        plan = plan_minimum_hands(RouteWorker("Farmer", (0, 0)), coords, grid)

        self.assertFalse(plan.feasible)
        self.assertIsNone(plan.min_required_hands)
        self.assertEqual(plan.tried_hand_counts, tuple(range(MAX_HIRE_COUNT + 1)))
        self.assertEqual(len(plan.team.workers), MAX_HIRE_COUNT + 1)
        self.assertEqual(plan.team.workers[1].start_coord, (0, 0))
        self.assertEqual(plan.team.workers[-1].start_coord, (9, 0))
        self.assertEqual(plan.team.workers[-1].worker_id, "Hand10")
        self.assertGreater(len(plan.team.unfinished_tasks), 0)
        claimed = _claimed(plan.team)
        self.assertEqual(len(claimed), len(set(claimed)))

    def test_optional_care_and_fertilizer_do_not_enter_the_crew(self) -> None:
        tomato = TaskBucket((4, 4), "TOMATO")
        tomato.tasks[WATER] = WaterTask(WATER, PENDING, False, needed=True, turns_until_weed=4, yield_gain=1)
        tomato.tasks[FERTILIZE] = FertilizeTask(FERTILIZE, PENDING, False)
        sheep = TaskBucket((8, 8), "SHEEP")
        sheep.tasks[FEED] = FeedTask(FEED, PENDING, False)
        sheep.tasks[CARE] = CareTask(CARE, PENDING, False)
        sheep.tasks[COLLECT_FERTILIZER] = CollectFertilizerTask(
            COLLECT_FERTILIZER, PENDING, False, fertilizer_ready=True
        )
        dying = TaskBucket((0, 1), "CARROT")
        dying.tasks[WATER] = WaterTask(WATER, PENDING, True, needed=True, turns_until_weed=0, yield_gain=1)
        hungry = TaskBucket((0, 2), "SHEEP")
        hungry.tasks[FEED] = FeedTask(FEED, PENDING, True, consecutive_unfed=1)

        plan = plan_minimum_hands(
            RouteWorker("Farmer", (0, 0)),
            [(8, 8)],
            _grid(tomato, sheep, dying, hungry),
        )

        self.assertEqual(plan.min_required_hands, 0)
        self.assertTrue(plan.feasible)
        self.assertEqual(
            set(plan.team.completed_tasks),
            {("WATER", (0, 1)), ("FEED", (0, 2))},
        )
        self.assertNotIn("CARE", [task[0] for task in _claimed(plan.team)])
        self.assertNotIn("FERTILIZE", [task[0] for task in _claimed(plan.team)])
        self.assertNotIn("COLLECT_FERTILIZER", [task[0] for task in _claimed(plan.team)])

    def test_water_and_harvest_on_one_tile_stay_with_one_worker(self) -> None:
        bucket = TaskBucket((2, 3), "TOMATO")
        bucket.tasks[WATER] = WaterTask(WATER, PENDING, True, needed=True, turns_until_weed=0, yield_gain=1)
        bucket.tasks[HARVEST] = HarvestTask(HARVEST, PENDING, True, yield_amount=1)
        crew = [RouteWorker("Farmer", (2, 3)), RouteWorker("Hand1", (8, 8))]

        plan = plan_team_routes(crew, _grid(bucket))

        self.assertTrue(plan.feasible)
        self.assertEqual(plan.total_move_count, 0)
        farmer, hand = plan.workers
        self.assertEqual([action.operation for action in farmer.actions_by_hour], ["WATER", "HARVEST"])
        self.assertEqual([action.hour for action in farmer.actions_by_hour], [1, 2])
        self.assertEqual(farmer.visited_tiles, ((2, 3),))
        self.assertEqual(hand.assigned_tasks, ())
        self.assertEqual(hand.actions_by_hour, ())

    def test_a_near_split_loses_to_the_shorter_joint_walk(self) -> None:
        grid = _grid(_harvest(0, 1), _harvest(5, 1))
        crew = [RouteWorker("Farmer", (0, 0)), RouteWorker("Hand1", (0, 2))]

        plan = plan_team_routes(crew, grid)

        self.assertTrue(plan.feasible)
        self.assertEqual(plan.total_move_count, 6)
        self.assertTrue(any(len(worker.assigned_tasks) == 2 for worker in plan.workers))
        self.assertEqual(len(_claimed(plan)), len(set(_claimed(plan))))

    def test_hour_zero_does_not_walk(self) -> None:
        plan = plan_team_routes([RouteWorker("Farmer", (1, 1))], _grid(_harvest(1, 1)), start_hour=0)

        self.assertEqual([action.hour for action in plan.workers[0].actions_by_hour], [1])
        self.assertTrue(all(action.hour >= 1 for action in plan.workers[0].actions_by_hour))
