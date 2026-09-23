"""An empty tile keeps one production chain on the same grid as the standing crops."""

from __future__ import annotations

import unittest

from lab.production_plan import (
    _account,
    _commit,
    _ledger,
    can_start_production,
    choose_production_plan,
    feed_runway_units,
    portfolio_summary,
    production_candidates,
    seedling_water_is_mandatory,
)
from lab.route14_economy import CASH_BUFFER, animal_yield_units, wheat_buy_price
from lab.route14_state import COMPLETED, parse_world
from lab.task_grid import (
    BUILD_PASTURE,
    CARE,
    FEED,
    PLACE_ANIMAL,
    PLANT,
    WATER,
    TileTask,
    WaterTask,
    build_task_grid,
)
from lab.test_route14_phase1 import _animal, _observation, _plant, _tiles


def _open(tiles: list[list], *spots: tuple[int, int]) -> None:
    for x, y in spots:
        tiles[y][x] = None


def _priced(observation: dict, **prices: int) -> dict:
    # An empty market book makes the economy use these quotes. A full book
    # would price the glut curve and ignore the quotes.
    observation["market"]["prices"] = dict(prices)
    observation["market"]["inventory"] = {}
    return observation


class ProductionPlanTests(unittest.TestCase):
    def test_one_empty_tile_keeps_only_the_best_plan(self) -> None:
        tiles = _tiles()
        _open(tiles, (4, 4))
        observation = _priced(
            _observation(tiles, day=0, money=5000),
            MELON=400,
            WHEAT=30,
            CARROT=1,
            TOMATO=1,
            STRAWBERRY=1,
            EGG=1,
            MILK=1,
            WOOL=1,
        )
        grid = build_task_grid(parse_world(observation), observation=observation)
        cell = grid[4][4]
        candidates = production_candidates(observation, (4, 4))

        self.assertEqual(len(candidates), 8)
        self.assertEqual(cell.production_plan.name, "MELON")
        self.assertEqual(set(cell.tasks), {PLANT, WATER})
        self.assertEqual(cell.tasks[PLANT].subject, "MELON")
        self.assertNotIn(BUILD_PASTURE, cell.tasks)

    def test_a_crop_plan_is_plant_then_water(self) -> None:
        tiles = _tiles()
        _open(tiles, (4, 4))
        observation = _priced(_observation(tiles, day=0, money=5000), MELON=400, WHEAT=1)
        plan = choose_production_plan(observation, (4, 4))
        assert plan is not None
        grid = build_task_grid(parse_world(observation), observation=observation)
        cell = grid[4][4]

        self.assertEqual([(action.operation, action.subject) for action in plan.actions], [(PLANT, "MELON"), (WATER, "MELON")])
        self.assertIsInstance(cell.tasks[WATER], TileTask)
        self.assertNotIsInstance(cell.tasks[WATER], WaterTask)
        self.assertEqual(cell.tasks[WATER].subject, "MELON")
        self.assertEqual(cell.tasks[WATER].depends_on, PLANT)
        self.assertFalse(cell.tasks[WATER].mandatory)
        self.assertFalse(seedling_water_is_mandatory(plan, "PENDING"))
        self.assertTrue(seedling_water_is_mandatory(plan, COMPLETED))

    def test_an_animal_plan_is_build_then_place(self) -> None:
        tiles = _tiles()
        _open(tiles, (4, 4))
        observation = _priced(
            _observation(tiles, day=0, money=0, shed={"SHEEP": 1, "WHEAT": 7}),
            WOOL=500,
            WHEAT=1,
            MELON=1,
            MILK=1,
            EGG=1,
        )
        grid = build_task_grid(parse_world(observation), observation=observation)
        plan = grid[4][4].production_plan

        self.assertEqual(plan.name, "SHEEP")
        self.assertEqual(plan.product, "WOOL")
        self.assertEqual(plan.startup_cash, 0)
        self.assertEqual(
            [(action.operation, action.subject) for action in plan.actions],
            [(BUILD_PASTURE, ""), (PLACE_ANIMAL, "SHEEP"), ("FEED", "SHEEP")],
        )
        self.assertEqual(grid[4][4].tasks[PLACE_ANIMAL].depends_on, BUILD_PASTURE)
        self.assertEqual(grid[4][4].tasks["FEED"].depends_on, PLACE_ANIMAL)
        self.assertNotIn("PICKUP", grid[4][4].tasks)
        self.assertFalse(can_start_production(plan, 3))
        self.assertTrue(can_start_production(plan, 4))

    def test_one_seed_can_start_only_one_tile(self) -> None:
        tiles = _tiles()
        _open(tiles, (4, 4), (0, 0))
        observation = _priced(_observation(tiles, day=0, money=0), MELON=400, WHEAT=1)
        observation["private"]["seeds"] = {"MELON": 1}
        grid = build_task_grid(parse_world(observation), observation=observation)
        plans = [grid[x][y].production_plan for x, y in ((4, 4), (0, 0)) if grid[x][y] and grid[x][y].production_plan]

        self.assertEqual([plan.name for plan in plans], ["MELON"])

    def test_one_sheep_in_the_shed_fills_only_one_tile(self) -> None:
        tiles = _tiles()
        _open(tiles, (4, 4), (0, 0))
        observation = _priced(
            _observation(tiles, day=0, money=0, shed={"SHEEP": 1, "WHEAT": 7}),
            WOOL=500,
            WHEAT=1,
            MELON=1,
        )
        grid = build_task_grid(parse_world(observation), observation=observation)
        sheep = [
            (x, y)
            for x, y in ((4, 4), (0, 0))
            if grid[x][y] and grid[x][y].production_plan and grid[x][y].production_plan.name == "SHEEP"
        ]

        self.assertEqual(sheep, [(4, 4)])

    def test_cash_for_one_cow_is_not_spent_twice(self) -> None:
        tiles = _tiles()
        _open(tiles, (4, 4), (0, 0))
        observation = _priced(
            _observation(tiles, day=0, money=609),
            MILK=500,
            WHEAT=1,
            MELON=1,
            EGG=1,
            WOOL=1,
        )
        grid = build_task_grid(parse_world(observation), observation=observation)
        cows = [
            (x, y)
            for x, y in ((4, 4), (0, 0))
            if grid[x][y] and grid[x][y].production_plan and grid[x][y].production_plan.name == "COW"
        ]

        self.assertEqual(cows, [(4, 4)])
        self.assertIsNone(grid[0][0].production_plan)

    def test_a_crop_that_cannot_finish_is_not_committed(self) -> None:
        tiles = _tiles()
        _open(tiles, (4, 4))
        observation = _priced(_observation(tiles, day=22, money=5000), MELON=400, WHEAT=1)
        melon = next(item for item in production_candidates(observation, (4, 4)) if item.name == "MELON")
        plan = choose_production_plan(observation, (4, 4))

        self.assertFalse(melon.feasible_before_end)
        self.assertTrue(plan is None or plan.name != "MELON")

    def test_the_better_line_takes_the_tile_closer_to_the_shed(self) -> None:
        tiles = _tiles()
        _open(tiles, (4, 4), (0, 0))
        observation = _priced(
            _observation(tiles, day=0, money=290),
            MELON=400,
            WHEAT=40,
            CARROT=1,
            TOMATO=1,
            STRAWBERRY=1,
            EGG=1,
            MILK=1,
            WOOL=1,
        )
        grid = build_task_grid(parse_world(observation), observation=observation)

        self.assertEqual(grid[4][4].production_plan.name, "MELON")
        self.assertEqual(grid[0][0].production_plan.name, "WHEAT")

    def test_locked_ground_has_no_plan(self) -> None:
        observation = _priced(_observation(_tiles(), day=0, money=5000), MELON=400)
        grid = build_task_grid(parse_world(observation), observation=observation)

        self.assertTrue(all(cell is None for row in grid._cells for cell in row))

    def test_standing_crops_and_animals_are_not_given_a_new_plan(self) -> None:
        tiles = _tiles()
        tiles[3][2] = _plant("WHEAT", dry=1)
        tiles[1][1] = _animal("SHEEP", unfed=1)
        _open(tiles, (4, 4))
        observation = _priced(_observation(tiles, day=0, money=5000), MELON=400, WHEAT=1)
        grid = build_task_grid(parse_world(observation), observation=observation)

        self.assertIsNone(grid[2][3].production_plan)
        self.assertNotIn(PLANT, grid[2][3].tasks)
        self.assertIsNone(grid[1][1].production_plan)
        self.assertNotIn(BUILD_PASTURE, grid[1][1].tasks)
        self.assertEqual(grid[4][4].production_plan.name, "MELON")

    def test_standing_work_and_an_empty_plan_share_one_grid(self) -> None:
        tiles = _tiles()
        tiles[3][2] = _plant("WHEAT", dry=1)
        tiles[1][1] = _animal("SHEEP", unfed=1)
        _open(tiles, (4, 4))
        observation = _priced(_observation(tiles, day=0, money=5000), MELON=400, WHEAT=1, WOOL=1)
        grid = build_task_grid(parse_world(observation), observation=observation)

        self.assertIn(WATER, grid[2][3].tasks)
        self.assertIsInstance(grid[2][3].tasks[WATER], WaterTask)
        self.assertEqual(grid[4][4].production_plan.name, "MELON")
        self.assertIsInstance(grid[4][4].tasks[WATER], TileTask)
        self.assertIn(FEED, grid[1][1].tasks)
        self.assertIn(CARE, grid[1][1].tasks)

    def test_a_crop_plan_does_not_start_when_only_the_sowing_fits(self) -> None:
        tiles = _tiles()
        _open(tiles, (4, 4))
        observation = _priced(_observation(tiles, day=0, money=5000), MELON=400, WHEAT=1)
        plan = choose_production_plan(observation, (4, 4))
        assert plan is not None

        self.assertTrue(plan.atomic_start)
        self.assertEqual([(action.operation, action.subject) for action in plan.actions], [(PLANT, "MELON"), (WATER, "MELON")])
        self.assertFalse(can_start_production(plan, 1))
        self.assertTrue(can_start_production(plan, 2))

    def test_a_sheep_that_cannot_be_fed_is_not_bought(self) -> None:
        tiles = _tiles()
        _open(tiles, (4, 4))
        observation = _priced(_observation(tiles, day=0, money=500), WOOL=500, WHEAT=25, MELON=1)
        sheep = next(item for item in production_candidates(observation, (4, 4)) if item.name == "SHEEP")

        self.assertEqual(sheep.startup_cash, 500)
        self.assertEqual(sheep.feed_units, feed_runway_units("SHEEP"))
        self.assertGreater(sheep.feed_reserve_cash, 0)
        self.assertFalse(sheep.affordable)
        plan = choose_production_plan(observation, (4, 4))
        self.assertTrue(plan is None or plan.name != "SHEEP")

    def test_wheat_already_in_the_shed_covers_the_first_sheep(self) -> None:
        tiles = _tiles()
        _open(tiles, (4, 4))
        observation = _priced(
            _observation(tiles, day=0, money=1000, shed={"WHEAT": 5}, inventories=[{"WHEAT": 2}]),
            WOOL=500,
            WHEAT=25,
            MELON=1,
        )
        sheep = next(item for item in production_candidates(observation, (4, 4)) if item.name == "SHEEP")

        self.assertEqual(sheep.feed_units, 7)
        self.assertEqual(sheep.feed_reserve_cash, 0)
        self.assertEqual(sheep.required_cash, 500)
        self.assertTrue(sheep.affordable)

    def test_the_second_sheep_cannot_reuse_the_first_sheep_wheat(self) -> None:
        tiles = _tiles()
        _open(tiles, (4, 4))
        observation = _priced(
            _observation(tiles, day=0, money=2000, shed={"WHEAT": 7}),
            WOOL=500,
            WHEAT=25,
            MELON=1,
        )
        books = _ledger(observation)
        first = next(item for item in production_candidates(observation, (4, 4), books) if item.name == "SHEEP")
        _account(books, observation, _commit(first), pay=True)
        second = next(item for item in production_candidates(observation, (4, 4), books) if item.name == "SHEEP")

        self.assertEqual(first.feed_reserve_cash, 0)
        self.assertEqual(second.feed_reserve_cash, 7 * wheat_buy_price(observation))
        self.assertEqual(books.reserved_feed_units, 7)

    def test_planned_wool_lowers_the_next_sheep(self) -> None:
        tiles = _tiles()
        _open(tiles, (4, 4))
        observation = _observation(tiles, day=0, money=5000, shed={"WHEAT": 20})
        books = _ledger(observation)
        first = next(item for item in production_candidates(observation, (4, 4), books) if item.name == "SHEEP")
        _account(books, observation, _commit(first), pay=True)
        second = next(item for item in production_candidates(observation, (4, 4), books) if item.name == "SHEEP")

        self.assertGreater(books.planned_supply["WOOL"], 0)
        self.assertEqual(books.planned_supply["WOOL"], animal_yield_units("SHEEP", 30))
        self.assertLess(second.money_per_day, first.money_per_day)

    def test_the_second_tile_can_switch_off_sheep(self) -> None:
        tiles = _tiles()
        _open(tiles, (4, 4), (0, 0))
        observation = _observation(tiles, day=0, money=5000, shed={"WHEAT": 20})
        grid = build_task_grid(parse_world(observation), observation=observation)

        self.assertEqual(grid[4][4].production_plan.name, "SHEEP")
        self.assertEqual(grid[0][0].production_plan.name, "MELON")
        self.assertEqual(grid.portfolio.animals, (("SHEEP", 1),))

    def test_opening_cash_is_not_all_spent_on_sheep(self) -> None:
        tiles = _tiles()
        for y in range(5):
            for x in range(5):
                tiles[y][x] = None
        observation = _observation(tiles, day=0, money=3000)
        grid = build_task_grid(parse_world(observation), observation=observation)
        portfolio = grid.portfolio
        free = portfolio.cash_remaining - portfolio.reserved_feed_cash - portfolio.reserved_hire_cash

        self.assertGreaterEqual(free, CASH_BUFFER)
        self.assertGreater(portfolio.reserved_feed_cash, 0)
        self.assertFalse(
            portfolio.cash_remaining == 0
            and portfolio.reserved_feed_cash == 0
            and dict(portfolio.animals).get("SHEEP") == 6
        )
        self.assertIn("feed reserve", portfolio_summary(portfolio))

    def test_owned_seeds_can_be_sown_when_cash_is_tight(self) -> None:
        tiles = _tiles()
        spots = [(x, 0) for x in range(5)]
        _open(tiles, *spots)
        observation = _priced(_observation(tiles, day=0, money=50), MELON=400, WHEAT=1)
        observation["private"]["seeds"] = {"MELON": 5}
        grid = build_task_grid(parse_world(observation), observation=observation)
        names = [grid[x][0].production_plan.name for x, _y in spots]

        self.assertEqual(names, ["MELON"] * 5)
        self.assertEqual(grid.portfolio.startup_spend, 0)
        self.assertEqual(grid[spots[0][0]][0].production_plan.startup_cash, 0)

    def test_an_owned_sheep_is_not_placed_without_feed(self) -> None:
        tiles = _tiles()
        _open(tiles, (4, 4))
        observation = _priced(
            _observation(tiles, day=0, money=50, shed={"SHEEP": 1}),
            WOOL=500,
            WHEAT=25,
            MELON=1,
        )
        sheep = next(item for item in production_candidates(observation, (4, 4)) if item.name == "SHEEP")

        self.assertEqual(sheep.startup_cash, 0)
        self.assertGreater(sheep.required_cash, 0)
        self.assertFalse(sheep.affordable)
        self.assertIsNone(choose_production_plan(observation, (4, 4)))

    def test_a_line_that_would_spend_the_cash_pad_is_refused(self) -> None:
        tiles = _tiles()
        _open(tiles, (4, 4))
        tight = _priced(
            _observation(tiles, day=0, money=699, shed={"WHEAT": 7}),
            WOOL=500,
            WHEAT=1,
            MELON=1,
        )
        safe = _priced(
            _observation(tiles, day=0, money=700, shed={"WHEAT": 7}),
            WOOL=500,
            WHEAT=1,
            MELON=1,
        )
        tight_sheep = next(item for item in production_candidates(tight, (4, 4)) if item.name == "SHEEP")
        safe_sheep = next(item for item in production_candidates(safe, (4, 4)) if item.name == "SHEEP")
        free_seed = _priced(_observation(tiles, day=0, money=0), MELON=400, WHEAT=1)
        free_seed["private"]["seeds"] = {"MELON": 1}
        melon = next(item for item in production_candidates(free_seed, (4, 4)) if item.name == "MELON")

        self.assertFalse(tight_sheep.affordable)
        self.assertTrue(safe_sheep.affordable)
        self.assertEqual(melon.startup_cash, 0)
        self.assertEqual(melon.required_cash, 0)
        self.assertTrue(melon.affordable)

    def test_the_same_morning_commits_the_same_portfolio(self) -> None:
        tiles = _tiles()
        for y in range(5):
            for x in range(5):
                tiles[y][x] = None
        observation = _observation(tiles, day=0, money=3000)

        def committed():
            grid = build_task_grid(parse_world(observation), observation=observation)
            plans = tuple(
                sorted(
                    (x, y, grid[x][y].production_plan.name)
                    for x in range(5)
                    for y in range(5)
                    if grid[x][y] is not None and grid[x][y].production_plan is not None
                )
            )
            portfolio = grid.portfolio
            return (
                plans,
                portfolio.crops,
                portfolio.animals,
                portfolio.startup_spend,
                portfolio.reserved_feed_cash,
                portfolio.reserved_hire_cash,
                portfolio.cash_remaining,
            )

        self.assertEqual(committed(), committed())


if __name__ == "__main__":
    unittest.main()
