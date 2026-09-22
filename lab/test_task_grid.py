"""TaskGrid records watering from parsed crop state, one cell at a time."""

from __future__ import annotations

import unittest

from kaggle_environments.envs.kaggriculture.kaggriculture import _apply_unit_action, _daily_refresh_plants

from lab.route14_state import COMPLETED, PENDING, SCHEDULED, parse_world
from lab.task_grid import (
    CARE,
    COLLECT_FERTILIZER,
    FEED,
    FERTILIZE,
    HARVEST,
    WATER,
    TaskGridBuilder,
    should_fertilize,
    water_yield_gain,
)
from lab.test_route14_phase1 import _animal, _observation, _plant, _tiles


def _world(
    crop: str,
    *,
    day: int,
    hour: int = 0,
    planted_day: int = 0,
    units: int = 0,
    dry: int = 0,
    watered: bool = False,
    fertilized_until: int = -1,
    shed: dict[str, int] | None = None,
):
    tiles = _tiles()
    plant = _plant(crop, planted_day=planted_day, units=units, dry=dry, watered=watered)
    plant["fertilized_until_day"] = fertilized_until
    tiles[3][2] = plant
    return parse_world(_observation(tiles, day=day, hour=hour, shed=shed))


def _engine_oneshot_gain(crop: str, day: int, planted_day: int, units: int, fertilized_until: int) -> int:
    farm = {
        "farmer": [0, 0],
        "hands": [],
        "tiles": [[
            {
                "kind": "PLANT",
                "crop": crop,
                "planted_day": planted_day,
                "yield_units": units,
                "watered_today": False,
                "fertilized_until_day": fertilized_until,
                "consecutive_unwatered": 0,
            }
        ]],
    }
    private = {"inventories": [{}]}
    _apply_unit_action(farm, private, 0, ["WATER"], 1, day, 24)
    return farm["tiles"][0][0]["yield_units"] - units


def _engine_ongoing_gain(crop: str, day: int, planted_day: int, units: int, fertilized_until: int) -> int:
    def after(watered: bool) -> int:
        farm = {
            "tiles": [[
                {
                    "kind": "PLANT",
                    "crop": crop,
                    "planted_day": planted_day,
                    "yield_units": units,
                    "watered_today": watered,
                    "fertilized_until_day": fertilized_until,
                    "consecutive_unwatered": 0,
                    "max_lifespan_step": -1,
                }
            ]]
        }
        _daily_refresh_plants(farm, day, 24)
        tile = farm["tiles"][0][0]
        if not isinstance(tile, dict) or "yield_units" not in tile:
            return 0
        return tile["yield_units"] - units

    return after(True) - after(False)


class TaskGridTests(unittest.TestCase):
    def test_watered_today_has_no_water_task(self) -> None:
        world = _world("WHEAT", day=2, units=1, watered=True)
        grid = TaskGridBuilder().build(world)
        self.assertIsNone(grid[0][0])
        self.assertNotIn(WATER, grid[2][3].tasks)

    def test_dying_tonight_is_mandatory(self) -> None:
        world = _world("WHEAT", day=0, units=1, dry=1, hour=5)
        water = TaskGridBuilder().build(world)[2][3].tasks[WATER]
        self.assertEqual(water.status, PENDING)
        self.assertTrue(water.needed)
        self.assertTrue(water.mandatory)
        self.assertEqual(water.turns_until_weed, 19)
        self.assertEqual(water.yield_gain, 0)
        self.assertIsNone(water.assigned_worker)

    def test_yield_only_water_is_not_mandatory(self) -> None:
        world = _world("WHEAT", day=2, units=1, dry=0, hour=5)
        water = TaskGridBuilder().build(world)[2][3].tasks[WATER]
        self.assertTrue(water.needed)
        self.assertFalse(water.mandatory)
        self.assertEqual(water.yield_gain, 1)
        self.assertEqual(water.turns_until_weed, 43)

    def test_water_adds_one_without_fertilizer(self) -> None:
        world = _world("WHEAT", day=2, units=1, dry=0)
        crop = world.farm.crops[0]
        self.assertEqual(water_yield_gain(crop), 1)
        self.assertEqual(water_yield_gain(crop), _engine_oneshot_gain("WHEAT", 2, 0, 1, -1))
        self.assertEqual(TaskGridBuilder().build(world)[2][3].tasks[WATER].yield_gain, 1)

    def test_water_adds_two_with_fertilizer(self) -> None:
        world = _world("WHEAT", day=2, units=1, dry=0, fertilized_until=2)
        crop = world.farm.crops[0]
        self.assertEqual(water_yield_gain(crop), 2)
        self.assertEqual(water_yield_gain(crop), _engine_oneshot_gain("WHEAT", 2, 0, 1, 2))
        self.assertEqual(TaskGridBuilder().build(world)[2][3].tasks[WATER].yield_gain, 2)

    def test_water_adds_nothing_at_max_yield(self) -> None:
        world = _world("WHEAT", day=4, units=6, dry=1, fertilized_until=4)
        water = TaskGridBuilder().build(world)[2][3].tasks[WATER]
        self.assertEqual(water.yield_gain, 0)
        self.assertEqual(water.yield_gain, _engine_oneshot_gain("WHEAT", 4, 0, 6, 4))
        self.assertTrue(water.mandatory)

    def test_one_cell_holds_water_and_harvest(self) -> None:
        tiles = _tiles()
        tiles[3][2] = _plant("WHEAT", units=4, dry=1)
        tiles[4][2] = None
        world = parse_world(_observation(tiles, day=2, hour=0))
        bucket = TaskGridBuilder().build(world)[2][3]
        self.assertEqual(bucket.tile_type, "WHEAT")
        self.assertEqual(bucket.coord, (2, 3))
        self.assertEqual(set(bucket.tasks), {WATER, HARVEST})
        self.assertEqual(bucket.tasks[HARVEST].yield_amount, 4)
        self.assertIsNone(TaskGridBuilder().build(world)[0][0])
        self.assertEqual(TaskGridBuilder().build(world)[2][4].tile_type, "EMPTY")
        self.assertEqual(TaskGridBuilder().build(world)[2][4].tasks, {})

    def test_unripe_crop_has_no_harvest(self) -> None:
        world = _world("WHEAT", day=0, units=1, dry=1)
        self.assertNotIn(HARVEST, TaskGridBuilder().build(world)[2][3].tasks)

    def test_ripe_harvest_completes_only_after_the_tile_is_empty(self) -> None:
        tiles = _tiles()
        tiles[3][2] = _plant("MELON", units=6)
        hour1 = parse_world(_observation(tiles, day=10, hour=1))
        grid = TaskGridBuilder().build(hour1)
        harvest = grid[2][3].tasks[HARVEST]
        self.assertEqual(harvest.status, PENDING)
        self.assertEqual(harvest.yield_amount, 6)
        grid.schedule(2, 3, HARVEST, "Farmer", 1)
        still_there = TaskGridBuilder().build(hour1, grid)
        self.assertEqual(still_there[2][3].tasks[HARVEST].status, SCHEDULED)
        self.assertEqual(still_there[2][3].tasks[HARVEST].assigned_worker, "Farmer")
        self.assertEqual(still_there[2][3].tasks[HARVEST].planned_hour, 1)
        tiles[3][2] = {"kind": "WEED"}
        weeded = TaskGridBuilder().build(parse_world(_observation(tiles, day=10, hour=2)), still_there)
        self.assertNotIn(HARVEST, weeded[2][3].tasks)
        tiles[3][2] = None
        done = TaskGridBuilder().build(parse_world(_observation(tiles, day=10, hour=2)), still_there)
        self.assertEqual(done[2][3].tasks[HARVEST].status, COMPLETED)
        self.assertEqual(done[2][3].tasks[HARVEST].yield_amount, 0)

    def test_ongoing_and_animal_harvests_finish_when_the_yield_is_gone(self) -> None:
        tiles = _tiles()
        tiles[3][2] = _plant("TOMATO", planted_day=0, units=1)
        tiles[1][1] = _animal("SHEEP", units=2)
        ripe = parse_world(_observation(tiles, day=10, hour=0))
        grid = TaskGridBuilder().build(ripe)
        self.assertEqual(grid[2][3].tasks[HARVEST].yield_amount, 1)
        self.assertEqual(grid[1][1].tasks[HARVEST].yield_amount, 2)
        self.assertEqual(grid[1][1].tile_type, "SHEEP")
        grid.schedule(2, 3, HARVEST, "Farmer", 2)
        grid.schedule(1, 1, HARVEST, "Hand1", 3)
        tiles[3][2]["yield_units"] = 0
        tiles[1][1]["yield_units"] = 0
        done = TaskGridBuilder().build(parse_world(_observation(tiles, day=10, hour=4)), grid)
        self.assertEqual(done[2][3].tile_type, "TOMATO")
        self.assertEqual(done[2][3].tasks[HARVEST].status, COMPLETED)
        self.assertEqual(done[1][1].tasks[HARVEST].status, COMPLETED)
        self.assertEqual(done[1][1].tasks[HARVEST].assigned_worker, "Hand1")

    def test_scheduled_water_completes_only_after_the_board_says_so(self) -> None:
        tiles = _tiles()
        tiles[3][2] = _plant("WHEAT", dry=1)
        hour5 = parse_world(_observation(tiles, day=1, hour=5))
        grid = TaskGridBuilder().build(hour5)
        grid.schedule(2, 3, WATER, "Hand1", 5)
        self.assertEqual(grid[2][3].tasks[WATER].status, SCHEDULED)
        still_dry = TaskGridBuilder().build(hour5, grid)
        self.assertEqual(still_dry[2][3].tasks[WATER].status, SCHEDULED)
        self.assertEqual(still_dry[2][3].tasks[WATER].assigned_worker, "Hand1")
        tiles[3][2]["watered_today"] = True
        hour6 = parse_world(_observation(tiles, day=1, hour=6))
        done = TaskGridBuilder().build(hour6, still_dry)
        self.assertEqual(done[2][3].tasks[WATER].status, COMPLETED)
        self.assertFalse(done[2][3].tasks[WATER].needed)

    def test_tomato_and_strawberry_gain_only_on_a_fertilized_production_night(self) -> None:
        tomato = _world("TOMATO", day=7, units=0, dry=0, fertilized_until=7)
        self.assertEqual(water_yield_gain(tomato.farm.crops[0]), 1)
        self.assertEqual(water_yield_gain(tomato.farm.crops[0]), _engine_ongoing_gain("TOMATO", 7, 0, 0, 7))
        plain = _world("TOMATO", day=7, units=0, dry=1)
        self.assertEqual(TaskGridBuilder().build(plain)[2][3].tasks[WATER].yield_gain, 0)
        self.assertEqual(water_yield_gain(plain.farm.crops[0]), _engine_ongoing_gain("TOMATO", 7, 0, 0, -1))
        off_night = _world("STRAWBERRY", day=10, units=0, dry=1, fertilized_until=10)
        self.assertEqual(TaskGridBuilder().build(off_night)[2][3].tasks[WATER].yield_gain, 0)
        berry = _world("STRAWBERRY", day=9, units=0, dry=0, fertilized_until=9)
        self.assertEqual(water_yield_gain(berry.farm.crops[0]), 1)
        self.assertEqual(water_yield_gain(berry.farm.crops[0]), _engine_ongoing_gain("STRAWBERRY", 9, 0, 0, 9))

    def test_animal_tile_has_feed_and_care_and_a_crop_tile_does_not(self) -> None:
        tiles = _tiles()
        tiles[3][2] = _plant("WHEAT", units=1, dry=1)
        tiles[1][1] = _animal("SHEEP", unfed=1)
        tiles[1][1]["pending_care_bonus"] = 2
        tiles[2][2] = {"kind": "COOP"}
        world = parse_world(_observation(tiles, day=3, hour=4))
        crop = TaskGridBuilder().build(world)[2][3]
        sheep = TaskGridBuilder().build(world)[1][1]
        coop = TaskGridBuilder().build(world)[2][2]
        self.assertNotIn(FEED, crop.tasks)
        self.assertNotIn(CARE, crop.tasks)
        self.assertNotIn(FEED, coop.tasks)
        self.assertNotIn(CARE, coop.tasks)
        feed = sheep.tasks[FEED]
        care = sheep.tasks[CARE]
        self.assertEqual(feed.status, PENDING)
        self.assertFalse(feed.fed_today)
        self.assertEqual(feed.consecutive_unfed, 1)
        self.assertEqual(feed.wheat_cost, 1)
        self.assertTrue(feed.mandatory)
        self.assertEqual(care.status, PENDING)
        self.assertFalse(care.cared_today)
        self.assertEqual(care.pending_care_bonus, 2)
        self.assertEqual(care.bonus_gain, 1)

    def test_feed_and_care_complete_only_after_the_animal_shows_it(self) -> None:
        tiles = _tiles()
        tiles[1][1] = _animal("SHEEP", unfed=0)
        hour4 = parse_world(_observation(tiles, day=3, hour=4))
        grid = TaskGridBuilder().build(hour4)
        self.assertFalse(grid[1][1].tasks[FEED].mandatory)
        grid.schedule(1, 1, FEED, "Hand1", 6)
        grid.schedule(1, 1, CARE, "Hand2", 7)
        still = TaskGridBuilder().build(hour4, grid)
        self.assertEqual(still[1][1].tasks[FEED].status, SCHEDULED)
        self.assertEqual(still[1][1].tasks[CARE].status, SCHEDULED)
        self.assertFalse(still[1][1].tasks[FEED].fed_today)
        self.assertEqual(still[1][1].tasks[FEED].wheat_cost, 1)
        tiles[1][1]["fed_today"] = True
        fed_only = TaskGridBuilder().build(parse_world(_observation(tiles, day=3, hour=6)), still)
        self.assertEqual(fed_only[1][1].tasks[FEED].status, COMPLETED)
        self.assertTrue(fed_only[1][1].tasks[FEED].fed_today)
        self.assertEqual(fed_only[1][1].tasks[FEED].wheat_cost, 0)
        self.assertEqual(fed_only[1][1].tasks[FEED].assigned_worker, "Hand1")
        self.assertEqual(fed_only[1][1].tasks[CARE].status, SCHEDULED)
        self.assertEqual(fed_only[1][1].tasks[CARE].bonus_gain, 1)
        tiles[1][1]["cared_today"] = True
        both = TaskGridBuilder().build(parse_world(_observation(tiles, day=3, hour=7)), fed_only)
        self.assertEqual(both[1][1].tasks[CARE].status, COMPLETED)
        self.assertTrue(both[1][1].tasks[CARE].cared_today)
        self.assertEqual(both[1][1].tasks[CARE].pending_care_bonus, 0)
        self.assertEqual(both[1][1].tasks[CARE].bonus_gain, 0)

    def test_ready_animal_gets_collect_fertilizer(self) -> None:
        tiles = _tiles()
        tiles[1][1] = _animal("SHEEP")
        tiles[1][1]["fertilizer_available"] = True
        grid = TaskGridBuilder().build(parse_world(_observation(tiles, day=4)))
        task = grid[1][1].tasks[COLLECT_FERTILIZER]
        self.assertEqual(task.status, PENDING)
        self.assertTrue(task.fertilizer_ready)
        self.assertIsNone(task.assigned_worker)

    def test_animal_without_fertilizer_has_no_collect_task(self) -> None:
        tiles = _tiles()
        tiles[1][1] = _animal("SHEEP")
        grid = TaskGridBuilder().build(parse_world(_observation(tiles, day=4)))
        self.assertNotIn(COLLECT_FERTILIZER, grid[1][1].tasks)

    def test_collect_stays_scheduled_while_fertilizer_remains(self) -> None:
        tiles = _tiles()
        tiles[1][1] = _animal("SHEEP")
        tiles[1][1]["fertilizer_available"] = True
        world = parse_world(_observation(tiles, day=4, hour=3))
        grid = TaskGridBuilder().build(world)
        grid.schedule(1, 1, COLLECT_FERTILIZER, "Hand1", 4)
        still = TaskGridBuilder().build(world, grid)
        task = still[1][1].tasks[COLLECT_FERTILIZER]
        self.assertEqual(task.status, SCHEDULED)
        self.assertEqual(task.assigned_worker, "Hand1")
        self.assertEqual(task.planned_hour, 4)
        self.assertTrue(task.fertilizer_ready)

    def test_collect_completes_when_fertilizer_is_gone(self) -> None:
        tiles = _tiles()
        tiles[1][1] = _animal("SHEEP")
        tiles[1][1]["fertilizer_available"] = True
        grid = TaskGridBuilder().build(parse_world(_observation(tiles, day=4, hour=3)))
        grid.schedule(1, 1, COLLECT_FERTILIZER, "Hand1", 4)
        tiles[1][1]["fertilizer_available"] = False
        done = TaskGridBuilder().build(parse_world(_observation(tiles, day=4, hour=4)), grid)
        task = done[1][1].tasks[COLLECT_FERTILIZER]
        self.assertEqual(task.status, COMPLETED)
        self.assertFalse(task.fertilizer_ready)
        self.assertEqual(task.assigned_worker, "Hand1")

    def test_fertilize_when_it_increases_yield(self) -> None:
        world = _world("WHEAT", day=2, units=1, shed={"FERTILIZER": 1})
        crop = world.farm.crops[0]
        self.assertTrue(should_fertilize(crop, world))
        task = TaskGridBuilder().build(world)[2][3].tasks[FERTILIZE]
        self.assertEqual(task.status, PENDING)
        self.assertFalse(task.fertilized_today)
        self.assertEqual(task.fertilizer_days_left, 0)

    def test_no_fertilize_without_fertilizer_on_hand(self) -> None:
        world = _world("WHEAT", day=2, units=1)
        self.assertFalse(should_fertilize(world.farm.crops[0], world))
        self.assertNotIn(FERTILIZE, TaskGridBuilder().build(world)[2][3].tasks)

    def test_no_fertilize_when_no_harvests_remain(self) -> None:
        world = _world("TOMATO", day=12, units=0, shed={"FERTILIZER": 1})
        self.assertEqual(world.farm.crops[0].remaining_harvests, 0)
        self.assertFalse(should_fertilize(world.farm.crops[0], world))
        self.assertNotIn(FERTILIZE, TaskGridBuilder().build(world)[2][3].tasks)

    def test_no_fertilize_when_already_fertilized(self) -> None:
        world = _world("WHEAT", day=2, units=1, fertilized_until=4, shed={"FERTILIZER": 1})
        crop = world.farm.crops[0]
        self.assertGreater(crop.fertilizer_days_left, 0)
        self.assertFalse(should_fertilize(crop, world))
        self.assertNotIn(FERTILIZE, TaskGridBuilder().build(world)[2][3].tasks)

    def test_fertilize_completes_only_after_the_crop_shows_it(self) -> None:
        tiles = _tiles()
        tiles[3][2] = _plant("WHEAT", units=1)
        hour2 = parse_world(_observation(tiles, day=2, hour=2, shed={"FERTILIZER": 1}))
        grid = TaskGridBuilder().build(hour2)
        grid.schedule(2, 3, FERTILIZE, "Farmer", 3)
        still = TaskGridBuilder().build(hour2, grid)
        self.assertEqual(still[2][3].tasks[FERTILIZE].status, SCHEDULED)
        tiles[3][2]["fertilized_until_day"] = 4
        done = TaskGridBuilder().build(parse_world(_observation(tiles, day=2, hour=3, shed={"FERTILIZER": 0})), still)
        task = done[2][3].tasks[FERTILIZE]
        self.assertEqual(task.status, COMPLETED)
        self.assertTrue(task.fertilized_today)
        self.assertEqual(task.assigned_worker, "Farmer")

    def test_fertilize_does_not_replace_water_or_harvest(self) -> None:
        world = _world("WHEAT", day=2, units=1, dry=1, shed={"FERTILIZER": 1})
        tasks = TaskGridBuilder().build(world)[2][3].tasks
        self.assertEqual(set(tasks), {WATER, HARVEST, FERTILIZE})
        self.assertEqual(tasks[HARVEST].yield_amount, 1)
        self.assertTrue(tasks[WATER].mandatory)
        self.assertNotIn(COLLECT_FERTILIZER, tasks)
        self.assertNotIn(FEED, tasks)


if __name__ == "__main__":
    unittest.main()
