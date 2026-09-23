"""The day route reads parsed land, crop, animal, inventory, and worker state."""

from __future__ import annotations

import unittest

from kaggle_environments.envs.kaggriculture.kaggriculture import market_price

from lab.route14_phase1 import make_route14_phase1_agent
from lab.route14_state import (
    COMPLETED,
    DIES_TONIGHT,
    ESCAPED,
    ESCAPES_TONIGHT,
    FALLOW,
    PENDING,
    RIPE,
    SCHEDULED,
    FieldTaskState,
    TaskAssignment,
    parse_world,
    settle_tasks,
)
from lab.test_route14_phase1 import _animal, _observation, _plant, _tiles


class Route14StateTests(unittest.TestCase):
    def test_empty_land_weed_and_coop_are_separate_states(self) -> None:
        tiles = _tiles()
        tiles[0][0] = None
        tiles[1][1] = {"kind": "WEED"}
        tiles[2][2] = {"kind": "COOP"}
        tiles[3][3] = {"kind": "PASTURE", "animal": "SHEEP", "placed_day": 0, "yield_units": 0, "consecutive_unfed": 0, "fed_today": False}
        world = parse_world(_observation(tiles, day=1))
        empty = next(land for land in world.farm.lands if land.position == (0, 0))
        weed = next(land for land in world.farm.lands if land.position == (1, 1))
        coop = next(building for building in world.farm.buildings if building.position == (2, 2))
        pasture = next(building for building in world.farm.buildings if building.position == (3, 3))
        self.assertTrue(empty.empty and empty.plantable and empty.can_build_coop and empty.can_build_pasture)
        self.assertEqual(empty.door_distance, 8)
        self.assertFalse(any(land.position == (9, 9) for land in world.farm.lands))
        self.assertTrue(weed.weed and weed.needs_dig and not weed.plantable)
        self.assertTrue(coop.empty and coop.can_dig)
        self.assertEqual(coop.accepted_animals, ("GOOSE",))
        self.assertFalse(pasture.empty or pasture.can_dig)
        self.assertEqual(pasture.accepted_animals, ("COW", "SHEEP"))

    def test_crop_and_animal_countdowns_follow_the_engine(self) -> None:
        tiles = _tiles()
        tiles[0][0] = _plant("WHEAT", dry=1)
        tiles[0][1] = _plant("WHEAT", dry=0, watered=True)
        tiles[0][2] = _plant("MELON", units=6, dry=1)
        tiles[0][2]["max_lifespan_step"] = 312
        tiles[1][0] = _plant("TOMATO", planted_day=0, units=1)
        tiles[2][0] = _animal("GOOSE", unfed=1)
        tiles[2][1] = _animal("SHEEP", unfed=0)
        tiles[2][2] = _animal("COW", unfed=0, fed=True)
        world = parse_world(_observation(tiles, day=10, hour=0))
        wheat = next(crop for crop in world.farm.crops if crop.position == (0, 0))
        watered = next(crop for crop in world.farm.crops if crop.position == (1, 0))
        melon = next(crop for crop in world.farm.crops if crop.position == (2, 0))
        tomato = next(crop for crop in world.farm.crops if crop.position == (0, 1))
        goose = next(animal for animal in world.farm.animals if animal.animal == "GOOSE")
        sheep = next(animal for animal in world.farm.animals if animal.animal == "SHEEP")
        cow = next(animal for animal in world.farm.animals if animal.animal == "COW")
        self.assertTrue(wheat.must_water)
        self.assertEqual(wheat.weed_countdown_days, 0)
        self.assertEqual(wheat.water_count_today, 0)
        self.assertEqual(watered.weed_countdown_days, 2)
        self.assertEqual(watered.water_count_today, 1)
        self.assertFalse(watered.must_water)
        self.assertTrue(melon.mature and melon.must_water)
        self.assertEqual(melon.harvest_countdown_days, 0)
        self.assertEqual(melon.remaining_harvests, 1)
        self.assertEqual(melon.rot_countdown_steps, 72)
        self.assertEqual(melon.next_price, 250)
        self.assertEqual(melon.next_revenue, 1500)
        self.assertEqual(tomato.remaining_harvests, 2)
        self.assertEqual(goose.escape_countdown_days, 0)
        self.assertTrue(goose.must_feed)
        self.assertEqual(goose.wheat_per_day, 1)
        self.assertEqual(goose.days_until_next_production, 1)
        self.assertEqual(sheep.escape_countdown_days, 1)
        self.assertFalse(sheep.must_feed)
        self.assertEqual(cow.escape_countdown_days, 2)
        self.assertEqual(cow.feed_count_today, 1)

    def test_inventory_reserves_feed_and_shop_countdown_is_in_hours(self) -> None:
        tiles = _tiles()
        tiles[0][0] = _animal("SHEEP", unfed=1)
        observation = _observation(
            tiles,
            day=2,
            hour=10,
            shed={"WHEAT": 1, "MELON": 2, "GOOSE": 1},
            inventories=[{"MELON": 6}],
            shops=["YARN_STORE", "YARN_STORE"],
        )
        world = parse_world(observation)
        inventory = world.farm.inventory
        assert inventory is not None
        self.assertEqual(inventory.capacity, 100)
        self.assertEqual(inventory.free_space, 96)
        self.assertEqual(inventory.wheat_reserved, 1)
        self.assertEqual(inventory.wheat_sellable, 0)
        self.assertEqual(inventory.unplaced_animals, {"GOOSE": 1})
        self.assertEqual(inventory.carried, {"MELON": 6})
        self.assertEqual(world.market.town_inventory["MELON"], 10000)
        self.assertEqual(world.market.shop_counts, {"YARN_STORE": 2})
        self.assertEqual(world.market.hours_until_shop_consume, 2)
        self.assertEqual(world.market.hours_until_center_consume, 14)
        self.assertEqual(world.market.hours_until_next_shop, 14)
        wool = world.market.price_forecast["WOOL"]
        self.assertEqual(wool[10], wool[12])
        self.assertEqual(wool[13], market_price("WOOL", 10000 - 4))
        self.assertEqual(world.market.price_forecast["MELON"][10], 250)
        self.assertEqual(world.market.price_forecast["MELON"][23], 250)
        self.assertEqual(world.hours_remaining, 30 * 24 - (2 * 24 + 10))
        self.assertEqual(world.farm.workers[0].carrying, {"MELON": 6})
        self.assertEqual(world.farm.workers[0].role, "farmer")
        self.assertIsNone(world.opponent.inventory if world.opponent else None)
        full = _observation(tiles, shops=["BAKERY"] * 8)
        self.assertIsNone(parse_world(full).market.hours_until_next_shop)

    def test_opponent_crops_are_public_and_the_route_is_written_onto_the_crop(self) -> None:
        tiles = _tiles()
        tiles[3][4] = _plant("MELON", units=6)
        rival = _tiles()
        rival[6][0] = _plant("MELON", units=30)
        observation = _observation(tiles, rival=rival, rival_farmer=(0, 6))
        world = parse_world(observation)
        assert world.opponent is not None
        self.assertEqual(world.opponent.crops[0].position, (0, 6))
        self.assertEqual(world.opponent.crops[0].yield_units, 30)
        self.assertIsNone(world.opponent.inventory)
        agent = make_route14_phase1_agent()
        agent(observation)
        planned = agent.telemetry["world"]
        melon = planned.farm.crops[0]
        worker = planned.farm.workers[0]
        self.assertTrue(melon.must_harvest)
        self.assertEqual(melon.status, SCHEDULED)
        self.assertEqual(melon.harvest_status, SCHEDULED)
        self.assertEqual(melon.assigned_worker, "Farmer")
        self.assertFalse(melon.harvest_completed)
        self.assertEqual(melon.planned_harvest_hour, 1)
        self.assertEqual(melon.planned_drop_hour, 3)
        self.assertEqual(melon.planned_sell_hour, 3)
        self.assertEqual(melon.next_revenue, melon.next_price * 6)
        self.assertEqual(worker.current_target, (4, 3))
        self.assertEqual(worker.remaining[0].operation, ("HARVEST",))
        self.assertEqual(worker.route[-1].operation, ("PLACE", "MELON", 6))
        harvests = [task for task in planned.farm.task_pool if task.kind == "HARVEST"]
        self.assertEqual(len(harvests), 1)
        self.assertEqual(harvests[0].status, SCHEDULED)
        self.assertEqual(harvests[0].assigned_worker, "Farmer")
        self.assertEqual(harvests[0].reason, RIPE)

    def test_issuing_harvest_does_not_finish_the_melon(self) -> None:
        tiles = _tiles()
        tiles[3][4] = _plant("MELON", units=6)
        observation = _observation(tiles)
        before = parse_world(observation).farm.crops[0]
        self.assertTrue(before.must_harvest)
        self.assertEqual(before.status, PENDING)
        self.assertFalse(before.harvest_completed)
        agent = make_route14_phase1_agent()
        agent(observation)
        observation["hour"] = 1
        observation["step"] = 10 * 24 + 1
        observation["farms"][0]["farmer"] = [4, 3]
        decision = agent(observation)
        self.assertEqual(decision["farmer"], ["HARVEST"])
        still = agent.telemetry["world"].farm.crops[0]
        self.assertEqual(still.status, SCHEDULED)
        self.assertEqual(still.assigned_worker, "Farmer")
        self.assertFalse(still.harvest_completed)
        tiles[3][4] = None
        observation["hour"] = 2
        observation["step"] = 10 * 24 + 2
        agent(observation)
        done = next(task for task in agent.telemetry["world"].farm.tasks if task.kind == "HARVEST")
        self.assertEqual(done.status, COMPLETED)
        self.assertTrue(done.harvest_completed)
        self.assertEqual(done.assigned_worker, "Farmer")
        self.assertEqual((done.planned_hour, done.planned_drop_hour, done.planned_sell_hour), (1, 3, 3))
        self.assertEqual(agent.telemetry["world"].farm.crops, [])
        bare = next(land for land in agent.telemetry["world"].farm.lands if land.position == (4, 3))
        self.assertTrue(bare.empty and bare.fallow and bare.plantable)
        self.assertEqual(bare.status, FALLOW)

    def test_water_and_feed_stay_scheduled_until_the_board_says_so(self) -> None:
        tiles = _tiles()
        tiles[0][0] = _plant("WHEAT", dry=1)
        observation = _observation(tiles, day=10, hour=0)
        world = parse_world(observation)
        wheat = world.farm.crops[0]
        self.assertTrue(wheat.must_water)
        self.assertEqual(wheat.status, PENDING)
        settle_tasks(world.farm, [TaskAssignment("WATER", (0, 0), "Hand2", 6)], [])
        self.assertEqual(wheat.status, SCHEDULED)
        self.assertEqual(wheat.water_status, SCHEDULED)
        self.assertEqual(wheat.assigned_worker, "Hand2")
        self.assertEqual(wheat.planned_water_hour, 6)
        self.assertFalse(wheat.watered_today)
        tiles[0][0]["watered_today"] = True
        tiles[0][0]["consecutive_unwatered"] = 0
        watered = parse_world(observation)
        settle_tasks(watered.farm, [TaskAssignment("WATER", (0, 0), "Hand2", 6)], world.farm.tasks)
        done = watered.farm.crops[0]
        self.assertTrue(done.watered_today)
        self.assertFalse(done.must_water)
        self.assertEqual(done.status, COMPLETED)
        self.assertEqual(done.water_status, COMPLETED)
        self.assertEqual(done.planned_water_hour, 6)

        tiles = _tiles()
        tiles[0][0] = _animal("SHEEP", unfed=1)
        observation = _observation(tiles, day=10, hour=0)
        world = parse_world(observation)
        sheep = world.farm.animals[0]
        self.assertTrue(sheep.must_feed)
        self.assertEqual(sheep.status, PENDING)
        settle_tasks(world.farm, [TaskAssignment("FEED", (0, 0), "Hand1", 8)], [])
        self.assertEqual(sheep.status, SCHEDULED)
        self.assertEqual(sheep.feed_status, SCHEDULED)
        self.assertEqual(sheep.assigned_worker, "Hand1")
        self.assertEqual(sheep.planned_feed_hour, 8)
        self.assertFalse(sheep.fed_today)
        tiles[0][0]["fed_today"] = True
        tiles[0][0]["consecutive_unfed"] = 0
        fed = parse_world(observation)
        settle_tasks(fed.farm, [TaskAssignment("FEED", (0, 0), "Hand1", 8)], world.farm.tasks)
        done_sheep = fed.farm.animals[0]
        self.assertTrue(done_sheep.fed_today)
        self.assertFalse(done_sheep.must_feed)
        self.assertEqual(done_sheep.status, COMPLETED)
        self.assertEqual(done_sheep.feed_status, COMPLETED)
        self.assertEqual(done_sheep.planned_feed_hour, 8)

    def test_task_pool_keeps_one_job_for_each_urgent_need(self) -> None:
        tiles = _tiles()
        tiles[0][0] = _plant("WHEAT", dry=1)
        tiles[0][1] = _plant("WHEAT", dry=0)
        tiles[0][2] = _plant("MELON", units=6)
        tiles[1][0] = _animal("SHEEP", unfed=1)
        tiles[1][1] = _animal("COW", unfed=0, units=2)
        world = parse_world(_observation(tiles, day=10, hour=0))
        pool = world.farm.task_pool
        reasons = {(task.kind, task.position): (task.reason, task.status) for task in pool}
        self.assertEqual(reasons[( "WATER", (0, 0))], (DIES_TONIGHT, PENDING))
        self.assertEqual(reasons[("HARVEST", (2, 0))], (RIPE, PENDING))
        self.assertEqual(reasons[("FEED", (0, 1))], (ESCAPES_TONIGHT, PENDING))
        self.assertEqual(reasons[("HARVEST", (1, 1))], (RIPE, PENDING))
        self.assertNotIn(("WATER", (1, 0)), reasons)
        self.assertNotIn(("FEED", (1, 1)), reasons)
        self.assertEqual(len(pool), len({(task.kind, task.position) for task in pool}))
        self.assertEqual(next(task for task in pool if task.kind == "FEED").reason, ESCAPES_TONIGHT)

    def test_each_tile_is_its_own_row_in_the_pool(self) -> None:
        tiles = _tiles()
        tiles[3][2] = _plant("STRAWBERRY", dry=1)
        tiles[4][2] = _plant("STRAWBERRY", dry=1)
        tiles[5][2] = _plant("WHEAT", units=4)
        world = parse_world(_observation(tiles, day=10, hour=0))
        self.assertEqual(
            [task.line for task in world.farm.task_pool],
            [
                "(2, 3) STRAWBERRY WATER",
                "(2, 4) STRAWBERRY WATER",
                "(2, 5) WHEAT HARVEST",
            ],
        )

    def test_a_weed_is_not_a_completed_harvest(self) -> None:
        tiles = _tiles()
        tiles[3][4] = {"kind": "WEED"}
        world = parse_world(_observation(tiles))
        previous = [
            FieldTaskState(
                "HARVEST",
                (4, 3),
                SCHEDULED,
                "MELON",
                assigned_worker="Farmer",
                planned_hour=1,
                planned_drop_hour=3,
                planned_sell_hour=3,
                units=6,
            )
        ]
        settle_tasks(world.farm, [], previous)
        self.assertFalse(any(task.status == COMPLETED for task in world.farm.tasks))

    def test_land_becomes_a_crop_or_a_shed_then_an_animal(self) -> None:
        tiles = _tiles()
        tiles[3][2] = None
        tiles[1][1] = None
        tiles[2][2] = {"kind": "WEED"}
        observation = _observation(tiles, day=10, hour=0)
        world = parse_world(observation)
        ground = next(land for land in world.farm.lands if land.position == (2, 3))
        pasture_site = next(land for land in world.farm.lands if land.position == (1, 1))
        weed = next(land for land in world.farm.lands if land.position == (2, 2))
        self.assertTrue(ground.empty and ground.plantable and not ground.reserved)
        self.assertIsNone(ground.status)

        settle_tasks(
            world.farm,
            [
                TaskAssignment("PLANT", (2, 3), "Hand1", 6, subject="WHEAT"),
                TaskAssignment("BUILD_PASTURE", (1, 1), "Hand2", 4, subject="PASTURE"),
                TaskAssignment("DIG", (2, 2), "Farmer", 2),
            ],
            [],
        )
        self.assertTrue(ground.reserved and pasture_site.reserved and weed.reserved)
        self.assertEqual(ground.status, SCHEDULED)
        self.assertEqual(ground.planned_change, "WHEAT")
        self.assertEqual(ground.planned_hour, 6)
        self.assertEqual(pasture_site.planned_change, "PASTURE")
        self.assertEqual(weed.planned_change, "DIG")
        self.assertFalse(world.farm.crops or world.farm.buildings)

        tiles[3][2] = _plant("WHEAT", planted_day=10)
        tiles[1][1] = {"kind": "PASTURE"}
        tiles[2][2] = None
        observation["hour"] = 7
        observation["step"] = 10 * 24 + 7
        changed = parse_world(observation)
        settle_tasks(changed.farm, [], world.farm.tasks)
        self.assertFalse(any(land.position == (2, 3) for land in changed.farm.lands))
        wheat = next(crop for crop in changed.farm.crops if crop.position == (2, 3))
        self.assertEqual(wheat.crop, "WHEAT")
        self.assertGreater(wheat.harvest_countdown_days, 0)
        pasture = next(building for building in changed.farm.buildings if building.position == (1, 1))
        self.assertEqual(pasture.structure, "PASTURE")
        self.assertTrue(pasture.empty and not pasture.reserved)
        dug = next(land for land in changed.farm.lands if land.position == (2, 2))
        self.assertTrue(dug.empty and dug.plantable and not dug.fallow)
        self.assertEqual({task.kind for task in changed.farm.tasks if task.status == COMPLETED}, {"PLANT", "BUILD_PASTURE", "DIG"})

        settle_tasks(changed.farm, [TaskAssignment("PLACE_ANIMAL", (1, 1), "Hand2", 9, subject="SHEEP")], changed.farm.tasks)
        self.assertTrue(pasture.reserved)
        self.assertEqual(pasture.status, SCHEDULED)
        self.assertEqual(pasture.planned_animal, "SHEEP")
        self.assertFalse(changed.farm.animals)

        tiles[1][1] = _animal("SHEEP")
        observation["hour"] = 10
        observation["step"] = 10 * 24 + 10
        stocked = parse_world(observation)
        settle_tasks(stocked.farm, [], changed.farm.tasks)
        sheep = stocked.farm.animals[0]
        self.assertEqual((sheep.animal, sheep.position, sheep.product), ("SHEEP", (1, 1), "WOOL"))
        held = next(building for building in stocked.farm.buildings if building.position == (1, 1))
        self.assertFalse(held.empty or held.reserved)
        place = next(task for task in stocked.farm.tasks if task.kind == "PLACE_ANIMAL")
        self.assertEqual(place.status, COMPLETED)

        tiles[1][1] = {"kind": "PASTURE"}
        tiles[3][2] = None
        observation["hour"] = 21
        observation["step"] = 10 * 24 + 21
        previous = [
            *stocked.farm.tasks,
            FieldTaskState("HARVEST", (2, 3), SCHEDULED, "WHEAT", assigned_worker="Hand1", planned_hour=20, units=6),
        ]
        cleared = parse_world(observation)
        settle_tasks(cleared.farm, [], previous, animals=[sheep.position])
        bare = next(land for land in cleared.farm.lands if land.position == (2, 3))
        self.assertTrue(bare.fallow and bare.plantable and not bare.reserved)
        self.assertEqual(bare.status, FALLOW)
        empty_pasture = next(building for building in cleared.farm.buildings if building.position == (1, 1))
        self.assertTrue(empty_pasture.empty and empty_pasture.escaped)
        self.assertEqual(empty_pasture.status, ESCAPED)
        self.assertFalse(cleared.farm.crops or cleared.farm.animals)

        tiles[3][2] = {"kind": "WEED"}
        observation["hour"] = 22
        observation["step"] = 10 * 24 + 22
        overgrown = parse_world(observation)
        settle_tasks(overgrown.farm, [], [], fallow=cleared.farm.fallow_positions, escaped=cleared.farm.escaped_positions)
        weeded = next(land for land in overgrown.farm.lands if land.position == (2, 3))
        self.assertTrue(weeded.weed and not weeded.fallow)
        self.assertNotIn((2, 3), overgrown.farm.fallow_positions)


if __name__ == "__main__":
    unittest.main()
