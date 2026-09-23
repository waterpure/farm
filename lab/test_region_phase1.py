"""The 5×5 route drives each real hour. It does not guess where a hand is born."""

from __future__ import annotations

import unittest

from kaggle_environments import make

from lab.region_phase1 import make_region_phase1_agent
from lab.route14_phase1 import _reserved_wheat
from lab.route14_state import COMPLETED, SCHEDULED
from lab.test_route14_phase1 import _animal, _observation, _plant, _tiles


def _copy_tiles(tiles: list[list]) -> list[list]:
    return [list(row) for row in tiles]


class RegionPhase1Tests(unittest.TestCase):
    def test_a_hand_is_planned_from_the_coordinate_where_it_actually_appeared(self) -> None:
        tiles = _tiles()
        for spot in ((0, 0), (0, 9), (9, 0), (9, 9)):
            tiles[spot[1]][spot[0]] = _plant("WHEAT", dry=1)
        agent = make_region_phase1_agent()
        morning = _observation(tiles, day=1, hour=0, farmer=(4, 4))
        hired = agent(morning)

        self.assertEqual(hired["farmer"], ["PASS"])
        self.assertIn(["HIRE"], hired["market"])
        self.assertIsNone(agent.telemetry["plan"])

        appeared = _observation(tiles, day=1, hour=1, farmer=(4, 4), hands=[(1, 2)])
        agent(appeared)

        self.assertIn(("Hand1", (1, 2)), agent.telemetry["planned_workers"])
        self.assertNotIn(("Hand1", (4, 3)), agent.telemetry["planned_workers"])
        self.assertNotIn(("Hand1", (5, 4)), agent.telemetry["planned_workers"])

    def test_each_hour_plays_the_expanded_route_without_searching_again(self) -> None:
        tiles = _tiles()
        tiles[0][1] = _plant("TOMATO", units=1, dry=1)
        agent = make_region_phase1_agent()
        position = (0, 0)
        played: list[str] = []
        for hour in (1, 2, 3):
            observation = _observation(_copy_tiles(tiles), day=8, hour=hour, farmer=position)
            action = agent(observation)
            played.append(action["farmer"][0])
            self.assertFalse(agent.telemetry["needs_replan"])
            if action["farmer"][0] == "EAST":
                position = (position[0] + 1, position[1])
        self.assertEqual(played, ["EAST", "WATER", "HARVEST"])
        operations = [item.operation for item in agent.telemetry["plan"].worker_routes[0].actions_by_hour]
        self.assertEqual(operations[:3], ["EAST", "WATER", "HARVEST"])

    def test_two_workers_act_from_their_own_routes(self) -> None:
        tiles = _tiles()
        tiles[4][4] = _plant("WHEAT", planted_day=-2, units=6)
        tiles[5][5] = _plant("WHEAT", planted_day=-2, units=6)
        observation = _observation(tiles, day=0, hour=1, farmer=(4, 4), hands=[(5, 5)])
        action = make_region_phase1_agent()(observation)

        self.assertEqual(action["farmer"], ["HARVEST"])
        self.assertEqual(action["hands"], [["HARVEST"]])

    def test_an_idle_worker_passes(self) -> None:
        tiles = _tiles()
        tiles[0][0] = _plant("WHEAT", planted_day=-2, units=6)
        observation = _observation(tiles, day=0, hour=1, farmer=(0, 0), hands=[(4, 0)])
        action = make_region_phase1_agent()(observation)

        self.assertEqual(action["farmer"], ["HARVEST"])
        self.assertEqual(action["hands"], [["PASS"]])

    def test_planning_schedules_the_job_and_the_next_board_completes_it(self) -> None:
        tiles = _tiles()
        tiles[0][0] = _plant("WHEAT", planted_day=-2, units=6)
        agent = make_region_phase1_agent()
        first = _observation(tiles, day=0, hour=1, farmer=(0, 0))
        self.assertEqual(agent(first)["farmer"], ["HARVEST"])
        harvest = agent.telemetry["grid"][0][0].tasks["HARVEST"]
        self.assertEqual(harvest.status, SCHEDULED)
        self.assertEqual(harvest.assigned_worker, "Farmer")
        self.assertEqual(harvest.planned_hour, 1)

        tiles[0][0] = None
        second = _observation(tiles, day=0, hour=2, farmer=(0, 0))
        self.assertEqual(agent(second)["farmer"], ["EAST"])
        self.assertEqual(agent.telemetry["grid"][0][0].tasks["HARVEST"].status, COMPLETED)
        self.assertFalse(agent.telemetry["needs_replan"])

    def test_a_worker_off_the_plan_is_replanned_from_where_they_stand(self) -> None:
        tiles = _tiles()
        tiles[0][1] = _plant("WHEAT", planted_day=-2, units=6)
        agent = make_region_phase1_agent()
        first = _observation(tiles, day=0, hour=1, farmer=(0, 0))
        self.assertEqual(agent(first)["farmer"], ["EAST"])
        stale = [item.operation for item in agent.telemetry["plan"].worker_routes[0].actions_by_hour if item.hour == 2]
        self.assertEqual(stale, ["HARVEST"])

        drifted = _observation(tiles, day=0, hour=2, farmer=(0, 2))
        action = agent(drifted)

        self.assertTrue(agent.telemetry["needs_replan"])
        self.assertNotEqual(action["farmer"], ["HARVEST"])
        self.assertEqual(agent.telemetry["planned_workers"], [("Farmer", (0, 2))])

    def test_optional_water_on_a_harvest_tile_is_scheduled_before_the_harvest(self) -> None:
        tiles = _tiles()
        tiles[3][2] = _plant("WHEAT", units=4, dry=1)
        agent = make_region_phase1_agent()
        first = _observation(tiles, day=2, hour=1, farmer=(2, 3))
        self.assertEqual(agent(first)["farmer"], ["WATER"])
        water = agent.telemetry["grid"][2][3].tasks["WATER"]
        self.assertEqual(water.status, SCHEDULED)
        self.assertFalse(water.mandatory)
        self.assertEqual(water.yield_gain, 1)

        tiles[3][2]["watered_today"] = True
        second = _observation(tiles, day=2, hour=2, farmer=(2, 3))
        self.assertEqual(agent(second)["farmer"], ["HARVEST"])
        self.assertEqual(agent.telemetry["grid"][2][3].tasks["WATER"].status, COMPLETED)

    def test_engine_steps_land_on_the_squares_the_route_already_named(self) -> None:
        environment = make(
            "kaggriculture",
            configuration={"episodeSteps": 8, "seed": 1},
            debug=True,
        )
        environment.reset()
        farm = environment.steps[0][0].observation.farms[0]
        plant = _plant("WHEAT", planted_day=-2, units=6)
        plant["max_lifespan_step"] = 10000
        plant["fertilized_until_day"] = -1
        farm["tiles"][4][2] = plant
        agent = make_region_phase1_agent()
        opponent = {"farmer": ["PASS"], "hands": [], "market": []}
        trace: list[tuple[str, tuple[int, int]]] = []
        for _ in range(4):
            observation = environment.steps[-1][0].observation
            action = agent(observation)
            before = tuple(observation.farms[0]["farmer"])
            trace.append((action["farmer"][0], before))
            environment.step([action, opponent])
            after = tuple(environment.steps[-1][0].observation.farms[0]["farmer"])
            self.assertEqual(after, _moved(before, action["farmer"][0]))

        self.assertEqual([step[0] for step in trace], ["PASS", "WEST", "WEST", "HARVEST"])
        self.assertEqual(trace[1][1], (4, 4))
        self.assertEqual(trace[3][1], (2, 4))
        self.assertIsNone(environment.steps[-1][0].observation.farms[0]["tiles"][4][2])

    def test_the_engine_marks_the_animal_fed_after_pickup(self) -> None:
        environment = make(
            "kaggriculture",
            configuration={"episodeSteps": 8, "seed": 1},
            debug=True,
        )
        environment.reset()
        observation = environment.steps[0][0].observation
        farm = observation.farms[0]
        animal = _animal("SHEEP", unfed=1)
        animal["pending_care_bonus"] = 0
        animal["fertilizer_available"] = False
        farm["tiles"][4][3] = animal
        farm["money"] = 0
        observation.private["shed"]["WHEAT"] = 1
        agent = make_region_phase1_agent()
        opponent = {"farmer": ["PASS"], "hands": [], "market": []}
        played: list[list] = []
        for _ in range(4):
            action = agent(environment.steps[-1][0].observation)
            played.append(list(action["farmer"]))
            environment.step([action, opponent])

        self.assertEqual(played[0], ["PASS"])
        self.assertEqual(played[1], ["PICKUP", "WHEAT", 1])
        self.assertEqual(played[2], ["WEST"])
        self.assertEqual(played[3], ["FEED"])
        self.assertTrue(environment.steps[-1][0].observation.farms[0]["tiles"][4][3]["fed_today"])

    def test_later_pickups_stay_out_of_the_wheat_sale(self) -> None:
        tiles = _tiles()
        tiles[2][4] = _animal("SHEEP", unfed=1)
        agent = make_region_phase1_agent()
        shared = dict(
            tiles=tiles,
            day=1,
            farmer=(0, 0),
            hands=[(4, 3)],
            shed={"WHEAT": 4},
            inventories=[{"WHEAT": 1}, {}],
        )
        agent(_observation(**shared, hour=0))
        action = agent(_observation(**shared, hour=1))

        pickups = [
            step
            for route in agent.telemetry["plan"].worker_routes
            if route.worker_id == "Hand1"
            for step in route.actions_by_hour
            if step.operation == "PICKUP"
        ]
        self.assertEqual([(step.hour, step.args) for step in pickups], [(2, ("WHEAT", 1))])
        old_route = agent.telemetry["market_route"]
        self.assertEqual(_reserved_wheat(old_route, [0] * len(old_route.actors)), 0)
        self.assertIn(["SELL", "WHEAT", 3], action["market"])
        self.assertNotIn(["SELL", "WHEAT", 4], action["market"])

        at_the_door = agent(_observation(**{**shared, "hour": 2, "hands": [(4, 4)]}))
        self.assertIn(["SELL", "WHEAT", 4], at_the_door["market"])

    def test_hour_zero_keeps_a_wheat_for_each_feed_before_the_new_hand_exists(self) -> None:
        tiles = _tiles()
        for spot in ((0, 0), (0, 9), (9, 0), (9, 9)):
            tiles[spot[1]][spot[0]] = _plant("WHEAT", dry=1)
        tiles[2][4] = _animal("SHEEP", unfed=1)
        tiles[6][6] = _animal("SHEEP", unfed=1)
        agent = make_region_phase1_agent()
        morning = _observation(
            tiles,
            day=1,
            hour=0,
            farmer=(0, 0),
            shed={"WHEAT": 4},
            inventories=[{"WHEAT": 1}],
        )
        opened = agent(morning)

        self.assertEqual(morning["farms"][0]["hands"], [])
        self.assertIn(["HIRE"], opened["market"])
        wheat_sales = [order for order in opened["market"] if order[:2] == ["SELL", "WHEAT"]]
        self.assertEqual(wheat_sales, [["SELL", "WHEAT", 3]])
        self.assertNotIn(["SELL", "WHEAT", 4], opened["market"])

        appeared = _observation(
            tiles,
            day=1,
            hour=1,
            farmer=(0, 0),
            hands=[(4, 3)],
            shed={"WHEAT": 4},
            inventories=[{"WHEAT": 1}, {}],
        )
        agent(appeared)
        plan = agent.telemetry["plan"]
        pickups = [
            step
            for route in plan.worker_routes
            if route.worker_id == "Hand1"
            for step in route.actions_by_hour
            if step.operation == "PICKUP"
        ]
        self.assertEqual([step.args for step in pickups], [("WHEAT", 1)])
        self.assertTrue(plan.feasible)
        self.assertEqual(plan.unfinished_visits, ())

    def test_the_morning_wheat_hold_shrinks_to_the_real_pickup(self) -> None:
        tiles = _tiles()
        for spot in ((0, 0), (0, 9), (9, 0), (9, 9)):
            tiles[spot[1]][spot[0]] = _plant("WHEAT", dry=1)
        tiles[2][4] = _animal("SHEEP", unfed=1)
        agent = make_region_phase1_agent()
        shared = dict(
            tiles=tiles,
            day=1,
            farmer=(0, 0),
            shed={"WHEAT": 4},
            inventories=[{"WHEAT": 1}],
        )
        agent(_observation(**shared, hour=0))
        hired = agent(_observation(**{**shared, "hour": 1, "hands": [(4, 3)], "inventories": [{"WHEAT": 1}, {}]}))
        self.assertIn(["SELL", "WHEAT", 3], hired["market"])
        self.assertNotIn(["SELL", "WHEAT", 4], hired["market"])

        at_the_door = agent(
            _observation(**{**shared, "hour": 2, "hands": [(4, 4)], "inventories": [{"WHEAT": 1}, {}]})
        )
        self.assertIn(["SELL", "WHEAT", 4], at_the_door["market"])

    def test_wheat_placed_this_hour_is_sold_this_hour(self) -> None:
        tiles = _tiles()
        tiles[4][4] = _plant("WHEAT", planted_day=-2, units=6)
        agent = make_region_phase1_agent()
        agent(_observation(tiles, day=0, hour=0, farmer=(4, 4), inventories=[{"WHEAT": 4}]))
        harvested = agent(_observation(tiles, day=0, hour=1, farmer=(4, 4), inventories=[{"WHEAT": 4}]))
        self.assertEqual(harvested["farmer"], ["HARVEST"])
        self.assertNotIn(["SELL", "WHEAT", 6], harvested["market"])

        tiles[4][4] = None
        unloaded = agent(_observation(_copy_tiles(tiles), day=0, hour=2, farmer=(4, 4), inventories=[{"WHEAT": 10}]))

        self.assertEqual(unloaded["farmer"], ["PLACE", "WHEAT", 6])
        self.assertIn(["SELL", "WHEAT", 6], unloaded["market"])
        self.assertNotIn(["SELL", "WHEAT", 10], unloaded["market"])

    def test_each_harvest_remembers_the_hour_its_goods_are_placed(self) -> None:
        tiles = _tiles()
        tiles[4][4] = _plant("WHEAT", planted_day=-2, units=5)
        tiles[4][3] = _plant("WHEAT", planted_day=-2, units=2)
        tiles[3][4] = _animal("COW", units=4, fed=True)
        tiles[4][2] = _animal("SHEEP", unfed=1)
        agent = make_region_phase1_agent()
        agent(
            _observation(
                tiles,
                day=0,
                hour=1,
                farmer=(4, 4),
                hands=[(2, 4)],
                shed={"WHEAT": 1},
            )
        )
        world = agent.telemetry["world"]
        plan = agent.telemetry["plan"]
        places = {
            (route.worker_id, action.args[0]): action.hour
            for route in plan.worker_routes
            for action in route.actions_by_hour
            if action.operation == "PLACE" and action.args
        }
        harvest_hours = {
            (route.worker_id, action.coord): action.hour
            for route in plan.worker_routes
            for action in route.actions_by_hour
            if action.operation == "HARVEST"
        }
        wheats = [crop for crop in world.farm.crops if crop.crop == "WHEAT"]
        cow = next(animal for animal in world.farm.animals if animal.animal == "COW")
        sheep = next(animal for animal in world.farm.animals if animal.animal == "SHEEP")

        self.assertEqual(len(wheats), 2)
        for crop in wheats:
            self.assertEqual(crop.planned_harvest_hour, harvest_hours[(crop.harvest_worker, crop.position)])
            self.assertEqual(crop.planned_drop_hour, places[(crop.harvest_worker, "WHEAT")])
            self.assertEqual(crop.planned_sell_hour, crop.planned_drop_hour)
        self.assertEqual(wheats[0].planned_drop_hour, wheats[1].planned_drop_hour)
        self.assertEqual(cow.planned_harvest_hour, harvest_hours[(cow.harvest_worker, cow.position)])
        self.assertEqual(cow.planned_drop_hour, places[(cow.harvest_worker, "MILK")])
        self.assertEqual(cow.planned_sell_hour, cow.planned_drop_hour)
        self.assertNotEqual(cow.planned_drop_hour, wheats[0].planned_drop_hour)
        self.assertIsNone(sheep.planned_drop_hour)
        self.assertIsNone(sheep.planned_sell_hour)
        self.assertIsNotNone(sheep.planned_feed_hour)
        for task in world.farm.tasks:
            if task.kind != "HARVEST":
                self.assertIsNone(task.planned_drop_hour)
                self.assertIsNone(task.planned_sell_hour)
                continue
            holder = next(
                (
                    item
                    for item in (*world.farm.crops, *world.farm.animals)
                    if item.position == task.position and item.planned_harvest_hour is not None
                ),
                None,
            )
            self.assertIsNotNone(holder)
            self.assertEqual(task.planned_hour, holder.planned_harvest_hour)
            self.assertEqual(task.planned_drop_hour, holder.planned_drop_hour)
            self.assertEqual(task.planned_sell_hour, holder.planned_sell_hour)

    def test_a_finished_harvest_keeps_its_unload_hour(self) -> None:
        tiles = _tiles()
        tiles[4][4] = _plant("WHEAT", planted_day=-2, units=6)
        agent = make_region_phase1_agent()
        agent(_observation(tiles, day=0, hour=1, farmer=(4, 4)))
        wheat = agent.telemetry["world"].farm.crops[0]
        self.assertEqual((wheat.planned_harvest_hour, wheat.planned_drop_hour, wheat.planned_sell_hour), (1, 2, 2))

        tiles[4][4] = None
        agent(_observation(tiles, day=0, hour=2, farmer=(4, 4)))
        done = next(task for task in agent.telemetry["world"].farm.tasks if task.kind == "HARVEST")
        self.assertEqual(done.status, COMPLETED)
        self.assertEqual((done.planned_hour, done.planned_drop_hour, done.planned_sell_hour), (1, 2, 2))

    def test_care_shifts_the_recorded_unload_hour(self) -> None:
        tiles = _tiles()
        tiles[3][4] = _animal("SHEEP", units=1, fed=True)
        agent = make_region_phase1_agent()
        agent(_observation(tiles, day=1, hour=1, farmer=(4, 4)))
        sheep = next(animal for animal in agent.telemetry["world"].farm.animals if animal.animal == "SHEEP")
        actions = agent.telemetry["plan"].worker_routes[0].actions_by_hour
        care = next(action.hour for action in actions if action.operation == "CARE")
        harvest = next(action.hour for action in actions if action.operation == "HARVEST")
        place = next(action.hour for action in actions if action.operation == "PLACE")

        self.assertLess(care, harvest)
        self.assertEqual(sheep.planned_care_hour, care)
        self.assertEqual(sheep.care_status, SCHEDULED)
        self.assertEqual(sheep.planned_harvest_hour, harvest)
        self.assertEqual(sheep.planned_drop_hour, place)
        self.assertEqual(sheep.planned_sell_hour, place)
        self.assertEqual(agent.telemetry["grid"][4][3].tasks["CARE"].status, SCHEDULED)

    def test_the_engine_marks_the_animal_fed_and_cared_on_separate_turns(self) -> None:
        environment = make(
            "kaggriculture",
            configuration={"episodeSteps": 8, "seed": 1},
            debug=True,
        )
        environment.reset()
        observation = environment.steps[0][0].observation
        farm = observation.farms[0]
        animal = _animal("SHEEP", unfed=1)
        animal["pending_care_bonus"] = 0
        animal["fertilizer_available"] = False
        animal["cared_today"] = False
        farm["tiles"][4][3] = animal
        farm["money"] = 0
        observation.private["inventories"] = [{"WHEAT": 1}]
        agent = make_region_phase1_agent()
        opponent = {"farmer": ["PASS"], "hands": [], "market": []}
        played: list[list] = []
        for _ in range(4):
            action = agent(environment.steps[-1][0].observation)
            played.append(list(action["farmer"]))
            environment.step([action, opponent])

        self.assertEqual(played[0], ["PASS"])
        self.assertEqual(played[1], ["WEST"])
        self.assertEqual(played[2], ["FEED"])
        self.assertEqual(played[3], ["CARE"])
        tile = environment.steps[-1][0].observation.farms[0]["tiles"][4][3]
        self.assertTrue(tile["fed_today"])
        self.assertTrue(tile["cared_today"])


def _moved(position: tuple[int, int], operation: str) -> tuple[int, int]:
    x, y = position
    if operation == "WEST":
        return (x - 1, y)
    if operation == "EAST":
        return (x + 1, y)
    if operation == "NORTH":
        return (x, y - 1)
    if operation == "SOUTH":
        return (x, y + 1)
    return position


if __name__ == "__main__":
    unittest.main()
