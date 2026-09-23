"""The 5×5 route drives each real hour. It does not guess where a hand is born."""

from __future__ import annotations

import unittest

from kaggle_environments import make

from lab.region_phase1 import make_region_phase1_agent
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
        tiles[0][0] = _plant("WHEAT", planted_day=-2, units=6)
        tiles[0][2] = _plant("WHEAT", planted_day=-2, units=6)
        observation = _observation(tiles, day=0, hour=1, farmer=(0, 0), hands=[(2, 0)])
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
        self.assertEqual(agent(second)["farmer"], ["PASS"])
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

    def test_optional_water_stays_on_the_grid_and_out_of_the_route(self) -> None:
        tiles = _tiles()
        tiles[3][2] = _plant("WHEAT", units=4, dry=1)
        observation = _observation(tiles, day=2, hour=1, farmer=(2, 3))
        agent = make_region_phase1_agent()
        action = agent(observation)

        self.assertEqual(action["farmer"], ["HARVEST"])
        self.assertNotIn("WATER", [item.operation for item in agent.telemetry["plan"].worker_routes[0].actions_by_hour])
        water = agent.telemetry["grid"][2][3].tasks["WATER"]
        self.assertFalse(water.mandatory)
        self.assertEqual(water.yield_gain, 1)

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
