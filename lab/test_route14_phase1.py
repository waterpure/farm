"""Phase 1 times harvests. It does not open new crops, animals, or land."""

from __future__ import annotations

import unittest
from typing import Any

from lab.route14_phase1 import choose_day_route, field_tasks, make_route14_phase1_agent, schedule_day


def _tiles() -> list[list[Any]]:
    return [["LOCKED"] * 10 for _ in range(10)]


def _plant(crop: str, *, planted_day: int = 0, units: int = 0, dry: int = 0, watered: bool = False) -> dict[str, Any]:
    return {
        "kind": "PLANT",
        "crop": crop,
        "planted_day": planted_day,
        "yield_units": units,
        "consecutive_unwatered": dry,
        "watered_today": watered,
    }


def _animal(name: str, *, unfed: int = 0, units: int = 0, fed: bool = False) -> dict[str, Any]:
    return {
        "kind": "COOP" if name == "GOOSE" else "PASTURE",
        "animal": name,
        "placed_day": 0,
        "yield_units": units,
        "consecutive_unfed": unfed,
        "fed_today": fed,
    }


def _observation(
    tiles: list[list[Any]],
    *,
    day: int = 10,
    hour: int = 0,
    money: int = 5000,
    farmer: tuple[int, int] = (4, 4),
    hands: list[tuple[int, int]] | None = None,
    shed: dict[str, int] | None = None,
    inventories: list[dict[str, int]] | None = None,
    rival: list[list[Any]] | None = None,
    rival_farmer: tuple[int, int] = (4, 4),
    rival_hands: list[tuple[int, int]] | None = None,
    shops: list[str] | None = None,
) -> dict[str, Any]:
    inventory = {item: 10000 for item in ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")}
    return {
        "player": 0,
        "day": day,
        "hour": hour,
        "step": day * 24 + hour,
        "farms": [
            {
                "money": money,
                "tiles": tiles,
                "farmer": list(farmer),
                "hands": [list(pos) for pos in (hands or [])],
                "hires_today": len(hands or []),
                "unlocked_quadrants": ["NW"],
            },
            {
                "money": money,
                "tiles": rival if rival is not None else _tiles(),
                "farmer": list(rival_farmer),
                "hands": [list(pos) for pos in (rival_hands or [])],
                "hires_today": len(rival_hands or []),
                "unlocked_quadrants": ["NW"],
            },
        ],
        "market": {"inventory": inventory, "prices": {"MELON": 250, "WHEAT": 25}},
        "town": {"unlocked_shops": list(shops or [])},
        "private": {
            "shed": dict(shed or {}),
            "seeds": {},
            "inventories": inventories or [{}],
        },
    }


class Route14Phase1Tests(unittest.TestCase):
    def test_water_is_mandatory_only_when_the_plant_would_die(self) -> None:
        tiles = _tiles()
        tiles[0][0] = _plant("WHEAT", dry=1)
        tiles[0][1] = _plant("WHEAT", dry=0)
        tiles[0][2] = _plant("MELON", units=6, dry=1)
        tasks = field_tasks(_observation(tiles, day=10))
        waters = [task.target for task in tasks if task.kind == "WATER"]
        harvests = [task.target for task in tasks if task.kind == "HARVEST"]
        self.assertEqual(waters, [(0, 0)])
        self.assertEqual(harvests, [(2, 0)])

    def test_feed_is_mandatory_for_every_unfed_animal(self) -> None:
        tiles = _tiles()
        tiles[0][0] = _animal("SHEEP", unfed=1)
        tiles[0][1] = _animal("SHEEP", unfed=0)
        tasks = field_tasks(_observation(tiles, day=3))
        self.assertEqual(
            [task.target for task in tasks if task.kind == "FEED"],
            [(0, 0), (1, 0)],
        )

    def test_sale_hour_is_the_drop_hour(self) -> None:
        tiles = _tiles()
        tiles[3][4] = _plant("MELON", units=6)
        route = schedule_day(_observation(tiles), 0)
        self.assertEqual(len(route.batches), 1)
        batch = route.batches[0]
        self.assertEqual(batch.harvest_hour, 1)
        self.assertEqual(batch.drop_hour, 3)
        self.assertEqual(batch.sell_hour, batch.drop_hour)

    def test_parallel_hands_deliver_the_same_wave(self) -> None:
        tiles = _tiles()
        spots = [(2, 4), (4, 2), (3, 3), (5, 3)]
        for spot in spots:
            tiles[spot[1]][spot[0]] = _plant("MELON", units=6)
        alone = schedule_day(_observation(tiles), 0)
        crew = schedule_day(_observation(tiles), 3)
        self.assertEqual(alone.harvests_delivered, len(spots))
        self.assertEqual(crew.harvests_delivered, len(spots))
        crew_hours = [batch.drop_hour for batch in crew.batches]
        alone_hours = [batch.drop_hour for batch in alone.batches]
        self.assertEqual(sorted(alone_hours), [5, 11, 17, 22])
        self.assertEqual(sorted(crew_hours), [5, 5, 7, 7])

    def test_without_a_rival_dump_the_smallest_crew_sells_the_melons(self) -> None:
        tiles = _tiles()
        for spot in ((4, 3), (3, 4), (4, 2)):
            tiles[spot[1]][spot[0]] = _plant("MELON", units=6)
        route = choose_day_route(_observation(tiles))
        self.assertEqual(route.extra_hires, 0)
        self.assertEqual(route.harvests_delivered, 3)
        self.assertTrue(all(batch.sell_hour == batch.drop_hour for batch in route.batches))

    def test_a_rival_dump_hires_enough_hands_to_sell_before_it(self) -> None:
        tiles = _tiles()
        for spot in ((4, 3), (3, 4), (5, 4)):
            tiles[spot[1]][spot[0]] = _plant("MELON", units=6)
        rival = _tiles()
        rival[6][0] = _plant("MELON", units=30)
        route = choose_day_route(_observation(tiles, rival=rival, rival_farmer=(0, 6)))
        self.assertGreaterEqual(route.extra_hires, 2)
        self.assertEqual(route.harvests_delivered, 3)
        self.assertTrue(all(batch.sell_hour < 6 for batch in route.batches))

    def test_dying_plants_at_dusk_need_a_second_worker(self) -> None:
        tiles = _tiles()
        tiles[4][0] = _plant("WHEAT", dry=1)
        tiles[4][8] = _plant("WHEAT", dry=1)
        route = choose_day_route(_observation(tiles, day=1, hour=18))
        self.assertTrue(route.survival_complete)
        self.assertEqual(route.extra_hires, 1)
        self.assertEqual(route.harvests_delivered, 0)

    def test_agent_walks_a_ripe_melon_to_the_door_and_sells_that_hour(self) -> None:
        tiles = _tiles()
        tiles[3][4] = _plant("MELON", units=6)
        observation = _observation(tiles)
        agent = make_route14_phase1_agent()
        position = [4, 4]
        inventory: dict[str, int] = {}
        shed: dict[str, int] = {}
        seen_sell = False
        for hour in range(4):
            observation["hour"] = hour
            observation["step"] = 10 * 24 + hour
            observation["farms"][0]["farmer"] = list(position)
            observation["private"]["inventories"] = [dict(inventory)]
            observation["private"]["shed"] = dict(shed)
            action = agent(observation)
            self.assertNotIn("BUY_SEED", {order[0] for order in action["market"]})
            self.assertNotIn("BUY_ANIMAL", {order[0] for order in action["market"]})
            self.assertNotIn("BUY_LAND", {order[0] for order in action["market"]})
            command = action["farmer"]
            self.assertNotIn(command[0], {"PLANT", "CARE", "FERTILIZE", "BUILD_COOP", "BUILD_PASTURE"})
            if command[0] == "NORTH":
                position[1] -= 1
            elif command[0] == "SOUTH":
                position[1] += 1
            elif command[0] == "HARVEST":
                inventory["MELON"] = 6
                tiles[3][4] = None
            elif command[0] == "PLACE":
                self.assertEqual(command, ["PLACE", "MELON", 6])
                self.assertEqual(hour, 3)
                shed["MELON"] = shed.get("MELON", 0) + 6
                inventory["MELON"] = 0
            sells = [order for order in action["market"] if order[0] == "SELL"]
            if hour == 3:
                self.assertEqual(sells, [["SELL", "MELON", 6]])
                seen_sell = True
            else:
                self.assertEqual(sells, [])
        self.assertTrue(seen_sell)

    def test_feed_keeps_the_wheat_and_sells_the_other_goods(self) -> None:
        tiles = _tiles()
        tiles[0][0] = _animal("SHEEP", unfed=1)
        observation = _observation(tiles, day=3, shed={"WHEAT": 1, "MELON": 2})
        route = choose_day_route(observation)
        self.assertEqual(route.survival_done, 1)
        self.assertEqual(route.wheat_buy, 0)
        steps = [(step.not_before, step.operation) for step in route.actors[0].steps]
        self.assertEqual(steps, [(0, ("PICKUP", "WHEAT", 1)), (9, ("FEED",))])
        action = make_route14_phase1_agent()(observation)
        self.assertEqual(action["farmer"], ["PICKUP", "WHEAT", 1])
        self.assertEqual(action["market"], [["SELL", "MELON", 2]])

    def test_rival_hands_appearing_rebuild_the_sale_route(self) -> None:
        tiles = _tiles()
        for spot in ((4, 3), (3, 4), (5, 4)):
            tiles[spot[1]][spot[0]] = _plant("MELON", units=6)
        observation = _observation(tiles)
        agent = make_route14_phase1_agent()
        agent(observation)
        self.assertEqual(agent.telemetry["route"].extra_hires, 0)
        rival = _tiles()
        rival[6][0] = _plant("MELON", units=30)
        observation["hour"] = 1
        observation["step"] = 10 * 24 + 1
        observation["farms"][1]["tiles"] = rival
        observation["farms"][1]["farmer"] = [0, 6]
        observation["farms"][1]["hands"] = [[4, 3], [3, 4], [5, 4], [4, 5]]
        agent(observation)
        route = agent.telemetry["route"]
        self.assertEqual(route.start_hour, 1)
        self.assertEqual(route.extra_hires, 2)
        self.assertEqual(sorted(batch.sell_hour for batch in route.batches), [4, 5, 6])

    def test_shed_goods_sell_now_and_empty_land_is_not_planted(self) -> None:
        tiles = _tiles()
        tiles[0][0] = None
        observation = _observation(tiles, shed={"MELON": 4})
        action = make_route14_phase1_agent()(observation)
        self.assertEqual(action["farmer"], ["PASS"])
        self.assertEqual(action["market"], [["SELL", "MELON", 4]])


if __name__ == "__main__":
    unittest.main()
