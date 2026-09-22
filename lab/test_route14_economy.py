"""Unit checks for the 14th-route economy rules."""

from __future__ import annotations

import unittest
from typing import Any

from lab.route14_economy import animal_labor_turns, batch_sale_revenue, choose_next_animal, choose_next_crop, choose_next_line, crop_labor_turns, crop_ready_to_harvest, door_distance, empty_slot_plan, may_harvest_melon_at, labor_price_per_turn, melon_harvest_cutoff, money_per_day, pack_market_orders, place_arrival_units, place_plan_by_value, plan_fill, planting_allowed, plantable_today, quota_tile_cap, rank_crop_lines, rank_lines, seed_fill_plan, sell_quantity, shed_place_target, should_buy_land, wheat_feed_order
from lab.route14_hub import hub_plan
from lab.route14_capacity import day_schedule, future_care_hires, proposed_crop_slots
from lab.route14_agent import make_route14_agent


PRICES = {
    "MELON": 250,
    "WHEAT": 25,
    "STRAWBERRY": 120,
    "TOMATO": 60,
    "CARROT": 35,
    "WOOL": 200,
    "MILK": 160,
    "EGG": 50,
}


def full_first_quadrant(crop: str = "MELON") -> list[list[object]]:
    tiles: list[list[object]] = [["LOCKED"] * 10 for _ in range(10)]
    for y in range(5):
        for x in range(5):
            tiles[y][x] = {"kind": "PLANT", "crop": crop, "yield_units": 0, "planted_day": 0}
    return tiles


class Route14EconomyTests(unittest.TestCase):
    def test_land_quota_is_seventy_percent_of_tiles(self) -> None:
        self.assertEqual(quota_tile_cap(10), 7)
        self.assertEqual(quota_tile_cap(25), 17)

    def test_next_crop_is_highest_money_per_day_not_shop_gap(self) -> None:
        tiles = [[None] * 5 for _ in range(5)]
        observation = {
            "player": 0,
            "day": 0,
            "farms": [{"money": 3000, "tiles": tiles, "unlocked_quadrants": ["NW"]}],
            "market": {"prices": {"MELON": 250, "WHEAT": 25, "STRAWBERRY": 120, "TOMATO": 60, "CARROT": 35}},
            "town": {"unlocked_shops": ["BAKERY", "PET_CAFE"]},
            "private": {"shed": {}, "inventories": [{}], "seeds": {}},
        }
        crop, role = choose_next_crop(observation)
        self.assertEqual(crop, "MELON")
        self.assertEqual(role, "quota")

    def test_harvest_uses_engine_first_yield_not_max_lifespan(self) -> None:
        self.assertFalse(crop_ready_to_harvest("MELON", age=9, yield_units=6))
        self.assertTrue(crop_ready_to_harvest("MELON", age=10, yield_units=6))
        self.assertTrue(crop_ready_to_harvest("MELON", age=10, yield_units=3))
        self.assertTrue(crop_ready_to_harvest("WHEAT", age=2, yield_units=2))
        self.assertTrue(crop_ready_to_harvest("CARROT", age=2, yield_units=2))
        self.assertTrue(crop_ready_to_harvest("TOMATO", age=8, yield_units=1))
        self.assertTrue(crop_ready_to_harvest("STRAWBERRY", age=10, yield_units=1))

    def test_sell_whatever_is_available_any_day(self) -> None:
        quantity, window = sell_quantity("MELON", day=10, days_left=20, price=100, window=None, available=5)
        self.assertEqual(quantity, 5)
        self.assertIsNone(window)
        quantity, window = sell_quantity("MELON", day=16, days_left=14, price=100, window=None, available=5)
        self.assertEqual(quantity, 5)
        self.assertIsNone(window)
        quantity, window = sell_quantity(
            "MELON", day=20, days_left=10, price=1, window={"armed": False, "max_price": 200}, available=5
        )
        self.assertEqual(quantity, 5)
        self.assertIsNone(window)
        quantity, _ = sell_quantity("MELON", day=29, days_left=1, price=10, window=None, available=3)
        self.assertEqual(quantity, 3)
        held, _ = sell_quantity("WHEAT", day=16, days_left=14, price=30, window=None, available=8, wheat_reserve=3)
        self.assertEqual(held, 5)
        none_left, _ = sell_quantity("WHEAT", day=16, days_left=14, price=30, window=None, available=2, wheat_reserve=3)
        self.assertEqual(none_left, 0)

    def test_sell_arrivals_even_when_field_still_ripe(self) -> None:
        quantity, window = sell_quantity("MELON", day=10, days_left=20, price=250, window=None, available=6)
        self.assertEqual(quantity, 6)
        self.assertIsNone(window)
        quantity, _ = sell_quantity("MELON", day=10, days_left=20, price=250, window=None, available=24)
        self.assertEqual(quantity, 24)

    def test_place_arrivals_count_this_hour_place_only(self) -> None:
        arrivals = place_arrival_units(
            [
                ["PLACE", "MELON", 6],
                ["EAST"],
                ["PLACE", "MELON", 6],
                ["PLACE", "SHEEP"],
                ["HARVEST"],
                ["PLACE", "WOOL", 2],
            ]
        )
        self.assertEqual(arrivals, {"MELON": 12, "WOOL": 2})

    def test_stand_on_door_unloads_even_if_that_door_is_taken(self) -> None:
        self.assertEqual(shed_place_target((4, 4), {(4, 4), (5, 4)}), (4, 4))
        self.assertEqual(shed_place_target((5, 5), {(4, 4)}), (5, 5))
        self.assertEqual(shed_place_target((6, 4), {(4, 4)}), (5, 4))

    def test_day10_harvests_near_ring_before_corners(self) -> None:
        tiles = [[None] * 5 for _ in range(5)]
        tiles[4][3] = {"kind": "PLANT", "crop": "MELON", "planted_day": 0, "yield_units": 6}
        tiles[0][0] = {"kind": "PLANT", "crop": "MELON", "planted_day": 0, "yield_units": 6}
        observation = {
            "player": 0,
            "day": 10,
            "farms": [{"tiles": tiles}],
        }
        self.assertEqual(melon_harvest_cutoff(observation), 5)
        self.assertTrue(may_harvest_melon_at(observation, (3, 4)))
        self.assertFalse(may_harvest_melon_at(observation, (0, 0)))
        tiles[4][3] = None
        self.assertEqual(melon_harvest_cutoff(observation), 9)
        self.assertTrue(may_harvest_melon_at(observation, (0, 0)))
        observation["day"] = 11
        tiles[4][3] = {"kind": "PLANT", "crop": "MELON", "planted_day": 0, "yield_units": 6}
        self.assertIsNone(melon_harvest_cutoff(observation))
        self.assertTrue(may_harvest_melon_at(observation, (0, 0)))

    def test_market_pack_keeps_seed_when_many_hires(self) -> None:
        packed = pack_market_orders(
            sells=[["SELL", "EGG", 2], ["SELL", "WOOL", 4]],
            seed=["BUY_SEED", "WHEAT", 6],
            animal=["BUY_ANIMAL", "SHEEP", 1],
            wheat=["BUY_PRODUCT", "WHEAT", 2],
            land=None,
            hires=8,
        )
        self.assertEqual(len(packed), 10)
        self.assertIn(["BUY_SEED", "WHEAT", 6], packed)
        self.assertEqual(sum(order[0] == "HIRE" for order in packed), 5)

    def test_hire_first_still_buys_feed_wheat(self) -> None:
        packed = pack_market_orders(
            sells=[["SELL", "MELON", 6]],
            seeds=[["BUY_SEED", "MELON", 4]],
            animal=None,
            wheat=["BUY_PRODUCT", "WHEAT", 2],
            land=None,
            hires=11,
            hire_first=True,
        )
        self.assertIn(["BUY_PRODUCT", "WHEAT", 2], packed)
        self.assertEqual(sum(order[0] == "HIRE" for order in packed), 8)

    def test_buy_feed_wheat_even_inside_cash_buffer(self) -> None:
        tiles = [[None] * 5 for _ in range(5)]
        tiles[0][0] = {"kind": "PASTURE", "animal": "SHEEP", "fed_today": False}
        observation = {
            "player": 0,
            "day": 2,
            "farms": [{"money": 220, "tiles": tiles, "unlocked_quadrants": ["NW"]}],
            "market": {"prices": {"WHEAT": 25, "MELON": 250, "WOOL": 200}},
            "private": {"shed": {}, "inventories": [{}]},
        }
        self.assertEqual(wheat_feed_order(observation, 220), ["BUY_PRODUCT", "WHEAT", 1])

    def test_land_wait_until_current_tiles_are_staffed(self) -> None:
        tiles = [[None] * 5 for _ in range(5)]
        observation = {
            "player": 0,
            "day": 0,
            "farms": [{"money": 3000, "tiles": tiles, "unlocked_quadrants": ["NW"]}],
            "market": {"prices": {"MELON": 40, "WHEAT": 20, "STRAWBERRY": 30}},
            "town": {"unlocked_shops": []},
            "private": {"shed": {}, "inventories": [{}]},
        }
        self.assertFalse(should_buy_land(observation, labor_short=False, workers=9))
        self.assertGreater(plantable_today(observation, 1), 0)

    def test_land_not_bought_while_any_tile_is_empty(self) -> None:
        plants = [[{"kind": "PLANT", "crop": "WHEAT"}] * 5 for _ in range(4)] + [[None] * 5]
        observation = {
            "player": 0,
            "day": 2,
            "farms": [{"money": 3000, "tiles": plants, "unlocked_quadrants": ["NW"]}],
            "market": {"prices": {"MELON": 40, "WHEAT": 20, "STRAWBERRY": 30}},
            "town": {"unlocked_shops": []},
            "private": {"shed": {}, "inventories": [{}]},
        }
        self.assertFalse(should_buy_land(observation, labor_short=False, workers=9))

    def test_seed_plan_fills_staffable_tiles_without_overbuying_one_crop(self) -> None:
        tiles = [[None] * 5 for _ in range(5)]
        observation = {
            "player": 0,
            "day": 0,
            "farms": [{"money": 3000, "tiles": tiles, "unlocked_quadrants": ["NW"]}],
            "market": {
                "prices": {
                    "MELON": 250,
                    "WHEAT": 25,
                    "STRAWBERRY": 120,
                    "TOMATO": 60,
                    "CARROT": 35,
                    "WOOL": 200,
                    "MILK": 160,
                    "EGG": 50,
                }
            },
            "town": {"unlocked_shops": []},
            "private": {"shed": {}, "inventories": [{}], "seeds": {}},
        }
        slots = empty_slot_plan(observation, 25, cash=3000)
        self.assertGreaterEqual(sum(kind == "animal" for kind, _name in slots), 1)
        self.assertGreaterEqual(sum(kind == "crop" and name == "MELON" for kind, name in slots), 12)
        plan = seed_fill_plan(observation, workers=4, cash=3000)
        crops = {order[1]: order[2] for order in plan}
        self.assertGreaterEqual(sum(crops.values()), 1)
        self.assertLessEqual(sum(crops.values()), 24)
        self.assertGreaterEqual(crops.get("MELON", 0), 1)

    def test_fill_plan_hires_to_cover_empty_tiles(self) -> None:
        tiles = [[None] * 5 for _ in range(5)]
        observation = {
            "player": 0,
            "day": 0,
            "hour": 0,
            "farms": [{"money": 3000, "tiles": tiles, "unlocked_quadrants": ["NW"]}],
            "market": {"prices": {"MELON": 250, "WHEAT": 25, "STRAWBERRY": 120, "TOMATO": 60, "CARROT": 35}},
            "town": {"unlocked_shops": []},
            "private": {"shed": {}, "inventories": [{}], "seeds": {}},
        }
        hired, seeds = plan_fill(observation, 3000)
        self.assertGreater(hired, 0)
        self.assertGreaterEqual(sum(order[2] for order in seeds), 12)

    def test_fill_plan_hires_to_water_full_field(self) -> None:
        plants = [[{"kind": "PLANT", "crop": "MELON", "yield_units": 0}] * 5 for _ in range(5)]
        observation = {
            "player": 0,
            "day": 2,
            "hour": 0,
            "farms": [{"money": 3000, "tiles": plants, "unlocked_quadrants": ["NW"]}],
            "market": {"prices": {"MELON": 250, "WHEAT": 25, "STRAWBERRY": 120, "TOMATO": 60, "CARROT": 35}},
            "town": {"unlocked_shops": []},
            "private": {"shed": {}, "inventories": [{}], "seeds": {}},
        }
        hired, seeds = plan_fill(observation, 3000)
        self.assertGreaterEqual(hired, 2)
        self.assertEqual(seeds, [])

    def test_dump_day_hires_double_crew(self) -> None:
        plants = [[{"kind": "PLANT", "crop": "CARROT", "yield_units": 3, "planted_day": 0}] * 5 for _ in range(5)]
        observation = {
            "player": 0,
            "day": 2,
            "hour": 0,
            "farms": [{"money": 3000, "tiles": plants, "unlocked_quadrants": ["NW"]}],
            "market": {"prices": {"MELON": 250, "WHEAT": 25, "STRAWBERRY": 120, "TOMATO": 60, "CARROT": 35}},
            "town": {"unlocked_shops": []},
            "private": {"shed": {}, "inventories": [{}], "seeds": {}},
        }
        hired, _ = plan_fill(observation, 3000)
        self.assertGreaterEqual(hired, 4)

    def test_day_10_hires_eleven_for_melon_rush(self) -> None:
        plants = [[{"kind": "PLANT", "crop": "MELON", "yield_units": 5, "planted_day": 0}] * 5 for _ in range(5)]
        observation = {
            "player": 0,
            "day": 10,
            "hour": 0,
            "farms": [{"money": 3000, "tiles": plants, "unlocked_quadrants": ["NW"], "hands": [], "hires_today": 0}],
            "market": {"prices": {"MELON": 250, "WHEAT": 25, "STRAWBERRY": 120, "TOMATO": 60, "CARROT": 35}},
            "town": {"unlocked_shops": []},
            "private": {"shed": {}, "inventories": [{}], "seeds": {}},
        }
        hired, _ = plan_fill(observation, 3000)
        self.assertGreaterEqual(hired, 11)
        packed = pack_market_orders(
            sells=[],
            seeds=[["BUY_SEED", "CARROT", 3]],
            animal=None,
            wheat=None,
            land=None,
            hires=hired,
            hire_first=True,
        )
        self.assertEqual(sum(order[0] == "HIRE" for order in packed), 10)
        self.assertNotIn(["BUY_SEED", "CARROT", 3], packed)

    def test_dump_does_not_buy_seeds_before_selling(self) -> None:
        plants = [[{"kind": "PLANT", "crop": "MELON", "yield_units": 5, "planted_day": 0}] * 4 + [None] for _ in range(5)]
        observation = {
            "player": 0,
            "day": 10,
            "hour": 0,
            "farms": [{"money": 3000, "tiles": plants, "unlocked_quadrants": ["NW"], "hands": [], "hires_today": 0}],
            "market": {"prices": {"MELON": 250, "WHEAT": 25, "STRAWBERRY": 120, "TOMATO": 60, "CARROT": 35}},
            "town": {"unlocked_shops": []},
            "private": {"shed": {"MELON": 6}, "inventories": [{}], "seeds": {}},
        }
        hired, seeds = plan_fill(observation, 3000)
        self.assertFalse(planting_allowed(observation))
        self.assertGreaterEqual(hired, 9)
        self.assertEqual(seeds, [])

    def test_replant_once_oneshot_crop_is_off_the_field(self) -> None:
        berries = [[{"kind": "PLANT", "crop": "STRAWBERRY", "yield_units": 1, "planted_day": 0}] * 2 + [None] * 3 for _ in range(5)]
        observation = {
            "player": 0,
            "day": 10,
            "hour": 8,
            "farms": [{"money": 3000, "tiles": berries, "unlocked_quadrants": ["NW"], "hands": [[]], "hires_today": 11}],
            "market": {"prices": {"MELON": 250, "WHEAT": 25, "STRAWBERRY": 120, "TOMATO": 60, "CARROT": 35}},
            "town": {"unlocked_shops": []},
            "private": {"shed": {"MELON": 12}, "inventories": [{}, {}], "seeds": {}},
        }
        self.assertTrue(planting_allowed(observation))
        hired, seeds = plan_fill(observation, 3000)
        self.assertGreater(sum(order[2] for order in seeds), 0)

    def test_one_leftover_melon_does_not_block_afternoon_replant(self) -> None:
        tiles = [[{"kind": "PLANT", "crop": "MELON", "yield_units": 6, "planted_day": 0}] + [None] * 4 for _ in range(5)]
        observation = {
            "player": 0,
            "day": 10,
            "hour": 8,
            "farms": [{"money": 3000, "tiles": tiles, "unlocked_quadrants": ["NW"], "hands": [[]], "hires_today": 11}],
            "market": {"prices": {"MELON": 250, "WHEAT": 25, "STRAWBERRY": 120, "TOMATO": 60, "CARROT": 35}},
            "town": {"unlocked_shops": []},
            "private": {"shed": {"MELON": 60}, "inventories": [{}, {}], "seeds": {}},
        }
        self.assertTrue(planting_allowed(observation))

    def test_cared_sheep_outrank_melon_at_base_prices(self) -> None:
        tiles = [[None] * 5 for _ in range(5)]
        observation = {
            "player": 0,
            "day": 0,
            "hour": 0,
            "farms": [{"money": 20000, "tiles": tiles, "unlocked_quadrants": ["NW"]}],
            "market": {
                "prices": {
                    "MELON": 250,
                    "WHEAT": 25,
                    "STRAWBERRY": 120,
                    "TOMATO": 60,
                    "CARROT": 35,
                    "WOOL": 200,
                    "MILK": 160,
                    "EGG": 50,
                }
            },
            "town": {"unlocked_shops": []},
            "private": {"shed": {}, "inventories": [{}], "seeds": {}},
        }
        ranked = rank_lines(observation)
        self.assertEqual(ranked[0][1], "SHEEP")
        self.assertEqual(ranked[1][1], "COW")
        slots = empty_slot_plan(observation, 25)
        self.assertEqual(sum(kind == "animal" for kind, _name in slots), 12)
        self.assertEqual(sum(kind == "crop" and name == "MELON" for kind, name in slots), 9)
        self.assertEqual(sum(kind == "crop" and name == "CARROT" for kind, name in slots), 4)

    def test_day10_empty_field_is_half_livestock_half_crops(self) -> None:
        tiles = [[None] * 5 for _ in range(5)]
        observation = {
            "player": 0,
            "day": 10,
            "hour": 8,
            "farms": [{"money": 20000, "tiles": tiles, "unlocked_quadrants": ["NW"]}],
            "market": {
                "prices": {
                    "MELON": 250,
                    "WHEAT": 25,
                    "STRAWBERRY": 120,
                    "TOMATO": 60,
                    "CARROT": 35,
                    "WOOL": 200,
                    "MILK": 160,
                    "EGG": 50,
                }
            },
            "town": {"unlocked_shops": []},
            "private": {"shed": {}, "inventories": [{}], "seeds": {}},
        }
        slots = empty_slot_plan(observation, 25)
        animals = sum(kind == "animal" for kind, _name in slots)
        melon = sum(kind == "crop" and name == "MELON" for kind, name in slots)
        carrot = sum(kind == "crop" and name == "CARROT" for kind, name in slots)
        self.assertEqual(animals, 12)
        self.assertEqual(melon, 9)
        self.assertEqual(carrot, 4)

    def test_empty_pastures_count_as_livestock_land(self) -> None:
        tiles = [[{"kind": "PASTURE"}] * 5 for _ in range(2)] + [[None] * 5 for _ in range(3)]
        observation = {
            "player": 0,
            "day": 10,
            "hour": 8,
            "farms": [{"money": 20000, "tiles": tiles, "unlocked_quadrants": ["NW"]}],
            "market": {
                "prices": {
                    "MELON": 250,
                    "WHEAT": 25,
                    "STRAWBERRY": 120,
                    "TOMATO": 60,
                    "CARROT": 35,
                    "WOOL": 200,
                    "MILK": 160,
                    "EGG": 50,
                }
            },
            "town": {"unlocked_shops": []},
            "private": {"shed": {}, "inventories": [{}], "seeds": {}},
        }
        slots = empty_slot_plan(observation, 15)
        self.assertEqual(sum(kind == "animal" for kind, _name in slots), 2)
        self.assertGreaterEqual(sum(kind == "crop" for kind, _name in slots), 9)

    def test_empty_sheds_still_buy_the_missing_animal(self) -> None:
        tiles = [[{"kind": "PASTURE"}] * 5 for _ in range(2)] + [
            [{"kind": "PLANT", "crop": "MELON", "planted_day": 0, "yield_units": 0}] * 5 for _ in range(3)
        ]
        observation = {
            "player": 0,
            "day": 2,
            "hour": 3,
            "farms": [{"money": 8000, "tiles": tiles, "unlocked_quadrants": ["NW"]}],
            "market": {
                "prices": {
                    "MELON": 250,
                    "WHEAT": 25,
                    "STRAWBERRY": 120,
                    "TOMATO": 60,
                    "CARROT": 35,
                    "WOOL": 200,
                    "MILK": 160,
                    "EGG": 50,
                }
            },
            "town": {"unlocked_shops": []},
            "private": {"shed": {}, "inventories": [{}], "seeds": {}},
        }
        self.assertEqual(choose_next_animal(observation), "SHEEP")

    def test_care_labour_is_charged_and_gets_dearer_with_a_big_crew(self) -> None:
        # The n-th hand of the day costs fib(n) and works 24 turns, so the same
        # care bill is cheap with a small crew and expensive with a large one.
        self.assertLess(labor_price_per_turn(self._plot(crew=2)), labor_price_per_turn(self._plot(crew=12)))
        # Livestock is tended every remaining day; a tile is tended until cleared.
        self.assertGreater(animal_labor_turns("COW", 20), crop_labor_turns("MELON", 20))
        prices = dict(PRICES)
        free = money_per_day("STRAWBERRY", prices, 20, 10000, 0, 0.0)
        costed = money_per_day("STRAWBERRY", prices, 20, 10000, 0, 5.0)
        assert free is not None and costed is not None
        self.assertAlmostEqual(free - costed, crop_labor_turns("STRAWBERRY", 20) * 5.0 / 16)

    def test_livestock_must_beat_the_crop_on_the_same_yardstick(self) -> None:
        # Wool and milk at their balance price: the free pasture slot is used.
        self.assertEqual(choose_next_line(self._plot(crew=4))[0], "animal")
        # Same free slot and same cash, but the animal products are now in glut,
        # so a head pays less per day than a melon tile and the land goes to the
        # crop. The herd still shows a profit, so this is a comparison, not a veto.
        glutted = self._plot(crew=4, glut={"WOOL": 30, "MILK": 30, "EGG": 300})
        animals = {name: value for kind, name, value in rank_lines(glutted) if kind == "animal"}
        self.assertTrue(all(value > 0 for value in animals.values()))
        self.assertEqual(choose_next_line(glutted)[0], "crop")

    @staticmethod
    def _plot(crew: int, glut: dict[str, int] | None = None) -> dict[str, Any]:
        inventory = {item: 10000 for item in PRICES}
        for item, over in (glut or {}).items():
            inventory[item] = 10000 + over
        return {
            "player": 0,
            "day": 6,
            "hour": 1,
            "farms": [{
                "money": 9000,
                "tiles": [[None] * 5 for _ in range(5)],
                "unlocked_quadrants": ["NW"],
                "hands": [[0, 0]] * crew,
                "hires_today": crew,
            }],
            "market": {"prices": dict(PRICES), "inventory": inventory},
            "town": {"unlocked_shops": []},
            "private": {"shed": {}, "inventories": [{}], "seeds": {}},
        }

    def test_last_day_reaps_the_herd_instead_of_feeding_it(self) -> None:
        tiles: list[list[object]] = [[None] * 5 for _ in range(5)]
        # An unfed cow holding milk: on any other day it is fed first and the milk
        # waits, but on the last day feeding buys nothing and the milk is the score.
        tiles[3][3] = {"kind": "PASTURE", "animal": "COW", "placed_day": 0, "fed_today": False, "cared_today": False, "yield_units": 6}
        observation = {
            "player": 0, "step": 0, "day": 29, "hour": 2,
            "farms": [
                {"money": 9000, "tiles": tiles, "farmer": [3, 3], "hands": [], "hires_today": 0, "unlocked_quadrants": ["NW"]},
                {"money": 0, "tiles": [[None] * 5 for _ in range(5)], "farmer": [0, 0], "hands": []},
            ],
            "market": {"prices": dict(PRICES), "inventory": {item: 10000 for item in PRICES}},
            "town": {"unlocked_shops": []},
            "private": {"shed": {"WHEAT": 20}, "inventories": [{}], "seeds": {}},
        }
        action = make_route14_agent()(observation)
        self.assertEqual(action["farmer"], ["HARVEST"])
        # Shed stock scores nothing, so the kept-back feed wheat goes out too.
        self.assertEqual(sum(order[2] for order in action["market"] if order[:2] == ["SELL", "WHEAT"]), 20)
        self.assertFalse(any(order[0] == "BUY_PRODUCT" for order in action["market"]))

    def test_batch_revenue_follows_the_engine_unit_by_unit(self) -> None:
        from kaggle_environments.envs.kaggriculture.kaggriculture import market_price

        for item, units in (("MELON", 6), ("WHEAT", 4), ("STRAWBERRY", 4)):
            for stock in (10000, 10060, 10150):
                self.assertEqual(
                    batch_sale_revenue(item, stock, units),
                    sum(market_price(item, stock + offset) for offset in range(units)),
                )

    def test_flooding_one_crop_hands_the_next_tile_to_another(self) -> None:
        def field(melon_tiles: int) -> dict[str, Any]:
            tiles: list[list[object]] = [[None] * 5 for _ in range(5)]
            for index in range(melon_tiles):
                tiles[index // 5][index % 5] = {"kind": "PLANT", "crop": "MELON", "planted_day": 0, "yield_units": 0}
            return {
                "player": 0,
                "day": 1,
                "hour": 1,
                "farms": [{"money": 9000, "tiles": tiles, "unlocked_quadrants": ["NW"]}],
                "market": {"prices": dict(PRICES), "inventory": {item: 10000 for item in PRICES}},
                "town": {"unlocked_shops": []},
                "private": {"shed": {}, "inventories": [{}], "seeds": {}},
            }

        # Melon pays best on an empty plot, but its glut curve is quadratic, so a
        # plot already full of melon must hand the next tile to another crop.
        empty_best = rank_crop_lines(field(0))
        flooded = rank_crop_lines(field(24))
        self.assertEqual(empty_best[0][1], "MELON")
        self.assertNotEqual(flooded[0][1], "MELON")
        melon_when_flooded = next(value for _kind, name, value in flooded if name == "MELON")
        self.assertLess(melon_when_flooded, empty_best[0][2])

    def test_no_seed_for_a_crop_that_cannot_reach_harvest(self) -> None:
        prices = {
            "MELON": 250,
            "WHEAT": 25,
            "STRAWBERRY": 120,
            "TOMATO": 60,
            "CARROT": 35,
            "WOOL": 200,
            "MILK": 160,
            "EGG": 50,
        }

        def late_day(day: int) -> dict[str, Any]:
            return {
                "player": 0,
                "day": day,
                "hour": 1,
                "farms": [{"money": 30000, "tiles": [[None] * 5 for _ in range(5)], "unlocked_quadrants": ["NW"]}],
                "market": {"prices": prices},
                "town": {"unlocked_shops": []},
                "private": {"shed": {}, "inventories": [{}], "seeds": {}},
            }

        # The quickest crops need 2 days to first yield, so day 27 is the last
        # sowing that can still be harvested; day 28 onward must stay idle.
        kind, name, _role = choose_next_line(late_day(27))
        self.assertEqual(kind, "crop")
        self.assertIn(name, {"WHEAT", "CARROT"})
        self.assertEqual(choose_next_line(late_day(28)), (None, None, "idle"))
        self.assertEqual(empty_slot_plan(late_day(28), 25), [])
        self.assertEqual(empty_slot_plan(late_day(29), 25), [])

    def test_rotation_does_not_assign_animals_the_purse_cannot_stock(self) -> None:
        tiles = [[None] * 5 for _ in range(5)]
        observation = {
            "player": 0,
            "day": 0,
            "hour": 0,
            "farms": [{"money": 1200, "tiles": tiles, "unlocked_quadrants": ["NW"]}],
            "market": {
                "prices": {
                    "MELON": 250,
                    "WHEAT": 25,
                    "STRAWBERRY": 120,
                    "TOMATO": 60,
                    "CARROT": 35,
                    "WOOL": 200,
                    "MILK": 160,
                    "EGG": 50,
                }
            },
            "town": {"unlocked_shops": []},
            "private": {"shed": {}, "inventories": [{}], "seeds": {}},
        }
        slots = empty_slot_plan(observation, 25)
        self.assertEqual(sum(kind == "animal" for kind, _name in slots), 0)
        self.assertGreaterEqual(sum(kind == "crop" and name == "MELON" for kind, name in slots), 1)
        plan = seed_fill_plan(observation, workers=9, cash=1200)
        crops = {order[1]: order[2] for order in plan}
        self.assertGreaterEqual(crops.get("MELON", 0), 1)

    def test_no_land_if_the_plot_price_does_not_fit(self) -> None:
        plants = [[{"kind": "PLANT", "crop": "MELON", "yield_units": 0, "planted_day": 0}] * 10 for _ in range(5)]
        observation = {
            "player": 0,
            "day": 18,
            "hour": 0,
            "farms": [
                {
                    "money": 1800,
                    "tiles": plants,
                    "unlocked_quadrants": ["NW", "NE"],
                    "hands": [],
                    "hires_today": 0,
                }
            ],
            "market": {"prices": {"MELON": 250, "WHEAT": 25, "STRAWBERRY": 120, "TOMATO": 60, "CARROT": 35}},
            "town": {"unlocked_shops": []},
            "private": {"shed": {}, "inventories": [{}], "seeds": {}},
        }
        self.assertFalse(should_buy_land(observation, labor_short=False, workers=9))

    def test_buy_land_when_full_staffed_and_profitable(self) -> None:
        plants = full_first_quadrant("STRAWBERRY")
        observation = {
            "player": 0,
            "day": 4,
            "hour": 0,
            "farms": [{"money": 8000, "tiles": plants, "unlocked_quadrants": ["NW"], "hands": [], "hires_today": 0}],
            "market": {"prices": {"MELON": 250, "WHEAT": 25, "STRAWBERRY": 120, "TOMATO": 60, "CARROT": 35}},
            "town": {"unlocked_shops": []},
            "private": {"shed": {}, "inventories": [{}], "seeds": {}},
        }
        self.assertTrue(should_buy_land(observation, labor_short=False, workers=4))
        observation["farms"][0]["money"] = 1100
        self.assertFalse(should_buy_land(observation, labor_short=False, workers=4))

    def test_higher_value_lines_take_tiles_closer_to_the_door(self) -> None:
        tiles = [["LOCKED"] * 10 for _ in range(10)]
        for y in range(5):
            for x in range(5):
                tiles[y][x] = None
        empties = [(x, y) for y in range(5) for x in range(5)]
        observation = {
            "player": 0,
            "day": 0,
            "farms": [{"money": 3000, "tiles": tiles, "unlocked_quadrants": ["NW"]}],
            "market": {"prices": {"MELON": 250, "WHEAT": 25, "STRAWBERRY": 120, "TOMATO": 60, "CARROT": 35}},
            "town": {"unlocked_shops": []},
            "private": {"shed": {}, "inventories": [{}], "seeds": {}},
        }
        plan = [("crop", "WHEAT")] * 8 + [("crop", "MELON")] * 17
        placed = dict(place_plan_by_value(observation, empties, plan))
        self.assertEqual(placed[(4, 4)], ("crop", "MELON"))
        self.assertEqual(placed[(3, 4)], ("crop", "MELON"))
        self.assertEqual(placed[(0, 0)], ("crop", "WHEAT"))
        self.assertEqual(placed[(1, 0)], ("crop", "WHEAT"))
        melon_far = max(door_distance(pos) for pos, line in placed.items() if line == ("crop", "MELON"))
        wheat_near = min(door_distance(pos) for pos, line in placed.items() if line == ("crop", "WHEAT"))
        self.assertLessEqual(melon_far, wheat_near)

    def test_whichever_crop_pays_more_today_gets_the_door_tiles(self) -> None:
        tiles = [["LOCKED"] * 10 for _ in range(10)]
        for y in range(5):
            for x in range(5):
                tiles[y][x] = None
        empties = [(x, y) for y in range(5) for x in range(5)]
        observation = {
            "player": 0,
            "day": 0,
            "farms": [{"money": 3000, "tiles": tiles, "unlocked_quadrants": ["NW"]}],
            "market": {"prices": {"MELON": 250, "WHEAT": 25, "STRAWBERRY": 700, "TOMATO": 60, "CARROT": 35}},
            "town": {"unlocked_shops": []},
            "private": {"shed": {}, "inventories": [{}], "seeds": {}},
        }
        plan = [("crop", "MELON")] * 12 + [("crop", "STRAWBERRY")] * 13
        placed = dict(place_plan_by_value(observation, empties, plan))
        self.assertEqual(placed[(4, 4)], ("crop", "STRAWBERRY"))
        self.assertEqual(placed[(0, 0)], ("crop", "MELON"))
        berry_far = max(door_distance(pos) for pos, line in placed.items() if line == ("crop", "STRAWBERRY"))
        melon_near = min(door_distance(pos) for pos, line in placed.items() if line == ("crop", "MELON"))
        self.assertLessEqual(berry_far, melon_near)


class Route14HubTests(unittest.TestCase):
    def test_waiting_animal_is_placed_before_another_purchase(self) -> None:
        tiles = [[None] * 5 for _ in range(5)]
        tiles[4][4] = {"kind": "PASTURE"}
        observation = {
            "player": 0, "step": 0, "day": 2, "hour": 0,
            "farms": [
                {"money": 8000, "tiles": tiles, "farmer": [4, 4], "hands": [], "hires_today": 0, "unlocked_quadrants": ["NW"]},
                {"money": 0, "tiles": tiles, "farmer": [0, 0], "hands": []},
            ],
            "market": {"prices": dict(PRICES)},
            "town": {"unlocked_shops": []},
            "private": {"shed": {"SHEEP": 1}, "inventories": [{}], "seeds": {}},
        }
        action = make_route14_agent()(observation)
        self.assertFalse(any(order[0] == "BUY_ANIMAL" for order in action["market"]))

    def test_worker_does_the_job_under_its_feet_instead_of_walking(self) -> None:
        # Bare corner to plant at (0,0); every other tile is a thirsty melon, so
        # the hand standing at (0,1) has watering under its feet on the way.
        tiles: list[list[object]] = [
            [{"kind": "PLANT", "crop": "MELON", "planted_day": 0, "yield_units": 0, "watered_today": False}] * 5
            for _ in range(5)
        ]
        tiles[0][0] = None
        observation = {
            "player": 0, "step": 0, "day": 3, "hour": 4,
            "farms": [
                {"money": 9000, "tiles": tiles, "farmer": [4, 4], "hands": [[0, 1]], "hires_today": 1, "unlocked_quadrants": ["NW"]},
                {"money": 0, "tiles": [[None] * 5 for _ in range(5)], "farmer": [0, 0], "hands": []},
            ],
            "market": {"prices": dict(PRICES), "inventory": {item: 10000 for item in PRICES}},
            "town": {"unlocked_shops": []},
            "private": {"shed": {}, "inventories": [{}, {}], "seeds": {"MELON": 2}},
        }
        action = make_route14_agent()(observation)
        self.assertEqual(action["hands"][0][0], "WATER")

    def test_loaded_worker_collects_a_ripe_tile_on_the_way_home(self) -> None:
        tiles: list[list[object]] = [[None] * 5 for _ in range(5)]
        tiles[1][2] = {"kind": "PLANT", "crop": "CARROT", "planted_day": 0, "yield_units": 3, "watered_today": True}
        observation = {
            "player": 0, "step": 0, "day": 3, "hour": 6,
            "farms": [
                # The hand at (2,2) is four steps from the shed and one step from
                # the ripe carrot, so collecting it costs nothing extra.
                {"money": 9000, "tiles": tiles, "farmer": [4, 4], "hands": [[2, 2]], "hires_today": 1, "unlocked_quadrants": ["NW"]},
                {"money": 0, "tiles": [[None] * 5 for _ in range(5)], "farmer": [0, 0], "hands": []},
            ],
            "market": {"prices": dict(PRICES), "inventory": {item: 10000 for item in PRICES}},
            "town": {"unlocked_shops": []},
            "private": {"shed": {}, "inventories": [{}, {"CARROT": 3}], "seeds": {}},
        }
        action = make_route14_agent()(observation)
        self.assertEqual(action["hands"][0], ["NORTH"])

    def test_feed_is_split_across_available_carers(self) -> None:
        tiles: list[list[object]] = [["LOCKED"] * 10 for _ in range(10)]
        for y in range(5):
            for x in range(5):
                tiles[y][x] = None
        for index in range(12):
            tiles[index // 5][index % 5] = {"kind": "PASTURE", "animal": "SHEEP", "placed_day": 0, "fed_today": False, "cared_today": False, "yield_units": 0}
        observation = {
            "player": 0, "step": 0, "day": 0, "hour": 0,
            "farms": [
                {"money": 5000, "tiles": tiles, "farmer": [4, 4], "hands": [[5, 4], [4, 5], [5, 5]], "hires_today": 3, "unlocked_quadrants": ["NW"]},
                {"money": 0, "tiles": tiles, "farmer": [0, 0], "hands": []},
            ],
            "market": {"prices": dict(PRICES)},
            "town": {"unlocked_shops": []},
            "private": {"shed": {"WHEAT": 12}, "inventories": [{}, {}, {}, {}], "seeds": {}},
        }
        action = make_route14_agent()(observation)
        pickups = [cmd for cmd in [action["farmer"], *action["hands"]] if cmd[0] == "PICKUP"]
        self.assertEqual(len(pickups), 4)
        self.assertEqual(sum(cmd[2] for cmd in pickups), 12)

    def test_new_plants_fit_plant_and_water_route(self) -> None:
        tiles = [["LOCKED"] * 10 for _ in range(10)]
        for y in range(5):
            for x in range(5):
                tiles[y][x] = None
        observation = {
            "player": 0, "day": 0, "hour": 0,
            "farms": [{"money": 3000, "tiles": tiles, "farmer": [4, 4], "hands": [], "hires_today": 0, "unlocked_quadrants": ["NW"]}],
            "market": {"prices": dict(PRICES)},
            "private": {"shed": {}, "inventories": [{}], "seeds": {}},
        }
        plan = hub_plan(observation, 3000)
        chosen = proposed_crop_slots(observation, 3000)[:plan.plant_slots]
        self.assertGreater(plan.plant_slots, 0)
        self.assertTrue(day_schedule(observation, plan.hires, chosen)["all_mandatory_tasks_witnessed"])
        self.assertIsNotNone(future_care_hires(observation, plants=chosen))

    def test_low_cash_preserves_existing_care(self) -> None:
        tiles = full_first_quadrant("STRAWBERRY")
        tiles[0][0] = None
        observation = {
            "player": 0, "day": 3, "hour": 0,
            "farms": [{"money": 210, "tiles": tiles, "farmer": [4, 4], "hands": [], "hires_today": 0, "unlocked_quadrants": ["NW"]}],
            "market": {"prices": dict(PRICES)},
            "private": {"shed": {}, "inventories": [{}], "seeds": {}},
        }
        plan = hub_plan(observation, 210)
        self.assertEqual(plan.seeds, [])
        self.assertEqual(plan.plant_slots, 0)
        self.assertFalse(plan.buy_land)
        self.assertIsNone(plan.animal)

    def test_hire_after_last_action_hour_is_not_bought(self) -> None:
        tiles = full_first_quadrant("STRAWBERRY")
        observation = {
            "player": 0, "day": 3, "hour": 23,
            "farms": [{"money": 5000, "tiles": tiles, "farmer": [4, 4], "hands": [], "hires_today": 0, "unlocked_quadrants": ["NW"]}],
            "market": {"prices": dict(PRICES)},
            "private": {"shed": {}, "inventories": [{}], "seeds": {}},
        }
        self.assertEqual(hub_plan(observation, 5000).hires, 0)

    def test_no_land_when_existing_harvest_exceeds_crew_capacity(self) -> None:
        observation = {
            "player": 0, "day": 4, "hour": 0,
            "farms": [{"money": 8000, "tiles": full_first_quadrant("MELON"), "farmer": [4, 4], "hands": [], "hires_today": 0, "unlocked_quadrants": ["NW"]}],
            "market": {"prices": dict(PRICES)},
            "private": {"shed": {}, "inventories": [{}], "seeds": {}},
        }
        self.assertFalse(hub_plan(observation, 8000).buy_land)

    def test_opening_day_does_not_hire_a_full_crew_just_because_people_are_cheap(self) -> None:
        tiles = [[None] * 5 for _ in range(5)]
        observation = {
            "player": 0,
            "day": 0,
            "hour": 0,
            "farms": [{"money": 3000, "tiles": tiles, "unlocked_quadrants": ["NW"]}],
            "market": {"prices": dict(PRICES)},
            "town": {"unlocked_shops": []},
            "private": {"shed": {}, "inventories": [{}], "seeds": {}},
        }
        plan = hub_plan(observation, 3000)
        self.assertLessEqual(plan.hires, 4)
        self.assertGreaterEqual(sum(order[2] for order in plan.seeds if order[1] == "MELON"), 12)
        self.assertFalse(plan.buy_land)

    def test_ripe_haul_hires_from_walking_not_a_fixed_day(self) -> None:
        plants = [[{"kind": "PLANT", "crop": "MELON", "yield_units": 5, "planted_day": 0}] * 5 for _ in range(5)]
        observation = {
            "player": 0,
            "day": 10,
            "hour": 0,
            "farms": [{"money": 3000, "tiles": plants, "unlocked_quadrants": ["NW"], "hands": [], "hires_today": 0}],
            "market": {"prices": dict(PRICES)},
            "town": {"unlocked_shops": []},
            "private": {"shed": {}, "inventories": [{}], "seeds": {}},
        }
        plan = hub_plan(observation, 3000)
        self.assertGreaterEqual(plan.hires, 11)
        self.assertTrue(plan.hire_first)
        self.assertEqual(plan.seeds, [])

    def test_empty_day_ten_does_not_force_eleven_hires(self) -> None:
        tiles = [[None] * 5 for _ in range(5)]
        observation = {
            "player": 0,
            "day": 10,
            "hour": 8,
            "farms": [{"money": 3000, "tiles": tiles, "unlocked_quadrants": ["NW"], "hands": [], "hires_today": 0}],
            "market": {"prices": dict(PRICES)},
            "town": {"unlocked_shops": []},
            "private": {"shed": {}, "inventories": [{}], "seeds": {}},
        }
        plan = hub_plan(observation, 3000)
        self.assertLess(plan.hires, 11)
        self.assertGreater(sum(order[2] for order in plan.seeds), 0)

    def test_animal_needs_runway_and_a_tile(self) -> None:
        tiles = [[{"kind": "PASTURE"}] * 5 for _ in range(2)] + [
            [{"kind": "PLANT", "crop": "MELON", "planted_day": 0, "yield_units": 0}] * 5 for _ in range(3)
        ]
        observation = {
            "player": 0,
            "day": 2,
            "hour": 3,
            "farms": [{"money": 8000, "tiles": tiles, "unlocked_quadrants": ["NW"]}],
            "market": {"prices": dict(PRICES)},
            "town": {"unlocked_shops": []},
            "private": {"shed": {}, "inventories": [{}], "seeds": {}},
        }
        self.assertEqual(hub_plan(observation, 8000).animal, "SHEEP")
        observation["farms"][0]["money"] = 400
        self.assertIsNone(hub_plan(observation, 400).animal)


    def test_land_waits_until_the_current_plot_is_full_and_cash_covers_new_tiles(self) -> None:
        plants = full_first_quadrant("STRAWBERRY")
        observation = {
            "player": 0,
            "day": 4,
            "hour": 0,
            "farms": [{"money": 8000, "tiles": plants, "unlocked_quadrants": ["NW"], "hands": [], "hires_today": 0}],
            "market": {"prices": dict(PRICES)},
            "town": {"unlocked_shops": []},
            "private": {"shed": {}, "inventories": [{}], "seeds": {}},
        }
        self.assertTrue(hub_plan(observation, 8000).buy_land)
        observation["farms"][0]["tiles"][0][0] = None
        self.assertFalse(hub_plan(observation, 8000).buy_land)


if __name__ == "__main__":
    unittest.main()
