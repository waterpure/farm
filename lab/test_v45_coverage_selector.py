"""Offline checks for the day-6 same-family 70% coverage selector."""

from __future__ import annotations

import unittest

from lab.baselines import load_v45_base_module
from lab.v45_variants import _family_routes, _route_family, _select_coverage_route


def _observation(shops: list[str], *, day: int = 6) -> dict:
    empty_row = [None] * 10
    tiles = [list(empty_row) for _ in range(10)]
    return {
        "player": 0,
        "day": day,
        "farms": [{"tiles": tiles, "money": 2000}, {"tiles": [], "money": 2000}],
        "town": {"unlocked_shops": shops},
        "market": {"prices": {}, "inventory": {}},
    }


class CoverageSelectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_v45_base_module()

    def test_pizza_yarn_stays_in_yarn_family_and_cannot_pick_pizza_route(self) -> None:
        family = _route_family(self.module, 7)
        allowed = _family_routes(self.module, 7)
        self.assertEqual(family, "yarn")
        self.assertIn(7, allowed)
        self.assertNotIn(123, allowed)
        decision = _select_coverage_route(
            self.module,
            _observation(["PIZZA_SHOP", "YARN_STORE"]),
            7,
        )
        self.assertEqual(decision["route"], 7)
        self.assertNotEqual(decision["route"], 123)

    def test_double_pizza_may_consider_heavy_milk_route_but_not_yarn_route(self) -> None:
        allowed = _family_routes(self.module, 105)
        self.assertEqual(_route_family(self.module, 105), "exp")
        self.assertIn(123, allowed)
        self.assertNotIn(7, allowed)
        decision = _select_coverage_route(
            self.module,
            _observation(["PIZZA_SHOP", "PIZZA_SHOP"]),
            105,
        )
        self.assertIn(decision["route"], allowed)
        self.assertNotEqual(decision["route"], 7)


if __name__ == "__main__":
    unittest.main()
