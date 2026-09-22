"""Regression checks for the transparent 70/30 shadow economics."""

from __future__ import annotations

import unittest

from lab.portfolio_model import allocate_ranked_capacity, choose_portfolio, evaluate_lines
from lab.task_router import route_mandatory_tasks


def _opening_observation() -> dict:
    return {
        "player": 0,
        "day": 0,
        "farms": [{"money": 3000, "tiles": []}, {"money": 3000, "tiles": []}],
        "market": {"prices": {"WHEAT": 25}, "inventory": {}},
        "town": {"unlocked_shops": []},
    }


class PortfolioEconomicsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.lines = {line.line: line for line in evaluate_lines(_opening_observation())}

    def test_ongoing_crops_have_four_total_events_not_four_per_event(self) -> None:
        self.assertEqual((self.lines["tomato"].yield_events, self.lines["tomato"].expected_output_units), (4, 4))
        self.assertEqual((self.lines["strawberry"].yield_events, self.lines["strawberry"].expected_output_units), (4, 4))

    def test_animal_care_outputs_and_field_action_lower_bounds(self) -> None:
        # Directly matches the official day-refresh semantics for an animal
        # placed at day 0, promptly harvested, fed daily, and cared daily only
        # for the cared alternatives. These are lower bounds: no walking.
        expected = {
            "goose_eggs_basic": (27, 41),
            "goose_eggs_cared": (56, 78),
            "cow_milk_basic": (12, 36),
            "cow_milk_cared": (39, 71),
            "sheep_wool_basic": (9, 36),
            "sheep_wool_cared": (38, 73),
        }
        actual = {name: (self.lines[name].expected_output_units, self.lines[name].field_action_lower_bound) for name in expected}
        self.assertEqual(actual, expected)

    def test_route_witness_accounts_for_walking_and_flags_unfittable_work(self) -> None:
        observation = {
            "player": 0,
            "hour": 0,
            "farms": [{"farmer": [0, 0], "hands": [[4, 0]]}, {}],
        }
        tasks = [
            {"kind": "FEED", "target": [2, 0], "priority": 100},
            {"kind": "WATER", "target": [4, 2], "priority": 80},
        ]
        witness = route_mandatory_tasks(observation, tasks)
        self.assertTrue(witness["all_mandatory_tasks_witnessed"])
        self.assertEqual([route["used_turns"] for route in witness["routes"]], [3, 3])

        observation["hour"] = 23
        too_late = route_mandatory_tasks(observation, [{"kind": "FEED", "target": [2, 0], "priority": 100}])
        self.assertFalse(too_late["all_mandatory_tasks_witnessed"])
        self.assertEqual(len(too_late["unassigned_groups"]), 1)

    def test_portfolio_secondary_is_a_different_product(self) -> None:
        portfolio = choose_portfolio(_opening_observation())
        self.assertIsNotNone(portfolio["primary"])
        self.assertIsNotNone(portfolio["secondary"])
        self.assertNotEqual(portfolio["primary"]["output"], portfolio["secondary"]["output"])

    def test_primary_constraint_releases_slots_to_secondary(self) -> None:
        allocation = allocate_ranked_capacity(
            ["WOOL", "STRAWBERRY", "MILK"],
            available_slots=10,
            safe_slot_cap_by_output={"WOOL": 5, "STRAWBERRY": 10, "MILK": 10},
        )
        self.assertEqual(allocation["primary_target_slots"], 7)
        self.assertEqual(
            [(row["output"], row["assigned_slots"]) for row in allocation["allocations"]],
            [("WOOL", 5), ("STRAWBERRY", 5)],
        )
        self.assertEqual(allocation["unallocated_slots"], 0)

    def test_ideal_split_then_third_product_and_reserve(self) -> None:
        normal = allocate_ranked_capacity(
            ["WOOL", "STRAWBERRY", "MILK"],
            available_slots=10,
            safe_slot_cap_by_output={"WOOL": 7, "STRAWBERRY": 3, "MILK": 10},
        )
        self.assertEqual(
            [(row["output"], row["assigned_slots"]) for row in normal["allocations"]],
            [("WOOL", 7), ("STRAWBERRY", 3)],
        )
        constrained = allocate_ranked_capacity(
            ["WOOL", "STRAWBERRY", "MILK"],
            available_slots=10,
            safe_slot_cap_by_output={"WOOL": 7, "STRAWBERRY": 1, "MILK": 1},
        )
        self.assertEqual(
            [(row["output"], row["assigned_slots"]) for row in constrained["allocations"]],
            [("WOOL", 7), ("STRAWBERRY", 1), ("MILK", 1)],
        )
        self.assertEqual(constrained["unallocated_slots"], 1)


if __name__ == "__main__":
    unittest.main()
