"""Season ledger: real fills, same seed, and restored engine hooks."""

from __future__ import annotations

import unittest

from kaggle_environments.envs.kaggriculture import kaggriculture as game

from lab.runner import _resolve_agent, run_match
from lab.season_eval import SeasonResult, evaluate_seasons, run_season, summarize


def idle_opponent(observation: dict, configuration: dict | None = None) -> dict:
    del observation, configuration
    return {"farmer": ["PASS"], "hands": [], "market": []}


def clerk(observation: dict, configuration: dict | None = None) -> dict:
    """Buy, sell one wheat, fail a melon sale, hire, and buy a goose. Hour 0 only."""

    del configuration
    if int(observation.get("hour") or 0) != 0:
        return {"farmer": ["PASS"], "hands": [], "market": []}
    return {
        "farmer": ["PASS"],
        "hands": [],
        "market": [
            ["BUY_PRODUCT", "WHEAT", 1],
            ["SELL", "WHEAT", 1],
            ["SELL", "MELON", 2],
            ["HIRE"],
            ["BUY_SEED", "WHEAT", 1],
            ["BUY_ANIMAL", "GOOSE", 1],
        ],
    }


def crash_agent(observation: dict, configuration: dict | None = None) -> dict:
    del observation, configuration
    raise RuntimeError("season-crash")


class SeasonEvalTest(unittest.TestCase):
    def test_builtin_starter_resolves_to_callable_for_seated_runs(self) -> None:
        agent = _resolve_agent("starter")
        self.assertTrue(callable(agent))
        action = agent({"farms": [], "player": 0, "private": {}}, {"episodeSteps": 6})
        self.assertEqual(action["farmer"], ["PASS"])

    def test_builtin_starter_can_be_the_evaluated_baseline(self) -> None:
        result = run_season("starter", "starter", seed=1, steps=6)
        self.assertEqual(result.status, "DONE")
        self.assertIsNone(result.exception)
        self.assertEqual(result.opponent, "starter")

    def test_short_season_returns_a_result(self) -> None:
        result = run_season("region_phase1", "starter", seed=1, steps=6)
        self.assertIsInstance(result, SeasonResult)
        self.assertEqual(result.status, "DONE")
        self.assertEqual(result.seed, 1)
        self.assertEqual(result.agent, "region_phase1")
        self.assertEqual(result.opponent, "starter")
        self.assertEqual(result.seat, "left")
        self.assertEqual(result.steps, 6)
        self.assertIsInstance(result.final_money, int)
        self.assertIsInstance(result.daily_money, tuple)

    def test_same_seed_is_deterministic(self) -> None:
        first = run_season("region_phase1", "starter", seed=1, steps=6)
        second = run_season("region_phase1", "starter", seed=1, steps=6)
        self.assertEqual(first.final_money, second.final_money)
        self.assertEqual(first.terminal_reward, second.terminal_reward)
        self.assertEqual(first.sell_revenue, second.sell_revenue)
        self.assertEqual(first.total_spend, second.total_spend)

    def test_cash_ledger_counts_only_successful_fills(self) -> None:
        result = run_season(clerk, idle_opponent, seed=3, steps=4)
        self.assertEqual(result.status, "DONE")
        self.assertEqual(result.initial_money, 3000)
        self.assertEqual(result.seed_spend, game.CROPS["WHEAT"]["seed"])
        self.assertEqual(result.seed_units_bought, 1)
        self.assertEqual(result.animal_spend, game.ANIMALS["GOOSE"]["cost"])
        self.assertEqual(result.animal_units_bought, 1)
        self.assertEqual(result.hire_spend, game._hire_cost(0))
        self.assertEqual(result.successful_hires, 1)
        self.assertEqual(result.sell_units, 1)
        self.assertGreater(result.sell_revenue, 0)
        self.assertEqual(result.sell_revenue, result.sell_revenue_by_item["WHEAT"])
        self.assertEqual(result.sell_revenue_by_item["MELON"], 0)
        self.assertGreater(result.product_spend, 0)
        self.assertGreaterEqual(result.failed_market_orders, 1)
        self.assertEqual(result.land_spend, 0)
        self.assertEqual(result.other_income, 0)
        self.assertEqual(result.other_spend, 0)
        self.assertEqual(result.cash_reconciliation_error, 0)
        self.assertEqual(
            result.initial_money + result.sell_revenue - result.total_spend,
            result.final_money,
        )
        self.assertEqual(result.net_cash_change, result.final_money - result.initial_money)
        self.assertTrue(result.reward_equals_final_money)
        self.assertGreaterEqual(result.pass_hours, 1)
        self.assertEqual(result.move_hours, 0)
        self.assertEqual(result.work_hours, 0)

    def test_instrumentation_does_not_change_the_score(self) -> None:
        plain = run_match(clerk, idle_opponent, steps=4, seed=3)
        watched = run_season(clerk, idle_opponent, seed=3, steps=4)
        self.assertEqual(watched.final_money, int(round(float(plain["rewards"][0]))))

    def test_paired_games_share_the_seed(self) -> None:
        report = evaluate_seasons(
            "region_phase1",
            "route14_phase1",
            "starter",
            seeds=[4],
            steps=4,
        )
        region, baseline = report["games"]
        self.assertEqual(region.agent, "region_phase1")
        self.assertEqual(baseline.agent, "route14_phase1")
        self.assertEqual(region.seed, baseline.seed)
        self.assertEqual(region.opponent, baseline.opponent)
        self.assertEqual(region.seat, baseline.seat)
        self.assertEqual(region.steps, baseline.steps)
        self.assertEqual(report["pairs"][0]["seed"], 4)
        self.assertEqual(
            report["pairs"][0]["delta_money"],
            region.final_money - baseline.final_money,
        )

    def test_summary_mean_median_and_wins(self) -> None:
        pairs = [
            {
                "seed": 1,
                "region_final_money": 110,
                "baseline_final_money": 100,
                "delta_money": 10,
                "region_sell_revenue": 50,
                "baseline_sell_revenue": 40,
                "region_total_spend": 20,
                "baseline_total_spend": 30,
                "region_hires": 2,
                "baseline_hires": 1,
                "region_failed_actions": 3,
                "baseline_failed_actions": 4,
                "region_status": "DONE",
                "baseline_status": "DONE",
            },
            {
                "seed": 2,
                "region_final_money": 90,
                "baseline_final_money": 95,
                "delta_money": -5,
                "region_sell_revenue": 10,
                "baseline_sell_revenue": 10,
                "region_total_spend": 5,
                "baseline_total_spend": 5,
                "region_hires": 0,
                "baseline_hires": 0,
                "region_failed_actions": 1,
                "baseline_failed_actions": 1,
                "region_status": "DONE",
                "baseline_status": "DONE",
            },
            {
                "seed": 3,
                "region_final_money": 80,
                "baseline_final_money": 80,
                "delta_money": 0,
                "region_sell_revenue": 0,
                "baseline_sell_revenue": 1,
                "region_total_spend": 0,
                "baseline_total_spend": 0,
                "region_hires": 1,
                "baseline_hires": 4,
                "region_failed_actions": 0,
                "baseline_failed_actions": 2,
                "region_status": "DONE",
                "baseline_status": "DONE",
            },
        ]
        games = [
            SeasonResult(seed=1, agent="candidate", opponent="starter", seat="left", reward_equals_final_money=True),
            SeasonResult(seed=1, agent="baseline", opponent="starter", seat="left", reward_equals_final_money=True),
            SeasonResult(seed=2, agent="candidate", opponent="starter", seat="left", reward_equals_final_money=False),
            SeasonResult(seed=2, agent="baseline", opponent="starter", seat="left", reward_equals_final_money=True),
        ]
        summary = summarize(pairs, games)
        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["region_mean_final_money"], round((110 + 90 + 80) / 3, 3))
        self.assertEqual(summary["baseline_mean_final_money"], round((100 + 95 + 80) / 3, 3))
        self.assertEqual(summary["mean_delta"], round((10 - 5) / 3, 3))
        self.assertEqual(summary["median_delta"], 0)
        self.assertEqual(summary["region_wins"], 1)
        self.assertEqual(summary["ties"], 1)
        self.assertEqual(summary["baseline_wins"], 1)
        self.assertEqual(summary["best_delta"], 10)
        self.assertEqual(summary["worst_delta"], -5)
        self.assertEqual(summary["best_seed"], 1)
        self.assertEqual(summary["worst_seed"], 2)
        self.assertEqual(summary["region_min_final_money"], 80)
        self.assertEqual(summary["region_max_final_money"], 110)
        self.assertEqual(summary["region_mean_sell_revenue"], round((50 + 10) / 3, 3))
        self.assertEqual(summary["baseline_mean_sell_revenue"], round((40 + 10 + 1) / 3, 3))
        self.assertEqual(summary["region_mean_spend"], round((20 + 5) / 3, 3))
        self.assertEqual(summary["baseline_mean_spend"], round((30 + 5) / 3, 3))
        self.assertEqual(summary["region_mean_hires"], round((2 + 1) / 3, 3))
        self.assertEqual(summary["baseline_mean_hires"], round((1 + 4) / 3, 3))
        self.assertEqual(summary["region_total_failed_actions"], 4)
        self.assertEqual(summary["baseline_total_failed_actions"], 7)
        self.assertFalse(summary["terminal_reward_equals_final_money"])
        self.assertTrue(summary["all_done"])

    def test_daily_curve_records_each_finished_day(self) -> None:
        midnight = run_season(idle_opponent, idle_opponent, seed=1, steps=25)
        self.assertEqual([point.day for point in midnight.daily], [0])
        self.assertEqual(midnight.daily_money[-1], midnight.final_money)

        stopped = run_season(idle_opponent, idle_opponent, seed=1, steps=48)
        self.assertEqual(stopped.status, "DONE")
        self.assertEqual([point.day for point in stopped.daily], [0, 1])
        self.assertEqual(stopped.daily_money, tuple(point.money for point in stopped.daily))
        self.assertEqual(stopped.daily_money[-1], stopped.final_money)
        self.assertTrue(all(point.owned_tiles > 0 for point in stopped.daily))

    def test_instrumentation_is_restored_after_a_crash(self) -> None:
        before = (
            game._apply_unit_action,
            game._commit_unit,
            game._do_hire,
            game._do_buy_land,
        )
        result = run_season(crash_agent, idle_opponent, seed=1, steps=4)
        after = (
            game._apply_unit_action,
            game._commit_unit,
            game._do_hire,
            game._do_buy_land,
        )
        self.assertEqual(before, after)
        self.assertEqual(result.status, "ERROR")
        self.assertIn("season-crash", result.exception or "")


if __name__ == "__main__":
    unittest.main()
