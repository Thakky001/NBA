import unittest
from analysis.betting_math import (
    american_to_prob,
    prob_to_american,
    decimal_to_prob,
    prob_to_decimal,
    convert_odds,
    devig,
    calculate_edge,
    calculate_ev,
    calculate_kelly,
    evaluate_market_pick,
)
from analysis.trend_analyzer import TrendAnalyzer
from unittest.mock import MagicMock


class TestBettingMath(unittest.TestCase):
    """Unit tests for quantitative betting calculations and risk engine."""

    def test_odds_conversions(self):
        # American to probability
        self.assertAlmostEqual(american_to_prob(-110), 110 / 210, places=4)
        self.assertAlmostEqual(american_to_prob(100), 0.50, places=4)
        self.assertAlmostEqual(american_to_prob(-150), 150 / 250, places=4)
        self.assertAlmostEqual(american_to_prob(150), 100 / 250, places=4)

        # Probability to American
        self.assertAlmostEqual(prob_to_american(0.50), -100.0, delta=1.0)
        self.assertAlmostEqual(prob_to_american(0.60), -150.0, places=1)
        self.assertAlmostEqual(prob_to_american(0.40), 150.0, places=1)

        # Decimal to probability
        self.assertAlmostEqual(decimal_to_prob(2.0), 0.50, places=4)
        self.assertAlmostEqual(decimal_to_prob(1.91), 1.0 / 1.91, places=4)

        # Comprehensive convert_odds
        c = convert_odds(-110, from_format="american")
        self.assertAlmostEqual(c["probability"], 0.52381, places=4)
        self.assertEqual(c["american"], -110.0)

    def test_devig_standard_spread(self):
        """Standard -110 / -110 market should de-vig to exactly 50% / 50% with ~4.76% vig."""
        res = devig("-110,-110", format="american")
        self.assertEqual(len(res["fair_probs"]), 2)
        self.assertAlmostEqual(res["fair_probs"][0], 0.50, places=4)
        self.assertAlmostEqual(res["fair_probs"][1], 0.50, places=4)
        self.assertAlmostEqual(res["vig_pct"], 4.76, delta=0.05)
        self.assertAlmostEqual(res["overround"], 1.0476, delta=0.001)

    def test_devig_asymmetric_market(self):
        """Asymmetric market -150 / +130 should correctly normalize probabilities."""
        res = devig("-150,+130", format="american")
        fair_home = res["fair_probs"][0]
        fair_away = res["fair_probs"][1]
        self.assertAlmostEqual(fair_home + fair_away, 1.0, places=5)
        self.assertTrue(fair_home > 0.55)
        self.assertTrue(fair_away < 0.45)

    def test_devig_invalid_inputs(self):
        """Invalid inputs should raise ValueError or TypeError safely."""
        with self.assertRaises(ValueError):
            devig("-110")  # Need at least 2 outcomes
        with self.assertRaises(ValueError):
            devig("-110,-110", format="unknown_format")

    def test_edge_and_ev(self):
        """Verify edge and EV calculation."""
        # 55% model prob vs 50% fair market prob
        edge = calculate_edge(0.55, 0.50)
        self.assertAlmostEqual(edge, 0.05, places=4)

        # Expected Value at -110 odds
        ev = calculate_ev(model_prob=0.55, market_odds=-110, odds_format="american")
        # Net payout on $1 at -110 is 100/110 = 0.90909
        # EV = 0.55 * 0.90909 - 0.45 = 0.5000 - 0.45 = +0.05 (5% ROI)
        self.assertAlmostEqual(ev, 0.05, delta=0.005)

        # Break-even probability at -110 is ~52.38%
        ev_breakeven = calculate_ev(model_prob=0.52381, market_odds=-110, odds_format="american")
        self.assertAlmostEqual(ev_breakeven, 0.0, delta=0.005)

    def test_kelly_criterion(self):
        """Verify Kelly criterion stake calculations."""
        # Positive edge case
        k = calculate_kelly(model_prob=0.55, market_odds=-110, fraction=0.25)
        self.assertTrue(k["full_kelly"] > 0)
        self.assertAlmostEqual(k["fractional_kelly"], k["full_kelly"] * 0.25, places=4)
        self.assertAlmostEqual(k["recommended_stake_pct"], k["fractional_kelly"] * 100.0, places=2)

        # Negative edge case -> stake should be 0.0
        k_neg = calculate_kelly(model_prob=0.45, market_odds=-110, fraction=0.25)
        self.assertEqual(k_neg["full_kelly"], 0.0)
        self.assertEqual(k_neg["fractional_kelly"], 0.0)
        self.assertEqual(k_neg["recommended_stake_pct"], 0.0)

    def test_evaluate_market_pick(self):
        """Verify evaluate_market_pick all-in-one helper."""
        eval_res = evaluate_market_pick(model_prob=0.56, book_odds="-110,-110", pick_idx=0, kelly_fraction=0.25)
        self.assertEqual(eval_res["model_prob"], 0.56)
        self.assertEqual(eval_res["fair_prob"], 0.50)
        self.assertEqual(eval_res["edge"], 0.06)
        self.assertTrue(eval_res["has_value"])
        self.assertIn("+EV", eval_res["rating"])
        self.assertTrue(eval_res["quarter_kelly"] > 0.0)

    def test_trend_analyzer_integration(self):
        """Verify TrendAnalyzer includes EV, Kelly, and Fair Prob in outputs."""
        mock_fe = MagicMock()
        mock_fe.compute_matchup_features.return_value = {
            'home_avg_points_scored': 118.0,
            'away_avg_points_scored': 114.0,
            'combined_avg_10': 232.0,
            'combined_avg_season': 230.0,
            'home_pace': 102.0,
            'away_pace': 100.0,
            'avg_pace': 101.0,
            'home_net_rating': 6.0,
            'away_net_rating': -2.0,
            'home_rest_days': 2,
            'away_rest_days': 1,
            'home_roster_rating': 85.0,
            'away_roster_rating': 80.0,
        }
        analyzer = TrendAnalyzer(mock_fe)

        # Test O/U Analysis with clear Over edge
        ou_res = analyzer.analyze_over_under('Boston Celtics', 'Miami Heat', over_under_line=215.0)
        self.assertIn('ev', ou_res)
        self.assertIn('kelly_quarter', ou_res)
        self.assertIn('fair_prob', ou_res)
        self.assertIn('market_prob', ou_res)
        self.assertIn('kelly_recommendation', ou_res)
        self.assertIn('STRONG OVER', ou_res['recommendation'])
        self.assertTrue(ou_res['ev'] > 0.0)
        self.assertTrue(ou_res['kelly_quarter'] > 0.0)
        self.assertIn('Value & Sizing', ou_res['summary'])

        # Test Spread Analysis with clear Home edge
        sp_res = analyzer.analyze_spread('Boston Celtics', 'Miami Heat', spread_line=-1.5)
        self.assertIn('ev', sp_res)
        self.assertIn('kelly_quarter', sp_res)
        self.assertIn('fair_prob', sp_res)
        self.assertIn('market_prob', sp_res)
        self.assertIn('kelly_recommendation', sp_res)
        self.assertIn('STRONG HOME', sp_res['recommendation'])
        self.assertTrue(sp_res['ev'] > 0.0)
        self.assertTrue(sp_res['kelly_quarter'] > 0.0)
        self.assertIn('Value & Sizing', sp_res['summary'])

    def test_kelly_safety_cap(self):
        """Verify Kelly criterion supports optional cap and unconstrained raw mode."""
        # Extreme win prob (e.g. 0.90) with explicit safety cap 5.0%
        k_capped = calculate_kelly(model_prob=0.90, market_odds=-110, fraction=0.25, max_cap=0.05)
        self.assertEqual(k_capped["recommended_stake_pct"], 5.0)
        self.assertEqual(k_capped["capped_kelly"], 0.05)
        self.assertGreater(k_capped["fractional_kelly"], 0.05)

        # Pure raw mode (default max_cap=None): unconstrained fractional Kelly
        k_raw = calculate_kelly(model_prob=0.90, market_odds=-110, fraction=0.25)
        self.assertAlmostEqual(k_raw["recommended_stake_pct"], k_raw["fractional_kelly"] * 100.0, places=2)
        self.assertGreater(k_raw["recommended_stake_pct"], 15.0)

    def test_anomaly_detection_and_raw_prob(self):
        """Verify extreme edge (>8.0 pts) triggers anomaly flag while preserving unconstrained win_prob."""
        mock_fe = MagicMock()
        mock_fe.compute_matchup_features.return_value = {
            'combined_avg_10': 250.0,
            'combined_avg_season': 248.0,
            'home_pace': 100.0,
            'away_pace': 100.0,
            'avg_pace': 100.0,
        }
        analyzer = TrendAnalyzer(mock_fe)
        # Line is 220. With 65% market anchor: projected_total is 230.5, edge is +10.5 pts (> 8 pts anomaly)
        ou_res = analyzer.analyze_over_under('Boston Celtics', 'Miami Heat', over_under_line=220.0)
        self.assertTrue(ou_res['is_anomaly'])
        self.assertIsNotNone(ou_res['anomaly_warning'])
        self.assertGreater(ou_res['win_prob'], 0.75)
        self.assertEqual(ou_res['win_prob'], ou_res['raw_win_prob'])
        self.assertIn('ANOMALY', ou_res['summary'].upper())

        # When testing unanchored (market_weight=0.0): edge is +30 pts and win_prob > 0.85
        ou_unanchored = analyzer.analyze_over_under('Boston Celtics', 'Miami Heat', over_under_line=220.0, market_weight=0.0)
        self.assertGreater(ou_unanchored['win_prob'], 0.85)


if __name__ == '__main__':
    unittest.main()
