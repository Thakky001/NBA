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
    calculate_live_pace_projection,
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
        """Verify Kelly criterion supports default cap (2.5%), custom cap, and unconstrained raw mode."""
        # Extreme win prob (e.g. 0.90) with default safety cap 2.5%
        k_default = calculate_kelly(model_prob=0.90, market_odds=-110, fraction=0.25)
        self.assertEqual(k_default["recommended_stake_pct"], 2.5)
        self.assertEqual(k_default["capped_kelly"], 0.025)
        self.assertGreater(k_default["fractional_kelly"], 0.025)

        # Extreme win prob with explicit custom safety cap 5.0%
        k_capped = calculate_kelly(model_prob=0.90, market_odds=-110, fraction=0.25, max_cap=0.05)
        self.assertEqual(k_capped["recommended_stake_pct"], 5.0)
        self.assertEqual(k_capped["capped_kelly"], 0.05)
        self.assertGreater(k_capped["fractional_kelly"], 0.05)

        # Pure raw mode (explicit max_cap=None): unconstrained fractional Kelly
        k_raw = calculate_kelly(model_prob=0.90, market_odds=-110, fraction=0.25, max_cap=None)
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


class TestLivePaceProjection(unittest.TestCase):
    """Unit tests for live scoring pace, Halftime, quarter breaks, and Overtime projections."""

    def test_pace_regular_in_play(self):
        """Test in-game live pace applies Bayesian shrinkage towards pregame baseline."""
        # Q1 with 6:00 remaining -> exactly 6.0 minutes elapsed (30 pts scored)
        # Without pregame total, defaults to 228.0 prior (4.75 pts/min)
        # w = 6/48 = 0.125. Live = 5.0 pts/min. Rem rate = 0.125*5 + 0.875*4.75 = 4.78125
        # Proj total = 30 + 42 * 4.78125 = 230.8 pts (regressed from naive 240.0)
        res = calculate_live_pace_projection(period=1, clock_str="06:00", home_score=15, away_score=15)
        self.assertIsNotNone(res)
        self.assertEqual(res['elapsed_mins'], 6.0)
        self.assertEqual(res['target_mins'], 48.0)
        self.assertEqual(res['proj_total'], 230.8)
        self.assertEqual(res['proj_margin'], 0.0)
        self.assertEqual(res['scoring_rate'], 5.0)
        self.assertEqual(res['phase'], 'IN_PLAY')
        self.assertIn("Q1 06:00", res['status_label'])

    def test_hawks_magic_q1_early_blowout_prevention(self):
        """Hawks @ Magic bug fix: 42 pts in 7.75 min must not produce 260.1 total or +24.8 margin."""
        # Hawks home 19, Magic away 23 (Magic lead by 4)
        # Pregame total 237.5, pregame spread +2.5 (Hawks +2.5 / Magic -2.5)
        res = calculate_live_pace_projection(
            period=1, clock_str="04:15", home_score=19, away_score=23,
            pregame_total=237.5, pregame_spread=2.5
        )
        self.assertIsNotNone(res)
        self.assertEqual(res['elapsed_mins'], 7.8)
        # Total must regress to realistic ~244 pts, NOT 260.1!
        self.assertLess(res['proj_total'], 250.0)
        self.assertGreater(res['proj_total'], 240.0)
        # Margin must regress to ~ -9.1 pts (Magic by 9), NOT -24.8 pts!
        self.assertGreater(res['proj_margin'], -12.0)
        self.assertLess(res['proj_margin'], -6.0)

    def test_pace_halftime_explicit(self):
        """Test explicit 'Halftime' or 'Half' clock string sets 24.0 mins elapsed."""
        for label in ["Halftime", "HALFTIME", "Half", "half", "End of 2nd"]:
            res = calculate_live_pace_projection(period=2, clock_str=label, home_score=55, away_score=45)
            self.assertIsNotNone(res)
            self.assertEqual(res['elapsed_mins'], 24.0)
            self.assertEqual(res['target_mins'], 48.0)
            self.assertEqual(res['phase'], 'HALFTIME')
            self.assertEqual(res['status_label'], 'ช่วงพักครึ่ง (Halftime)')
            # 100 pts at half, live rate 4.167, prior 4.75 -> rem rate 4.458 -> 100 + 107.0 = 207.0
            self.assertEqual(res['proj_total'], 207.0)
            # +10 lead at half, 50% regressed 2nd half outscores by +5 -> proj_margin = 15.0
            self.assertEqual(res['proj_margin'], 15.0)

    def test_pace_halftime_zero_clock(self):
        """Test period 2 with clock '0.0' or '0:00' is identified as Halftime."""
        for zero_clk in ["0.0", "0:00", "0"]:
            res = calculate_live_pace_projection(period=2, clock_str=zero_clk, home_score=50, away_score=50)
            self.assertIsNotNone(res)
            self.assertEqual(res['elapsed_mins'], 24.0)
            self.assertEqual(res['phase'], 'HALFTIME')
            self.assertEqual(res['status_label'], 'ช่วงพักครึ่ง (Halftime)')
            self.assertEqual(res['proj_total'], 207.0)

    def test_pace_quarter_breaks(self):
        """Test End of 1st (12 mins) and End of 3rd (36 mins) quarter breaks."""
        # End of Q1 (60 pts in 12 min -> 5.0 rate, w=0.25 -> rem rate 0.25*5 + 0.75*4.75 = 4.8125 -> 60 + 36*4.8125 = 233.25)
        q1_res = calculate_live_pace_projection(period=1, clock_str="End of 1st", home_score=30, away_score=30)
        self.assertIsNotNone(q1_res)
        self.assertEqual(q1_res['elapsed_mins'], 12.0)
        self.assertEqual(q1_res['phase'], 'QUARTER_BREAK')
        self.assertEqual(q1_res['status_label'], 'จบควอเตอร์ 1 (End of Q1)')
        self.assertEqual(q1_res['proj_total'], 233.2)

        # End of Q3 (180 pts in 36 min -> 5.0 rate, w=0.75 -> rem rate 0.75*5 + 0.25*4.75 = 4.9375 -> 180 + 12*4.9375 = 239.25)
        q3_res = calculate_live_pace_projection(period=3, clock_str="End of 3rd", home_score=90, away_score=90)
        self.assertIsNotNone(q3_res)
        self.assertEqual(q3_res['elapsed_mins'], 36.0)
        self.assertEqual(q3_res['phase'], 'QUARTER_BREAK')
        self.assertEqual(q3_res['status_label'], 'จบควอเตอร์ 3 (End of Q3)')
        self.assertEqual(q3_res['proj_total'], 239.2)

    def test_pace_overtime(self):
        """Test Overtime (OT1, Period 5) scales expected total to 53.0 minutes."""
        # Period 5 with 2:30 remaining in 5-minute OT1 -> 48 + 2.5 = 50.5 mins elapsed
        ot_res = calculate_live_pace_projection(period=5, clock_str="02:30", home_score=105, away_score=100)
        self.assertIsNotNone(ot_res)
        self.assertEqual(ot_res['elapsed_mins'], 50.5)
        self.assertEqual(ot_res['target_mins'], 53.0)
        self.assertEqual(ot_res['phase'], 'OVERTIME')
        self.assertIn("OT1", ot_res['status_label'])
        self.assertGreater(ot_res['proj_total'], 210.0)

    def test_pace_invalid_inputs(self):
        """Test safety guards return None on invalid or zero scores/times."""
        self.assertIsNone(calculate_live_pace_projection(period=1, clock_str="12:00", home_score=None, away_score=10))
        self.assertIsNone(calculate_live_pace_projection(period=1, clock_str="12:00", home_score=0, away_score=0))
        self.assertIsNone(calculate_live_pace_projection(period=1, clock_str="11:50", home_score=2, away_score=0))  # <= 0.5 mins
        self.assertIsNone(calculate_live_pace_projection(period=1, clock_str="invalid_clock", home_score=10, away_score=10))

    def test_line_movement_live_isolation(self):
        """Verify is_live=True suppresses STEAM_WARNING and SHARP_CONFIRMATION."""
        from analysis.betting_math import analyze_line_movement
        # When game is live and total shifts by 5.0 pts (e.g. 237.5 -> 242.5):
        # Must NOT trigger STEAM_WARNING!
        res_live = analyze_line_movement(open_line=237.5, current_line=242.5, market_type='total', recommendation='STRONG UNDER', is_live=True)
        self.assertEqual(res_live['status'], 'LIVE_IN_PLAY')
        self.assertEqual(res_live['verdict'], 'LIVE_TRACKING')
        self.assertIn('Live', res_live['badge'])

        # When game is pregame (is_live=False), 5.0 pt shift opposing model DOES trigger STEAM_WARNING
        res_pregame = analyze_line_movement(open_line=237.5, current_line=242.5, market_type='total', recommendation='STRONG UNDER', is_live=False)
        self.assertEqual(res_pregame['status'], 'STEAM_WARNING')
        self.assertEqual(res_pregame['verdict'], 'FOLLOW_MARKET')


if __name__ == '__main__':
    unittest.main()
