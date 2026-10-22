import unittest
from analysis.betting_math import market_anchored_projection, analyze_line_movement
from analysis.trend_analyzer import TrendAnalyzer
from unittest.mock import MagicMock


class TestLineMovementAndAnchoring(unittest.TestCase):
    """Unit tests for Market-Anchored Projections and Line Movement Detection."""

    def test_market_anchored_projection(self):
        """Verify Bayesian market anchoring properly blends market line with model raw projection."""
        # Knicks vs 76ers scenario: Market 232.5, Raw model 212.9
        # 0.65 * 232.5 + 0.35 * 212.9 = 151.125 + 74.515 = 225.64
        anchored = market_anchored_projection(raw_model_val=212.9, market_line=232.5, market_weight=0.65)
        self.assertAlmostEqual(anchored, 225.64, places=2)

        # 50/50 blend
        half_blend = market_anchored_projection(raw_model_val=200.0, market_line=220.0, market_weight=0.50)
        self.assertEqual(half_blend, 210.0)

        # 0% market weight -> pure raw model projection
        raw_only = market_anchored_projection(raw_model_val=212.9, market_line=232.5, market_weight=0.0)
        self.assertEqual(raw_only, 212.9)

        # 100% market weight -> market line only
        market_only = market_anchored_projection(raw_model_val=212.9, market_line=232.5, market_weight=1.0)
        self.assertEqual(market_only, 232.5)

        # None or invalid market line fallback
        fallback = market_anchored_projection(raw_model_val=215.0, market_line=None)
        self.assertEqual(fallback, 215.0)

    def test_line_movement_totals_aligned_with_model(self):
        lm = analyze_line_movement(open_line=235.0, current_line=232.5, market_type='total', recommendation='STRONG UNDER')
        self.assertTrue(lm['has_movement'])
        self.assertEqual(lm['status'], 'MOVE_WITH_MODEL')
        self.assertIn('Move With Model', lm['badge'])
        self.assertAlmostEqual(lm['diff'], -2.5, places=1)
        self.assertEqual(lm['verdict'], 'MOVE_ALIGNED')
        self.assertIsNone(lm['market_flow_prob'])
        self.assertEqual(lm['market_flow_direction'], 'ต่ำ (UNDER)')

    def test_line_movement_totals_contrary_to_model(self):
        # A line move alone cannot identify sharp action or quantify probability.
        lm = analyze_line_movement(open_line=229.5, current_line=232.5, market_type='total', recommendation='STRONG UNDER')
        self.assertTrue(lm['has_movement'])
        self.assertEqual(lm['status'], 'MOVE_AGAINST_MODEL')
        self.assertIn('Move Against Model', lm['badge'])
        self.assertAlmostEqual(lm['diff'], 3.0, places=1)
        self.assertEqual(lm['verdict'], 'MOVE_CONFLICT')
        self.assertIsNone(lm['market_flow_prob'])
        self.assertEqual(lm['market_flow_direction'], 'สูง (OVER)')

        # Case B: Mild steam (diff = +1.0 <= 1.5) -> FOLLOW_MODEL
        lm_mild = analyze_line_movement(open_line=228.5, current_line=229.5, market_type='total', recommendation='STRONG UNDER')
        self.assertEqual(lm_mild['status'], 'MOVE_AGAINST_MODEL')
        self.assertEqual(lm_mild['verdict'], 'MOVE_CONFLICT')
        self.assertIsNone(lm_mild['market_flow_prob'])

    def test_line_movement_totals_stable(self):
        """Line difference < 0.5 -> Stable Line."""
        lm = analyze_line_movement(open_line=225.0, current_line=225.0, market_type='total', recommendation='LEAN OVER')
        self.assertTrue(lm['has_movement'])
        self.assertEqual(lm['status'], 'STABLE')
        self.assertIn('Stable Line', lm['badge'])
        self.assertEqual(lm['verdict'], 'STABLE_MODEL')
        self.assertIn('ตามสถิติ', lm['verdict_title'])
        self.assertIsNone(lm['market_flow_prob'])

    def test_line_movement_spread_model_alignment(self):
        """Spread movement toward home/away properly matches model pick."""
        # Home spread moves from -3.5 to -4.5 (1 pt toward Home)
        # Case A: Model likes HOME
        lm_home = analyze_line_movement(open_line=-3.5, current_line=-4.5, market_type='spread', recommendation='STRONG HOME')
        self.assertEqual(lm_home['status'], 'MOVE_WITH_MODEL')
        self.assertIn('Move With Model', lm_home['badge'])

        # Case B: Model likes AWAY
        lm_away = analyze_line_movement(open_line=-3.5, current_line=-4.5, market_type='spread', recommendation='STRONG AWAY')
        self.assertEqual(lm_away['status'], 'MOVE_AGAINST_MODEL')
        self.assertIn('Move Against Model', lm_away['badge'])

    def test_trend_analyzer_full_integration(self):
        """Verify full_matchup_analysis accepts open lines and returns anchored & line movement metrics."""
        mock_fe = MagicMock()
        mock_fe.compute_matchup_features.return_value = {
            'combined_avg_10': 210.0,
            'combined_avg_season': 212.0,
            'home_pace': 99.0,
            'away_pace': 99.0,
            'avg_pace': 99.0,
            'home_net_rating': 4.0,
            'away_net_rating': -2.0,
            'home_rest_days': 2,
            'away_rest_days': 1,
            'home_roster_power': 85.0,
            'away_roster_power': 80.0,
        }
        analyzer = TrendAnalyzer(mock_fe)
        res = analyzer.full_matchup_analysis(
            'New York Knicks', 'Philadelphia 76ers',
            spread_line=-4.5, over_under_line=232.5,
            open_spread=-3.5, open_line=229.5
        )

        ou = res['over_under']
        self.assertIsNotNone(ou)
        self.assertIn('projected_total', ou)
        self.assertIn('raw_projected_total', ou)
        self.assertIn('line_movement', ou)
        self.assertEqual(ou['line_movement']['status'], 'MOVE_AGAINST_MODEL')

        sp = res['spread']
        self.assertIsNotNone(sp)
        self.assertIn('projected_margin', sp)
        self.assertIn('raw_projected_margin', sp)
        self.assertIn('line_movement', sp)


if __name__ == '__main__':
    unittest.main()
