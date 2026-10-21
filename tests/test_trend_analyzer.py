import unittest
import pandas as pd
from unittest.mock import MagicMock
from analysis.trend_analyzer import TrendAnalyzer


class TestTrendAnalyzer(unittest.TestCase):
    """Tests for TrendAnalyzer addressing Bug 2 and Bug 7."""

    def setUp(self):
        self.mock_fe = MagicMock()
        self.mock_fe.compute_matchup_features.return_value = {
            'home_avg_points_scored': 112.0,
            'away_avg_points_scored': 108.0,
            'combined_avg': 220.0,
            'home_pace': 99.0,
            'away_pace': 101.0,
            'avg_pace': 100.0,
            'home_net_rating': 4.5,
            'away_net_rating': -1.2,
            'home_point_diff_10': 5.2,
            'away_point_diff_10': -2.1,
            'home_home_record': (15, 5),
            'away_away_record': (8, 12),
            'home_streak': 1.0,  # Standings float streak
            'away_streak': -2.0, # Standings float streak
            'h2h_record': (2, 1),
            'h2h_avg_total': 218.0,
            'home_rest_days': 2,
            'away_rest_days': 1,
            'home_roster_power': 85.0,
            'away_roster_power': 82.0,
            'home_full_roster_power': 85.0,
            'away_full_roster_power': 82.0,
            'home_ou_record': (10, 10),
            'away_ou_record': (11, 9)
        }
        self.analyzer = TrendAnalyzer(self.mock_fe)

    def test_over_under_none_or_zero_bug2(self):
        """Bug 2: None or 0.0 over_under_line must return NEUTRAL recommendation without huge edge."""
        for invalid_line in [None, 0.0, 0]:
            res = self.analyzer.analyze_over_under('Boston Celtics', 'Miami Heat', invalid_line)
            self.assertEqual(res['recommendation'], 'NEUTRAL')
            self.assertEqual(res['total_edge'], 0.0)
            self.assertEqual(res['win_prob'], 0.50)
            self.assertIn('NEUTRAL', res['summary'])

    def test_parse_streak_bug7(self):
        """Bug 7: parse_streak should properly handle floats, strings, None, and NaN."""
        # Floats from ESPN standings
        self.assertEqual(TrendAnalyzer.parse_streak(1.0), 1)
        self.assertEqual(TrendAnalyzer.parse_streak(-2.0), -2)
        self.assertEqual(TrendAnalyzer.parse_streak(5), 5)

        # Standard strings
        self.assertEqual(TrendAnalyzer.parse_streak('W3'), 3)
        self.assertEqual(TrendAnalyzer.parse_streak('L4'), -4)
        self.assertEqual(TrendAnalyzer.parse_streak('W'), 1)
        self.assertEqual(TrendAnalyzer.parse_streak('L'), -1)

        # Nulls / Edge cases
        self.assertEqual(TrendAnalyzer.parse_streak(None), 0)
        self.assertEqual(TrendAnalyzer.parse_streak('None'), 0)
        self.assertEqual(TrendAnalyzer.parse_streak(float('nan')), 0)

    def test_analyze_spread_with_float_streaks(self):
        """Verify analyze_spread executes smoothly when streak features are floats."""
        res = self.analyzer.analyze_spread('Boston Celtics', 'Miami Heat', spread_line=-4.5)
        self.assertIn('recommendation', res)
        self.assertIn(res['recommendation'], ['STRONG HOME', 'LEAN HOME', 'NEUTRAL', 'LEAN AWAY', 'STRONG AWAY'])
        self.assertTrue(any('Recent Streak' in f['name'] for f in res['factors']))

    def test_dean_oliver_possession_model_no_double_pace(self):
        """Verify Dean Oliver possession model computes total from efficiency and pace without double-counting."""
        self.mock_fe.compute_matchup_features.return_value = {
            'home_pace': 102.0,
            'away_pace': 100.0,
            'home_ortg': 116.0,
            'home_drtg': 110.0,
            'away_ortg': 112.0,
            'away_drtg': 114.0,
            'home_rest_days': 2,
            'away_rest_days': 2,
            'combined_avg_10': 230.0,
        }
        # Expected Pace = (102 * 100) / 99.5 = 102.5126
        # Home exp eff = (116 + 114) / 2 = 115.0 -> Home pts = 102.5126 * 1.15 = 117.89
        # Away exp eff = (112 + 110) / 2 = 111.0 -> Away pts = 102.5126 * 1.11 = 113.79
        # Raw Total = 117.89 + 113.79 = 231.68 (Not double-multiplied by 102/99.5!)
        res = self.analyzer.analyze_over_under('Boston Celtics', 'Miami Heat', over_under_line=230.0)
        self.assertAlmostEqual(res['raw_projected_total'], 231.68, delta=0.5)

    def test_injury_replacement_impact_on_total(self):
        """Verify key missing starters drop projected total via replacement-level formula."""
        self.mock_fe.compute_matchup_features.return_value = {
            'combined_avg_10': 220.0,
            'home_full_roster_rating': 120.0,
            'home_roster_rating': 100.0,  # 20 EFF lost to injury
            'away_full_roster_rating': 110.0,
            'away_roster_rating': 110.0,
            'home_rest_days': 2,
            'away_rest_days': 2,
        }
        res = self.analyzer.analyze_over_under('Boston Celtics', 'Miami Heat', over_under_line=220.0)
        # 20 * 0.22 = -4.4 pts drop -> raw_projected_total = 220.0 - 4.4 = 215.6
        self.assertTrue(any('Injury Impact on Total' in f['name'] for f in res['factors']))
        self.assertLess(res['raw_projected_total'], 220.0)

    def test_b2b_fatigue_impact_on_total(self):
        """Verify road Back-to-Back (0 days rest) applies negative fatigue modifier to total."""
        self.mock_fe.compute_matchup_features.return_value = {
            'combined_avg_10': 220.0,
            'home_rest_days': 2,
            'away_rest_days': 0,  # Away B2B
        }
        res = self.analyzer.analyze_over_under('Boston Celtics', 'Miami Heat', over_under_line=220.0)
        self.assertTrue(any('Schedule & Rest Fatigue' in f['name'] for f in res['factors']))
        # Road B2B modifier is -1.8 pts -> raw total = 220.0 - 1.8 = 218.2
        self.assertAlmostEqual(res['raw_projected_total'], 218.2, places=1)

    def test_market_anchoring_to_open_line(self):
        """Verify Bayesian projection anchors to Open Line so steam line drops reveal OVER value."""
        self.mock_fe.compute_matchup_features.return_value = {
            'combined_avg_10': 224.0,
            'home_rest_days': 2,
            'away_rest_days': 2,
        }
        # Open line is 224.0, but market steamed down to 220.0
        # Anchoring to Open Line 224: 0.65 * 224.0 + 0.35 * 224.0 = 224.0
        # Edge against current line (220.0) = 224.0 - 220.0 = +4.0 (STRONG OVER value!)
        res = self.analyzer.analyze_over_under('Boston Celtics', 'Miami Heat', over_under_line=220.0, open_line=224.0)
        self.assertEqual(res['projected_total'], 224.0)
        self.assertAlmostEqual(res['total_edge'], 4.0, places=1)
        self.assertEqual(res['recommendation'], 'STRONG OVER')


if __name__ == '__main__':
    unittest.main()
