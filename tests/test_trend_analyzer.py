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


if __name__ == '__main__':
    unittest.main()
