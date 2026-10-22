import unittest
import pandas as pd
import numpy as np
from processing.feature_engineering import FeatureEngineer


class TestFeatureEngineering(unittest.TestCase):
    """Tests for FeatureEngineer addressing Bug 1, Bug 3, and Bug 8."""

    def setUp(self):
        self.sample_logs = pd.DataFrame({
            'date': [
                '2026-01-10T20:00Z',
                '2026-01-12',
                '2026-01-15T19:30Z',
                '2026-01-18',
                '2026-01-20T22:00Z'
            ],
            'home_team': ['Boston Celtics', 'Miami Heat', 'Boston Celtics', 'New York Knicks', 'Boston Celtics'],
            'away_team': ['Miami Heat', 'Boston Celtics', 'Chicago Bulls', 'Boston Celtics', 'Toronto Raptors'],
            'home_score': [110, 105, 115, 102, 120],
            'away_score': [100, 108, 105, 99, 112],
            'over_under': [215.0, 210.0, 218.0, 205.0, 222.0]
        })

    def test_mixed_date_formats_bug1(self):
        """Bug 1: Mixed date formats should parse without producing NaT."""
        fe = FeatureEngineer(self.sample_logs)
        self.assertEqual(fe.game_logs['date'].isna().sum(), 0)
        self.assertEqual(len(fe.game_logs), 5)
        # Dates should be tz-naive datetimes
        self.assertIsNone(fe.game_logs['date'].dt.tz)

    def test_tz_aware_before_date_filtering_bug3(self):
        """Bug 3: Filtering with tz-aware string should not raise TypeError."""
        fe = FeatureEngineer(self.sample_logs)
        # Query recent form before a tz-aware date string
        form = fe.get_team_recent_form('Boston Celtics', n_games=5, before_date='2026-01-19T00:00Z')
        self.assertGreater(form['wins'] + form['losses'], 0)

        # Query _get_team_games directly with tz-aware string
        games = fe._get_team_games('Boston Celtics', before_date='2026-01-19T00:00Z')
        self.assertEqual(len(games), 4)

    def test_rolling_average_virtual_columns_bug8(self):
        """Bug 8: Virtual columns like 'score' and 'points_scored' should return valid rolling series."""
        fe = FeatureEngineer(self.sample_logs)
        
        # Test 'score'
        rolling_score = fe.rolling_average('Boston Celtics', n_games=3, column='score')
        self.assertFalse(rolling_score.empty)
        self.assertEqual(len(rolling_score), 5)
        self.assertAlmostEqual(rolling_score.iloc[-1], (115 + 99 + 120) / 3.0)

        # Test 'points_scored'
        rolling_pts = fe.rolling_average('Boston Celtics', n_games=3, column='points_scored')
        self.assertFalse(rolling_pts.empty)
        self.assertEqual(len(rolling_pts), 5)
        self.assertAlmostEqual(rolling_pts.iloc[-1], rolling_score.iloc[-1])

        # Test 'points_allowed'
        rolling_allowed = fe.rolling_average('Boston Celtics', n_games=3, column='points_allowed')
        self.assertFalse(rolling_allowed.empty)
        self.assertEqual(len(rolling_allowed), 5)

    def test_unplayed_0_0_games_excluded(self):
        """Verify unplayed scheduled games (0-0 scores) are dropped from analysis logs."""
        df_with_unplayed = pd.concat([
            self.sample_logs,
            pd.DataFrame({
                'date': ['2026-01-22'],
                'home_team': ['Boston Celtics'],
                'away_team': ['Brooklyn Nets'],
                'home_score': [0],
                'away_score': [0],
                'over_under': [225.0]
            })
        ], ignore_index=True)
        fe = FeatureEngineer(df_with_unplayed)
        self.assertEqual(len(fe.game_logs), 5)
        # Verify 0-0 game is not present
        celtics_games = fe._get_team_games('Boston Celtics')
        self.assertFalse((celtics_games['home_score'] == 0).any())

    def test_small_sample_is_reported_without_unfitted_shrinkage(self):
        early_df = pd.DataFrame({
            'date': ['2026-10-22', '2026-10-24', '2026-10-26'],
            'home_team': ['Detroit Pistons', 'Detroit Pistons', 'Detroit Pistons'],
            'away_team': ['Miami Heat', 'Chicago Bulls', 'Atlanta Hawks'],
            'home_score': [90, 92, 94],  # Raw mean = 92.0
            'away_score': [100, 102, 104], # Raw mean = 102.0
            'over_under': [210.0, 212.0, 214.0]
        })
        fe = FeatureEngineer(early_df)
        form = fe.get_team_recent_form('Detroit Pistons')
        self.assertFalse(form['sample_size_regressed'])
        self.assertEqual(form['raw_points_scored'], 92.0)
        self.assertEqual(form['avg_points_scored'], 92.0)

    def test_historical_features_exclude_untimestamped_advanced_snapshot(self):
        advanced = pd.DataFrame({
            'team': ['Boston Celtics', 'Miami Heat'],
            'pace': [120.0, 80.0],
            'off_rating': [130.0, 90.0],
            'def_rating': [90.0, 130.0],
            'net_rating': [40.0, -40.0],
        })
        fe = FeatureEngineer(self.sample_logs, team_advanced=advanced)
        features = fe.compute_matchup_features('Boston Celtics', 'Miami Heat', game_date='2026-01-16T00:00:00Z')
        self.assertIsNone(features['home_pace'])
        self.assertIsNone(features['away_ortg'])

    def test_head_to_head_and_splits_are_as_of(self):
        fe = FeatureEngineer(self.sample_logs)
        h2h = fe.get_head_to_head('Boston Celtics', 'Miami Heat', before_date='2026-01-11')
        self.assertEqual(h2h['total_games'], 1)
        splits = fe.get_home_away_splits('Boston Celtics', before_date='2026-01-11')
        self.assertEqual(splits['home_wins'] + splits['home_losses'], 1)


if __name__ == '__main__':
    unittest.main()
