import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from processing.feature_engineering import FeatureEngineer


class TestFeatureEngineering(unittest.TestCase):
    """Tests for FeatureEngineer addressing Bug 1, Bug 3, and Bug 8."""

    def setUp(self):
        self.inj_patch = patch('collectors.espn_collector.ESPNCollector.get_injuries', return_value=pd.DataFrame())
        self.inj_patch.start()
        self.rw_patch = patch('collectors.rotowire_collector.RotoWireCollector.get_todays_lineups', return_value={})
        self.rw_patch.start()

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

    def tearDown(self):
        self.inj_patch.stop()
        self.rw_patch.stop()

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




    def test_preseason_and_playoffs_excluded(self):
        df = pd.DataFrame({
            'date': ['2026-10-10', '2026-10-15', '2026-10-22', '2026-10-24'],
            'home_team': ['Boston Celtics', 'Boston Celtics', 'Boston Celtics', 'Boston Celtics'],
            'away_team': ['Miami Heat', 'Miami Heat', 'Miami Heat', 'Miami Heat'],
            'home_score': [95, 98, 120, 125],
            'away_score': [90, 92, 110, 115],
            'season_type': [1, 3, 2, 2]
        })
        fe = FeatureEngineer(df)
        self.assertEqual(len(fe.game_logs), 2)
        self.assertTrue((fe.game_logs['home_score'].isin([120, 125])).all())

    def test_data_quality_flags(self):
        """Verify data quality flags in compute_matchup_features."""
        fe = FeatureEngineer(self.sample_logs)
        features = fe.compute_matchup_features('Boston Celtics', 'Miami Heat', game_date='2026-01-22')
        self.assertIn('dq_is_reliable', features)
        self.assertIn('dq_extreme_roster_gap', features)
        self.assertIn('dq_missing_net_rating', features)
        self.assertIn('dq_unconfirmed_lineup', features)
        self.assertIn('dq_stale_data', features)

    def test_cross_season_form_isolation_phoenix_case(self):
        """Phase 1.1: Ensure previous season games (e.g. May 2026) are not leaked into current season form."""
        df_cross_season = pd.DataFrame({
            'date': [
                '2026-05-15',  # Prior season playoffs/May
                '2026-05-18',  # Prior season playoffs/May
                '2026-10-25',  # Current season regular season
                '2026-10-28',  # Current season regular season
                '2026-11-02'   # Current season regular season
            ],
            'home_team': ['Phoenix Suns', 'Phoenix Suns', 'Phoenix Suns', 'Denver Nuggets', 'Phoenix Suns'],
            'away_team': ['Minnesota Timberwolves', 'Minnesota Timberwolves', 'Golden State Warriors', 'Phoenix Suns', 'LA Clippers'],
            'home_score': [118, 120, 110, 105, 115],
            'away_score': [100, 105, 108, 112, 102],
            'season_type': [3, 3, 2, 2, 2]  # 3 = playoffs, 2 = regular season
        })
        fe = FeatureEngineer(df_cross_season)
        form = fe.get_team_recent_form('Phoenix Suns', n_games=10, before_date='2026-11-15')
        # Only 3 current regular season games should be included
        self.assertEqual(form['wins'] + form['losses'], 3)
        # Scores from May 2026 (118, 120) should NOT be in the average
        # Current games: 110 (W), 112 (W away), 115 (W) -> 3 wins, avg points scored = (110 + 112 + 115) / 3 = 112.33
        self.assertAlmostEqual(form['avg_points_scored'], 112.3, places=1)

    def test_h2h_missing_returns_none(self):
        """Phase 1.3: h2h_record should be None, not [0,0] when no h2h games exist."""
        fe = FeatureEngineer(self.sample_logs)
        h2h = fe.get_head_to_head('Boston Celtics', 'Utah Jazz')
        self.assertEqual(h2h['total_games'], 0)
        self.assertIsNone(h2h['team_a_wins'])
        self.assertIsNone(h2h['avg_point_diff'])
        
        spread_feat = fe.compute_spread_features('Boston Celtics', 'Utah Jazz')
        self.assertIsNone(spread_feat['h2h_record'])
        self.assertIsNone(spread_feat['h2h_avg_margin'])

    def test_net_rating_and_power_gap_fallback(self):
        """Phase 1.3: net_rating and power_gap fallback from game logs when advanced stats are missing."""
        fe = FeatureEngineer(self.sample_logs, team_advanced=None)
        spread_feat = fe.compute_spread_features('Boston Celtics', 'Miami Heat', game_date='2026-01-22')
        # Should be derived from game logs
        self.assertIsNotNone(spread_feat['home_net_rating'])
        self.assertIsNotNone(spread_feat['away_net_rating'])
        self.assertIsNotNone(spread_feat['power_gap'])
        # Aliases present
        self.assertIn('home_last10_record', spread_feat)
        self.assertIn('away_last10_record', spread_feat)

    def test_matchup_pace_style_fallback_from_league(self):
        """Phase 1.3: matchup_pace_style falls back gracefully instead of UNKNOWN."""
        fe = FeatureEngineer(self.sample_logs, team_advanced=None)
        features = fe.compute_matchup_features('Boston Celtics', 'Miami Heat', game_date='2026-01-22')
        self.assertNotEqual(features['matchup_pace_style'], 'UNKNOWN')
        self.assertIn(features['matchup_pace_style'], ['BALANCED_PACE', 'FAST_VS_FAST', 'SLOW_VS_SLOW', 'FAST_VS_SLOW'])

    def test_dynamic_early_season_math(self):
        """Phase 1.2: Verify dynamic market shrinkage and dynamic sigma."""
        from analysis.betting_math import dynamic_early_season_market_weight, dynamic_early_season_sigma
        w_0 = dynamic_early_season_market_weight(0)
        w_10 = dynamic_early_season_market_weight(10)
        w_20 = dynamic_early_season_market_weight(20)
        w_40 = dynamic_early_season_market_weight(40)
        
        self.assertEqual(w_0, 0.85)
        self.assertEqual(w_10, 0.65)
        self.assertEqual(w_20, 0.45)
        self.assertEqual(w_40, 0.25)
        
        sig_0 = dynamic_early_season_sigma(12.5, 0)
        sig_10 = dynamic_early_season_sigma(12.5, 10)
        sig_30 = dynamic_early_season_sigma(12.5, 30)
        self.assertGreater(sig_0, sig_10)
        self.assertGreater(sig_10, sig_30)
        self.assertGreater(sig_30, 12.5)

    def test_current_season_games_played_isolation(self):
        """Verify home_games_played strictly counts current season games and does not bleed prior season."""
        cross_season_logs = pd.DataFrame({
            'date': [
                '2025-11-10', '2025-12-15', '2026-01-20', # Prior 2025-26 season
                '2026-10-25', '2026-11-02', '2026-11-10'  # Current 2026-27 season
            ],
            'home_team': ['Boston Celtics'] * 6,
            'away_team': ['Miami Heat'] * 6,
            'home_score': [110, 105, 115, 108, 112, 120],
            'away_score': [100, 102, 108, 100, 105, 115],
            'over_under': [215.0] * 6,
            'season_type': [2] * 6
        })
        fe = FeatureEngineer(cross_season_logs)
        feat = fe.compute_matchup_features('Boston Celtics', 'Miami Heat', game_date='2026-11-15')
        # Only the 3 games in Oct-Nov 2026 should be counted, NEVER the 2025 games!
        self.assertEqual(feat['home_games_played'], 3)
        self.assertEqual(feat['away_games_played'], 3)

        # Team with 0 games in current season should return 0
        feat_other = fe.compute_matchup_features('Golden State Warriors', 'Los Angeles Lakers', game_date='2026-11-15')
        self.assertEqual(feat_other['home_games_played'], 0)
        self.assertEqual(feat_other['away_games_played'], 0)

class TestRosterRatingProvenance(unittest.TestCase):
    """Roster features must carry the timestamp of the injury snapshot they used."""

    def test_injuries_as_of_is_propagated(self):
        fe = object.__new__(FeatureEngineer)
        stub = MagicMock()
        stub.calculate_team_roster_rating.return_value = {
            'roster_rating': 100.0,
            'full_roster_rating': 110.0,
            'off_loss': 0.0,
            'def_loss': 0.0,
            'top_players': [],
            'starters': [],
            'lineup_status': 'Confirmed',
            'is_lineup_confirmed': True,
            'injuries': ['Star Guy [OUT - Knee]'],
            'injuries_as_of': '2026-11-18 09:30',
        }
        fe._player_engine = stub

        out = fe.get_roster_ratings('Boston Celtics', 'Miami Heat')

        self.assertEqual(out['home_injuries_as_of'], '2026-11-18 09:30')
        self.assertEqual(out['away_injuries_as_of'], '2026-11-18 09:30')
        self.assertEqual(out['home_injuries'], ['Star Guy [OUT - Knee]'])


if __name__ == '__main__':
    unittest.main()
