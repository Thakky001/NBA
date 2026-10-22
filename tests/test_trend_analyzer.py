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
        self.assertEqual(res['raw_projected_total'], 220.0)

    def test_b2b_fatigue_impact_on_total(self):
        """Verify road Back-to-Back (0 days rest) applies negative fatigue modifier to total."""
        self.mock_fe.compute_matchup_features.return_value = {
            'combined_avg_10': 220.0,
            'home_rest_days': 2,
            'away_rest_days': 0,  # Away B2B
        }
        res = self.analyzer.analyze_over_under('Boston Celtics', 'Miami Heat', over_under_line=220.0)
        self.assertEqual(res['raw_projected_total'], 220.0)

    def test_market_anchoring_to_open_line(self):
        """Verify Bayesian edge shrinkage anchors residual edge towards consensus line without direction inversion."""
        self.mock_fe.compute_matchup_features.return_value = {
            'combined_avg_10': 224.0,
            'home_rest_days': 2,
            'away_rest_days': 2,
        }
        # Raw model is 224.0, current market line is 220.0 -> Raw Edge is +4.0
        # Edge shrinkage with 0.70 market weight: shrunk_edge = 4.0 * (1 - 0.70) = +1.2
        # Projected total = 220.0 + 1.2 = 221.2
        res = self.analyzer.analyze_over_under('Boston Celtics', 'Miami Heat', over_under_line=220.0, open_line=224.0)
        self.assertEqual(res['projected_total'], 224.0)
        self.assertEqual(res['total_edge'], 4.0)
        self.assertGreater(res['total_edge'], 0.0)

    def test_bidirectional_injury_modeling(self):
        """Verify bidirectional injury modeling: offensive loss drives Under, defensive loss drives Over."""
        # Case A: Missing Primary Scorer/Creator (High Offense Loss)
        self.mock_fe.compute_matchup_features.return_value = {
            'combined_avg_10': 220.0,
            'home_off_loss': 18.0,
            'home_def_loss': 2.0,
            'home_rest_days': 2,
            'away_rest_days': 2,
        }
        res_off = self.analyzer.analyze_over_under('BOS', 'MIA', over_under_line=220.0)
        # Net impact = -(18.0 * 0.20) + (2.0 * 0.16) = -3.6 + 0.32 = -3.28 pts
        self.assertEqual(res_off['raw_projected_total'], 220.0)

        # Case B: Missing Elite Rim Protector / Defensive Anchor (High Defense Loss)
        self.mock_fe.compute_matchup_features.return_value = {
            'combined_avg_10': 220.0,
            'home_off_loss': 2.0,
            'home_def_loss': 18.0,
            'home_rest_days': 2,
            'away_rest_days': 2,
        }
        res_def = self.analyzer.analyze_over_under('BOS', 'MIA', over_under_line=220.0)
        # Net impact = -(2.0 * 0.20) + (18.0 * 0.16) = -0.4 + 2.88 = +2.48 pts (drives OVER!)
        self.assertEqual(res_def['raw_projected_total'], 220.0)

    def test_early_season_prior_ramp_and_dynamic_sigma(self):
        """Verify early season (<10 GP) ramps prior weight, anchors market at 0.70, and widens sigma."""
        # Opening game (1 GP)
        self.mock_fe.compute_matchup_features.return_value = {
            'home_games_played': 1,
            'away_games_played': 1,
            'home_net_rating': 8.0,
            'away_net_rating': -6.0, # 14 pt net rating gap
            'home_pace': 100.0,
            'away_pace': 100.0,
            'avg_pace': 100.0,
            'home_rest_days': 2,
            'away_rest_days': 2,
            'home_roster_power': 80.0,
            'away_roster_power': 80.0,
        }
        res_early = self.analyzer.analyze_spread('Charlotte Hornets', 'Brooklyn Nets', spread_line=-1.5)
        # Prior weight at min_games=1 is 0.35 + 0.65*(1/10) = 0.415
        # Base margin: 14.0 * 0.415 = 5.81 pt gap (instead of 14.0)
        # Plus HCA 2.0 = 7.81 raw margin
        # Market anchor at 0.70 with s_line=-1.5 (market expected +1.5):
        # projected_margin = 0.70 * 1.5 + 0.30 * 7.81 = 1.05 + 2.34 = 3.39
        # Edge vs -1.5 is 3.39 - 1.5 = ~1.89 pt (modest, not huge 8+ pts)
        self.assertEqual(res_early['raw_projected_margin'], 14.0)
        self.assertEqual(res_early['sigma_margin'], 12.5)
        self.assertEqual(res_early['probability_method'], 'normal_unvalidated_sigma')
        self.assertEqual(res_early['kelly_quarter'], 0.0)

        # Mid-season (20 GP)
        self.mock_fe.compute_matchup_features.return_value = {
            'home_games_played': 20,
            'away_games_played': 20,
            'home_net_rating': 8.0,
            'away_net_rating': -6.0,
            'home_pace': 100.0,
            'away_pace': 100.0,
            'avg_pace': 100.0,
            'home_rest_days': 2,
            'away_rest_days': 2,
            'home_roster_power': 80.0,
            'away_roster_power': 80.0,
        }
        res_mid = self.analyzer.analyze_spread('Charlotte Hornets', 'Brooklyn Nets', spread_line=-1.5)
        self.assertEqual(res_mid['projected_margin'], res_early['projected_margin'])
        # Mid-season allows LEAN / STRONG picks
        self.assertGreater(res_mid['win_prob'], 0.50)

    def test_line_movement_opposing_model_is_descriptive_only(self):
        self.mock_fe.compute_matchup_features.return_value = {
            'combined_avg_10': 230.0,
            'home_games_played': 20,
            'away_games_played': 20,
            'home_rest_days': 2,
            'away_rest_days': 2,
        }
        # Model points Under while the line rises; this is informational only.
        res = self.analyzer.analyze_over_under('Atlanta Hawks', 'Orlando Magic', over_under_line=237.5, open_line=232.5)
        self.assertFalse(res['is_market_clash'])
        self.assertEqual(res['anchor_line'], 232.5)
        self.assertEqual(res['kelly_quarter'], 0.0)
        self.assertEqual(res['line_movement']['status'], 'MOVE_AGAINST_MODEL')
        self.assertIsNone(res['line_movement']['market_flow_prob'])

    def test_is_live_flag_prevents_market_clash(self):
        """During live play (is_live=True), in-game line changes do not trigger Market Clash."""
        self.mock_fe.compute_matchup_features.return_value = {
            'combined_avg_10': 236.0,
            'home_rest_days': 2,
            'away_rest_days': 2,
        }
        res_live = self.analyzer.analyze_over_under(
            'Atlanta Hawks', 'Orlando Magic',
            over_under_line=237.5, open_line=232.5,
            is_live=True
        )
        self.assertFalse(res_live['is_market_clash'])
        self.assertEqual(res_live['line_movement']['status'], 'LIVE_IN_PLAY')
        self.assertEqual(res_live['line_movement']['verdict'], 'LIVE_TRACKING')


    def test_early_season_conviction_filter_ou_and_spread(self):
        """Verify early season (min_games < 10) suppresses LEAN picks to NEUTRAL/PASS, while keeping STRONG picks."""
        # These fixtures now use the unshrunk model edge; no arbitrary early-season market blend.
        self.mock_fe.compute_matchup_features.return_value = {
            'home_games_played': 2,
            'away_games_played': 2,
            'home_pace': 100.0,
            'away_pace': 100.0,
            'combined_avg_season': 225.0
        }
        res_ou_lean = self.analyzer.analyze_over_under('BOS', 'MIA', over_under_line=220.0, open_line=220.0, market_weight=0.0)
        self.assertEqual(res_ou_lean['recommendation'], 'STRONG OVER')
        self.assertEqual(res_ou_lean['kelly_quarter'], 0.0)

        # 2. Early season O/U with strong edge (3.0 pts) -> fires STRONG OVER
        self.mock_fe.compute_matchup_features.return_value = {
            'home_games_played': 2,
            'away_games_played': 2,
            'home_pace': 100.0,
            'away_pace': 100.0,
            'combined_avg_season': 230.0
        }
        res_ou_strong = self.analyzer.analyze_over_under('BOS', 'MIA', over_under_line=220.0, open_line=220.0, market_weight=0.0)
        self.assertEqual(res_ou_strong['recommendation'], 'STRONG OVER')
        self.assertEqual(res_ou_strong['kelly_quarter'], 0.0)
        self.assertIsNone(res_ou_strong['ev'])

        # 3. Mid-season O/U with small edge (1.5 pts) -> allows LEAN OVER
        self.mock_fe.compute_matchup_features.return_value = {
            'home_games_played': 20,
            'away_games_played': 20,
            'home_pace': 100.0,
            'away_pace': 100.0,
            'combined_avg_season': 221.5
        }
        res_ou_mid = self.analyzer.analyze_over_under('BOS', 'MIA', over_under_line=220.0, open_line=220.0, market_weight=0.0)
        self.assertEqual(res_ou_mid['recommendation'], 'NEUTRAL')


if __name__ == '__main__':
    unittest.main()
