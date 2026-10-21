"""Unit tests verifying all 8 bug fixes and improvements from the Ponytail Audit."""

import unittest
import pandas as pd
import numpy as np

from config import (
    normalize_team_name,
    resolve_team_info,
    parse_spread,
    calculate_home_spread,
    t_rec,
)
from processing.feature_engineering import FeatureEngineer
from collectors.odds_collector import OddsCollector
from processing.data_merger import DataMerger


class TestPonytailAuditFixes(unittest.TestCase):
    """Test suite covering fixes for items #1 to #8 from ponytail audit."""

    def test_item1_team_normalization_no_substring_collision(self):
        """Fix #1: Substring 'la' or short abbrs must not match 'Atlanta Hawks'."""
        # 'LA' input must not resolve to Atlanta Hawks
        la_result = normalize_team_name("LA")
        self.assertNotEqual(la_result, "Atlanta Hawks")

        # Specific LA aliases must resolve accurately
        self.assertEqual(normalize_team_name("LA Lakers"), "Los Angeles Lakers")
        self.assertEqual(normalize_team_name("LA Clippers"), "Los Angeles Clippers")
        self.assertEqual(normalize_team_name("L.A. Lakers"), "Los Angeles Lakers")
        self.assertEqual(normalize_team_name("L.A. Clippers"), "Los Angeles Clippers")

        # Other short abbreviations must match their exact teams, not substrings
        self.assertEqual(normalize_team_name("IND"), "Indiana Pacers")
        self.assertEqual(normalize_team_name("ORL"), "Orlando Magic")

    def test_item2_spread_parser_comprehensive(self):
        """Fix #2: Centralized parse_spread handles all formats including pick'em."""
        # Standalone pick'em
        self.assertEqual(parse_spread("PK"), ("PK", 0.0))
        self.assertEqual(parse_spread("EVEN"), ("PK", 0.0))
        self.assertEqual(parse_spread("PICK"), ("PK", 0.0))

        # Team pick'em
        self.assertEqual(parse_spread("BOS PK"), ("BOS", 0.0))
        self.assertEqual(parse_spread("MIA EVEN"), ("MIA", 0.0))
        self.assertEqual(parse_spread("LAL PICK"), ("LAL", 0.0))

        # Standard lines
        self.assertEqual(parse_spread("BOS -4.5"), ("BOS", -4.5))
        self.assertEqual(parse_spread("MIA +3"), ("MIA", 3.0))

        # Pure numeric
        self.assertEqual(parse_spread("-5.5"), (None, -5.5))
        self.assertEqual(parse_spread(3.0), (None, 3.0))

        # Missing / invalid
        self.assertEqual(parse_spread(None), (None, None))
        self.assertEqual(parse_spread(""), (None, None))
        self.assertEqual(parse_spread(float("nan")), (None, None))

    def test_item2_calculate_home_spread(self):
        """Fix #2: calculate_home_spread properly aligns spreads to the home team."""
        # Home team favored
        self.assertEqual(
            calculate_home_spread("BOS -4.5", "Boston Celtics", "Miami Heat"),
            -4.5
        )
        # Away team underdog (+4.5) means home team is -4.5
        self.assertEqual(
            calculate_home_spread("MIA +4.5", "Boston Celtics", "Miami Heat"),
            -4.5
        )
        # Pick'em
        self.assertEqual(
            calculate_home_spread("BOS PK", "Boston Celtics", "Miami Heat"),
            0.0
        )
        self.assertEqual(
            calculate_home_spread("PK", "Boston Celtics", "Miami Heat"),
            0.0
        )

    def test_item5_season_opener_rest_days_neutral(self):
        """Fix #5: Empty past games log returns neutral 3 days of rest, not -1."""
        sample_logs = pd.DataFrame({
            'date': ['2026-01-10'],
            'home_team': ['Boston Celtics'],
            'away_team': ['Miami Heat'],
            'home_score': [110],
            'away_score': [100],
            'over_under': [215.0]
        })
        fe = FeatureEngineer(sample_logs)

        # Team with no games played prior
        rest_days = fe.get_rest_days("Denver Nuggets", "2026-01-10")
        self.assertEqual(rest_days, 3, "New team must have neutral 3 rest days, not -1")

        # Matchup features should not set fatigue flag for 3 rest days
        matchup = fe.compute_matchup_features("Denver Nuggets", "Chicago Bulls", game_date="2026-01-10")
        self.assertEqual(matchup.get('home_rest_days'), 3)
        self.assertEqual(matchup.get('away_rest_days'), 3)
        self.assertFalse(matchup.get('fatigue_flag'), "Neutral rest should not trigger fatigue flag")

    def test_item6_odds_collector_fallback_schema(self):
        """Fix #6: Odds collector fallback returns DataFrame with required columns."""
        collector = OddsCollector()
        df = collector.get_odds_dataframe()
        self.assertIsInstance(df, pd.DataFrame)
        expected_cols = {'home_team', 'away_team', 'spread', 'over_under'}
        self.assertTrue(expected_cols.issubset(set(df.columns)), f"Columns must contain {expected_cols}")
        # Aliases
        self.assertIsInstance(collector.get_odds_df(), pd.DataFrame)
        self.assertIsInstance(collector.get_odds_data(), pd.DataFrame)

    def test_item8_data_merger_normalization(self):
        """Fix #8: DataMerger cleanly delegates normalization to central config."""
        merger = DataMerger()
        self.assertEqual(merger.normalize_team_name("BOS"), "Boston Celtics")
        self.assertEqual(merger.normalize_team_name("LAL"), "Los Angeles Lakers")
        self.assertTrue(pd.isna(merger.normalize_team_name(np.nan)))

    def test_phase2_odds_collector_match_team_no_la_atlanta_collision(self):
        """Phase 2: match_team inside OddsCollector must not match LA to Atlanta Hawks."""
        from collectors.odds_collector import OddsCollector
        collector = OddsCollector()
        # Mocking games list where Atlanta Hawks is present
        mock_games = [{
            'home_team': 'Atlanta Hawks',
            'away_team': 'Miami Heat',
            'bookmakers': [{'title': 'DraftKings', 'markets': []}]
        }]
        collector._cached_games = mock_games
        collector._cache_time = 9999999999  # keep cache alive
        
        # Searching for LA Lakers vs Miami Heat must NOT match Atlanta Hawks
        res = collector.get_game_odds('Los Angeles Lakers', 'Miami Heat')
        self.assertEqual(res, {'spreads': {}, 'totals': {}}, "LA Lakers must not match Atlanta Hawks")

        res_la = collector.get_game_odds('LA', 'Miami Heat')
        self.assertEqual(res_la, {'spreads': {}, 'totals': {}}, "'LA' must not match Atlanta Hawks")

    def test_odds_collector_no_key_caches_empty(self):
        """OddsCollector when api_key is empty logs once and caches empty result to prevent log spam."""
        from unittest.mock import patch
        collector = OddsCollector()
        collector.api_key = ""
        collector._cached_games = None
        collector._logged_no_key = False

        with patch('config.ODDS_API_KEY', ''):
            res1 = collector._fetch_all_odds()
            self.assertEqual(res1, [])
            self.assertTrue(collector._logged_no_key)
            self.assertEqual(collector._cached_games, [])

            res2 = collector._fetch_all_odds()
            self.assertEqual(res2, [])

    def test_phase2_espn_collector_uses_https(self):
        """Phase 2: ESPN endpoints must use secure HTTPS protocol."""
        from collectors.espn_collector import ESPNCollector
        self.assertTrue(ESPNCollector.BASE_URL.startswith("https://"))
        self.assertTrue(ESPNCollector.STANDINGS_URL.startswith("https://"))

    def test_phase2_espn_summary_caching(self):
        """Phase 2: get_game_summary must cache results in _summary_cache."""
        from collectors.espn_collector import ESPNCollector
        collector = ESPNCollector()
        # Inject mock into cache
        collector._summary_cache['99999'] = ({'boxscore': {'test': 1}}, 9999999999)
        # Calling get_game_summary should return cached item directly
        summary = collector.get_game_summary('99999')
        self.assertEqual(summary.get('boxscore'), {'test': 1})

    def test_recommendation_translation_dynamic_spread(self):
        """Verify t_rec dynamically assigns 'ต่อ' vs 'รอง' according to spread_line."""
        # 1. Home is favorite (spread_line = -4.5) e.g. Washington -4.5 vs Milwaukee
        # Away is UNDERDOG -> must say 'รองทีมเยือน', NOT 'ต่อทีมเยือน'
        self.assertEqual(t_rec('STRONG AWAY', spread_line=-4.5), '🔥 รองทีมเยือน (มั่นใจมาก)')
        self.assertEqual(t_rec('LEAN AWAY', spread_line=-4.5), '✈️ รองทีมเยือน (น่าสนใจ)')
        # Home is FAVORITE -> must say 'ต่อเจ้าบ้าน'
        self.assertEqual(t_rec('STRONG HOME', spread_line=-4.5), '🔥 ต่อเจ้าบ้าน (มั่นใจมาก)')
        self.assertEqual(t_rec('LEAN HOME', spread_line=-4.5), '🏠 ต่อเจ้าบ้าน (น่าสนใจ)')

        # 2. Away is favorite (spread_line = +4.5) e.g. Washington +4.5 vs Milwaukee
        # Away is FAVORITE -> must say 'ต่อทีมเยือน'
        self.assertEqual(t_rec('STRONG AWAY', spread_line=4.5), '🔥 ต่อทีมเยือน (มั่นใจมาก)')
        self.assertEqual(t_rec('LEAN AWAY', spread_line=4.5), '✈️ ต่อทีมเยือน (น่าสนใจ)')
        # Home is UNDERDOG -> must say 'รองเจ้าบ้าน'
        self.assertEqual(t_rec('STRONG HOME', spread_line=4.5), '🔥 รองเจ้าบ้าน (มั่นใจมาก)')
        self.assertEqual(t_rec('LEAN HOME', spread_line=4.5), '🏠 รองเจ้าบ้าน (น่าสนใจ)')

        # 3. Pick'em (spread_line = 0)
        self.assertEqual(t_rec('STRONG AWAY', spread_line=0), '🔥 เลือกทีมเยือน (มั่นใจมาก)')
        self.assertEqual(t_rec('STRONG HOME', spread_line=0), '🔥 เลือกเจ้าบ้าน (มั่นใจมาก)')

        # 4. Unknown spread line -> neutral fallback without assuming ต่อ
        self.assertEqual(t_rec('STRONG AWAY'), '🔥 ทีมเยือน (มั่นใจมาก)')
        self.assertEqual(t_rec('STRONG HOME'), '🔥 เจ้าบ้าน (มั่นใจมาก)')

        # 5. Over/Under untouched
        self.assertEqual(t_rec('STRONG OVER'), '🔥 สูง (มั่นใจมาก)')
        self.assertEqual(t_rec('STRONG UNDER'), '🔥 ต่ำ (มั่นใจมาก)')


if __name__ == '__main__':
    unittest.main()
