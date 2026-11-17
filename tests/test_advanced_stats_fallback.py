"""Tests for AdvancedStatsCollector fallback and ESPN supplementation."""

import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from collectors.advanced_stats_collector import AdvancedStatsCollector


class TestGameLogsFallback(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.raw = os.path.join(self.tmp, 'raw')
        os.makedirs(self.raw)
        self.collector = AdvancedStatsCollector(season='2026-27', max_retries=1)
        self.dir_patch = patch.object(self.collector, '_resolve_raw_dir', return_value=self.raw)
        self.dir_patch.start()

    def tearDown(self):
        self.dir_patch.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write_cache(self, latest_date='2026-11-01'):
        df = pd.DataFrame({
            'GAME_ID': ['g1', 'g1'],
            'GAME_DATE': [latest_date, latest_date],
            'TEAM_NAME': ['Boston Celtics', 'Miami Heat'],
            'MATCHUP': ['Boston Celtics vs. Miami Heat', 'Miami Heat @ Boston Celtics'],
            'WL': ['W', 'L'],
            'PTS': [110, 100],
            'PLUS_MINUS': [10, -10],
        })
        df.to_csv(os.path.join(self.raw, 'nba_api_game_logs.csv'), index=False)

    def _write_espn(self):
        df = pd.DataFrame({
            'game_id': ['g2'],
            'date': ['2026-11-15'],
            'home_team': ['Los Angeles Lakers'],
            'away_team': ['Golden State Warriors'],
            'home_score': [120],
            'away_score': [115],
        })
        df.to_csv(os.path.join(self.raw, 'espn_recent_games.csv'), index=False)

    def test_api_failure_returns_cached_data_with_schema(self):
        self._write_cache()
        with patch.object(self.collector, '_retry_call', return_value=None):
            df = self.collector.get_game_logs('T')

        self.assertFalse(df.empty)
        self.assertEqual(len(df), 2)
        # Expected schema columns are coerced even when absent from the cache
        self.assertIn('SEASON_ID', df.columns)
        self.assertIn('TEAM_ID', df.columns)

    def test_api_exception_returns_cached_data(self):
        self._write_cache()
        with patch.object(self.collector, '_retry_call', side_effect=RuntimeError('boom')):
            df = self.collector.get_game_logs('T')

        self.assertFalse(df.empty)
        self.assertEqual(len(df), 2)

    def test_stale_cache_is_supplemented_from_espn(self):
        self._write_cache(latest_date='2026-11-01')
        self._write_espn()
        with patch.object(self.collector, '_retry_call', return_value=None):
            df = self.collector.get_game_logs('T')

        # 2 cached team-rows + 2 supplemental rows (home + away) for the new game
        self.assertEqual(len(df), 4)
        self.assertIn('Los Angeles Lakers', set(df['TEAM_NAME']))
        self.assertIn('Golden State Warriors', set(df['TEAM_NAME']))
        new_rows = df[df['TEAM_NAME'] == 'Los Angeles Lakers']
        self.assertEqual(float(new_rows.iloc[0]['PTS']), 120.0)
        self.assertEqual(new_rows.iloc[0]['WL'], 'W')

    def test_fresh_cache_is_not_supplemented(self):
        self._write_cache(latest_date='2026-11-15')
        self._write_espn()
        with patch.object(self.collector, '_retry_call', return_value=None):
            df = self.collector.get_game_logs('T')

        self.assertEqual(len(df), 2)
        self.assertNotIn('Los Angeles Lakers', set(df['TEAM_NAME']))

    def test_no_cache_returns_empty_frame(self):
        with patch.object(self.collector, '_retry_call', return_value=None):
            df = self.collector.get_game_logs('T')

        self.assertTrue(df.empty)


if __name__ == '__main__':
    unittest.main()
