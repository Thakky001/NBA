import unittest
import pandas as pd
from unittest.mock import patch, MagicMock
from collectors.sdv_collector import SDVCollector


class TestSDVCollector(unittest.TestCase):
    """Tests for SDVCollector addressing Bug 5 (Polars DataFrame compatibility)."""

    def setUp(self):
        self.collector = SDVCollector(max_retries=1, retry_delay=0.01)

    def test_to_pandas_df_with_pandas(self):
        """Pass-through of standard pandas DataFrame."""
        df = pd.DataFrame({'team': ['BOS'], 'pts': [115]})
        res = self.collector._to_pandas_df(df)
        self.assertIsInstance(res, pd.DataFrame)
        self.assertFalse(res.empty)
        self.assertEqual(len(res), 1)

    def test_to_pandas_df_with_none(self):
        """None should return None safely."""
        self.assertIsNone(self.collector._to_pandas_df(None))

    def test_to_pandas_df_with_polars_bug5(self):
        """Bug 5: Polars DataFrame should be converted to pandas.DataFrame with .empty attribute."""
        try:
            import polars as pl
            pldf = pl.DataFrame({'team': ['LAL', 'BOS'], 'pts': [108, 112]})
            res = self.collector._to_pandas_df(pldf)
            self.assertIsInstance(res, pd.DataFrame)
            self.assertTrue(hasattr(res, 'empty'))
            self.assertFalse(res.empty)
            self.assertEqual(len(res), 2)
            self.assertEqual(list(res.columns), ['team', 'pts'])
        except ImportError:
            # Fallback mock if polars is not present
            mock_pl = MagicMock()
            mock_pl.to_pandas.return_value = pd.DataFrame({'team': ['LAL'], 'pts': [108]})
            res = self.collector._to_pandas_df(mock_pl)
            self.assertIsInstance(res, pd.DataFrame)
            self.assertFalse(res.empty)

    def test_get_team_season_averages_fallback_robustness(self):
        """Verify get_team_season_averages executes without crashing on Polars return."""
        # Mock load_nba_team_boxscore returning a mock Polars object
        fake_pandas = pd.DataFrame({
            'team_name': ['Boston Celtics', 'Los Angeles Lakers'],
            'team_id': [2, 13],
            'season': [2026, 2026],
            'team_score': [115, 110],
            'field_goal_percentage': [48.5, 46.2],
            'three_point_field_goal_percentage': [38.2, 35.1],
            'free_throw_percentage': [81.0, 78.5]
        })
        mock_polars = MagicMock()
        mock_polars.to_pandas.return_value = fake_pandas
        del mock_polars.empty  # Simulate polars not having .empty

        with patch('sportsdataverse.nba.load_nba_team_boxscore', return_value=mock_polars):
            res = self.collector.get_team_season_averages(seasons=[2026])
            self.assertIsInstance(res, pd.DataFrame)
            self.assertFalse(res.empty)
            self.assertIn('team_name', res.columns)


if __name__ == '__main__':
    unittest.main()
