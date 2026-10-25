import unittest
import pandas as pd
from processing.data_merger import DataMerger


class TestDataMerger(unittest.TestCase):
    """Tests for DataMerger addressing Bug 6 (Metric mapping and team normalization)."""

    def setUp(self):
        self.merger = DataMerger()

    def test_merge_team_stats_metric_mapping_bug6(self):
        """Bug 6: Advanced stats with SDV/NBA API column names (tm_tov_pct, oreb_pct, fta_rate, e_pace) should map correctly."""
        espn_standings = pd.DataFrame({
            'team': ['Boston Celtics', 'Los Angeles Clippers'],
            'conference': ['Eastern', 'Western'],
            'wins': [35, 30],
            'losses': [12, 18],
            'win_pct': [0.745, 0.625]
        })

        adv_stats = pd.DataFrame({
            'team_name': ['Boston Celtics', 'LA Clippers'],
            'off_rtg': [118.5, 114.2],
            'def_rtg': [110.1, 111.0],
            'net_rtg': [8.4, 3.2],
            'e_pace': [99.2, 98.4]
        })

        four_factors = pd.DataFrame({
            'teamname': ['BOS', 'LAC'],
            'efg_pct': [0.552, 0.538],
            'tm_tov_pct': [0.121, 0.134],
            'oreb_pct': [0.285, 0.272],
            'fta_rate': [0.245, 0.260]
        })

        merged = self.merger.merge_team_stats(espn_standings, adv_stats, four_factors)

        self.assertEqual(len(merged), 2)
        # Check mapped columns are present and populated
        for col in ['pace', 'off_rating', 'def_rating', 'net_rating', 'tov_pct', 'orb_pct', 'ft_rate']:
            self.assertIn(col, merged.columns, f"Column {col} missing from merged output")
            self.assertTrue(merged[col].notna().all(), f"Column {col} has null values: {merged[col].tolist()}")

        # Check values match
        bos_row = merged[merged['team'] == 'Boston Celtics'].iloc[0]
        self.assertAlmostEqual(bos_row['pace'], 99.2)
        self.assertAlmostEqual(bos_row['tov_pct'], 0.121)
        self.assertAlmostEqual(bos_row['orb_pct'], 0.285)
        self.assertAlmostEqual(bos_row['ft_rate'], 0.245)

    def test_normalize_team_name(self):
        """Verify normalization works for various inputs."""
        self.assertEqual(self.merger.normalize_team_name("LA Clippers"), "Los Angeles Clippers")
        self.assertEqual(self.merger.normalize_team_name("LAC"), "Los Angeles Clippers")
        self.assertEqual(self.merger.normalize_team_name("Clippers"), "Los Angeles Clippers")
        self.assertEqual(self.merger.normalize_team_name("Boston Celtics"), "Boston Celtics")

    def test_merge_game_data_deduplication(self):
        """Verify merge_game_data concatenates and deduplicates games by game_id."""
        espn_games = pd.DataFrame({
            'game_id': ['401584001', '401584002'],
            'date': ['2026-01-10', '2026-01-11'],
            'home_team': ['BOS', 'MIA'],
            'away_team': ['MIA', 'NYK'],
            'home_score': [110, 105],
            'away_score': [100, 102]
        })
        sdv_games = pd.DataFrame({
            'game_id': ['401584001', '401584003'],  # 401584001 is duplicate
            'date': ['2026-01-10', '2026-01-12'],
            'home_team': ['Boston Celtics', 'Denver Nuggets'],
            'away_team': ['Miami Heat', 'LAL'],
            'home_score': [110, 120],
            'away_score': [100, 115]
        })

        merged = self.merger.merge_game_data(espn_games, sdv_games)
        self.assertEqual(len(merged), 3)  # 401584001, 401584002, 401584003



    def test_merge_game_data_excludes_preseason_and_playoffs(self):
        espn_games = pd.DataFrame({
            'game_id': ['1', '2', '3', '4'],
            'date': ['2026-10-10', '2026-05-15', '2026-10-22', '2026-10-24'],
            'home_team': ['BOS', 'BOS', 'BOS', 'BOS'],
            'away_team': ['MIA', 'MIA', 'MIA', 'MIA'],
            'home_score': [95, 98, 120, 125],
            'away_score': [90, 92, 110, 115],
            'season_type': [1, 3, 2, 2]
        })
        merged = self.merger.merge_game_data(espn_games, pd.DataFrame())
        self.assertEqual(len(merged), 2)
        self.assertTrue((merged['game_id'].isin(['3', '4'])).all())

if __name__ == '__main__':
    unittest.main()
