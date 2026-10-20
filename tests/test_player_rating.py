import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from config import TEAMS, resolve_team_info, normalize_team_name
from processing.player_rating import PlayerRatingEngine


class TestPlayerRating(unittest.TestCase):
    """Tests for PlayerRatingEngine and team resolution addressing Bug 4."""

    def test_all_30_teams_resolution(self):
        """Ensure all 30 NBA teams resolve properly from full name, abbrev, and common aliases."""
        for abbrev, info in TEAMS.items():
            name = info['name']
            # Test full name
            resolved_by_full = resolve_team_info(name)
            self.assertIsNotNone(resolved_by_full, f"Failed to resolve {name}")
            self.assertEqual(resolved_by_full['abbreviation'], abbrev)

            # Test abbrev
            resolved_by_abbrev = resolve_team_info(abbrev)
            self.assertIsNotNone(resolved_by_abbrev, f"Failed to resolve {abbrev}")
            self.assertEqual(resolved_by_abbrev['abbreviation'], abbrev)

    def test_clippers_aliases_resolution_bug4(self):
        """Bug 4: Clippers variations should all resolve to LAC and ESPN ID 12."""
        variations = ["Los Angeles Clippers", "LA Clippers", "Clippers", "LAC"]
        for variant in variations:
            info = resolve_team_info(variant)
            self.assertIsNotNone(info, f"Failed for {variant}")
            self.assertEqual(info['abbreviation'], 'LAC')
            self.assertEqual(info['espn_id'], 12)
            self.assertEqual(normalize_team_name(variant), 'Los Angeles Clippers')

    def test_clippers_roster_rating_calculation(self):
        """Verify PlayerRatingEngine calculates roster rating for Clippers without fallback."""
        engine = PlayerRatingEngine()
        rating = engine.calculate_team_roster_rating("Los Angeles Clippers")
        self.assertEqual(rating['status'], 'success')
        self.assertGreater(rating['roster_rating'], 50.0)

        # Same for LA Clippers alias
        rating_alias = engine.calculate_team_roster_rating("LA Clippers")
        self.assertEqual(rating_alias['status'], 'success')
        self.assertEqual(rating_alias['roster_rating'], rating['roster_rating'])

    def test_api_failure_cache_fallback(self):
        """Verify engine falls back to local cache if nba_api or remote fetch raises Exception."""
        # Force file_age_days > 7 by setting mtime far in the past or mocking time.time
        with patch('time.time', return_value=9999999999.0):
            # When LeagueDashPlayerStats raises Exception, it should fall back to cache
            with patch('nba_api.stats.endpoints.leaguedashplayerstats.LeagueDashPlayerStats', side_effect=Exception("API connection timeout")):
                engine = PlayerRatingEngine()
                self.assertFalse(engine.player_stats.empty)
                self.assertIn('PLAYER_NAME', engine.player_stats.columns)


if __name__ == '__main__':
    unittest.main()
