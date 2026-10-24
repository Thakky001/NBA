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

    def test_rookie_tiers_and_small_sample_blending(self):
        """Verify elevated rookie base tiers and small-sample (GP < 5) blending."""
        engine = PlayerRatingEngine()
        # Mock active roster with 4 mock players
        engine._roster_cache[999] = pd.DataFrame([
            {'name': 'Rookie TopPick'},
            {'name': 'Rookie Lottery'},
            {'name': 'Rookie RoundOne'},
            {'name': 'Undrafted Guy'},
            {'name': 'Hot Starter'},
        ])
        # Mock draft history
        engine.draft_history = pd.DataFrame([
            {'PLAYER_NAME': 'Rookie TopPick', 'OVERALL_PICK': 2},
            {'PLAYER_NAME': 'Rookie Lottery', 'OVERALL_PICK': 10},
            {'PLAYER_NAME': 'Rookie RoundOne', 'OVERALL_PICK': 22},
            {'PLAYER_NAME': 'Hot Starter', 'OVERALL_PICK': 8},
        ])
        # Mock player stats: Hot Starter has played 2 games with EFF 20.0
        engine.player_stats = pd.DataFrame([
            {'PLAYER_NAME': 'Hot Starter', 'GP': 2, 'EFF_PG': 20.0},
        ])
        engine.injuries_df = pd.DataFrame()

        # Resolve team ID to 999
        with patch('config.resolve_team_info', return_value={'espn_id': 999}):
            res = engine.calculate_team_roster_rating("Mock Team")
            top_players_str = " | ".join(res['top_players'])

            # Hot Starter: pick 8 (base 9.0), 2 GP @ 20.0 EFF -> (20*2 + 9*3)/5 = 13.4 EFF
            self.assertIn('Hot Starter (13.4)', top_players_str)

            # Rookie TopPick: pick 2 -> 14.0 base
            self.assertIn('Rookie TopPick (Base) (14.0)', top_players_str)

            # Rookie Lottery: pick 10 -> 9.0 base
            self.assertIn('Rookie Lottery (Base) (9.0)', top_players_str)

            # Rookie RoundOne: pick 22 -> 6.0 base
            self.assertIn('Rookie RoundOne (Base) (6.0)', top_players_str)

            # Undrafted Guy: unmapped -> 4.5 base
            self.assertIn('Undrafted Guy (Base) (4.5)', top_players_str)

    def test_injury_matching_with_base_tag(self):
        """Verify cold-start players marked (Base) correctly match injuries and drop active_eff to 0.0."""
        engine = PlayerRatingEngine()
        engine._roster_cache[888] = pd.DataFrame([
            {'name': 'Kon Knueppel'},
            {'name': 'Healthy Player'},
        ])
        engine.draft_history = pd.DataFrame([
            {'PLAYER_NAME': 'Kon Knueppel', 'OVERALL_PICK': 4},
        ])
        engine.player_stats = pd.DataFrame() # No stats yet (cold start)
        engine.injuries_df = pd.DataFrame([
            {'player_name': 'Kon Knueppel', 'status': 'Out', 'detail': 'Hamstring'}
        ])

        with patch('config.resolve_team_info', return_value={'espn_id': 888}):
            res = engine.calculate_team_roster_rating("Mock Hornets")
            self.assertEqual(res['status'], 'success')
            # Injury must be recognized in injuries list
            self.assertTrue(any('Kon Knueppel' in inj and 'OUT' in inj for inj in res['injuries']))
            # Active roster rating must be penalized (Kon Knueppel gets 0.0 active EFF)
            top_players_str = " | ".join(res['top_players'])
            self.assertIn('Kon Knueppel (Base) (0.0 - OUT - Hamstring)', top_players_str)

    def test_rotowire_70_30_weighted_rating(self):
        """Verify that 5 confirmed starters receive 70% weight and bench receives 30% weight."""
        mock_rw = MagicMock()
        mock_rw.get_lineup_for_team.return_value = {
            'lineup_status': 'Confirmed',
            'is_confirmed': True,
            'starters': [
                {'name': 'Player One', 'position': 'PG'},
                {'name': 'Player Two', 'position': 'SG'},
                {'name': 'Player Three', 'position': 'SF'},
                {'name': 'Player Four', 'position': 'PF'},
                {'name': 'Player Five', 'position': 'C'},
            ]
        }
        engine = PlayerRatingEngine(rotowire_collector=mock_rw)
        engine._roster_cache[777] = pd.DataFrame([
            {'name': 'Player One'},
            {'name': 'Player Two'},
            {'name': 'Player Three'},
            {'name': 'Player Four'},
            {'name': 'Player Five'},
            {'name': 'Bench Six'},
            {'name': 'Bench Seven'},
            {'name': 'Bench Eight'},
        ])
        engine.draft_history = pd.DataFrame()
        # Starters have 20.0 EFF each, Bench have 10.0 EFF each
        engine.player_stats = pd.DataFrame([
            {'PLAYER_NAME': 'Player One', 'GP': 10, 'EFF_PG': 20.0},
            {'PLAYER_NAME': 'Player Two', 'GP': 10, 'EFF_PG': 20.0},
            {'PLAYER_NAME': 'Player Three', 'GP': 10, 'EFF_PG': 20.0},
            {'PLAYER_NAME': 'Player Four', 'GP': 10, 'EFF_PG': 20.0},
            {'PLAYER_NAME': 'Player Five', 'GP': 10, 'EFF_PG': 20.0},
            {'PLAYER_NAME': 'Bench Six', 'GP': 10, 'EFF_PG': 10.0},
            {'PLAYER_NAME': 'Bench Seven', 'GP': 10, 'EFF_PG': 10.0},
            {'PLAYER_NAME': 'Bench Eight', 'GP': 10, 'EFF_PG': 10.0},
        ])
        engine.injuries_df = pd.DataFrame()

        with patch('config.resolve_team_info', return_value={'espn_id': 777}):
            res = engine.calculate_team_roster_rating("Mock Team")
            self.assertEqual(res['status'], 'success')
            self.assertEqual(res['lineup_status'], 'Confirmed')
            self.assertTrue(res['is_lineup_confirmed'])
            self.assertEqual(len(res['starters']), 5)
            # Weighted expected rating: 8 * (0.70 * 20.0 + 0.30 * 10.0) = 8 * (14.0 + 3.0) = 8 * 17.0 = 136.0
            self.assertAlmostEqual(res['roster_rating'], 136.0, places=1)

    def test_rotowire_missing_silent_fallback(self):
        """Verify clean fallback to legacy top 8 unweighted calculation when RotoWire is unavailable."""
        mock_rw = MagicMock()
        mock_rw.get_lineup_for_team.return_value = None
        engine = PlayerRatingEngine(rotowire_collector=mock_rw)
        engine._roster_cache[777] = pd.DataFrame([
            {'name': 'Player One'},
            {'name': 'Player Two'},
            {'name': 'Player Three'},
            {'name': 'Player Four'},
            {'name': 'Player Five'},
            {'name': 'Bench Six'},
            {'name': 'Bench Seven'},
            {'name': 'Bench Eight'},
        ])
        engine.draft_history = pd.DataFrame()
        engine.player_stats = pd.DataFrame([
            {'PLAYER_NAME': f'Player {i}', 'GP': 10, 'EFF_PG': 15.0} for i in ['One', 'Two', 'Three', 'Four', 'Five']
        ] + [
            {'PLAYER_NAME': f'Bench {j}', 'GP': 10, 'EFF_PG': 10.0} for j in ['Six', 'Seven', 'Eight']
        ])
        engine.injuries_df = pd.DataFrame()

        with patch('config.resolve_team_info', return_value={'espn_id': 777}):
            res = engine.calculate_team_roster_rating("Mock Team")
            self.assertEqual(res['status'], 'success')
            self.assertEqual(res['lineup_status'], 'Unconfirmed')
            self.assertFalse(res['is_lineup_confirmed'])
            # Legacy expected rating: 5 * 15.0 + 3 * 10.0 = 75.0 + 30.0 = 105.0
            self.assertAlmostEqual(res['roster_rating'], 105.0, places=1)


if __name__ == '__main__':
    unittest.main()
