import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from config import TEAMS, resolve_team_info, normalize_team_name
from processing.player_rating import PlayerRatingEngine


class TestPlayerRating(unittest.TestCase):
    """Tests for PlayerRatingEngine and team resolution addressing Bug 4."""

    def setUp(self):
        self.inj_patch = patch('collectors.espn_collector.ESPNCollector.get_injuries', return_value=pd.DataFrame())
        self.inj_patch.start()
        self.rw_patch = patch('collectors.rotowire_collector.RotoWireCollector.get_todays_lineups', return_value={})
        self.rw_patch.start()

    def tearDown(self):
        self.inj_patch.stop()
        self.rw_patch.stop()

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

    def test_rotowire_missing_uses_the_same_formula(self):
        """A missing lineup must not switch the team onto a different scale."""
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
            # Same 70/30 split as a team with a lineup: 8 * (0.70 * 15.0 + 0.30 * 10.0) = 108.0
            self.assertAlmostEqual(res['roster_rating'], 108.0, places=1)


    def test_starter_suffix_matching_and_fallback(self):
        """Verify starters with missing/differing suffixes (Jr, III) and unlisted players match to 5 starters."""
        mock_rw = MagicMock()
        mock_rw.get_lineup_for_team.return_value = {
            'lineup_status': 'Expected',
            'is_confirmed': False,
            'starters': [
                {'name': 'Jabari Smith', 'position': 'PF'},
                {'name': 'Jaren Jackson', 'position': 'C'},
                {'name': 'Trey Murphy', 'position': 'SF'},
                {'name': 'Jaime Jaquez', 'position': 'SG'},
                {'name': 'New Guy', 'position': 'PG'},
            ]
        }
        engine = PlayerRatingEngine(rotowire_collector=mock_rw)
        engine._roster_cache[666] = pd.DataFrame([
            {'name': 'Jabari Smith Jr.'},
            {'name': 'Jaren Jackson Jr.'},
            {'name': 'Trey Murphy III'},
            {'name': 'Jaime Jaquez Jr.'},
            {'name': 'Bench Ten'},
        ])
        engine.draft_history = pd.DataFrame()
        engine.player_stats = pd.DataFrame([
            {'PLAYER_NAME': 'Jabari Smith Jr.', 'GP': 10, 'EFF_PG': 18.0},
            {'PLAYER_NAME': 'Jaren Jackson Jr.', 'GP': 10, 'EFF_PG': 20.0},
            {'PLAYER_NAME': 'Trey Murphy III', 'GP': 10, 'EFF_PG': 16.0},
            {'PLAYER_NAME': 'Jaime Jaquez Jr.', 'GP': 10, 'EFF_PG': 15.0},
            {'PLAYER_NAME': 'Bench Ten', 'GP': 10, 'EFF_PG': 10.0},
        ])
        engine.injuries_df = pd.DataFrame()

        with patch('config.resolve_team_info', return_value={'espn_id': 666}):
            res = engine.calculate_team_roster_rating('Mock Suffix Team')
            self.assertEqual(res['status'], 'success')
            self.assertEqual(len(res['starters']), 5)
            # 4 matched via suffix removal, 1 matched via fallback dummy starter
            names = [s['name'] for s in res['starters']]
            self.assertIn('Jabari Smith Jr.', names)
            self.assertIn('Jaren Jackson Jr.', names)
            self.assertIn('Trey Murphy III', names)
            self.assertIn('Jaime Jaquez Jr.', names)
            self.assertIn('New Guy', names)


    def test_next_man_up_starter_replacement_when_out(self):
        """Verify that when a confirmed starter is OUT, the 6th man is promoted to starters."""
        mock_rw = MagicMock()
        mock_rw.get_lineup_for_team.return_value = {
            'lineup_status': 'Confirmed',
            'is_confirmed': True,
            'starters': [
                {'name': 'Injured Star', 'position': 'PG'},
                {'name': 'Player Two', 'position': 'SG'},
                {'name': 'Player Three', 'position': 'SF'},
                {'name': 'Player Four', 'position': 'PF'},
                {'name': 'Player Five', 'position': 'C'},
            ]
        }
        engine = PlayerRatingEngine(rotowire_collector=mock_rw)
        engine._roster_cache[555] = pd.DataFrame([
            {'name': 'Injured Star'},
            {'name': 'Player Two'},
            {'name': 'Player Three'},
            {'name': 'Player Four'},
            {'name': 'Player Five'},
            {'name': 'Sixth Man'},
            {'name': 'Bench Seven'},
            {'name': 'Bench Eight'},
            {'name': 'Bench Nine'},
        ])
        engine.draft_history = pd.DataFrame()
        engine.player_stats = pd.DataFrame([
            {'PLAYER_NAME': 'Injured Star', 'GP': 10, 'EFF_PG': 30.0},
            {'PLAYER_NAME': 'Player Two', 'GP': 10, 'EFF_PG': 20.0},
            {'PLAYER_NAME': 'Player Three', 'GP': 10, 'EFF_PG': 20.0},
            {'PLAYER_NAME': 'Player Four', 'GP': 10, 'EFF_PG': 20.0},
            {'PLAYER_NAME': 'Player Five', 'GP': 10, 'EFF_PG': 20.0},
            {'PLAYER_NAME': 'Sixth Man', 'GP': 10, 'EFF_PG': 16.0},
            {'PLAYER_NAME': 'Bench Seven', 'GP': 10, 'EFF_PG': 12.0},
            {'PLAYER_NAME': 'Bench Eight', 'GP': 10, 'EFF_PG': 10.0},
            {'PLAYER_NAME': 'Bench Nine', 'GP': 10, 'EFF_PG': 8.0},
        ])
        # Injured Star is OUT
        engine.injuries_df = pd.DataFrame([
            {'player_name': 'Injured Star', 'status': 'Out', 'detail': 'Knee'}
        ])

        with patch('config.resolve_team_info', return_value={'espn_id': 555}):
            res = engine.calculate_team_roster_rating("Mock Team")
            self.assertEqual(res['status'], 'success')
            starter_names = [s['name'] for s in res['starters']]
            # Injured Star was replaced in starting 5 by Sixth Man
            self.assertIn('Sixth Man', starter_names)
            self.assertNotIn('Injured Star', starter_names)
            # Full roster rating preserves Injured Star (30.0)
            self.assertGreater(res['full_roster_rating'], 150.0)
            # Active roster rating uses Sixth Man (16.0) instead of 0.0 penalty
            self.assertAlmostEqual(res['roster_rating'], 131.5, delta=1.0)
            # Star still listed in injuries
            self.assertTrue(any('Injured Star' in inj and 'OUT' in inj for inj in res['injuries']))

    def test_prefix_name_matching_herb_and_nic(self):
        """Verify prefix matching for common abbreviations (Herb -> Herbert, Nic -> Nicolas)."""
        mock_rw = MagicMock()
        mock_rw.get_lineup_for_team.return_value = {
            'lineup_status': 'Expected',
            'is_confirmed': False,
            'starters': [
                {'name': 'Herb Jones', 'position': 'SF'},
                {'name': 'Nic Claxton', 'position': 'C'},
                {'name': 'Cam Thomas', 'position': 'SG'},
                {'name': 'Normal Player', 'position': 'PG'},
                {'name': 'Fifth Player', 'position': 'PF'},
            ]
        }
        engine = PlayerRatingEngine(rotowire_collector=mock_rw)
        engine._roster_cache[444] = pd.DataFrame([
            {'name': 'Herbert Jones'},
            {'name': 'Nicolas Claxton'},
            {'name': 'Cameron Thomas'},
            {'name': 'Normal Player'},
            {'name': 'Fifth Player'},
        ])
        engine.draft_history = pd.DataFrame()
        engine.player_stats = pd.DataFrame([
            {'PLAYER_NAME': 'Herbert Jones', 'GP': 10, 'EFF_PG': 14.0},
            {'PLAYER_NAME': 'Nicolas Claxton', 'GP': 10, 'EFF_PG': 16.0},
            {'PLAYER_NAME': 'Cameron Thomas', 'GP': 10, 'EFF_PG': 18.0},
            {'PLAYER_NAME': 'Normal Player', 'GP': 10, 'EFF_PG': 15.0},
            {'PLAYER_NAME': 'Fifth Player', 'GP': 10, 'EFF_PG': 13.0},
        ])
        engine.injuries_df = pd.DataFrame()

        with patch('config.resolve_team_info', return_value={'espn_id': 444}):
            res = engine.calculate_team_roster_rating("Mock Team")
            names = [s['name'] for s in res['starters']]
            self.assertIn('Herbert Jones', names)
            self.assertIn('Nicolas Claxton', names)
            self.assertIn('Cameron Thomas', names)


class TestInjurySnapshotRefresh(unittest.TestCase):
    """The engine must not serve one load-once injury report for its whole life.

    The web app caches a single analyzer per process, so a frozen snapshot is how
    lineups end up built from a two-day-old injury report.
    """

    def _mk_engine(self, injuries):
        mock_rw = MagicMock()
        mock_rw.get_lineup_for_team.return_value = None
        with patch('collectors.espn_collector.ESPNCollector.get_injuries', return_value=injuries):
            return PlayerRatingEngine(rotowire_collector=mock_rw)

    def test_snapshot_is_not_refetched_inside_ttl(self):
        initial = pd.DataFrame([{'player_name': 'Star Guy', 'status': 'Out', 'detail': 'Knee'}])
        engine = self._mk_engine(initial)

        with patch('collectors.espn_collector.ESPNCollector.get_injuries',
                   return_value=pd.DataFrame()) as fetch:
            engine._refresh_injuries()
            fetch.assert_not_called()
        self.assertEqual(engine.injuries_df.iloc[0]['player_name'], 'Star Guy')

    def test_snapshot_refreshes_after_ttl(self):
        initial = pd.DataFrame([{'player_name': 'Star Guy', 'status': 'Out', 'detail': 'Knee'}])
        updated = pd.DataFrame([{'player_name': 'Star Guy', 'status': 'Day-To-Day', 'detail': 'Knee'}])
        engine = self._mk_engine(initial)

        engine._injuries_ts -= engine.INJURY_REFRESH_TTL_SECONDS + 1
        with patch('collectors.espn_collector.ESPNCollector.get_injuries',
                   return_value=updated) as fetch:
            engine._refresh_injuries()
            fetch.assert_called_once()
        self.assertEqual(engine.injuries_df.iloc[0]['status'], 'Day-To-Day')

    def test_failed_refresh_keeps_last_good_snapshot(self):
        initial = pd.DataFrame([{'player_name': 'Star Guy', 'status': 'Out', 'detail': 'Knee'}])
        engine = self._mk_engine(initial)

        engine._injuries_ts -= engine.INJURY_REFRESH_TTL_SECONDS + 1
        with patch('collectors.espn_collector.ESPNCollector.get_injuries',
                   return_value=pd.DataFrame()) as fetch:
            engine._refresh_injuries()
            fetch.assert_called_once()
        self.assertEqual(len(engine.injuries_df), 1)

    def test_rating_reports_injury_timestamp_and_refreshes(self):
        initial = pd.DataFrame([{'player_name': 'Star Guy', 'status': 'Out', 'detail': 'Knee'}])
        engine = self._mk_engine(initial)
        engine._roster_cache[321] = pd.DataFrame([{'name': 'Star Guy'}])
        engine.draft_history = pd.DataFrame()
        engine.player_stats = pd.DataFrame([{'PLAYER_NAME': 'Star Guy', 'GP': 10, 'EFF_PG': 20.0}])
        engine._injuries_ts -= engine.INJURY_REFRESH_TTL_SECONDS + 1

        with patch('config.resolve_team_info', return_value={'espn_id': 321}), \
                patch('collectors.espn_collector.ESPNCollector.get_injuries',
                      return_value=pd.DataFrame([{'player_name': 'Other Guy', 'status': 'Out',
                                                  'detail': 'Ankle'}])) as fetch:
            res = engine.calculate_team_roster_rating("Mock Team")

        fetch.assert_called_once()
        self.assertRegex(res['injuries_as_of'], r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$')
        # The stale OUT tag cleared once the fresh report arrived.
        self.assertEqual(res['injuries'], [])


class TestRosterRatingBaselines(unittest.TestCase):
    """Full-health baseline, injury cap, EFF split and cold-start fallback."""

    ROSTER = [
        'Star', 'Player Two', 'Player Three', 'Player Four', 'Player Five',
        'Sixth Man', 'Bench Seven', 'Bench Eight', 'Bench Nine',
    ]
    STATS = {
        'Star': 30.0, 'Player Two': 20.0, 'Player Three': 20.0,
        'Player Four': 20.0, 'Player Five': 20.0, 'Sixth Man': 16.0,
        'Bench Seven': 12.0, 'Bench Eight': 10.0, 'Bench Nine': 8.0,
    }

    def _mk_engine(self, starters, stats=None, injuries=None):
        mock_rw = MagicMock()
        mock_rw.get_lineup_for_team.return_value = (
            {
                'lineup_status': 'Expected',
                'is_confirmed': False,
                'starters': [{'name': n, 'position': ''} for n in starters],
            }
            if starters else None
        )
        engine = PlayerRatingEngine(rotowire_collector=mock_rw)
        engine._roster_cache[999] = pd.DataFrame([{'name': n} for n in self.ROSTER])
        engine.draft_history = pd.DataFrame()
        engine.prev_season_eff = {}
        table = dict(stats if stats is not None else self.STATS)
        rows = []
        for name, eff in table.items():
            rows.append({
                'PLAYER_NAME': name, 'GP': 10, 'EFF_PG': eff,
                'PTS': eff * 10, 'AST': eff * 2, 'OREB': eff,
                'STL': 1.0, 'BLK': 1.0, 'DREB': eff * 2,
            })
        engine.player_stats = pd.DataFrame(rows)
        engine.injuries_df = pd.DataFrame(injuries or [])
        return engine

    def _rate(self, engine):
        with patch('config.resolve_team_info', return_value={'espn_id': 999}):
            return engine.calculate_team_roster_rating('Mock Team')

    def test_full_rating_ignores_which_five_rotowire_names(self):
        """The healthy baseline is the best five on the roster, not tonight's five."""
        healthy = self._rate(self._mk_engine(
            ['Star', 'Player Two', 'Player Three', 'Player Four', 'Player Five']))
        depleted = self._rate(self._mk_engine(
            ['Sixth Man', 'Bench Seven', 'Bench Eight', 'Bench Nine', 'Star']))

        self.assertAlmostEqual(
            healthy['full_roster_rating'], depleted['full_roster_rating'], delta=0.01,
            msg="RotoWire naming a bench five must not move the full-health baseline",
        )

    def test_injury_cap_applies_without_a_rotowire_lineup(self):
        """A missing lineup must not bypass the 22-point ceiling."""
        stats = dict(self.STATS)
        stats['Star'] = 40.0
        engine = self._mk_engine([], stats=stats,
                                 injuries=[{'player_name': 'Star', 'status': 'Out', 'detail': 'Knee'}])
        res = self._rate(engine)

        self.assertEqual(res['status'], 'success')
        # No RotoWire lineup: same formula, same ceiling.
        self.assertAlmostEqual(res['full_roster_rating'] - res['roster_rating'], 22.0, delta=0.05)

    def test_cap_is_reported_with_its_uncapped_value(self):
        """The pre-cap rating must survive, or the cap can never be tuned."""
        stats = dict(self.STATS)
        stats['Star'] = 40.0
        capped = self._rate(self._mk_engine(
            [], stats=stats,
            injuries=[{'player_name': 'Star', 'status': 'Out', 'detail': 'Knee'}]))

        self.assertTrue(capped['injury_cap_bound'])
        # full 164.8 - uncapped 131.5 = 33.3, clipped to 22.0
        self.assertAlmostEqual(
            capped['full_roster_rating'] - capped['uncapped_roster_rating'], 33.28, delta=0.1)
        self.assertAlmostEqual(
            capped['full_roster_rating'] - capped['roster_rating'], 22.0, delta=0.05)

        healthy = self._rate(self._mk_engine([]))
        self.assertFalse(healthy['injury_cap_bound'])
        self.assertAlmostEqual(
            healthy['uncapped_roster_rating'], healthy['roster_rating'], delta=0.01)

    def test_off_and_def_add_up_to_eff(self):
        """off_loss + def_loss must equal the real EFF drop, with no inflation."""
        engine = self._mk_engine(
            ['Player Five', 'Star', 'Player Two', 'Player Three', 'Player Four'],
            injuries=[{'player_name': 'Star', 'status': 'Out', 'detail': 'Knee'}])
        res = self._rate(engine)

        self.assertAlmostEqual(
            res['full_off_rating'] + res['full_def_rating'], res['full_roster_rating'], delta=0.05)
        self.assertAlmostEqual(
            res['active_off_rating'] + res['active_def_rating'], res['roster_rating'], delta=0.05)
        self.assertAlmostEqual(
            res['off_loss'] + res['def_loss'],
            res['full_roster_rating'] - res['roster_rating'], delta=0.05)

    def test_cold_start_uses_previous_season_eff(self):
        """A returning player with no current-season line must not read as 4.5."""
        engine = self._mk_engine([])
        # Only eight players have a current line; "Star" is the returning one.
        stats = {k: v for k, v in self.STATS.items() if k != 'Star'}
        engine.player_stats = pd.DataFrame([
            {'PLAYER_NAME': n, 'GP': 10, 'EFF_PG': e} for n, e in stats.items()
        ])
        engine.prev_season_eff = {'star': {'eff_pg': 21.5, 'gp': 40}}
        res = self._rate(engine)

        # top-5 = 21.5, 20, 20, 20, 20 (avg 20.3); next-3 = 16, 12, 10 (avg 12.67)
        # 8 * (0.70 * 20.3 + 0.30 * 12.67) = 144.1
        self.assertAlmostEqual(res['full_roster_rating'], 144.1, delta=0.1)

    def test_cold_start_falls_back_to_draft_tier_without_a_sample(self):
        """A thin previous season (GP < 10) still uses the draft tier."""
        engine = self._mk_engine([])
        stats = {k: v for k, v in self.STATS.items() if k != 'Star'}
        engine.player_stats = pd.DataFrame([
            {'PLAYER_NAME': n, 'GP': 10, 'EFF_PG': e} for n, e in stats.items()
        ])
        engine.prev_season_eff = {'star': {'eff_pg': 21.5, 'gp': 3}}
        engine.draft_history = pd.DataFrame([{'PLAYER_NAME': 'Star', 'OVERALL_PICK': 3}])
        res = self._rate(engine)

        # top-5 = 20, 20, 20, 20, 16 (avg 19.2); next-3 = 14.0 (top-5 pick tier), 12, 10 (avg 12.0)
        # 8 * (0.70 * 19.2 + 0.30 * 12.0) = 136.3
        self.assertAlmostEqual(res['full_roster_rating'], 136.3, delta=0.1)


    def test_active_rating_ignores_which_five_rotowire_names(self):
        """Tonight's rating comes from value, not from RotoWire's five."""
        healthy = self._rate(self._mk_engine(
            ['Star', 'Player Two', 'Player Three', 'Player Four', 'Player Five']))
        depleted_lineup = self._rate(self._mk_engine(
            ['Sixth Man', 'Bench Seven', 'Bench Eight', 'Bench Nine', 'Star']))

        self.assertAlmostEqual(
            healthy['roster_rating'], depleted_lineup['roster_rating'], delta=0.01,
            msg="RotoWire naming a bench five must not move the active rating",
        )
        # The lineup card still reports what RotoWire actually said.
        self.assertEqual(
            [s['name'] for s in depleted_lineup['starters']],
            ['Sixth Man', 'Bench Seven', 'Bench Eight', 'Bench Nine', 'Star'],
        )

    def test_replacement_gets_starter_minutes(self):
        """A bench body covering an injured starter is floored to 12.0 EFF."""
        stats = {
            'Star': 24.0, 'Player Two': 20.0, 'Player Three': 20.0,
            'Player Four': 20.0, 'Player Five': 20.0, 'Sixth Man': 9.0,
            'Bench Seven': 8.0, 'Bench Eight': 7.0, 'Bench Nine': 6.0,
        }
        out = self._rate(self._mk_engine(
            ['Player Two', 'Player Three', 'Player Four', 'Player Five', 'Sixth Man'],
            stats=stats,
            injuries=[{'player_name': 'Star', 'status': 'Out', 'detail': 'Knee'}]))

        # top-5 = 24, 20, 20, 20, 20 (avg 20.8); next-3 = 9, 8, 7 (avg 8.0)
        # full = 8 * (0.70 * 20.8 + 0.30 * 8.0) = 135.7
        self.assertAlmostEqual(out['full_roster_rating'], 135.7, delta=0.1)
        # Sixth Man (9.0) covers for Star, so he is valued at the 12.0 floor:
        # 8 * (0.70 * ((80 + 12) / 5) + 0.30 * ((8 + 7 + 6) / 3)) = 119.8
        self.assertAlmostEqual(out['roster_rating'], 119.8, delta=0.2)


    def test_unavailable_player_is_never_given_the_floor(self):
        """A ruled-out body stays at zero even when the roster is short."""
        engine = self._mk_engine([])
        engine._roster_cache[999] = pd.DataFrame(
            [{'name': n} for n in ['A', 'B', 'C', 'D', 'E']]
        )
        engine.player_stats = pd.DataFrame([
            {'PLAYER_NAME': 'A', 'GP': 10, 'EFF_PG': 20.0},
            {'PLAYER_NAME': 'B', 'GP': 10, 'EFF_PG': 20.0},
            {'PLAYER_NAME': 'C', 'GP': 10, 'EFF_PG': 20.0},
            {'PLAYER_NAME': 'D', 'GP': 10, 'EFF_PG': 6.0},
            {'PLAYER_NAME': 'E', 'GP': 10, 'EFF_PG': 8.0},
        ])
        engine.injuries_df = pd.DataFrame(
            [{'player_name': 'E', 'status': 'Out', 'detail': 'Ankle'}]
        )
        res = self._rate(engine)

        # healthy top-5 = 20, 20, 20, 8, 6 (avg 14.8), no bench left
        self.assertAlmostEqual(res['full_roster_rating'], 82.9, delta=0.1)
        # active core = 20, 20, 20, 6, 0 - E is out and must not take the floor
        # 8 * 0.70 * (66 / 5) = 73.9
        self.assertAlmostEqual(res['roster_rating'], 73.9, delta=0.2)


if __name__ == '__main__':
    unittest.main()
