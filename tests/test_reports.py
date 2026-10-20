import unittest
from reports.daily_slate import _parse_spread
from reports.game_report import GameReporter


class TestReports(unittest.TestCase):
    """Tests for reporting modules addressing Bug 9 and Bug 10."""

    def test_parse_spread_bug9(self):
        """Bug 9: Spread parsing should correctly handle EVEN, PK, PICK, team spreads, and bare numbers."""
        # Pick'em variations
        for pk in ['EVEN', 'even', 'PK', 'pk', 'PICK', 'Pick']:
            team, val = _parse_spread(pk)
            self.assertEqual(team, 'PK')
            self.assertEqual(val, 0.0)

        # Standard team spreads
        team, val = _parse_spread('BOS -4.5')
        self.assertEqual(team, 'BOS')
        self.assertEqual(val, -4.5)

        team, val = _parse_spread('MIA +3.0')
        self.assertEqual(team, 'MIA')
        self.assertEqual(val, 3.0)

        # Bare numbers
        team, val = _parse_spread('-5.5')
        self.assertIsNone(team)
        self.assertEqual(val, -5.5)

        # Nulls
        team, val = _parse_spread(None)
        self.assertIsNone(team)
        self.assertIsNone(val)

    def test_format_matchup_report_with_none_profiles_bug10(self):
        """Bug 10: format_matchup_report should not raise AttributeError when home_profile or away_profile is None."""
        reporter = GameReporter()
        analysis = {
            'home_team': 'Los Angeles Clippers',
            'away_team': 'Denver Nuggets',
            'home_profile': None,
            'away_profile': None,
            'spread': None,
            'over_under': None
        }
        report = reporter.format_matchup_report(analysis, game_info={'venue': 'Crypto.com Arena'})
        self.assertIn('Los Angeles Clippers vs Denver Nuggets', report)
        self.assertIn('TEAM PROFILES:', report)
        self.assertIn('Home: 0-0', report)
        self.assertIn('Away: 0-0', report)

    def test_format_team_profile_with_none(self):
        """Verify format_team_profile_report safely handles None profile or None advanced_stats."""
        reporter = GameReporter()
        report = reporter.format_team_profile_report(None, "Boston Celtics")
        self.assertIn('TEAM PROFILE: Boston Celtics', report)
        self.assertIn('Pace: N/A', report)

    def test_format_quick_pick_with_none(self):
        """Verify format_quick_pick safely handles None or partial analysis dicts."""
        reporter = GameReporter()
        quick = reporter.format_quick_pick({'home_team': 'BOS', 'away_team': 'LAL', 'over_under': None, 'spread': None})
        self.assertIn('BOS vs LAL', quick)
        self.assertIn('N/A', quick)

    def test_format_daily_slate_flow(self):
        """Verify format_daily_slate formats a list of games containing None profiles."""
        reporter = GameReporter()
        games = [
            {
                'analysis': {
                    'home_team': 'BOS',
                    'away_team': 'MIA',
                    'home_profile': None,
                    'away_profile': None
                },
                'game_info': {'venue': 'TD Garden'}
            }
        ]
        slate_report = reporter.format_daily_slate(games)
        self.assertIn('DAILY SLATE SUMMARY', slate_report)
        self.assertIn('BOS vs MIA', slate_report)


if __name__ == '__main__':
    unittest.main()
