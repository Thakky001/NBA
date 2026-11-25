"""Unit tests for Central Publication Gate and Single-Writer Ledger Architecture."""

import unittest
import pandas as pd

from analysis.publication import (
    resolve_market,
    resolve_analysis,
    resolve_row,
    is_pick_recommendation,
    CLASH_VERDICTS
)
from analysis.prediction_tracker import (
    ensure_published_columns,
    calculate_track_record,
    grade_spread_pick,
    grade_over_under_pick
)


class TestPublicationGate(unittest.TestCase):
    """Test publication gate logic and verdict resolution."""

    def test_market_clash_suppresses_pick(self):
        """When a line-movement verdict is in CLASH_VERDICTS, recommendation becomes PASS."""
        res = resolve_market(
            {'recommendation': 'STRONG HOME', 'line_movement': {'verdict': 'MOVE_CONFLICT'}}
        )
        self.assertFalse(res['is_published'])
        self.assertEqual(res['published_recommendation'], 'PASS')
        self.assertEqual(res['publish_reason'], 'MARKET_CLASH_MOVE_CONFLICT')
        self.assertTrue(res['is_market_clash'])

    def test_capital_preservation_suppresses_pick(self):
        """CAPITAL_PRESERVATION suppresses even STRONG picks."""
        res = resolve_market(
            {'recommendation': 'STRONG OVER', 'line_movement': {'verdict': 'CAPITAL_PRESERVATION'}}
        )
        self.assertFalse(res['is_published'])
        self.assertEqual(res['published_recommendation'], 'PASS')
        self.assertEqual(res['publish_reason'], 'MARKET_CLASH_CAPITAL_PRESERVATION')

    def test_clean_pick_publishes(self):
        """When there is no clash, pick is published as-is."""
        res = resolve_market(
            {'recommendation': 'LEAN AWAY', 'line_movement': {'verdict': 'MOVE_ALIGNED'}}
        )
        self.assertTrue(res['is_published'])
        self.assertEqual(res['published_recommendation'], 'LEAN AWAY')
        self.assertIsNone(res['publish_reason'])
        self.assertFalse(res['is_market_clash'])

    def test_neutral_is_not_published(self):
        """NEUTRAL recommendations are not published as bets."""
        res = resolve_market(
            {'recommendation': 'NEUTRAL', 'line_movement': {'verdict': 'STABLE_MODEL'}}
        )
        self.assertFalse(res['is_published'])
        self.assertEqual(res['published_recommendation'], 'NEUTRAL')
        self.assertEqual(res['publish_reason'], 'NEUTRAL')


class TestPublicationLedgerIntegration(unittest.TestCase):
    """Test ledger publication columns, effective is_actionable, and track record calculation."""

    def test_is_actionable_enforces_model_and_published(self):
        """is_actionable must be model_is_actionable AND is_published."""
        df = pd.DataFrame([{
            'game_id': '101',
            'date': '2026-11-25',
            'home_team': 'ORL',
            'away_team': 'DET',
            'spread_line': -4.5,
            'total_line': 218.0,
            'sp_recommendation': 'STRONG AWAY',
            'sp_verdict': 'MOVE_CONFLICT',
            'sp_is_actionable': True,
            'ou_recommendation': 'LEAN OVER',
            'ou_verdict': 'MOVE_CONFLICT',
            'ou_is_actionable': True,
            'status': 'PENDING',
            'model_version': 'nba-baseline-v3'
        }])

        ensured = ensure_published_columns(df, derive=True)
        # Because of MOVE_CONFLICT, published is False
        self.assertFalse(ensured.loc[0, 'sp_is_published'])
        self.assertEqual(ensured.loc[0, 'sp_published_recommendation'], 'PASS')
        # sp_is_actionable must be False despite raw model thinking it was actionable
        self.assertFalse(ensured.loc[0, 'sp_is_actionable'])
        self.assertTrue(ensured.loc[0, 'sp_model_is_actionable'])
        self.assertEqual(ensured.loc[0, 'sp_suppressed_reason'], 'MARKET_CLASH_MOVE_CONFLICT')

    def test_track_record_published_only(self):
        """calculate_track_record(published_only=True) only counts published bets."""
        df = pd.DataFrame([
            {
                'game_id': '201',
                'status': 'GRADED',
                'model_version': 'nba-baseline-v3',
                'sp_recommendation': 'STRONG HOME',
                'sp_result': 'WIN',
                'ou_recommendation': 'LEAN OVER',
                'ou_result': 'LOSS',
                'sp_is_published': True,
                'sp_published_recommendation': 'STRONG HOME',
                'sp_published_result': 'WIN',
                'ou_is_published': False,
                'ou_published_recommendation': 'PASS',
                'ou_published_result': 'SKIPPED',
            },
            {
                'game_id': '202',
                'status': 'GRADED',
                'model_version': 'nba-baseline-v3',
                'sp_recommendation': 'STRONG AWAY',
                'sp_result': 'WIN',
                'ou_recommendation': 'LEAN UNDER',
                'ou_result': 'LOSS',
                'sp_is_published': False,
                'sp_published_recommendation': 'PASS',
                'sp_published_result': 'SKIPPED',
                'ou_is_published': True,
                'ou_published_recommendation': 'LEAN UNDER',
                'ou_published_result': 'LOSS',
            }
        ])

        # Raw calculate_track_record counts all 4 results
        raw_stats = calculate_track_record(df, published_only=False)
        self.assertEqual(raw_stats['overall']['bets'], 4)

        # Published only counts strictly published (2 bets: 1 WIN, 1 LOSS)
        pub_stats = calculate_track_record(df, published_only=True)
        self.assertEqual(pub_stats['overall']['bets'], 2)
        self.assertEqual(pub_stats['overall']['wins'], 1)
        self.assertEqual(pub_stats['overall']['losses'], 1)
        self.assertEqual(pub_stats['strong']['bets'], 1)
        self.assertEqual(pub_stats['strong']['wins'], 1)
        self.assertEqual(pub_stats['lean']['bets'], 1)
        self.assertEqual(pub_stats['lean']['losses'], 1)


if __name__ == '__main__':
    unittest.main()
