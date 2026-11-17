"""Tests for the unified decision gate (strict-sample gate v2.4)."""

import os
import unittest
from unittest.mock import patch

from analysis.recommendation import GATE_VERSION, STRONG_MIN_GAMES, decide_pick


def _flags(home_games, away_games, **extra):
    flags = {'home_games_played': home_games, 'away_games_played': away_games}
    flags.update(extra)
    return flags


class TestStrictSampleGate(unittest.TestCase):
    """STRONG requires >= 15 current-season games for both teams."""

    def test_gate_version_is_strict_sample(self):
        self.assertEqual(GATE_VERSION, 'gate-v2.4-strict-sample')
        self.assertEqual(STRONG_MIN_GAMES, 15)

    def test_spread_14_games_demoted_to_lean(self):
        res = decide_pick(
            market_type='spread',
            current_line=-3.5,
            projected_val=8.0,
            edge=4.5,
            fair_prob=0.56,
            data_quality_flags=_flags(14, 14),
        )
        self.assertEqual(res['recommendation'], 'LEAN HOME')
        self.assertEqual(res['suppressed_reason'], 'EARLY_SEASON_SAMPLE_RESTRICTION')
        self.assertFalse(res['is_actionable'])

    def test_spread_15_games_allows_strong(self):
        res = decide_pick(
            market_type='spread',
            current_line=-3.5,
            projected_val=8.0,
            edge=4.5,
            fair_prob=0.56,
            data_quality_flags=_flags(15, 15),
        )
        self.assertEqual(res['recommendation'], 'STRONG HOME')
        self.assertTrue(res['is_actionable'])
        self.assertIsNone(res['suppressed_reason'])
        self.assertEqual(res['gate_version'], GATE_VERSION)

    def test_spread_20_games_allows_strong(self):
        res = decide_pick(
            market_type='spread',
            current_line=-3.5,
            projected_val=7.0,
            edge=3.5,
            fair_prob=0.54,
            data_quality_flags=_flags(20, 20),
        )
        self.assertEqual(res['recommendation'], 'STRONG HOME')
        self.assertTrue(res['is_actionable'])

    def test_away_side_uses_away_probability(self):
        res = decide_pick(
            market_type='spread',
            current_line=3.5,
            projected_val=-8.0,
            edge=-4.5,
            fair_prob=0.36,
            data_quality_flags=_flags(15, 15),
        )
        self.assertEqual(res['recommendation'], 'STRONG AWAY')
        self.assertGreater(res['pick_prob'], 0.50)

    def test_unknown_sample_size_fails_closed(self):
        res = decide_pick(
            market_type='spread',
            current_line=-3.5,
            projected_val=8.0,
            edge=4.5,
            fair_prob=0.56,
        )
        self.assertEqual(res['recommendation'], 'LEAN HOME')
        self.assertEqual(res['suppressed_reason'], 'SAMPLE_SIZE_UNVERIFIED')
        self.assertFalse(res['is_actionable'])

    def test_early_season_flag_does_not_bypass_floor(self):
        res = decide_pick(
            market_type='spread',
            current_line=-3.5,
            projected_val=8.0,
            edge=6.0,
            fair_prob=0.60,
            data_quality_flags=_flags(14, 14),
            is_early_season=True,
        )
        self.assertIn('LEAN', res['recommendation'])
        self.assertFalse(res['is_actionable'])

    def test_total_14_games_demoted_to_lean(self):
        res = decide_pick(
            market_type='total',
            current_line=220.0,
            projected_val=226.0,
            edge=6.0,
            fair_prob=0.56,
            data_quality_flags=_flags(14, 14),
        )
        self.assertEqual(res['recommendation'], 'LEAN OVER')
        self.assertEqual(res['suppressed_reason'], 'EARLY_SEASON_SAMPLE_RESTRICTION')
        self.assertFalse(res['is_actionable'])

    def test_total_15_games_allows_strong(self):
        res = decide_pick(
            market_type='total',
            current_line=220.0,
            projected_val=225.5,
            edge=5.5,
            fair_prob=0.54,
            data_quality_flags=_flags(15, 15),
        )
        self.assertEqual(res['recommendation'], 'STRONG OVER')
        self.assertTrue(res['is_actionable'])

    def test_total_under_15_games_allows_strong(self):
        res = decide_pick(
            market_type='total',
            current_line=220.0,
            projected_val=217.0,
            edge=-3.0,
            fair_prob=0.44,
            data_quality_flags=_flags(15, 15),
        )
        self.assertEqual(res['recommendation'], 'STRONG UNDER')
        self.assertTrue(res['is_actionable'])

    def test_extreme_edge_still_vetoed(self):
        res = decide_pick(
            market_type='spread',
            current_line=-3.5,
            projected_val=13.0,
            edge=9.5,
            fair_prob=0.65,
            data_quality_flags=_flags(20, 20),
        )
        self.assertEqual(res['recommendation'], 'NEUTRAL')
        self.assertEqual(res['suppressed_reason'], 'EXTREME_EDGE_MODEL_ANOMALY')
        self.assertFalse(res['is_actionable'])

    def test_unreliable_data_still_vetoed(self):
        res = decide_pick(
            market_type='spread',
            current_line=-3.5,
            projected_val=8.0,
            edge=4.5,
            fair_prob=0.56,
            data_quality_flags=_flags(20, 20, dq_is_reliable=False),
        )
        self.assertEqual(res['recommendation'], 'NEUTRAL')
        self.assertEqual(res['suppressed_reason'], 'VETO_UNRELIABLE_DATA_QUALITY')
        self.assertFalse(res['is_actionable'])

    def test_low_sample_veto_unchanged(self):
        res = decide_pick(
            market_type='spread',
            current_line=-3.5,
            projected_val=8.0,
            edge=4.5,
            fair_prob=0.56,
            data_quality_flags=_flags(3, 3),
        )
        self.assertEqual(res['recommendation'], 'NEUTRAL')
        self.assertEqual(res['suppressed_reason'], 'VETO_LOW_SAMPLE_SIZE')


def _strong_total(**overrides):
    kwargs = dict(
        market_type='total',
        current_line=220.0,
        projected_val=225.5,
        edge=5.5,
        fair_prob=0.54,
        data_quality_flags=_flags(20, 20),
    )
    kwargs.update(overrides)
    return decide_pick(**kwargs)


class TestSelectiveGate(unittest.TestCase):
    """The selective gate is off by default and can only suppress picks."""

    def test_disabled_by_default(self):
        with patch.dict(os.environ, {'NBA_OU_SELECTIVE_GATE': 'false'}):
            res = _strong_total()
        self.assertFalse(res['selective_gate_enabled'])
        self.assertEqual(res['recommendation'], 'STRONG OVER')
        self.assertIsNone(res['selective_rule'])

    def test_enabled_without_profile_fails_closed(self):
        with patch.dict(os.environ, {'NBA_OU_SELECTIVE_GATE': 'true'}):
            res = _strong_total()
        self.assertTrue(res['selective_gate_enabled'])
        self.assertEqual(res['recommendation'], 'NEUTRAL')
        self.assertEqual(res['suppressed_reason'], 'SELECTIVE_GATE_NO_MATCH')
        self.assertFalse(res['is_actionable'])

    def test_matching_rule_keeps_pick(self):
        with patch.dict(os.environ, {'NBA_OU_SELECTIVE_GATE': 'true'}):
            res = _strong_total(
                selective_profile={
                    'version': 'ou-selective-v1.0',
                    'allowed_rules': ['strong_gp15'],
                }
            )
        self.assertEqual(res['recommendation'], 'STRONG OVER')
        self.assertEqual(res['selective_rule'], 'strong_gp15')
        self.assertTrue(res['is_actionable'])

    def test_rule_cannot_rescue_demoted_sample(self):
        with patch.dict(os.environ, {'NBA_OU_SELECTIVE_GATE': 'true'}):
            res = _strong_total(
                data_quality_flags=_flags(14, 14),
                selective_profile={'allowed_rules': ['strong_gp15']},
            )
        self.assertEqual(res['recommendation'], 'NEUTRAL')
        self.assertEqual(res['suppressed_reason'], 'SELECTIVE_GATE_NO_MATCH')

    def test_side_specific_rule_suppresses_other_side(self):
        with patch.dict(os.environ, {'NBA_OU_SELECTIVE_GATE': 'true'}):
            res = _strong_total(
                selective_profile={'allowed_rules': ['under_only']},
            )
        self.assertEqual(res['recommendation'], 'NEUTRAL')

    def test_profile_thresholds_apply(self):
        with patch.dict(os.environ, {'NBA_OU_SELECTIVE_GATE': 'true'}):
            res = _strong_total(
                selective_profile={
                    'allowed_rules': ['all_picks'],
                    'min_edge': 6.0,
                }
            )
        self.assertEqual(res['recommendation'], 'NEUTRAL')
        self.assertEqual(res['suppressed_reason'], 'SELECTIVE_GATE_NO_MATCH')

    def test_spread_is_not_affected_by_ou_gate(self):
        with patch.dict(os.environ, {'NBA_OU_SELECTIVE_GATE': 'true'}):
            res = decide_pick(
                market_type='spread',
                current_line=-3.5,
                projected_val=8.0,
                edge=4.5,
                fair_prob=0.56,
                data_quality_flags=_flags(20, 20),
            )
        self.assertEqual(res['recommendation'], 'STRONG HOME')
        self.assertIsNone(res['selective_rule'])


if __name__ == '__main__':
    unittest.main()
