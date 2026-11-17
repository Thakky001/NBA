"""Tests for pre-registered subset research and multiple-testing control."""

import json
import unittest

import pandas as pd

from analysis.ou_subsets import (
    MIN_PROMOTION_BETS,
    SUBSET_RULES,
    _binomial_upper_p,
    _rule_mask,
    benjamini_hochberg,
    build_subset_frame,
    run_subset_research,
)


def synthetic_ledger(n=260):
    rows = []
    styles = ['FAST_VS_FAST', 'SLOW_VS_SLOW', 'AVERAGE']
    for i in range(n):
        edge = float((i % 9) - 4)
        line = 225.0
        projection = line + edge
        actual = line + (4.0 if i % 4 == 0 else -4.0)
        if edge >= 2:
            rec = 'STRONG OVER'
        elif edge <= -2:
            rec = 'STRONG UNDER'
        elif edge > 0:
            rec = 'LEAN OVER'
        else:
            rec = 'LEAN UNDER'
        rows.append({
            'game_id': f'g{i:04d}',
            'date': (pd.Timestamp('2026-10-21') + pd.Timedelta(days=i)).strftime('%Y-%m-%d'),
            'home_team': 'Home',
            'away_team': 'Away',
            'total_line': line,
            'actual_total': actual,
            'projected_total': projection,
            'ou_recommendation': rec,
            'ou_edge': edge,
            'ou_win_prob': 0.54,
            'features_snapshot': json.dumps({
                'home_games_played': 20,
                'away_games_played': 20,
                'combined_avg_season': 220.0,
                'matchup_pace_style': styles[i % len(styles)],
                'injury_net_total_impact': -5.0 if i % 5 == 0 else 0.0,
                'fatigue_flag': i % 6 == 0,
            }),
        })
    return pd.DataFrame(rows)


class TestSubsetFrame(unittest.TestCase):
    def test_frame_and_derived_fields(self):
        frame, diagnostics = build_subset_frame(synthetic_ledger(60))
        self.assertEqual(diagnostics['rows_usable'], 60)
        self.assertGreater(len(frame), 0)
        self.assertIn('line_minus_season', frame.columns)
        self.assertIn('pace_style', frame.columns)
        self.assertTrue((frame['line_minus_season'] == 5.0).all())

    def test_invalid_snapshot_skipped(self):
        ledger = synthetic_ledger(20)
        ledger.loc[0, 'features_snapshot'] = 'not json'
        frame, diagnostics = build_subset_frame(ledger)
        self.assertEqual(diagnostics['skipped']['invalid_snapshot'], 1)
        self.assertEqual(len(frame), 19)


class TestSubsetRules(unittest.TestCase):
    def setUp(self):
        self.frame, _ = build_subset_frame(synthetic_ledger(260))

    def test_every_registered_rule_has_a_mask(self):
        for rule in SUBSET_RULES:
            mask = _rule_mask(self.frame, rule['id'])
            self.assertEqual(len(mask), len(self.frame), rule['id'])

    def test_gp_rule_requires_both_teams(self):
        frame = self.frame.copy()
        frame.loc[0, 'games_played'] = 14
        mask = _rule_mask(frame, 'strong_gp15')
        self.assertFalse(bool(mask.iloc[0]))

    def test_strong_under_only_selects_under(self):
        subset = self.frame[_rule_mask(self.frame, 'strong_under_gp15')]
        if not subset.empty:
            self.assertTrue((subset['side'] == 'UNDER').all())

    def test_fast_and_slow_rules_use_style(self):
        fast = self.frame[_rule_mask(self.frame, 'fast_vs_fast_over')]
        slow = self.frame[_rule_mask(self.frame, 'slow_vs_slow_under')]
        if not fast.empty:
            self.assertTrue((fast['pace_style'] == 'FAST_VS_FAST').all())
            self.assertTrue((fast['side'] == 'OVER').all())
        if not slow.empty:
            self.assertTrue((slow['pace_style'] == 'SLOW_VS_SLOW').all())
            self.assertTrue((slow['side'] == 'UNDER').all())

    def test_unknown_rule_selects_nothing(self):
        self.assertFalse(_rule_mask(self.frame, 'does_not_exist').any())


class TestStatistics(unittest.TestCase):
    def test_binomial_p_value_direction(self):
        low = _binomial_upper_p(5, 10)
        high = _binomial_upper_p(9, 10)
        self.assertGreater(low, 0.5)
        self.assertLess(high, 0.5)

    def test_bh_flags_only_strong_signal(self):
        result = benjamini_hochberg(
            {'a': 0.0001, 'b': 0.4, 'c': 0.6},
            q=0.10,
        )
        self.assertEqual(result['significant'], ['a'])

    def test_promotion_floor_is_200(self):
        self.assertEqual(MIN_PROMOTION_BETS, 200)


class TestSubsetReport(unittest.TestCase):
    def test_small_sample_fails_closed(self):
        report = run_subset_research(synthetic_ledger(40))
        self.assertEqual(report['status'], 'INSUFFICIENT_DATA')
        self.assertFalse(report['decision']['production_enabled'])

    def test_report_keeps_everything_disabled(self):
        report = run_subset_research(synthetic_ledger(260))
        self.assertIn(report['status'], {'KEEP_DISABLED', 'CANDIDATE_FOR_SHADOW'})
        self.assertFalse(report['decision']['production_enabled'])
        self.assertTrue(report['decision']['paper_only'])
        self.assertEqual(
            set(report['multiple_testing']['pre_registered']),
            {rule['id'] for rule in SUBSET_RULES},
        )
        for rule_id, result in report['rules'].items():
            if result.get('n_picks', 0) > 0:
                self.assertIn('p_value_one_sided', result)
                self.assertFalse(result.get('sufficient_sample', False), rule_id)

    def test_empty_ledger_fails_closed(self):
        report = run_subset_research(pd.DataFrame())
        self.assertEqual(report['status'], 'INSUFFICIENT_DATA')
        self.assertFalse(report['decision']['production_enabled'])


if __name__ == '__main__':
    unittest.main()
