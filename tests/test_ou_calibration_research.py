"""Tests for walk-forward calibration and selective-rule research."""

import json
import unittest

import pandas as pd

from analysis.ou_calibration_research import (
    SELECTIVE_RULES,
    _build_frame,
    _rule_mask,
    fit_temperature,
    run_calibration_research,
    temperature_scale,
)
from analysis.ou_subsets import benjamini_hochberg


def synthetic_ledger(n=150):
    rows = []
    for i in range(n):
        edge = float((i % 7) - 3)
        line = 220.0
        projection = line + edge
        actual = line + (3.0 if i % 3 == 0 else -3.0)
        if edge >= 2:
            rec = 'STRONG OVER'
            prob = 0.56
        elif edge <= -2:
            rec = 'STRONG UNDER'
            prob = 0.54
        elif edge > 0:
            rec = 'LEAN OVER'
            prob = 0.52
        else:
            rec = 'NEUTRAL'
            prob = 0.50
        rows.append({
            'game_id': f'g{i:03d}',
            'date': (pd.Timestamp('2026-10-21') + pd.Timedelta(days=i)).strftime('%Y-%m-%d'),
            'home_team': 'Home',
            'away_team': 'Away',
            'total_line': line,
            'actual_total': actual,
            'projected_total': projection,
            'ou_recommendation': rec,
            'ou_edge': edge,
            'ou_win_prob': prob,
            'features_snapshot': json.dumps({
                'home_games_played': 20,
                'away_games_played': 21,
                'matchup_pace_style': 'AVERAGE',
            }),
        })
    return pd.DataFrame(rows)


class TestTemperatureScaling(unittest.TestCase):
    def test_temperature_above_one_shrinks_toward_half(self):
        p = temperature_scale(0.9, 2.0)
        self.assertLess(p, 0.9)
        self.assertGreater(p, 0.5)

    def test_temperature_below_one_sharpens(self):
        p = temperature_scale(0.7, 0.5)
        self.assertGreater(p, 0.7)

    def test_fit_temperature_prefers_shrink_for_overconfident_probabilities(self):
        probabilities = [0.9] * 20 + [0.1] * 20
        outcomes = [1, 0] * 20
        fit = fit_temperature(probabilities, outcomes)
        self.assertIsNotNone(fit)
        self.assertGreater(fit['temperature'], 1.0)

    def test_fit_temperature_fails_closed_with_single_class(self):
        self.assertIsNone(fit_temperature([0.6] * 20, [1] * 20))
        self.assertIsNone(fit_temperature([0.6] * 5, [1, 0] * 2 + [1]))


class TestFrameAndRules(unittest.TestCase):
    def test_frame_builds_and_skips_bad_rows(self):
        ledger = synthetic_ledger(30)
        ledger.loc[0, 'total_line'] = None
        ledger.loc[1, 'actual_total'] = 220.0  # push
        ledger.loc[2, 'projected_total'] = None
        frame, diagnostics = _build_frame(ledger)
        self.assertEqual(diagnostics['rows_usable'], 27)
        self.assertEqual(diagnostics['skipped']['missing_line'], 1)
        self.assertEqual(diagnostics['skipped']['push'], 1)
        self.assertEqual(diagnostics['skipped']['missing_projection'], 1)
        self.assertIn('is_pick', frame.columns)

    def test_rule_masks_match_definitions(self):
        ledger = synthetic_ledger(30)
        frame, _ = _build_frame(ledger)
        self.assertTrue(_rule_mask(frame, 'strong_gp15').any())
        under = frame[_rule_mask(frame, 'strong_under')]
        self.assertTrue((under['side'] == 'UNDER').all())
        all_picks = _rule_mask(frame, 'all_picks')
        self.assertEqual(int(all_picks.sum()), int(frame['is_pick'].sum()))

    def test_probability_rule_uses_calibrated_column(self):
        ledger = synthetic_ledger(30)
        frame, _ = _build_frame(ledger)
        frame['calibrated_pick_prob'] = 0.60
        self.assertTrue(_rule_mask(frame, 'prob_ge_053').any())
        frame['calibrated_pick_prob'] = 0.51
        self.assertFalse(_rule_mask(frame, 'prob_ge_053').any())


class TestMultipleTesting(unittest.TestCase):
    def test_bh_adjusts_and_flags_significant(self):
        result = benjamini_hochberg(
            {'a': 0.001, 'b': 0.02, 'c': 0.5, 'd': None},
            q=0.10,
        )
        self.assertEqual(result['family_size'], 3)
        self.assertIn('a', result['significant'])
        self.assertNotIn('c', result['significant'])
        self.assertGreaterEqual(result['adjusted']['a'], 0.001)

    def test_empty_family_is_safe(self):
        result = benjamini_hochberg({}, q=0.10)
        self.assertEqual(result['family_size'], 0)
        self.assertEqual(result['adjusted'], {})


class TestCalibrationReport(unittest.TestCase):
    def test_pre_registered_rules_are_present(self):
        ids = {rule[0] for rule in SELECTIVE_RULES}
        self.assertIn('strong_gp15', ids)
        self.assertIn('edge_ge_2_prob_ge_053', ids)

    def test_small_sample_fails_closed(self):
        report = run_calibration_research(synthetic_ledger(50))
        self.assertEqual(report['status'], 'INSUFFICIENT_DATA')
        self.assertFalse(report['decision']['production_enabled'])
        self.assertFalse(report['decision']['enable_selective_gate'])

    def test_report_runs_and_never_enables_betting(self):
        report = run_calibration_research(synthetic_ledger(150))
        self.assertIn(report['status'], {'KEEP_DISABLED', 'CANDIDATE_FOR_SHADOW'})
        self.assertFalse(report['decision']['production_enabled'])
        self.assertFalse(report['decision']['enable_selective_gate'])
        self.assertGreater(report['walk_forward']['out_of_sample_rows'], 0)
        self.assertIn('calibration', report)
        self.assertIn('selective_rules', report)
        for rule_id, result in report['selective_rules'].items():
            self.assertEqual(result['rule'], rule_id)
            self.assertFalse(result.get('sufficient_sample', False))

    def test_no_picks_still_disables_gate(self):
        ledger = synthetic_ledger(150)
        ledger['ou_recommendation'] = 'NEUTRAL'
        report = run_calibration_research(ledger)
        self.assertEqual(report['status'], 'INSUFFICIENT_DATA')
        self.assertFalse(report['decision']['production_enabled'])


if __name__ == '__main__':
    unittest.main()
