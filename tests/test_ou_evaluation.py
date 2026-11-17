"""Tests for the O/U evaluation harness (Phase 0)."""

import json
import unittest

import pandas as pd

from analysis.ou_evaluation import evaluate_market_baseline, evaluate_ou_frame, evaluate_ou_slice
from analysis.prediction_tracker import calculate_ou_clv


def _row(
    result,
    rec='LEAN OVER',
    pick_prob=0.55,
    edge=2.5,
    total_line=220.0,
    actual_total=224.0,
    projected_total=223.0,
    gp=12,
    closing_line=None,
    clv=None,
):
    return {
        'game_id': '1',
        'date': '2026-11-17',
        'ou_recommendation': rec,
        'ou_pick_side': 'OVER' if 'OVER' in rec else 'UNDER',
        'ou_result': result,
        'ou_pick_prob': pick_prob,
        'ou_edge': edge,
        'total_line': total_line,
        'actual_total': actual_total,
        'projected_total': projected_total,
        'closing_line': closing_line,
        'clv': clv,
        'features_snapshot': json.dumps({
            'home_games_played': gp,
            'away_games_played': gp,
        }),
    }


class TestEvaluateOUSlice(unittest.TestCase):

    def test_empty_slice_has_no_profit_metrics(self):
        res = evaluate_ou_slice(pd.DataFrame(), min_sample=1)
        self.assertEqual(res['n_picks'], 0)
        self.assertEqual(res['units'], 0.0)
        self.assertIsNone(res['roi_pct'])
        self.assertFalse(res['sufficient_sample'])

    def test_basic_record_and_units(self):
        df = pd.DataFrame([
            _row('WIN', rec='LEAN OVER', actual_total=224.0),
            _row('LOSS', rec='LEAN OVER', actual_total=216.0),
            _row('WIN', rec='LEAN UNDER', actual_total=216.0),
            _row('PUSH', rec='LEAN UNDER', actual_total=220.0),
        ])
        res = evaluate_ou_slice(df)
        self.assertEqual(res['n_picks'], 4)
        self.assertEqual(res['wins'], 2)
        self.assertEqual(res['losses'], 1)
        self.assertEqual(res['pushes'], 1)
        self.assertAlmostEqual(res['win_rate'], 66.67, places=2)
        self.assertAlmostEqual(res['units'], 0.818, places=3)

    def test_market_and_model_mae(self):
        df = pd.DataFrame([
            _row('WIN', actual_total=224.0, projected_total=223.0),
            _row('LOSS', actual_total=216.0, projected_total=221.0),
        ])
        res = evaluate_ou_slice(df)
        self.assertAlmostEqual(res['model_mae'], 3.0, places=3)
        self.assertAlmostEqual(res['market_mae'], 4.0, places=3)
        self.assertAlmostEqual(res['mean_actual_minus_line'], 0.0, places=3)

    def test_point_clv_is_positive_when_over_beats_close(self):
        df = pd.DataFrame([
            _row('WIN', rec='LEAN OVER', total_line=220.0, closing_line=223.0),
            _row('WIN', rec='LEAN UNDER', total_line=230.0, closing_line=227.0),
        ])
        res = evaluate_ou_slice(df)
        self.assertAlmostEqual(res['mean_clv_points'], 3.0, places=3)
        self.assertEqual(res['clv_sample'], 2)

    def test_neutral_rows_are_excluded(self):
        df = pd.DataFrame([
            _row('SKIPPED', rec='NEUTRAL'),
            _row('WIN', rec='LEAN OVER'),
        ])
        res = evaluate_ou_slice(df)
        self.assertEqual(res['n_picks'], 1)


class TestFrameBreakdowns(unittest.TestCase):

    def test_side_tier_and_gp_breakdowns(self):
        df = pd.DataFrame([
            _row('WIN', rec='STRONG OVER', gp=20),
            _row('LOSS', rec='LEAN UNDER', gp=10),
            _row('WIN', rec='LEAN OVER', gp=3),
        ])
        report = evaluate_ou_frame(df)
        self.assertEqual(report['overall']['n_picks'], 3)
        self.assertIn('OVER', report['by_side'])
        self.assertIn('UNDER', report['by_side'])
        self.assertIn('STRONG', report['by_tier'])
        self.assertIn('LEAN', report['by_tier'])
        self.assertIn('0-5', report['by_gp_bucket'])
        self.assertEqual(report['market_baseline']['graded_games'], 3)


class TestCLVHelper(unittest.TestCase):

    def test_over_and_under_direction(self):
        self.assertAlmostEqual(calculate_ou_clv('LEAN OVER', 220.0, 223.0), 3.0)
        self.assertAlmostEqual(calculate_ou_clv('LEAN UNDER', 230.0, 227.0), 3.0)
        self.assertIsNone(calculate_ou_clv('NEUTRAL', 220.0, 223.0))


class TestMarketBaseline(unittest.TestCase):

    def test_bias_and_over_rate(self):
        df = pd.DataFrame([
            _row('SKIPPED', rec='NEUTRAL', total_line=220.0, actual_total=224.0),
            _row('SKIPPED', rec='NEUTRAL', total_line=220.0, actual_total=216.0),
            _row('SKIPPED', rec='NEUTRAL', total_line=220.0, actual_total=221.0),
        ])
        baseline = evaluate_market_baseline(df)
        self.assertEqual(baseline['graded_games'], 3)
        self.assertAlmostEqual(baseline['bias_actual_minus_line'], 0.333, places=3)
        self.assertAlmostEqual(baseline['over_rate'], 2 / 3, places=3)


if __name__ == '__main__':
    unittest.main()
