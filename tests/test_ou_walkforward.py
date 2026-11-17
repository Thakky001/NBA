"""Tests for the market-anchored residual walk-forward evaluation."""

import json
import unittest

import numpy as np
import pandas as pd

from analysis.ai_residual_model import (
    MARKET_OU_FEATURE_COLUMNS,
    MarketAnchoredResidualModel,
    build_market_residual_features,
)
from analysis.ou_walkforward import (
    build_market_residual_frame,
    run_market_residual_walkforward,
)


def synthetic_ledger(n=160):
    rows = []
    for i in range(n):
        line = 220.0 + (i % 5)
        projection = line + ((i % 7) - 3)
        actual = line + float(np.random.default_rng(i).normal(0, 10))
        rows.append({
            'game_id': f'g{i:04d}',
            'date': (pd.Timestamp('2026-10-21') + pd.Timedelta(days=i)).strftime('%Y-%m-%d'),
            'home_team': 'Home',
            'away_team': 'Away',
            'total_line': line,
            'actual_total': actual,
            'projected_total': projection,
            'ou_recommendation': 'STRONG OVER' if projection > line else 'STRONG UNDER',
            'ou_edge': projection - line,
            'ou_win_prob': 0.54,
            'features_snapshot': json.dumps({
                'combined_avg_10': line - 1.0,
                'combined_avg_5': line + 1.0,
                'combined_avg_season': line - 2.0,
                'home_rest_days': 2,
                'away_rest_days': 1,
                'home_point_diff_10': 3.0,
                'away_point_diff_10': -2.0,
                'home_roster_rating': 100.0,
                'away_roster_rating': 97.0,
                'injury_net_total_impact': -1.0,
                'fatigue_flag': False,
                'matchup_pace_style': 'AVERAGE',
                'home_games_played': 20,
                'away_games_played': 21,
            }),
        })
    return pd.DataFrame(rows)


class TestFeatureBuilder(unittest.TestCase):
    def test_builds_expected_features(self):
        features = build_market_residual_features(
            total_line=225.0,
            projected_total=227.5,
            features={
                'combined_avg_10': 224.0,
                'combined_avg_5': 226.0,
                'combined_avg_season': 222.0,
                'home_rest_days': 3,
                'away_rest_days': 1,
                'home_point_diff_10': 4.0,
                'away_point_diff_10': -1.0,
                'home_roster_rating': 101.0,
                'away_roster_rating': 98.0,
                'injury_net_total_impact': -3.0,
                'matchup_pace_style': 'FAST_VS_FAST',
            },
            game_date='2026-11-17',
        )
        self.assertEqual(features['market_line'], 225.0)
        self.assertEqual(features['model_edge'], 2.5)
        self.assertEqual(features['line_minus_combined10'], 1.0)
        self.assertEqual(features['rest_diff'], 2.0)
        self.assertEqual(features['fast_vs_fast'], 1.0)
        self.assertGreater(features['days_into_season'], 0)

    def test_missing_snapshot_keeps_none(self):
        features = build_market_residual_features(225.0, 227.0, '{}')
        self.assertIsNone(features['line_minus_combined10'])
        self.assertIsNone(features['injury_impact'])

    def test_requires_line_and_projection(self):
        with self.assertRaises(ValueError):
            build_market_residual_features(None, 227.0, {})
        with self.assertRaises(ValueError):
            build_market_residual_features(225.0, None, {})
        with self.assertRaises(ValueError):
            build_market_residual_features(225.0, 227.0, 'not json')


class TestResidualModel(unittest.TestCase):
    def _frame(self, n=30):
        rows = []
        for i in range(n):
            row = {name: float((i % 5) - 2) for name in MARKET_OU_FEATURE_COLUMNS}
            row['market_line'] = 220.0 + (i % 3)
            row['target_residual'] = 2.0 if i % 2 == 0 else -2.0
            rows.append(row)
        return pd.DataFrame(rows)

    def test_fit_and_predict_shapes(self):
        frame = self._frame()
        model = MarketAnchoredResidualModel(alpha=10.0)
        summary = model.fit(frame)
        self.assertTrue(model.is_fitted)
        self.assertEqual(summary['train_samples'], len(frame))
        predictions = model.predict_residuals(frame)
        self.assertEqual(len(predictions), len(frame))
        self.assertTrue(np.all(np.abs(predictions) <= model.clip + 1e-9))
        coefficients = model.coefficients()
        self.assertEqual(set(coefficients), set(MARKET_OU_FEATURE_COLUMNS))

    def test_fit_does_not_mutate_input(self):
        frame = self._frame()
        original = frame.copy(deep=True)
        MarketAnchoredResidualModel().fit(frame)
        pd.testing.assert_frame_equal(frame, original)

    def test_missing_columns_raise(self):
        frame = self._frame().drop(columns=['rest_diff'])
        with self.assertRaises(ValueError):
            MarketAnchoredResidualModel().fit(frame)

    def test_missing_feature_values_are_imputed(self):
        frame = self._frame()
        model = MarketAnchoredResidualModel()
        model.fit(frame)
        row = {name: 0.0 for name in MARKET_OU_FEATURE_COLUMNS}
        row['market_line'] = 221.0
        row['rest_diff'] = None
        prediction = model.predict_residual(row)
        self.assertTrue(np.isfinite(prediction))


class TestWalkforwardReport(unittest.TestCase):
    def test_frame_skips_unusable_rows(self):
        ledger = synthetic_ledger(40)
        ledger.loc[0, 'total_line'] = None
        ledger.loc[1, 'actual_total'] = None
        ledger.loc[2, 'projected_total'] = None
        frame, diagnostics = build_market_residual_frame(ledger)
        self.assertEqual(diagnostics['rows_usable'], 37)
        self.assertEqual(diagnostics['skipped']['missing_market_line'], 1)
        self.assertEqual(diagnostics['skipped']['missing_actual_total'], 1)
        self.assertEqual(diagnostics['skipped']['missing_projection'], 1)

    def test_invalid_snapshot_is_skipped(self):
        ledger = synthetic_ledger(20)
        ledger.loc[0, 'features_snapshot'] = 'not json'
        frame, diagnostics = build_market_residual_frame(ledger)
        self.assertEqual(diagnostics['skipped']['invalid_feature_snapshot'], 1)
        self.assertEqual(len(frame), 19)

    def test_small_sample_fails_closed(self):
        report = run_market_residual_walkforward(synthetic_ledger(50))
        self.assertEqual(report['status'], 'INSUFFICIENT_DATA')
        self.assertFalse(report['decision']['go_live_ready'])

    def test_report_runs_without_claiming_an_edge(self):
        report = run_market_residual_walkforward(synthetic_ledger(160))
        self.assertIn(
            report['status'],
            {'INSUFFICIENT_DATA', 'NO_MARKET_EDGE', 'NO_BETTING_EDGE', 'CANDIDATE_FOR_SHADOW'},
        )
        self.assertFalse(report['decision']['go_live_ready'])
        self.assertTrue(report['decision']['paper_only'])
        self.assertGreater(report['walk_forward']['out_of_sample_rows'], 0)
        self.assertIn('market', report['variants'])
        self.assertIn('market_residual_model', report['variants'])
        self.assertIn('coefficients', report)

    def test_empty_ledger_fails_closed(self):
        report = run_market_residual_walkforward(pd.DataFrame())
        self.assertEqual(report['status'], 'INSUFFICIENT_DATA')
        self.assertFalse(report['decision']['go_live_ready'])


if __name__ == '__main__':
    unittest.main()
