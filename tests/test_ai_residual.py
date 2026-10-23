"""Unit tests for the AI CatBoost Residual Models for Over/Under and Spread."""

import unittest
import os
import tempfile
import pandas as pd
import numpy as np

from analysis.ai_residual_model import (
    OverUnderResidualModel,
    SpreadResidualModel,
    OU_FEATURE_COLUMNS,
    SPREAD_FEATURE_COLUMNS,
    CATEGORICAL_FEATURES,
    parse_streak_val,
)
from processing.feature_engineering import FeatureEngineer
from analysis.trend_analyzer import TrendAnalyzer


class TestAIResidualModel(unittest.TestCase):

    def setUp(self):
        # Create minimal synthetic game logs
        dates = pd.date_range('2026-01-01', periods=15, freq='D')
        self.mock_games = pd.DataFrame({
            'game_id': [f"g_{i}" for i in range(15)],
            'date': dates,
            'home_team': ['Denver Nuggets', 'Oklahoma City Thunder', 'Los Angeles Lakers'] * 5,
            'away_team': ['Boston Celtics', 'Golden State Warriors', 'Miami Heat'] * 5,
            'home_score': [110 + i for i in range(15)],
            'away_score': [105 + i for i in range(15)],
            'spread': [-4.5] * 15,
            'over_under': [215.5] * 15,
            'season_type': [2] * 15,
        })
        self.fe = FeatureEngineer(game_logs=self.mock_games)
        self.analyzer = TrendAnalyzer(self.fe)

    def test_streak_parsing(self):
        self.assertEqual(parse_streak_val('W3'), 3.0)
        self.assertEqual(parse_streak_val('L2'), -2.0)
        self.assertEqual(parse_streak_val('None'), 0.0)
        self.assertEqual(parse_streak_val(''), 0.0)

    # --- Over/Under Model Tests ---

    def test_ou_feature_extraction(self):
        model = OverUnderResidualModel(model_path="non_existent.cbm")
        feats = model.extract_game_features(
            home_team='Denver Nuggets',
            away_team='Boston Celtics',
            fe=self.fe,
            analyzer=self.analyzer,
            game_date='2026-01-10',
            over_under_line=220.5
        )

        for col in OU_FEATURE_COLUMNS:
            self.assertIn(col, feats, f"Missing OU feature column: {col}")

        self.assertEqual(feats['home_team'], 'Denver Nuggets')
        self.assertEqual(feats['away_team'], 'Boston Celtics')
        self.assertEqual(feats['over_under_line'], 220.5)
        self.assertIsInstance(feats['math_projected_total'], float)

    def test_ou_unfitted_model_fallback(self):
        model = OverUnderResidualModel(model_path="non_existent.cbm")
        self.assertFalse(model.is_fitted)

        feats = {col: 1.0 for col in OU_FEATURE_COLUMNS}
        feats['home_team'] = 'Denver Nuggets'
        feats['away_team'] = 'Boston Celtics'

        residual = model.predict_residual(feats)
        self.assertEqual(residual, 0.0)

        proj = model.get_adjusted_projection(
            'Denver Nuggets', 'Boston Celtics', self.fe, self.analyzer, over_under_line=220.0
        )
        self.assertEqual(proj['ai_residual'], 0.0)
        self.assertEqual(proj['base_math_total'], proj['adjusted_total'])

    def test_ou_train_save_and_predict(self):
        np.random.seed(42)
        n = 50
        df_train = pd.DataFrame({
            'home_team': np.random.choice(['Denver Nuggets', 'Boston Celtics'], size=n),
            'away_team': np.random.choice(['Los Angeles Lakers', 'Miami Heat'], size=n),
            'math_projected_total': np.random.uniform(210, 230, size=n),
            'over_under_line': np.random.uniform(210, 230, size=n),
            'home_rest_days': np.random.randint(1, 4, size=n).astype(float),
            'away_rest_days': np.random.randint(1, 4, size=n).astype(float),
            'home_b2b': np.random.choice([0, 1], size=n),
            'away_b2b': np.random.choice([0, 1], size=n),
            'rest_diff': np.random.uniform(-2, 2, size=n),
            'home_streak': np.random.uniform(-3, 3, size=n),
            'away_streak': np.random.uniform(-3, 3, size=n),
            'pace_diff': np.random.uniform(-3, 3, size=n),
            'rolling_combined_avg': np.random.uniform(210, 230, size=n),
            'line_vs_math_diff': np.random.uniform(-5, 5, size=n),
            'target_residual': np.random.uniform(-6, 6, size=n),
        })

        with tempfile.NamedTemporaryFile(suffix='.cbm', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            model = OverUnderResidualModel(model_path=tmp_path)
            res = model.train(df_train, iterations=20, depth=2)

            self.assertTrue(model.is_fitted)
            self.assertIn('mae_improvement', res)

            sample_feats = df_train.iloc[0].to_dict()
            pred = model.predict_residual(sample_feats)
            self.assertIsInstance(pred, float)

            reloaded = OverUnderResidualModel(model_path=tmp_path)
            self.assertTrue(reloaded.is_fitted)
            pred_reloaded = reloaded.predict_residual(sample_feats)
            self.assertAlmostEqual(pred, pred_reloaded, places=4)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # --- Spread Model Tests ---

    def test_spread_feature_extraction(self):
        model = SpreadResidualModel(model_path="non_existent_spread.cbm")
        feats = model.extract_spread_features(
            home_team='Denver Nuggets',
            away_team='Boston Celtics',
            fe=self.fe,
            analyzer=self.analyzer,
            game_date='2026-01-10',
            spread_line=-4.5
        )

        for col in SPREAD_FEATURE_COLUMNS:
            self.assertIn(col, feats, f"Missing Spread feature column: {col}")

        self.assertEqual(feats['home_team'], 'Denver Nuggets')
        self.assertEqual(feats['away_team'], 'Boston Celtics')
        self.assertEqual(feats['spread_line'], -4.5)
        self.assertEqual(feats['market_expected_margin'], 4.5)
        self.assertIsInstance(feats['math_projected_margin'], float)

    def test_spread_unfitted_model_fallback(self):
        model = SpreadResidualModel(model_path="non_existent_spread.cbm")
        self.assertFalse(model.is_fitted)

        feats = {col: 1.0 for col in SPREAD_FEATURE_COLUMNS}
        feats['home_team'] = 'Denver Nuggets'
        feats['away_team'] = 'Boston Celtics'

        residual = model.predict_residual(feats)
        self.assertEqual(residual, 0.0)

        proj = model.get_adjusted_spread_projection(
            'Denver Nuggets', 'Boston Celtics', self.fe, self.analyzer, spread_line=-5.0
        )
        self.assertEqual(proj['ai_residual'], 0.0)
        self.assertEqual(proj['base_math_margin'], proj['adjusted_margin'])
        self.assertEqual(proj['cover_edge'], round(proj['adjusted_margin'] - 5.0, 2))

    def test_spread_train_save_and_predict(self):
        np.random.seed(42)
        n = 50
        df_train = pd.DataFrame({
            'home_team': np.random.choice(['Denver Nuggets', 'Boston Celtics'], size=n),
            'away_team': np.random.choice(['Los Angeles Lakers', 'Miami Heat'], size=n),
            'math_projected_margin': np.random.uniform(-8, 8, size=n),
            'spread_line': np.random.uniform(-8, 8, size=n),
            'market_expected_margin': np.random.uniform(-8, 8, size=n),
            'market_discrepancy': np.random.uniform(-5, 5, size=n),
            'home_rest_days': np.random.randint(1, 4, size=n).astype(float),
            'away_rest_days': np.random.randint(1, 4, size=n).astype(float),
            'rest_diff': np.random.uniform(-2, 2, size=n),
            'home_b2b': np.random.choice([0, 1], size=n),
            'away_b2b': np.random.choice([0, 1], size=n),
            'net_rating_diff': np.random.uniform(-10, 10, size=n),
            'home_streak': np.random.uniform(-3, 3, size=n),
            'away_streak': np.random.uniform(-3, 3, size=n),
            'home_pt_diff_avg': np.random.uniform(-8, 8, size=n),
            'away_pt_diff_avg': np.random.uniform(-8, 8, size=n),
            'pt_diff_gap': np.random.uniform(-10, 10, size=n),
            'win_pct_diff': np.random.uniform(-0.5, 0.5, size=n),
            'pace_diff': np.random.uniform(-3, 3, size=n),
            'target_residual': np.random.uniform(-6, 6, size=n),
        })

        with tempfile.NamedTemporaryFile(suffix='.cbm', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            model = SpreadResidualModel(model_path=tmp_path)
            res = model.train(df_train, iterations=20, depth=2)

            self.assertTrue(model.is_fitted)
            self.assertIn('mae_improvement', res)

            sample_feats = df_train.iloc[0].to_dict()
            pred = model.predict_residual(sample_feats)
            self.assertIsInstance(pred, float)

            reloaded = SpreadResidualModel(model_path=tmp_path)
            self.assertTrue(reloaded.is_fitted)
            pred_reloaded = reloaded.predict_residual(sample_feats)
            self.assertAlmostEqual(pred, pred_reloaded, places=4)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == '__main__':
    unittest.main()
