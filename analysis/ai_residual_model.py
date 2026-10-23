"""AI Residual Models for NBA Over/Under and Spread Prediction.

Enhances foundational quantitative basketball analytics (Pace/Efficiency, FiveThirtyEight Elo)
by using CatBoost to learn non-linear contextual adjustments (rest days, fatigue,
travel, style matchups, market discrepancies) from historical residual errors:
    OU Residual = Actual Total - Math Projected Total
    Spread Residual = Actual Margin - Math Projected Margin
"""

import os
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Feature column definitions for Over/Under
OU_FEATURE_COLUMNS = [
    'home_team',
    'away_team',
    'math_projected_total',
    'over_under_line',
    'home_rest_days',
    'away_rest_days',
    'home_b2b',
    'away_b2b',
    'rest_diff',
    'home_streak',
    'away_streak',
    'pace_diff',
    'rolling_combined_avg',
    'line_vs_math_diff',
]

# Backward compatibility alias
FEATURE_COLUMNS = OU_FEATURE_COLUMNS
CATEGORICAL_FEATURES = ['home_team', 'away_team']

# Feature column definitions for Spread / ATS
SPREAD_FEATURE_COLUMNS = [
    'home_team',
    'away_team',
    'math_projected_margin',
    'spread_line',
    'market_expected_margin',
    'market_discrepancy',
    'home_rest_days',
    'away_rest_days',
    'rest_diff',
    'home_b2b',
    'away_b2b',
    'net_rating_diff',
    'home_streak',
    'away_streak',
    'home_pt_diff_avg',
    'away_pt_diff_avg',
    'pt_diff_gap',
    'win_pct_diff',
    'pace_diff',
]


def parse_streak_val(streak_str: str) -> float:
    """Converts streak strings like 'W3' -> +3.0, 'L2' -> -2.0."""
    if not streak_str or streak_str == 'None':
        return 0.0
    s_type = streak_str[0].upper()
    try:
        val = float(streak_str[1:])
        return val if s_type == 'W' else -val
    except (ValueError, IndexError):
        return 0.0


class OverUnderResidualModel:
    """CatBoost-based residual predictor for NBA Over/Under game totals."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'data', 'models', 'catboost_ou_residual.cbm'
        )
        self.model = None
        self.is_fitted = False
        self._load_if_exists()

    def _load_if_exists(self):
        """Attempts to load a pre-trained CatBoost model from disk if available."""
        if os.path.exists(self.model_path) and os.path.getsize(self.model_path) > 0:
            try:
                from catboost import CatBoostRegressor
                self.model = CatBoostRegressor()
                self.model.load_model(self.model_path)
                self.is_fitted = True
                logger.info(f"Loaded existing CatBoost OU residual model from {self.model_path}")
            except Exception as e:
                logger.warning(f"Could not load CatBoost OU model from {self.model_path}: {e}")
                self.model = None
                self.is_fitted = False

    @staticmethod
    def _parse_streak_val(streak_str: str) -> float:
        return parse_streak_val(streak_str)

    def extract_game_features(
        self,
        home_team: str,
        away_team: str,
        fe: Any,
        analyzer: Any,
        game_date: Optional[str] = None,
        over_under_line: Optional[float] = None
    ) -> Dict[str, Any]:
        """Extract all contextual and quantitative features for a single matchup."""
        from config import normalize_team_name
        h_team = normalize_team_name(home_team) or home_team
        a_team = normalize_team_name(away_team) or away_team

        math_analysis = analyzer.analyze_over_under(
            home_team=h_team,
            away_team=a_team,
            over_under_line=over_under_line,
            game_date=game_date
        )
        math_proj = math_analysis.get('raw_projected_total') or math_analysis.get('projected_total', 220.0)

        h_form = fe.get_team_recent_form(h_team, n_games=5, before_date=game_date)
        a_form = fe.get_team_recent_form(a_team, n_games=5, before_date=game_date)

        g_date_str = str(game_date) if game_date else pd.Timestamp.now().strftime('%Y-%m-%d')
        h_rest = fe.get_rest_days(h_team, game_date=g_date_str)
        a_rest = fe.get_rest_days(a_team, game_date=g_date_str)

        try:
            h_b2b = 1 if float(h_rest) <= 1.0 else 0
        except (ValueError, TypeError):
            h_b2b = 0
        try:
            a_b2b = 1 if float(a_rest) <= 1.0 else 0
        except (ValueError, TypeError):
            a_b2b = 0

        h_pace = fe._get_team_adv_stat(h_team, 'pace', game_date) if hasattr(fe, '_get_team_adv_stat') else None
        a_pace = fe._get_team_adv_stat(a_team, 'pace', game_date) if hasattr(fe, '_get_team_adv_stat') else None
        h_pace = float(h_pace) if h_pace is not None else 99.5
        a_pace = float(a_pace) if a_pace is not None else 99.5

        h_scored = h_form.get('avg_points_scored', 114.0)
        a_scored = a_form.get('avg_points_scored', 114.0)
        comb_avg = h_scored + a_scored

        ou_line = float(over_under_line) if (over_under_line is not None and pd.notna(over_under_line)) else float(comb_avg)
        line_diff = math_proj - ou_line

        return {
            'home_team': h_team,
            'away_team': a_team,
            'math_projected_total': float(math_proj),
            'over_under_line': float(ou_line),
            'home_rest_days': float(min(h_rest, 7)),
            'away_rest_days': float(min(a_rest, 7)),
            'home_b2b': int(h_b2b),
            'away_b2b': int(a_b2b),
            'rest_diff': float(h_rest - a_rest),
            'home_streak': parse_streak_val(h_form.get('current_streak', 'None')),
            'away_streak': parse_streak_val(a_form.get('current_streak', 'None')),
            'pace_diff': float(h_pace - a_pace),
            'rolling_combined_avg': float(comb_avg),
            'line_vs_math_diff': float(line_diff),
        }

    def predict_residual(self, features: Dict[str, Any]) -> float:
        """Predicts the residual adjustment (in points) for a matchup."""
        if not self.is_fitted or self.model is None:
            return 0.0

        try:
            df = pd.DataFrame([features])[OU_FEATURE_COLUMNS]
            pred = self.model.predict(df)[0]
            return float(np.clip(pred, -15.0, 15.0))
        except Exception as e:
            logger.warning(f"Error predicting OU residual with CatBoost: {e}")
            return 0.0

    def get_adjusted_projection(
        self,
        home_team: str,
        away_team: str,
        fe: Any,
        analyzer: Any,
        game_date: Optional[str] = None,
        over_under_line: Optional[float] = None
    ) -> Dict[str, Any]:
        """Returns base math projection, AI residual, and final combined total."""
        features = self.extract_game_features(
            home_team, away_team, fe, analyzer, game_date, over_under_line
        )
        base_math = features['math_projected_total']
        ai_residual = self.predict_residual(features)
        adjusted_total = round(base_math + ai_residual, 2)

        return {
            'base_math_total': base_math,
            'ai_residual': round(ai_residual, 2),
            'adjusted_total': adjusted_total,
            'features': features
        }

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        iterations: int = 350,
        learning_rate: float = 0.03,
        depth: int = 4
    ) -> Dict[str, Any]:
        """Trains the CatBoost OU residual regressor."""
        from catboost import CatBoostRegressor

        X_train = train_df[OU_FEATURE_COLUMNS]
        y_train = train_df['target_residual']

        eval_set = None
        if val_df is not None and not val_df.empty:
            eval_set = (val_df[OU_FEATURE_COLUMNS], val_df['target_residual'])

        self.model = CatBoostRegressor(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            loss_function='MAE',
            eval_metric='MAE',
            cat_features=CATEGORICAL_FEATURES,
            random_seed=42,
            verbose=False
        )

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            early_stopping_rounds=40 if eval_set else None,
            verbose=False
        )
        self.is_fitted = True

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model.save_model(self.model_path)
        logger.info(f"Successfully trained and saved CatBoost OU model to {self.model_path}")

        preds = self.model.predict(X_train)
        base_mae = float(np.mean(np.abs(y_train)))
        model_mae = float(np.mean(np.abs(y_train - preds)))

        return {
            'train_samples': len(X_train),
            'base_math_mae': round(base_mae, 3),
            'ai_residual_mae': round(model_mae, 3),
            'mae_improvement': round(base_mae - model_mae, 3),
            'pct_improvement': round(((base_mae - model_mae) / base_mae) * 100, 2) if base_mae > 0 else 0.0
        }


class SpreadResidualModel:
    """CatBoost-based residual predictor for NBA Spread (Margin of Victory / ATS)."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'data', 'models', 'catboost_spread_residual.cbm'
        )
        self.model = None
        self.is_fitted = False
        self._load_if_exists()

    def _load_if_exists(self):
        """Attempts to load a pre-trained CatBoost model from disk if available."""
        if os.path.exists(self.model_path) and os.path.getsize(self.model_path) > 0:
            try:
                from catboost import CatBoostRegressor
                self.model = CatBoostRegressor()
                self.model.load_model(self.model_path)
                self.is_fitted = True
                logger.info(f"Loaded existing CatBoost Spread residual model from {self.model_path}")
            except Exception as e:
                logger.warning(f"Could not load CatBoost Spread model from {self.model_path}: {e}")
                self.model = None
                self.is_fitted = False

    def extract_spread_features(
        self,
        home_team: str,
        away_team: str,
        fe: Any,
        analyzer: Any,
        game_date: Optional[str] = None,
        spread_line: Optional[float] = None
    ) -> Dict[str, Any]:
        """Extract all contextual and quantitative features for Spread / Margin prediction.
        
        Strict walk-forward guarantee: all features calculated strictly BEFORE game_date.
        """
        from config import normalize_team_name
        h_team = normalize_team_name(home_team) or home_team
        a_team = normalize_team_name(away_team) or away_team

        # 1. Base math projection from TrendAnalyzer
        math_analysis = analyzer.analyze_spread(
            home_team=h_team,
            away_team=a_team,
            spread_line=spread_line,
            game_date=game_date
        )
        math_proj = math_analysis.get('raw_projected_margin')
        if math_proj is None:
            math_proj = math_analysis.get('projected_margin', 0.0)

        # 2. Team recent forms strictly before game_date
        h_form = fe.get_team_recent_form(h_team, n_games=5, before_date=game_date)
        a_form = fe.get_team_recent_form(a_team, n_games=5, before_date=game_date)

        g_date_str = str(game_date) if game_date else pd.Timestamp.now().strftime('%Y-%m-%d')
        h_rest = fe.get_rest_days(h_team, game_date=g_date_str)
        a_rest = fe.get_rest_days(a_team, game_date=g_date_str)

        try:
            h_b2b = 1 if float(h_rest) <= 1.0 else 0
        except (ValueError, TypeError):
            h_b2b = 0
        try:
            a_b2b = 1 if float(a_rest) <= 1.0 else 0
        except (ValueError, TypeError):
            a_b2b = 0

        # Advanced stats (Net Rating & Pace)
        h_net = fe._get_team_adv_stat(h_team, 'net_rating', game_date) if hasattr(fe, '_get_team_adv_stat') else None
        a_net = fe._get_team_adv_stat(a_team, 'net_rating', game_date) if hasattr(fe, '_get_team_adv_stat') else None
        h_net = float(h_net) if h_net is not None else 0.0
        a_net = float(a_net) if a_net is not None else 0.0
        net_diff = h_net - a_net

        h_pace = fe._get_team_adv_stat(h_team, 'pace', game_date) if hasattr(fe, '_get_team_adv_stat') else None
        a_pace = fe._get_team_adv_stat(a_team, 'pace', game_date) if hasattr(fe, '_get_team_adv_stat') else None
        h_pace = float(h_pace) if h_pace is not None else 99.5
        a_pace = float(a_pace) if a_pace is not None else 99.5

        h_pt_diff = h_form.get('avg_point_diff', 0.0)
        a_pt_diff = a_form.get('avg_point_diff', 0.0)
        pt_diff_gap = h_pt_diff - a_pt_diff

        h_win_pct = h_form.get('win_pct', 0.5)
        a_win_pct = a_form.get('win_pct', 0.5)

        s_line = float(spread_line) if (spread_line is not None and pd.notna(spread_line)) else 0.0
        market_margin = -s_line
        market_disc = float(math_proj) - market_margin

        return {
            'home_team': h_team,
            'away_team': a_team,
            'math_projected_margin': float(math_proj),
            'spread_line': float(s_line),
            'market_expected_margin': float(market_margin),
            'market_discrepancy': float(market_disc),
            'home_rest_days': float(min(h_rest, 7)),
            'away_rest_days': float(min(a_rest, 7)),
            'rest_diff': float(h_rest - a_rest),
            'home_b2b': int(h_b2b),
            'away_b2b': int(a_b2b),
            'net_rating_diff': float(net_diff),
            'home_streak': parse_streak_val(h_form.get('current_streak', 'None')),
            'away_streak': parse_streak_val(a_form.get('current_streak', 'None')),
            'home_pt_diff_avg': float(h_pt_diff),
            'away_pt_diff_avg': float(a_pt_diff),
            'pt_diff_gap': float(pt_diff_gap),
            'win_pct_diff': float(h_win_pct - a_win_pct),
            'pace_diff': float(h_pace - a_pace),
        }

    def predict_residual(self, features: Dict[str, Any]) -> float:
        """Predicts the residual adjustment to Home Margin (in points)."""
        if not self.is_fitted or self.model is None:
            return 0.0

        try:
            df = pd.DataFrame([features])[SPREAD_FEATURE_COLUMNS]
            pred = self.model.predict(df)[0]
            # Safety clamp: spread residual should be within [-12.0, +12.0] points
            return float(np.clip(pred, -12.0, 12.0))
        except Exception as e:
            logger.warning(f"Error predicting Spread residual with CatBoost: {e}")
            return 0.0

    def get_adjusted_spread_projection(
        self,
        home_team: str,
        away_team: str,
        fe: Any,
        analyzer: Any,
        game_date: Optional[str] = None,
        spread_line: Optional[float] = None
    ) -> Dict[str, Any]:
        """Returns base math margin, AI residual, and final combined projected margin."""
        features = self.extract_spread_features(
            home_team, away_team, fe, analyzer, game_date, spread_line
        )
        base_math = features['math_projected_margin']
        ai_residual = self.predict_residual(features)
        adjusted_margin = round(base_math + ai_residual, 2)

        s_line = features['spread_line']
        cover_edge = round(adjusted_margin + s_line, 2)

        return {
            'base_math_margin': base_math,
            'ai_residual': round(ai_residual, 2),
            'adjusted_margin': adjusted_margin,
            'cover_edge': cover_edge,
            'features': features
        }

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        iterations: int = 350,
        learning_rate: float = 0.03,
        depth: int = 4
    ) -> Dict[str, Any]:
        """Trains the CatBoost Spread residual regressor."""
        from catboost import CatBoostRegressor

        X_train = train_df[SPREAD_FEATURE_COLUMNS]
        y_train = train_df['target_residual']

        eval_set = None
        if val_df is not None and not val_df.empty:
            eval_set = (val_df[SPREAD_FEATURE_COLUMNS], val_df['target_residual'])

        self.model = CatBoostRegressor(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            loss_function='MAE',
            eval_metric='MAE',
            cat_features=CATEGORICAL_FEATURES,
            random_seed=42,
            verbose=False
        )

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            early_stopping_rounds=40 if eval_set else None,
            verbose=False
        )
        self.is_fitted = True

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model.save_model(self.model_path)
        logger.info(f"Successfully trained and saved CatBoost Spread model to {self.model_path}")

        preds = self.model.predict(X_train)
        base_mae = float(np.mean(np.abs(y_train)))
        model_mae = float(np.mean(np.abs(y_train - preds)))

        return {
            'train_samples': len(X_train),
            'base_math_mae': round(base_mae, 3),
            'ai_residual_mae': round(model_mae, 3),
            'mae_improvement': round(base_mae - model_mae, 3),
            'pct_improvement': round(((base_mae - model_mae) / base_mae) * 100, 2) if base_mae > 0 else 0.0
        }
