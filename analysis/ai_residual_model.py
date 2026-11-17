"""AI Residual Models for NBA Over/Under and Spread Prediction.

Enhances foundational quantitative basketball analytics (Pace/Efficiency, FiveThirtyEight Elo)
by using CatBoost to learn non-linear contextual adjustments (rest days, fatigue,
travel, style matchups, market discrepancies) from historical residual errors:
    OU Residual = Actual Total - Math Projected Total
    Spread Residual = Actual Margin - Math Projected Margin
"""

import os
import json
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
        over_under_line: Optional[float] = None,
        market_weight: Optional[float] = None
    ) -> Dict[str, Any]:
        """Extract all contextual and quantitative features for a single matchup."""
        from config import normalize_team_name
        h_team = normalize_team_name(home_team) or home_team
        a_team = normalize_team_name(away_team) or away_team

        math_analysis = analyzer.analyze_over_under(
            home_team=h_team,
            away_team=a_team,
            over_under_line=over_under_line,
            game_date=game_date,
            market_weight=market_weight
        )
        # Use market-anchored projected_total to ensure AI residual trains and adjusts on anchored baseline
        math_proj = math_analysis.get('projected_total') or math_analysis.get('raw_projected_total', 220.0)

        h_form = fe.get_team_recent_form(h_team, n_games=5, before_date=game_date)
        a_form = fe.get_team_recent_form(a_team, n_games=5, before_date=game_date)

        g_date_str = str(game_date) if game_date else pd.Timestamp.now().strftime('%Y-%m-%d')
        h_rest_raw = fe.get_rest_days(h_team, game_date=g_date_str)
        a_rest_raw = fe.get_rest_days(a_team, game_date=g_date_str)

        try:
            h_rest = float(h_rest_raw)
        except (ValueError, TypeError):
            h_rest = 2.0
        try:
            a_rest = float(a_rest_raw)
        except (ValueError, TypeError):
            a_rest = 2.0

        h_b2b = 1 if h_rest <= 1.0 else 0
        a_b2b = 1 if a_rest <= 1.0 else 0

        h_pace = fe._get_team_adv_stat(h_team, 'pace', game_date) if hasattr(fe, '_get_team_adv_stat') else None
        a_pace = fe._get_team_adv_stat(a_team, 'pace', game_date) if hasattr(fe, '_get_team_adv_stat') else None
        h_pace = float(h_pace) if h_pace is not None else 99.5
        a_pace = float(a_pace) if a_pace is not None else 99.5

        h_scored = h_form.get('avg_points_scored', 114.0)
        a_scored = a_form.get('avg_points_scored', 114.0)
        comb_avg = h_scored + a_scored

        ou_line = float(over_under_line) if (over_under_line is not None and pd.notna(over_under_line)) else float(comb_avg)
        line_diff = math_proj - ou_line

        # Extract sample size for early-season damping
        min_games = math_analysis.get('min_games_played')
        if min_games is None:
            features_dict = math_analysis.get('features', {})
            hg = features_dict.get('home_games_played')
            ag = features_dict.get('away_games_played')
            if hg is not None and ag is not None:
                min_games = min(hg, ag)
            else:
                min_games = 20

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
            'min_games_played': float(min_games),
        }

    def predict_residual(self, features: Dict[str, Any]) -> float:
        """Predicts the residual adjustment (in points) for a matchup."""
        if not self.is_fitted or self.model is None:
            return 0.0

        try:
            df = pd.DataFrame([features])[OU_FEATURE_COLUMNS]
            pred = float(self.model.predict(df)[0])
            min_games = float(features.get('min_games_played', 20))
            if min_games < 15:
                # Dynamic Sample-Size Damping
                damping = min(1.0, max(0.25, min_games / 20.0))
                # Early-Season Clamping +-3.5 pts
                return float(np.clip(pred * damping, -3.5, 3.5))
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
        over_under_line: Optional[float] = None,
        market_weight: Optional[float] = None
    ) -> Dict[str, Any]:
        """Returns base math projection, AI residual, and final combined total."""
        features = self.extract_game_features(
            home_team, away_team, fe, analyzer, game_date, over_under_line, market_weight=market_weight
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
        spread_line: Optional[float] = None,
        market_weight: Optional[float] = None
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
            game_date=game_date,
            market_weight=market_weight
        )
        # Use market-anchored projected_margin to ensure AI residual trains and adjusts on anchored baseline
        math_proj = math_analysis.get('projected_margin')
        if math_proj is None:
            math_proj = math_analysis.get('raw_projected_margin', 0.0)

        # 2. Team recent forms strictly before game_date
        h_form = fe.get_team_recent_form(h_team, n_games=5, before_date=game_date)
        a_form = fe.get_team_recent_form(a_team, n_games=5, before_date=game_date)

        g_date_str = str(game_date) if game_date else pd.Timestamp.now().strftime('%Y-%m-%d')
        h_rest_raw = fe.get_rest_days(h_team, game_date=g_date_str)
        a_rest_raw = fe.get_rest_days(a_team, game_date=g_date_str)

        try:
            h_rest = float(h_rest_raw)
        except (ValueError, TypeError):
            h_rest = 2.0
        try:
            a_rest = float(a_rest_raw)
        except (ValueError, TypeError):
            a_rest = 2.0

        h_b2b = 1 if h_rest <= 1.0 else 0
        a_b2b = 1 if a_rest <= 1.0 else 0

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

        # Extract sample size for early-season damping
        min_games = math_analysis.get('min_games_played')
        if min_games is None:
            features_dict = math_analysis.get('features', {})
            hg = features_dict.get('home_games_played')
            ag = features_dict.get('away_games_played')
            if hg is not None and ag is not None:
                min_games = min(hg, ag)
            else:
                min_games = 20

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
            'min_games_played': float(min_games),
        }

    def predict_residual(self, features: Dict[str, Any]) -> float:
        """Predicts the residual adjustment to Home Margin (in points)."""
        if not self.is_fitted or self.model is None:
            return 0.0

        try:
            df = pd.DataFrame([features])[SPREAD_FEATURE_COLUMNS]
            pred = float(self.model.predict(df)[0])
            min_games = float(features.get('min_games_played', 20))
            if min_games < 15:
                # Dynamic Sample-Size Damping
                damping = min(1.0, max(0.25, min_games / 20.0))
                # Early-Season Clamping +-3.0 pts
                return float(np.clip(pred * damping, -3.0, 3.0))
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
        spread_line: Optional[float] = None,
        market_weight: Optional[float] = None
    ) -> Dict[str, Any]:
        """Returns base math margin, AI residual, and final combined projected margin."""
        features = self.extract_spread_features(
            home_team, away_team, fe, analyzer, game_date, spread_line, market_weight=market_weight
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


# ---------------------------------------------------------------------------
# Market-anchored O/U residual model
# ---------------------------------------------------------------------------
# The legacy CatBoost residual models target ``actual - math_projection``. That
# can look skilful while still being worse than simply using the market line.
# This model instead uses the market total as the base forecast and learns only
# the residual ``actual_total - market_total``. A ridge penalty keeps the tiny
# historical samples from memorising noise, and coefficients are fitted inside
# walk-forward folds only.

MARKET_OU_FEATURE_COLUMNS = [
    'market_line',
    'model_edge',
    'line_minus_combined10',
    'line_minus_combined5',
    'line_minus_season',
    'avg5_minus_avg10',
    'avg10_minus_season',
    'rest_diff',
    'fatigue',
    'injury_impact',
    'point_diff_gap',
    'roster_gap',
    'days_into_season',
    'fast_vs_fast',
    'slow_vs_slow',
]


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def build_market_residual_features(
    total_line: Any,
    projected_total: Any,
    features: Any,
    game_date: Any = None,
) -> Dict[str, Optional[float]]:
    """Build leakage-free features for ``actual_total - market_total``.

    ``features`` is the pre-game feature snapshot stored in the ledger. Missing
    inputs stay ``None`` and are median-imputed by the model, so a damaged
    snapshot cannot silently become a zero.
    """
    line = _safe_float(total_line)
    projection = _safe_float(projected_total)
    if line is None or line <= 0 or projection is None:
        raise ValueError("total_line and projected_total are required")

    if isinstance(features, str):
        try:
            features = json.loads(features)
        except Exception as exc:
            raise ValueError(f"features snapshot is not valid JSON: {exc}") from exc
    if not isinstance(features, dict):
        features = {}

    def _feature(key: str) -> Optional[float]:
        return _safe_float(features.get(key))

    combined10 = _feature('combined_avg_10')
    combined5 = _feature('combined_avg_5')
    combined_season = _feature('combined_avg_season')
    home_rest = _feature('home_rest_days')
    away_rest = _feature('away_rest_days')
    home_diff = _feature('home_point_diff_10')
    away_diff = _feature('away_point_diff_10')
    home_roster = _feature('home_roster_rating')
    away_roster = _feature('away_roster_rating')
    style = str(features.get('matchup_pace_style', '') or '').upper()

    days_into_season = None
    if game_date is not None:
        try:
            date = pd.to_datetime(game_date)
            season_year = date.year if date.month >= 10 else date.year - 1
            season_start = pd.Timestamp(year=season_year, month=10, day=1)
            days_into_season = float(max(0, (date.normalize() - season_start).days))
        except (TypeError, ValueError):
            days_into_season = None

    return {
        'market_line': line,
        'model_edge': projection - line,
        'line_minus_combined10': line - combined10 if combined10 is not None else None,
        'line_minus_combined5': line - combined5 if combined5 is not None else None,
        'line_minus_season': line - combined_season if combined_season is not None else None,
        'avg5_minus_avg10': (
            combined5 - combined10
            if combined5 is not None and combined10 is not None else None
        ),
        'avg10_minus_season': (
            combined10 - combined_season
            if combined10 is not None and combined_season is not None else None
        ),
        'rest_diff': (
            max(-10.0, min(10.0, home_rest - away_rest))
            if home_rest is not None and away_rest is not None else None
        ),
        'fatigue': 1.0 if features.get('fatigue_flag') else 0.0,
        'injury_impact': _feature('injury_net_total_impact'),
        'point_diff_gap': (
            home_diff - away_diff
            if home_diff is not None and away_diff is not None else None
        ),
        'roster_gap': (
            home_roster - away_roster
            if home_roster is not None and away_roster is not None else None
        ),
        'days_into_season': days_into_season,
        'fast_vs_fast': 1.0 if style == 'FAST_VS_FAST' else 0.0,
        'slow_vs_slow': 1.0 if style == 'SLOW_VS_SLOW' else 0.0,
    }


class MarketAnchoredResidualModel:
    """Ridge regression on the market residual, with train-only preprocessing.

    Design choices are deliberately conservative:

    - The market total is the base forecast; the model only predicts an
      adjustment.
    - Features are median-imputed and standardised using training rows only.
    - A ridge penalty (default 10.0) is strong relative to the sample size.
    - Predictions are clipped to a plausible total adjustment.
    - ``fit`` never mutates the input frame and never sees rows passed to
      ``predict`` during walk-forward evaluation.
    """

    def __init__(self, alpha: float = 10.0, clip: float = 12.0):
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        if clip <= 0:
            raise ValueError("clip must be positive")
        self.alpha = float(alpha)
        self.clip = float(clip)
        self.feature_columns = list(MARKET_OU_FEATURE_COLUMNS)
        self.coef_ = np.zeros(len(self.feature_columns), dtype=float)
        self.intercept_ = 0.0
        self.feature_medians_ = np.zeros(len(self.feature_columns), dtype=float)
        self.feature_mean_ = np.zeros(len(self.feature_columns), dtype=float)
        self.feature_std_ = np.ones(len(self.feature_columns), dtype=float)
        self.is_fitted = False

    def _matrix(self, df: pd.DataFrame) -> np.ndarray:
        if isinstance(df, dict):
            df = pd.DataFrame([df])
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        missing = [c for c in self.feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"missing market-residual features: {missing}")
        matrix = df[self.feature_columns].apply(pd.to_numeric, errors='coerce')
        matrix = matrix.replace([np.inf, -np.inf], np.nan)
        return matrix.to_numpy(dtype=float)

    def fit(self, df: pd.DataFrame, target_col: str = 'target_residual') -> Dict[str, Any]:
        if target_col not in df.columns:
            raise ValueError(f"target column '{target_col}' is required")
        if len(df) < 2:
            raise ValueError("at least two training rows are required")

        X_raw = self._matrix(df)
        y = pd.to_numeric(df[target_col], errors='coerce').to_numpy(dtype=float)
        valid = np.isfinite(y)
        if valid.sum() < 2:
            raise ValueError("at least two finite target values are required")
        X_raw = X_raw[valid]
        y = y[valid]

        self.feature_medians_ = np.nanmedian(X_raw, axis=0)
        self.feature_medians_ = np.where(
            np.isfinite(self.feature_medians_), self.feature_medians_, 0.0
        )
        X = np.where(np.isfinite(X_raw), X_raw, self.feature_medians_)
        self.feature_mean_ = X.mean(axis=0)
        self.feature_std_ = X.std(axis=0)
        self.feature_std_ = np.where(self.feature_std_ > 1e-9, self.feature_std_, 1.0)
        Z = (X - self.feature_mean_) / self.feature_std_

        self.intercept_ = float(y.mean())
        centered = y - self.intercept_
        penalty = self.alpha * np.eye(Z.shape[1])
        self.coef_ = np.linalg.solve(Z.T @ Z + penalty, Z.T @ centered)

        self.is_fitted = True
        fitted = self.intercept_ + Z @ self.coef_
        return {
            'train_samples': int(len(y)),
            'alpha': self.alpha,
            'train_mae': round(float(np.mean(np.abs(y - fitted))), 4),
            'market_mae': round(float(np.mean(np.abs(y))), 4),
            'mae_improvement': round(
                float(np.mean(np.abs(y)) - np.mean(np.abs(y - fitted))), 4
            ),
        }

    def predict_residuals(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("model is not fitted")
        X_raw = self._matrix(df)
        X = np.where(np.isfinite(X_raw), X_raw, self.feature_medians_)
        Z = (X - self.feature_mean_) / self.feature_std_
        predicted = self.intercept_ + Z @ self.coef_
        return np.clip(predicted, -self.clip, self.clip)

    def predict_residual(self, features: Dict[str, Any]) -> float:
        return float(self.predict_residuals(pd.DataFrame([features]))[0])

    def coefficients(self) -> Dict[str, float]:
        return {
            name: round(float(value), 5)
            for name, value in zip(self.feature_columns, self.coef_)
        }
