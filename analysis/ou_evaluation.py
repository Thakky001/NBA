"""Over/Under evaluation helpers for research and go-live gating.

This module is deliberately independent from the live alert path. It turns the
predictions ledger into the metrics the O/U research programme needs:

- realised win rate / ROI after vig
- model vs market MAE and bias
- edge-vs-result correlation
- probability calibration (Brier, log loss, ECE)
- closing-line value (CLV) when a closing line was captured
- breakdowns by side, tier, edge size and team sample maturity

Only graded O/U picks are used for pick metrics. Market baselines use every
graded game with a total, because the market line exists even when the model
stays neutral.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from analysis.backtesting import (
    bootstrap_mean_interval,
    calibration_bins,
    probability_metrics,
)


WIN_PAYOUT = 100.0 / 110.0  # standard -110


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _result_pnl(result: Any) -> float:
    value = str(result).upper()
    if value == 'WIN':
        return WIN_PAYOUT
    if value == 'LOSS':
        return -1.0
    return 0.0


def _pick_mask(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=bool)
    rec = df.get('ou_recommendation', pd.Series(index=df.index, dtype=object)).astype(str).str.upper()
    result = df.get('ou_result', pd.Series(index=df.index, dtype=object)).astype(str).str.upper()
    return result.isin(['WIN', 'LOSS', 'PUSH']) & rec.str.contains('OVER|UNDER', regex=True)


def _graded_mask(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=bool)
    result = df.get('ou_result', pd.Series(index=df.index, dtype=object)).astype(str).str.upper()
    return result.isin(['WIN', 'LOSS', 'PUSH'])


def _market_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with both a final total and a market line, including neutral games."""
    if df is None or df.empty:
        return pd.DataFrame()
    actual = pd.to_numeric(df.get('actual_total'), errors='coerce')
    line = pd.to_numeric(df.get('total_line'), errors='coerce')
    return df[actual.notna() & line.notna()].copy()


def _side(row: pd.Series) -> Optional[str]:
    rec = str(row.get('ou_recommendation', '')).upper()
    if 'OVER' in rec:
        return 'OVER'
    if 'UNDER' in rec:
        return 'UNDER'
    side = str(row.get('ou_pick_side', '')).upper()
    return side if side in ('OVER', 'UNDER') else None


def _tier(row: pd.Series) -> str:
    rec = str(row.get('ou_recommendation', '')).upper()
    return 'STRONG' if 'STRONG' in rec else ('LEAN' if 'LEAN' in rec else 'NEUTRAL')


def _games_played(row: pd.Series) -> Optional[int]:
    """Read min(home, away) games played from the stored feature snapshot."""
    raw = row.get('features_snapshot')
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    try:
        features = json.loads(raw) if isinstance(raw, str) else raw
        h = features.get('home_games_played')
        a = features.get('away_games_played')
        if h is None or a is None:
            return None
        return int(min(float(h), float(a)))
    except Exception:
        return None


def ou_side(row: pd.Series) -> Optional[str]:
    """Public helper: OVER/UNDER side of a ledger row."""
    return _side(row)


def ou_tier(row: pd.Series) -> str:
    """Public helper: STRONG/LEAN/NEUTRAL tier of a ledger row."""
    return _tier(row)


def ou_games_played(row: pd.Series) -> Optional[int]:
    """Public helper: min(home, away) current-season games from the snapshot."""
    return _games_played(row)


def _edge_bucket(edge: Optional[float]) -> str:
    if edge is None or pd.isna(edge):
        return 'unknown'
    value = abs(float(edge))
    if value < 1.0:
        return '0-1'
    if value < 2.0:
        return '1-2'
    if value < 3.0:
        return '2-3'
    if value < 4.0:
        return '3-4'
    if value < 5.0:
        return '4-5'
    return '5+'


def _gp_bucket(gp: Optional[int]) -> str:
    if gp is None:
        return 'unknown'
    if gp < 5:
        return '0-5'
    if gp < 10:
        return '5-10'
    if gp < 15:
        return '10-15'
    if gp < 20:
        return '15-20'
    return '20+'


def _calibration_error(probabilities: List[float], outcomes: List[int], bins: int = 10) -> Optional[float]:
    if not probabilities or len(probabilities) != len(outcomes):
        return None
    groups = calibration_bins(probabilities, outcomes, bins=bins)
    total = sum(group['count'] for group in groups)
    if total == 0:
        return None
    return sum(
        group['count'] / total * abs(group['mean_probability'] - group['observed_rate'])
        for group in groups
    )


def _point_clv(row: pd.Series) -> Optional[float]:
    """Point-based CLV: positive means the bet beat the closing line."""
    closing = _to_float(row.get('closing_line'))
    bet_line = _to_float(row.get('total_line'))
    side = _side(row)
    if closing is None or bet_line is None or side is None:
        return None
    return (closing - bet_line) if side == 'OVER' else (bet_line - closing)


def evaluate_ou_slice(
    df: pd.DataFrame,
    min_sample: int = 1,
    include_pushes_in_units: bool = False,
) -> Dict[str, Any]:
    """Evaluate a slice of O/U picks.

    The slice is expected to contain ledger rows. Neutral rows are ignored for
    pick metrics. The function returns ``None`` for metrics that need more data
    than the slice provides, so an empty slice can never look profitable.
    """
    if df is None:
        df = pd.DataFrame()
    picks = df[_pick_mask(df)].copy() if not df.empty else df.copy()

    wins = int((picks.get('ou_result', pd.Series(dtype=object)).astype(str).str.upper() == 'WIN').sum()) if not picks.empty else 0
    losses = int((picks.get('ou_result', pd.Series(dtype=object)).astype(str).str.upper() == 'LOSS').sum()) if not picks.empty else 0
    pushes = int((picks.get('ou_result', pd.Series(dtype=object)).astype(str).str.upper() == 'PUSH').sum()) if not picks.empty else 0
    n = len(picks)
    decided = wins + losses

    pnl = picks['ou_result'].map(_result_pnl) if not picks.empty else pd.Series(dtype=float)
    units = float(pnl.sum()) if not pnl.empty else 0.0
    roi = (units / n * 100.0) if n > 0 else None

    actual = pd.to_numeric(picks.get('actual_total'), errors='coerce') if not picks.empty else pd.Series(dtype=float)
    line = pd.to_numeric(picks.get('total_line'), errors='coerce') if not picks.empty else pd.Series(dtype=float)
    projected = pd.to_numeric(picks.get('projected_total'), errors='coerce') if not picks.empty else pd.Series(dtype=float)
    edge = pd.to_numeric(picks.get('ou_edge'), errors='coerce') if not picks.empty else pd.Series(dtype=float)
    miss = actual - line
    model_error = (actual - projected).abs()

    valid_prob = pd.Series(dtype=float, index=picks.index)
    if not picks.empty:
        if 'ou_pick_prob' in picks.columns:
            valid_prob = picks['ou_pick_prob'].apply(_to_float)
            if 'ou_win_prob' in picks.columns:
                fallback_prob = picks['ou_win_prob'].apply(_to_float)
                valid_prob = valid_prob.where(valid_prob.notna(), fallback_prob)
        elif 'ou_win_prob' in picks.columns:
            valid_prob = picks['ou_win_prob'].apply(_to_float)
    binary = picks['ou_result'].astype(str).str.upper().map({'WIN': 1, 'LOSS': 0}) if not picks.empty else pd.Series(dtype=float)
    prob_pairs = [
        (p, int(y))
        for p, y in zip(valid_prob, binary)
        if p is not None and not pd.isna(p) and 0.0 <= float(p) <= 1.0 and not pd.isna(y)
    ]
    probabilities = [p for p, _ in prob_pairs]
    outcomes = [y for _, y in prob_pairs]

    prob_metrics = probability_metrics(probabilities, outcomes) if probabilities else {}
    ece = _calibration_error(probabilities, outcomes) if probabilities else None

    clv_values = picks.apply(_point_clv, axis=1).dropna() if not picks.empty else pd.Series(dtype=float)
    explicit_clv = pd.to_numeric(picks.get('clv'), errors='coerce').dropna() if 'clv' in picks.columns and not picks.empty else pd.Series(dtype=float)
    clv_series = explicit_clv if not explicit_clv.empty else clv_values

    market_rows = _market_rows(df)
    market_actual = pd.to_numeric(market_rows.get('actual_total'), errors='coerce')
    market_line = pd.to_numeric(market_rows.get('total_line'), errors='coerce')
    market_mae = (market_actual - market_line).abs().mean() if len(market_rows) else None

    sufficient = decided >= min_sample and n > 0
    roai_interval = bootstrap_mean_interval(pnl.tolist()) if len(pnl) >= 10 else None

    return {
        'n_picks': n,
        'wins': wins,
        'losses': losses,
        'pushes': pushes,
        'decided': decided,
        'win_rate': round(wins / decided * 100.0, 2) if decided else None,
        'units': round(units, 3) if n else 0.0,
        'roi_pct': round(roi, 2) if roi is not None else None,
        'roi_95ci': [round(roai_interval[0] * 100.0, 2), round(roai_interval[1] * 100.0, 2)] if roai_interval else None,
        'sufficient_sample': sufficient,
        'min_sample': min_sample,
        'avg_edge': round(float(edge.mean()), 3) if edge.notna().any() else None,
        'mean_actual_minus_line': round(float(miss.mean()), 3) if miss.notna().any() else None,
        'model_mae': round(float(model_error.mean()), 3) if model_error.notna().any() else None,
        'market_mae': round(float(market_mae), 3) if market_mae is not None and not pd.isna(market_mae) else None,
        'bias_actual_minus_projected': round(float((actual - projected).mean()), 3) if (actual - projected).notna().any() else None,
        'corr_edge_miss': round(float(edge.corr(miss)), 4) if edge.notna().sum() > 2 and miss.notna().sum() > 2 else None,
        'brier': round(prob_metrics.get('brier', 0.0), 5) if prob_metrics else None,
        'log_loss': round(prob_metrics.get('log_loss', 0.0), 5) if prob_metrics else None,
        'calibration_error': round(float(ece), 5) if ece is not None else None,
        'mean_clv_points': round(float(clv_series.mean()), 3) if len(clv_series) else None,
        'clv_sample': int(len(clv_series)),
        'mean_pick_prob': round(float(valid_prob.dropna().mean()), 4) if valid_prob.notna().any() else None,
    }


def _evaluate_group(df: pd.DataFrame, group_col: str, min_sample: int = 1) -> Dict[str, Dict[str, Any]]:
    if df is None or df.empty or group_col not in df.columns:
        return {}
    result = {}
    for key, group in df.groupby(group_col, dropna=False):
        label = str(key)
        result[label] = evaluate_ou_slice(group, min_sample=min_sample)
    return result


def evaluate_market_baseline(df: pd.DataFrame) -> Dict[str, Any]:
    """Describe how the closing total itself performed, independent of picks."""
    if df is None or df.empty:
        return {
            'graded_games': 0,
            'market_mae': None,
            'bias_actual_minus_line': None,
            'over_rate': None,
            'under_rate': None,
        }
    graded = _market_rows(df)
    actual = pd.to_numeric(graded.get('actual_total'), errors='coerce')
    line = pd.to_numeric(graded.get('total_line'), errors='coerce')
    valid = actual.notna() & line.notna()
    actual = actual[valid]
    line = line[valid]
    if actual.empty:
        return {
            'graded_games': 0,
            'market_mae': None,
            'bias_actual_minus_line': None,
            'over_rate': None,
            'under_rate': None,
        }
    miss = actual - line
    return {
        'graded_games': int(len(miss)),
        'market_mae': round(float(miss.abs().mean()), 3),
        'bias_actual_minus_line': round(float(miss.mean()), 3),
        'over_rate': round(float((miss > 0).mean()), 4),
        'under_rate': round(float((miss < 0).mean()), 4),
    }


def evaluate_ou_frame(df: pd.DataFrame, min_sample: int = 1) -> Dict[str, Any]:
    """Full O/U report: overall metrics plus useful breakdowns."""
    if df is None:
        df = pd.DataFrame()
    picks = df[_pick_mask(df)].copy() if not df.empty else df.copy()

    if not picks.empty:
        picks['_side'] = picks.apply(_side, axis=1)
        picks['_tier'] = picks.apply(_tier, axis=1)
        picks['_edge_bucket'] = picks.apply(lambda r: _edge_bucket(_to_float(r.get('ou_edge'))), axis=1)
        picks['_gp_bucket'] = picks.apply(lambda r: _gp_bucket(_games_played(r)), axis=1)

    return {
        'overall': evaluate_ou_slice(picks, min_sample=min_sample),
        'by_side': _evaluate_group(picks, '_side', min_sample),
        'by_tier': _evaluate_group(picks, '_tier', min_sample),
        'by_edge_bucket': _evaluate_group(picks, '_edge_bucket', min_sample),
        'by_gp_bucket': _evaluate_group(picks, '_gp_bucket', min_sample),
        'market_baseline': evaluate_market_baseline(df),
    }
