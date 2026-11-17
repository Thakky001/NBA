"""Walk-forward evaluation of a market-anchored O/U residual model.

The historical ledger is the only source of genuine market totals in this
repository.  This module therefore treats the ledger as research data, rebuilds
a leakage-safe feature frame from the *pre-game* feature snapshots, and runs an
expanding-window walk-forward evaluation.  It never fits on a row it evaluates.

Two comparisons matter:

1. **Market baseline** - predict the final total with the market line.  This is
   the number to beat on MAE, bias and betting ROI.
2. **Math-only model** - use the deployed model projection as the forecast.
   This isolates whether the market residual model adds anything beyond the
   existing quantitative model.

The module is deliberately read-only with respect to the production ledger.
"""

from __future__ import annotations

import math
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from analysis.ai_residual_model import (
    MARKET_OU_FEATURE_COLUMNS,
    MarketAnchoredResidualModel,
    build_market_residual_features,
)
from analysis.backtesting import (
    bootstrap_mean_interval,
    probability_metrics,
    walk_forward_splits,
)
from analysis.ou_evaluation import ou_side, ou_tier
from analysis.prediction_tracker import load_ledger

WIN_PAYOUT = 100.0 / 110.0
DEFAULT_MIN_TRAIN = 80
DEFAULT_VALIDATION = 20
DEFAULT_TEST = 10
MIN_GO_LIVE_BETS = 100
SHRINKAGE_GRID = (0.0, 0.25, 0.50, 0.75, 1.00)


def _blocked_decision(reason: str) -> Dict[str, Any]:
    return {
        'status': 'INSUFFICIENT_DATA',
        'reason': reason,
        'go_live_ready': False,
        'paper_only': True,
        'beats_market_mae': False,
        'beats_market_mae_significantly': False,
        'beats_math_model': False,
        'qualifying_thresholds': [],
        'minimum_go_live_bets': MIN_GO_LIVE_BETS,
        'n_test_rows': 0,
        'n_test_picks': 0,
    }


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _ece(probabilities: Sequence[float], outcomes: Sequence[int], bins: int = 5) -> Optional[float]:
    if not probabilities or len(probabilities) != len(outcomes):
        return None
    groups: List[List[Tuple[float, int]]] = [[] for _ in range(bins)]
    for p, y in zip(probabilities, outcomes):
        if not 0.0 <= p <= 1.0:
            continue
        groups[min(int(p * bins), bins - 1)].append((float(p), int(y)))
    total = sum(len(group) for group in groups)
    if total == 0:
        return None
    return sum(
        len(group) / total * abs(
            (sum(p for p, _ in group) / len(group))
            - (sum(y for _, y in group) / len(group))
        )
        for group in groups if group
    )


def _result_for_side(actual_total: float, line: float, side: str) -> str:
    if abs(actual_total - line) < 1e-9:
        return 'PUSH'
    if side == 'OVER':
        return 'WIN' if actual_total > line else 'LOSS'
    return 'WIN' if actual_total < line else 'LOSS'


def _pnl(result: str) -> float:
    if result == 'WIN':
        return WIN_PAYOUT
    if result == 'LOSS':
        return -1.0
    return 0.0


def build_market_residual_frame(
    ledger: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Build a leakage-safe research frame from the predictions ledger.

    Required columns are checked explicitly.  Rows without a usable market line,
    final total, projection or feature snapshot are excluded and counted so the
    caller can fail closed instead of silently shrinking the sample.
    """
    if ledger is None:
        ledger = load_ledger()
    if ledger is None or ledger.empty:
        return pd.DataFrame(), {
            'rows_in': 0,
            'rows_usable': 0,
            'skipped': {'empty_ledger': 1},
        }

    required = ['date', 'total_line', 'actual_total', 'projected_total', 'features_snapshot']
    missing = [col for col in required if col not in ledger.columns]
    if missing:
        return pd.DataFrame(), {
            'rows_in': int(len(ledger)),
            'rows_usable': 0,
            'skipped': {'missing_columns': missing},
        }

    skipped: Counter = Counter()
    rows: List[Dict[str, Any]] = []

    for _, row in ledger.iterrows():
        line = _to_float(row.get('total_line'))
        actual = _to_float(row.get('actual_total'))
        projection = _to_float(row.get('projected_total'))
        game_date = row.get('date')
        if line is None or line <= 0:
            skipped['missing_market_line'] += 1
            continue
        if actual is None or actual <= 0:
            skipped['missing_actual_total'] += 1
            continue
        if projection is None:
            skipped['missing_projection'] += 1
            continue
        try:
            features = build_market_residual_features(
                total_line=line,
                projected_total=projection,
                features=row.get('features_snapshot'),
                game_date=game_date,
            )
        except Exception:
            skipped['invalid_feature_snapshot'] += 1
            continue

        record: Dict[str, Any] = {
            'game_id': str(row.get('game_id', '')),
            'date': game_date,
            'home_team': row.get('home_team'),
            'away_team': row.get('away_team'),
            'total_line': line,
            'projected_total': projection,
            'actual_total': actual,
            'target_residual': actual - line,
            'model_residual': projection - line,
            'recommendation': str(row.get('ou_recommendation', 'NEUTRAL') or 'NEUTRAL'),
            'tier': ou_tier(row),
            'side': ou_side(row),
            'ou_edge': _to_float(row.get('ou_edge')),
            'ou_win_prob': _to_float(row.get('ou_win_prob')),
            'features_snapshot': row.get('features_snapshot'),
        }
        record.update(features)
        rows.append(record)

    if not rows:
        return pd.DataFrame(), {
            'rows_in': int(len(ledger)),
            'rows_usable': 0,
            'skipped': dict(skipped),
        }

    frame = pd.DataFrame(rows)
    frame['date'] = pd.to_datetime(frame['date'], errors='coerce')
    frame = frame[frame['date'].notna()].copy()
    frame = frame.sort_values(['date', 'game_id'], kind='stable').reset_index(drop=True)
    frame['is_pick'] = frame['recommendation'].str.upper().str.contains('OVER|UNDER', regex=True)
    frame['post_split'] = frame['date'] >= pd.Timestamp('2026-11-06')

    diagnostics = {
        'rows_in': int(len(ledger)),
        'rows_usable': int(len(frame)),
        'skipped': dict(skipped),
        'date_range': [
            str(frame['date'].min().date()) if len(frame) else None,
            str(frame['date'].max().date()) if len(frame) else None,
        ],
        'picks': int(frame['is_pick'].sum()),
        'post_split_rows': int(frame['post_split'].sum()),
        'post_split_picks': int((frame['is_pick'] & frame['post_split']).sum()),
        'unique_dates': int(frame['date'].nunique()),
        'market_line_coverage': round(float(frame['total_line'].notna().mean()), 4),
    }
    return frame, diagnostics


def _evaluate_predictions(
    frame: pd.DataFrame,
    prediction_col: str,
    sigma_col: Optional[str] = None,
    thresholds: Iterable[float] = (1.0, 2.0, 3.0, 4.0),
) -> Dict[str, Any]:
    """Evaluate one forecast column without fitting anything."""
    if frame is None or frame.empty or prediction_col not in frame.columns:
        return {'n_rows': 0, 'status': 'NO_DATA'}

    working = frame.copy()
    working['_prediction'] = pd.to_numeric(working[prediction_col], errors='coerce')
    working['_line'] = pd.to_numeric(working['total_line'], errors='coerce')
    working['_actual'] = pd.to_numeric(working['actual_total'], errors='coerce')
    working['_target'] = working['_actual'] - working['_line']
    working = working[
        working['_prediction'].notna()
        & working['_line'].notna()
        & working['_actual'].notna()
    ].copy()
    if working.empty:
        return {'n_rows': 0, 'status': 'NO_DATA'}

    prediction = working['_prediction'].to_numpy(dtype=float)
    target = working['_target'].to_numpy(dtype=float)
    abs_market = np.abs(target)
    abs_model = np.abs(target - prediction)
    mae_market = float(abs_market.mean())
    mae_model = float(abs_model.mean())
    improvement = abs_market - abs_model
    interval = (
        bootstrap_mean_interval(improvement.tolist())
        if len(improvement) >= 10 else None
    )

    correlation = None
    if len(prediction) >= 3 and np.std(prediction) > 1e-9 and np.std(target) > 1e-9:
        correlation = float(np.corrcoef(prediction, target)[0, 1])

    sigma = None
    if sigma_col and sigma_col in working.columns:
        sigma_values = pd.to_numeric(working[sigma_col], errors='coerce').dropna()
        sigma = float(sigma_values.median()) if not sigma_values.empty else None
    if sigma is None or sigma <= 0:
        sigma = max(6.0, float(np.std(target - prediction, ddof=1))) if len(target) > 2 else 13.5

    result: Dict[str, Any] = {
        'n_rows': int(len(working)),
        'mae': round(mae_model, 3),
        'market_mae': round(mae_market, 3),
        'mae_improvement_vs_market': round(mae_market - mae_model, 3),
        'mae_improvement_95ci': (
            [round(interval[0], 3), round(interval[1], 3)] if interval else None
        ),
        'beats_market_mae': bool(mae_model < mae_market),
        'beats_market_mae_significantly': bool(
            interval is not None and interval[0] > 0
        ),
        'rmse': round(float(np.sqrt(np.mean((target - prediction) ** 2))), 3),
        'bias_actual_minus_predicted': round(float((target - prediction).mean()), 3),
        'corr_prediction_target': round(correlation, 4) if correlation is not None else None,
        'sigma_used': round(float(sigma), 3),
        'thresholds': {},
    }

    for threshold in thresholds:
        mask = np.abs(prediction) >= float(threshold)
        n = int(mask.sum())
        if n == 0:
            result['thresholds'][f'{threshold:g}'] = {
                'n_picks': 0,
                'status': 'NO_PICKS',
            }
            continue
        selected = working.loc[mask].copy()
        selected['_side'] = np.where(selected['_prediction'] > 0, 'OVER', 'UNDER')
        selected['_result'] = [
            _result_for_side(float(a), float(l), str(s))
            for a, l, s in zip(selected['_actual'], selected['_line'], selected['_side'])
        ]
        selected['_pnl'] = selected['_result'].map(_pnl)
        wins = int((selected['_result'] == 'WIN').sum())
        losses = int((selected['_result'] == 'LOSS').sum())
        pushes = int((selected['_result'] == 'PUSH').sum())
        decided = wins + losses
        pnl = selected['_pnl'].to_numpy(dtype=float)
        roi = float(pnl.sum() / n) if n else None
        roi_ci = bootstrap_mean_interval(pnl.tolist()) if n >= 10 else None

        selected_sigma = (
            pd.to_numeric(selected[sigma_col], errors='coerce').fillna(sigma)
            if sigma_col and sigma_col in selected.columns
            else pd.Series([sigma] * len(selected), index=selected.index)
        )
        probabilities = []
        outcomes = []
        for _, row in selected.iterrows():
            pred = float(row['_prediction'])
            row_sigma = max(6.0, float(selected_sigma.loc[row.name]))
            p_over = _normal_cdf(pred / row_sigma)
            p_side = p_over if row['_side'] == 'OVER' else 1.0 - p_over
            outcome = 1 if row['_result'] == 'WIN' else (0 if row['_result'] == 'LOSS' else None)
            if outcome is not None:
                probabilities.append(float(p_side))
                outcomes.append(outcome)

        prob_metrics = probability_metrics(probabilities, outcomes) if probabilities else {}
        result['thresholds'][f'{threshold:g}'] = {
            'n_picks': n,
            'wins': wins,
            'losses': losses,
            'pushes': pushes,
            'win_rate': round(wins / decided * 100.0, 2) if decided else None,
            'units': round(float(pnl.sum()), 3),
            'roi_pct': round(roi * 100.0, 2) if roi is not None else None,
            'roi_95ci_pct': (
                [round(roi_ci[0] * 100.0, 2), round(roi_ci[1] * 100.0, 2)]
                if roi_ci else None
            ),
            'brier': round(prob_metrics.get('brier', 0.0), 5) if prob_metrics else None,
            'log_loss': round(prob_metrics.get('log_loss', 0.0), 5) if prob_metrics else None,
            'ece': round(_ece(probabilities, outcomes), 5) if probabilities else None,
            'sufficient_sample': n >= MIN_GO_LIVE_BETS,
        }

    return result


def _fit_fold(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    alpha: float = 10.0,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Fit on train, choose shrinkage on validation, refit, predict test."""
    if len(train_df) < 10:
        raise ValueError("training fold is too small")
    model = MarketAnchoredResidualModel(alpha=alpha)
    train_fit = model.fit(train_df, target_col='target_residual')

    alpha_used = 1.0
    validation_mae = None
    if len(validation_df) >= 5:
        val_pred = model.predict_residuals(validation_df)
        val_market_mae = float(np.mean(np.abs(validation_df['target_residual'])))
        best = None
        for candidate in SHRINKAGE_GRID:
            val_mae = float(
                np.mean(
                    np.abs(
                        validation_df['target_residual'].to_numpy(dtype=float)
                        - candidate * val_pred
                    )
                )
            )
            if best is None or val_mae < best[0]:
                best = (val_mae, candidate)
        validation_mae, alpha_used = best
        if val_market_mae <= validation_mae:
            alpha_used = 0.0

    combined = pd.concat([train_df, validation_df], ignore_index=True)
    final_model = MarketAnchoredResidualModel(alpha=alpha)
    final_fit = final_model.fit(combined, target_col='target_residual')
    predicted = alpha_used * final_model.predict_residuals(test_df)

    out = test_df.copy()
    out['predicted_residual'] = predicted
    out['predicted_total'] = out['total_line'].to_numpy(dtype=float) + predicted
    out['shrinkage_alpha'] = alpha_used

    # Probability scale must come from held-back residuals, never from the test
    # fold.  Validation is the first choice; train-only residuals are the
    # fallback when a fold has no validation block.
    if len(validation_df) >= 5:
        val_residuals = (
            validation_df['target_residual'].to_numpy(dtype=float)
            - alpha_used * model.predict_residuals(validation_df)
        )
    else:
        train_residuals = (
            train_df['target_residual'].to_numpy(dtype=float)
            - alpha_used * model.predict_residuals(train_df)
        )
        val_residuals = train_residuals
    sigma = float(np.std(val_residuals, ddof=1)) if len(val_residuals) > 1 else 13.5
    out['sigma_residual'] = max(6.0, sigma)

    math_sigma = float(
        np.std(
            train_df['target_residual'].to_numpy(dtype=float)
            - train_df['model_residual'].to_numpy(dtype=float),
            ddof=1,
        )
    ) if len(train_df) > 2 else 13.5
    out['sigma_math'] = max(6.0, math_sigma)

    summary = {
        'train_rows': int(len(train_df)),
        'validation_rows': int(len(validation_df)),
        'test_rows': int(len(test_df)),
        'alpha_used': alpha_used,
        'validation_mae': round(float(validation_mae), 3) if validation_mae is not None else None,
        'train_fit': train_fit,
        'final_fit': final_fit,
        'coefficients': final_model.coefficients(),
    }
    return out, summary


def _aggregate_coefficients(folds: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not folds:
        return {}
    names = MARKET_OU_FEATURE_COLUMNS
    matrix = np.array(
        [[fold.get('coefficients', {}).get(name, 0.0) for name in names] for fold in folds],
        dtype=float,
    )
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    return {
        name: {
            'mean': round(float(m), 5),
            'std': round(float(s), 5),
            'sign_stability': round(
                float(max((matrix[:, i] >= 0).mean(), (matrix[:, i] <= 0).mean())), 3
            ),
        }
        for i, (name, m, s) in enumerate(zip(names, mean, std))
    }


def _decision(
    residual_eval: Dict[str, Any],
    math_eval: Dict[str, Any],
    n_test_rows: int,
    n_test_picks: int,
) -> Dict[str, Any]:
    thresholds = residual_eval.get('thresholds', {})
    qualifying = [
        (key, value) for key, value in thresholds.items()
        if value.get('n_picks', 0) >= MIN_GO_LIVE_BETS
        and value.get('roi_95ci_pct') is not None
        and value['roi_95ci_pct'][0] > 0
    ]

    if n_test_rows < 50 or n_test_picks < 30:
        status = 'INSUFFICIENT_DATA'
        reason = (
            f"only {n_test_rows} out-of-sample rows / {n_test_picks} model picks; "
            "need a much larger prospective sample"
        )
    elif not residual_eval.get('beats_market_mae_significantly'):
        status = 'NO_MARKET_EDGE'
        reason = (
            "market-anchored residual model does not beat the market line on "
            "out-of-sample MAE with a 95% interval above zero"
        )
    elif qualifying:
        status = 'CANDIDATE_FOR_SHADOW'
        reason = "residual model beats the market and at least one threshold has a positive ROI interval"
    else:
        status = 'NO_BETTING_EDGE'
        reason = "residual model may fit totals better but no threshold has a positive ROI interval"

    return {
        'status': status,
        'reason': reason,
        'go_live_ready': False,
        'paper_only': True,
        'beats_market_mae': bool(residual_eval.get('beats_market_mae')),
        'beats_market_mae_significantly': bool(
            residual_eval.get('beats_market_mae_significantly')
        ),
        'beats_math_model': bool(
            residual_eval.get('mae', 1e9) < math_eval.get('mae', 1e9)
        ),
        'qualifying_thresholds': [key for key, _ in qualifying],
        'minimum_go_live_bets': MIN_GO_LIVE_BETS,
        'n_test_rows': int(n_test_rows),
        'n_test_picks': int(n_test_picks),
    }


def run_market_residual_walkforward(
    ledger: Optional[pd.DataFrame] = None,
    min_train: int = DEFAULT_MIN_TRAIN,
    validation_size: int = DEFAULT_VALIDATION,
    test_size: int = DEFAULT_TEST,
    ridge_alpha: float = 10.0,
) -> Dict[str, Any]:
    """Run the expanding-window market-residual evaluation."""
    frame, diagnostics = build_market_residual_frame(ledger)
    generated_at = datetime.now(timezone.utc).isoformat()

    if frame.empty:
        reason = 'no usable rows with a market total, final total and feature snapshot'
        return {
            'report_version': 'ou-market-residual-v1.0',
            'generated_at': generated_at,
            'status': 'INSUFFICIENT_DATA',
            'reason': reason,
            'data': diagnostics,
            'decision': _blocked_decision(reason),
            'warnings': ['The ledger is the only source of historical market totals.'],
        }

    dates = frame['date'].tolist()
    try:
        folds = walk_forward_splits(
            dates,
            min_train=min_train,
            validation_size=validation_size,
            test_size=test_size,
            step=test_size,
        )
    except ValueError as exc:
        return {
            'report_version': 'ou-market-residual-v1.0',
            'generated_at': generated_at,
            'status': 'INSUFFICIENT_DATA',
            'reason': str(exc),
            'data': diagnostics,
            'decision': _blocked_decision(str(exc)),
        }

    if not folds:
        reason = (
            f"not enough rows for min_train={min_train}, "
            f"validation={validation_size}, test={test_size}"
        )
        return {
            'report_version': 'ou-market-residual-v1.0',
            'generated_at': generated_at,
            'status': 'INSUFFICIENT_DATA',
            'reason': reason,
            'data': diagnostics,
            'decision': _blocked_decision(reason),
        }

    fold_summaries: List[Dict[str, Any]] = []
    test_rows: List[pd.DataFrame] = []

    for fold_index, fold in enumerate(folds):
        train_df = frame.iloc[fold['train']].copy()
        validation_df = frame.iloc[fold['validation']].copy()
        test_df = frame.iloc[fold['test']].copy()
        try:
            predicted, summary = _fit_fold(
                train_df,
                validation_df,
                test_df,
                alpha=ridge_alpha,
            )
        except Exception as exc:
            fold_summaries.append({
                'fold': fold_index,
                'status': 'SKIPPED',
                'reason': str(exc),
            })
            continue
        summary['fold'] = fold_index
        summary['test_start'] = str(test_df['date'].min().date())
        summary['test_end'] = str(test_df['date'].max().date())
        fold_summaries.append(summary)
        test_rows.append(predicted)

    if not test_rows:
        reason = 'every walk-forward fold failed to fit'
        return {
            'report_version': 'ou-market-residual-v1.0',
            'generated_at': generated_at,
            'status': 'INSUFFICIENT_DATA',
            'reason': reason,
            'data': diagnostics,
            'folds': fold_summaries,
            'decision': _blocked_decision(reason),
        }

    out = pd.concat(test_rows, ignore_index=True)
    out = out.sort_values(['date', 'game_id'], kind='stable').reset_index(drop=True)

    market_eval = _evaluate_predictions(
        out.assign(market_residual=0.0),
        'market_residual',
        sigma_col=None,
    )
    math_eval = _evaluate_predictions(out, 'model_residual', sigma_col='sigma_math')
    residual_eval = _evaluate_predictions(
        out, 'predicted_residual', sigma_col='sigma_residual'
    )

    n_test_rows = int(len(out))
    n_test_picks = int(
        (
            (out['predicted_residual'].abs() >= 1.0)
            & out['total_line'].notna()
            & out['actual_total'].notna()
        ).sum()
    )
    decision = _decision(residual_eval, math_eval, n_test_rows, n_test_picks)

    warnings = [
        "Only the predictions ledger supplies genuine historical market totals; "
        "the cached residual dataset used synthetic rolling-average 'lines' and is excluded.",
        "The ledger has no closing-line captures yet, so CLV is not part of this evaluation.",
        "Rows before 2026-11-06 were generated by older gate versions and recorded "
        "games-played counts from contaminated team pairings; use the post-split window "
        "for cleaner prospective evidence.",
        "The residual model is a regularised linear model, not a CatBoost model. "
        "With fewer than ~500 market-lined games, a high-variance learner would overfit.",
        "This is research output only. The production pause flag stays enabled until "
        "the separate prospective go-live gates pass.",
    ]

    return {
        'report_version': 'ou-market-residual-v1.0',
        'generated_at': generated_at,
        'status': decision['status'],
        'data': diagnostics,
        'walk_forward': {
            'min_train': min_train,
            'validation_size': validation_size,
            'test_size': test_size,
            'ridge_alpha': ridge_alpha,
            'folds_requested': len(folds),
            'folds_completed': len([f for f in fold_summaries if f.get('status') != 'SKIPPED']),
            'date_range': [
                str(out['date'].min().date()),
                str(out['date'].max().date()),
            ],
            'out_of_sample_rows': n_test_rows,
            'out_of_sample_picks': n_test_picks,
        },
        'variants': {
            'market': market_eval,
            'math_model': math_eval,
            'market_residual_model': residual_eval,
        },
        'folds': fold_summaries,
        'coefficients': _aggregate_coefficients(
            [f for f in fold_summaries if f.get('coefficients')]
        ),
        'decision': decision,
        'warnings': warnings,
        'feature_columns': list(MARKET_OU_FEATURE_COLUMNS),
    }
