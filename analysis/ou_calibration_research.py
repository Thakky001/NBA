"""Walk-forward probability calibration and selective-betting research.

The production gate already applies a fixed temperature (T=2.0) fitted
in-sample.  This module asks a stricter question: if temperature is fitted only
on earlier games and applied to later games, does calibration improve Brier,
log loss and ECE?  It also evaluates a small, pre-registered set of selective
filters on held-out games.

Nothing here can enable real-money betting.  The report is an input to the
shadow-mode review, and the production selector stays disabled by default.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from analysis.backtesting import (
    bootstrap_mean_interval,
    probability_metrics,
    walk_forward_splits,
)
from analysis.ou_evaluation import ou_games_played, ou_side, ou_tier
from analysis.prediction_tracker import load_ledger

WIN_PAYOUT = 100.0 / 110.0
MIN_SELECTIVE_BETS = 100
MIN_TRAIN_ROWS = 80
VALIDATION_ROWS = 20
TEST_ROWS = 10
TEMPERATURE_GRID = tuple(round(0.25 + 0.05 * i, 2) for i in range(196))

# These rules are pre-registered in docs/ou_hypotheses.md.  Adding a rule after
# looking at the results invalidates its p-value and must be marked exploratory.
SELECTIVE_RULES = (
    ('all_picks', 'ทุกไม้ O/U ที่โมเดลเลือก'),
    ('strong_only', 'STRONG เท่านั้น'),
    ('strong_gp15', 'STRONG และทั้งสองทีมเล่น >= 15 เกม'),
    ('under_only', 'UNDER เท่านั้น'),
    ('strong_under', 'STRONG UNDER เท่านั้น'),
    ('edge_ge_2', 'ขนาด edge >= 2.0 แต้ม'),
    ('prob_ge_053', 'calibrated pick probability >= 0.53'),
    ('edge_ge_2_prob_ge_053', 'edge >= 2.0 และ calibrated probability >= 0.53'),
)


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _logit(probability: float) -> float:
    p = min(max(float(probability), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def temperature_scale(probability: float, temperature: float) -> float:
    """Apply temperature scaling to a binary probability."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return _sigmoid(_logit(probability) / float(temperature))


def _log_loss_single(probability: float, outcome: int) -> float:
    p = min(max(float(probability), 1e-12), 1.0 - 1e-12)
    return -(outcome * math.log(p) + (1 - outcome) * math.log(1.0 - p))


def fit_temperature(probabilities: Sequence[float], outcomes: Sequence[int]) -> Optional[Dict[str, Any]]:
    """Grid-search T that minimises log loss on the supplied rows only."""
    pairs = [
        (float(p), int(y))
        for p, y in zip(probabilities, outcomes)
        if p is not None and 0.0 < float(p) < 1.0 and y in (0, 1)
    ]
    if len(pairs) < 10:
        return None
    if len({y for _, y in pairs}) < 2:
        return None
    best = None
    for temperature in TEMPERATURE_GRID:
        loss = sum(
            _log_loss_single(temperature_scale(p, temperature), y)
            for p, y in pairs
        ) / len(pairs)
        if best is None or loss < best[1]:
            best = (temperature, loss)
    return {
        'temperature': best[0],
        'train_log_loss': round(float(best[1]), 5),
        'train_samples': len(pairs),
    }


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


def _build_frame(ledger: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if ledger is None:
        ledger = load_ledger()
    required = ['date', 'total_line', 'actual_total', 'projected_total', 'ou_recommendation']
    if ledger is None or ledger.empty:
        return pd.DataFrame(), {'rows_in': 0, 'rows_usable': 0, 'reason': 'empty ledger'}
    missing = [c for c in required if c not in ledger.columns]
    if missing:
        return pd.DataFrame(), {
            'rows_in': int(len(ledger)),
            'rows_usable': 0,
            'reason': f'missing columns: {missing}',
        }

    rows = []
    skipped = {'missing_line': 0, 'missing_actual': 0, 'missing_projection': 0, 'push': 0}
    for _, row in ledger.iterrows():
        line = _to_float(row.get('total_line'))
        actual = _to_float(row.get('actual_total'))
        projection = _to_float(row.get('projected_total'))
        if line is None or line <= 0:
            skipped['missing_line'] += 1
            continue
        if actual is None or actual <= 0:
            skipped['missing_actual'] += 1
            continue
        if projection is None:
            skipped['missing_projection'] += 1
            continue
        if abs(actual - line) < 1e-9:
            skipped['push'] += 1
            continue
        side = ou_side(row)
        rec = str(row.get('ou_recommendation', 'NEUTRAL') or 'NEUTRAL').upper()
        rows.append({
            'game_id': str(row.get('game_id', '')),
            'date': row.get('date'),
            'total_line': line,
            'actual_total': actual,
            'projected_total': projection,
            'recommendation': rec,
            'tier': ou_tier(row),
            'side': side,
            'games_played': ou_games_played(row),
            'ou_edge': _to_float(row.get('ou_edge')),
            'stored_pick_prob': _to_float(row.get('ou_win_prob')),
            'outcome_over': 1 if actual > line else 0,
            'is_pick': bool(side and ('OVER' in rec or 'UNDER' in rec)),
        })

    if not rows:
        return pd.DataFrame(), {
            'rows_in': int(len(ledger)),
            'rows_usable': 0,
            'skipped': skipped,
        }

    frame = pd.DataFrame(rows)
    frame['date'] = pd.to_datetime(frame['date'], errors='coerce')
    frame = frame[frame['date'].notna()].copy()
    frame = frame.sort_values(['date', 'game_id'], kind='stable').reset_index(drop=True)
    return frame, {
        'rows_in': int(len(ledger)),
        'rows_usable': int(len(frame)),
        'skipped': skipped,
        'date_range': [
            str(frame['date'].min().date()),
            str(frame['date'].max().date()),
        ],
        'picks': int(frame['is_pick'].sum()),
        'unique_dates': int(frame['date'].nunique()),
    }


def _evaluate_probabilities(
    probabilities: Sequence[float],
    outcomes: Sequence[int],
) -> Dict[str, Any]:
    pairs = [
        (float(p), int(y)) for p, y in zip(probabilities, outcomes)
        if p is not None and 0.0 <= float(p) <= 1.0 and y in (0, 1)
    ]
    if not pairs:
        return {'n': 0}
    probs = [p for p, _ in pairs]
    ys = [y for _, y in pairs]
    metrics = probability_metrics(probs, ys)
    return {
        'n': len(pairs),
        'brier': round(metrics['brier'], 5),
        'log_loss': round(metrics['log_loss'], 5),
        'ece': round(_ece(probs, ys), 5),
        'mean_probability': round(float(np.mean(probs)), 4),
        'observed_rate': round(float(np.mean(ys)), 4),
    }


def _evaluate_betting(
    frame: pd.DataFrame,
    prob_col: str,
    rule_id: str,
) -> Dict[str, Any]:
    if frame is None or frame.empty:
        return {'rule': rule_id, 'n_picks': 0, 'status': 'NO_PICKS'}

    selected = frame[frame['is_pick']].copy()
    if selected.empty:
        return {'rule': rule_id, 'n_picks': 0, 'status': 'NO_PICKS'}

    selected['_side'] = selected['side']
    selected['_result'] = np.where(
        selected['_side'] == 'OVER',
        np.where(selected['actual_total'] > selected['total_line'], 'WIN', 'LOSS'),
        np.where(selected['actual_total'] < selected['total_line'], 'WIN', 'LOSS'),
    )
    selected['_pnl'] = selected['_result'].map({'WIN': WIN_PAYOUT, 'LOSS': -1.0})
    n = len(selected)
    wins = int((selected['_result'] == 'WIN').sum())
    losses = int((selected['_result'] == 'LOSS').sum())
    pnl = selected['_pnl'].to_numpy(dtype=float)
    roi_ci = bootstrap_mean_interval(pnl.tolist()) if n >= 10 else None
    probabilities = []
    outcomes = []
    if prob_col in selected.columns:
        for p, result in zip(selected[prob_col], selected['_result']):
            p = _to_float(p)
            if p is not None:
                probabilities.append(p)
                outcomes.append(1 if result == 'WIN' else 0)
    prob_metrics = probability_metrics(probabilities, outcomes) if probabilities else {}
    return {
        'rule': rule_id,
        'n_picks': n,
        'wins': wins,
        'losses': losses,
        'win_rate': round(wins / (wins + losses) * 100.0, 2) if wins + losses else None,
        'units': round(float(pnl.sum()), 3),
        'roi_pct': round(float(pnl.mean()) * 100.0, 2),
        'roi_95ci_pct': (
            [round(roi_ci[0] * 100.0, 2), round(roi_ci[1] * 100.0, 2)]
            if roi_ci else None
        ),
        'brier': round(prob_metrics.get('brier', 0.0), 5) if prob_metrics else None,
        'log_loss': round(prob_metrics.get('log_loss', 0.0), 5) if prob_metrics else None,
        'ece': round(_ece(probabilities, outcomes), 5) if probabilities else None,
        'sufficient_sample': n >= MIN_SELECTIVE_BETS,
    }


def _rule_mask(frame: pd.DataFrame, rule_id: str) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=bool)
    base = frame['is_pick']
    if rule_id == 'all_picks':
        return base
    if rule_id == 'strong_only':
        return base & (frame['tier'] == 'STRONG')
    if rule_id == 'strong_gp15':
        gp = pd.to_numeric(frame['games_played'], errors='coerce')
        return base & (frame['tier'] == 'STRONG') & (gp >= 15)
    if rule_id == 'under_only':
        return base & (frame['side'] == 'UNDER')
    if rule_id == 'strong_under':
        return base & (frame['tier'] == 'STRONG') & (frame['side'] == 'UNDER')
    if rule_id == 'edge_ge_2':
        return base & (pd.to_numeric(frame['ou_edge'], errors='coerce').abs() >= 2.0)
    if rule_id == 'prob_ge_053':
        return base & (pd.to_numeric(frame['calibrated_pick_prob'], errors='coerce') >= 0.53)
    if rule_id == 'edge_ge_2_prob_ge_053':
        return (
            base
            & (pd.to_numeric(frame['ou_edge'], errors='coerce').abs() >= 2.0)
            & (pd.to_numeric(frame['calibrated_pick_prob'], errors='coerce') >= 0.53)
        )
    return base


def run_calibration_research(
    ledger: Optional[pd.DataFrame] = None,
    min_train: int = MIN_TRAIN_ROWS,
    validation_size: int = VALIDATION_ROWS,
    test_size: int = TEST_ROWS,
) -> Dict[str, Any]:
    """Run leakage-safe calibration and selective-rule evaluation."""
    frame, diagnostics = _build_frame(ledger)
    generated_at = datetime.now(timezone.utc).isoformat()
    if frame.empty:
        return {
            'report_version': 'ou-calibration-v1.0',
            'generated_at': generated_at,
            'status': 'INSUFFICIENT_DATA',
            'reason': diagnostics.get('reason', 'no usable rows'),
            'data': diagnostics,
            'decision': {'enable_selective_gate': False, 'production_enabled': False},
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
            'report_version': 'ou-calibration-v1.0',
            'generated_at': generated_at,
            'status': 'INSUFFICIENT_DATA',
            'reason': str(exc),
            'data': diagnostics,
            'decision': {'enable_selective_gate': False, 'production_enabled': False},
        }

    if not folds:
        return {
            'report_version': 'ou-calibration-v1.0',
            'generated_at': generated_at,
            'status': 'INSUFFICIENT_DATA',
            'reason': (
                f'not enough rows for min_train={min_train}, '
                f'validation={validation_size}, test={test_size}'
            ),
            'data': diagnostics,
            'decision': {'enable_selective_gate': False, 'production_enabled': False},
        }

    fold_summaries: List[Dict[str, Any]] = []
    test_rows: List[pd.DataFrame] = []

    for fold_index, fold in enumerate(folds):
        train_df = frame.iloc[fold['train']].copy()
        test_df = frame.iloc[fold['test']].copy()
        if len(train_df) < 10 or test_df.empty:
            continue

        train_over = train_df['outcome_over'].to_numpy(dtype=int)
        train_residual = (
            train_df['actual_total'].to_numpy(dtype=float)
            - train_df['projected_total'].to_numpy(dtype=float)
        )
        sigma = float(np.std(train_residual, ddof=1)) if len(train_residual) > 1 else 13.5
        sigma = max(6.0, sigma)

        raw_over = [
            _normal_cdf((float(proj) - float(line)) / sigma)
            for proj, line in zip(train_df['projected_total'], train_df['total_line'])
        ]
        temp_fit = fit_temperature(raw_over, train_over)

        test_sigma = sigma
        test_raw_over = [
            _normal_cdf((float(proj) - float(line)) / test_sigma)
            for proj, line in zip(test_df['projected_total'], test_df['total_line'])
        ]
        temperature = temp_fit['temperature'] if temp_fit else 1.0
        test_cal_over = [temperature_scale(p, temperature) for p in test_raw_over]

        test_df = test_df.copy()
        test_df['raw_over_prob'] = test_raw_over
        test_df['calibrated_over_prob'] = test_cal_over
        test_df['sigma_used'] = test_sigma
        test_df['temperature'] = temperature
        test_df['fold'] = fold_index

        def _pick_prob(over_prob: float, side: Optional[str]) -> Optional[float]:
            if side == 'OVER':
                return over_prob
            if side == 'UNDER':
                return 1.0 - over_prob
            return None

        test_df['raw_pick_prob'] = [
            _pick_prob(p, s) for p, s in zip(test_df['raw_over_prob'], test_df['side'])
        ]
        test_df['calibrated_pick_prob'] = [
            _pick_prob(p, s) for p, s in zip(test_df['calibrated_over_prob'], test_df['side'])
        ]
        test_df['outcome_pick'] = np.where(
            test_df['side'] == 'OVER',
            (test_df['actual_total'] > test_df['total_line']).astype(int),
            (test_df['actual_total'] < test_df['total_line']).astype(int),
        )

        fold_summaries.append({
            'fold': fold_index,
            'train_rows': int(len(train_df)),
            'test_rows': int(len(test_df)),
            'test_start': str(test_df['date'].min().date()),
            'test_end': str(test_df['date'].max().date()),
            'sigma': round(test_sigma, 3),
            'temperature': temperature,
            'temperature_fit': temp_fit,
        })
        test_rows.append(test_df)

    if not test_rows:
        return {
            'report_version': 'ou-calibration-v1.0',
            'generated_at': generated_at,
            'status': 'INSUFFICIENT_DATA',
            'reason': 'every calibration fold was too small to fit',
            'data': diagnostics,
            'folds': fold_summaries,
            'decision': {'enable_selective_gate': False, 'production_enabled': False},
        }

    out = pd.concat(test_rows, ignore_index=True)
    out = out.sort_values(['date', 'game_id'], kind='stable').reset_index(drop=True)

    raw_calibration = _evaluate_probabilities(out['raw_over_prob'], out['outcome_over'])
    calibrated_calibration = _evaluate_probabilities(
        out['calibrated_over_prob'], out['outcome_over']
    )
    raw_pick_calibration = _evaluate_probabilities(
        out.loc[out['is_pick'], 'raw_pick_prob'],
        out.loc[out['is_pick'], 'outcome_pick'],
    )
    calibrated_pick_calibration = _evaluate_probabilities(
        out.loc[out['is_pick'], 'calibrated_pick_prob'],
        out.loc[out['is_pick'], 'outcome_pick'],
    )

    rule_results = {}
    for rule_id, _description in SELECTIVE_RULES:
        mask = _rule_mask(out, rule_id)
        subset = out[mask].copy()
        rule_results[rule_id] = _evaluate_betting(subset, 'calibrated_pick_prob', rule_id)

    # Nested adaptive rule: pick the minimum edge on the training fold only,
    # then apply it to the next fold.  This is reported separately so it cannot
    # be confused with a pre-registered rule.
    adaptive_rows = []
    for fold_index, fold in enumerate(folds):
        train_df = frame.iloc[fold['train']].copy()
        test_df = out[out['fold'] == fold_index].copy()
        if train_df.empty or test_df.empty:
            continue
        best = None
        for threshold in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
            mask = train_df['is_pick'] & (
                pd.to_numeric(train_df['ou_edge'], errors='coerce').abs() >= threshold
            )
            train_subset = train_df[mask]
            if len(train_subset) < 20:
                continue
            wins = (
                ((train_subset['side'] == 'OVER') & (train_subset['actual_total'] > train_subset['total_line']))
                | ((train_subset['side'] == 'UNDER') & (train_subset['actual_total'] < train_subset['total_line']))
            ).sum()
            roi = (wins * WIN_PAYOUT - (len(train_subset) - wins)) / len(train_subset)
            if best is None or roi > best[1]:
                best = (threshold, roi, len(train_subset))
        if best is None or best[1] <= 0:
            continue
        threshold = best[0]
        mask = test_df['is_pick'] & (
            pd.to_numeric(test_df['ou_edge'], errors='coerce').abs() >= threshold
        )
        adaptive_rows.append(test_df[mask].assign(adaptive_threshold=threshold))
    adaptive = (
        pd.concat(adaptive_rows, ignore_index=True)
        if adaptive_rows else pd.DataFrame()
    )
    adaptive_result = _evaluate_betting(adaptive, 'calibrated_pick_prob', 'adaptive_train_threshold')

    promotable_rules = [
        rule_id for rule_id, result in rule_results.items()
        if result.get('sufficient_sample')
        and result.get('roi_95ci_pct')
        and result['roi_95ci_pct'][0] > 0
        and result.get('brier') is not None
        and result['brier'] < 0.25
    ]

    temperature_max = max(TEMPERATURE_GRID)
    boundary_hits = sum(
        1 for fold in fold_summaries
        if fold.get('temperature') is not None
        and float(fold['temperature']) >= temperature_max - 1e-9
    )

    calibrated_better = (
        calibrated_calibration.get('brier') is not None
        and raw_calibration.get('brier') is not None
        and calibrated_calibration['brier'] < raw_calibration['brier']
    )
    if not out['is_pick'].any():
        status = 'INSUFFICIENT_DATA'
        reason = 'no out-of-sample model picks to evaluate'
    elif len(out) < 50:
        status = 'INSUFFICIENT_DATA'
        reason = f'only {len(out)} out-of-sample rows'
    elif promotable_rules:
        status = 'CANDIDATE_FOR_SHADOW'
        reason = 'at least one pre-registered rule is profitable with sufficient sample'
    else:
        status = 'KEEP_DISABLED'
        reason = 'no pre-registered selective rule passed the out-of-sample promotion gates'

    return {
        'report_version': 'ou-calibration-v1.0',
        'generated_at': generated_at,
        'status': status,
        'reason': reason,
        'data': diagnostics,
        'walk_forward': {
            'min_train': min_train,
            'validation_size': validation_size,
            'test_size': test_size,
            'folds_requested': len(folds),
            'folds_completed': len(fold_summaries),
            'out_of_sample_rows': int(len(out)),
            'out_of_sample_picks': int(out['is_pick'].sum()),
            'date_range': [
                str(out['date'].min().date()),
                str(out['date'].max().date()),
            ],
        },
        'calibration': {
            'all_rows_raw': raw_calibration,
            'all_rows_calibrated': calibrated_calibration,
            'picks_raw': raw_pick_calibration,
            'picks_calibrated': calibrated_pick_calibration,
            'calibrated_better_brier': bool(calibrated_better),
            'temperatures': [fold.get('temperature') for fold in fold_summaries],
            'temperature_grid': [min(TEMPERATURE_GRID), temperature_max],
            'temperature_boundary_hits': boundary_hits,
        },
        'selective_rules': rule_results,
        'adaptive_train_threshold': adaptive_result,
        'promotable_rules': promotable_rules,
        'folds': fold_summaries,
        'decision': {
            'enable_selective_gate': False,
            'production_enabled': False,
            'promotable_rules': promotable_rules,
            'minimum_selective_bets': MIN_SELECTIVE_BETS,
            'requires_prospective_validation': True,
        },
        'warnings': [
            'Temperature is fitted on earlier ledger rows only; rows before 2026-11-06 '
            'still carry older-gate and contaminated-sample caveats.',
            'The ledger has no closing lines, so these results do not yet establish CLV.',
            'Selective rules are pre-registered; any rule added after inspection is exploratory '
            'and cannot be promoted on this sample.',
            'The production selector must stay disabled until the prospective shadow period '
            'and all go-live criteria pass.',
        ] + (
            [
                f'Temperature hit the grid maximum ({temperature_max}) in '
                f'{boundary_hits}/{len(fold_summaries)} folds; the raw probability '
                'scale is effectively uninformative on this sample.'
            ]
            if boundary_hits else []
        ),
    }
