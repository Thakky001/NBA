"""Pre-registered situational O/U subset research.

Subset fishing is the fastest way to manufacture a fake edge: try twenty splits
on 122 bets and one will look profitable.  This module prevents that by fixing
the hypothesis list in advance (see ``docs/ou_hypotheses.md``), evaluating each
subset only on walk-forward out-of-sample folds, and applying a
Benjamini-Hochberg false-discovery-rate correction across the whole family.

Promotion is deliberately brutal:

- at least 200 out-of-sample bets in the subset;
- positive ROI with a 95% bootstrap interval above zero;
- BH-adjusted one-sided p-value below 0.10;
- no post-hoc rule additions.

With the current ledger every subset fails the sample floor.  That is the
correct result, not a bug.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from analysis.backtesting import bootstrap_mean_interval, walk_forward_splits
from analysis.ou_evaluation import ou_games_played, ou_side, ou_tier
from analysis.prediction_tracker import load_ledger

WIN_PAYOUT = 100.0 / 110.0
BREAKEVEN = 100.0 / 210.0
MIN_PROMOTION_BETS = 200
FDR_Q = 0.10
MIN_TRAIN_ROWS = 80
VALIDATION_ROWS = 20
TEST_ROWS = 10

# The pre-registered family.  Each rule names a plausible mechanism *before*
# looking at results.  Adding a rule after inspection makes it exploratory and
# ineligible for promotion.
SUBSET_RULES: Tuple[Dict[str, Any], ...] = (
    {
        'id': 'strong_gp15',
        'label': 'STRONG + both teams >= 15 games',
        'rationale': 'Production gate; the only tier treated as actionable.',
        'fdr_family': True,
    },
    {
        'id': 'strong_under_gp15',
        'label': 'STRONG UNDER + both teams >= 15 games',
        'rationale': 'UNDER showed a positive baseline; test whether it survives the sample gate.',
        'fdr_family': True,
    },
    {
        'id': 'strong_over_gp15',
        'label': 'STRONG OVER + both teams >= 15 games',
        'rationale': 'Symmetric check for an OVER/UNDER asymmetry.',
        'fdr_family': True,
    },
    {
        'id': 'edge_ge_3',
        'label': 'Absolute edge >= 3.0 points',
        'rationale': 'Larger model disagreement should carry more signal than noise.',
        'fdr_family': True,
    },
    {
        'id': 'edge_ge_4',
        'label': 'Absolute edge >= 4.0 points',
        'rationale': 'Test the high-conviction tail separately from the 3-4 band.',
        'fdr_family': True,
    },
    {
        'id': 'line_above_season_3_under',
        'label': 'Line >= season average + 3, model UNDER',
        'rationale': 'Recency-inflated totals may revert; only take the model UNDER.',
        'fdr_family': True,
    },
    {
        'id': 'line_below_season_3_over',
        'label': 'Line <= season average - 3, model OVER',
        'rationale': 'Symmetric reversion check on deflated totals.',
        'fdr_family': True,
    },
    {
        'id': 'fast_vs_fast_over',
        'label': 'FAST_VS_FAST pace matchup, model OVER',
        'rationale': 'Two fast teams should create more possessions and points.',
        'fdr_family': True,
    },
    {
        'id': 'slow_vs_slow_under',
        'label': 'SLOW_VS_SLOW pace matchup, model UNDER',
        'rationale': 'Two slow teams should suppress possessions and points.',
        'fdr_family': True,
    },
    {
        'id': 'injury_impact_le_minus4_under',
        'label': 'Injury impact <= -4, model UNDER',
        'rationale': 'Missing rotation players depress scoring; take the model UNDER.',
        'fdr_family': True,
    },
    {
        'id': 'fatigue_under',
        'label': 'Fatigue flag, model UNDER',
        'rationale': 'Fatigued teams may lose offensive efficiency.',
        'fdr_family': True,
    },
)


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_features(raw: Any) -> Optional[Dict[str, Any]]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def build_subset_frame(ledger: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Build a leakage-safe row frame with the situational attributes."""
    if ledger is None:
        ledger = load_ledger()
    if ledger is None or ledger.empty:
        return pd.DataFrame(), {'rows_in': 0, 'rows_usable': 0, 'reason': 'empty ledger'}

    required = ['date', 'total_line', 'actual_total', 'projected_total', 'ou_recommendation']
    missing = [col for col in required if col not in ledger.columns]
    if missing:
        return pd.DataFrame(), {
            'rows_in': int(len(ledger)),
            'rows_usable': 0,
            'reason': f'missing columns: {missing}',
        }

    skipped = {
        'missing_line': 0,
        'missing_actual': 0,
        'missing_projection': 0,
        'invalid_snapshot': 0,
        'push': 0,
    }
    rows: List[Dict[str, Any]] = []

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
        features = _parse_features(row.get('features_snapshot'))
        if features is None:
            skipped['invalid_snapshot'] += 1
            continue

        side = ou_side(row)
        rec = str(row.get('ou_recommendation', 'NEUTRAL') or 'NEUTRAL').upper()
        if side is None or not ('OVER' in rec or 'UNDER' in rec):
            continue

        result = (
            'WIN'
            if (side == 'OVER' and actual > line) or (side == 'UNDER' and actual < line)
            else 'LOSS'
        )
        gp = ou_games_played(row)
        if gp is None:
            gp = min(
                _to_float(features.get('home_games_played')) or 0,
                _to_float(features.get('away_games_played')) or 0,
            )

        combined_season = _to_float(features.get('combined_avg_season'))
        rows.append({
            'game_id': str(row.get('game_id', '')),
            'date': row.get('date'),
            'total_line': line,
            'actual_total': actual,
            'projected_total': projection,
            'recommendation': rec,
            'tier': ou_tier(row),
            'side': side,
            'edge': _to_float(row.get('ou_edge')),
            'win_prob': _to_float(row.get('ou_win_prob')),
            'games_played': gp,
            'line_minus_season': (
                line - combined_season if combined_season is not None else None
            ),
            'pace_style': str(features.get('matchup_pace_style') or '').upper(),
            'injury_impact': _to_float(features.get('injury_net_total_impact')),
            'fatigue': bool(features.get('fatigue_flag')),
            'result': result,
            'pnl': WIN_PAYOUT if result == 'WIN' else -1.0,
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
        'picks': int(len(frame)),
        'unique_dates': int(frame['date'].nunique()),
    }


def _rule_mask(frame: pd.DataFrame, rule_id: str) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=bool)
    tier = frame['tier']
    side = frame['side']
    edge = pd.to_numeric(frame['edge'], errors='coerce')
    gp = pd.to_numeric(frame['games_played'], errors='coerce')
    if rule_id == 'strong_gp15':
        return (tier == 'STRONG') & (gp >= 15)
    if rule_id == 'strong_under_gp15':
        return (tier == 'STRONG') & (side == 'UNDER') & (gp >= 15)
    if rule_id == 'strong_over_gp15':
        return (tier == 'STRONG') & (side == 'OVER') & (gp >= 15)
    if rule_id == 'edge_ge_3':
        return edge.abs() >= 3.0
    if rule_id == 'edge_ge_4':
        return edge.abs() >= 4.0
    if rule_id == 'line_above_season_3_under':
        line_gap = pd.to_numeric(frame['line_minus_season'], errors='coerce')
        return (side == 'UNDER') & (line_gap >= 3.0)
    if rule_id == 'line_below_season_3_over':
        line_gap = pd.to_numeric(frame['line_minus_season'], errors='coerce')
        return (side == 'OVER') & (line_gap <= -3.0)
    if rule_id == 'fast_vs_fast_over':
        return (frame['pace_style'] == 'FAST_VS_FAST') & (side == 'OVER')
    if rule_id == 'slow_vs_slow_under':
        return (frame['pace_style'] == 'SLOW_VS_SLOW') & (side == 'UNDER')
    if rule_id == 'injury_impact_le_minus4_under':
        impact = pd.to_numeric(frame['injury_impact'], errors='coerce')
        return (side == 'UNDER') & (impact <= -4.0)
    if rule_id == 'fatigue_under':
        return frame['fatigue'] & (side == 'UNDER')
    return pd.Series(False, index=frame.index)


def _binomial_upper_p(wins: int, n: int, p0: float = BREAKEVEN) -> Optional[float]:
    """Exact one-sided p-value for H1: win rate > breakeven."""
    if n <= 0:
        return None
    wins = max(0, min(int(wins), int(n)))
    p0 = min(max(float(p0), 1e-9), 1.0 - 1e-9)
    tail = sum(
        math.comb(n, k) * (p0 ** k) * ((1.0 - p0) ** (n - k))
        for k in range(wins, n + 1)
    )
    return max(0.0, min(1.0, float(tail)))


def benjamini_hochberg(
    p_values: Dict[str, Optional[float]],
    q: float = FDR_Q,
) -> Dict[str, Any]:
    """Benjamini-Hochberg adjustment over a fixed hypothesis family."""
    ordered = sorted(
        ((float(p), key) for key, p in p_values.items() if p is not None),
        key=lambda item: item[0],
    )
    m = len(ordered)
    adjusted: Dict[str, float] = {}
    if m == 0:
        return {'adjusted': adjusted, 'significant': [], 'q': q, 'family_size': 0}

    running = 1.0
    for rank in range(m, 0, -1):
        p, key = ordered[rank - 1]
        running = min(running, p * m / rank)
        adjusted[key] = min(1.0, running)

    return {
        'adjusted': {
            key: round(float(value), 6) for key, value in adjusted.items()
        },
        'significant': sorted(
            key for key, value in adjusted.items() if value <= q
        ),
        'q': q,
        'family_size': m,
    }


def _evaluate_subset(frame: pd.DataFrame) -> Dict[str, Any]:
    n = int(len(frame))
    if n == 0:
        return {'n_picks': 0, 'status': 'NO_PICKS'}
    wins = int((frame['result'] == 'WIN').sum())
    losses = int((frame['result'] == 'LOSS').sum())
    decided = wins + losses
    pnl = frame['pnl'].to_numpy(dtype=float)
    roi = float(pnl.mean()) if n else None
    p_value = _binomial_upper_p(wins, decided)
    roi_ci = bootstrap_mean_interval(pnl.tolist()) if n >= 10 else None
    return {
        'n_picks': n,
        'wins': wins,
        'losses': losses,
        'win_rate': round(wins / decided * 100.0, 2) if decided else None,
        'units': round(float(pnl.sum()), 3),
        'roi_pct': round(roi * 100.0, 2) if roi is not None else None,
        'roi_95ci_pct': (
            [round(roi_ci[0] * 100.0, 2), round(roi_ci[1] * 100.0, 2)]
            if roi_ci else None
        ),
        'p_value_one_sided': round(p_value, 6) if p_value is not None else None,
        'sufficient_sample': n >= MIN_PROMOTION_BETS,
    }


def run_subset_research(
    ledger: Optional[pd.DataFrame] = None,
    min_train: int = MIN_TRAIN_ROWS,
    validation_size: int = VALIDATION_ROWS,
    test_size: int = TEST_ROWS,
) -> Dict[str, Any]:
    """Evaluate the pre-registered subsets on walk-forward test folds only."""
    frame, diagnostics = build_subset_frame(ledger)
    generated_at = datetime.now(timezone.utc).isoformat()

    if frame.empty:
        return {
            'report_version': 'ou-subsets-v1.0',
            'generated_at': generated_at,
            'status': 'INSUFFICIENT_DATA',
            'reason': diagnostics.get('reason', 'no usable picks'),
            'data': diagnostics,
            'rules': {},
            'decision': {'production_enabled': False, 'paper_only': True},
        }

    try:
        folds = walk_forward_splits(
            frame['date'].tolist(),
            min_train=min_train,
            validation_size=validation_size,
            test_size=test_size,
            step=test_size,
        )
    except ValueError as exc:
        return {
            'report_version': 'ou-subsets-v1.0',
            'generated_at': generated_at,
            'status': 'INSUFFICIENT_DATA',
            'reason': str(exc),
            'data': diagnostics,
            'rules': {},
            'decision': {'production_enabled': False, 'paper_only': True},
        }

    if not folds:
        return {
            'report_version': 'ou-subsets-v1.0',
            'generated_at': generated_at,
            'status': 'INSUFFICIENT_DATA',
            'reason': (
                f'not enough rows for min_train={min_train}, '
                f'validation={validation_size}, test={test_size}'
            ),
            'data': diagnostics,
            'rules': {},
            'decision': {'production_enabled': False, 'paper_only': True},
        }

    test_indices: List[int] = []
    for fold in folds:
        test_indices.extend(fold['test'])
    oos = frame.iloc[test_indices].copy()
    oos = oos.sort_values(['date', 'game_id'], kind='stable').reset_index(drop=True)

    rule_results: Dict[str, Any] = {}
    family_p_values: Dict[str, Optional[float]] = {}

    reference = _evaluate_subset(oos)
    reference.update({
        'id': 'all_picks',
        'label': 'All O/U picks (reference, not a hypothesis)',
        'fdr_family': False,
    })

    for rule in SUBSET_RULES:
        mask = _rule_mask(oos, rule['id'])
        result = _evaluate_subset(oos[mask])
        result.update({
            'id': rule['id'],
            'label': rule['label'],
            'rationale': rule['rationale'],
            'fdr_family': bool(rule.get('fdr_family', True)),
        })
        rule_results[rule['id']] = result
        if rule.get('fdr_family', True):
            family_p_values[rule['id']] = result.get('p_value_one_sided')

    bh = benjamini_hochberg(family_p_values, q=FDR_Q)
    for rule_id, adjusted in bh['adjusted'].items():
        rule_results[rule_id]['p_value_bh'] = adjusted
        rule_results[rule_id]['significant_after_bh'] = rule_id in bh['significant']

    promotable = []
    for rule_id, result in rule_results.items():
        if (
            result.get('sufficient_sample')
            and result.get('roi_95ci_pct')
            and result['roi_95ci_pct'][0] > 0
            and result.get('p_value_bh') is not None
            and result['p_value_bh'] <= FDR_Q
        ):
            promotable.append(rule_id)

    if not oos.empty and len(promotable) > 0:
        status = 'CANDIDATE_FOR_SHADOW'
        reason = 'at least one pre-registered subset passed every promotion gate'
    else:
        status = 'KEEP_DISABLED'
        reason = (
            'no pre-registered subset reached 200 out-of-sample bets with a '
            'positive lower ROI bound and a BH-adjusted p-value <= 0.10'
        )

    return {
        'report_version': 'ou-subsets-v1.0',
        'generated_at': generated_at,
        'status': status,
        'reason': reason,
        'data': diagnostics,
        'walk_forward': {
            'min_train': min_train,
            'validation_size': validation_size,
            'test_size': test_size,
            'folds': len(folds),
            'out_of_sample_picks': int(len(oos)),
            'date_range': (
                [str(oos['date'].min().date()), str(oos['date'].max().date())]
                if not oos.empty else None
            ),
        },
        'reference': reference,
        'rules': rule_results,
        'multiple_testing': {
            'method': 'benjamini-hochberg',
            'q': FDR_Q,
            'family_size': bh['family_size'],
            'significant': bh['significant'],
            'pre_registered': [rule['id'] for rule in SUBSET_RULES],
            'note': (
                'The family is fixed in docs/ou_hypotheses.md. Any post-hoc rule '
                'is exploratory and cannot be promoted on this sample.'
            ),
        },
        'decision': {
            'status': status,
            'promotable_rules': promotable,
            'production_enabled': False,
            'paper_only': True,
            'minimum_promotion_bets': MIN_PROMOTION_BETS,
            'requires_prospective_validation': True,
        },
        'warnings': [
            'Subsets are evaluated only on walk-forward test folds; training rows never count.',
            'The ledger has no closing lines, so CLV is not part of subset evaluation.',
            'Rows before 2026-11-06 used older gates and contaminated games-played values.',
            'With fewer than 200 bets per subset, no result can be promoted regardless of ROI.',
            'Production betting remains paused; this report never enables a bet.',
        ],
    }
