"""Replay Backtest & Calibration Evaluation Module (Phase 3).

Replays historical games in predictions_ledger.csv as-of date using
the modernized Phase 1 & Phase 2 pipeline:
- Current line anchor (Phase 2.4)
- Dynamic early-season market weight and sigma (Phase 1.2)
- Unified decision gate with DQ veto and extreme edge soft cap (Phase 2.1 & 2.5)
- Full metrics: MAE, ATS Win Rate, ROI after vig, Calibration Error, Max Drawdown
"""

import os
import sys
import json
import math
import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import RAW_DIR
from analysis.recommendation import decide_pick, GATE_VERSION
from analysis.betting_math import (
    calculate_spread_probabilities,
    dynamic_early_season_market_weight,
    dynamic_early_season_sigma,
    market_anchored_projection
)
from analysis.backtesting import calibration_bins, regression_metrics, probability_metrics

logger = logging.getLogger('replay_backtest')


def calculate_max_drawdown(pnl_series: List[float]) -> float:
    """Compute maximum drawdown in units from a cumulative PnL series."""
    if not pnl_series:
        return 0.0
    cum = np.cumsum(pnl_series)
    peak = np.maximum.accumulate(cum)
    drawdowns = peak - cum
    return float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0


def run_replay_backtest(ledger_path: Optional[str] = None) -> Dict[str, Any]:
    """Execute chronological replay backtest on predictions ledger."""
    path = ledger_path or os.path.join(RAW_DIR, 'predictions_ledger.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Ledger file not found: {path}")

    df = pd.read_csv(path)
    # Filter to graded games with final scores
    graded = df[(df['status'] == 'GRADED') & (df['actual_margin'].notna()) & (df['actual_total'].notna())].copy()
    graded.sort_values(by='date', inplace=True)
    graded.reset_index(drop=True, inplace=True)

    sp_records = []
    ou_records = []

    for idx, row in graded.iterrows():
        gid = str(row['game_id'])
        date_str = str(row['date'])
        home = str(row['home_team'])
        away = str(row['away_team'])
        act_margin = float(row['actual_margin'])
        act_total = float(row['actual_total'])

        sp_line = float(row['spread_line']) if pd.notna(row.get('spread_line')) else None
        ou_line = float(row['total_line']) if pd.notna(row.get('total_line')) else None

        # Parse features snapshot
        features = {}
        if pd.notna(row.get('features_snapshot')):
            try:
                features = json.loads(str(row['features_snapshot']))
            except Exception:
                features = {}

        min_games = min(features.get('home_games_played', 10), features.get('away_games_played', 10))
        is_early = min_games < 20

        # --- Replay Spread Pick ---
        if sp_line is not None:
            # Raw projected margin
            raw_margin = float(row.get('projected_margin')) if pd.notna(row.get('projected_margin')) else 0.0
            eff_weight = dynamic_early_season_market_weight(min_games)
            market_exp_margin = -sp_line
            anchored_margin = market_anchored_projection(raw_margin, market_exp_margin, market_weight=eff_weight)
            cover_edge = round(anchored_margin + sp_line, 2)

            sigma = dynamic_early_season_sigma(12.5, min_games)
            probs = calculate_spread_probabilities(sp_line, anchored_margin, sigma)
            win_prob_home = probs['win']

            decision = decide_pick(
                market_type='spread',
                current_line=sp_line,
                projected_val=anchored_margin,
                edge=cover_edge,
                fair_prob=win_prob_home,
                home_team=home,
                away_team=away,
                data_quality_flags=features,
                is_early_season=is_early,
                push_prob=probs.get('push', 0.0)
            )

            rec = decision['recommendation']
            if rec != 'NEUTRAL':
                # Grade outcome
                # Spread cover formula: actual_margin + spread_line
                ats_diff = act_margin + sp_line
                if abs(ats_diff) < 1e-6:
                    res = 'PUSH'
                    pnl = 0.0
                elif 'HOME' in rec:
                    if ats_diff > 0:
                        res = 'WIN'
                        pnl = 0.909  # -110 odds
                    else:
                        res = 'LOSS'
                        pnl = -1.0
                elif 'AWAY' in rec:
                    if ats_diff < 0:
                        res = 'WIN'
                        pnl = 0.909
                    else:
                        res = 'LOSS'
                        pnl = -1.0
                else:
                    res = 'SKIPPED'
                    pnl = 0.0

                sp_records.append({
                    'game_id': gid,
                    'date': date_str,
                    'market': 'SPREAD',
                    'tier': 'STRONG' if 'STRONG' in rec else 'LEAN',
                    'rec': rec,
                    'pick_side': decision['pick_side'],
                    'pick_prob': decision['pick_prob'],
                    'edge': cover_edge,
                    'line': sp_line,
                    'actual': act_margin,
                    'result': res,
                    'pnl': pnl,
                    'binary_win': 1 if res == 'WIN' else (0 if res == 'LOSS' else None)
                })

        # --- Replay O/U Pick ---
        if ou_line is not None and ou_line > 0:
            raw_total = float(row.get('projected_total')) if pd.notna(row.get('projected_total')) else ou_line
            eff_weight = dynamic_early_season_market_weight(min_games)
            anchored_total = market_anchored_projection(raw_total, ou_line, market_weight=eff_weight)
            ou_edge = round(anchored_total - ou_line, 2)

            sigma_ou = dynamic_early_season_sigma(12.5, min_games)
            probs_ou = calculate_spread_probabilities(-ou_line, anchored_total, sigma_ou)
            win_prob_over = probs_ou['win']

            decision_ou = decide_pick(
                market_type='total',
                current_line=ou_line,
                projected_val=anchored_total,
                edge=ou_edge,
                fair_prob=win_prob_over,
                home_team=home,
                away_team=away,
                data_quality_flags=features,
                is_early_season=is_early,
                push_prob=probs_ou.get('push', 0.0)
            )

            rec_ou = decision_ou['recommendation']
            if rec_ou != 'NEUTRAL':
                ou_diff = act_total - ou_line
                if abs(ou_diff) < 1e-6:
                    res = 'PUSH'
                    pnl = 0.0
                elif 'OVER' in rec_ou:
                    if ou_diff > 0:
                        res = 'WIN'
                        pnl = 0.909
                    else:
                        res = 'LOSS'
                        pnl = -1.0
                elif 'UNDER' in rec_ou:
                    if ou_diff < 0:
                        res = 'WIN'
                        pnl = 0.909
                    else:
                        res = 'LOSS'
                        pnl = -1.0
                else:
                    res = 'SKIPPED'
                    pnl = 0.0

                ou_records.append({
                    'game_id': gid,
                    'date': date_str,
                    'market': 'TOTAL',
                    'tier': 'STRONG' if 'STRONG' in rec_ou else 'LEAN',
                    'rec': rec_ou,
                    'pick_side': decision_ou['pick_side'],
                    'pick_prob': decision_ou['pick_prob'],
                    'edge': ou_edge,
                    'line': ou_line,
                    'actual': act_total,
                    'result': res,
                    'pnl': pnl,
                    'binary_win': 1 if res == 'WIN' else (0 if res == 'LOSS' else None)
                })

    all_records = sp_records + ou_records
    # Sort chronologically
    all_records.sort(key=lambda x: x['date'])

    def evaluate_slice(records: List[Dict[str, Any]]) -> Dict[str, Any]:
        valid = [r for r in records if r['result'] in ('WIN', 'LOSS')]
        n_total = len(records)
        n_graded = len(valid)
        n_wins = sum(1 for r in valid if r['result'] == 'WIN')
        n_losses = sum(1 for r in valid if r['result'] == 'LOSS')
        n_pushes = sum(1 for r in records if r['result'] == 'PUSH')
        win_rate = (n_wins / n_graded * 100.0) if n_graded > 0 else 0.0
        pnl_units = sum(r['pnl'] for r in records)
        roi_pct = (pnl_units / n_graded * 100.0) if n_graded > 0 else 0.0
        max_dd = calculate_max_drawdown([r['pnl'] for r in records])

        # Calibration
        brier = 0.0
        calib_error = 0.0
        if valid:
            probs = [r['pick_prob'] for r in valid]
            outcomes = [r['binary_win'] for r in valid]
            prob_metrics = probability_metrics(probs, outcomes)
            brier = prob_metrics['brier']
            bins = calibration_bins(probs, outcomes, bins=5)
            # Weighted average calibration error: sum |p - y| * n / N
            total_n = sum(b['count'] for b in bins)
            calib_error = sum(abs(b['mean_probability'] - b['observed_rate']) * b['count'] for b in bins) / total_n if total_n > 0 else 0.0

        return {
            'total_picks': n_total,
            'graded_picks': n_graded,
            'wins': n_wins,
            'losses': n_losses,
            'pushes': n_pushes,
            'win_rate': round(win_rate, 2),
            'pnl_units': round(pnl_units, 2),
            'roi_pct': round(roi_pct, 2),
            'max_drawdown_units': round(max_dd, 2),
            'brier_score': round(brier, 4),
            'calibration_error_pct': round(calib_error * 100.0, 2)
        }

    overall = evaluate_slice(all_records)
    strong_slice = evaluate_slice([r for r in all_records if r['tier'] == 'STRONG'])
    lean_slice = evaluate_slice([r for r in all_records if r['tier'] == 'LEAN'])
    spread_slice = evaluate_slice(sp_records)
    ou_slice = evaluate_slice(ou_records)

    # Go-Live Verification Criteria (Phase 3.4)
    # 1. Total picks >= 100
    # 2. ROI after vig > +2%
    # 3. Calibration error < 5%
    # 4. Max drawdown < 10 units
    go_live_checks = {
        'criterion_1_min_100_picks': overall['total_picks'] >= 100,
        'criterion_2_roi_above_2pct': overall['roi_pct'] > 2.0,
        'criterion_3_calibration_under_5pct': overall['calibration_error_pct'] < 5.0,
        'criterion_4_max_dd_under_10u': overall['max_drawdown_units'] < 10.0
    }
    go_live_checks['all_passed'] = all(go_live_checks.values())

    return {
        'total_games_replayed': len(graded),
        'overall': overall,
        'strong_tier': strong_slice,
        'lean_tier': lean_slice,
        'spread_market': spread_slice,
        'total_market': ou_slice,
        'go_live_checks': go_live_checks,
        'records': all_records
    }


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    res = run_replay_backtest()
    print("\n" + "=" * 60)
    print("REPLAY BACKTEST RESULTS (MODERNIZED PHASE 1 & 2 PIPELINE)")
    print("=" * 60)
    print(f"Total Games Evaluated: {res['total_games_replayed']}")
    print(f"Total Actionable Picks: {res['overall']['total_picks']}")
    print(f"Graded Picks: {res['overall']['graded_picks']} (W: {res['overall']['wins']}, L: {res['overall']['losses']}, P: {res['overall']['pushes']})")
    print(f"Win Rate: {res['overall']['win_rate']}%")
    print(f"PnL: {res['overall']['pnl_units']:+.2f} Units")
    print(f"ROI (after -110 vig): {res['overall']['roi_pct']:+.2f}%")
    print(f"Max Drawdown: {res['overall']['max_drawdown_units']:.2f} Units")
    print(f"Calibration Error: {res['overall']['calibration_error_pct']:.2f}%")
    print("-" * 60)
    print("BREAKDOWN BY TIER:")
    print(f"  STRONG Tier: {res['strong_tier']['total_picks']} picks | Win Rate: {res['strong_tier']['win_rate']}% | ROI: {res['strong_tier']['roi_pct']:+.2f}% | Max DD: {res['strong_tier']['max_drawdown_units']:.2f}u")
    print(f"  LEAN Tier:   {res['lean_tier']['total_picks']} picks | Win Rate: {res['lean_tier']['win_rate']}% | ROI: {res['lean_tier']['roi_pct']:+.2f}% | Max DD: {res['lean_tier']['max_drawdown_units']:.2f}u")
    print("-" * 60)
    print("BREAKDOWN BY MARKET:")
    print(f"  Spread: {res['spread_market']['total_picks']} picks | Win Rate: {res['spread_market']['win_rate']}% | ROI: {res['spread_market']['roi_pct']:+.2f}%")
    print(f"  Total:  {res['total_market']['total_picks']} picks | Win Rate: {res['total_market']['win_rate']}% | ROI: {res['total_market']['roi_pct']:+.2f}%")
    print("-" * 60)
    print("GO-LIVE GATES VERIFICATION (PHASE 3.4):")
    for k, v in res['go_live_checks'].items():
        status_icon = "✅ PASS" if v else "❌ FAIL"
        print(f"  {k}: {status_icon}")
    print("=" * 60)
