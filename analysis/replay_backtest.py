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
from unittest.mock import MagicMock
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import RAW_DIR
from analysis.trend_analyzer import TrendAnalyzer
from analysis.betting_math import dynamic_early_season_market_weight
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
    # Filter to graded games with final scores and features snapshot
    graded = df[(df['status'] == 'GRADED') & (df['actual_margin'].notna()) & (df['actual_total'].notna()) & (df['features_snapshot'].notna())].copy()
    graded.sort_values(by='date', inplace=True)
    graded.reset_index(drop=True, inplace=True)

    sp_records = []
    ou_records = []

    mock_fe = MagicMock()
    analyzer = TrendAnalyzer(mock_fe)

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
        try:
            features = json.loads(str(row['features_snapshot']))
        except Exception:
            features = {}

        mock_fe.compute_matchup_features.return_value = features
        min_games = min(features.get('home_games_played', 10), features.get('away_games_played', 10))
        eff_weight = dynamic_early_season_market_weight(min_games)

        # --- Replay Spread Pick (Raw Math with Single Anchoring) ---
        if sp_line is not None:
            sp_res = analyzer.analyze_spread(home, away, spread_line=sp_line, market_weight=eff_weight)
            rec = sp_res.get('recommendation', 'NEUTRAL')
            if rec != 'NEUTRAL':
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
                    'pick_side': sp_res.get('pick_side'),
                    'pick_prob': sp_res.get('win_prob', 0.5),
                    'edge': sp_res.get('cover_edge', 0.0),
                    'line': sp_line,
                    'actual': act_margin,
                    'result': res,
                    'pnl': pnl,
                    'binary_win': 1 if res == 'WIN' else (0 if res == 'LOSS' else None)
                })

        # --- Replay O/U Pick (Raw Math with Single Anchoring) ---
        if ou_line is not None and ou_line > 0:
            ou_res = analyzer.analyze_over_under(home, away, over_under_line=ou_line, market_weight=eff_weight)
            rec_ou = ou_res.get('recommendation', 'NEUTRAL')
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
                    'pick_side': ou_res.get('pick_side'),
                    'pick_prob': ou_res.get('win_prob', 0.5),
                    'edge': ou_res.get('total_edge', 0.0),
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

    # Go-Live Verification Criteria (Phase 3.4 - Evaluated strictly on Actionable STRONG Tier)
    # 1. Total Actionable picks >= 100
    # 2. ROI after vig > +2%
    # 3. Calibration error < 5%
    # 4. Max drawdown < 10 units
    go_live_checks = {
        'criterion_1_min_100_strong_picks': strong_slice['total_picks'] >= 100,
        'criterion_2_strong_roi_above_2pct': strong_slice['roi_pct'] > 2.0,
        'criterion_3_strong_calibration_under_5pct': strong_slice['calibration_error_pct'] < 5.0,
        'criterion_4_strong_max_dd_under_10u': strong_slice['max_drawdown_units'] < 10.0
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
    # Check historical ledger baseline
    try:
        from analysis.prediction_tracker import load_ledger
        raw_ledger = load_ledger()
        raw_sp_strong = raw_ledger['sp_recommendation'].astype(str).str.contains('STRONG').sum()
        raw_ou_strong = raw_ledger['ou_recommendation'].astype(str).str.contains('STRONG').sum()
        raw_tot_strong = raw_sp_strong + raw_ou_strong
        print(f"Historical Ledger Baseline: {len(raw_ledger)} games | {raw_tot_strong} Recorded STRONG picks (Spread: {raw_sp_strong}, Total: {raw_ou_strong})")
    except Exception:
        pass
    print(f"Total Games Replayed: {res['total_games_replayed']} (with valid feature snapshots)")
    print(f"Rigorous Model Picks: {res['overall']['total_picks']} (Actionable STRONG: {res['strong_tier']['total_picks']}, Paper LEAN: {res['lean_tier']['total_picks']})")
    print(f"Total Graded Picks: {res['overall']['graded_picks']} (W: {res['overall']['wins']}, L: {res['overall']['losses']}, P: {res['overall']['pushes']})")
    print(f"Overall Model Win Rate: {res['overall']['win_rate']}%")
    print(f"Overall Model PnL: {res['overall']['pnl_units']:+.2f} Units")
    print(f"Overall Model ROI (after -110 vig): {res['overall']['roi_pct']:+.2f}%")
    print(f"Overall Max Drawdown: {res['overall']['max_drawdown_units']:.2f} Units")
    print(f"Overall Calibration Error: {res['overall']['calibration_error_pct']:.2f}%")
    print("-" * 60)
    print(f"ACTIONABLE PERFORMANCE (STRONG TIER - {res['strong_tier']['total_picks']} PICKS):")
    print(f"  Win Rate: {res['strong_tier']['win_rate']}% (W: {res['strong_tier']['wins']}, L: {res['strong_tier']['losses']}, P: {res['strong_tier']['pushes']})")
    print(f"  PnL: {res['strong_tier']['pnl_units']:+.2f} Units | ROI: {res['strong_tier']['roi_pct']:+.2f}%")
    print(f"  Max Drawdown: {res['strong_tier']['max_drawdown_units']:.2f} Units | Calibration Error: {res['strong_tier']['calibration_error_pct']:.2f}%")
    print("-" * 60)
    print(f"PAPER MONITORING PERFORMANCE (LEAN TIER - {res['lean_tier']['total_picks']} PICKS):")
    print(f"  Win Rate: {res['lean_tier']['win_rate']}% (W: {res['lean_tier']['wins']}, L: {res['lean_tier']['losses']}, P: {res['lean_tier']['pushes']})")
    print(f"  PnL: {res['lean_tier']['pnl_units']:+.2f} Units | ROI: {res['lean_tier']['roi_pct']:+.2f}%")
    print(f"  Max Drawdown: {res['lean_tier']['max_drawdown_units']:.2f} Units | Calibration Error: {res['lean_tier']['calibration_error_pct']:.2f}%")
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
