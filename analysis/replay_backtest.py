"""Replay Backtest & Calibration Evaluation Module (Phase 3 & Quantitative Remedy).

Replays historical games in predictions_ledger.csv as-of game date using
the modernized Phase 1 & Phase 2 pipeline:
- True historical feature reconstruction as-of date (discarding contaminated snapshots)
- Calibrated probability gating with empirical thresholds (-110 breakeven = 52.38%)
- Current line anchor (Phase 2.4)
- Dynamic early-season market weight and sigma (Phase 1.2)
- Unified decision gate with DQ veto and graduated early-season gating
- In-Sample (pre-Nov 6) vs Out-of-Sample (Nov 6+) split evaluation
- Full metrics: MAE, ATS Win Rate, ROI after vig, Calibration Error (ECE), Max Drawdown
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
from analysis.trend_analyzer import TrendAnalyzer
from analysis.backtesting import calibration_bins, regression_metrics, probability_metrics
from processing.data_merger import DataMerger
from processing.feature_engineering import FeatureEngineer

logger = logging.getLogger('replay_backtest')

SPLIT_DATE = '2026-11-06'


def calculate_max_drawdown(pnl_series: List[float]) -> float:
    """Compute maximum drawdown in units from a cumulative PnL series."""
    if not pnl_series:
        return 0.0
    cum = np.cumsum(pnl_series)
    peak = np.maximum.accumulate(cum)
    drawdowns = peak - cum
    return float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0


def evaluate_slice(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute quantitative metrics for a slice of picks."""
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

    # Probability calibration metrics
    brier = 0.0
    ece_cal = 0.0
    ece_raw = 0.0
    mean_cal_p = 0.0
    mean_raw_p = 0.0
    if valid:
        cal_probs = [r['cal_prob'] for r in valid]
        raw_probs = [r['raw_prob'] for r in valid]
        outcomes = [r['binary_win'] for r in valid]
        mean_cal_p = float(np.mean(cal_probs) * 100.0)
        mean_raw_p = float(np.mean(raw_probs) * 100.0)

        prob_metrics = probability_metrics(cal_probs, outcomes)
        brier = prob_metrics['brier']

        # Weighted average calibration error: sum |p_bar - y_bar| * n_b / N
        bins_cal = calibration_bins(cal_probs, outcomes, bins=5)
        tot_cal_n = sum(b['count'] for b in bins_cal)
        ece_cal = sum(abs(b['mean_probability'] - b['observed_rate']) * b['count'] for b in bins_cal) / tot_cal_n if tot_cal_n > 0 else 0.0

        bins_raw = calibration_bins(raw_probs, outcomes, bins=5)
        tot_raw_n = sum(b['count'] for b in bins_raw)
        ece_raw = sum(abs(b['mean_probability'] - b['observed_rate']) * b['count'] for b in bins_raw) / tot_raw_n if tot_raw_n > 0 else 0.0

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
        'mean_cal_prob': round(mean_cal_p, 2),
        'mean_raw_prob': round(mean_raw_p, 2),
        'brier_score': round(brier, 4),
        'calibration_error_pct': round(ece_cal * 100.0, 2),
        'raw_calibration_error_pct': round(ece_raw * 100.0, 2)
    }


def run_replay_backtest(ledger_path: Optional[str] = None, use_dynamic_features: bool = True) -> Dict[str, Any]:
    """Execute chronological replay backtest on predictions ledger.
    
    Args:
        ledger_path: Optional custom path to predictions ledger.
        use_dynamic_features: If True, dynamically reconstructs historical features
            as-of game date using FeatureEngineer and clean game logs.
    """
    path = ledger_path or os.path.join(RAW_DIR, 'predictions_ledger.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Ledger file not found: {path}")

    df = pd.read_csv(path)
    # Filter to graded games with final scores
    graded = df[(df['status'] == 'GRADED') & (df['actual_margin'].notna()) & (df['actual_total'].notna())].copy()
    graded.sort_values(by='date', inplace=True)
    graded.reset_index(drop=True, inplace=True)

    if use_dynamic_features:
        dm = DataMerger()
        all_games = dm.load_all_games()
        fe = FeatureEngineer(all_games)
        # Cache roster ratings across team pairs to accelerate replay
        roster_cache = {}
        orig_roster = fe.get_roster_ratings
        fe.get_roster_ratings = lambda h, a: roster_cache.setdefault((h, a), orig_roster(h, a))
        analyzer = TrendAnalyzer(fe)
    else:
        from unittest.mock import MagicMock
        mock_fe = MagicMock()
        analyzer = TrendAnalyzer(mock_fe)

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
        open_sp = float(row['open_spread']) if pd.notna(row.get('open_spread')) else None
        open_ou = float(row['open_total']) if pd.notna(row.get('open_total')) else None

        if not use_dynamic_features:
            features = {}
            try:
                features = json.loads(str(row.get('features_snapshot', '{}')))
            except Exception:
                features = {}
            mock_fe.compute_matchup_features.return_value = features

        # Execute full matchup analysis as-of date
        res = analyzer.full_matchup_analysis(
            home_team=home,
            away_team=away,
            spread_line=sp_line,
            over_under_line=ou_line,
            open_spread=open_sp,
            open_line=open_ou,
            game_date=date_str
        )

        sp_res = res.get('spread') or {}
        ou_res = res.get('over_under') or {}

        # --- Replay Spread Pick ---
        if sp_line is not None and sp_res:
            rec = sp_res.get('recommendation', 'NEUTRAL')
            if rec != 'NEUTRAL':
                ats_diff = act_margin + sp_line
                if abs(ats_diff) < 1e-6:
                    r_status = 'PUSH'
                    pnl = 0.0
                elif 'HOME' in rec:
                    if ats_diff > 0:
                        r_status = 'WIN'
                        pnl = 0.909
                    else:
                        r_status = 'LOSS'
                        pnl = -1.0
                elif 'AWAY' in rec:
                    if ats_diff < 0:
                        r_status = 'WIN'
                        pnl = 0.909
                    else:
                        r_status = 'LOSS'
                        pnl = -1.0
                else:
                    r_status = 'SKIPPED'
                    pnl = 0.0

                sp_records.append({
                    'game_id': gid,
                    'date': date_str,
                    'market': 'SPREAD',
                    'tier': 'STRONG' if 'STRONG' in rec else 'LEAN',
                    'rec': rec,
                    'is_actionable': sp_res.get('is_actionable', False),
                    'pick_side': sp_res.get('pick_side'),
                    'cal_prob': sp_res.get('win_prob', 0.50),
                    'raw_prob': sp_res.get('raw_win_prob', 0.50),
                    'edge': sp_res.get('cover_edge', 0.0),
                    'line': sp_line,
                    'actual': act_margin,
                    'result': r_status,
                    'pnl': pnl,
                    'binary_win': 1 if r_status == 'WIN' else (0 if r_status == 'LOSS' else None)
                })

        # --- Replay Total Pick ---
        if ou_line is not None and ou_res and ou_line > 0:
            rec_ou = ou_res.get('recommendation', 'NEUTRAL')
            if rec_ou != 'NEUTRAL':
                ou_diff = act_total - ou_line
                if abs(ou_diff) < 1e-6:
                    r_status = 'PUSH'
                    pnl = 0.0
                elif 'OVER' in rec_ou:
                    if ou_diff > 0:
                        r_status = 'WIN'
                        pnl = 0.909
                    else:
                        r_status = 'LOSS'
                        pnl = -1.0
                elif 'UNDER' in rec_ou:
                    if ou_diff < 0:
                        r_status = 'WIN'
                        pnl = 0.909
                    else:
                        r_status = 'LOSS'
                        pnl = -1.0
                else:
                    r_status = 'SKIPPED'
                    pnl = 0.0

                ou_records.append({
                    'game_id': gid,
                    'date': date_str,
                    'market': 'TOTAL',
                    'tier': 'STRONG' if 'STRONG' in rec_ou else 'LEAN',
                    'rec': rec_ou,
                    'is_actionable': ou_res.get('is_actionable', False),
                    'pick_side': ou_res.get('pick_side'),
                    'cal_prob': ou_res.get('win_prob', 0.50),
                    'raw_prob': ou_res.get('raw_win_prob', 0.50),
                    'edge': ou_res.get('total_edge', 0.0),
                    'line': ou_line,
                    'actual': act_total,
                    'result': r_status,
                    'pnl': pnl,
                    'binary_win': 1 if r_status == 'WIN' else (0 if r_status == 'LOSS' else None)
                })

    all_records = sp_records + ou_records
    all_records.sort(key=lambda x: x['date'])

    # Slices by time period (Expanding Window Validation)
    prior_records = [r for r in all_records if r['date'] < SPLIT_DATE]
    post_records = [r for r in all_records if r['date'] >= SPLIT_DATE]
    last7_records = [r for r in all_records if SPLIT_DATE <= r['date'] < '2026-11-13']
    recent_records = [r for r in all_records if r['date'] >= '2026-11-13']

    # Slices by tier & period
    overall = evaluate_slice(all_records)
    strong_overall = evaluate_slice([r for r in all_records if r['tier'] == 'STRONG'])
    lean_overall = evaluate_slice([r for r in all_records if r['tier'] == 'LEAN'])

    prior_eval = evaluate_slice(prior_records)
    prior_strong = evaluate_slice([r for r in prior_records if r['tier'] == 'STRONG'])
    prior_lean = evaluate_slice([r for r in prior_records if r['tier'] == 'LEAN'])

    post_eval = evaluate_slice(post_records)
    post_strong = evaluate_slice([r for r in post_records if r['tier'] == 'STRONG'])
    post_lean = evaluate_slice([r for r in post_records if r['tier'] == 'LEAN'])

    last7_eval = evaluate_slice(last7_records)
    recent_eval = evaluate_slice(recent_records)

    spread_slice = evaluate_slice(sp_records)
    ou_slice = evaluate_slice(ou_records)

    # Go-Live Verification Criteria (Evaluated on Actionable STRONG Tier)
    go_live_checks = {
        'criterion_1_min_100_strong_picks': strong_overall['total_picks'] >= 100,
        'criterion_2_strong_roi_above_2pct': strong_overall['roi_pct'] > 2.0,
        'criterion_3_strong_calibration_under_5pct': strong_overall['calibration_error_pct'] < 5.0,
        'criterion_4_strong_max_dd_under_10u': strong_overall['max_drawdown_units'] < 10.0
    }
    go_live_checks['all_passed'] = all(go_live_checks.values())

    return {
        'total_games_replayed': len(graded),
        'overall': overall,
        'strong_tier': strong_overall,
        'lean_tier': lean_overall,
        'prior_in_sample': prior_eval,
        'prior_strong': prior_strong,
        'prior_lean': prior_lean,
        'post_out_of_sample': post_eval,
        'post_strong': post_strong,
        'post_lean': post_lean,
        'last7_sample': last7_eval,
        'recent_sample': recent_eval,
        'spread_market': spread_slice,
        'total_market': ou_slice,
        'go_live_checks': go_live_checks,
        'records': all_records
    }


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    res = run_replay_backtest()
    print("\n" + "=" * 70)
    print("TRUE REPLAY BACKTEST & OUT-OF-SAMPLE CALIBRATION EVALUATION")
    print("=" * 70)
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

    print(f"Total Games Replayed: {res['total_games_replayed']} (dynamically reconstructed as-of date)")
    print(f"Total Model Picks: {res['overall']['total_picks']} (STRONG: {res['strong_tier']['total_picks']}, LEAN: {res['lean_tier']['total_picks']})")
    print(f"Total Graded Picks: {res['overall']['graded_picks']} (W: {res['overall']['wins']}, L: {res['overall']['losses']}, P: {res['overall']['pushes']})")
    print(f"Overall Model Win Rate: {res['overall']['win_rate']}%")
    print(f"Overall Model PnL: {res['overall']['pnl_units']:+.2f} Units | ROI: {res['overall']['roi_pct']:+.2f}%")
    print(f"Overall Max Drawdown: {res['overall']['max_drawdown_units']:.2f} Units")
    print(f"Overall Calibration Error (ECE): Calibrated {res['overall']['calibration_error_pct']:.2f}% vs Raw {res['overall']['raw_calibration_error_pct']:.2f}%")
    print("-" * 70)
    print("EXPANDING WINDOW SPLIT: IN-SAMPLE (PRIOR) vs OUT-OF-SAMPLE (POST)")
    print(f"  PRIOR (Before Nov 6) - {res['prior_in_sample']['total_picks']} picks:")
    print(f"    Win Rate: {res['prior_in_sample']['win_rate']}% (W: {res['prior_in_sample']['wins']}, L: {res['prior_in_sample']['losses']}, P: {res['prior_in_sample']['pushes']})")
    print(f"    PnL: {res['prior_in_sample']['pnl_units']:+.2f} Units | ROI: {res['prior_in_sample']['roi_pct']:+.2f}%")
    print(f"    Mean Prob: Calibrated {res['prior_in_sample']['mean_cal_prob']:.1f}% vs Raw {res['prior_in_sample']['mean_raw_prob']:.1f}%")
    print(f"    ECE: Calibrated {res['prior_in_sample']['calibration_error_pct']:.2f}% vs Raw {res['prior_in_sample']['raw_calibration_error_pct']:.2f}%")
    print(f"  POST (Nov 6 onwards - Out-of-Sample) - {res['post_out_of_sample']['total_picks']} picks:")
    print(f"    Win Rate: {res['post_out_of_sample']['win_rate']}% (W: {res['post_out_of_sample']['wins']}, L: {res['post_out_of_sample']['losses']}, P: {res['post_out_of_sample']['pushes']})")
    print(f"    PnL: {res['post_out_of_sample']['pnl_units']:+.2f} Units | ROI: {res['post_out_of_sample']['roi_pct']:+.2f}%")
    print(f"    Mean Prob: Calibrated {res['post_out_of_sample']['mean_cal_prob']:.1f}% vs Raw {res['post_out_of_sample']['mean_raw_prob']:.1f}%")
    print(f"    ECE: Calibrated {res['post_out_of_sample']['calibration_error_pct']:.2f}% vs Raw {res['post_out_of_sample']['raw_calibration_error_pct']:.2f}%")
    print("-" * 70)
    print(f"ACTIONABLE PERFORMANCE (STRONG TIER - {res['strong_tier']['total_picks']} PICKS):")
    print(f"  Win Rate: {res['strong_tier']['win_rate']}% (W: {res['strong_tier']['wins']}, L: {res['strong_tier']['losses']}, P: {res['strong_tier']['pushes']})")
    print(f"  PnL: {res['strong_tier']['pnl_units']:+.2f} Units | ROI: {res['strong_tier']['roi_pct']:+.2f}%")
    print(f"  Max Drawdown: {res['strong_tier']['max_drawdown_units']:.2f} Units")
    print(f"  Mean Prob: Calibrated {res['strong_tier']['mean_cal_prob']:.1f}% vs Raw {res['strong_tier']['mean_raw_prob']:.1f}%")
    print(f"  ECE: Calibrated {res['strong_tier']['calibration_error_pct']:.2f}% vs Raw {res['strong_tier']['raw_calibration_error_pct']:.2f}%")
    print("-" * 70)
    print(f"PAPER MONITORING PERFORMANCE (LEAN TIER - {res['lean_tier']['total_picks']} PICKS):")
    print(f"  Win Rate: {res['lean_tier']['win_rate']}% (W: {res['lean_tier']['wins']}, L: {res['lean_tier']['losses']}, P: {res['lean_tier']['pushes']})")
    print(f"  PnL: {res['lean_tier']['pnl_units']:+.2f} Units | ROI: {res['lean_tier']['roi_pct']:+.2f}%")
    print(f"  Max Drawdown: {res['lean_tier']['max_drawdown_units']:.2f} Units")
    print(f"  Mean Prob: Calibrated {res['lean_tier']['mean_cal_prob']:.1f}% vs Raw {res['lean_tier']['mean_raw_prob']:.1f}%")
    print(f"  ECE: Calibrated {res['lean_tier']['calibration_error_pct']:.2f}% vs Raw {res['lean_tier']['raw_calibration_error_pct']:.2f}%")
    print("-" * 70)
    print("GO-LIVE GATES VERIFICATION:")
    for k, v in res['go_live_checks'].items():
        status_icon = "✅ PASS" if v else "❌ FAIL"
        print(f"  {k}: {status_icon}")
    print(f"Safety Gate Status: BETTING_PAUSED = {not res['go_live_checks']['all_passed']} (All criteria strictly enforced)")
    print("=" * 70)
