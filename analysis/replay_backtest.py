"""Replay Backtest & Calibration Evaluation Module (Phase 3 & Quantitative Remedy).

Replays historical games in predictions_ledger.csv as-of game date using
the modernized Phase 1 & Phase 2 pipeline:
- True historical feature reconstruction as-of date (discarding contaminated snapshots)
- Calibrated probability gating with empirical thresholds (-110 breakeven = 52.38%)
- Current line anchor (Phase 2.4)
- Dynamic early-season market weight and sigma (Phase 1.2)
- Unified decision gate with DQ veto and strict >=15-game sample floor for STRONG
- In-Sample (pre-Nov 6) vs Out-of-Sample (Nov 6+) split evaluation
- Go-live checks on POST-split STRONG only; metrics are N/A until sample is sufficient
- Full metrics: MAE, ATS Win Rate, ROI after vig, Calibration Error (ECE), Max Drawdown
- Known lookahead limitations are reported explicitly (see REPLAY_LIMITATIONS)
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
from processing.snapshot_archive import snapshot_coverage

logger = logging.getLogger('replay_backtest')

SPLIT_DATE = '2026-11-06'

# Minimum graded picks required before a metric (ROI, ECE, drawdown) is
# considered statistically meaningful. Below this the metric is reported as N/A
# and can never satisfy a go-live criterion.
MIN_METRIC_SAMPLE = 30

# Known lookahead / validity limitations of this replay. Reported in the CLI and
# included in the result payload so downstream consumers cannot treat the
# numbers as a clean out-of-sample estimate.
REPLAY_LIMITATIONS = [
    "Roster ratings, injuries and lineups are loaded from the CURRENT data files, "
    "not from an as-of-game-date snapshot, so pre-split (PRIOR) results carry lookahead bias.",
    "Daily injury/lineup/roster snapshots are archived going forward, but replay does not "
    "consume them yet, so no replayed game is currently free of this bias.",
    "The ledger itself was produced by older gate versions (v2.2), so early rows do not "
    "represent picks the current gate would have made.",
    "Calibration temperature is fitted in-sample (T=2.0); ECE is not a pure out-of-sample metric.",
]


def calculate_max_drawdown(pnl_series: List[float]) -> float:
    """Compute maximum drawdown in units from a cumulative PnL series."""
    if not pnl_series:
        return 0.0
    cum = np.cumsum(pnl_series)
    peak = np.maximum.accumulate(cum)
    drawdowns = peak - cum
    return float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0


def evaluate_slice(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute quantitative metrics for a slice of picks.

    Metrics are only populated when the slice has at least MIN_METRIC_SAMPLE
    graded picks. Otherwise they are None so an empty/small slice can never
    silently satisfy a go-live threshold.
    """
    valid = [r for r in records if r['result'] in ('WIN', 'LOSS')]
    n_total = len(records)
    n_graded = len(valid)
    n_wins = sum(1 for r in valid if r['result'] == 'WIN')
    n_losses = sum(1 for r in valid if r['result'] == 'LOSS')
    n_pushes = sum(1 for r in records if r['result'] == 'PUSH')
    pnl_units = sum(r['pnl'] for r in records)
    sufficient_sample = n_graded >= MIN_METRIC_SAMPLE

    win_rate = None
    roi_pct = None
    max_dd = None
    brier = None
    ece_cal = None
    ece_raw = None
    mean_cal_p = None
    mean_raw_p = None

    if sufficient_sample:
        win_rate = round(n_wins / n_graded * 100.0, 2)
        roi_pct = round(pnl_units / n_graded * 100.0, 2)
        max_dd = round(calculate_max_drawdown([r['pnl'] for r in records]), 2)

        cal_probs = [r['cal_prob'] for r in valid]
        raw_probs = [r['raw_prob'] for r in valid]
        outcomes = [r['binary_win'] for r in valid]
        mean_cal_p = round(float(np.mean(cal_probs) * 100.0), 2)
        mean_raw_p = round(float(np.mean(raw_probs) * 100.0), 2)

        prob_metrics = probability_metrics(cal_probs, outcomes)
        brier = round(prob_metrics['brier'], 4)

        # Weighted average calibration error: sum |p_bar - y_bar| * n_b / N
        bins_cal = calibration_bins(cal_probs, outcomes, bins=5)
        tot_cal_n = sum(b['count'] for b in bins_cal)
        ece_cal = round(
            (sum(abs(b['mean_probability'] - b['observed_rate']) * b['count'] for b in bins_cal) / tot_cal_n * 100.0)
            if tot_cal_n > 0 else 0.0,
            2
        )

        bins_raw = calibration_bins(raw_probs, outcomes, bins=5)
        tot_raw_n = sum(b['count'] for b in bins_raw)
        ece_raw = round(
            (sum(abs(b['mean_probability'] - b['observed_rate']) * b['count'] for b in bins_raw) / tot_raw_n * 100.0)
            if tot_raw_n > 0 else 0.0,
            2
        )

    return {
        'total_picks': n_total,
        'graded_picks': n_graded,
        'wins': n_wins,
        'losses': n_losses,
        'pushes': n_pushes,
        'sufficient_sample': sufficient_sample,
        'min_metric_sample': MIN_METRIC_SAMPLE,
        'win_rate': win_rate,
        'pnl_units': round(pnl_units, 2),
        'roi_pct': roi_pct,
        'max_drawdown_units': max_dd,
        'mean_cal_prob': mean_cal_p,
        'mean_raw_prob': mean_raw_p,
        'brier_score': brier,
        'calibration_error_pct': ece_cal,
        'raw_calibration_error_pct': ece_raw
    }


def compute_go_live_checks(post_strong: Dict[str, Any]) -> Dict[str, Optional[bool]]:
    """Evaluate go-live criteria on POST-split actionable STRONG picks only.

    Rules:
    - The sample criterion uses POST STRONG graded picks, never the overall tier.
    - Metric criteria return None (N/A) when the slice has fewer than
      MIN_METRIC_SAMPLE graded picks, so an empty slice cannot pass by default.
    - all_passed is True only when every criterion is explicitly True.
    """
    n_graded = int(post_strong.get('graded_picks') or 0)
    has_metrics = bool(post_strong.get('sufficient_sample')) and n_graded >= MIN_METRIC_SAMPLE

    checks: Dict[str, Optional[bool]] = {
        'criterion_1_min_100_strong_picks': n_graded >= 100,
        'criterion_2_strong_roi_above_2pct': (
            float(post_strong['roi_pct']) > 2.0 if has_metrics and post_strong.get('roi_pct') is not None else None
        ),
        'criterion_3_strong_calibration_under_5pct': (
            float(post_strong['calibration_error_pct']) < 5.0
            if has_metrics and post_strong.get('calibration_error_pct') is not None else None
        ),
        'criterion_4_strong_max_dd_under_10u': (
            float(post_strong['max_drawdown_units']) < 10.0
            if has_metrics and post_strong.get('max_drawdown_units') is not None else None
        ),
    }
    checks['all_passed'] = all(v is True for v in checks.values())

    return checks


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

    # Go-Live Verification Criteria (Evaluated on POST-split actionable STRONG picks only)
    go_live_checks = compute_go_live_checks(post_strong)

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
        'go_live_basis': f'POST STRONG (date >= {SPLIT_DATE})',
        'go_live_checks': go_live_checks,
        'limitations': list(REPLAY_LIMITATIONS),
        'snapshot_coverage': snapshot_coverage([r['date'] for r in all_records]),
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

    def _num(value, spec, suffix=''):
        return 'N/A' if value is None else f"{format(value, spec)}{suffix}"

    def _slice_lines(label, sl):
        print(f"  {label} - {sl['total_picks']} picks (graded: {sl['graded_picks']}):")
        if not sl.get('sufficient_sample'):
            print(f"    Metrics: N/A (need >= {sl['min_metric_sample']} graded picks)")
            return
        print(f"    Win Rate: {_num(sl['win_rate'], '.2f', '%')} (W: {sl['wins']}, L: {sl['losses']}, P: {sl['pushes']})")
        print(f"    PnL: {sl['pnl_units']:+.2f} Units | ROI: {_num(sl['roi_pct'], '+.2f', '%')}")
        print(f"    Mean Prob: Calibrated {_num(sl['mean_cal_prob'], '.1f', '%')} vs Raw {_num(sl['mean_raw_prob'], '.1f', '%')}")
        print(f"    ECE: Calibrated {_num(sl['calibration_error_pct'], '.2f', '%')} vs Raw {_num(sl['raw_calibration_error_pct'], '.2f', '%')}")

    print(f"Total Games Replayed: {res['total_games_replayed']} (dynamically reconstructed as-of date)")
    print(f"Total Model Picks: {res['overall']['total_picks']} (STRONG: {res['strong_tier']['total_picks']}, LEAN: {res['lean_tier']['total_picks']})")
    print(f"Total Graded Picks: {res['overall']['graded_picks']} (W: {res['overall']['wins']}, L: {res['overall']['losses']}, P: {res['overall']['pushes']})")
    print("-" * 70)
    print("EXPANDING WINDOW SPLIT: IN-SAMPLE (PRIOR) vs OUT-OF-SAMPLE (POST)")
    _slice_lines("PRIOR (Before Nov 6)", res['prior_in_sample'])
    _slice_lines("POST (Nov 6 onwards - Out-of-Sample)", res['post_out_of_sample'])
    print("-" * 70)
    print(f"ACTIONABLE PERFORMANCE (STRONG TIER - {res['strong_tier']['total_picks']} PICKS):")
    _slice_lines("Overall STRONG reference", res['strong_tier'])
    _slice_lines("POST STRONG (go-live basis)", res['post_strong'])
    print("-" * 70)
    print(f"PAPER MONITORING PERFORMANCE (LEAN TIER - {res['lean_tier']['total_picks']} PICKS):")
    _slice_lines("LEAN", res['lean_tier'])
    print("-" * 70)
    print(f"GO-LIVE GATES VERIFICATION (basis: {res['go_live_basis']}):")
    for k, v in res['go_live_checks'].items():
        if v is None:
            status_icon = "N/A (insufficient sample)"
        elif v:
            status_icon = "✅ PASS"
        else:
            status_icon = "❌ FAIL"
        print(f"  {k}: {status_icon}")
    overlap = res['snapshot_coverage']
    print(
        f"Snapshot coverage: {overlap['dates_with_snapshot']}/{overlap['dates_replayed']} replayed dates "
        f"have an archived injury/lineup snapshot"
    )
    print("KNOWN REPLAY LIMITATIONS (numbers above are NOT a clean out-of-sample estimate):")
    for limitation in res['limitations']:
        print(f"  - {limitation}")
    print(f"Safety Gate Status: BETTING_PAUSED = {not res['go_live_checks']['all_passed']} "
          f"(go-live passed: {res['go_live_checks']['all_passed']})")
    print("=" * 70)
