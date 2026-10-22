"""Prediction Ledger & Performance Tracking Module.

Provides functions to record pre-game predictions, automatically grade finished
games against actual scores, compute win rates, ROI, and breakdown metrics,
and backfill historical records.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np

from config import RAW_DIR, normalize_team_name, calculate_home_spread

logger = logging.getLogger('prediction_tracker')

LEDGER_PATH = os.path.join(RAW_DIR, 'predictions_ledger.csv')
THAI_TZ = timezone(timedelta(hours=7))

LEDGER_COLUMNS = [
    'game_id',
    'date',
    'home_team',
    'away_team',
    'spread_line',
    'total_line',
    'ou_recommendation',
    'ou_edge',
    'ou_win_prob',
    'ou_verdict',
    'sp_recommendation',
    'sp_cover_edge',
    'sp_win_prob',
    'sp_verdict',
    'home_score',
    'away_score',
    'actual_total',
    'actual_margin',
    'ou_result',
    'sp_result',
    'verdict_result',
    'status',
    'forecast_as_of',
    'model_version',
    'features_snapshot',
    'projected_margin',
    'projected_total',
    'spread_odds',
    'total_odds'
]


STRING_COLUMNS = [
    'game_id', 'date', 'home_team', 'away_team',
    'ou_recommendation', 'ou_verdict', 'sp_recommendation', 'sp_verdict',
    'ou_result', 'sp_result', 'verdict_result', 'status'
]


def load_ledger() -> pd.DataFrame:
    """Load existing predictions ledger or return empty DataFrame with schema."""
    if os.path.exists(LEDGER_PATH):
        try:
            df = pd.read_csv(LEDGER_PATH, dtype={'game_id': str})
            # Ensure all expected columns exist
            for col in LEDGER_COLUMNS:
                if col not in df.columns:
                    df[col] = None
            for col in STRING_COLUMNS:
                if col in df.columns:
                    df[col] = df[col].astype(object)

            # Auto-prune any accidental future PENDING predictions from earlier bugs
            now_thai_str = datetime.now(THAI_TZ).strftime('%Y-%m-%d')
            future_pending = (df['date'].astype(str) > now_thai_str) & (df['status'] == 'PENDING')
            if future_pending.any():
                logger.info(f"Purging {future_pending.sum()} premature future pending prediction(s) from ledger.")
                df = df[~future_pending].reset_index(drop=True)
                save_ledger(df)

            return df
        except Exception as exc:
            logger.warning(f"Error reading {LEDGER_PATH}: {exc}; returning empty ledger.")
    df = pd.DataFrame(columns=LEDGER_COLUMNS)
    for col in STRING_COLUMNS:
        df[col] = df[col].astype(object)
    return df


def save_ledger(df: pd.DataFrame) -> None:
    """Save predictions ledger to disk safely."""
    try:
        os.makedirs(os.path.dirname(LEDGER_PATH), exist_ok=True)
        # Reorder columns to standard schema
        cols = [c for c in LEDGER_COLUMNS if c in df.columns]
        df[cols].to_csv(LEDGER_PATH, index=False, encoding='utf-8')
    except Exception as exc:
        logger.error(f"Failed to save predictions ledger to {LEDGER_PATH}: {exc}")


def record_slate_predictions(
    date_str: str,
    scoreboard_df: pd.DataFrame,
    analyzer: Any,
    espn: Optional[Any] = None,
    odds_coll: Optional[Any] = None,
    max_minutes_before: Optional[float] = 60.0,
    force: bool = False
) -> int:
    """Record pre-game predictions for games in the slate within the pre-game window.

    Locks in official pre-game predictions and closing lines before tip-off.
    Never records future dates or games already in progress/finished.

    Args:
        date_str: Date string ('YYYY-MM-DD').
        scoreboard_df: ESPN scoreboard DataFrame.
        analyzer: Matchup TrendAnalyzer.
        espn: ESPN collector instance.
        odds_coll: Odds collector instance.
        max_minutes_before: Only record games within N minutes before tip-off (default: 60 min).
            If None, records all scheduled games for today regardless of tip-off time.
        force: If True, bypass timing filters (used in tests or offline grading).

    Returns:
        int: Number of new predictions recorded.
    """
    if scoreboard_df is None or scoreboard_df.empty:
        return 0

    now_thai = datetime.now(THAI_TZ)
    today_thai_str = now_thai.strftime('%Y-%m-%d')

    # Guard 1: Never record future dates (games on future dates are preview-only)
    if not force and date_str > today_thai_str:
        logger.info(
            f"Bypassing prediction recording for future date {date_str} (today is {today_thai_str}). "
            f"Pre-game predictions will lock on game day within the pre-game window."
        )
        return 0

    ledger_df = load_ledger()
    existing_gids = set(ledger_df['game_id'].astype(str).tolist()) if not ledger_df.empty else set()

    new_rows = []
    for _, game in scoreboard_df.iterrows():
        gid = str(game.get('game_id', '')).strip()
        if not gid or gid in existing_gids:
            continue

        status_str = str(game.get('status', '')).upper()

        # Guard 2: Skip games already in progress or completed
        # Predictions MUST be locked in PRE-GAME, never retroactively during or after the match!
        if not force and any(s in status_str for s in ['FINAL', 'PROGRESS', 'LIVE', 'HALFTIME']):
            continue

        # Guard 3: Check tip-off time window (only record if within max_minutes_before, e.g. 60m)
        if not force and max_minutes_before is not None:
            thai_dt = game.get('thai_datetime')
            if thai_dt is not None and not pd.isna(thai_dt):
                if isinstance(thai_dt, pd.Timestamp):
                    game_time = thai_dt.to_pydatetime()
                elif isinstance(thai_dt, str):
                    try:
                        game_time = pd.to_datetime(thai_dt).to_pydatetime()
                    except Exception:
                        game_time = None
                else:
                    game_time = thai_dt

                if game_time is not None:
                    if game_time.tzinfo is None:
                        game_time = game_time.replace(tzinfo=THAI_TZ)

                    minutes_until = (game_time - now_thai).total_seconds() / 60.0

                    # Defer recording if game is too far in the future (e.g. > 60 mins away)
                    if minutes_until > max_minutes_before:
                        continue
                    # Skip if game already tipped off
                    if minutes_until < -5.0:
                        continue

        raw_home = game.get('home_team', 'Unknown')
        raw_away = game.get('away_team', 'Unknown')
        home = normalize_team_name(raw_home) or raw_home
        away = normalize_team_name(raw_away) or raw_away

        # 1. Odds API consensus lines
        c_spread, c_total = None, None
        spread_odds = total_odds = None
        if odds_coll:
            try:
                odds_data = odds_coll.get_game_odds(home, away)
                c_spread, c_total = odds_coll.get_consensus_lines(odds_data)
                for book in odds_data.get('spreads', {}).values():
                    if book.get('home') == c_spread and book.get('home_price') is not None and book.get('away_price') is not None:
                        spread_odds = [book['home_price'], book['away_price']]
                        break
                for book in odds_data.get('totals', {}).values():
                    if book.get('over') == c_total and book.get('over_price') is not None and book.get('under_price') is not None:
                        total_odds = [book['over_price'], book['under_price']]
                        break
            except Exception:
                pass

        # 2. Scoreboard lines
        raw_spread = game.get('spread')
        raw_ou = game.get('over_under')
        sp_val = calculate_home_spread(raw_spread, home, away)
        ou_val = float(raw_ou) if pd.notna(raw_ou) else None

        if c_spread is not None: sp_val = c_spread
        if c_total is not None: ou_val = c_total

        # 3. ESPN pickcenter line movement fallback
        lm_data = {}
        if espn and hasattr(espn, 'get_game_line_movement'):
            try:
                lm_data = espn.get_game_line_movement(str(gid))
            except Exception:
                pass

        open_total = lm_data.get('open_total')
        open_spread = lm_data.get('open_spread')

        if sp_val is None:
            sp_val = lm_data.get('current_spread') or lm_data.get('open_spread')
        if ou_val is None:
            ou_val = lm_data.get('current_total') or lm_data.get('open_total')

        # Run analysis
        try:
            forecast_as_of = datetime.now(timezone.utc).isoformat()
            prior = load_ledger()
            margin_residuals, total_residuals = [], []
            if not prior.empty and {'forecast_as_of', 'model_version', 'status', 'actual_margin', 'actual_total', 'projected_margin', 'projected_total'}.issubset(prior.columns):
                prior['forecast_as_of'] = pd.to_datetime(prior['forecast_as_of'], utc=True, errors='coerce')
                eligible = prior[(prior['model_version'] == 'nba-baseline-v2') & (prior['status'] == 'GRADED') & (prior['forecast_as_of'] < pd.Timestamp(forecast_as_of))]
                for _, prior_row in eligible.iterrows():
                    if pd.notna(prior_row['actual_margin']) and pd.notna(prior_row['projected_margin']):
                        margin_residuals.append(float(prior_row['actual_margin']) - float(prior_row['projected_margin']))
                    if pd.notna(prior_row['actual_total']) and pd.notna(prior_row['projected_total']):
                        total_residuals.append(float(prior_row['actual_total']) - float(prior_row['projected_total']))
            margin_residuals = margin_residuals if len(margin_residuals) >= 2 else None
            total_residuals = total_residuals if len(total_residuals) >= 2 else None
            analysis = analyzer.full_matchup_analysis(
                home_team=home,
                away_team=away,
                spread_line=sp_val,
                over_under_line=ou_val,
                open_spread=open_spread,
                open_line=open_total,
                game_date=forecast_as_of,
                spread_odds=spread_odds,
                total_odds=total_odds,
                margin_residuals=margin_residuals,
                total_residuals=total_residuals
            )
        except Exception as exc:
            logger.warning(f"Error analyzing game {gid} ({away} @ {home}): {exc}")
            continue

        ou_res = analysis.get('over_under') or {}
        sp_res = analysis.get('spread') or {}

        ou_lm = ou_res.get('line_movement') or {}
        sp_lm = sp_res.get('line_movement') or {}

        new_rows.append({
            'game_id': gid,
            'date': date_str,
            'home_team': home,
            'away_team': away,
            'spread_line': sp_val,
            'total_line': ou_val,
            'ou_recommendation': ou_res.get('recommendation'),
            'ou_edge': round(ou_res.get('total_edge', 0.0), 2) if ou_res.get('total_edge') is not None else None,
            'ou_win_prob': round(ou_res.get('win_prob', 0.5), 4) if ou_res.get('win_prob') is not None else None,
            'ou_verdict': ou_lm.get('verdict'),
            'sp_recommendation': sp_res.get('recommendation'),
            'sp_cover_edge': round(sp_res.get('cover_edge', 0.0), 2) if sp_res.get('cover_edge') is not None else None,
            'sp_win_prob': round(sp_res.get('win_prob', 0.5), 4) if sp_res.get('win_prob') is not None else None,
            'sp_verdict': sp_lm.get('verdict'),
            'home_score': None,
            'away_score': None,
            'actual_total': None,
            'actual_margin': None,
            'ou_result': None,
            'sp_result': None,
            'verdict_result': None,
            'status': 'PENDING',
            'forecast_as_of': forecast_as_of,
            'model_version': 'nba-baseline-v2',
            'features_snapshot': json.dumps(analysis.get('features', {}), default=str, sort_keys=True),
            'projected_margin': sp_res.get('projected_margin'),
            'projected_total': ou_res.get('projected_total'),
            'spread_odds': json.dumps(spread_odds),
            'total_odds': json.dumps(total_odds)
        })
        existing_gids.add(gid)

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        ledger_df = pd.concat([ledger_df, new_df], ignore_index=True)
        save_ledger(ledger_df)
        logger.info(f"Recorded {len(new_rows)} new predictions to ledger.")

    return len(new_rows)


def grade_finished_games(scoreboard_df: pd.DataFrame) -> int:
    """Grade pending predictions against final scores.

    Matches game_id with completed games (STATUS_FINAL) in scoreboard_df,
    computes actual total & margin, and evaluates WIN / LOSS / PUSH.

    Returns:
        int: Number of newly graded predictions.
    """
    if scoreboard_df is None or scoreboard_df.empty:
        return 0

    ledger_df = load_ledger()
    if ledger_df.empty:
        return 0

    # Build map of final games
    final_games = {}
    for _, game in scoreboard_df.iterrows():
        st_raw = str(game.get('status', '')).upper()
        if 'FINAL' in st_raw:
            gid = str(game.get('game_id', '')).strip()
            h_score = game.get('home_score')
            a_score = game.get('away_score')
            if pd.notna(h_score) and pd.notna(a_score) and str(h_score).strip() != '' and str(a_score).strip() != '':
                try:
                    final_games[gid] = (float(h_score), float(a_score))
                except (ValueError, TypeError):
                    pass

    if not final_games:
        return 0

    for col in ['ou_result', 'sp_result', 'verdict_result', 'status']:
        if col in ledger_df.columns:
            ledger_df[col] = ledger_df[col].astype(object)

    graded_count = 0
    for idx, row in ledger_df.iterrows():
        gid = str(row['game_id'])
        if gid in final_games and (row['status'] != 'GRADED' or pd.isna(row['ou_result'])):
            h_pts, a_pts = final_games[gid]
            act_total = h_pts + a_pts
            act_margin = h_pts - a_pts  # Home - Away

            t_line = row['total_line']
            s_line = row['spread_line']

            ou_res = grade_over_under_pick(row['ou_recommendation'], t_line, act_total)
            sp_res = grade_spread_pick(row['sp_recommendation'], s_line, act_margin)
            v_res = grade_verdict_pick(row['ou_verdict'], row['ou_recommendation'], t_line, act_total)

            ledger_df.at[idx, 'home_score'] = h_pts
            ledger_df.at[idx, 'away_score'] = a_pts
            ledger_df.at[idx, 'actual_total'] = act_total
            ledger_df.at[idx, 'actual_margin'] = act_margin
            ledger_df.at[idx, 'ou_result'] = ou_res
            ledger_df.at[idx, 'sp_result'] = sp_res
            ledger_df.at[idx, 'verdict_result'] = v_res
            ledger_df.at[idx, 'status'] = 'GRADED'
            graded_count += 1

    if graded_count > 0:
        save_ledger(ledger_df)
        logger.info(f"Graded {graded_count} finished predictions in ledger.")

    return graded_count


def grade_over_under_pick(recommendation: Any, total_line: Any, actual_total: float) -> str:
    """Evaluate Over/Under pick outcome."""
    if pd.isna(recommendation) or pd.isna(total_line) or actual_total is None:
        return 'SKIPPED'
    try:
        line = float(total_line)
    except (ValueError, TypeError):
        return 'SKIPPED'

    rec = str(recommendation).upper()
    if 'OVER' in rec:
        if actual_total > line: return 'WIN'
        elif actual_total < line: return 'LOSS'
        else: return 'PUSH'
    elif 'UNDER' in rec:
        if actual_total < line: return 'WIN'
        elif actual_total > line: return 'LOSS'
        else: return 'PUSH'
    return 'SKIPPED'


def grade_spread_pick(recommendation: Any, spread_line: Any, actual_margin: float) -> str:
    """Evaluate Spread pick outcome (spread_line is relative to home team)."""
    if pd.isna(recommendation) or pd.isna(spread_line) or actual_margin is None:
        return 'SKIPPED'
    try:
        s_val = float(spread_line)
    except (ValueError, TypeError):
        return 'SKIPPED'

    # Home covers if actual_margin + spread_line > 0
    # e.g. Home wins by 5 with -4.5 -> 5 + (-4.5) = +0.5 > 0 -> Home covers
    cover_diff = actual_margin + s_val
    rec = str(recommendation).upper()

    if 'HOME' in rec:
        if cover_diff > 0: return 'WIN'
        elif cover_diff < 0: return 'LOSS'
        else: return 'PUSH'
    elif 'AWAY' in rec:
        if cover_diff < 0: return 'WIN'
        elif cover_diff > 0: return 'LOSS'
        else: return 'PUSH'
    return 'SKIPPED'


def grade_verdict_pick(verdict: Any, recommendation: Any, total_line: Any, actual_total: float) -> str:
    """Evaluate Directional Verdict outcome."""
    if pd.isna(verdict) or verdict in ['NO_DATA', 'PASS', 'CAPITAL_PRESERVATION', 'LIVE_TRACKING', None]:
        return 'SKIPPED'
    v_str = str(verdict).upper()
    if v_str in ['FOLLOW_MODEL', 'CONFIRMED_ALIGNED', 'STABLE_MODEL']:
        return grade_over_under_pick(recommendation, total_line, actual_total)
    return 'SKIPPED'


def calculate_track_record(ledger_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute comprehensive performance metrics and win rates from ledger."""
    empty_stats = {
        'total_predictions': 0,
        'graded_games': 0,
        'pending_games': 0,
        'overall': {'bets': 0, 'wins': 0, 'losses': 0, 'pushes': 0, 'win_rate': 0.0, 'net_units': 0.0, 'roi_pct': 0.0},
        'strong': {'bets': 0, 'wins': 0, 'losses': 0, 'pushes': 0, 'win_rate': 0.0, 'net_units': 0.0, 'roi_pct': 0.0},
        'lean': {'bets': 0, 'wins': 0, 'losses': 0, 'pushes': 0, 'win_rate': 0.0, 'net_units': 0.0, 'roi_pct': 0.0},
        'over_under': {'bets': 0, 'wins': 0, 'losses': 0, 'pushes': 0, 'win_rate': 0.0, 'net_units': 0.0, 'roi_pct': 0.0},
        'spread': {'bets': 0, 'wins': 0, 'losses': 0, 'pushes': 0, 'win_rate': 0.0, 'net_units': 0.0, 'roi_pct': 0.0},
        'verdict': {'bets': 0, 'wins': 0, 'losses': 0, 'pushes': 0, 'win_rate': 0.0, 'net_units': 0.0, 'roi_pct': 0.0},
    }

    if ledger_df is None or ledger_df.empty:
        return empty_stats

    def _calc_subset(results: List[str]) -> Dict[str, Any]:
        valid = [r for r in results if r in ['WIN', 'LOSS', 'PUSH']]
        w = valid.count('WIN')
        l = valid.count('LOSS')
        p = valid.count('PUSH')
        decided = w + l
        wr = round(w / decided * 100.0, 1) if decided > 0 else 0.0
        # Standard -110 juice flat betting: risking 1.10 units to win 1.0 unit (or risk 1.0 to win 0.909)
        # Net units: w * 0.9091 - l * 1.0
        net = round(w * 0.9091 - l * 1.0, 2)
        total_risked = decided * 1.0
        roi = round(net / total_risked * 100.0, 1) if total_risked > 0 else 0.0
        return {
            'bets': decided + p,
            'wins': w,
            'losses': l,
            'pushes': p,
            'win_rate': wr,
            'net_units': net,
            'roi_pct': roi
        }

    graded = ledger_df[ledger_df['status'] == 'GRADED']
    pending_count = len(ledger_df[ledger_df['status'] != 'GRADED'])

    all_results = []
    ou_results = []
    sp_results = []
    verdict_results = []
    strong_results = []
    lean_results = []
    strong_sp_results = []
    strong_ou_results = []
    lean_sp_results = []
    lean_ou_results = []

    for _, row in graded.iterrows():
        ou_r = row.get('ou_result')
        sp_r = row.get('sp_result')
        v_r = row.get('verdict_result')
        ou_rec = str(row.get('ou_recommendation', ''))
        sp_rec = str(row.get('sp_recommendation', ''))

        if ou_r in ['WIN', 'LOSS', 'PUSH']:
            ou_results.append(ou_r)
            all_results.append(ou_r)
            if 'STRONG' in ou_rec:
                strong_results.append(ou_r)
                strong_ou_results.append(ou_r)
            elif 'LEAN' in ou_rec:
                lean_results.append(ou_r)
                lean_ou_results.append(ou_r)

        if sp_r in ['WIN', 'LOSS', 'PUSH']:
            sp_results.append(sp_r)
            all_results.append(sp_r)
            if 'STRONG' in sp_rec:
                strong_results.append(sp_r)
                strong_sp_results.append(sp_r)
            elif 'LEAN' in sp_rec:
                lean_results.append(sp_r)
                lean_sp_results.append(sp_r)

        if v_r in ['WIN', 'LOSS', 'PUSH']:
            verdict_results.append(v_r)

    return {
        'total_predictions': len(ledger_df),
        'graded_games': len(graded),
        'pending_games': pending_count,
        'overall': _calc_subset(all_results),
        'strong': _calc_subset(strong_results),
        'lean': _calc_subset(lean_results),
        'over_under': _calc_subset(ou_results),
        'spread': _calc_subset(sp_results),
        'strong_spread': _calc_subset(strong_sp_results),
        'strong_ou': _calc_subset(strong_ou_results),
        'lean_spread': _calc_subset(lean_sp_results),
        'lean_ou': _calc_subset(lean_ou_results),
        'verdict': _calc_subset(verdict_results)
    }


def get_available_ledger_dates(ledger_df: pd.DataFrame) -> List[str]:
    """Return sorted list of unique dates from ledger in descending order."""
    if ledger_df is None or ledger_df.empty or 'date' not in ledger_df.columns:
        return []
    dates = [str(d).strip() for d in ledger_df['date'].dropna().unique() if str(d).strip() != '']
    return sorted(dates, reverse=True)


def backfill_season_history(analyzer: Any) -> int:
    """Backfill predictions and grading for finished 2025-26 season games."""
    from collectors.espn_collector import ESPNCollector
    collector = ESPNCollector()

    recent_games_path = os.path.join(RAW_DIR, 'espn_recent_games.csv')
    if not os.path.exists(recent_games_path):
        logger.warning(f"No recent games file at {recent_games_path} for backfill.")
        return 0

    games_df = pd.read_csv(recent_games_path)
    if games_df.empty:
        return 0

    # Filter to completed regular season games
    completed = games_df[
        (pd.to_numeric(games_df['home_score'], errors='coerce') > 0) &
        (pd.to_numeric(games_df['away_score'], errors='coerce') > 0)
    ].copy()

    logger.info(f"Starting historical backfill for {len(completed)} completed games...")
    count_new = 0
    ledger_df = load_ledger()
    existing_gids = set(ledger_df['game_id'].astype(str).tolist()) if not ledger_df.empty else set()

    new_rows = []
    for _, game in completed.iterrows():
        gid = str(game.get('game_id', '')).strip()
        if not gid or gid in existing_gids:
            continue

        raw_home = game.get('home_team', 'Unknown')
        raw_away = game.get('away_team', 'Unknown')
        home = normalize_team_name(raw_home) or raw_home
        away = normalize_team_name(raw_away) or raw_away

        h_score = float(game.get('home_score'))
        a_score = float(game.get('away_score'))
        act_total = h_score + a_score
        act_margin = h_score - a_score

        # Get lines
        raw_sp = game.get('spread')
        raw_ou = game.get('over_under')
        sp_val = calculate_home_spread(raw_sp, home, away)
        ou_val = float(raw_ou) if pd.notna(raw_ou) else None

        # Fetch pickcenter line movement if available
        lm_data = {}
        try:
            lm_data = collector.get_game_line_movement(str(gid))
        except Exception:
            pass

        open_total = lm_data.get('open_total')
        open_spread = lm_data.get('open_spread')
        if sp_val is None:
            sp_val = lm_data.get('current_spread') or lm_data.get('open_spread')
        if ou_val is None:
            ou_val = lm_data.get('current_total') or lm_data.get('open_total')

        if sp_val is None and ou_val is None:
            continue

        date_val = str(game.get('date', ''))[:10]

        try:
            analysis = analyzer.full_matchup_analysis(
                home_team=home,
                away_team=away,
                spread_line=sp_val,
                over_under_line=ou_val,
                open_spread=open_spread,
                open_line=open_total
            )
        except Exception:
            continue

        ou_res = analysis.get('over_under') or {}
        sp_res = analysis.get('spread') or {}
        ou_lm = ou_res.get('line_movement') or {}
        sp_lm = sp_res.get('line_movement') or {}

        ou_pick = ou_res.get('recommendation')
        sp_pick = sp_res.get('recommendation')
        v_pick = ou_lm.get('verdict')

        ou_grade = grade_over_under_pick(ou_pick, ou_val, act_total)
        sp_grade = grade_spread_pick(sp_pick, sp_val, act_margin)
        v_grade = grade_verdict_pick(v_pick, ou_pick, ou_val, act_total)

        new_rows.append({
            'game_id': gid,
            'date': date_val,
            'home_team': home,
            'away_team': away,
            'spread_line': sp_val,
            'total_line': ou_val,
            'ou_recommendation': ou_pick,
            'ou_edge': round(ou_res.get('total_edge', 0.0), 2) if ou_res.get('total_edge') is not None else None,
            'ou_win_prob': round(ou_res.get('win_prob', 0.5), 4) if ou_res.get('win_prob') is not None else None,
            'ou_verdict': v_pick,
            'sp_recommendation': sp_pick,
            'sp_cover_edge': round(sp_res.get('cover_edge', 0.0), 2) if sp_res.get('cover_edge') is not None else None,
            'sp_win_prob': round(sp_res.get('win_prob', 0.5), 4) if sp_res.get('win_prob') is not None else None,
            'sp_verdict': sp_lm.get('verdict'),
            'home_score': h_score,
            'away_score': a_score,
            'actual_total': act_total,
            'actual_margin': act_margin,
            'ou_result': ou_grade,
            'sp_result': sp_grade,
            'verdict_result': v_grade,
            'status': 'GRADED'
        })
        existing_gids.add(gid)
        count_new += 1

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        ledger_df = pd.concat([ledger_df, new_df], ignore_index=True)
        save_ledger(ledger_df)
        logger.info(f"Backfilled {count_new} completed games into predictions ledger.")

    return count_new
