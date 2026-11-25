#!/usr/bin/env python3
"""NBA Pre-Game Wave Alert Dispatcher.

Scans the NBA scoreboard in Thai timezone, identifies games tipping off in 20-45 minutes,
runs quantitative matchup analysis, formats compact actionable cards, and dispatches
them to Telegram.
"""

import os
import sys
import json
import logging
import argparse
import re
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional

# Fix Windows console encoding for emoji/unicode
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from config import (
    DATA_DIR,
    RAW_DIR,
    ensure_dirs,
    normalize_team_name,
    calculate_home_spread,
    resolve_team_info,
    t_rec,
)
from collectors.espn_collector import ESPNCollector
from collectors.odds_collector import OddsCollector
from processing.data_merger import DataMerger
from processing.feature_engineering import FeatureEngineer
from processing.snapshot_archive import archive_daily_snapshot
from analysis.trend_analyzer import TrendAnalyzer
from analysis.prediction_tracker import (
    grade_finished_games,
    record_slate_predictions,
    compute_game_analysis,
    get_historical_residuals,
    load_ledger,
    calculate_daily_performance_report,
)
from notifications.telegram_bot import TelegramNotifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger('nba_alerts')

THAI_TZ = timezone(timedelta(hours=7))
ALERT_LEDGER_PATH = os.path.join(RAW_DIR, 'alerted_games.json')
DEFAULT_DASHBOARD_URL = os.getenv('STREAMLIT_APP_URL', 'https://hetu8dr75jdcavumnqk6jz.streamlit.app/')
ENABLE_AI_RESIDUAL = False  # Disabled during early season: base math outperforms residual model (MAE 13.62 vs 14.39)
BETTING_PAUSED = os.getenv('NBA_BETTING_PAUSED', 'true').lower() in ('true', '1', 'yes')  # Paper mode until go-live criteria met


def load_alerted_ledger() -> Dict[str, List[str]]:
    """Load map of date -> list of alerted game IDs."""
    if os.path.exists(ALERT_LEDGER_PATH):
        try:
            with open(ALERT_LEDGER_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception as exc:
            logger.warning(f"Error reading alert ledger, starting fresh: {exc}")
    return {}


def save_alerted_ledger(ledger: Dict[str, List[str]]) -> None:
    """Save map of date -> list of alerted game IDs with 7-day pruning."""
    ensure_dirs()
    # Prune dates older than 7 days
    today = datetime.now(THAI_TZ).date()
    cutoff_str = (today - timedelta(days=7)).strftime('%Y-%m-%d')
    pruned = {d: gids for d, gids in ledger.items() if d >= cutoff_str}

    try:
        with open(ALERT_LEDGER_PATH, 'w', encoding='utf-8') as f:
            json.dump(pruned, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        logger.error(f"Failed to persist alert ledger: {exc}")


def mark_games_alerted(date_str: str, game_ids: List[str]) -> None:
    """Mark given game IDs as alerted for the date."""
    ledger = load_alerted_ledger()
    existing = set(ledger.get(date_str, []))
    for gid in game_ids:
        existing.add(str(gid))
    ledger[date_str] = list(existing)
    save_alerted_ledger(ledger)


def is_game_already_alerted(date_str: str, game_id: str) -> bool:
    """Check if game was already alerted."""
    ledger = load_alerted_ledger()
    return str(game_id) in ledger.get(date_str, [])


def format_game_card(analysis: Dict[str, Any], game_info: Dict[str, Any], odds_data: Optional[Dict[str, Any]] = None) -> str:
    odds_data = odds_data or {}
    """Format an actionable, compact HTML card for a single NBA game (6-line compact mobile layout)."""
    home = game_info.get('home_team', 'Home')
    away = game_info.get('away_team', 'Away')
    time_str = game_info.get('thai_time', 'N/A')

    h_info = resolve_team_info(home) or {}
    a_info = resolve_team_info(away) or {}
    h_abbr = h_info.get('abbreviation') or home[:3].upper()
    a_abbr = a_info.get('abbreviation') or away[:3].upper()

    ou_res = analysis.get('over_under') or {}
    sp_res = analysis.get('spread') or {}
    feat = sp_res.get('features') or ou_res.get('features') or analysis.get('features') or {}

    card_lines = []

    # Line 1: Matchup Header with 3-letter abbreviations
    card_lines.append(f"🏀 <b>{a_abbr} @ {h_abbr}</b> (⏰ {time_str} น.)")

    # Line 2: Lines and Lineup status
    sp_val = sp_res.get('features', {}).get('spread_line')
    ou_val = ou_res.get('features', {}).get('over_under_line')
    sp_text = f"{h_abbr} {sp_val:+g}" if sp_val is not None else "N/A"
    ou_text = f"{ou_val:.1f}" if ou_val is not None else "N/A"

    h_st = feat.get('home_lineup_status', 'Unconfirmed')
    a_st = feat.get('away_lineup_status', 'Unconfirmed')
    if h_st == 'Confirmed' and a_st == 'Confirmed':
        st_badge = "✅ Confirmed"
    elif h_st in ['Confirmed', 'Expected'] or a_st in ['Confirmed', 'Expected']:
        st_badge = "⚠️ Expected"
    else:
        st_badge = "Projected"

    card_lines.append(f"📊 <b>เรต:</b> {sp_text} | O/U {ou_text} ({st_badge})")

    # Helper to extract surname properly handling generational suffixes (Jr., III, etc.)
    def _get_player_surname(full_name: str) -> str:
        if not full_name:
            return ""
        parts = full_name.strip().split()
        if len(parts) <= 1:
            return parts[0]
        suffixes = {'jr.', 'jr', 'sr.', 'sr', 'ii', 'iii', 'iv', 'v'}
        if parts[-1].lower().rstrip('.') in suffixes:
            if len(parts) >= 2:
                return f"{parts[-2]} {parts[-1]}"
        return parts[-1]

    # Line 3: 5 Starters (only surnames, 1 line)
    h_starters = feat.get('home_starters', [])
    a_starters = feat.get('away_starters', [])
    starter_parts = []
    if a_starters:
        a_surnames = ", ".join(_get_player_surname(s.get('name', '')) for s in a_starters[:5] if s.get('name'))
        if a_surnames:
            starter_parts.append(f"{a_abbr}: {a_surnames}")
    if h_starters:
        h_surnames = ", ".join(_get_player_surname(s.get('name', '')) for s in h_starters[:5] if s.get('name'))
        if h_surnames:
            starter_parts.append(f"{h_abbr}: {h_surnames}")
    if starter_parts:
        card_lines.append(f"⛹️ <b>ตัวจริง:</b> {' | '.join(starter_parts)}")

    # Line 4: Key Injuries (condensed to surname & status, 1 line)
    def _clean_inj(inj_list):
        items = []
        for item in inj_list[:3]:
            match = re.search(r"(.*?)(?:\s+\(Base\))?\s*\[(.*?)\]", item)
            if match:
                raw_name = match.group(1).strip()
                name_part = _get_player_surname(raw_name)
                status_raw = match.group(2).split('-')[0].strip()
                items.append(f"{name_part} [{status_raw}]")
            else:
                items.append(_get_player_surname(item))
        return ", ".join(items)

    h_inj = feat.get('home_injuries', [])
    a_inj = feat.get('away_injuries', [])
    inj_parts = []
    if a_inj:
        cleaned_a = _clean_inj(a_inj)
        if cleaned_a:
            inj_parts.append(f"{a_abbr}: {cleaned_a}")
    if h_inj:
        cleaned_h = _clean_inj(h_inj)
        if cleaned_h:
            inj_parts.append(f"{h_abbr}: {cleaned_h}")

    if inj_parts:
        card_lines.append(f"🏥 <b>ตัวเจ็บ:</b> {' | '.join(inj_parts)}")
    else:
        card_lines.append("🏥 <b>ตัวเจ็บ:</b> สภาพทีมสมบูรณ์ทั้งสองทีม")

    # Lines 5 & 6: Picks & Supporting Stats
    sp_rec = sp_res.get('recommendation', 'NEUTRAL')
    ou_rec = ou_res.get('recommendation', 'NEUTRAL')

    # Market Clash / PASS Strict Filter: Suppress picks where market line flow conflicts with model
    CLASH_VERDICTS = {'CAPITAL_PRESERVATION', 'PASS', 'MOVE_CONFLICT'}
    sp_lm = sp_res.get('line_movement') or {}
    ou_lm = ou_res.get('line_movement') or {}
    sp_v = sp_lm.get('verdict')
    ou_v = ou_lm.get('verdict')

    global_clash = analysis.get('is_market_clash', False) or (analysis.get('line_movement', {}).get('verdict') in CLASH_VERDICTS)
    sp_clash = global_clash or sp_res.get('is_market_clash', False) or (sp_v in CLASH_VERDICTS)
    ou_clash = global_clash or ou_res.get('is_market_clash', False) or (ou_v in CLASH_VERDICTS)

    picks = []
    if ('STRONG' in sp_rec or 'LEAN' in sp_rec) and not sp_clash:
        picks.append((sp_res, 'Spread'))
    if ('STRONG' in ou_rec or 'LEAN' in ou_rec) and not ou_clash:
        picks.append((ou_res, 'Total'))

    if picks:
        for res, p_type in picks:
            rec = res.get('recommendation', '')
            edge = res.get('cover_edge') if p_type == 'Spread' else res.get('total_edge', 0.0)
            win_p = res.get('win_prob', 0.5)
            ev = res.get('ev', 0.0)
            q_kelly = res.get('kelly_quarter', 0.0)
            ai_adj = res.get('ai_residual')

            lm_pick = res.get('line_movement') or {}
            is_confirmed = lm_pick.get('verdict') in ['CONFIRMED_ALIGNED', 'MOVE_ALIGNED', 'STABLE_MODEL']
            pick_badge = "🔥 [CONFIRMED]" if is_confirmed else "🎯"

            if BETTING_PAUSED:
                tier_badge = "🧪 [Paper Strong]" if 'STRONG' in rec else "📝 [Paper Lean]"
            else:
                tier_badge = "🟢 [ไม้หลัก]" if 'STRONG' in rec else "🟡 [ไม้เล็ก 0.25-0.5U]"
            if p_type == 'Spread':
                action_text = t_rec(rec, spread_line=sp_val, home_team=home, away_team=away)
                card_lines.append(f"{pick_badge} {tier_badge} <b>แต้มต่อ: {action_text}</b>")
            else:
                action_text = t_rec(rec)
                line_str = f" {ou_val:.1f}" if ou_val is not None else ""
                card_lines.append(f"{pick_badge} {tier_badge} <b>สูง/ต่ำ: {action_text}{line_str}</b>")

            stat_items = [
                f"Win <code>{win_p:.1%}</code>",
                f"Edge <code>{edge:+.1f} pts</code>"
            ]
            if is_confirmed:
                stat_items.append("🌊 ตลาดหนุน")
            if ai_adj is not None and ai_adj != 0.0:
                stat_items.append(f"🤖 AI: <code>{ai_adj:+.1f}</code>")
            if ev is not None and ev > 0:
                ev_str = f"⚡ +EV <code>+{ev*100:.1f}%</code>"
                if q_kelly and q_kelly > 0:
                    ev_str += f" (Kelly <code>{q_kelly*100:.1f}%</code>)"
                stat_items.append(ev_str)

            card_lines.append(f"📈 {' | '.join(stat_items)}")
    elif global_clash or sp_v == 'MOVE_CONFLICT' or ou_v == 'MOVE_CONFLICT' or (sp_res.get('is_market_clash') and ('STRONG' in sp_rec or 'LEAN' in sp_rec)) or (ou_res.get('is_market_clash') and ('STRONG' in ou_rec or 'LEAN' in ou_rec)):
        card_lines.append("⚖️ <b>คำแนะนำ:</b> ข้าม (PASS - สัญญาณขัดแย้งกับกระแสเงินตลาด)")
    elif sp_clash or ou_clash:
        card_lines.append("⚖️ <b>คำแนะนำ:</b> ข้าม (PASS - แนะนำรักษาเงินทุน สถิติสูสีไม่มี Edge คุ้มเสี่ยง)")
    else:
        card_lines.append("⚖️ <b>คำแนะนำ:</b> เรตราคาเปิดมาสมดุล (NEUTRAL)")

    return "\n".join(card_lines)


def run_alert_cycle(
    target_date_str: Optional[str] = None,
    dry_run: bool = False,
    force: bool = False,
    window_min: int = 8,
    window_max: int = 25,
    dashboard_url: str = DEFAULT_DASHBOARD_URL
) -> int:
    """Run one alert check cycle. Returns count of dispatched games."""
    now_thai = datetime.now(THAI_TZ)
    if target_date_str:
        today_thai_date = datetime.strptime(target_date_str, '%Y-%m-%d').date()
    else:
        today_thai_date = now_thai.date()
    date_str = today_thai_date.strftime('%Y-%m-%d')

    logger.info(f"Scanning scoreboard for {date_str} (current Thai time: {now_thai.strftime('%H:%M:%S')})...")

    espn = ESPNCollector()
    odds_coll = OddsCollector()
    notifier = TelegramNotifier()

    scoreboard = espn.get_scoreboard_by_thai_date(today_thai_date)

    # 1. Auto-grade finished games in the predictions ledger
    try:
        g_count = grade_finished_games(scoreboard)
        ledger_df = load_ledger()
        if not ledger_df.empty and (ledger_df['status'] == 'PENDING').any():
            yesterday_sb = espn.get_scoreboard_by_thai_date(today_thai_date - timedelta(days=1))
            if not yesterday_sb.empty:
                g_count += grade_finished_games(yesterday_sb)
        if g_count > 0:
            logger.info(f"Auto-graded {g_count} finished prediction(s) in ledger.")
            try:
                perf_report = calculate_daily_performance_report()
                logger.info(
                    f"Daily performance report updated: as_of={perf_report.get('as_of_date')}, "
                    f"7d_wr={perf_report.get('rolling_7d', {}).get('win_rate')}%, "
                    f"7d_roi={perf_report.get('rolling_7d', {}).get('roi_pct')}%"
                )
            except Exception as rpt_err:
                logger.warning(f"Failed to generate daily performance report: {rpt_err}")
    except Exception as exc:
        logger.warning(f"Error auto-grading predictions: {exc}")

    if scoreboard.empty:
        logger.info("No games scheduled on scoreboard today.")
        return 0

    candidate_games = []

    for _, game in scoreboard.iterrows():
        gid = str(game.get('game_id', ''))
        status_str = str(game.get('status', '')).upper()

        # Skip games already in progress or completed
        if any(s in status_str for s in ['FINAL', 'PROGRESS', 'LIVE', 'HALFTIME']):
            continue

        # Skip already alerted games
        if is_game_already_alerted(date_str, gid) and not force:
            continue

        # Check tip-off time
        thai_dt = game.get('thai_datetime')
        if pd.isna(thai_dt) or thai_dt is None:
            continue

        if isinstance(thai_dt, pd.Timestamp):
            game_time = thai_dt.to_pydatetime()
        else:
            game_time = thai_dt

        minutes_until = (game_time - now_thai).total_seconds() / 60.0

        if force or (window_min <= minutes_until <= window_max):
            candidate_games.append({
                'game': game,
                'minutes_until': minutes_until,
                'game_time': game_time
            })

    if not candidate_games:
        logger.info(f"No games tipping off in the next {window_min}-{window_max} minutes window.")
        return 0

    logger.info(f"Found {len(candidate_games)} game(s) matching pre-game alert criteria!")

    # Refresh live lineups right before analyzing pre-game candidates to catch confirmed starters
    try:
        from collectors.rotowire_collector import RotoWireCollector
        RotoWireCollector().get_todays_lineups(force_refresh=True)
    except Exception as rw_err:
        logger.warning(f"Could not force refresh RotoWire lineups: {rw_err}")

    # Archive today's injury/lineup/roster inputs so a future replay can avoid lookahead bias.
    # Best-effort: failures are logged inside the helper and never block alerts.
    try:
        archived = archive_daily_snapshot(today_thai_date)
        if archived:
            logger.info(f"Daily snapshot archived: {sorted(archived)}")
    except Exception as snap_err:
        logger.warning(f"Could not archive daily snapshot: {snap_err}")

    # Lazy load heavy models only when games qualify
    merger = DataMerger()
    game_history = merger.load_all_games()
    adv_path = os.path.join(DATA_DIR, 'raw', 'nba_api_advanced.csv')
    if os.path.exists(adv_path):
        team_adv = pd.read_csv(adv_path)
        mtime = os.path.getmtime(adv_path)
        team_adv['snapshot_at'] = pd.to_datetime(mtime, unit='s', utc=True)
    else:
        team_adv = None
    fe = FeatureEngineer(game_history, team_advanced=team_adv)
    analyzer = TrendAnalyzer(fe)

    # Load AI Residual Models with Graceful Fallback
    ai_ou_model = None
    ai_sp_model = None
    if ENABLE_AI_RESIDUAL:
        try:
            from analysis.ai_residual_model import OverUnderResidualModel, SpreadResidualModel
            ai_ou_model = OverUnderResidualModel()
            ai_sp_model = SpreadResidualModel()
            logger.info(f"AI Residual Models loaded (OU fitted: {ai_ou_model.is_fitted}, Spread fitted: {ai_sp_model.is_fitted})")
        except Exception as e:
            logger.warning(f"Could not load AI models, falling back to pure math: {e}")

    # 2. Build unified analysis objects for candidate games (single run, identical residuals/timestamp/lines)
    forecast_as_of = datetime.now(timezone.utc).isoformat()
    margin_res, total_res = get_historical_residuals(forecast_as_of)
    candidate_analyses = {}

    for item in candidate_games:
        game = item['game']
        gid = str(game.get('game_id', ''))
        g_dict = game.to_dict() if isinstance(game, pd.Series) else dict(game)
        try:
            analysis = compute_game_analysis(
                game=g_dict,
                analyzer=analyzer,
                espn=espn,
                odds_coll=odds_coll,
                forecast_as_of=forecast_as_of,
                margin_residuals=margin_res,
                total_residuals=total_res
            )
            candidate_analyses[gid] = analysis
        except Exception as exc:
            logger.warning(f"Error computing unified analysis for game {gid}: {exc}")

    # 3. Format Telegram alert cards using the EXACT SAME analysis object
    cards = []
    alerted_gids = []

    for item in candidate_games:
        game = item['game']
        gid = str(game.get('game_id', ''))
        analysis = candidate_analyses.get(gid)
        if not analysis:
            continue
        try:
            g_dict = game.to_dict() if isinstance(game, pd.Series) else dict(game)
            odds_data = analysis.get('odds_data')
            card = format_game_card(analysis, g_dict, odds_data)
            cards.append(card)
            alerted_gids.append(gid)
        except Exception as game_err:
            logger.error(f"Failed to format card for game {gid}: {game_err}", exc_info=True)
            continue

    # Compose wave message
    if BETTING_PAUSED:
        header = (
            f"⚠️ <b>[PAPER TEST ONLY / BETTING PAUSED]</b>\n"
            f"🧪 โมเดลอยู่ในโหมดจำลองกระดาษ (งดวางเงินจริงจนกว่าจะผ่านเกณฑ์ Go-Live)\n"
            f"📅 ข้อมูลวิเคราะห์ล่วงหน้า 15 นาทีก่อนเริ่มแข่งขัน\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
        )
    else:
        header = (
            f"🔔 <b>[NBA PRE-GAME INTELLIGENCE]</b>\n"
            f"📅 ข้อมูลวิเคราะห์ล่วงหน้า 15 นาทีก่อนเริ่มแข่งขัน\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
        )
    separator = "\n────────────────────\n"
    footer = (
        f"\n━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 <b>เปิดดู Dashboard วิเคราะห์สดเต็มรูปแบบ:</b>\n"
        f"👉 <a href='{dashboard_url}'>{dashboard_url}</a>"
    )

    full_message = header + separator.join(cards) + footer

    if dry_run:
        print("\n=== [DRY RUN MESSAGE PREVIEW] ===")
        print(full_message)
        print("=================================\n")
        logger.info(f"Dry-run complete for {len(alerted_gids)} games (no Telegram API call made, ledger untouched).")
        return len(alerted_gids)

    # Dispatch to Telegram (Sound enabled 24/7 as requested)
    success = notifier.send_message(full_message, disable_notification=False)
    if success:
        mark_games_alerted(date_str, alerted_gids)
        logger.info(f"Successfully dispatched alert for {len(alerted_gids)} game(s): {alerted_gids}")

        # Single writer: record into predictions ledger ONLY after alert is successfully sent
        published_at = datetime.now(timezone.utc).isoformat()
        try:
            n_rec = record_slate_predictions(
                date_str,
                scoreboard,
                analyzer,
                espn=espn,
                odds_coll=odds_coll,
                force=True,
                only_game_ids=alerted_gids,
                precomputed_analyses=candidate_analyses,
                record_source='alert',
                published_at=published_at
            )
            if n_rec > 0:
                logger.info(f"Locked and recorded {n_rec} published prediction(s) into ledger.")
        except Exception as exc:
            logger.warning(f"Error recording published slate predictions into ledger: {exc}")

        return len(alerted_gids)
    else:
        logger.error("Failed to deliver alert via Telegram.")
        return 0


def main():
    parser = argparse.ArgumentParser(description="NBA 15-Minute Pre-Game Alert Dispatcher")
    parser.add_argument('--date', type=str, default=None, help="Target date YYYY-MM-DD (default: today in Thai time)")
    parser.add_argument('--dry-run', action='store_true', help="Preview output without sending Telegram message")
    parser.add_argument('--force', action='store_true', help="Force evaluate upcoming games regardless of 8-25m window")
    parser.add_argument('--test-message', action='store_true', help="Send test ping to Telegram to verify setup")
    parser.add_argument('--window-min', type=int, default=8, help="Min minutes before tip-off (default: 8)")
    parser.add_argument('--window-max', type=int, default=25, help="Max minutes before tip-off (default: 25)")
    parser.add_argument('--dashboard-url', type=str, default=DEFAULT_DASHBOARD_URL, help="URL to Streamlit Dashboard")

    args = parser.parse_args()

    if args.test_message:
        notifier = TelegramNotifier()
        if not notifier.is_configured:
            print("❌ Error: TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set.")
            sys.exit(1)
        test_txt = (
            "🏀 <b>NBA Analytics Bot Connected!</b>\n"
            "ระบบแจ้งเตือนผลวิเคราะห์บาสเกตบอลล่วงหน้า 15 นาทีก่อนแข่งพร้อมทำงานเรียบร้อยครับ! 🔔"
        )
        if notifier.send_message(test_txt, disable_notification=False):
            print("✅ Test notification dispatched successfully with sound!")
            sys.exit(0)
        else:
            print("❌ Failed to send test notification. Check your Bot Token and Chat ID.")
            sys.exit(1)

    dispatched = run_alert_cycle(
        target_date_str=args.date,
        dry_run=args.dry_run,
        force=args.force,
        window_min=args.window_min,
        window_max=args.window_max,
        dashboard_url=args.dashboard_url
    )
    print(f"Cycle completed. Games processed: {dispatched}")


if __name__ == '__main__':
    main()
