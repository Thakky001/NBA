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
)
from collectors.espn_collector import ESPNCollector
from collectors.odds_collector import OddsCollector
from processing.data_merger import DataMerger
from processing.feature_engineering import FeatureEngineer
from analysis.trend_analyzer import TrendAnalyzer
from analysis.prediction_tracker import grade_finished_games, record_slate_predictions, load_ledger
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


def format_game_card(analysis: Dict[str, Any], game_info: Dict[str, Any], odds_data: Dict[str, Any]) -> str:
    """Format an actionable, compact HTML card for a single NBA game."""
    home = game_info.get('home_team', 'Home')
    away = game_info.get('away_team', 'Away')
    time_str = game_info.get('thai_time', 'N/A')
    venue = game_info.get('venue', 'TBD')

    ou_res = analysis.get('over_under') or {}
    sp_res = analysis.get('spread') or {}
    feat = sp_res.get('features') or ou_res.get('features') or analysis.get('features') or {}

    # 1. Injuries
    h_inj = feat.get('home_injuries', [])
    a_inj = feat.get('away_injuries', [])
    inj_parts = []
    if h_inj:
        inj_parts.append(f"{home}: {', '.join(h_inj[:2])}")
    if a_inj:
        inj_parts.append(f"{away}: {', '.join(a_inj[:2])}")
    inj_str = " | ".join(inj_parts) if inj_parts else "สภาพทีมสมบูรณ์ / ไม่มีตัวหลักเจ็บ"

    # 2. Lines
    sp_val = sp_res.get('features', {}).get('spread_line')
    ou_val = ou_res.get('features', {}).get('over_under_line')
    sp_text = f"{home} {sp_val:+g}" if sp_val is not None else "ยังไม่ออกราคา"
    ou_text = f"{ou_val:.1f}" if ou_val is not None else "ยังไม่ออกราคา"

    # 3. Recommendations
    sp_rec = sp_res.get('recommendation', 'NEUTRAL')
    ou_rec = ou_res.get('recommendation', 'NEUTRAL')

    picks = []
    if 'STRONG' in sp_rec or 'LEAN' in sp_rec:
        picks.append((sp_res, 'Spread'))
    if 'STRONG' in ou_rec or 'LEAN' in ou_rec:
        picks.append((ou_res, 'Total'))

    # Best pick summary
    card_lines = [
        f"🏀 <b>{away} @ {home}</b> (⏰ {time_str} น.)",
        f"📍 สนาม: {venue}",
        f"🏥 <i>{inj_str}</i>",
        f"📊 เรต: {sp_text} | O/U {ou_text}"
    ]

    if picks:
        for res, p_type in picks:
            rec = res.get('recommendation', '')
            edge = res.get('cover_edge') if p_type == 'Spread' else res.get('total_edge', 0.0)
            win_p = res.get('win_prob', 0.5)
            ev = res.get('ev', 0.0)
            q_kelly = res.get('kelly_quarter', 0.0)

            badge = "🔥" if "STRONG" in rec else "📈"
            card_lines.append(
                f"{badge} <b>{p_type}: {rec}</b> (Edge: <code>{edge:+.1f} pts</code> | Win Prob: <code>{win_p:.1%}</code>)"
            )
            if ev > 0:
                card_lines.append(f"  └ ⚡ <b>+EV:</b> <code>+{ev*100:.1f}%</code> | <b>Kelly:</b> <code>{q_kelly*100:.1f}%</code>")

            # Line shopping hint
            if p_type == 'Total' and odds_data:
                is_over = 'OVER' in rec.upper()
                b_book, b_val = None, None
                for bookie, totals in odds_data.get('totals', {}).items():
                    val = totals.get('over' if is_over else 'under')
                    if val is not None and (b_val is None or (val < b_val if is_over else val > b_val)):
                        b_val, b_book = val, bookie
                if b_book:
                    card_lines.append(f"  └ 🛒 <b>Line Shopping:</b> {b_book.upper()} ({b_val})")
    else:
        card_lines.append("⚖️ <b>คำแนะนำ:</b> เรตราคาเปิดมาสมดุล (NEUTRAL)")

    return "\n".join(card_lines)


def run_alert_cycle(
    target_date_str: Optional[str] = None,
    dry_run: bool = False,
    force: bool = False,
    window_min: int = 20,
    window_max: int = 45,
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

    # Lazy load heavy models only when games qualify
    merger = DataMerger()
    game_history = merger.load_all_games()
    adv_path = os.path.join(DATA_DIR, 'raw', 'nba_api_advanced.csv')
    team_adv = pd.read_csv(adv_path) if os.path.exists(adv_path) else None
    fe = FeatureEngineer(game_history, team_advanced=team_adv)
    analyzer = TrendAnalyzer(fe)

    # 2. Record pre-game predictions in ledger (locks lines & picks before tip-off)
    try:
        n_rec = record_slate_predictions(date_str, scoreboard, analyzer, espn=espn, odds_coll=odds_coll)
        if n_rec > 0:
            logger.info(f"Locked and recorded {n_rec} pre-game prediction(s) into ledger.")
    except Exception as exc:
        logger.warning(f"Error recording slate predictions: {exc}")

    cards = []
    alerted_gids = []

    for item in candidate_games:
        game = item['game']
        gid = str(game.get('game_id', ''))
        raw_home = game.get('home_team', 'Unknown')
        raw_away = game.get('away_team', 'Unknown')
        home = normalize_team_name(raw_home) or raw_home
        away = normalize_team_name(raw_away) or raw_away

        # Odds & Line movement
        odds_data = odds_coll.get_game_odds(home, away)
        c_spread, c_total = odds_coll.get_consensus_lines(odds_data)
        raw_spread = game.get('spread')
        raw_ou = game.get('over_under')

        sp_val = calculate_home_spread(raw_spread, home, away)
        ou_val = float(raw_ou) if pd.notna(raw_ou) else None
        if c_spread is not None:
            sp_val = c_spread
        if c_total is not None:
            ou_val = c_total

        lm_data = espn.get_game_line_movement(gid) if gid else {}
        open_total = lm_data.get('open_total')
        open_spread = lm_data.get('open_spread')

        # Fallback to ESPN pickcenter if still None
        if sp_val is None:
            if lm_data.get('current_spread') is not None:
                sp_val = lm_data.get('current_spread')
            elif lm_data.get('open_spread') is not None:
                sp_val = lm_data.get('open_spread')
        if ou_val is None:
            if lm_data.get('current_total') is not None:
                ou_val = lm_data.get('current_total')
            elif lm_data.get('open_total') is not None:
                ou_val = lm_data.get('open_total')

        analysis = analyzer.full_matchup_analysis(
            home_team=home,
            away_team=away,
            spread_line=sp_val,
            over_under_line=ou_val,
            open_spread=open_spread,
            open_line=open_total
        )

        card = format_game_card(analysis, game.to_dict(), odds_data)
        cards.append(card)
        alerted_gids.append(gid)

    # Compose wave message
    header = (
        f"🔔 <b>[NBA PRE-GAME INTELLIGENCE]</b>\n"
        f"📅 ข้อมูลวิเคราะห์ล่วงหน้า 30 นาทีก่อนเริ่มแข่งขัน\n"
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
        logger.info(f"Dry-run complete for {len(alerted_gids)} games (no Telegram API call made).")
        return len(alerted_gids)

    # Dispatch to Telegram (Sound enabled 24/7 as requested)
    success = notifier.send_message(full_message, disable_notification=False)
    if success:
        mark_games_alerted(date_str, alerted_gids)
        logger.info(f"Successfully dispatched alert for {len(alerted_gids)} game(s): {alerted_gids}")
        return len(alerted_gids)
    else:
        logger.error("Failed to deliver alert via Telegram.")
        return 0


def main():
    parser = argparse.ArgumentParser(description="NBA 30-Minute Pre-Game Alert Dispatcher")
    parser.add_argument('--date', type=str, default=None, help="Target date YYYY-MM-DD (default: today in Thai time)")
    parser.add_argument('--dry-run', action='store_true', help="Preview output without sending Telegram message")
    parser.add_argument('--force', action='store_true', help="Force evaluate upcoming games regardless of 20-45m window")
    parser.add_argument('--test-message', action='store_true', help="Send test ping to Telegram to verify setup")
    parser.add_argument('--window-min', type=int, default=20, help="Min minutes before tip-off (default: 20)")
    parser.add_argument('--window-max', type=int, default=45, help="Max minutes before tip-off (default: 45)")
    parser.add_argument('--dashboard-url', type=str, default=DEFAULT_DASHBOARD_URL, help="URL to Streamlit Dashboard")

    args = parser.parse_args()

    if args.test_message:
        notifier = TelegramNotifier()
        if not notifier.is_configured:
            print("❌ Error: TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set.")
            sys.exit(1)
        test_txt = (
            "🏀 <b>NBA Analytics Bot Connected!</b>\n"
            "ระบบแจ้งเตือนผลวิเคราะห์บาสเกตบอลล่วงหน้า 30 นาทีก่อนแข่งพร้อมทำงานเรียบร้อยครับ! 🔔"
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
