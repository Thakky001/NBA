import streamlit as st
import pandas as pd
from datetime import datetime, timezone, timedelta
import sys
import os
import re
import logging
import warnings

# Suppress framework deprecation warnings to keep logs clean
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("streamlit.deprecation_util").setLevel(logging.ERROR)

THAI_TZ = timezone(timedelta(hours=7))

# Ensure current directory is at the beginning of sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from typing import Any, Dict, Optional
import importlib

# Resilient imports with auto-recovery for Streamlit Cloud hot-reloading race conditions
try:
    from collectors.espn_collector import ESPNCollector
    from collectors.odds_collector import OddsCollector
    from processing.feature_engineering import FeatureEngineer
    from analysis.trend_analyzer import TrendAnalyzer
except (ImportError, KeyError, AttributeError):
    for mod in ['collectors', 'collectors.espn_collector', 'collectors.odds_collector',
                'processing', 'processing.feature_engineering',
                'analysis', 'analysis.trend_analyzer']:
        sys.modules.pop(mod, None)
    from collectors.espn_collector import ESPNCollector
    from collectors.odds_collector import OddsCollector
    from processing.feature_engineering import FeatureEngineer
    from analysis.trend_analyzer import TrendAnalyzer

try:
    from analysis.betting_math import calculate_live_pace_projection, evaluate_live_pick
except (ImportError, KeyError, AttributeError):
    try:
        sys.modules.pop('analysis.betting_math', None)
        from analysis.betting_math import calculate_live_pace_projection, evaluate_live_pick
    except Exception:
        evaluate_live_pick = None
        def calculate_live_pace_projection(
            period: Any,
            clock_str: Any,
            home_score: Any,
            away_score: Any
        ) -> Optional[Dict[str, Any]]:
            """Fallback live pace projection engine."""
            try:
                if home_score is None or away_score is None:
                    return None
                h_pts = float(home_score)
                a_pts = float(away_score)
                total_pts = h_pts + a_pts
                if total_pts <= 0:
                    return None
                p_int = int(period) if period else 1
                raw_clock = str(clock_str).strip() if clock_str is not None else ""
                clock_lower = raw_clock.lower()
                phase = "IN_PLAY"
                status_label = f"Q{p_int} {raw_clock}" if p_int <= 4 else f"OT{p_int - 4} {raw_clock}"
                elapsed_total = 0.0

                if "half" in clock_lower or (p_int == 2 and raw_clock in ["0.0", "0:00", "0", ""]):
                    phase, status_label, elapsed_total = "HALFTIME", "ช่วงพักครึ่ง (Halftime)", 24.0
                elif "end of 1" in clock_lower or (p_int == 1 and raw_clock in ["0.0", "0:00", "0"]):
                    phase, status_label, elapsed_total = "QUARTER_BREAK", "จบควอเตอร์ 1 (End of Q1)", 12.0
                elif "end of 2" in clock_lower:
                    phase, status_label, elapsed_total = "HALFTIME", "ช่วงพักครึ่ง (Halftime)", 24.0
                elif "end of 3" in clock_lower or (p_int == 3 and raw_clock in ["0.0", "0:00", "0"]):
                    phase, status_label, elapsed_total = "QUARTER_BREAK", "จบควอเตอร์ 3 (End of Q3)", 36.0
                elif "end of 4" in clock_lower or (p_int == 4 and raw_clock in ["0.0", "0:00", "0"]):
                    phase, status_label, elapsed_total = "REGULATION_END", "จบควอเตอร์ 4 (End of Q4)", 48.0
                else:
                    if ":" in raw_clock:
                        mins, secs = map(float, raw_clock.split(":"))
                        rem = mins + secs / 60.0
                    else:
                        rem = float(raw_clock) / 60.0
                    p_len = 12.0 if p_int <= 4 else 5.0
                    elap = max(0.0, min(p_len, p_len - rem))
                    if p_int <= 4:
                        elapsed_total = (p_int - 1) * 12.0 + elap
                    else:
                        phase, elapsed_total = "OVERTIME", 48.0 + (p_int - 5) * 5.0 + elap

                if elapsed_total <= 0.5:
                    return None
                expected_total = 48.0 if p_int <= 4 else 48.0 + (p_int - 4) * 5.0
                pace = expected_total / elapsed_total
                proj_h = h_pts * pace
                proj_a = a_pts * pace
                return {
                    'elapsed_mins': round(elapsed_total, 1),
                    'target_mins': round(expected_total, 1),
                    'proj_total': round(proj_h + proj_a, 1),
                    'proj_margin': round(proj_h - proj_a, 1),
                    'proj_home': round(proj_h, 1),
                    'proj_away': round(proj_a, 1),
                    'scoring_rate': round(total_pts / elapsed_total, 2),
                    'phase': phase,
                    'status_label': status_label
                }
            except Exception:
                return None

# Resilient import for prediction_tracker with auto-reload
try:
    from analysis.prediction_tracker import (
        load_ledger,
        grade_finished_games,
        calculate_track_record,
        record_slate_predictions,
    )
except (ImportError, AttributeError):
    try:
        import analysis.prediction_tracker
        importlib.reload(analysis.prediction_tracker)
        from analysis.prediction_tracker import (
            load_ledger,
            grade_finished_games,
            calculate_track_record,
            record_slate_predictions,
        )
    except Exception:
        def load_ledger(): return pd.DataFrame()
        def grade_finished_games(*a, **k): return 0
        def calculate_track_record(*a, **k): return {}
        def record_slate_predictions(*a, **k): return 0
from config import parse_spread, calculate_home_spread, resolve_team_info, normalize_team_name, t_rec

def t_dir(direction, factor_type):
    if factor_type == 'ou':
        return "📈" if direction == 'over' else "📉"
    else:
        return "🏠" if direction == 'home' else "✈️" if direction == 'away' else "➖"

# Page config
st.set_page_config(page_title="NBA Statistical Trends & Edge Engine", layout="wide", page_icon="🏀")

# ซ่อนปุ่ม Deploy ที่มุมขวาบน
st.markdown("""
    <style>
    .stAppDeployButton {display:none;}
    </style>
""", unsafe_allow_html=True)

st.title("🏀 ระบบวิเคราะห์สถิติและแต้มต่อ NBA (Advanced Statistical Trends & Edge Engine)")
st.markdown("ระบบวิเคราะห์เชิงสถิติขั้นสูง คำนวณจากฟอร์มการเล่น ความเร็วเกม (Pace) สภาพความพร้อมผู้เล่นบาดเจ็บ และราคาต่อรองสด (ESPN / The Odds API)")

@st.cache_resource
def get_espn_collector():
    return ESPNCollector()

@st.cache_resource
def get_odds_collector():
    return OddsCollector()

espn = get_espn_collector()
odds_coll = get_odds_collector()



# Sidebar
st.sidebar.header("📅 ตั้งค่าการค้นหา")
today_thai = datetime.now(THAI_TZ).date()
selected_date = st.sidebar.date_input("เลือกวันที่แข่งขัน (เวลาไทย)", today_thai)

auto_refresh = st.sidebar.toggle(
    "🔄 Live Auto-refresh (10s)",
    value=True,
    help="อัปเดตสกอร์สดและราคาต่อรองอัตโนมัติทุกๆ 10 วินาที"
)

st.sidebar.markdown("---")
if st.sidebar.button("🔄 อัปเดตฐานข้อมูลสถิติ (Refresh)"):
    progress = st.sidebar.progress(0, text="กำลังเตรียมการอัปเดตข้อมูล...")
    try:
        from main import _load_game_history
        from collectors.espn_collector import ESPNCollector
        from collectors.odds_collector import OddsCollector
        from config import resolve_team_info
        
        progress.progress(15, text="1/5 กำลังดึงผลการแข่งขันและตารางจาก ESPN...")
        collector_espn = ESPNCollector()
        _load_game_history(collector_espn)

        progress.progress(35, text="2/5 กำลังอัปเดตรายชื่อตัวเจ็บ (Injuries) ล่าสุดจาก ESPN...")
        collector_espn.get_injuries()

        progress.progress(55, text="3/5 กำลังดึงรายชื่อผู้เล่นสด (Active Rosters) ของทีมที่มีแข่งวันนี้...")
        sb_today = collector_espn.get_scoreboard_by_thai_date(selected_date)
        if not sb_today.empty:
            for _, g in sb_today.iterrows():
                for t_name in [g.get('home_team'), g.get('away_team')]:
                    t_info = resolve_team_info(t_name)
                    if t_info and 'espn_id' in t_info:
                        try:
                            collector_espn.get_team_roster(int(t_info['espn_id']))
                        except Exception:
                            pass
        
        progress.progress(75, text="4/5 กำลังอัปเดตราคาต่อรองล่าสุด...")
        collector_odds = OddsCollector()
        collector_odds.get_odds_dataframe()
        
        progress.progress(90, text="5/5 กำลังล้างแคชและประมวลผลโมเดลใหม่...")
        st.cache_data.clear()
        st.cache_resource.clear()
        
        progress.progress(100, text="อัปเดตข้อมูลสำเร็จเรียบร้อย!")
        st.sidebar.success("อัปเดตข้อมูลสำเร็จ!")
        st.rerun()
    except Exception as exc:
        st.sidebar.error(f"อัปเดตข้อมูลล้มเหลว: {exc}")

# Load Data with Streamlit Caching
@st.cache_resource
def get_analyzer(cache_version: str = "v9_early_season_conviction_and_auto_sync"):
    from processing.data_merger import DataMerger
    from processing.feature_engineering import FeatureEngineer
    from analysis.trend_analyzer import TrendAnalyzer

    merger = DataMerger()
    game_history = merger.load_all_games()
    
    # Check for advanced stats
    team_advanced = None
    if os.path.exists('data/raw/nba_api_advanced.csv'):
        team_advanced = pd.read_csv('data/raw/nba_api_advanced.csv')
        
    fe = FeatureEngineer(game_history, team_advanced=team_advanced)
    return TrendAnalyzer(fe)

@st.cache_data(ttl=10)
def get_cached_line_movement(game_id: str):
    """Fetch line movement with 10-second caching for real-time live odds."""
    if not game_id or not hasattr(espn, 'get_game_line_movement'):
        return {}
    return espn.get_game_line_movement(str(game_id))

@st.cache_data(ttl=300)
def get_cached_analysis(_analyzer, home_team, away_team, spread_line, over_under_line, open_spread, open_line, is_live=False):
    """Cache computationally heavy matchup simulation for 5 minutes."""
    try:
        return _analyzer.full_matchup_analysis(
            home_team=home_team, away_team=away_team, 
            spread_line=spread_line, over_under_line=over_under_line,
            open_spread=open_spread, open_line=open_line,
            is_live=is_live
        )
    except TypeError:
        # Fallback if _analyzer is an in-memory cached instance from prior code version
        return _analyzer.full_matchup_analysis(
            home_team=home_team, away_team=away_team, 
            spread_line=spread_line, over_under_line=over_under_line,
            open_spread=open_spread, open_line=open_line
        )

try:
    analyzer = get_analyzer()
except Exception as e:
    st.warning("⚠️ ไม่พบฐานข้อมูลสถิติ กรุณากด 'อัปเดตฐานข้อมูลสถิติ' ที่แถบด้านซ้ายก่อนใช้งานครับ")
    st.stop()

is_today = (selected_date == today_thai)
refresh_interval = 10 if (auto_refresh and is_today) else None

@st.fragment(run_every=refresh_interval)
def render_game_slate(selected_date, is_live_refresh):
    st.subheader(f"โปรแกรมการแข่งขันวันที่ {selected_date.strftime('%d/%m/%Y')} (เวลาประเทศไทย)")
    if is_live_refresh:
        now_str = datetime.now(THAI_TZ).strftime('%H:%M:%S')
        st.caption(f"⚡ **Live Sync Active:** กำลังอัปเดตข้อมูลสดทุก 10 วินาที *(ล่าสุดเมื่อ: `{now_str}` น.)*")


    with st.spinner("กำลังประมวลผลการวิเคราะห์..."):
        scoreboard = espn.get_scoreboard_by_thai_date(selected_date)
        if scoreboard.empty:
            st.info(f"ไม่มีโปรแกรมแข่งขันในวันที่ {selected_date.strftime('%d/%m/%Y')} (เวลาไทย) ครับ")
        else:
            # Auto-grade finished games (for today or past dates) & lock pre-game predictions for today only
            try:
                if selected_date <= today_thai:
                    grade_finished_games(scoreboard)
                if selected_date == today_thai:
                    record_slate_predictions(
                        selected_date.strftime('%Y-%m-%d'),
                        scoreboard,
                        analyzer,
                        espn=espn,
                        odds_coll=odds_coll,
                        max_minutes_before=60.0
                    )
            except Exception:
                pass

            def is_team_match(abbr, full_name):
                if not abbr or not full_name: return False
                abbr_info = resolve_team_info(abbr)
                full_info = resolve_team_info(full_name)
                if abbr_info and full_info:
                    return abbr_info.get('abbreviation') == full_info.get('abbreviation')
                return abbr.strip().lower() == full_name.strip().lower()

            for _, game in scoreboard.iterrows():
                raw_home = game.get('home_team', 'Unknown')
                raw_away = game.get('away_team', 'Unknown')
                home = normalize_team_name(raw_home) or raw_home
                away = normalize_team_name(raw_away) or raw_away
            
                # Fetch from cache
                odds_data = odds_coll.get_game_odds(home, away)
                c_spread, c_total = odds_coll.get_consensus_lines(odds_data)
            
                # Fallback to ESPN if missing
                raw_spread = game.get('spread')
                ou = game.get('over_under')
            
                sp_val = calculate_home_spread(raw_spread, home, away)
                ou_val = float(ou) if pd.notna(ou) else None
            
                # Prefer Odds API Consensus
                if c_spread is not None: sp_val = c_spread
                if c_total is not None: ou_val = c_total
            
                # Fetch Line Movement from ESPN (cached 10 min)
                gid = game.get('game_id')
                lm_data = get_cached_line_movement(str(gid)) if gid else {}
                open_total = lm_data.get('open_total')
                open_spread = lm_data.get('open_spread')

                # Fallback to ESPN pickcenter (close or open) if scoreboard/Odds API missing odds
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

                st_raw = str(game.get('status', ''))
                is_live = 'PROGRESS' in st_raw.upper() or 'LIVE' in st_raw.upper()

                # Analyze (with 5-minute cache to make 10s refreshes instant)
                analysis = get_cached_analysis(
                    analyzer,
                    home_team=home, away_team=away, 
                    spread_line=sp_val, over_under_line=ou_val,
                    open_spread=open_spread, open_line=open_total,
                    is_live=is_live
                )
            
                thai_time_str = game.get('thai_time')
                time_badge = f" ⏰ {thai_time_str} น." if thai_time_str else ""
                status_thai = ""
                if 'FINAL' in st_raw.upper():
                    status_thai = " [จบการแข่งขัน]"
                elif is_live:
                    status_thai = " [🔴 กำลังแข่งขัน]"
            
                with st.expander(f"🏀 {away} (เยือน) @ {home} (เหย้า){time_badge}{status_thai}", expanded=True):
                    # --- MATCH INFO (Kickoff time & Venue) ---
                    venue_str = game.get('venue') or 'TBD'
                    full_time_display = game.get('thai_time_display') or ''
                    info_col1, info_col2 = st.columns(2)
                    with info_col1:
                        if full_time_display:
                            st.markdown(f"🕒 **เวลาแข่งขัน (ไทย):** `{full_time_display}`")
                    with info_col2:
                        if venue_str:
                            st.markdown(f"📍 **สนาม:** {venue_str}")

                    # --- LIVE / FINAL SCORE BANNER ---
                    h_score = game.get('home_score')
                    a_score = game.get('away_score')
                    if pd.notna(h_score) and pd.notna(a_score) and str(h_score).strip() != '' and str(a_score).strip() != '':
                        try:
                            h_pts = int(float(h_score))
                            a_pts = int(float(a_score))
                            total_pts = h_pts + a_pts
                            lead = h_pts - a_pts
                            lead_str = f"{home} นำ +{lead}" if lead > 0 else (f"{away} นำ +{-lead}" if lead < 0 else "เสมอกัน")
                        
                            if 'PROGRESS' in st_raw.upper() or 'LIVE' in st_raw.upper():
                                c_detail = str(game.get('clock_detail') or (f"Q{game.get('period', '')} {game.get('clock', '')}".strip())).strip()
                                clock_str = f" (เวลา: `{c_detail}`)" if c_detail else ""
                                st.info(f"🔴 **สกอร์สด (Live Score):** **{away} {a_pts} - {h_pts} {home}**{clock_str} ({lead_str} | ผลรวมแต้มปัจจุบัน: **{total_pts}** PTS)")
                            elif 'FINAL' in st_raw.upper():
                                st.success(f"🏁 **ผลการแข่งขัน (Final):** **{away} {a_pts} - {h_pts} {home}** (ผลรวมจริง: **{total_pts}** PTS)")
                        except Exception:
                            pass
                
                    # --- TEAM PROFILES (Always visible even if no odds) ---
                    if 'home_profile' in analysis and 'away_profile' in analysis:
                        hp = analysis['home_profile']
                        ap = analysis['away_profile']
                        h_rec = f"{hp['wins']}-{hp['losses']}"
                        a_rec = f"{ap['wins']}-{ap['losses']}"
                        st.markdown(f"**ฟอร์มทีม:** {home} ({h_rec} | {hp['avg_points_scored']:.1f} PTS) 🆚 {away} ({a_rec} | {ap['avg_points_scored']:.1f} PTS)")

                    # --- INJURY BANNER ---
                    spread_dict = analysis.get('spread') or {}
                    ou_dict = analysis.get('over_under') or {}
                    feat = spread_dict.get('features') or ou_dict.get('features') or analysis.get('features') or {}
                    h_inj = feat.get('home_injuries', [])
                    a_inj = feat.get('away_injuries', [])
                    if h_inj or a_inj:
                        inj_notes = []
                        if h_inj: inj_notes.append(f"🏥 **{home}:** {', '.join(h_inj)}")
                        if a_inj: inj_notes.append(f"🏥 **{away}:** {', '.join(a_inj)}")
                        st.warning("  \n".join(inj_notes))
                
                    st.divider()

                    live_proj = calculate_live_pace_projection(
                        game.get('period'), game.get('clock'), h_score, a_score,
                        pregame_total=ou_val, pregame_spread=sp_val
                    ) if is_live else None
                
                    col1, col2 = st.columns(2)
                
                    # O/U Column
                    with col1:
                        st.markdown("### 📊 วิเคราะห์ สูง/ต่ำ (Over/Under)")
                        if is_live and live_proj:
                            st.markdown(f"**⚡ Live Projected Final PTS:** `{live_proj['proj_total']:.1f}` แต้ม")
                            tgt_mins = live_proj.get('target_mins', 48.0)
                            status_lbl = live_proj.get('status_label', '')
                            status_info = f" ({status_lbl} | {live_proj['elapsed_mins']:.1f}/{tgt_mins:.0f} นาที)" if status_lbl else f" *({live_proj['elapsed_mins']:.1f}/{tgt_mins:.0f} นาที)*"
                            st.caption(f"⏱️ **Live Scoring Pace:** `{live_proj['scoring_rate']:.2f}` แต้ม/นาที{status_info}")
                            if ou_val is not None:
                                diff = live_proj['proj_total'] - ou_val
                                diff_sign = "+" if diff > 0 else ""
                                direction = "สูงกว่าเรตก่อนแข่ง" if diff > 0 else "ต่ำกว่าเรตก่อนแข่ง"
                                st.caption(f"🎯 **เทียบเรตก่อนแข่ง ({ou_val:.1f}):** สปีดสด `{diff_sign}{diff:.1f}` แต้ม ({direction})")
                            live_t = lm_data.get('live_total')
                            if live_t is not None:
                                ldiff = live_proj['proj_total'] - live_t
                                ldiff_sign = "+" if ldiff > 0 else ""
                                ldir = "สูงกว่าเรตสด" if ldiff > 0 else "ต่ำกว่าเรตสด"
                                st.caption(f"🔴 **เทียบเรตสดในเกม ({live_t:.1f}):** สปีดสด `{ldiff_sign}{ldiff:.1f}` แต้ม ({ldir})")
                            st.divider()

                        if open_total is not None and ou_val is not None and abs(open_total - ou_val) >= 0.5:
                            st.markdown(f"**เรตราคาตลาด:** เปิด `{open_total:.1f}` ➔ ปัจจุบัน `{ou_val:.1f}` (ขยับ `{ou_val - open_total:+.1f}` แต้ม)")
                        elif ou_val is not None:
                            st.markdown(f"**เรตราคาตลาด (Consensus):** `{ou_val:.1f}`")
                        else:
                            st.markdown("**เรตราคาตลาด:** `ยังไม่ออกราคา`")
                    
                        ou_res = analysis.get('over_under')
                        if ou_res:
                            # In-play Live Evaluation: compare live projected final vs current live line
                            live_eval_ou = None
                            if is_live and live_proj and evaluate_live_pick:
                                live_t = lm_data.get('live_total')
                                effective_ou_line = live_t if live_t is not None else ou_val
                                rem_mins = live_proj.get('rem_mins', max(0.0, live_proj.get('target_mins', 48.0) - live_proj.get('elapsed_mins', 0.0)))
                                if effective_ou_line is not None:
                                    live_eval_ou = evaluate_live_pick(
                                        live_proj['proj_total'],
                                        current_line=effective_ou_line,
                                        bet_type='total',
                                        rem_mins=rem_mins
                                    )

                            if live_eval_ou:
                                rec_text = live_eval_ou['rec_text']
                                t_edge = live_eval_ou['edge']
                                win_p = live_eval_ou['win_prob']
                                ev_val = live_eval_ou['ev']
                                edge_pct_val = live_eval_ou['edge_pct']
                                q_kelly = live_eval_ou['quarter_kelly']
                                fair_p = live_eval_ou['fair_prob']
                                is_bet_live = live_eval_ou['is_bet']
                                rec_tag = " (Live In-Play)"
                            else:
                                rec_text = t_rec(ou_res['recommendation'])
                                t_edge = ou_res.get('total_edge', 0.0)
                                win_p = ou_res.get('win_prob', 0.5)
                                ev_val = ou_res.get('ev', 0.0)
                                edge_pct_val = ou_res.get('edge_pct', 0.0)
                                q_kelly = ou_res.get('kelly_quarter', 0.0)
                                fair_p = ou_res.get('fair_prob', 0.5)
                                is_bet_live = ou_res.get('recommendation') != 'NEUTRAL' and ev_val is not None
                                rec_tag = ""

                            is_clash_ou = ou_res.get('is_market_clash', False) and not is_live
                        
                            clash_tag = " *(⚠️ สัญญาณเงินตลาดสวนทาง)*" if is_clash_ou else ""
                            st.markdown(f"**คำแนะนำ{rec_tag}:** **{rec_text}** (Edge: `{t_edge:+.1f}` แต้ม){clash_tag}")
                            st.progress(win_p, text=f"ความน่าจะเป็นทางสถิติ (Win Prob): {win_p:.1%}")
                        
                            # --- LINE MOVEMENT BADGE & TWO-SIDED VERDICT ---
                            lm_ou = ou_res.get('line_movement')
                            if lm_ou and lm_ou.get('badge'):
                                if is_clash_ou:
                                    st.warning(f"**{lm_ou['badge']}:** {lm_ou['detail']}")
                                elif lm_ou.get('status') == 'SHARP_CONFIRMATION':
                                    st.success(f"**{lm_ou['badge']}:** {lm_ou['detail']}")
                                elif lm_ou.get('status') == 'STEAM_WARNING':
                                    st.warning(f"**{lm_ou['badge']}:** {lm_ou['detail']}")
                                else:
                                    st.caption(f"**{lm_ou['badge']}:** {lm_ou['detail']}")

                                # Two-sided comparison & Directional Verdict
                                if lm_ou.get('market_flow_prob') is not None and lm_ou.get('status') != 'NO_DATA':
                                    m_prob = lm_ou['market_flow_prob']
                                    m_dir = lm_ou.get('market_flow_direction', '')
                                    diff_val = lm_ou.get('diff', 0.0)
                                    diff_str = f" *(แรงไหล {diff_val:+.1f} แต้ม)*" if abs(diff_val) >= 0.5 else ""
                                
                                    st.markdown(f"""
                                    ⚖️ **เปรียบเทียบ 2 ฝั่ง:**
                                    * 📊 **ฝั่งโมเดลสถิติ:** {rec_text} (`{win_p:.1%}`)
                                    * 🌊 **ฝั่งกระแสเงินตลาด:** ดัน{m_dir} (`{m_prob:.1%}`){diff_str}
                                    """)
                                
                                    v_title = lm_ou.get('verdict_title', '')
                                    v_detail = lm_ou.get('verdict_detail', '')
                                    if is_clash_ou or lm_ou.get('verdict') in ['CAPITAL_PRESERVATION', 'PASS']:
                                        st.warning(f"🎯 **ฟันธง (Unified Verdict):** **ข้าม (PASS) / งดวางเงิน** — {v_detail or 'สัญญาณขัดแย้งรุนแรงหรือไม่มี Edge ที่คุ้มค่า แนะนำรักษาเงินทุน'}")
                                    elif lm_ou.get('verdict') == 'CONFIRMED_ALIGNED':
                                        st.success(f"🎯 **ฟันธง:** **{v_title}** — *{v_detail}*")
                                    else:
                                        st.info(f"🎯 **ฟันธง:** **{v_title}** — *{v_detail}*")
                        
                            # --- MARKET-ANCHORED PROJECTION INFO ---
                            if ou_res.get('raw_projected_total') is not None and ou_res.get('projected_total') is not None:
                                anchor_used = ou_res.get('anchor_line') or ou_val or 0.0
                                st.caption(f"🎯 **Market-Anchored Projection:** `{ou_res['projected_total']:.1f}` PTS *(คำนวณจาก: ราคาเปิด {anchor_used:.1f} × 65% + โมเดลดิบ {ou_res['raw_projected_total']:.1f} × 35%)*")
                        
                            # --- QUANTITATIVE VALUE & KELLY METRICS ---
                            if is_bet_live:
                                m_col1, m_col2 = st.columns(2)
                                with m_col1:
                                    ev_delta_color = "normal" if ev_val >= 0 else "inverse"
                                    st.metric("Expected Value (+EV)", f"{ev_val:+.1%}", delta=f"{edge_pct_val:+.1f}% Edge", delta_color=ev_delta_color)
                                with m_col2:
                                    st.metric("Market Fair Prob", f"{fair_p:.1%}", help="คำนวณผ่าน De-vigging ถอดค่าน้ำของตลาดออก (ราคาที่ตลาดมองว่ายุติธรรม)")
                                if is_clash_ou:
                                    st.info("💡 **คำแนะนำบริหารหน้าตัก (Kelly):** ลงทุน **0.0%** (ข้ามคู่ที่มีสัญญาณขัดแย้งกับตลาด)")
                                elif q_kelly > 0:
                                    st.info(f"💡 **คำแนะนำบริหารหน้าตัก{'สด ' if live_eval_ou else ''}(Kelly):** ลงทุน **{q_kelly*100:.1f}%** ของพอร์ต (1/4 Kelly)")
                        
                            if ou_res.get('is_anomaly'):
                                st.warning("⚠️ **ระวังข้อมูลผันผวน:** ส่วนต่างแต้มห่างจากราคาเปิดเกิน 8 แต้ม (ข้อมูลต้นฤดูกาลผันผวน)")
                        
                            # --- LINE SHOPPING O/U ---
                            if odds_data and ('STRONG' in ou_res['recommendation'] or 'LEAN' in ou_res['recommendation']):
                                is_over = 'OVER' in ou_res['recommendation'].upper()
                                b_book = "N/A"
                                b_val = -999 if is_over else 999
                                b_price = ""
                                for bookie, totals in odds_data.get('totals', {}).items():
                                    if is_over and 'over' in totals:
                                        o_val = totals['over']
                                        if o_val is not None and (b_val == -999 or o_val < b_val):
                                            b_val, b_book, b_price = o_val, bookie, totals.get('over_price', '')
                                    elif not is_over and 'under' in totals:
                                        u_val = totals['under']
                                        if u_val is not None and (b_val == 999 or u_val > b_val):
                                            b_val, b_book, b_price = u_val, bookie, totals.get('under_price', '')
                                if b_book != "N/A":
                                    if is_clash_ou:
                                        st.caption(f"🛒 **Line Watch:** ราคาตลาดดีที่สุดขณะนี้คือ {b_book.upper()} ({b_val}) — เฝ้าระวัง")
                                    else:
                                        st.success(f"🛒 **Line Shopping:** ควรแทง {'สูง' if is_over else 'ต่ำ'} {b_val} ที่ **{b_book.upper()}** (ค่าน้ำ {b_price})")
                        
                            with st.expander("อ่านเหตุผลประกอบ"):
                                for f in ou_res['factors']:
                                    icon = t_dir(f['direction'], 'ou')
                                    st.write(f"- {icon} **{f['name']}**: {f['value']}")
                        else:
                            st.write("ไม่สามารถวิเคราะห์ได้เนื่องจากไม่มีราคา")
                        
                    # Spread Column
                    with col2:
                        st.markdown("### ⚖️ วิเคราะห์ แต้มต่อ (Spread)")
                        if is_live and live_proj:
                            margin = live_proj['proj_margin']
                            proj_leader = home if margin > 0 else away
                            margin_abs = abs(margin)
                            st.markdown(f"**⚡ Live Projected Final Margin:** `{proj_leader} +{margin_abs:.1f}` แต้ม")
                            st.caption(f"🏀 **คาดการณ์สกอร์จบเกม:** `{away} {live_proj['proj_away']:.1f} - {live_proj['proj_home']:.1f} {home}`")
                            if sp_val is not None:
                                sp_str = f"{home} {sp_val:+.1f}" if sp_val != 0.0 else f"{home} PK (0.0)"
                                cover_diff = margin + sp_val
                                cover_team = home if cover_diff > 0 else away
                                st.caption(f"🎯 **เทียบแต้มต่อเปิด ({sp_str}):** สปีดสดชี้ `{cover_team}` ได้เปรียบ `{abs(cover_diff):.1f}` แต้ม")
                            live_s = lm_data.get('live_spread')
                            if live_s is not None:
                                lsp_str = f"{home} {live_s:+.1f}" if live_s != 0.0 else f"{home} PK (0.0)"
                                lcover_diff = margin + live_s
                                lcover_team = home if lcover_diff > 0 else away
                                st.caption(f"🔴 **เทียบแต้มต่อสดในเกม ({lsp_str}):** สปีดสดชี้ `{lcover_team}` ได้เปรียบ `{abs(lcover_diff):.1f}` แต้ม")
                            st.divider()

                        if sp_val is not None:
                            try:
                                s_val = float(sp_val)
                                if s_val < 0:
                                    role_display = f"🔴 **{home}** (ทีมต่อ `{s_val:+.1f}`) | 🟢 **{away}** (ทีมรอง `+{abs(s_val):.1f}`)"
                                elif s_val > 0:
                                    role_display = f"🔴 **{away}** (ทีมต่อ `{-s_val:+.1f}`) | 🟢 **{home}** (ทีมรอง `+{s_val:.1f}`)"
                                else:
                                    role_display = f"⚖️ **ราคาเสมอ (Pick'em):** `{home}` vs `{away}` ไม่มีใครต่อใครรอง (0.0)"
                            except Exception:
                                sp_str = f"{home} {sp_val:+.1f}" if sp_val != 0.0 else f"{home} PK (0.0)"
                                role_display = f"`{sp_str}`"
                            st.markdown(f"**บทบาทแต้มต่อ:** {role_display}")
                        else:
                            st.markdown("**บทบาทแต้มต่อ:** `ยังไม่ออกราคา`")
                    
                        sp_res = analysis.get('spread')
                        if sp_res:
                            # In-play Live Evaluation: compare live projected margin vs current live spread
                            live_eval_sp = None
                            if is_live and live_proj and evaluate_live_pick:
                                live_s = lm_data.get('live_spread')
                                effective_sp_line = live_s if live_s is not None else sp_val
                                rem_mins = live_proj.get('rem_mins', max(0.0, live_proj.get('target_mins', 48.0) - live_proj.get('elapsed_mins', 0.0)))
                                if effective_sp_line is not None:
                                    live_eval_sp = evaluate_live_pick(
                                        live_proj['proj_margin'],
                                        current_line=effective_sp_line,
                                        bet_type='spread',
                                        rem_mins=rem_mins,
                                        home_team=home,
                                        away_team=away
                                    )

                            if live_eval_sp:
                                rec_text = live_eval_sp['rec_text']
                                c_edge = live_eval_sp['edge']
                                win_p = live_eval_sp['win_prob']
                                ev_val = live_eval_sp['ev']
                                edge_pct_val = live_eval_sp['edge_pct']
                                q_kelly = live_eval_sp['quarter_kelly']
                                fair_p = live_eval_sp['fair_prob']
                                is_bet_live_sp = live_eval_sp['is_bet']
                                rec_tag = " (Live In-Play)"
                            else:
                                rec_text = t_rec(sp_res['recommendation'], spread_line=sp_val, home_team=home, away_team=away)
                                c_edge = sp_res.get('cover_edge', 0.0)
                                win_p = sp_res.get('win_prob', 0.5)
                                ev_val = sp_res.get('ev', 0.0)
                                edge_pct_val = sp_res.get('edge_pct', 0.0)
                                q_kelly = sp_res.get('kelly_quarter', 0.0)
                                fair_p = sp_res.get('fair_prob', 0.5)
                                is_bet_live_sp = sp_res.get('recommendation') != 'NEUTRAL' and ev_val is not None
                                rec_tag = ""

                            is_clash_sp = sp_res.get('is_market_clash', False) and not is_live
                        
                            clash_tag = " *(⚠️ สัญญาณเงินตลาดสวนทาง)*" if is_clash_sp else ""
                            st.markdown(f"🎯 **คำแนะนำ{rec_tag}:** **{rec_text}** (Cover Edge: `{c_edge:+.1f}` แต้ม){clash_tag}")
                            st.progress(win_p, text=f"ความน่าจะเป็นทางสถิติ (Win Prob): {win_p:.1%}")
                        
                            # --- LINE MOVEMENT BADGE & TWO-SIDED VERDICT ---
                            lm_sp = sp_res.get('line_movement')
                            if lm_sp and lm_sp.get('badge'):
                                if is_clash_sp:
                                    st.warning(f"**{lm_sp['badge']}:** {lm_sp['detail']}")
                                elif lm_sp.get('status') == 'SHARP_CONFIRMATION':
                                    st.success(f"**{lm_sp['badge']}:** {lm_sp['detail']}")
                                elif lm_sp.get('status') == 'STEAM_WARNING':
                                    st.warning(f"**{lm_sp['badge']}:** {lm_sp['detail']}")
                                else:
                                    st.caption(f"**{lm_sp['badge']}:** {lm_sp['detail']}")

                                # Two-sided comparison & Directional Verdict
                                if lm_sp.get('market_flow_prob') is not None and lm_sp.get('status') != 'NO_DATA':
                                    m_prob = lm_sp['market_flow_prob']
                                    m_dir = lm_sp.get('market_flow_direction', '')
                                    diff_val = lm_sp.get('diff', 0.0)
                                    diff_str = f" *(แรงไหล {diff_val:+.1f} แต้ม)*" if abs(diff_val) >= 0.5 else ""
                                
                                    st.markdown(f"""
                                    ⚖️ **เปรียบเทียบ 2 ฝั่ง:**
                                    * 📊 **ฝั่งโมเดลสถิติ:** {rec_text} (`{win_p:.1%}`)
                                    * 🌊 **ฝั่งกระแสเงินตลาด:** ดัน{m_dir} (`{m_prob:.1%}`){diff_str}
                                    """)
                                
                                    v_title = lm_sp.get('verdict_title', '')
                                    v_detail = lm_sp.get('verdict_detail', '')
                                    if is_clash_sp or lm_sp.get('verdict') in ['CAPITAL_PRESERVATION', 'PASS']:
                                        st.warning(f"🎯 **ฟันธง (Unified Verdict):** **ข้าม (PASS) / งดวางเงิน** — {v_detail or 'สัญญาณขัดแย้งรุนแรงหรือไม่มี Edge ที่คุ้มค่า แนะนำรักษาเงินทุน'}")
                                    elif lm_sp.get('verdict') == 'CONFIRMED_ALIGNED':
                                        st.success(f"🎯 **ฟันธง:** **{v_title}** — *{v_detail}*")
                                    else:
                                        st.info(f"🎯 **ฟันธง:** **{v_title}** — *{v_detail}*")
                        
                            # --- MARKET-ANCHORED PROJECTION INFO ---
                            if sp_res.get('raw_projected_margin') is not None and sp_res.get('projected_margin') is not None:
                                anchor_sp = sp_res.get('anchor_spread', sp_val) or 0.0
                                st.caption(f"🎯 **Market-Anchored Margin:** `{sp_res['projected_margin']:+.1f}` PTS *(คำนวณจาก: แต้มต่อเปิด {anchor_sp:+.1f} × 65% + โมเดลดิบ {sp_res['raw_projected_margin']:+.1f} × 35%)*")
                        
                            # --- QUANTITATIVE VALUE & KELLY METRICS ---
                            if is_bet_live_sp:
                                m_col1, m_col2 = st.columns(2)
                                with m_col1:
                                    ev_delta_color = "normal" if ev_val >= 0 else "inverse"
                                    st.metric("Expected Value (+EV)", f"{ev_val:+.1%}", delta=f"{edge_pct_val:+.1f}% Edge", delta_color=ev_delta_color)
                                with m_col2:
                                    st.metric("Market Fair Prob", f"{fair_p:.1%}", help="คำนวณผ่าน De-vigging ถอดค่าน้ำของตลาดออก (ราคาที่ตลาดมองว่ายุติธรรม)")
                                if is_clash_sp:
                                    st.info("💡 **คำแนะนำบริหารหน้าตัก (Kelly):** ลงทุน **0.0%** (ข้ามคู่ที่มีสัญญาณขัดแย้งกับตลาด)")
                                elif q_kelly > 0:
                                    st.info(f"💡 **คำแนะนำบริหารหน้าตัก{'สด ' if live_eval_sp else ''}(Kelly):** ลงทุน **{q_kelly*100:.1f}%** ของพอร์ต (1/4 Kelly)")
                        
                            if sp_res.get('is_anomaly'):
                                st.warning("⚠️ **ระวังข้อมูลผันผวน:** แต้มต่อห่างจากโมเดลเกิน 6 แต้ม (ข้อมูลต้นฤดูกาลผันผวน)")
                        
                            # --- LINE SHOPPING SPREAD ---
                            if odds_data and ('STRONG' in sp_res['recommendation'] or 'LEAN' in sp_res['recommendation']):
                                is_home = 'HOME' in sp_res['recommendation'].upper()
                                pick_team = home if is_home else away
                            
                                b_book = "N/A"
                                b_val = -999  # Always want highest spread
                                b_price = ""
                            
                                for bookie, spreads in odds_data.get('spreads', {}).items():
                                    if is_home and 'home' in spreads:
                                        h_val = spreads['home']
                                        if h_val is not None and h_val > b_val:
                                            b_val, b_book, b_price = h_val, bookie, spreads.get('home_price', '')
                                    elif not is_home and 'away' in spreads:
                                        a_val = spreads['away']
                                        if a_val is not None and a_val > b_val:
                                            b_val, b_book, b_price = a_val, bookie, spreads.get('away_price', '')
                                        
                                if b_book != "N/A":
                                    if is_clash_sp:
                                        st.caption(f"🛒 **Line Watch:** ราคาตลาดดีที่สุดขณะนี้คือ {pick_team} {b_val} ที่ {b_book.upper()} — เฝ้าระวัง")
                                    else:
                                        st.success(f"🛒 **Line Shopping:** ควรแทง {pick_team} {b_val} ที่ **{b_book.upper()}** (ค่าน้ำ {b_price})")
                        
                            with st.expander("อ่านเหตุผลประกอบ"):
                                for f in sp_res['factors']:
                                    icon = t_dir(f['direction'], 'sp')
                                    st.write(f"- {icon} **{f['name']}**: {f['value']}")
                        else:
                            st.write("ไม่สามารถวิเคราะห์ได้เนื่องจากไม่มีราคา")



def render_track_record_tab():
    st.subheader("🏆 สถิติความแม่นยำและการตรวจสอบผลย้อนหลัง (Track Record)")
    st.caption("ระบบบันทึกราคาปิด (Closing Line) และคำทำนายก่อนเริ่มแข่งอัตโนมัติ พร้อมตรวจผลสกอร์จริงเมื่อจบการแข่งขัน (`WIN` / `LOSS` / `PUSH`)")

    # Sync button row
    c_btn1, c_btn2 = st.columns([3, 1])
    with c_btn2:
        btn_clicked = False
        try:
            btn_clicked = st.button("🔄 ตรวจผลและอัปเดตสถิติทันที", width="stretch", key="btn_sync_ledger")
        except Exception:
            btn_clicked = st.button("🔄 ตรวจผลและอัปเดตสถิติทันที", key="btn_sync_ledger")
        if btn_clicked:
            with st.spinner("กำลังดึงผลการแข่งขันล่าสุดและอัปเดตสถานะ..."):
                g1 = grade_finished_games(espn.get_scoreboard_by_thai_date(today_thai))
                g2 = grade_finished_games(espn.get_scoreboard_by_thai_date(today_thai - timedelta(days=1)))
                st.success(f"ตรวจผลสำเร็จ! มีการอัปเดตผลใหม่ {g1 + g2} รายการ")
                st.rerun()

    ledger_df = load_ledger()
    if ledger_df.empty:
        st.info("ยังไม่มีข้อมูลบันทึกคำทำนายในระบบ (ระบบจะบันทึกอัตโนมัติก่อนเริ่มแข่ง 15-30 นาที)")
        return

    stats = calculate_track_record(ledger_df)

    # 4 Main KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        wr = stats['overall']['win_rate']
        st.metric(
            label="🎯 Overall Win Rate",
            value=f"{wr:.1f}%",
            delta=f"{stats['overall']['wins']}W - {stats['overall']['losses']}L ({stats['overall']['bets']} บิล)"
        )
    with col2:
        swr = stats['strong']['win_rate']
        st.metric(
            label="🔥 ไม้เน้น (Strong Picks)",
            value=f"{swr:.1f}%",
            delta=f"{stats['strong']['wins']}W - {stats['strong']['losses']}L ({stats['strong']['bets']} บิล)"
        )
    with col3:
        ou_wr = stats['over_under']['win_rate']
        sp_wr = stats['spread']['win_rate']
        st.metric(
            label="⚖️ O/U vs แต้มต่อ",
            value=f"{ou_wr:.0f}% / {sp_wr:.0f}%",
            delta=f"สูง-ต่ำ {stats['over_under']['wins']}W | แต้มต่อ {stats['spread']['wins']}W"
        )
    with col4:
        roi = stats['overall']['roi_pct']
        units = stats['overall']['net_units']
        d_color = "normal" if roi >= 0 else "inverse"
        st.metric(
            label="💰 ผลตอบแทนจำลอง (Flat ROI)",
            value=f"{roi:+.1f}%",
            delta=f"{units:+.2f} Units",
            delta_color=d_color
        )

    st.markdown("---")

    # Detailed Category Breakdown in expander
    with st.expander("📊 ดูรายละเอียดแยกตามหมวดหมู่ (Breakdown by Category)", expanded=True):
        b_cols = st.columns(4)
        with b_cols[0]:
            st.markdown("##### 🏀 สูง/ต่ำ (Over/Under)")
            ou = stats['over_under']
            st.write(f"- ชนะ/แพ้: **{ou['wins']}W - {ou['losses']}L - {ou['pushes']}P**")
            st.write(f"- Win Rate: **{ou['win_rate']:.1f}%**")
            st.write(f"- กำไร/ขาดทุน: **{ou['net_units']:+.2f} Units** (`{ou['roi_pct']:+.1f}% ROI`)")
        with b_cols[1]:
            st.markdown("##### 🛡️ แต้มต่อ (Spread)")
            sp = stats['spread']
            st.write(f"- ชนะ/แพ้: **{sp['wins']}W - {sp['losses']}L - {sp['pushes']}P**")
            st.write(f"- Win Rate: **{sp['win_rate']:.1f}%**")
            st.write(f"- กำไร/ขาดทุน: **{sp['net_units']:+.2f} Units** (`{sp['roi_pct']:+.1f}% ROI`)")
        with b_cols[2]:
            st.markdown("##### 📈 น่าสนใจ (Lean Picks)")
            ln = stats['lean']
            st.write(f"- ชนะ/แพ้: **{ln['wins']}W - {ln['losses']}L - {ln['pushes']}P**")
            st.write(f"- Win Rate: **{ln['win_rate']:.1f}%**")
            st.write(f"- กำไร/ขาดทุน: **{ln['net_units']:+.2f} Units** (`{ln['roi_pct']:+.1f}% ROI`)")
        with b_cols[3]:
            st.markdown("##### 🌊 ฟันธงกระแสเงิน (Verdict)")
            vd = stats['verdict']
            st.write(f"- ชนะ/แพ้: **{vd['wins']}W - {vd['losses']}L - {vd['pushes']}P**")
            st.write(f"- Win Rate: **{vd['win_rate']:.1f}%**")
            st.write(f"- กำไร/ขาดทุน: **{vd['net_units']:+.2f} Units** (`{vd['roi_pct']:+.1f}% ROI`)")

    st.markdown("### 📋 ตารางประวัติคำทำนายย้อนหลัง")

    # Filter controls
    f_c1, f_c2 = st.columns([2, 2])
    with f_c1:
        filter_status = st.selectbox("สถานะการแข่งขัน", ["ทั้งหมด", "ตรวจผลแล้ว (GRADED)", "รอผล (PENDING)"])
    with f_c2:
        filter_tier = st.selectbox("ระดับความมั่นใจ", ["ทั้งหมด", "เฉพาะไม้เน้น (🔥 STRONG)", "เฉพาะน่าสนใจ (LEAN)"])

    filtered_df = ledger_df.copy()
    if filter_status == "ตรวจผลแล้ว (GRADED)":
        filtered_df = filtered_df[filtered_df['status'] == 'GRADED']
    elif filter_status == "รอผล (PENDING)":
        filtered_df = filtered_df[filtered_df['status'] == 'PENDING']

    if filter_tier == "เฉพาะไม้เน้น (🔥 STRONG)":
        filtered_df = filtered_df[
            filtered_df['ou_recommendation'].astype(str).str.contains('STRONG', na=False) |
            filtered_df['sp_recommendation'].astype(str).str.contains('STRONG', na=False)
        ]
    elif filter_tier == "เฉพาะน่าสนใจ (LEAN)":
        filtered_df = filtered_df[
            filtered_df['ou_recommendation'].astype(str).str.contains('LEAN', na=False) |
            filtered_df['sp_recommendation'].astype(str).str.contains('LEAN', na=False)
        ]

    # Helper badge formatter
    def format_badge(res):
        if pd.isna(res) or res is None or str(res).strip() == '':
            return "⏳ รอผล"
        r = str(res).upper().strip()
        if r == 'WIN': return "🟢 WIN"
        if r == 'LOSS': return "🔴 LOSS"
        if r == 'PUSH': return "🟡 PUSH"
        if r == 'SKIPPED': return "⚪ ข้าม"
        return str(res)

    def t_verdict(v):
        mapping = {
            'FOLLOW_MARKET': '🌊 ตามกระแสเงินตลาด',
            'FOLLOW_MODEL': '🛡️ ยึดโมเดลสถิติ',
            'CONFIRMED_ALIGNED': '🔥 สถิติ & ตลาดตรงกัน',
            'NEUTRAL_WATCH': '⚖️ รอดูสถานการณ์',
        }
        val = str(v).strip()
        if not val or val == 'None' or val == 'nan':
            return '-'
        return mapping.get(val, val)

    def format_score(row):
        if pd.notna(row.get('home_score')) and pd.notna(row.get('away_score')):
            try:
                return f"{int(row['away_score'])} - {int(row['home_score'])}"
            except Exception:
                return f"{row['away_score']} - {row['home_score']}"
        return "⏳ รอแข่ง/ยังไม่จบ"

    display_rows = []
    for _, row in filtered_df.iterrows():
        ou_rec = str(row.get('ou_recommendation', ''))
        sp_rec = str(row.get('sp_recommendation', ''))
        ou_line = f"O/U {row.get('total_line')}" if pd.notna(row.get('total_line')) else "-"
        sp_line = f"{row.get('spread_line'):+g}" if pd.notna(row.get('spread_line')) else "-"

        display_rows.append({
            "วันที่": row.get('date', '-'),
            "คู่แข่งขัน": f"{row.get('away_team', '')} @ {row.get('home_team', '')}",
            "สกอร์จริง (เยือน-เหย้า)": format_score(row),
            "เรตราคาปิด": f"{ou_line} | ต่อ {sp_line}",
            "คำทำนาย สูง/ต่ำ": f"{t_rec(ou_rec)} [{format_badge(row.get('ou_result'))}]",
            "คำทำนาย แต้มต่อ": f"{t_rec(sp_rec, spread_line=row.get('spread_line'))} [{format_badge(row.get('sp_result'))}]",
            "ฟันธงกระแสเงิน": f"{t_verdict(row.get('ou_verdict'))} [{format_badge(row.get('verdict_result'))}]",
            "สถานะ": "✅ จบแล้ว" if row.get('status') == 'GRADED' else "⏳ รอผล"
        })

    if display_rows:
        table_df = pd.DataFrame(display_rows)
        st.dataframe(table_df, hide_index=True)
    else:
        st.info("ไม่พบรายการที่ตรงกับเงื่อนไขการค้นหา")

    # CSV Download
    csv_bytes = ledger_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 ดาวน์โหลดไฟล์ประวัติคำทำนาย (CSV)",
        data=csv_bytes,
        file_name="nba_predictions_ledger.csv",
        mime="text/csv"
    )


tab_live, tab_track = st.tabs(["🏀 โปรแกรม & วิเคราะห์สด", "🏆 สถิติความแม่นยำย้อนหลัง (Track Record)"])

with tab_live:
    render_game_slate(selected_date, auto_refresh and is_today)

with tab_track:
    render_track_record_tab()
