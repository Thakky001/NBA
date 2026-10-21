import streamlit as st
import pandas as pd
from datetime import datetime, timezone, timedelta
import sys
import os
import re

THAI_TZ = timezone(timedelta(hours=7))

# Add current directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from collectors.espn_collector import ESPNCollector
from processing.feature_engineering import FeatureEngineer
from analysis.trend_analyzer import TrendAnalyzer
from config import parse_spread, calculate_home_spread, resolve_team_info, normalize_team_name

# Translation mapping for recommendations
def t_rec(rec):
    mapping = {
        'STRONG OVER': '🔥 สูง (มั่นใจมาก)', 
        'LEAN OVER': '📈 สูง (น่าสนใจ)',
        'STRONG UNDER': '🔥 ต่ำ (มั่นใจมาก)', 
        'LEAN UNDER': '📉 ต่ำ (น่าสนใจ)',
        'STRONG HOME': '🔥 ต่อเจ้าบ้าน (มั่นใจมาก)', 
        'LEAN HOME': '🏠 เจ้าบ้าน (น่าสนใจ)',
        'STRONG AWAY': '🔥 ต่อทีมเยือน (มั่นใจมาก)', 
        'LEAN AWAY': '✈️ ทีมเยือน (น่าสนใจ)',
        'NEUTRAL': '⚖️ ข้าม (สูสีมาก)',
        'NEUTRAL (slight home)': '⚖️ ข้าม (แต่แอบเอียงไปทางเจ้าบ้าน)',
        'NEUTRAL (slight away)': '⚖️ ข้าม (แต่แอบเอียงไปทางทีมเยือน)',
        'NEUTRAL (slight under)': '⚖️ ข้าม (แต่แอบเอียงไปทางต่ำ)',
        'NEUTRAL (slight over)': '⚖️ ข้าม (แต่แอบเอียงไปทางสูง)'
    }
    return mapping.get(rec, rec)

def t_dir(direction, factor_type):
    if factor_type == 'ou':
        return "📈" if direction == 'over' else "📉"
    else:
        return "🏠" if direction == 'home' else "✈️" if direction == 'away' else "➖"

def calculate_live_pace_projection(period, clock_str, home_score, away_score):
    """Calculate Pure Live Pace Projected Final PTS and Margin."""
    try:
        if home_score is None or away_score is None:
            return None
        h_pts = float(home_score)
        a_pts = float(away_score)
        total_pts = h_pts + a_pts
        if total_pts <= 0:
            return None
        period = int(period) if period else 1

        clock_str = str(clock_str).strip()
        if ':' in clock_str:
            mins, secs = map(float, clock_str.split(':'))
            rem_in_period = mins + secs / 60.0
        else:
            rem_in_period = float(clock_str) / 60.0

        period_length = 12.0 if period <= 4 else 5.0
        elapsed_in_period = max(0.0, min(period_length, period_length - rem_in_period))

        if period <= 4:
            elapsed_total = (period - 1) * 12.0 + elapsed_in_period
        else:
            elapsed_total = 48.0 + (period - 5) * 5.0 + elapsed_in_period

        if elapsed_total <= 0.5:
            return None

        pace_mult = 48.0 / elapsed_total
        proj_home = h_pts * pace_mult
        proj_away = a_pts * pace_mult
        proj_total = proj_home + proj_away
        proj_margin = proj_home - proj_away

        return {
            'elapsed_mins': elapsed_total,
            'proj_total': round(proj_total, 1),
            'proj_margin': round(proj_margin, 1),
            'proj_home': round(proj_home, 1),
            'proj_away': round(proj_away, 1),
            'scoring_rate': round(total_pts / elapsed_total, 2)
        }
    except Exception:
        return None

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

from collectors.espn_collector import ESPNCollector
from collectors.odds_collector import OddsCollector

espn = ESPNCollector()
odds_coll = OddsCollector()



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
        
        progress.progress(25, text="1/3 กำลังดึงผลการแข่งขันและตารางจาก ESPN...")
        collector_espn = ESPNCollector()
        _load_game_history(collector_espn)
        
        progress.progress(60, text="2/3 กำลังอัปเดตราคาต่อรองล่าสุด...")
        collector_odds = OddsCollector()
        collector_odds.get_odds_dataframe()
        
        progress.progress(90, text="3/3 กำลังล้างแคชและประมวลผลโมเดลใหม่...")
        st.cache_data.clear()
        st.cache_resource.clear()
        
        progress.progress(100, text="อัปเดตข้อมูลสำเร็จเรียบร้อย!")
        st.sidebar.success("อัปเดตข้อมูลสำเร็จ!")
        st.rerun()
    except Exception as exc:
        st.sidebar.error(f"อัปเดตข้อมูลล้มเหลว: {exc}")

# Load Data with Streamlit Caching
@st.cache_resource
def get_analyzer(cache_version: str = "v6_historical_dataset_restored"):
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
def get_cached_analysis(_analyzer, home_team, away_team, spread_line, over_under_line, open_spread, open_line):
    """Cache computationally heavy matchup simulation for 5 minutes."""
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

                # Analyze (with 5-minute cache to make 10s refreshes instant)
                analysis = get_cached_analysis(
                    analyzer,
                    home_team=home, away_team=away, 
                    spread_line=sp_val, over_under_line=ou_val,
                    open_spread=open_spread, open_line=open_total
                )
            
                thai_time_str = game.get('thai_time')
                time_badge = f" ⏰ {thai_time_str} น." if thai_time_str else ""
                status_thai = ""
                st_raw = str(game.get('status', ''))
                if 'FINAL' in st_raw.upper():
                    status_thai = " [จบการแข่งขัน]"
                elif 'PROGRESS' in st_raw.upper() or 'LIVE' in st_raw.upper():
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

                    is_live = 'PROGRESS' in st_raw.upper() or 'LIVE' in st_raw.upper()
                    live_proj = calculate_live_pace_projection(game.get('period'), game.get('clock'), h_score, a_score) if is_live else None
                
                    col1, col2 = st.columns(2)
                
                    # O/U Column
                    with col1:
                        st.markdown("### 📊 วิเคราะห์ สูง/ต่ำ (Over/Under)")
                        if is_live and live_proj:
                            st.markdown(f"**⚡ Live Projected Final PTS:** `{live_proj['proj_total']:.1f}` แต้ม")
                            st.caption(f"⏱️ **Live Scoring Pace:** `{live_proj['scoring_rate']:.2f}` แต้ม/นาที *(เล่นไปแล้ว {live_proj['elapsed_mins']:.1f}/48 นาที)*")
                            if ou_val is not None:
                                diff = live_proj['proj_total'] - ou_val
                                diff_sign = "+" if diff > 0 else ""
                                direction = "สูงกว่าราคาเปิด" if diff > 0 else "ต่ำกว่าราคาเปิด"
                                st.caption(f"🎯 **เทียบราคาเปิด ({ou_val:.1f}):** สปีดสด `{diff_sign}{diff:.1f}` แต้ม ({direction})")
                            live_t = lm_data.get('live_total')
                            if live_t is not None:
                                ldiff = live_proj['proj_total'] - live_t
                                ldiff_sign = "+" if ldiff > 0 else ""
                                ldir = "สูงกว่าเรตสด" if ldiff > 0 else "ต่ำกว่าเรตสด"
                                st.caption(f"🔴 **เทียบเรตสดในเกม ({live_t:.1f}):** สปีดสด `{ldiff_sign}{ldiff:.1f}` แต้ม ({ldir})")
                            st.divider()

                        st.markdown(f"**เรตราคาเปิดมาที่:** `{ou_val:.1f}`" if ou_val is not None else "**เรตราคาเปิดมาที่:** `ยังไม่ออกราคา`")
                    
                        ou_res = analysis.get('over_under')
                        if ou_res:
                            rec_text = t_rec(ou_res['recommendation'])
                            t_edge = ou_res.get('total_edge', 0.0)
                            win_p = ou_res.get('win_prob', 0.5)
                        
                            st.markdown(f"**คำแนะนำ:** **{rec_text}** (Edge: `{t_edge:+.1f}` แต้ม)")
                            st.progress(win_p, text=f"ความน่าจะเป็นทางสถิติ (Win Prob): {win_p:.1%}")
                        
                            # --- LINE MOVEMENT BADGE & TWO-SIDED VERDICT ---
                            lm_ou = ou_res.get('line_movement')
                            if lm_ou and lm_ou.get('badge'):
                                if lm_ou.get('status') == 'SHARP_CONFIRMATION':
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
                                    if lm_ou.get('verdict') == 'FOLLOW_MARKET':
                                        st.error(f"🎯 **ฟันธง:** **{v_title}** — *{v_detail}*")
                                    elif lm_ou.get('verdict') == 'CONFIRMED_ALIGNED':
                                        st.success(f"🎯 **ฟันธง:** **{v_title}** — *{v_detail}*")
                                    else:
                                        st.info(f"🎯 **ฟันธง:** **{v_title}** — *{v_detail}*")
                        
                            # --- MARKET-ANCHORED PROJECTION INFO ---
                            if ou_res.get('raw_projected_total') is not None and ou_res.get('projected_total') is not None:
                                st.caption(f"🎯 **Market-Anchored Projection:** `{ou_res['projected_total']:.1f}` PTS *(ตลาด 65% + สถิติ 35% | โมเดลดิบ: {ou_res['raw_projected_total']:.1f})*")
                        
                            # --- QUANTITATIVE VALUE & KELLY METRICS ---
                            if ou_res.get('recommendation') != 'NEUTRAL' and ou_res.get('ev') is not None:
                                ev_val = ou_res.get('ev', 0.0)
                                q_kelly = ou_res.get('kelly_quarter', 0.0)
                                fair_p = ou_res.get('fair_prob', 0.5)
                                m_col1, m_col2 = st.columns(2)
                                with m_col1:
                                    ev_delta_color = "normal" if ev_val >= 0 else "inverse"
                                    st.metric("Expected Value (+EV)", f"{ev_val:+.1%}", delta=f"{ou_res.get('edge_pct', 0.0):+.1f}% Edge", delta_color=ev_delta_color)
                                with m_col2:
                                    st.metric("Market Fair Prob", f"{fair_p:.1%}", help="คำนวณผ่าน De-vigging ถอดค่าน้ำของตลาดออก (ราคาที่ตลาดมองว่ายุติธรรม)")
                                if q_kelly > 0:
                                    st.info(f"💡 **คำแนะนำบริหารหน้าตัก (Kelly):** ลงทุน **{q_kelly*100:.1f}%** ของพอร์ต (1/4 Kelly)")
                        
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
                            sp_str = f"{home} {sp_val:+.1f}" if sp_val != 0.0 else f"{home} PK (0.0)"
                            st.markdown(f"**เรตราคาเปิดมาที่:** `{sp_str}`")
                        else:
                            st.markdown("**เรตราคาเปิดมาที่:** `ยังไม่ออกราคา`")
                    
                        sp_res = analysis.get('spread')
                        if sp_res:
                            rec_text = t_rec(sp_res['recommendation'])
                            c_edge = sp_res.get('cover_edge', 0.0)
                            win_p = sp_res.get('win_prob', 0.5)
                        
                            st.markdown(f"**คำแนะนำ:** **{rec_text}** (Cover Edge: `{c_edge:+.1f}` แต้ม)")
                            st.progress(win_p, text=f"ความน่าจะเป็นทางสถิติ (Win Prob): {win_p:.1%}")
                        
                            # --- LINE MOVEMENT BADGE & TWO-SIDED VERDICT ---
                            lm_sp = sp_res.get('line_movement')
                            if lm_sp and lm_sp.get('badge'):
                                if lm_sp.get('status') == 'SHARP_CONFIRMATION':
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
                                    if lm_sp.get('verdict') == 'FOLLOW_MARKET':
                                        st.error(f"🎯 **ฟันธง:** **{v_title}** — *{v_detail}*")
                                    elif lm_sp.get('verdict') == 'CONFIRMED_ALIGNED':
                                        st.success(f"🎯 **ฟันธง:** **{v_title}** — *{v_detail}*")
                                    else:
                                        st.info(f"🎯 **ฟันธง:** **{v_title}** — *{v_detail}*")
                        
                            # --- MARKET-ANCHORED PROJECTION INFO ---
                            if sp_res.get('raw_projected_margin') is not None and sp_res.get('projected_margin') is not None:
                                st.caption(f"🎯 **Market-Anchored Margin:** `{sp_res['projected_margin']:+.1f}` PTS *(ตลาด 65% + สถิติ 35% | โมเดลดิบ: {sp_res['raw_projected_margin']:+.1f})*")
                        
                            # --- QUANTITATIVE VALUE & KELLY METRICS ---
                            if sp_res.get('recommendation') != 'NEUTRAL' and sp_res.get('ev') is not None:
                                ev_val = sp_res.get('ev', 0.0)
                                q_kelly = sp_res.get('kelly_quarter', 0.0)
                                fair_p = sp_res.get('fair_prob', 0.5)
                                m_col1, m_col2 = st.columns(2)
                                with m_col1:
                                    ev_delta_color = "normal" if ev_val >= 0 else "inverse"
                                    st.metric("Expected Value (+EV)", f"{ev_val:+.1%}", delta=f"{sp_res.get('edge_pct', 0.0):+.1f}% Edge", delta_color=ev_delta_color)
                                with m_col2:
                                    st.metric("Market Fair Prob", f"{fair_p:.1%}", help="คำนวณผ่าน De-vigging ถอดค่าน้ำของตลาดออก (ราคาที่ตลาดมองว่ายุติธรรม)")
                                if q_kelly > 0:
                                    st.info(f"💡 **คำแนะนำบริหารหน้าตัก (Kelly):** ลงทุน **{q_kelly*100:.1f}%** ของพอร์ต (1/4 Kelly)")
                        
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
                                    st.success(f"🛒 **Line Shopping:** ควรแทง {pick_team} {b_val} ที่ **{b_book.upper()}** (ค่าน้ำ {b_price})")
                        
                            with st.expander("อ่านเหตุผลประกอบ"):
                                for f in sp_res['factors']:
                                    icon = t_dir(f['direction'], 'sp')
                                    st.write(f"- {icon} **{f['name']}**: {f['value']}")
                        else:
                            st.write("ไม่สามารถวิเคราะห์ได้เนื่องจากไม่มีราคา")


render_game_slate(selected_date, auto_refresh and is_today)
