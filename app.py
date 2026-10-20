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
from config import parse_spread, calculate_home_spread, resolve_team_info

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
def get_analyzer(cache_version: str = "v5_market_anchored"):
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

@st.cache_data(ttl=600)
def get_cached_line_movement(game_id: str):
    """Fetch line movement with 10-minute caching to eliminate sequential HTTP latency."""
    if not game_id or not hasattr(espn, 'get_game_line_movement'):
        return {}
    return espn.get_game_line_movement(str(game_id))

try:
    analyzer = get_analyzer()
except Exception as e:
    st.warning("⚠️ ไม่พบฐานข้อมูลสถิติ กรุณากด 'อัปเดตฐานข้อมูลสถิติ' ที่แถบด้านซ้ายก่อนใช้งานครับ")
    st.stop()

st.subheader(f"โปรแกรมการแข่งขันวันที่ {selected_date.strftime('%d/%m/%Y')} (เวลาประเทศไทย)")

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

            # Analyze
            analysis = analyzer.full_matchup_analysis(
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
                
                col1, col2 = st.columns(2)
                
                # O/U Column
                with col1:
                    st.markdown("### 📊 วิเคราะห์ สูง/ต่ำ (Over/Under)")
                    st.markdown(f"**เรตราคาเปิดมาที่:** `{ou_val:.1f}`" if ou_val is not None else "**เรตราคาเปิดมาที่:** `ยังไม่ออกราคา`")
                    
                    ou_res = analysis.get('over_under')
                    if ou_res:
                        rec_text = t_rec(ou_res['recommendation'])
                        t_edge = ou_res.get('total_edge', 0.0)
                        win_p = ou_res.get('win_prob', 0.5)
                        
                        st.markdown(f"**คำแนะนำ:** **{rec_text}** (Edge: `{t_edge:+.1f}` แต้ม)")
                        st.progress(win_p, text=f"ความน่าจะเป็นทางสถิติ (Win Prob): {win_p:.1%}")
                        
                        # --- LINE MOVEMENT BADGE ---
                        lm_ou = ou_res.get('line_movement')
                        if lm_ou and lm_ou.get('badge'):
                            if lm_ou.get('status') == 'SHARP_CONFIRMATION':
                                st.success(f"**{lm_ou['badge']}:** {lm_ou['detail']}")
                            elif lm_ou.get('status') == 'STEAM_WARNING':
                                st.warning(f"**{lm_ou['badge']}:** {lm_ou['detail']}")
                            else:
                                st.caption(f"**{lm_ou['badge']}:** {lm_ou['detail']}")
                        
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
                                st.metric("Fair Probability", f"{fair_p:.1%}", help="คำนวณผ่าน De-vigging ถอดค่าน้ำของตลาดออก")
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
                        
                        # --- LINE MOVEMENT BADGE ---
                        lm_sp = sp_res.get('line_movement')
                        if lm_sp and lm_sp.get('badge'):
                            if lm_sp.get('status') == 'SHARP_CONFIRMATION':
                                st.success(f"**{lm_sp['badge']}:** {lm_sp['detail']}")
                            elif lm_sp.get('status') == 'STEAM_WARNING':
                                st.warning(f"**{lm_sp['badge']}:** {lm_sp['detail']}")
                            else:
                                st.caption(f"**{lm_sp['badge']}:** {lm_sp['detail']}")
                        
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
                                st.metric("Fair Probability", f"{fair_p:.1%}", help="คำนวณผ่าน De-vigging ถอดค่าน้ำของตลาดออก")
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
