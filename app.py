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

def render_html(html_str: str):
    """Safely renders raw HTML in Streamlit without 4-space markdown codeblock wrapping."""
    cleaned = "\n".join(line.lstrip() for line in html_str.strip().splitlines())
    if hasattr(st, 'html'):
        st.html(cleaned)
    else:
        st.markdown(cleaned, unsafe_allow_html=True)

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
        get_available_ledger_dates,
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
            get_available_ledger_dates,
        )
    except Exception:
        def load_ledger(): return pd.DataFrame()
        def grade_finished_games(*a, **k): return 0
        def calculate_track_record(*a, **k): return {}
        def record_slate_predictions(*a, **k): return 0
        def get_available_ledger_dates(*a, **k): return []
from config import parse_spread, calculate_home_spread, resolve_team_info, normalize_team_name, t_rec, DATA_DIR, RAW_DIR

def t_dir(direction, factor_type):
    if factor_type == 'ou':
        return "📈" if direction == 'over' else "📉"
    else:
        return "🏠" if direction == 'home' else "✈️" if direction == 'away' else "➖"

# Page config - Lock expanded sidebar for desktop view
st.set_page_config(
    page_title="NBA Statistical Trends & Edge Engine",
    layout="wide",
    page_icon="🏀",
    initial_sidebar_state="expanded"
)

# Design System & Responsive Architecture (frontend-design + better-layout)
st.markdown("""
    <style>
    /* 1. Global Reset & Layout Constraints */
    .stAppDeployButton { display: none !important; }
    html, body {
        overflow-x: hidden;
        scroll-behavior: smooth;
        scroll-padding-top: 4rem;
    }
    
    /* Remove wasteful default Streamlit top margin */
    .main .block-container {
        padding-top: 1.25rem !important;
        padding-bottom: 2rem !important;
        padding-left: 0.85rem !important;
        padding-right: 0.85rem !important;
        max-width: 1200px !important;
    }

    /* 2. Quant Typography & Tokens (frontend-design) */
    :root {
        --quant-hero-height: 94px;
        --quant-font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
        --quant-bg: #0B0F17;
        --quant-card-bg: #111827;
        --quant-card-border: rgba(255, 255, 255, 0.08);
        --quant-card-hover: rgba(59, 130, 246, 0.35);
        --quant-accent-blue: #3B82F6;
        --quant-accent-green: #10B981;
        --quant-accent-cyan: #06B6D4;
        --quant-accent-amber: #F59E0B;
        --quant-accent-rose: #F43F5E;
    }

    /* Monospaced Tabular Numbers */
    code, pre, [data-testid="stMetricValue"], [data-testid="stMetricDelta"] {
        font-family: var(--quant-font-mono) !important;
        font-variant-numeric: tabular-nums !important;
        letter-spacing: -0.02em;
    }

    /* Transparent Streamlit Header so sticky elements can reach top: 0 cleanly */
    header[data-testid="stHeader"] {
        background: transparent !important;
        pointer-events: none !important;
    }
    header[data-testid="stHeader"] button,
    header[data-testid="stHeader"] a {
        pointer-events: auto !important;
    }

    /* 3. Bespoke Sticky Quant Hero Component (frontend-design) */
    div[data-testid="stElementContainer"]:has(.quant-hero),
    div[data-testid="element-container"]:has(.quant-hero) {
        position: sticky !important;
        top: 0 !important;
        z-index: 998 !important;
    }
    .quant-hero {
        position: sticky !important;
        top: 0 !important;
        z-index: 998 !important;
        background: rgba(11, 15, 23, 0.96) !important;
        backdrop-filter: blur(14px) !important;
        -webkit-backdrop-filter: blur(14px) !important;
        border: 1px solid var(--quant-card-border);
        border-bottom: none !important;
        border-radius: 14px 14px 0 0 !important;
        padding: 0.75rem 1.15rem;
        margin-bottom: 0 !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
        position: relative;
        overflow: hidden;
    }
    .quant-hero::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, #3B82F6, #10B981, #06B6D4);
    }
    .quant-hero-badge {
        display: inline-flex;
        align-items: center;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 9999px;
        padding: 2px 10px;
        font-size: 0.70rem;
        font-weight: 700;
        color: #34D399;
        letter-spacing: 0.05em;
        margin-bottom: 0.25rem;
    }
    .quant-pulse-dot {
        display: inline-block;
        width: 6px;
        height: 6px;
        background: #10B981;
        border-radius: 50%;
        margin-right: 6px;
        box-shadow: 0 0 8px #10B981;
    }
    .quant-hero-title {
        font-size: 1.35rem;
        font-weight: 800;
        color: #F8FAFC;
        letter-spacing: -0.02em;
        line-height: 1.2;
        margin-bottom: 0.2rem;
    }
    .quant-hero-subtitle {
        font-size: 0.78rem;
        color: #94A3B8;
        line-height: 1.3;
    }

    /* 4. Matchday Status Bar */
    .matchday-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid var(--quant-card-border);
        border-radius: 10px;
        padding: 0.6rem 0.85rem;
        margin-bottom: 0.85rem;
    }
    .matchday-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #E2E8F0;
    }
    .matchday-sync {
        font-size: 0.76rem;
        font-weight: 600;
        color: #38BDF8;
        background: rgba(56, 189, 248, 0.1);
        border: 1px solid rgba(56, 189, 248, 0.25);
        padding: 2px 8px;
        border-radius: 6px;
    }

    /* 4.1 Match Meta Strip (Time & Venue) */
    .match-meta-strip {
        display: flex;
        flex-wrap: wrap;
        gap: 8px 16px;
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid var(--quant-card-border);
        border-radius: 8px;
        padding: 6px 12px;
        font-size: 0.82rem;
        color: #94A3B8;
        margin-bottom: 0.75rem;
        align-items: center;
    }
    .match-meta-strip span.time-highlight {
        color: #38BDF8;
        font-family: var(--quant-font-mono);
        font-weight: 600;
    }

    /* 4.2 Sports Scoreboard Banner */
    .scoreboard-banner {
        background: linear-gradient(180deg, rgba(30, 41, 59, 0.75) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.85rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.35);
    }
    .scoreboard-status-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
        padding-bottom: 0.4rem;
    }
    .scoreboard-badge {
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.05em;
        padding: 2px 8px;
        border-radius: 4px;
        text-transform: uppercase;
    }
    .scoreboard-badge.final {
        background: rgba(148, 163, 184, 0.15);
        color: #CBD5E1;
        border: 1px solid rgba(148, 163, 184, 0.3);
    }
    .scoreboard-badge.live {
        background: rgba(239, 68, 68, 0.2);
        color: #F87171;
        border: 1px solid #EF4444;
        animation: pulse-glow 1.5s infinite;
    }
    .scoreboard-teams {
        display: flex;
        justify-content: space-around;
        align-items: center;
        gap: 12px;
        margin: 0.4rem 0;
    }
    .score-team-box {
        display: flex;
        flex-direction: column;
        align-items: center;
        flex: 1;
    }
    .score-team-name {
        font-size: 0.92rem;
        font-weight: 700;
        color: #E2E8F0;
        text-align: center;
    }
    .score-team-pts {
        font-size: 1.85rem;
        font-weight: 900;
        font-family: var(--quant-font-mono);
        color: #F8FAFC;
        line-height: 1.1;
    }
    .score-team-pts.winner {
        color: #34D399;
        text-shadow: 0 0 12px rgba(52, 211, 153, 0.35);
    }
    .score-vs-divider {
        font-size: 1.1rem;
        font-weight: 800;
        color: #475569;
    }
    .scoreboard-footer {
        display: flex;
        justify-content: space-between;
        font-size: 0.78rem;
        color: #94A3B8;
        padding-top: 0.4rem;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
    }
    .scoreboard-footer strong {
        color: #38BDF8;
        font-family: var(--quant-font-mono);
    }

    /* 4.3 Head-to-Head Team Form Grid (frontend-design + better-layout) */
    .team-form-grid {
        display: flex;
        align-items: center;
        gap: 8px;
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid var(--quant-card-border);
        border-radius: 10px;
        padding: 8px 10px;
        margin-bottom: 0.75rem;
    }
    .team-form-card {
        flex: 1;
        display: flex;
        flex-direction: column;
        gap: 4px;
        min-width: 0;
    }
    .team-form-card.home {
        text-align: right;
        align-items: flex-end;
    }
    .team-form-header {
        display: flex;
        align-items: center;
        gap: 6px;
        width: 100%;
    }
    .team-form-card.home .team-form-header {
        justify-content: flex-end;
    }
    .team-role-tag {
        font-size: 0.68rem;
        font-weight: 700;
        padding: 1px 6px;
        border-radius: 4px;
        background: rgba(148, 163, 184, 0.12);
        color: #94A3B8;
        white-space: nowrap;
        flex-shrink: 0;
    }
    .team-role-tag.home {
        background: rgba(59, 130, 246, 0.15);
        color: #60A5FA;
    }
    .team-name-label {
        font-size: 0.88rem;
        font-weight: 700;
        color: #F8FAFC;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* 4.6 Modern Quantitative Analysis Cards (Spread & Over/Under) */
    .analysis-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid var(--quant-card-border);
        border-radius: 12px;
        padding: 12px 14px;
        margin-bottom: 0.75rem;
        display: flex;
        flex-direction: column;
        gap: 10px;
    }
    .analysis-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 6px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
        padding-bottom: 6px;
    }
    .analysis-title {
        font-size: 0.94rem;
        font-weight: 700;
        color: #F8FAFC;
        display: flex;
        align-items: center;
        gap: 6px;
        white-space: nowrap;
    }
    .market-line-pill {
        font-family: var(--quant-font-mono);
        font-size: 0.76rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 6px;
        background: rgba(56, 189, 248, 0.12);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.25);
        white-space: nowrap;
    }
    .live-pace-banner {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(239, 68, 68, 0.12);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 6px;
        padding: 4px 8px;
        font-size: 0.76rem;
        color: #FCA5A5;
    }
    
    /* Structured Matchup Subrows (Spread & O/U) - Responsive & Clean */
    .matchup-roles-card {
        display: flex;
        flex-direction: column;
        gap: 5px;
        background: rgba(0, 0, 0, 0.22);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 6px 10px;
    }
    .matchup-role-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        min-height: 24px;
        gap: 8px;
    }
    .matchup-role-row:not(:last-child) {
        border-bottom: 1px dashed rgba(255, 255, 255, 0.06);
        padding-bottom: 4px;
    }
    .matchup-team-side {
        display: flex;
        align-items: center;
        gap: 8px;
        min-width: 0;
        overflow: hidden;
    }
    .role-pill {
        font-size: 0.72rem;
        font-weight: 700;
        padding: 2px 7px;
        border-radius: 4px;
        white-space: nowrap;
        flex-shrink: 0;
    }
    .role-pill.fav {
        background: rgba(239, 68, 68, 0.18);
        color: #FCA5A5;
        border: 1px solid rgba(239, 68, 68, 0.35);
    }
    .role-pill.dog {
        background: rgba(16, 185, 129, 0.18);
        color: #6EE7B7;
        border: 1px solid rgba(16, 185, 129, 0.35);
    }
    .role-pill.neutral {
        background: rgba(148, 163, 184, 0.15);
        color: #CBD5E1;
        border: 1px solid rgba(148, 163, 184, 0.3);
    }
    .matchup-team-name {
        font-size: 0.84rem;
        font-weight: 700;
        color: #F1F5F9;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .matchup-sub-desc {
        font-size: 0.74rem;
        color: #94A3B8;
        white-space: nowrap;
    }
    .matchup-point-val {
        font-family: var(--quant-font-mono);
        font-size: 0.88rem;
        font-weight: 800;
        white-space: nowrap;
        flex-shrink: 0;
    }
    .matchup-point-val.fav { color: #FCA5A5; }
    .matchup-point-val.dog { color: #6EE7B7; }
    .matchup-point-val.open { color: #CBD5E1; }
    .matchup-point-val.current { color: #38BDF8; }
    .trend-chip {
        font-size: 0.70rem;
        font-weight: 700;
        padding: 1px 6px;
        border-radius: 4px;
        white-space: nowrap;
        flex-shrink: 0;
    }
    .trend-chip.up {
        background: rgba(239, 68, 68, 0.18);
        color: #FCA5A5;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    .trend-chip.down {
        background: rgba(16, 185, 129, 0.18);
        color: #6EE7B7;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .trend-chip.flat {
        background: rgba(148, 163, 184, 0.15);
        color: #CBD5E1;
        border: 1px solid rgba(148, 163, 184, 0.25);
    }
    .hero-pick-box {
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.14), rgba(16, 185, 129, 0.08));
        border: 1px solid rgba(59, 130, 246, 0.35);
        border-radius: 10px;
        padding: 10px 12px;
        display: flex;
        flex-direction: column;
        gap: 6px;
    }
    .hero-pick-box.clash {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.12), rgba(239, 68, 68, 0.08));
        border-color: rgba(245, 158, 11, 0.35);
    }
    .hero-pick-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 6px;
    }
    .hero-pick-text {
        font-size: 1.02rem;
        font-weight: 800;
        color: #F8FAFC;
    }
    .hero-pick-badges {
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .confidence-pill {
        font-size: 0.72rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 4px;
        white-space: nowrap;
    }
    .confidence-pill.high {
        background: rgba(16, 185, 129, 0.2);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    .confidence-pill.medium {
        background: rgba(56, 189, 248, 0.2);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.4);
    }
    .confidence-pill.caution {
        background: rgba(245, 158, 11, 0.2);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.4);
    }
    .confidence-pill.neutral {
        background: rgba(148, 163, 184, 0.2);
        color: #94A3B8;
        border: 1px solid rgba(148, 163, 184, 0.4);
    }
    .edge-pill {
        font-family: var(--quant-font-mono);
        font-size: 0.74rem;
        font-weight: 700;
        padding: 2px 7px;
        border-radius: 4px;
        background: rgba(255, 255, 255, 0.06);
        color: #E2E8F0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        white-space: nowrap;
    }
    .win-prob-wrapper {
        display: flex;
        flex-direction: column;
        gap: 4px;
        margin-top: 2px;
    }
    .win-prob-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.76rem;
        color: #94A3B8;
        font-weight: 600;
    }
    .win-prob-header strong {
        color: #38BDF8;
        font-family: var(--quant-font-mono);
    }
    .custom-prob-bar {
        height: 6px;
        background: rgba(255, 255, 255, 0.08);
        border-radius: 999px;
        overflow: hidden;
        position: relative;
    }
    .custom-prob-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #0284C7, #10B981);
        transition: width 0.4s ease;
    }
    .custom-prob-fill.clash {
        background: linear-gradient(90deg, #F59E0B, #EF4444);
    }
    .market-callout {
        border-radius: 8px;
        padding: 8px 10px;
        font-size: 0.78rem;
        display: flex;
        flex-direction: column;
        gap: 6px;
    }
    .market-callout.warning {
        background: rgba(245, 158, 11, 0.08);
        border: 1px solid rgba(245, 158, 11, 0.25);
        color: #FDE68A;
    }
    .market-callout.success {
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.25);
        color: #A7F3D0;
    }
    .market-callout.info {
        background: rgba(56, 189, 248, 0.08);
        border: 1px solid rgba(56, 189, 248, 0.25);
        color: #BAE6FD;
    }
    .market-compare-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin-top: 3px;
    }
    .market-compare-col {
        flex: 1 1 135px;
        background: rgba(0, 0, 0, 0.25);
        padding: 5px 8px;
        border-radius: 5px;
        font-size: 0.74rem;
    }
    .verdict-banner {
        display: block;
        font-size: 0.82rem;
        line-height: 1.5;
        padding: 6px 10px;
        border-radius: 6px;
        margin-top: 4px;
        word-break: break-word;
        overflow-wrap: break-word;
    }
    .verdict-banner.pass {
        background: rgba(239, 68, 68, 0.15);
        color: #FCA5A5;
        border: 1px solid rgba(239, 68, 68, 0.35);
    }
    .verdict-banner.confirm {
        background: rgba(16, 185, 129, 0.15);
        color: #6EE7B7;
        border: 1px solid rgba(16, 185, 129, 0.35);
    }
    .verdict-badge {
        font-weight: 800;
        white-space: nowrap;
        display: inline-block;
        margin-right: 4px;
    }
    .verdict-title {
        font-weight: 700;
        display: inline;
    }
    .verdict-sep {
        color: rgba(255, 255, 255, 0.45);
        margin: 0 4px;
        display: inline;
    }
    .verdict-detail {
        font-weight: 400;
        opacity: 0.95;
        display: inline;
    }
    .model-metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(115px, 1fr));
        gap: 6px;
        background: rgba(255, 255, 255, 0.015);
        border: 1px solid rgba(255, 255, 255, 0.04);
        border-radius: 8px;
        padding: 6px 8px;
    }
    .model-metric-item {
        display: flex;
        flex-direction: column;
        gap: 2px;
    }
    .model-metric-label {
        font-size: 0.65rem;
        color: #94A3B8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    .model-metric-val {
        font-family: var(--quant-font-mono);
        font-size: 0.84rem;
        font-weight: 700;
        color: #F1F5F9;
    }
    .model-metric-val.pos { color: #34D399; }
    .model-metric-sub {
        font-size: 0.64rem;
        color: #64748B;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .team-stat-badges {
        display: flex;
        gap: 6px;
        flex-wrap: wrap;
    }
    .team-form-card.home .team-stat-badges {
        justify-content: flex-end;
    }
    .stat-chip {
        font-size: 0.76rem;
        font-family: var(--quant-font-mono);
        font-weight: 600;
        padding: 2px 7px;
        border-radius: 5px;
        white-space: nowrap;
        display: inline-flex;
        align-items: center;
    }
    .stat-chip.record {
        background: rgba(255, 255, 255, 0.06);
        color: #F1F5F9;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stat-chip.scoring {
        background: rgba(56, 189, 248, 0.1);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.2);
    }
    .team-form-vs-badge {
        font-size: 0.7rem;
        font-weight: 800;
        color: #64748B;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        width: 24px;
        height: 24px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 50%;
        flex-shrink: 0;
    }

    /* 4.4 Injury & Lineup Modern Card */
    .injury-card {
        background: rgba(245, 158, 11, 0.04);
        border: 1px solid rgba(245, 158, 11, 0.2);
        border-radius: 8px;
        padding: 8px 12px;
        margin-bottom: 0.75rem;
        font-size: 0.82rem;
    }
    .injury-card-title {
        font-weight: 700;
        color: #FBBF24;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 4px;
    }
    .inj-player-chip {
        display: inline-flex;
        align-items: center;
        background: rgba(239, 68, 68, 0.12);
        border: 1px solid rgba(239, 68, 68, 0.25);
        border-radius: 4px;
        padding: 1px 6px;
        margin: 2px 4px 2px 0;
        font-size: 0.78rem;
        color: #FCA5A5;
    }

    /* 4.5 Starting 5 Lineup Board (Position-ordered, No clutter) */
    .starting5-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid var(--quant-card-border);
        border-radius: 10px;
        padding: 8px 12px;
        margin-bottom: 0.65rem;
    }
    .starting5-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.84rem;
        font-weight: 700;
        color: #E2E8F0;
        margin-bottom: 6px;
        padding-bottom: 4px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    .starting5-list {
        display: flex;
        flex-direction: column;
        gap: 4px;
    }
    .starter-row {
        display: flex;
        align-items: center;
        gap: 8px;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.04);
        border-radius: 6px;
        padding: 4px 8px;
        transition: background 0.15s ease;
    }
    .starter-row:hover {
        background: rgba(255, 255, 255, 0.06);
    }
    .pos-badge {
        font-size: 0.72rem;
        font-weight: 800;
        font-family: var(--quant-font-mono);
        padding: 1px 6px;
        border-radius: 4px;
        min-width: 28px;
        text-align: center;
        flex-shrink: 0;
    }
    .pos-badge.pos-pg { background: rgba(2, 132, 199, 0.2); color: #38BDF8; border: 1px solid rgba(2, 132, 199, 0.4); }
    .pos-badge.pos-sg { background: rgba(14, 165, 233, 0.2); color: #7DD3FC; border: 1px solid rgba(14, 165, 233, 0.4); }
    .pos-badge.pos-sf { background: rgba(139, 92, 246, 0.2); color: #C4B5FD; border: 1px solid rgba(139, 92, 246, 0.4); }
    .pos-badge.pos-pf { background: rgba(245, 158, 11, 0.2); color: #FCD34D; border: 1px solid rgba(245, 158, 11, 0.4); }
    .pos-badge.pos-c  { background: rgba(16, 185, 129, 0.2); color: #6EE7B7; border: 1px solid rgba(16, 185, 129, 0.4); }
    .pos-badge.pos-g  { background: rgba(56, 189, 248, 0.2); color: #BAE6FD; border: 1px solid rgba(56, 189, 248, 0.4); }
    .pos-badge.pos-f  { background: rgba(168, 85, 247, 0.2); color: #E9D5FF; border: 1px solid rgba(168, 85, 247, 0.4); }
    .starter-name {
        font-size: 0.84rem;
        font-weight: 600;
        color: #F1F5F9;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* 5. Modern Pill Tabs - Sticky & Docked to Quant Hero (better-layout) */
    div[data-testid="stTabs"] {
        overflow: visible !important;
    }
    div[data-baseweb="tab-list"] {
        position: sticky !important;
        top: var(--quant-hero-height) !important;
        z-index: 997 !important;
        display: flex !important;
        width: 100% !important;
        background: rgba(11, 15, 23, 0.96) !important;
        backdrop-filter: blur(14px) !important;
        -webkit-backdrop-filter: blur(14px) !important;
        border-radius: 0 0 12px 12px !important;
        padding: 4px 6px !important;
        border: 1px solid var(--quant-card-border) !important;
        border-top: 1px dashed rgba(255, 255, 255, 0.08) !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.45) !important;
        gap: 6px !important;
        margin-top: 0 !important;
        margin-bottom: 1rem !important;
    }
    div[data-baseweb="tab"] {
        flex: 1 1 50% !important;
        justify-content: center !important;
        text-align: center !important;
        padding: 8px 12px !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        color: #94A3B8 !important;
        border: none !important;
        background: transparent !important;
        transition: all 0.2s ease !important;
    }
    div[data-baseweb="tab"][aria-selected="true"] {
        background: rgba(37, 99, 235, 0.25) !important;
        border: 1px solid rgba(59, 130, 246, 0.5) !important;
        color: #60A5FA !important;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.25) !important;
    }

    /* 6. Matchup Card Elevation (details & summary) */
    div[data-testid="stExpander"], details {
        border-radius: 12px !important;
        border: 1px solid var(--quant-card-border) !important;
        background-color: var(--quant-card-bg) !important;
        margin-bottom: 0.75rem !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.2) !important;
        overflow: hidden !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stExpander"]:hover, details:hover {
        border-color: var(--quant-card-hover) !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3) !important;
    }
    div[data-testid="stExpander"] summary, details summary {
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 0.75rem 1rem !important;
        background: rgba(255, 255, 255, 0.02) !important;
        color: #F8FAFC !important;
        cursor: pointer !important;
    }

    /* 7. Metric Containers */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid var(--quant-card-border) !important;
        border-radius: 10px !important;
        padding: 0.65rem 0.85rem !important;
    }

    /* ========================================================
       DESKTOP: Lock sidebar permanently open (min-width: 992px)
       ======================================================== */
    @media (min-width: 992px) {
        /* Hide all sidebar collapse buttons on desktop so user cannot close it */
        [data-testid="stSidebarCollapseButton"],
        [data-testid="stSidebarHeader"] button,
        section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"],
        section[data-testid="stSidebar"] button[aria-label*="sidebar" i],
        section[data-testid="stSidebar"] button[aria-label*="close" i],
        section[data-testid="stSidebar"] button[aria-label*="collapse" i],
        [data-testid="collapsedControl"] {
            display: none !important;
            visibility: hidden !important;
            pointer-events: none !important;
        }

        /* Permanently lock sidebar open and fixed width on desktop */
        section[data-testid="stSidebar"] {
            display: block !important;
            visibility: visible !important;
            transform: none !important;
            margin-left: 0 !important;
            min-width: 310px !important;
            max-width: 320px !important;
            width: 315px !important;
        }
    }

    /* 8. Mobile-First Layout & Touch Ergonomics (better-layout) */
    @media (max-width: 768px) {
        /* In game card analysis, stack Spread and Over/Under into full-width cards on mobile */
        div[data-testid="stHorizontalBlock"]:has(.analysis-card) {
            flex-direction: column !important;
            gap: 12px !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.analysis-card) > div[data-testid="column"] {
            width: 100% !important;
            min-width: 100% !important;
            flex: 1 1 100% !important;
        }
        .analysis-card {
            padding: 10px 10px !important;
            margin-bottom: 8px !important;
        }
        .analysis-title {
            font-size: 0.88rem !important;
        }
        .market-line-pill {
            font-size: 0.72rem !important;
            padding: 2px 6px !important;
        }
        .matchup-roles-card {
            padding: 5px 8px !important;
        }
        .matchup-team-name {
            font-size: 0.80rem !important;
        }
        .role-pill {
            font-size: 0.68rem !important;
            padding: 1px 5px !important;
        }
        .matchup-point-val {
            font-size: 0.84rem !important;
        }
        .hero-pick-text {
            font-size: 0.95rem !important;
        }
        .market-compare-col {
            font-size: 0.74rem !important;
            padding: 5px 8px !important;
        }
        .verdict-banner {
            font-size: 0.78rem !important;
            padding: 6px 8px !important;
        }
        :root {
            --quant-hero-height: 70px;
        }
        .quant-hero {
            padding: 0.45rem 0.85rem !important;
            border-radius: 10px 10px 0 0 !important;
        }
        .quant-hero-badge {
            font-size: 0.65rem !important;
            padding: 1px 6px !important;
            margin-bottom: 1px !important;
        }
        .quant-hero-title {
            font-size: 1.05rem !important;
            line-height: 1.15 !important;
            margin-bottom: 1px !important;
        }
        .quant-hero-subtitle {
            font-size: 0.68rem !important;
            line-height: 1.2 !important;
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
        }
        div[data-baseweb="tab-list"] {
            top: var(--quant-hero-height) !important;
            padding: 3px !important;
            border-radius: 0 0 10px 10px !important;
            gap: 4px !important;
            margin-bottom: 0.75rem !important;
        }
        div[data-baseweb="tab"] {
            font-size: 0.80rem !important;
            padding: 6px 8px !important;
        }

        /* Enforce Touch Target >= 44px on interactive controls */
        .stButton > button {
            min-height: 44px !important;
            padding: 0.6rem 1.1rem !important;
            font-size: 0.95rem !important;
            border-radius: 8px !important;
            width: 100% !important;
        }

        /* Compact metric padding for mobile */
        div[data-testid="stMetric"] {
            padding: 0.45rem 0.6rem !important;
            margin-bottom: 0.5rem !important;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.25rem !important;
        }
        div[data-testid="stMetricDelta"] {
            font-size: 0.78rem !important;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Sleek Custom Hero (frontend-design)
st.markdown("""
<div class="quant-hero">
    <div class="quant-hero-badge">
        <span class="quant-pulse-dot"></span>
        <span>QUANT SPORTS INTELLIGENCE • LIVE ENGINE</span>
    </div>
    <div class="quant-hero-title">🏀 NBA Statistical Edge Engine</div>
    <div class="quant-hero-subtitle">ระบบวิเคราะห์สถิติขั้นสูง • Pace & Efficiency • FiveThirtyEight MOV-Elo • CatBoost Residuals</div>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def get_espn_collector():
    return ESPNCollector()

@st.cache_resource
def get_odds_collector():
    return OddsCollector()

espn = get_espn_collector()
odds_coll = get_odds_collector()

@st.cache_data(ttl=3600)
def get_espn_position_map():
    pos_map = {}
    data_dir = globals().get('DATA_DIR') or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    p_path = os.path.join(data_dir, 'raw', 'espn_rosters.csv')
    if os.path.exists(p_path):
        try:
            df = pd.read_csv(p_path)
            for _, r in df.iterrows():
                p_name = str(r.get('name', '')).strip().lower()
                pos = str(r.get('position', '')).strip().upper()
                if p_name and pos and pos != 'NAN':
                    pos_map[p_name] = pos
                    # Suffix-stripped alias
                    clean_p = re.sub(r'[-_]', ' ', p_name)
                    clean_p = re.sub(r'[^\w\s]', '', clean_p)
                    clean_p = re.sub(r'\s+(?:jr|sr|ii|iii|iv|v)$', '', clean_p).strip()
                    clean_p = re.sub(r'\s+', ' ', clean_p)
                    if clean_p:
                        pos_map[clean_p] = pos
        except Exception:
            pass
    return pos_map

POS_ORDER = {'PG': 1, 'SG': 2, 'G': 2.5, 'SF': 3, 'F': 3.5, 'PF': 4, 'C': 5}

def format_and_sort_starters(starters_list, pos_map):
    formatted = []
    default_positions = ['PG', 'SG', 'SF', 'PF', 'C']
    for idx, s in enumerate(starters_list[:5]):
        name = s.get('name', '') if isinstance(s, dict) else str(s)
        pos = s.get('position', '').strip().upper() if isinstance(s, dict) else ''
        if not pos or pos not in POS_ORDER:
            clean_name = re.sub(r'[-_]', ' ', name.lower().strip())
            clean_name = re.sub(r'[^\w\s]', '', clean_name)
            clean_name = re.sub(r'\s+(?:jr|sr|ii|iii|iv|v)$', '', clean_name).strip()
            clean_name = re.sub(r'\s+', ' ', clean_name)
            raw_pos = pos_map.get(name.lower().strip()) or pos_map.get(clean_name, '')
            if 'POINT' in raw_pos or raw_pos == 'PG': pos = 'PG'
            elif 'SHOOTING' in raw_pos or raw_pos == 'SG': pos = 'SG'
            elif 'CENTER' in raw_pos or raw_pos == 'C': pos = 'C'
            elif 'POWER' in raw_pos or raw_pos == 'PF': pos = 'PF'
            elif 'SMALL' in raw_pos or raw_pos == 'SF': pos = 'SF'
            elif 'GUARD' in raw_pos or raw_pos == 'G': pos = 'PG' if idx == 0 else 'SG'
            elif 'FORWARD' in raw_pos or raw_pos == 'F': pos = 'SF' if idx <= 2 else 'PF'
            else:
                pos = default_positions[idx] if idx < len(default_positions) else 'G'
        formatted.append({'name': name, 'position': pos})
    formatted.sort(key=lambda x: POS_ORDER.get(x['position'], 9))
    return formatted


def _render_spread_analysis_card(home, away, sp_val, open_spread, sp_res, rec_text, c_edge, win_p, is_clash_sp, is_live=False, live_proj=None, lm_data=None):
    lm_data = lm_data or {}
    
    # 1. Consensus line pill (Compact with team abbreviation)
    if sp_val is not None:
        try:
            s_val = float(sp_val)
            h_info = resolve_team_info(home)
            h_abbr = h_info.get('abbreviation', home) if h_info else home
            sp_line_pill = f"{h_abbr} {s_val:+.1f}" if s_val != 0.0 else "Pick'em (0.0)"
        except Exception:
            sp_line_pill = str(sp_val)
    else:
        sp_line_pill = "ยังไม่ออกราคา"

    # 2. Roles Bar (Favorite vs Underdog - 2-row clean structured matchup)
    roles_html = ""
    if sp_val is not None:
        try:
            s_val = float(sp_val)
            if s_val < 0:
                fav_team, fav_pts = home, f"{s_val:+.1f}"
                dog_team, dog_pts = away, f"+{abs(s_val):.1f}"
            elif s_val > 0:
                fav_team, fav_pts = away, f"{-s_val:+.1f}"
                dog_team, dog_pts = home, f"+{s_val:.1f}"
            else:
                fav_team, fav_pts = home, "0.0"
                dog_team, dog_pts = away, "0.0"

            roles_html = f'''<div class="matchup-roles-card"><div class="matchup-role-row"><div class="matchup-team-side"><span class="role-pill fav">🔴 ทีมต่อ</span><span class="matchup-team-name">{fav_team}</span></div><span class="matchup-point-val fav">{fav_pts}</span></div><div class="matchup-role-row"><div class="matchup-team-side"><span class="role-pill dog">🟢 ทีมรอง</span><span class="matchup-team-name">{dog_team}</span></div><span class="matchup-point-val dog">{dog_pts}</span></div></div>'''
        except Exception:
            roles_html = f'''<div class="matchup-roles-card"><div class="matchup-role-row"><div class="matchup-team-side"><span class="role-pill neutral">⚖️ เรตราคา</span><span class="matchup-team-name">{sp_line_pill}</span></div></div></div>'''

    # 3. Live Pace Banner if in-play
    live_banner_html = ""
    if is_live and live_proj:
        margin = live_proj.get('proj_margin', 0.0)
        proj_leader = home if margin > 0 else away
        live_banner_html = f'''<div class="live-pace-banner"><span>⚡ <strong>Live Proj Margin:</strong> {proj_leader} +{abs(margin):.1f} PTS</span><span style="color:#94A3B8;">({away} {live_proj.get("proj_away",0):.0f} - {live_proj.get("proj_home",0):.0f} {home})</span></div>'''

    # 4. Hero Pick Box
    is_strong = 'STRONG' in sp_res.get('recommendation', '') or 'มั่นใจมาก' in rec_text
    is_lean = 'LEAN' in sp_res.get('recommendation', '') or 'น่าสนใจ' in rec_text
    
    if is_clash_sp:
        conf_badge = '<span class="confidence-pill caution">⚠️ สัญญาณขัดแย้ง</span>'
        box_class = "hero-pick-box clash"
        fill_class = "custom-prob-fill clash"
    elif is_strong:
        conf_badge = '<span class="confidence-pill high">🔥 มั่นใจมาก</span>'
        box_class = "hero-pick-box"
        fill_class = "custom-prob-fill"
    elif is_lean:
        conf_badge = '<span class="confidence-pill medium">⚡ น่าสนใจ</span>'
        box_class = "hero-pick-box"
        fill_class = "custom-prob-fill"
    else:
        conf_badge = '<span class="confidence-pill neutral">⚖️ รอดู / สูสี</span>'
        box_class = "hero-pick-box"
        fill_class = "custom-prob-fill"

    prob_pct = f"{win_p * 100:.1f}%"
    edge_chip = f'<span class="edge-pill">Cover Edge: {c_edge:+.1f} แต้ม</span>' if c_edge else ''
    clean_rec = rec_text.replace('(มั่นใจมาก)', '').replace('(น่าสนใจ)', '').replace('🔥', '').replace('🏠', '').replace('✈️', '').replace('🎯', '').strip()

    # 5. Market Flow & Line Movement Callout
    lm_sp = sp_res.get('line_movement') or {}
    market_html = ""
    if lm_sp and lm_sp.get('badge'):
        badge_txt = lm_sp.get('badge', '')
        detail_txt = lm_sp.get('detail', '')
        m_prob = lm_sp.get('market_flow_prob')
        m_dir = lm_sp.get('market_flow_direction', '')
        diff_val = lm_sp.get('diff', 0.0)
        v_title = lm_sp.get('verdict_title', '')
        v_detail = lm_sp.get('verdict_detail', '')

        compare_html = ""
        if m_prob is not None and lm_sp.get('status') != 'NO_DATA':
            diff_str = f" ({diff_val:+.1f})" if abs(diff_val) >= 0.5 else ""
            compare_html = f'''<div class="market-compare-grid"><div class="market-compare-col"><span style="color:#94A3B8;">📊 ฝั่งโมเดล:</span><br><strong style="color:#F1F5F9;">{clean_rec}</strong> <span style="color:#38BDF8;">({prob_pct})</span></div><div class="market-compare-col"><span style="color:#94A3B8;">🌊 ฝั่งกระแสเงิน:</span><br><strong style="color:#F1F5F9;">ดัน{m_dir}</strong> <span style="color:#FBBF24;">({m_prob:.1%}{diff_str})</span></div></div>'''

        verdict_html = ""
        if is_clash_sp or lm_sp.get('verdict') in ['CAPITAL_PRESERVATION', 'PASS']:
            detail_sub = f' <span class="verdict-sep">—</span> <span class="verdict-detail">{v_detail or "สัญญาณขัดแย้งกับตลาด แนะนำรักษาเงินทุน"}</span>'
            verdict_html = f'''<div class="verdict-banner pass"><span class="verdict-badge">🎯 ฟันธง:</span> <strong class="verdict-title">ข้าม (PASS) / งดวางเงิน</strong>{detail_sub}</div>'''
        elif lm_sp.get('verdict') == 'CONFIRMED_ALIGNED' or v_title:
            detail_sub = f' <span class="verdict-sep">—</span> <span class="verdict-detail">{v_detail}</span>' if v_detail else ""
            verdict_html = f'''<div class="verdict-banner confirm"><span class="verdict-badge">🎯 ฟันธง:</span> <strong class="verdict-title">{v_title}</strong>{detail_sub}</div>'''

        callout_type = "warning" if (is_clash_sp or lm_sp.get('status') == 'STEAM_WARNING') else ("success" if lm_sp.get('status') == 'SHARP_CONFIRMATION' else "info")
        market_html = f'''<div class="market-callout {callout_type}"><div><strong>{badge_txt}:</strong> {detail_txt}</div>{compare_html}{verdict_html}</div>'''

    # 6. Model Deep-Dive Metrics
    proj_margin = sp_res.get('projected_margin')
    raw_margin = sp_res.get('raw_projected_margin')
    ai_res = sp_res.get('ai_residual')
    anchor_sp = sp_res.get('anchor_spread', sp_val) or 0.0

    metrics_html = ""
    if proj_margin is not None:
        ai_val_str = f"{ai_res:+.1f} PTS" if (ai_res is not None and ai_res != 0.0) else "0.0 PTS"
        raw_val_str = f"{raw_margin:+.1f} PTS" if raw_margin is not None else "-"
        metrics_html = f'''<div class="model-metrics-grid"><div class="model-metric-item"><span class="model-metric-label">Market-Anchored</span><span class="model-metric-val pos">{proj_margin:+.1f} PTS</span><span class="model-metric-sub">เปิด {anchor_sp:+.1f}×65% + ดิบ×35%</span></div><div class="model-metric-item"><span class="model-metric-label">AI Tuned (CatBoost)</span><span class="model-metric-val">{ai_val_str}</span><span class="model-metric-sub">ปรับความล้า / B2B / ฟอร์ม</span></div><div class="model-metric-item"><span class="model-metric-label">Raw Model</span><span class="model-metric-val">{raw_val_str}</span><span class="model-metric-sub">โมเดลสถิติดั้งเดิม</span></div></div>'''

    return f'''<div class="analysis-card"><div class="analysis-header"><span class="analysis-title">⚖️ วิเคราะห์ แต้มต่อ (Spread)</span><span class="market-line-pill">เรตตลาด: {sp_line_pill}</span></div>{live_banner_html}{roles_html}<div class="{box_class}"><div class="hero-pick-row"><span class="hero-pick-text">🎯 {clean_rec}</span><div class="hero-pick-badges">{conf_badge}{edge_chip}</div></div><div class="win-prob-wrapper"><div class="win-prob-header"><span>โอกาสชนะทางสถิติ (Win Prob)</span><strong>{prob_pct}</strong></div><div class="custom-prob-bar"><div class="{fill_class}" style="width: {min(100.0, max(0.0, win_p * 100)):.1f}%;"></div></div></div></div>{market_html}{metrics_html}</div>'''


def _render_ou_analysis_card(ou_val, open_total, ou_res, rec_text, t_edge, win_p, is_clash_ou, is_live=False, live_proj=None, lm_data=None):
    lm_data = lm_data or {}
    
    # 1. Consensus line pill
    if ou_val is not None:
        ou_line_pill = f"O/U {ou_val:.1f}"
    else:
        ou_line_pill = "ยังไม่ออกราคา"

    # 2. Market Line Movement Bar (Open vs Current - 2-row clean structured matchup)
    roles_html = ""
    if open_total is not None and ou_val is not None:
        move_diff = ou_val - open_total
        diff_sign = "+" if move_diff > 0 else ""
        if move_diff > 0:
            trend_class = "up"
            move_label = "▲ ดันสูงขึ้น"
            trend_text = f"{move_label} ({diff_sign}{move_diff:.1f})"
        elif move_diff < 0:
            trend_class = "down"
            move_label = "▼ ไหลต่ำลง"
            trend_text = f"{move_label} ({diff_sign}{move_diff:.1f})"
        else:
            trend_class = "flat"
            move_label = "คงที่"
            trend_text = f"{move_label} (0.0)"

        roles_html = f'''<div class="matchup-roles-card"><div class="matchup-role-row"><div class="matchup-team-side"><span class="role-pill neutral">📌 ราคาเปิด</span><span class="matchup-sub-desc">เรตเปิดแรกของตลาด</span></div><span class="matchup-point-val open">{open_total:.1f}</span></div><div class="matchup-role-row"><div class="matchup-team-side"><span class="role-pill neutral">🎯 ปัจจุบัน</span><span class="trend-chip {trend_class}">{trend_text}</span></div><span class="matchup-point-val current">{ou_val:.1f}</span></div></div>'''
    elif ou_val is not None:
        roles_html = f'''<div class="matchup-roles-card"><div class="matchup-role-row"><div class="matchup-team-side"><span class="role-pill neutral">🎯 ปัจจุบัน</span><span class="matchup-sub-desc">Consensus Line</span></div><span class="matchup-point-val current">{ou_val:.1f}</span></div></div>'''

    # 3. Live Pace Banner if in-play
    live_banner_html = ""
    if is_live and live_proj:
        live_banner_html = f'''<div class="live-pace-banner"><span>⚡ <strong>Live Projected Total:</strong> {live_proj.get("proj_total", 0):.1f} PTS</span><span style="color:#94A3B8;">(Pace: {live_proj.get("scoring_rate", 0):.2f} pts/m)</span></div>'''

    # 4. Hero Pick Box
    is_strong = 'STRONG' in ou_res.get('recommendation', '') or 'มั่นใจมาก' in rec_text
    is_lean = 'LEAN' in ou_res.get('recommendation', '') or 'น่าสนใจ' in rec_text
    
    if is_clash_ou:
        conf_badge = '<span class="confidence-pill caution">⚠️ สัญญาณขัดแย้ง</span>'
        box_class = "hero-pick-box clash"
        fill_class = "custom-prob-fill clash"
    elif is_strong:
        conf_badge = '<span class="confidence-pill high">🔥 มั่นใจมาก</span>'
        box_class = "hero-pick-box"
        fill_class = "custom-prob-fill"
    elif is_lean:
        conf_badge = '<span class="confidence-pill medium">⚡ น่าสนใจ</span>'
        box_class = "hero-pick-box"
        fill_class = "custom-prob-fill"
    else:
        conf_badge = '<span class="confidence-pill neutral">⚖️ รอดู / สูสี</span>'
        box_class = "hero-pick-box"
        fill_class = "custom-prob-fill"

    prob_pct = f"{win_p * 100:.1f}%"
    edge_chip = f'<span class="edge-pill">Edge: {t_edge:+.1f} แต้ม</span>' if t_edge else ''
    clean_rec = rec_text.replace('(มั่นใจมาก)', '').replace('(น่าสนใจ)', '').replace('🔥', '').replace('📈', '').replace('📉', '').replace('🎯', '').strip()

    # 5. Market Flow & Line Movement Callout
    lm_ou = ou_res.get('line_movement') or {}
    market_html = ""
    if lm_ou and lm_ou.get('badge'):
        badge_txt = lm_ou.get('badge', '')
        detail_txt = lm_ou.get('detail', '')
        m_prob = lm_ou.get('market_flow_prob')
        m_dir = lm_ou.get('market_flow_direction', '')
        diff_val = lm_ou.get('diff', 0.0)
        v_title = lm_ou.get('verdict_title', '')
        v_detail = lm_ou.get('verdict_detail', '')

        compare_html = ""
        if m_prob is not None and lm_ou.get('status') != 'NO_DATA':
            diff_str = f" ({diff_val:+.1f})" if abs(diff_val) >= 0.5 else ""
            compare_html = f'''<div class="market-compare-grid"><div class="market-compare-col"><span style="color:#94A3B8;">📊 ฝั่งโมเดล:</span><br><strong style="color:#F1F5F9;">{clean_rec}</strong> <span style="color:#38BDF8;">({prob_pct})</span></div><div class="market-compare-col"><span style="color:#94A3B8;">🌊 ฝั่งกระแสเงิน:</span><br><strong style="color:#F1F5F9;">ดัน{m_dir}</strong> <span style="color:#FBBF24;">({m_prob:.1%}{diff_str})</span></div></div>'''

        verdict_html = ""
        if is_clash_ou or lm_ou.get('verdict') in ['CAPITAL_PRESERVATION', 'PASS']:
            detail_sub = f' <span class="verdict-sep">—</span> <span class="verdict-detail">{v_detail or "สัญญาณขัดแย้งกับตลาด แนะนำรักษาเงินทุน"}</span>'
            verdict_html = f'''<div class="verdict-banner pass"><span class="verdict-badge">🎯 ฟันธง:</span> <strong class="verdict-title">ข้าม (PASS) / งดวางเงิน</strong>{detail_sub}</div>'''
        elif lm_ou.get('verdict') == 'CONFIRMED_ALIGNED' or v_title:
            detail_sub = f' <span class="verdict-sep">—</span> <span class="verdict-detail">{v_detail}</span>' if v_detail else ""
            verdict_html = f'''<div class="verdict-banner confirm"><span class="verdict-badge">🎯 ฟันธง:</span> <strong class="verdict-title">{v_title}</strong>{detail_sub}</div>'''

        callout_type = "warning" if (is_clash_ou or lm_ou.get('status') == 'STEAM_WARNING') else ("success" if lm_ou.get('status') == 'SHARP_CONFIRMATION' else "info")
        market_html = f'''<div class="market-callout {callout_type}"><div><strong>{badge_txt}:</strong> {detail_txt}</div>{compare_html}{verdict_html}</div>'''

    # 6. Model Deep-Dive Metrics
    proj_total = ou_res.get('projected_total')
    raw_total = ou_res.get('raw_projected_total')
    ai_res = ou_res.get('ai_residual')
    anchor_used = ou_res.get('anchor_line') or ou_val or 0.0

    metrics_html = ""
    if proj_total is not None:
        ai_val_str = f"{ai_res:+.1f} PTS" if (ai_res is not None and ai_res != 0.0) else "0.0 PTS"
        raw_val_str = f"{raw_total:.1f} PTS" if raw_total is not None else "-"
        metrics_html = f'''<div class="model-metrics-grid"><div class="model-metric-item"><span class="model-metric-label">Market-Anchored</span><span class="model-metric-val pos">{proj_total:.1f} PTS</span><span class="model-metric-sub">เปิด {anchor_used:.1f}×65% + ดิบ×35%</span></div><div class="model-metric-item"><span class="model-metric-label">AI Tuned (CatBoost)</span><span class="model-metric-val">{ai_val_str}</span><span class="model-metric-sub">ปรับความล้า / B2B / สปีด</span></div><div class="model-metric-item"><span class="model-metric-label">Raw Model</span><span class="model-metric-val">{raw_val_str}</span><span class="model-metric-sub">โมเดลสถิติดั้งเดิม</span></div></div>'''

    return f'''<div class="analysis-card"><div class="analysis-header"><span class="analysis-title">📊 วิเคราะห์ สูง/ต่ำ (Over/Under)</span><span class="market-line-pill">เรตตลาด: {ou_line_pill}</span></div>{live_banner_html}{roles_html}<div class="{box_class}"><div class="hero-pick-row"><span class="hero-pick-text">🎯 {clean_rec}</span><div class="hero-pick-badges">{conf_badge}{edge_chip}</div></div><div class="win-prob-wrapper"><div class="win-prob-header"><span>โอกาสชนะทางสถิติ (Win Prob)</span><strong>{prob_pct}</strong></div><div class="custom-prob-bar"><div class="{fill_class}" style="width: {min(100.0, max(0.0, win_p * 100)):.1f}%;"></div></div></div></div>{market_html}{metrics_html}</div>'''



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
        
        progress.progress(75, text="4/5 กำลังอัปเดตราคาต่อรองและรายชื่อ 5 ตัวจริง (RotoWire)...")
        try:
            from collectors.rotowire_collector import RotoWireCollector
            RotoWireCollector().get_todays_lineups(force_refresh=True)
        except Exception:
            pass
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
def get_analyzer(cache_version: str = "v11_rotowire_lineups"):
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
    now_badge = f'<div class="matchday-sync">⚡ Live Sync: {datetime.now(THAI_TZ).strftime("%H:%M:%S")} น.</div>' if is_live_refresh else ''
    st.markdown(f"""
    <div class="matchday-bar">
        <div class="matchday-title">📅 โปรแกรมการแข่งขันวันที่ {selected_date.strftime('%d/%m/%Y')} (เวลาไทย)</div>
        {now_badge}
    </div>
    """, unsafe_allow_html=True)


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
            
                sp_feat = (analysis.get('spread') or {}).get('features') or {}
                ou_feat = (analysis.get('over_under') or {}).get('features') or {}
                m_feat = sp_feat or ou_feat or analysis.get('features') or {}
                h_st = m_feat.get('home_lineup_status')
                a_st = m_feat.get('away_lineup_status')
                lineup_tag = " [Confirmed ✅]" if (h_st == 'Confirmed' and a_st == 'Confirmed') else ""

                thai_time_str = game.get('thai_time')
                status_pill = "🏁 จบเกม" if 'FINAL' in st_raw.upper() else ("🔴 LIVE" if is_live else f"⏰ {thai_time_str} น." if thai_time_str else "TBD")
                lineup_pill = " • ✅ Lineup" if (h_st == 'Confirmed' and a_st == 'Confirmed') else ""
            
                with st.expander(f"🏀 {away} @ {home}  •  {status_pill}{lineup_pill}", expanded=True):
                    # --- MATCH INFO (Kickoff time & Venue) ---
                    venue_str = game.get('venue') or 'TBD'
                    full_time_display = game.get('thai_time_display') or ''
                    time_html = f'<div class="meta-item">🕒 <span class="time-highlight">{full_time_display}</span></div>' if full_time_display else ''
                    venue_html = f'<div class="meta-item">📍 {venue_str}</div>' if venue_str else ''
                    render_html(f'<div class="match-meta-strip">{time_html}{venue_html}</div>')

                    # --- SPORTS SCOREBOARD BANNER (frontend-design) ---
                    h_score = game.get('home_score')
                    a_score = game.get('away_score')
                    if pd.notna(h_score) and pd.notna(a_score) and str(h_score).strip() != '' and str(a_score).strip() != '':
                        try:
                            h_pts = int(float(h_score))
                            a_pts = int(float(a_score))
                            total_pts = h_pts + a_pts
                            h_winner = " winner" if h_pts > a_pts else ""
                            a_winner = " winner" if a_pts > h_pts else ""
                        
                            if is_live:
                                c_detail = str(game.get('clock_detail') or (f"Q{game.get('period', '')} {game.get('clock', '')}".strip())).strip()
                                status_html = f'<span class="scoreboard-badge live">🔴 LIVE {c_detail}</span>'
                                lead = h_pts - a_pts
                                lead_str = f"{home} +{lead}" if lead > 0 else (f"{away} +{-lead}" if lead < 0 else "เสมอกัน")
                                sub_info = f'สถานะ: <strong>{lead_str}</strong>'
                            else:
                                status_html = '<span class="scoreboard-badge final">🏁 FINAL</span>'
                                sub_info = f'ผู้ชนะ: <strong>{home if h_pts > a_pts else away}</strong>'

                            render_html(f"""
                            <div class="scoreboard-banner">
                                <div class="scoreboard-status-bar">
                                    {status_html}
                                    <span style="font-size: 0.78rem; color: #94A3B8;">NBA Game</span>
                                </div>
                                <div class="scoreboard-teams">
                                    <div class="score-team-box">
                                        <span class="score-team-name">{away}</span>
                                        <span class="score-team-pts{a_winner}">{a_pts}</span>
                                    </div>
                                    <div class="score-vs-divider">-</div>
                                    <div class="score-team-box">
                                        <span class="score-team-name">{home}</span>
                                        <span class="score-team-pts{h_winner}">{h_pts}</span>
                                    </div>
                                </div>
                                <div class="scoreboard-footer">
                                    <div>{sub_info}</div>
                                    <div>แต้มรวมจริง: <strong>{total_pts} PTS</strong></div>
                                </div>
                            </div>
                            """)
                        except Exception:
                            pass
                
                    # --- TEAM PROFILES BAR (Head-to-Head Comparison) ---
                    if 'home_profile' in analysis and 'away_profile' in analysis:
                        hp = analysis['home_profile']
                        ap = analysis['away_profile']
                        h_rec = f"{hp['wins']}-{hp['losses']}"
                        a_rec = f"{ap['wins']}-{ap['losses']}"
                        render_html(f"""
                        <div class="team-form-grid">
                            <div class="team-form-card away">
                                <div class="team-form-header">
                                    <span class="team-role-tag">เยือน</span>
                                    <span class="team-name-label">{away}</span>
                                </div>
                                <div class="team-stat-badges">
                                    <span class="stat-chip record">สถิติ {a_rec}</span>
                                    <span class="stat-chip scoring">{ap['avg_points_scored']:.1f} PTS/G</span>
                                </div>
                            </div>
                            <div class="team-form-vs-badge">VS</div>
                            <div class="team-form-card home">
                                <div class="team-form-header">
                                    <span class="team-name-label">{home}</span>
                                    <span class="team-role-tag home">เหย้า</span>
                                </div>
                                <div class="team-stat-badges">
                                    <span class="stat-chip record">สถิติ {h_rec}</span>
                                    <span class="stat-chip scoring">{hp['avg_points_scored']:.1f} PTS/G</span>
                                </div>
                            </div>
                        </div>
                        """)

                    # --- INJURY REPORT CARD ---
                    spread_dict = analysis.get('spread') or {}
                    ou_dict = analysis.get('over_under') or {}
                    feat = spread_dict.get('features') or ou_dict.get('features') or analysis.get('features') or {}
                    h_inj = feat.get('home_injuries', [])
                    a_inj = feat.get('away_injuries', [])
                    if h_inj or a_inj:
                        def _render_chips(inj_list):
                            return "".join(f'<span class="inj-player-chip">🏥 {p}</span>' for p in inj_list)
                        away_chips = f"<div><strong style='color:#E2E8F0;'>{away}:</strong> {_render_chips(a_inj)}</div>" if a_inj else ""
                        home_chips = f"<div style='margin-top: 4px;'><strong style='color:#E2E8F0;'>{home}:</strong> {_render_chips(h_inj)}</div>" if h_inj else ""
                        render_html(f"""
                        <div class="injury-card">
                            <div class="injury-card-title">🏥 ผู้เล่นบาดเจ็บ / พักแข่ง (Injury Report)</div>
                            {away_chips}
                            {home_chips}
                        </div>
                        """)

                    # Lineup status & 5 starters
                    h_lineup_st = feat.get('home_lineup_status', 'Unconfirmed')
                    a_lineup_st = feat.get('away_lineup_status', 'Unconfirmed')

                    h_starters = feat.get('home_starters', [])
                    a_starters = feat.get('away_starters', [])
                    if h_starters or a_starters:
                        pos_map = get_espn_position_map()
                        sorted_a = format_and_sort_starters(a_starters, pos_map) if a_starters else []
                        sorted_h = format_and_sort_starters(h_starters, pos_map) if h_starters else []

                        def _render_starter_card(team_label, s_list, l_status):
                            if not s_list: return ""
                            pill_color = "rgba(16, 185, 129, 0.2)" if l_status == 'Confirmed' else "rgba(245, 158, 11, 0.2)"
                            pill_text_color = "#34D399" if l_status == 'Confirmed' else "#FBBF24"
                            pill_html = f'<span style="font-size:0.7rem; padding:1px 6px; border-radius:4px; background:{pill_color}; color:{pill_text_color}; font-weight:700;">{l_status}</span>'
                            
                            rows_html = "".join(
                                f'<div class="starter-row"><span class="pos-badge pos-{s["position"].lower()}">{s["position"]}</span><span class="starter-name">{s["name"]}</span></div>'
                                for s in s_list
                            )
                            return f'<div class="starting5-card"><div class="starting5-header"><span>🏀 5 ตัวจริง {team_label}</span>{pill_html}</div><div class="starting5-list">{rows_html}</div></div>'

                        sc_col1, sc_col2 = st.columns(2)
                        with sc_col1:
                            if sorted_a:
                                render_html(_render_starter_card(away, sorted_a, a_lineup_st))
                        with sc_col2:
                            if sorted_h:
                                render_html(_render_starter_card(home, sorted_h, h_lineup_st))
                
                    st.divider()

                    live_proj = calculate_live_pace_projection(
                        game.get('period'), game.get('clock'), h_score, a_score,
                        pregame_total=ou_val, pregame_spread=sp_val
                    ) if is_live else None
                
                    col1, col2 = st.columns(2)
                
                    # O/U Column
                    with col1:
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

                            # Render the modern, structured Over/Under Analysis Card
                            render_html(_render_ou_analysis_card(
                                ou_val=ou_val,
                                open_total=open_total,
                                ou_res=ou_res,
                                rec_text=rec_text,
                                t_edge=t_edge,
                                win_p=win_p,
                                is_clash_ou=is_clash_ou,
                                is_live=is_live,
                                live_proj=live_proj,
                                lm_data=lm_data
                            ))

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

                            # Render the modern, structured Spread Analysis Card
                            render_html(_render_spread_analysis_card(
                                home=home,
                                away=away,
                                sp_val=sp_val,
                                open_spread=open_spread,
                                sp_res=sp_res,
                                rec_text=rec_text,
                                c_edge=c_edge,
                                win_p=win_p,
                                is_clash_sp=is_clash_sp,
                                is_live=is_live,
                                live_proj=live_proj,
                                lm_data=lm_data
                            ))

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
                                    icon = t_dir(f['direction'], 'spread')
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

    # Detailed Category Breakdown in expander (2 rows: Strong vs Lean + Verdict summary)
    with st.expander("📊 ดูรายละเอียดแยกตามหมวดหมู่ (Breakdown by Category)", expanded=True):
        st.markdown("#### 🔥 ไม้เน้น (Strong Picks - โมเดลมั่นใจสูง)")
        b_r1_c1, b_r1_c2 = st.columns(2)
        with b_r1_c1:
            st.markdown("##### 🛡️ มั่นใจแต้มต่อ (Strong Spread)")
            s_sp = stats.get('strong_spread', {})
            st.write(f"- ชนะ/แพ้: **{s_sp.get('wins', 0)}W - {s_sp.get('losses', 0)}L - {s_sp.get('pushes', 0)}P**")
            st.write(f"- Win Rate: **{s_sp.get('win_rate', 0.0):.1f}%**")
            st.write(f"- กำไร/ขาดทุน: **{s_sp.get('net_units', 0.0):+.2f} Units** (`{s_sp.get('roi_pct', 0.0):+.1f}% ROI`)")
        with b_r1_c2:
            st.markdown("##### 🏀 มั่นใจสูง/ต่ำ (Strong O/U)")
            s_ou = stats.get('strong_ou', {})
            st.write(f"- ชนะ/แพ้: **{s_ou.get('wins', 0)}W - {s_ou.get('losses', 0)}L - {s_ou.get('pushes', 0)}P**")
            st.write(f"- Win Rate: **{s_ou.get('win_rate', 0.0):.1f}%**")
            st.write(f"- กำไร/ขาดทุน: **{s_ou.get('net_units', 0.0):+.2f} Units** (`{s_ou.get('roi_pct', 0.0):+.1f}% ROI`)")

        st.markdown("---")
        st.markdown("#### 📈 ไม้น่าสนใจ (Lean Picks - สัญญาณน่าจับตา)")
        b_r2_c1, b_r2_c2 = st.columns(2)
        with b_r2_c1:
            st.markdown("##### ✈️ น่าสนใจแต้มต่อ (Lean Spread)")
            l_sp = stats.get('lean_spread', {})
            st.write(f"- ชนะ/แพ้: **{l_sp.get('wins', 0)}W - {l_sp.get('losses', 0)}L - {l_sp.get('pushes', 0)}P**")
            st.write(f"- Win Rate: **{l_sp.get('win_rate', 0.0):.1f}%**")
            st.write(f"- กำไร/ขาดทุน: **{l_sp.get('net_units', 0.0):+.2f} Units** (`{l_sp.get('roi_pct', 0.0):+.1f}% ROI`)")
        with b_r2_c2:
            st.markdown("##### 🏀 น่าสนใจสูง/ต่ำ (Lean O/U)")
            l_ou = stats.get('lean_ou', {})
            st.write(f"- ชนะ/แพ้: **{l_ou.get('wins', 0)}W - {l_ou.get('losses', 0)}L - {l_ou.get('pushes', 0)}P**")
            st.write(f"- Win Rate: **{l_ou.get('win_rate', 0.0):.1f}%**")
            st.write(f"- กำไร/ขาดทุน: **{l_ou.get('net_units', 0.0):+.2f} Units** (`{l_ou.get('roi_pct', 0.0):+.1f}% ROI`)")

        st.markdown("---")
        vd = stats.get('verdict', {})
        st.caption(
            f"🌊 **สรุปฟันธงกระแสเงิน (Verdict Summary):** ชนะ/แพ้ **{vd.get('wins', 0)}W - {vd.get('losses', 0)}L - {vd.get('pushes', 0)}P** "
            f"| Win Rate **{vd.get('win_rate', 0.0):.1f}%** | กำไร/ขาดทุน **{vd.get('net_units', 0.0):+.2f} Units** (`{vd.get('roi_pct', 0.0):+.1f}% ROI`)"
        )

    st.markdown("### 📋 ตารางประวัติคำทำนายย้อนหลัง")

    # Available dates for date filtering
    avail_dates = get_available_ledger_dates(ledger_df)

    # 3 Filter controls
    f_c1, f_c2, f_c3 = st.columns([1.5, 1.2, 2.0])
    with f_c1:
        date_options = ["ทั้งหมด (All Dates)"] + avail_dates
        filter_date = st.selectbox("📅 เลือกวันที่", date_options)
    with f_c2:
        filter_status = st.selectbox("สถานะการแข่งขัน", ["ทั้งหมด", "ตรวจผลแล้ว (GRADED)", "รอผล (PENDING)"])
    with f_c3:
        filter_tier = st.selectbox(
            "ระดับความมั่นใจและประเภทไม้",
            [
                "ทั้งหมด",
                "🔥 มั่นใจแต้มต่อ (Strong Spread)",
                "🔥 มั่นใจสูงต่ำ (Strong O/U)",
                "✈️ น่าสนใจแต้มต่อ (Lean Spread)",
                "📈 น่าสนใจสูงต่ำ (Lean O/U)",
                "เฉพาะไม้เน้นทั้งหมด (Strong All)"
            ]
        )

    # Daily performance banner when a specific date is selected
    if filter_date != "ทั้งหมด (All Dates)":
        daily_ledger = ledger_df[ledger_df['date'] == filter_date]
        d_stats = calculate_track_record(daily_ledger)
        d_total_games = len(daily_ledger)
        d_graded_games = d_stats.get('graded_games', 0)
        d_wins = d_stats['overall']['wins']
        d_losses = d_stats['overall']['losses']
        d_pushes = d_stats['overall']['pushes']
        d_wr = d_stats['overall']['win_rate']
        d_units = d_stats['overall']['net_units']
        d_roi = d_stats['overall']['roi_pct']

        st.info(
            f"📅 **สรุปผลงานวันที่ {filter_date}:** แข่งขันทั้งหมด **{d_total_games} คู่** (จบแล้ว {d_graded_games} คู่) | "
            f"ผลรวมคำทำนาย: ชนะ **{d_wins}W - {d_losses}L - {d_pushes}P** "
            f"(`{d_wr:.1f}%` Win Rate) | "
            f"ผลตอบแทนจำลอง **{d_units:+.2f} Units** (`{d_roi:+.1f}%` ROI)"
        )

    filtered_df = ledger_df.copy()
    if filter_date != "ทั้งหมด (All Dates)":
        filtered_df = filtered_df[filtered_df['date'] == filter_date]

    if filter_status == "ตรวจผลแล้ว (GRADED)":
        filtered_df = filtered_df[filtered_df['status'] == 'GRADED']
    elif filter_status == "รอผล (PENDING)":
        filtered_df = filtered_df[filtered_df['status'] == 'PENDING']

    if filter_tier == "🔥 มั่นใจแต้มต่อ (Strong Spread)":
        filtered_df = filtered_df[filtered_df['sp_recommendation'].astype(str).str.contains('STRONG', na=False)]
    elif filter_tier == "🔥 มั่นใจสูงต่ำ (Strong O/U)":
        filtered_df = filtered_df[filtered_df['ou_recommendation'].astype(str).str.contains('STRONG', na=False)]
    elif filter_tier == "✈️ น่าสนใจแต้มต่อ (Lean Spread)":
        filtered_df = filtered_df[filtered_df['sp_recommendation'].astype(str).str.contains('LEAN', na=False)]
    elif filter_tier == "📈 น่าสนใจสูงต่ำ (Lean O/U)":
        filtered_df = filtered_df[filtered_df['ou_recommendation'].astype(str).str.contains('LEAN', na=False)]
    elif filter_tier == "เฉพาะไม้เน้นทั้งหมด (Strong All)":
        filtered_df = filtered_df[
            filtered_df['ou_recommendation'].astype(str).str.contains('STRONG', na=False) |
            filtered_df['sp_recommendation'].astype(str).str.contains('STRONG', na=False)
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


tab_live, tab_track = st.tabs(["🏀 วิเคราะห์สด (Live)", "🏆 สถิติย้อนหลัง (Track)"])

with tab_live:
    render_game_slate(selected_date, auto_refresh and is_today)

with tab_track:
    render_track_record_tab()
