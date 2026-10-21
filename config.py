"""Configuration module for NBA data analysis system.

Contains league constants, endpoint URLs, team metadata, and storage path settings.
"""

import logging
import os
import re
from typing import Any, Dict, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

# Setup module logger
logger = logging.getLogger(__name__)

# Season and league settings
NBA_SEASON: int = 2026
NBA_API_SEASON: str = "2025-26"  # nba_api uses this format
LEAGUE_ID: str = "00"  # NBA league ID for nba_api

# ESPN API base URLs
ESPN_BASE: str = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
ESPN_STANDINGS: str = "https://site.api.espn.com/apis/v2/sports/basketball/nba/standings"

# NBA CDN URLs
CDN_SCHEDULE: str = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"
CDN_SCOREBOARD: str = "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"
CDN_BOXSCORE_TEMPLATE: str = "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{game_id}.json"



# API Keys
ODDS_API_KEY: str = os.getenv("ODDS_API_KEY", "")
if not ODDS_API_KEY:
    try:
        import streamlit as st
        ODDS_API_KEY = st.secrets.get("ODDS_API_KEY", "")
    except Exception:
        pass

# stats.nba.com Headers (required for nba_api / direct requests)
STATS_HEADERS: Dict[str, str] = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Referer": "https://stats.nba.com/",
    "Origin": "https://stats.nba.com",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Host": "stats.nba.com",
}

# TEAMS dictionary mapping team abbreviations to full info
TEAMS: Dict[str, Dict[str, Any]] = {
    "ATL": {"name": "Atlanta Hawks", "espn_id": 1, "conference": "Eastern"},
    "BOS": {"name": "Boston Celtics", "espn_id": 2, "conference": "Eastern"},
    "NOP": {"name": "New Orleans Pelicans", "espn_id": 3, "conference": "Western"},
    "CHI": {"name": "Chicago Bulls", "espn_id": 4, "conference": "Eastern"},
    "CLE": {"name": "Cleveland Cavaliers", "espn_id": 5, "conference": "Eastern"},
    "DAL": {"name": "Dallas Mavericks", "espn_id": 6, "conference": "Western"},
    "DEN": {"name": "Denver Nuggets", "espn_id": 7, "conference": "Western"},
    "DET": {"name": "Detroit Pistons", "espn_id": 8, "conference": "Eastern"},
    "GSW": {"name": "Golden State Warriors", "espn_id": 9, "conference": "Western"},
    "HOU": {"name": "Houston Rockets", "espn_id": 10, "conference": "Western"},
    "IND": {"name": "Indiana Pacers", "espn_id": 11, "conference": "Eastern"},
    "LAC": {"name": "Los Angeles Clippers", "espn_id": 12, "conference": "Western"},
    "LAL": {"name": "Los Angeles Lakers", "espn_id": 13, "conference": "Western"},
    "MIA": {"name": "Miami Heat", "espn_id": 14, "conference": "Eastern"},
    "MIL": {"name": "Milwaukee Bucks", "espn_id": 15, "conference": "Eastern"},
    "MIN": {"name": "Minnesota Timberwolves", "espn_id": 16, "conference": "Western"},
    "BKN": {"name": "Brooklyn Nets", "espn_id": 17, "conference": "Eastern"},
    "NYK": {"name": "New York Knicks", "espn_id": 18, "conference": "Eastern"},
    "ORL": {"name": "Orlando Magic", "espn_id": 19, "conference": "Eastern"},
    "PHI": {"name": "Philadelphia 76ers", "espn_id": 20, "conference": "Eastern"},
    "PHX": {"name": "Phoenix Suns", "espn_id": 21, "conference": "Western"},
    "POR": {"name": "Portland Trail Blazers", "espn_id": 22, "conference": "Western"},
    "SAC": {"name": "Sacramento Kings", "espn_id": 23, "conference": "Western"},
    "SAS": {"name": "San Antonio Spurs", "espn_id": 24, "conference": "Western"},
    "OKC": {"name": "Oklahoma City Thunder", "espn_id": 25, "conference": "Western"},
    "UTA": {"name": "Utah Jazz", "espn_id": 26, "conference": "Western"},
    "WAS": {"name": "Washington Wizards", "espn_id": 27, "conference": "Eastern"},
    "TOR": {"name": "Toronto Raptors", "espn_id": 28, "conference": "Eastern"},
    "MEM": {"name": "Memphis Grizzlies", "espn_id": 29, "conference": "Western"},
    "CHA": {"name": "Charlotte Hornets", "espn_id": 30, "conference": "Eastern"},
}

# Filesystem and data storage directories
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
DATA_DIR: str = os.path.join(BASE_DIR, "data")
RAW_DIR: str = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR: str = os.path.join(DATA_DIR, "processed")


def ensure_dirs() -> None:
    """Ensure that project data directories exist.

    Creates DATA_DIR, RAW_DIR, and PROCESSED_DIR if they do not already exist.

    Raises:
        OSError: If any of the directories cannot be created due to permissions or OS errors.
    """
    for directory in [DATA_DIR, RAW_DIR, PROCESSED_DIR]:
        try:
            os.makedirs(directory, exist_ok=True)
            logger.debug("Ensured directory exists: %s", directory)
        except OSError as exc:
            logger.error("Failed to create directory '%s': %s", directory, exc)
            raise


# Extra alias mapping for alternative abbreviations and common team aliases
_TEAM_ALIASES: Dict[str, str] = {
    "la clippers": "LAC",
    "clippers": "LAC",
    "l.a. clippers": "LAC",
    "la lakers": "LAL",
    "lakers": "LAL",
    "l.a. lakers": "LAL",
    "sixers": "PHI",
    "76ers": "PHI",
    "blazers": "POR",
    "trail blazers": "POR",
    "cavs": "CLE",
    "wolves": "MIN",
    "t-wolves": "MIN",
    "timberwolves": "MIN",
    "gs": "GSW",
    "gsw": "GSW",
    "warriors": "GSW",
    "golden state": "GSW",
    "ny": "NYK",
    "nyk": "NYK",
    "knicks": "NYK",
    "new york": "NYK",
    "no": "NOP",
    "nop": "NOP",
    "pelicans": "NOP",
    "new orleans": "NOP",
    "sa": "SAS",
    "sas": "SAS",
    "spurs": "SAS",
    "san antonio": "SAS",
    "wsh": "WAS",
    "was": "WAS",
    "wizards": "WAS",
    "washington": "WAS",
    "utah": "UTA",
    "uta": "UTA",
    "jazz": "UTA",
    "pho": "PHX",
    "phx": "PHX",
    "suns": "PHX",
    "phoenix": "PHX",
    "bk": "BKN",
    "bkn": "BKN",
    "nets": "BKN",
    "brooklyn": "BKN",
    "okc": "OKC",
    "thunder": "OKC",
    "oklahoma city": "OKC",
}


def resolve_team_info(identifier: Any) -> Optional[Dict[str, Any]]:
    """Resolve a team abbreviation, full name, alias, or ESPN ID to its TEAMS dictionary entry.

    Returns the team dict augmented with 'abbreviation' or None if unresolved.
    """
    if identifier is None:
        return None
    raw = str(identifier).strip()
    if not raw or raw.lower() == 'nan':
        return None

    # Check direct abbreviation
    upper_raw = raw.upper()
    if upper_raw in TEAMS:
        res = dict(TEAMS[upper_raw])
        res["abbreviation"] = upper_raw
        return res

    # Check numeric ESPN ID
    try:
        espn_num = int(raw)
        for abbr, data in TEAMS.items():
            if data.get("espn_id") == espn_num:
                res = dict(data)
                res["abbreviation"] = abbr
                return res
    except (ValueError, TypeError):
        pass

    lower_raw = raw.lower()

    # Check alias dictionary
    if lower_raw in _TEAM_ALIASES:
        abbr = _TEAM_ALIASES[lower_raw]
        res = dict(TEAMS[abbr])
        res["abbreviation"] = abbr
        return res

    # Check match against TEAMS full names and city/nicknames
    for abbr, data in TEAMS.items():
        name = data["name"]
        lower_name = name.lower()
        if lower_raw == lower_name or lower_raw == abbr.lower():
            res = dict(data)
            res["abbreviation"] = abbr
            return res
        nickname = lower_name.split()[-1]
        if lower_raw == nickname:
            res = dict(data)
            res["abbreviation"] = abbr
            return res
        if lower_name in lower_raw:
            res = dict(data)
            res["abbreviation"] = abbr
            return res
        # Use word boundaries so abbreviations like "la", "in", "or" don't match inside other city names
        if re.search(r'\b' + re.escape(lower_raw) + r'\b', lower_name):
            res = dict(data)
            res["abbreviation"] = abbr
            return res

    return None


def normalize_team_name(name: Any) -> str:
    """Standardize any team representation to the official full team name in TEAMS."""
    if name is None:
        return name
    info = resolve_team_info(name)
    if info:
        return info["name"]
    return str(name).strip()


def parse_spread(spread_text: Any) -> Tuple[Optional[str], Optional[float]]:
    """Parse spread text like 'BOS -4.5', 'BOS PK', 'PK', 'EVEN', '-3.5' into (team_abbr, spread_value).
    Returns (None, None) if parsing fails or spread_text is empty/NaN.
    """
    if spread_text is None:
        return None, None
    try:
        import pandas as pd
        if pd.isna(spread_text):
            return None, None
    except ImportError:
        if spread_text != spread_text:
            return None, None

    text = str(spread_text).strip()
    if not text or text.lower() == 'nan':
        return None, None

    # Standalone pick'em lines
    if text.upper() in ['PK', 'PICK', 'EVEN']:
        return "PK", 0.0

    # Team + PK/EVEN pattern e.g. "BOS PK", "MIA EVEN", "BOS PICK"
    m_pk = re.match(r'^([A-Za-z]+)\s+(?:PK|PICK|EVEN)$', text, re.IGNORECASE)
    if m_pk:
        return m_pk.group(1).upper(), 0.0

    # Team + numeric spread e.g. "BOS -4.5", "DAL +6", "NYK -3.5"
    m_spread = re.match(r'^([A-Za-z]+)\s*([+-]?\d+(?:\.\d+)?)$', text)
    if m_spread:
        return m_spread.group(1).upper(), float(m_spread.group(2))

    # Just a number e.g. "-4.5", "3"
    try:
        return None, float(text)
    except (ValueError, TypeError):
        pass

    return None, None


def calculate_home_spread(spread_text: Any, home_team: str, away_team: str) -> Optional[float]:
    """Calculate the point spread from the home team's perspective.
    Negative means home team is favored (e.g. -4.5).
    Positive means away team is favored.
    """
    team_abbr, spread_val = parse_spread(spread_text)
    if spread_val is None:
        return None
    if team_abbr is None or team_abbr == "PK":
        return spread_val

    # Determine if the parsed team is the away team
    info = resolve_team_info(team_abbr)
    away_info = resolve_team_info(away_team)
    
    is_away = False
    if info and away_info and info.get("abbreviation") == away_info.get("abbreviation"):
        is_away = True
    elif info and info.get("name", "").lower() in away_team.lower():
        is_away = True
    elif team_abbr.upper() == away_team.strip().upper():
        is_away = True

    return -spread_val if is_away else spread_val


def translate_recommendation(
    rec: Any,
    spread_line: Optional[float] = None,
    home_team: Optional[str] = None,
    away_team: Optional[str] = None
) -> str:
    """Translate model recommendation to human-readable Thai text.
    
    Dynamically accounts for whether the picked team is the favorite (ต่อ)
    or underdog (รอง) based on spread_line (from home team perspective).
    If home_team and away_team are provided, embeds the target team name and line.
    
    If spread_line < 0: Home is favorite (-), Away is underdog (+)
    If spread_line > 0: Home is underdog (+), Away is favorite (-)
    If spread_line == 0: Even line / Pick'em (เลือก)
    """
    if not rec:
        return ""
    rec_str = str(rec).strip().upper()

    # Totals (Over / Under)
    if 'OVER' in rec_str:
        if 'STRONG' in rec_str:
            return '🔥 สูง (มั่นใจมาก)'
        elif 'LEAN' in rec_str:
            return '📈 สูง (น่าสนใจ)'
        return 'สูง (OVER)'
    if 'UNDER' in rec_str:
        if 'STRONG' in rec_str:
            return '🔥 ต่ำ (มั่นใจมาก)'
        elif 'LEAN' in rec_str:
            return '📉 ต่ำ (น่าสนใจ)'
        return 'ต่ำ (UNDER)'

    # Spread recommendations (HOME / AWAY)
    is_home = 'HOME' in rec_str
    is_away = 'AWAY' in rec_str
    is_strong = 'STRONG' in rec_str
    is_lean = 'LEAN' in rec_str

    if is_home or is_away:
        role = ""
        pts = ""
        if spread_line is not None:
            try:
                s_val = float(spread_line)
                if s_val < 0:
                    # Home is favorite (-), Away is underdog (+)
                    if is_home:
                        role = "ต่อ"
                        pts = f"{s_val:+.1f}"
                    else:
                        role = "รอง"
                        pts = f"+{abs(s_val):.1f}"
                elif s_val > 0:
                    # Home is underdog (+), Away is favorite (-)
                    if is_home:
                        role = "รอง"
                        pts = f"+{s_val:.1f}"
                    else:
                        role = "ต่อ"
                        pts = f"{-s_val:+.1f}"
                else:
                    role = "เลือก"
                    pts = "0.0"
            except (ValueError, TypeError):
                role = ""
                pts = ""

        # Determine target team display
        target_name = None
        if is_home and home_team:
            target_name = home_team
        elif is_away and away_team:
            target_name = away_team

        if target_name:
            pts_str = f" ({pts})" if pts else ""
            label = f"{role} {target_name}{pts_str}" if role else target_name
        else:
            base_target = "เจ้าบ้าน" if is_home else "ทีมเยือน"
            label = f"{role}{base_target}" if role else base_target

        if is_strong:
            return f"🔥 {label} (มั่นใจมาก)"
        elif is_lean:
            icon = "🏠" if is_home else "✈️"
            return f"{icon} {label} (น่าสนใจ)"
        return label

    # Neutral variants
    if 'NEUTRAL' in rec_str:
        if 'HOME' in rec_str:
            return '⚖️ ข้าม (แต่แอบเอียงไปทางเจ้าบ้าน)'
        if 'AWAY' in rec_str:
            return '⚖️ ข้าม (แต่แอบเอียงไปทางทีมเยือน)'
        if 'UNDER' in rec_str:
            return '⚖️ ข้าม (แต่แอบเอียงไปทางต่ำ)'
        if 'OVER' in rec_str:
            return '⚖️ ข้าม (แต่แอบเอียงไปทางสูง)'
        return '⚖️ ข้าม (สูสีมาก)'

    return str(rec)


t_rec = translate_recommendation

