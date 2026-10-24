"""RotoWire NBA Lineups Collector.

Collects real-time confirmed and expected starting lineups (PG, SG, SF, PF, C)
and injury statuses from RotoWire NBA Lineups:
https://www.rotowire.com/basketball/nba-lineups.php

Features:
- Extracts confirmed vs expected lineup status
- Extracts 5 starters per team with positional roles
- Caches scraped data for 20 minutes to prevent rate limiting / IP bans
- 100% silent fallback on timeout, parse error, or network error
"""

import os
import time
import json
import logging
from typing import Dict, Any, List, Optional
import requests
from bs4 import BeautifulSoup

from config import RAW_DIR, resolve_team_info, normalize_team_name

logger = logging.getLogger(__name__)


class RotoWireCollector:
    """Scrapes and caches NBA starting lineups and lineup status from RotoWire."""

    URL = "https://www.rotowire.com/basketball/nba-lineups.php"
    CACHE_FILENAME = "rotowire_lineups.json"
    CACHE_TTL_SECONDS = 1200  # 20 minutes

    def __init__(self, cache_ttl: int = CACHE_TTL_SECONDS, timeout: int = 10):
        self.cache_ttl = cache_ttl
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        })
        self._memory_cache: Optional[Dict[str, Any]] = None
        self._cache_mtime: float = 0.0

    def _get_cache_path(self) -> str:
        """Resolve path to local cache JSON file."""
        os.makedirs(RAW_DIR, exist_ok=True)
        return os.path.join(RAW_DIR, self.CACHE_FILENAME)

    def _load_cache(self) -> Optional[Dict[str, Any]]:
        """Load cached lineups from disk if not expired."""
        cache_path = self._get_cache_path()
        if not os.path.exists(cache_path):
            return None
        try:
            mtime = os.path.getmtime(cache_path)
            if (time.time() - mtime) < self.cache_ttl:
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._memory_cache = data
                self._cache_mtime = mtime
                logger.debug(f"Loaded RotoWire lineups from cache ({len(data)} teams)")
                return data
        except Exception as e:
            logger.warning(f"Error reading RotoWire cache: {e}")
        return None

    def _save_cache(self, data: Dict[str, Any]) -> None:
        """Save lineups data to local cache JSON file."""
        if not data:
            return
        cache_path = self._get_cache_path()
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self._memory_cache = data
            self._cache_mtime = time.time()
        except Exception as e:
            logger.warning(f"Error saving RotoWire cache: {e}")

    def parse_html(self, html_text: str) -> Dict[str, Dict[str, Any]]:
        """Parse RotoWire lineups HTML page into structured team dictionary."""
        if not html_text:
            return {}

        soup = BeautifulSoup(html_text, "html.parser")
        boxes = soup.find_all("div", class_=lambda c: c and "lineup" in c and "is-nba" in c)
        if not boxes:
            # Fallback to general lineup class
            boxes = soup.find_all("div", class_="lineup")

        results: Dict[str, Dict[str, Any]] = {}

        for box in boxes:
            # 1. Identify visit (away) and home teams
            visit_link = box.find(class_=lambda c: c and "lineup__team" in c and "is-visit" in c)
            home_link = box.find(class_=lambda c: c and "lineup__team" in c and "is-home" in c)

            visit_abbr_el = visit_link.find(class_="lineup__abbr") if visit_link else None
            home_abbr_el = home_link.find(class_="lineup__abbr") if home_link else None

            raw_visit_abbr = visit_abbr_el.get_text(strip=True).upper() if visit_abbr_el else ""
            raw_home_abbr = home_abbr_el.get_text(strip=True).upper() if home_abbr_el else ""

            if not raw_visit_abbr or not raw_home_abbr:
                continue

            # Standardize abbreviations to project conventions
            visit_info = resolve_team_info(raw_visit_abbr)
            home_info = resolve_team_info(raw_home_abbr)

            visit_team_name = visit_info["name"] if visit_info else raw_visit_abbr
            home_team_name = home_info["name"] if home_info else raw_home_abbr

            visit_abbr = visit_info["abbreviation"] if visit_info else raw_visit_abbr
            home_abbr = home_info["abbreviation"] if home_info else raw_home_abbr

            # 2. Extract lineup lists (away and home)
            list_visit = box.find(class_=lambda c: c and "lineup__list" in c and "is-visit" in c)
            list_home = box.find(class_=lambda c: c and "lineup__list" in c and "is-home" in c)

            for side, team_name, team_abbr, opp_name, opp_abbr, is_home, team_list in [
                ("away", visit_team_name, visit_abbr, home_team_name, home_abbr, False, list_visit),
                ("home", home_team_name, home_abbr, visit_team_name, visit_abbr, True, list_home),
            ]:
                if not team_list:
                    continue

                # Status (Confirmed vs Expected)
                status_el = team_list.find(class_=lambda c: c and "lineup__status" in c)
                status_text = status_el.get_text(strip=True) if status_el else "Expected Lineup"
                is_confirmed = "confirmed" in status_text.lower()
                status_clean = "Confirmed" if is_confirmed else "Expected"

                # Extract Starters (items before 'lineup__title is-middle')
                starters: List[Dict[str, str]] = []
                injuries: List[Dict[str, str]] = []

                reached_injuries = False
                for li in team_list.find_all("li"):
                    classes = li.get("class") or []
                    if any("lineup__title" in c for c in classes) or "may not play" in li.get_text().lower():
                        reached_injuries = True
                        continue

                    if not any("lineup__player" in c for c in classes):
                        continue

                    pos_el = li.find(class_="lineup__pos")
                    pos = pos_el.get_text(strip=True) if pos_el else ""
                    a_tag = li.find("a")
                    player_name = a_tag.get("title") or a_tag.get_text(strip=True) if a_tag else ""
                    if not player_name:
                        # Fallback to text inside li without position
                        full_txt = li.get_text(" ", strip=True)
                        player_name = full_txt.replace(pos, "").strip()

                    play_status = li.get("title", "Likely")

                    player_dict = {
                        "name": player_name,
                        "position": pos,
                        "status": play_status
                    }

                    if not reached_injuries and len(starters) < 5:
                        starters.append(player_dict)
                    else:
                        injuries.append(player_dict)

                entry = {
                    "team": team_name,
                    "abbreviation": team_abbr,
                    "opponent": opp_name,
                    "opponent_abbr": opp_abbr,
                    "is_home": is_home,
                    "lineup_status": status_clean,
                    "is_confirmed": is_confirmed,
                    "starters": starters,
                    "injuries": injuries,
                    "scraped_at": time.strftime("%Y-%m-%d %H:%M:%S")
                }

                # Store with both full team name and abbreviation for easy lookup
                results[team_name] = entry
                results[team_abbr] = entry

        return results

    def get_todays_lineups(self, force_refresh: bool = False) -> Dict[str, Dict[str, Any]]:
        """Fetch today's lineups from RotoWire or load from cache.
        Returns dictionary keyed by team name and abbreviation.
        """
        # 1. Check in-memory / file cache first
        if not force_refresh:
            cached = self._load_cache()
            if cached:
                return cached

        # 2. Make live HTTP GET request
        try:
            logger.info(f"Fetching live NBA lineups from {self.URL}...")
            response = self.session.get(self.URL, timeout=self.timeout)
            response.raise_for_status()
            parsed = self.parse_html(response.text)
            if parsed:
                self._save_cache(parsed)
                logger.info(f"Successfully scraped and cached {len(parsed)//2} matchups from RotoWire.")
                return parsed
            else:
                logger.warning("RotoWire returned 200 OK but 0 matchups parsed.")
        except Exception as e:
            logger.warning(f"Failed to fetch live RotoWire lineups: {e}")

        # 3. Fallback to expired cache if available
        cache_path = self._get_cache_path()
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    expired_data = json.load(f)
                logger.info("Using expired RotoWire cache as fallback.")
                return expired_data
            except Exception:
                pass

        return {}

    def get_lineup_for_team(self, team_identifier: str) -> Optional[Dict[str, Any]]:
        """Lookup lineup data for a specific team (by full name, alias, or abbreviation)."""
        lineups = self.get_todays_lineups()
        if not lineups:
            return None

        # 1. Direct hit
        if team_identifier in lineups:
            return lineups[team_identifier]

        # 2. Resolve via canonical config
        info = resolve_team_info(team_identifier)
        if info:
            abbr = info.get("abbreviation")
            name = info.get("name")
            if abbr and abbr in lineups:
                return lineups[abbr]
            if name and name in lineups:
                return lineups[name]

        return None
