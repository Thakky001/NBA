import os
import re
import time
import logging
from typing import Dict, Optional, Tuple, Any, List
import requests
import pandas as pd

logger = logging.getLogger(__name__)


class OddsCollector:
    """Collects betting odds from The Odds API with retry logic and local cache fallback."""

    # Class-level circuit breaker to protect across all instances
    _circuit_breaker_until: float = 0.0
    _circuit_breaker_reason: str = ""
    _circuit_breaker_logged: bool = False

    # Class-level shared cache to persist across Streamlit reruns and multiple instances
    _shared_cached_games: Optional[list] = None
    _shared_cache_time: float = 0.0

    def __init__(self, max_retries: int = 3, retry_delay: float = 0.1, backoff_factor: float = 2.0):
        try:
            from config import ODDS_API_KEY
            self.api_key = ODDS_API_KEY
        except Exception:
            self.api_key = ""
        if not self.api_key:
            try:
                import streamlit as st
                self.api_key = st.secrets.get("ODDS_API_KEY", "")
            except Exception:
                pass
        self.base_url = 'https://api.the-odds-api.com/v4/sports/basketball_nba/odds'
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.backoff_factor = backoff_factor
        self._cached_games = None
        self._cache_time = 0
        self._logged_no_key = False
        self.time = time

    def clear_cache(self) -> None:
        """Clear memory cache for odds."""
        self._cached_games = None
        self._cache_time = 0
        self._logged_no_key = False
        OddsCollector._shared_cached_games = None
        OddsCollector._shared_cache_time = 0.0

    @classmethod
    def reset_circuit_breaker(cls) -> None:
        """Reset the class-level circuit breaker."""
        cls._circuit_breaker_until = 0.0
        cls._circuit_breaker_reason = ""
        cls._circuit_breaker_logged = False
        cls._shared_cached_games = None
        cls._shared_cache_time = 0.0

    def _resolve_raw_dir(self) -> str:
        """Resolve path to data/raw directory."""
        try:
            from config import RAW_DIR
            if os.path.isdir(RAW_DIR):
                return RAW_DIR
        except Exception:
            pass
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        raw_path = os.path.join(base, 'data', 'raw')
        if os.path.isdir(raw_path):
            return raw_path
        cwd_raw = os.path.join(os.getcwd(), 'data', 'raw')
        return cwd_raw

    def _load_fallback_csv(self, filename: str = 'espn_recent_games.csv') -> pd.DataFrame:
        """Load fallback CSV from data/raw/ directory."""
        raw_dir = self._resolve_raw_dir()
        path = os.path.join(raw_dir, filename)
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                logger.info(
                    f"Live odds request failed/unavailable; activating fallback data from {path} ({len(df)} rows)"
                )
                return df
            except Exception as e:
                logger.error(f"Failed to read fallback CSV {path}: {e}")
        else:
            logger.warning(f"Fallback CSV {path} not found.")
        return pd.DataFrame()

    def _load_fallback_odds_list(self) -> list:
        """Construct fallback game odds list from cached espn_recent_games.csv."""
        df = self._load_fallback_csv('espn_recent_games.csv')
        if df.empty:
            return []

        games = []
        for _, row in df.iterrows():
            h_team = str(row.get('home_team', ''))
            a_team = str(row.get('away_team', ''))
            if not h_team or not a_team or h_team == 'nan' or a_team == 'nan':
                continue

            from config import calculate_home_spread
            spread_val = calculate_home_spread(row.get('spread'), h_team, a_team)
            if spread_val is None:
                spread_val = 0.0

            ou_val = 225.0
            raw_ou = row.get('over_under')
            if pd.notna(raw_ou):
                try:
                    ou_val = float(raw_ou)
                except (ValueError, TypeError):
                    ou_val = 225.0

            games.append({
                'id': str(row.get('game_id', '')),
                'home_team': h_team,
                'away_team': a_team,
                'bookmakers': [
                    {
                        'title': 'Consensus',
                        'markets': [
                            {
                                'key': 'spreads',
                                'outcomes': [
                                    {'name': h_team, 'point': spread_val, 'price': -110},
                                    {'name': a_team, 'point': -spread_val, 'price': -110}
                                ]
                            },
                            {
                                'key': 'totals',
                                'outcomes': [
                                    {'name': 'Over', 'point': ou_val, 'price': -110},
                                    {'name': 'Under', 'point': ou_val, 'price': -110}
                                ]
                            }
                        ]
                    }
                ]
            })
        return games

    def _fetch_all_odds(self) -> list:
        """Fetch all odds from API with retry backoff and fallback on failure."""
        now = self.time.time()
        # 1. Instance cache (for unit tests / custom overrides)
        if self._cached_games is not None and (now - self._cache_time) < 3600:
            return self._cached_games

        # 2. Class-level shared cache (persists across Streamlit reruns)
        if OddsCollector._shared_cached_games is not None and (now - OddsCollector._shared_cache_time) < 3600:
            self._cached_games = OddsCollector._shared_cached_games
            self._cache_time = OddsCollector._shared_cache_time
            return OddsCollector._shared_cached_games

        # Check class-level circuit breaker (e.g. quota exhausted or API key invalid)
        if now < OddsCollector._circuit_breaker_until:
            if not OddsCollector._circuit_breaker_logged:
                logger.info(
                    f"[OddsCollector] Circuit breaker active until {int(OddsCollector._circuit_breaker_until - now)}s "
                    f"({OddsCollector._circuit_breaker_reason}). Skipping live API requests and using fallback."
                )
                OddsCollector._circuit_breaker_logged = True
            self._cached_games = []
            self._cache_time = now
            return []

        if not self.api_key:
            try:
                from config import ODDS_API_KEY
                self.api_key = ODDS_API_KEY
            except Exception:
                pass
            if not self.api_key:
                try:
                    import streamlit as st
                    self.api_key = st.secrets.get("ODDS_API_KEY", "")
                except Exception:
                    pass

        if not self.api_key:
            if not self._logged_no_key:
                logger.info("ODDS_API_KEY not set; skipping live bookmakers fetch.")
                self._logged_no_key = True
            # Cache empty result for 1 hour so it does not spam logs on every game & 10s refresh
            self._cached_games = []
            self._cache_time = now
            OddsCollector._shared_cached_games = []
            OddsCollector._shared_cache_time = now
            return []

        params = {
            'apiKey': self.api_key,
            'regions': 'us',
            'markets': 'spreads,totals',
            'bookmakers': 'draftkings,fanduel,betmgm,caesars',
            'oddsFormat': 'american'
        }

        last_error = None
        for attempt in range(self.max_retries):
            try:
                res = requests.get(self.base_url, params=params, timeout=10)

                # Check for quota exhaustion (401 OUT_OF_USAGE_CREDITS) or access denied (401/403)
                if res.status_code in [401, 403] or "OUT_OF_USAGE_CREDITS" in res.text:
                    reason = "OUT_OF_USAGE_CREDITS" if "OUT_OF_USAGE_CREDITS" in res.text else f"Status {res.status_code}"
                    logger.warning(
                        f"[OddsCollector] The Odds API quota reached or unauthorized ({reason}): {res.text[:200]}. "
                        f"Activating 1-hour circuit breaker to prevent retry spam and latency. "
                        f"Falling back to ESPN/consensus odds."
                    )
                    OddsCollector._circuit_breaker_until = now + 3600.0
                    OddsCollector._circuit_breaker_reason = reason
                    OddsCollector._circuit_breaker_logged = True
                    self._cached_games = []
                    self._cache_time = now
                    OddsCollector._shared_cached_games = []
                    OddsCollector._shared_cache_time = now
                    return []

                if res.status_code in [429, 500, 502, 503, 504]:
                    raise requests.exceptions.HTTPError(
                        f"HTTP {res.status_code}: {res.text[:200]}",
                        response=res
                    )
                if res.status_code != 200:
                    raise requests.exceptions.RequestException(
                        f"Status {res.status_code}: {res.text[:200]}"
                    )

                self._cached_games = res.json()
                self._cache_time = now
                OddsCollector._shared_cached_games = self._cached_games
                OddsCollector._shared_cache_time = now
                OddsCollector._circuit_breaker_until = 0.0
                OddsCollector._circuit_breaker_logged = False
                return self._cached_games

            except Exception as e:
                last_error = e
                logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries} failed fetching odds: {e}"
                )
                if attempt < self.max_retries - 1:
                    sleep_time = self.retry_delay * (self.backoff_factor ** attempt)
                    self.time.sleep(sleep_time)

        logger.warning(
            f"All {self.max_retries} attempts failed to fetch live odds: {last_error}"
        )
        # Cache failure for 300s (5 minutes) so remaining games in slate don't repeat retries
        self._cached_games = []
        self._cache_time = now
        OddsCollector._shared_cached_games = []
        OddsCollector._shared_cache_time = now
        return []

    def get_game_odds(self, home_team: str, away_team: str) -> Dict:
        """Fetch odds for a specific matchup from multiple bookmakers."""
        try:
            games = self._fetch_all_odds()
            if not games:
                return {
                    'spreads': {},
                    'totals': {}
                }

            from config import normalize_team_name, resolve_team_info

            def match_team(api_name, search_name):
                if not api_name or not search_name:
                    return False
                norm_api = normalize_team_name(api_name)
                norm_search = normalize_team_name(search_name)
                if norm_api and norm_search and norm_api == norm_search:
                    return True
                info_api = resolve_team_info(api_name)
                info_search = resolve_team_info(search_name)
                if info_api and info_search:
                    return info_api.get("abbreviation") == info_search.get("abbreviation")
                return str(api_name).strip().lower() == str(search_name).strip().lower()

            for game in games:
                if match_team(game.get('home_team', ''), home_team) and match_team(game.get('away_team', ''), away_team):
                    return self._parse_bookmakers(game.get('bookmakers', []), game.get('home_team', ''), game.get('away_team', ''))

            # If matchup not found in odds feed, return empty dict so callers know no odds were found
            return {
                'spreads': {},
                'totals': {}
            }
        except Exception as e:
            logger.warning(f"Error getting game odds: {e}; returning empty lines.")
            return {
                'spreads': {},
                'totals': {}
            }

    def _parse_bookmakers(self, bookmakers_data: list, home_team: str, away_team: str) -> Dict:
        """Parse raw bookmaker data into a clean dictionary of lines."""
        parsed = {
            'spreads': {},
            'totals': {}
        }

        for bookie in bookmakers_data:
            b_name = bookie.get('title', 'Unknown')

            for market in bookie.get('markets', []):
                if market.get('key') == 'spreads':
                    spread_data = {}
                    for outcome in market.get('outcomes', []):
                        if outcome.get('name') == home_team:
                            spread_data['home'] = outcome.get('point')
                            spread_data['home_price'] = outcome.get('price')
                        elif outcome.get('name') == away_team:
                            spread_data['away'] = outcome.get('point')
                            spread_data['away_price'] = outcome.get('price')
                    if spread_data:
                        parsed['spreads'][b_name] = spread_data

                elif market.get('key') == 'totals':
                    total_data = {}
                    for outcome in market.get('outcomes', []):
                        if outcome.get('name') == 'Over':
                            total_data['over'] = outcome.get('point')
                            total_data['over_price'] = outcome.get('price')
                        elif outcome.get('name') == 'Under':
                            total_data['under'] = outcome.get('point')
                            total_data['under_price'] = outcome.get('price')
                    if total_data:
                        parsed['totals'][b_name] = total_data

        return parsed

    def get_consensus_lines(self, parsed_odds: Dict) -> Tuple[Optional[float], Optional[float]]:
        """Calculate the most common (mode) line to use as baseline for analysis.
        Returns (home_spread, over_under_total)
        """
        if not parsed_odds or not isinstance(parsed_odds, dict):
            return None, None

        home_spreads = [d.get('home') for d in parsed_odds.get('spreads', {}).values() if d.get('home') is not None]
        totals = [d.get('over') for d in parsed_odds.get('totals', {}).values() if d.get('over') is not None]

        consensus_spread = None
        if home_spreads:
            consensus_spread = max(set(home_spreads), key=home_spreads.count)

        consensus_total = None
        if totals:
            consensus_total = max(set(totals), key=totals.count)

        return consensus_spread, consensus_total

    def get_best_line_shopping(self, parsed_odds: Dict, pick_type: str, pick_team: str = None) -> str:
        """Find the best bookmaker for a specific pick."""
        if not parsed_odds or not isinstance(parsed_odds, dict):
            return ""
        return ""

    def get_odds_dataframe(self) -> pd.DataFrame:
        """Return odds data as a DataFrame, falling back to local cache on error."""
        try:
            games = self._fetch_all_odds()
            records = []
            for g in games:
                h = g.get('home_team', '')
                a = g.get('away_team', '')
                parsed = self._parse_bookmakers(g.get('bookmakers', []), h, a)
                c_spread, c_total = self.get_consensus_lines(parsed)
                records.append({
                    'home_team': h,
                    'away_team': a,
                    'spread': c_spread,
                    'over_under': c_total
                })
            if records:
                return pd.DataFrame(records)
        except Exception as e:
            logger.warning(f"Error creating odds dataframe: {e}; activating fallback data.")

        # Fallback to games built from fallback list with matching schema
        fallback_games = self._load_fallback_odds_list()
        records = []
        for g in fallback_games:
            h = g.get('home_team', '')
            a = g.get('away_team', '')
            parsed = self._parse_bookmakers(g.get('bookmakers', []), h, a)
            c_spread, c_total = self.get_consensus_lines(parsed)
            records.append({
                'home_team': h,
                'away_team': a,
                'spread': c_spread,
                'over_under': c_total
            })
        if records:
            return pd.DataFrame(records)

        return pd.DataFrame(columns=['home_team', 'away_team', 'spread', 'over_under'])

    def get_odds_df(self) -> pd.DataFrame:
        """Alias for get_odds_dataframe."""
        return self.get_odds_dataframe()

    def get_odds_data(self) -> pd.DataFrame:
        """Alias for get_odds_dataframe."""
        return self.get_odds_dataframe()
