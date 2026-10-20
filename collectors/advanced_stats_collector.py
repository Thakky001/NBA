import os
import time
import logging
from typing import Optional, Dict, Any, List
import pandas as pd

logger = logging.getLogger(__name__)


class AdvancedStatsCollector:
    """Collects NBA advanced stats via nba_api with retry logic and local cache fallback."""

    LEAGUE_ID = '00'

    def __init__(self, season: str = '2025-26', max_retries: int = 3, retry_delay: float = 0.1, backoff_factor: float = 2.0):
        self.season = season
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.backoff_factor = backoff_factor
        try:
            from nba_api.stats.library.http import NBAStatsHTTP
        except ImportError:
            logger.warning("nba_api not installed. Run: pip install nba_api")

    EXPECTED_SCHEMAS = {
        'nba_api_base.csv': ['TEAM_ID', 'TEAM_NAME', 'GP', 'W', 'L', 'W_PCT', 'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK', 'BLKA', 'PF', 'PFD', 'PTS', 'PLUS_MINUS'],
        'nba_api_advanced.csv': ['TEAM_ID', 'TEAM_NAME', 'GP', 'W', 'L', 'W_PCT', 'MIN', 'E_OFF_RATING', 'OFF_RATING', 'E_DEF_RATING', 'DEF_RATING', 'E_NET_RATING', 'NET_RATING', 'AST_PCT', 'AST_TO', 'AST_RATIO', 'OREB_PCT', 'DREB_PCT', 'REB_PCT', 'TM_TOV_PCT', 'EFG_PCT', 'TS_PCT', 'E_PACE', 'PACE', 'PACE_PER40', 'POSS', 'PIE'],
        'nba_api_four_factors.csv': ['TEAM_ID', 'TEAM_NAME', 'GP', 'W', 'L', 'W_PCT', 'MIN', 'EFG_PCT', 'FTA_RATE', 'TM_TOV_PCT', 'OREB_PCT', 'OPP_EFG_PCT', 'OPP_FTA_RATE', 'OPP_TOV_PCT', 'OPP_OREB_PCT'],
        'draft_history.csv': ['PERSON_ID', 'PLAYER_NAME', 'SEASON', 'ROUND_NUMBER', 'ROUND_PICK', 'OVERALL_PICK'],
        'nba_api_draft_history.csv': ['PERSON_ID', 'PLAYER_NAME', 'SEASON', 'ROUND_NUMBER', 'ROUND_PICK', 'OVERALL_PICK'],
        'nba_api_players.csv': ['PLAYER_ID', 'PLAYER_NAME', 'NICKNAME', 'TEAM_ID', 'TEAM_ABBREVIATION', 'AGE', 'GP', 'W', 'L', 'W_PCT', 'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK', 'BLKA', 'PF', 'PFD', 'PTS', 'PLUS_MINUS'],
        'nba_api_game_logs.csv': ['SEASON_ID', 'TEAM_ID', 'TEAM_ABBREVIATION', 'TEAM_NAME', 'GAME_ID', 'GAME_DATE', 'MATCHUP', 'WL', 'MIN', 'PTS'],
    }

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

    def _load_fallback_csv(self, filename: str, expected_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """Load fallback CSV from data/raw/ directory with schema coercion and corruption tolerance."""
        if expected_cols is None:
            expected_cols = self.EXPECTED_SCHEMAS.get(filename)
        raw_dir = self._resolve_raw_dir()
        path = os.path.join(raw_dir, filename)
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, on_bad_lines='skip')
                if expected_cols:
                    for col in expected_cols:
                        if col not in df.columns:
                            df[col] = None
                logger.warning(
                    f"API request failed; activating fallback data from {path} ({len(df)} rows)"
                )
                return df
            except Exception as e:
                logger.error(f"Failed to read fallback CSV {path}: {e}")
        else:
            logger.warning(f"Fallback CSV {path} not found.")
        return pd.DataFrame(columns=expected_cols if expected_cols else None)

    def _retry_call(self, func, *args, **kwargs):
        """Execute a callable with retry logic and exponential backoff."""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries} failed for API call: {e}"
                )
                if attempt < self.max_retries - 1:
                    sleep_time = self.retry_delay * (self.backoff_factor ** attempt)
                    if hasattr(e, 'response') and e.response is not None:
                        retry_after = e.response.headers.get('Retry-After')
                        if retry_after:
                            try:
                                sleep_time = max(float(retry_after), sleep_time)
                            except (ValueError, TypeError):
                                pass
                    time.sleep(sleep_time)
        logger.warning(f"All {self.max_retries} attempts failed for API call: {last_error}")
        return None

    def get_team_stats_base(self) -> pd.DataFrame:
        """Get base team stats (PTS, REB, AST, FG%, etc.)."""
        logger.info(f"Fetching base team stats for {self.season}")
        def _fetch():
            from nba_api.stats.endpoints import leaguedashteamstats
            result = leaguedashteamstats.LeagueDashTeamStats(
                league_id_nullable=self.LEAGUE_ID,
                season=self.season,
                season_type_all_star='Regular Season'
            )
            frames = result.get_data_frames()
            return frames[0] if frames else pd.DataFrame()

        try:
            df = self._retry_call(_fetch)
            if df is not None and not df.empty:
                logger.info(f"Got base stats for {len(df)} teams")
                return df
        except Exception as e:
            logger.warning(f"Unexpected error fetching base team stats: {e}")

        return self._load_fallback_csv('nba_api_base.csv')

    def get_team_stats_advanced(self) -> pd.DataFrame:
        """Get advanced team stats (Pace, ORtg, DRtg, Net Rating, eFG%, TS%).
        Key method for O/U and Spread analysis.
        """
        logger.info(f"Fetching advanced team stats for {self.season}")
        def _fetch():
            from nba_api.stats.endpoints import leaguedashteamstats
            result = leaguedashteamstats.LeagueDashTeamStats(
                league_id_nullable=self.LEAGUE_ID,
                season=self.season,
                season_type_all_star='Regular Season',
                measure_type_detailed_defense='Advanced'
            )
            frames = result.get_data_frames()
            return frames[0] if frames else pd.DataFrame()

        try:
            df = self._retry_call(_fetch)
            if df is not None and not df.empty:
                logger.info(f"Got advanced stats for {len(df)} teams")
                return df
        except Exception as e:
            logger.warning(f"Unexpected error fetching advanced team stats: {e}")

        return self._load_fallback_csv('nba_api_advanced.csv')

    def get_team_four_factors(self) -> pd.DataFrame:
        """Get Four Factors stats (eFG%, TOV%, ORB%, FTr).
        The Four Factors are key predictors of winning/scoring.
        """
        logger.info(f"Fetching four factors for {self.season}")
        def _fetch():
            from nba_api.stats.endpoints import leaguedashteamstats
            result = leaguedashteamstats.LeagueDashTeamStats(
                league_id_nullable=self.LEAGUE_ID,
                season=self.season,
                season_type_all_star='Regular Season',
                measure_type_detailed_defense='Four Factors'
            )
            frames = result.get_data_frames()
            return frames[0] if frames else pd.DataFrame()

        try:
            df = self._retry_call(_fetch)
            if df is not None and not df.empty:
                logger.info(f"Got four factors for {len(df)} teams")
                return df
        except Exception as e:
            logger.warning(f"Unexpected error fetching four factors: {e}")

        return self._load_fallback_csv('nba_api_four_factors.csv')

    def fetch_draft_history(self, num_years: int = 2) -> pd.DataFrame:
        """Fetch recent draft history to evaluate rookies."""
        logger.info("Fetching Draft History for Rookie evaluation...")
        def _fetch():
            from nba_api.stats.endpoints import drafthistory
            from config import STATS_HEADERS, NBA_API_SEASON
            season_year = int(NBA_API_SEASON.split('-')[0])
            dh = drafthistory.DraftHistory(
                headers=STATS_HEADERS,
                timeout=15
            )
            frames = dh.get_data_frames()
            if not frames:
                return pd.DataFrame()
            df_live = frames[0]
            recent_years = [str(season_year - i) for i in range(num_years)]
            if 'SEASON' in df_live.columns:
                df_live = df_live[df_live['SEASON'].astype(str).isin(recent_years)].copy()
            cols = ['PERSON_ID', 'PLAYER_NAME', 'SEASON', 'ROUND_NUMBER', 'ROUND_PICK', 'OVERALL_PICK']
            avail_cols = [c for c in cols if c in df_live.columns]
            return df_live[avail_cols]

        try:
            df = self._retry_call(_fetch)
            if df is not None and not df.empty:
                return df
        except Exception as e:
            logger.warning(f"Unexpected error fetching draft history: {e}")

        # Try draft_history.csv, then nba_api_draft_history.csv
        cached = self._load_fallback_csv('draft_history.csv')
        if cached.empty:
            cached = self._load_fallback_csv('nba_api_draft_history.csv')
        return cached

    def get_player_stats(self) -> pd.DataFrame:
        """Get player-level stats for the season."""
        logger.info(f"Fetching player stats for {self.season}")
        def _fetch():
            from nba_api.stats.endpoints import leaguedashplayerstats
            result = leaguedashplayerstats.LeagueDashPlayerStats(
                league_id_nullable=self.LEAGUE_ID,
                season=self.season,
                season_type_all_star='Regular Season'
            )
            frames = result.get_data_frames()
            return frames[0] if frames else pd.DataFrame()

        try:
            df = self._retry_call(_fetch)
            if df is not None and not df.empty:
                logger.info(f"Got stats for {len(df)} players")
                return df
        except Exception as e:
            logger.warning(f"Unexpected error fetching player stats: {e}")

        return self._load_fallback_csv('nba_api_players.csv')

    def get_game_logs(self, player_or_team: str = 'T') -> pd.DataFrame:
        """Get game-by-game logs for the season.
        Args:
            player_or_team: 'P' for player logs, 'T' for team logs
        """
        logger.info(f"Fetching {'team' if player_or_team == 'T' else 'player'} game logs for {self.season}")
        def _fetch():
            from nba_api.stats.endpoints import leaguegamelog
            result = leaguegamelog.LeagueGameLog(
                season=self.season,
                season_type_all_star='Regular Season',
                player_or_team_abbreviation=player_or_team
            )
            frames = result.get_data_frames()
            return frames[0] if frames else pd.DataFrame()

        try:
            df = self._retry_call(_fetch)
            if df is not None and not df.empty:
                logger.info(f"Got {len(df)} game log entries")
                return df
        except Exception as e:
            logger.warning(f"Unexpected error fetching game logs: {e}")

        return self._load_fallback_csv('nba_api_game_logs.csv')

    def get_all_advanced_data(self) -> dict:
        """Fetch all advanced datasets with delays between requests.
        Returns dict with keys: 'base', 'advanced', 'four_factors', 'players', 'game_logs', 'draft_history'
        """
        data = {}
        data['base'] = self.get_team_stats_base()
        time.sleep(0.1)

        data['advanced'] = self.get_team_stats_advanced()
        time.sleep(0.1)

        data['four_factors'] = self.get_team_four_factors()
        time.sleep(0.1)

        data['players'] = self.get_player_stats()
        time.sleep(0.1)

        data['game_logs'] = self.get_game_logs('T')
        time.sleep(0.1)

        data['draft_history'] = self.fetch_draft_history()
        return data

    def save_raw_data(self, output_dir: str = None):
        """Download and save all advanced stats to disk."""
        if output_dir is None:
            output_dir = self._resolve_raw_dir()
        os.makedirs(output_dir, exist_ok=True)

        data = self.get_all_advanced_data()
        for name, df in data.items():
            if df is not None and not df.empty:
                fname = f'draft_history.csv' if name == 'draft_history' else f'nba_api_{name}.csv'
                path = os.path.join(output_dir, fname)
                try:
                    df.to_csv(path, index=False)
                    logger.info(f"Saved {name} to {path} ({len(df)} rows)")
                except Exception as e:
                    logger.error(f"Failed to save {name} to {path}: {e}")
