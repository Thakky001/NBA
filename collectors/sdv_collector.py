import os
import time
import logging
from typing import List, Optional, Dict, Any
import pandas as pd

logger = logging.getLogger(__name__)


class SDVCollector:
    """Collects historical NBA data via sportsdataverse package with retry logic and local cache fallback."""

    DEFAULT_SEASONS = [2024, 2025, 2026]

    def __init__(self, max_retries: int = 3, retry_delay: float = 0.1, backoff_factor: float = 2.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.backoff_factor = backoff_factor

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

    def _load_fallback_csv(self, filename: str) -> pd.DataFrame:
        """Load fallback CSV or parquet from data/raw/ directory."""
        raw_dir = self._resolve_raw_dir()
        path = os.path.join(raw_dir, filename)
        if os.path.exists(path):
            try:
                if filename.endswith('.parquet'):
                    df = pd.read_parquet(path)
                else:
                    df = pd.read_csv(path)
                logger.warning(
                    f"Live request failed; activating fallback data from {path} ({len(df)} rows)"
                )
                return df
            except Exception as e:
                logger.error(f"Failed to read fallback file {path}: {e}")
        return pd.DataFrame()

    def _retry_call(self, func, *args, **kwargs):
        """Execute a callable with retry logic and exponential backoff."""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries} failed for SDV call: {e}"
                )
                if attempt < self.max_retries - 1:
                    sleep_time = self.retry_delay * (self.backoff_factor ** attempt)
                    time.sleep(sleep_time)
        logger.warning(f"All {self.max_retries} attempts failed for SDV call: {last_error}")
        return None

    def _to_pandas_df(self, df: Any) -> Optional[pd.DataFrame]:
        """Convert input (Polars or Pandas) to a valid non-empty pandas.DataFrame."""
        if df is None:
            return None
        if hasattr(df, 'to_pandas'):
            try:
                df = df.to_pandas()
            except Exception as e:
                logger.warning(f"Failed to convert object to pandas: {e}")
                return None
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df
        return None

    def get_team_box_scores(self, seasons: List[int] = None) -> pd.DataFrame:
        """Load team-level box scores for specified seasons.
        Contains per-game team stats: points, rebounds, assists, FG%, etc.
        """
        seasons = seasons or self.DEFAULT_SEASONS
        logger.info(f"Loading team box scores for seasons: {seasons}")
        def _fetch():
            try:
                from sportsdataverse.nba import load_nba_team_boxscore as loader
            except ImportError:
                from sportsdataverse.nba import load_nba_team_box as loader
            return loader(seasons=seasons)

        try:
            raw_df = self._retry_call(_fetch)
            df = self._to_pandas_df(raw_df)
            if df is not None:
                logger.info(f"Loaded {len(df)} team box score records")
                return df
        except Exception as e:
            logger.warning(f"Unexpected error loading team box scores: {e}")

        # Fallback to sdv_team_box.csv or nba_api_game_logs.csv
        cached = self._load_fallback_csv('sdv_team_box.csv')
        if cached.empty:
            cached = self._load_fallback_csv('nba_api_game_logs.csv')
        return cached

    def get_player_box_scores(self, seasons: List[int] = None) -> pd.DataFrame:
        """Load player-level box scores for specified seasons.
        Contains per-game player stats: points, rebounds, assists, minutes, etc.
        """
        seasons = seasons or self.DEFAULT_SEASONS
        logger.info(f"Loading player box scores for seasons: {seasons}")
        def _fetch():
            try:
                from sportsdataverse.nba import load_nba_player_boxscore as loader
            except ImportError:
                from sportsdataverse.nba import load_nba_player_box as loader
            return loader(seasons=seasons)

        try:
            raw_df = self._retry_call(_fetch)
            df = self._to_pandas_df(raw_df)
            if df is not None:
                logger.info(f"Loaded {len(df)} player box score records")
                return df
        except Exception as e:
            logger.warning(f"Unexpected error loading player box scores: {e}")

        # Fallback to sdv_player_box.csv or nba_api_players.csv
        cached = self._load_fallback_csv('sdv_player_box.csv')
        if cached.empty:
            cached = self._load_fallback_csv('nba_api_players.csv')
        return cached

    def get_schedules(self, seasons: List[int] = None) -> pd.DataFrame:
        """Load schedules with game results for specified seasons.
        Contains: game_id, date, teams, scores, venue, attendance.
        """
        seasons = seasons or self.DEFAULT_SEASONS
        logger.info(f"Loading schedules for seasons: {seasons}")
        def _fetch():
            from sportsdataverse.nba import load_nba_schedule
            return load_nba_schedule(seasons=seasons)

        try:
            raw_df = self._retry_call(_fetch)
            df = self._to_pandas_df(raw_df)
            if df is not None:
                logger.info(f"Loaded {len(df)} schedule records")
                return df
        except Exception as e:
            logger.warning(f"Unexpected error loading schedules: {e}")

        # Fallback to sdv_schedule.csv or espn_recent_games.csv
        cached = self._load_fallback_csv('sdv_schedule.csv')
        if cached.empty:
            cached = self._load_fallback_csv('espn_recent_games.csv')
        return cached

    def get_play_by_play(self, seasons: List[int] = None) -> pd.DataFrame:
        """Load full play-by-play data for specified seasons."""
        seasons = seasons or [2026]
        logger.info(f"Loading play-by-play for seasons: {seasons}")
        def _fetch():
            from sportsdataverse.nba import load_nba_pbp
            return load_nba_pbp(seasons=seasons)

        try:
            raw_df = self._retry_call(_fetch)
            df = self._to_pandas_df(raw_df)
            if df is not None:
                logger.info(f"Loaded {len(df)} play-by-play records")
                return df
        except Exception as e:
            logger.warning(f"Unexpected error loading play-by-play: {e}")

        # Fallback to sdv_play_by_play.csv or nba_api_game_logs.csv
        cached = self._load_fallback_csv('sdv_play_by_play.csv')
        if cached.empty:
            cached = self._load_fallback_csv('nba_api_game_logs.csv')
        return cached

    def get_team_season_averages(self, seasons: List[int] = None) -> pd.DataFrame:
        """Compute team season averages from box scores.
        Returns per-game averages: PPG, RPG, APG, FG%, 3P%, FT%, TOV, etc.
        """
        df = self.get_team_box_scores(seasons)
        if df.empty:
            return self._load_fallback_csv('nba_api_base.csv')

        logger.info("Computing team season averages from box scores")

        group_cols = []
        for col in ['season', 'season_type', 'team_display_name', 'team_name', 'team_short_display_name', 'TEAM_NAME', 'TEAM_ABBREVIATION']:
            if col in df.columns:
                group_cols.append(col)

        numeric_cols = [c for c in df.select_dtypes(include='number').columns.tolist() if c not in group_cols]

        if not group_cols or not numeric_cols:
            logger.warning("Could not identify grouping/numeric columns; falling back to base stats.")
            cached = self._load_fallback_csv('nba_api_base.csv')
            return cached if not cached.empty else df

        try:
            avg_df = df.groupby(group_cols)[numeric_cols].mean().reset_index()
            if not avg_df.empty:
                return avg_df
        except Exception as e:
            logger.warning(f"Error computing season averages: {e}; falling back to base stats.")

        cached = self._load_fallback_csv('nba_api_base.csv')
        return cached if not cached.empty else df

    def save_raw_data(self, output_dir: str = None):
        """Download and save all raw data to disk as parquet files."""
        if output_dir is None:
            output_dir = self._resolve_raw_dir()
        os.makedirs(output_dir, exist_ok=True)

        datasets = {
            'team_box': self.get_team_box_scores,
            'player_box': self.get_player_box_scores,
            'schedule': self.get_schedules,
        }

        for name, func in datasets.items():
            logger.info(f"Saving {name}...")
            try:
                df = func()
                if df is not None and not df.empty:
                    path = os.path.join(output_dir, f'sdv_{name}.parquet')
                    df.to_parquet(path, index=False)
                    logger.info(f"Saved {name} to {path} ({len(df)} rows)")
            except Exception as e:
                logger.error(f"Failed to save {name}: {e}")
