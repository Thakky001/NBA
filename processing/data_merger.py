import pandas as pd
import numpy as np
import logging
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

class DataMerger:
    """Merges data from multiple sources (ESPN, SDV, nba_api) into unified DataFrames."""
    
    def __init__(self):
        # Initialized for backward compatibility
        self.team_mapping = {}
    
    def merge_game_data(self, espn_games: pd.DataFrame, sdv_games: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Merge ESPN scoreboard data with SDV historical data.
        Standardizes column names and combines sources.
        Priority: ESPN for recent data + odds, SDV for historical depth.
        """
        logger.info("Merging game data from available sources...")
        
        def standardize_game_df(df: pd.DataFrame, source: str) -> pd.DataFrame:
            if df is None or df.empty:
                return pd.DataFrame()
                
            df_std = df.copy()

            # Handle team-level box score (e.g. sdv_team_box: 2 rows per game with MATCHUP)
            matchup_col = None
            for col in df_std.columns:
                if str(col).lower() in ['matchup']:
                    matchup_col = col
                    break

            if matchup_col and 'home_team' not in [str(c).lower() for c in df_std.columns]:
                # Find game_id column
                gid_col = next((c for c in df_std.columns if str(c).lower() in ['game_id', 'gameid']), None)
                if gid_col:
                    h_rows = df_std[df_std[matchup_col].astype(str).str.contains('vs.', case=False, na=False)].copy()
                    a_rows = df_std[df_std[matchup_col].astype(str).str.contains('@', case=False, na=False)].copy()
                    if not h_rows.empty and not a_rows.empty:
                        m_box = pd.merge(h_rows, a_rows, on=gid_col, suffixes=('_h', '_a'))
                        # Extract paired fields
                        date_col_h = next((c for c in m_box.columns if 'game_date' in str(c).lower() or 'date' in str(c).lower()), None)
                        team_col_h = next((c for c in m_box.columns if 'team_name_h' in str(c).lower() or 'team_h' in str(c).lower()), None)
                        team_col_a = next((c for c in m_box.columns if 'team_name_a' in str(c).lower() or 'team_a' in str(c).lower()), None)
                        pts_col_h = next((c for c in m_box.columns if 'pts_h' in str(c).lower()), None)
                        pts_col_a = next((c for c in m_box.columns if 'pts_a' in str(c).lower()), None)
                        
                        df_std = pd.DataFrame({
                            'game_id': m_box[gid_col].astype(str),
                            'date': m_box[date_col_h] if date_col_h else None,
                            'home_team': m_box[team_col_h] if team_col_h else None,
                            'away_team': m_box[team_col_a] if team_col_a else None,
                            'home_score': m_box[pts_col_h] if pts_col_h else None,
                            'away_score': m_box[pts_col_a] if pts_col_a else None,
                        })
            
            # Map columns to standard format
            col_mapping = {}
            for col in df_std.columns:
                lower_col = str(col).lower()
                if lower_col in ['game_id', 'id', 'gameid']: col_mapping[col] = 'game_id'
                elif lower_col in ['date', 'gamedate', 'game_date']: col_mapping[col] = 'date'
                elif lower_col in ['season', 'year']: col_mapping[col] = 'season'
                elif lower_col in ['home_team', 'home', 'hometeam']: col_mapping[col] = 'home_team'
                elif lower_col in ['away_team', 'away', 'visitor', 'awayteam']: col_mapping[col] = 'away_team'
                elif lower_col in ['home_score', 'homescore', 'home_pts']: col_mapping[col] = 'home_score'
                elif lower_col in ['away_score', 'awayscore', 'away_pts']: col_mapping[col] = 'away_score'
                elif lower_col in ['spread', 'line']: col_mapping[col] = 'spread'
                elif lower_col in ['over_under', 'ou', 'total']: col_mapping[col] = 'over_under'
                elif lower_col in ['home_ml', 'homemoneyline']: col_mapping[col] = 'home_ml'
                elif lower_col in ['away_ml', 'awaymoneyline']: col_mapping[col] = 'away_ml'
                elif lower_col in ['venue', 'location', 'arena']: col_mapping[col] = 'venue'
                
            df_std.rename(columns=col_mapping, inplace=True)
            
            required_cols = ['game_id', 'date', 'home_team', 'away_team', 'home_score', 'away_score']
            for col in required_cols:
                if col not in df_std.columns:
                    df_std[col] = None
                    
            # Normalize team names
            if 'home_team' in df_std.columns:
                df_std['home_team'] = df_std['home_team'].apply(self.normalize_team_name)
            if 'away_team' in df_std.columns:
                df_std['away_team'] = df_std['away_team'].apply(self.normalize_team_name)
                
            # Add computed columns if possible
            try:
                df_std['home_score'] = pd.to_numeric(df_std['home_score'])
                df_std['away_score'] = pd.to_numeric(df_std['away_score'])
                df_std['total_points'] = df_std['home_score'] + df_std['away_score']
                df_std['point_diff'] = df_std['home_score'] - df_std['away_score']
            except Exception:
                pass
                
            df_std['source'] = source
            return df_std
            
        std_espn = standardize_game_df(espn_games, 'espn')
        std_sdv = standardize_game_df(sdv_games, 'sdv')
        
        if std_espn.empty and std_sdv.empty:
            return pd.DataFrame()
        elif std_sdv.empty:
            combined = std_espn
        elif std_espn.empty:
            combined = std_sdv
        else:
            combined = pd.concat([std_espn, std_sdv], ignore_index=True)
            
        if 'game_id' in combined.columns:
            mask = combined['game_id'].notna()
            deduped = combined[mask].drop_duplicates(subset=['game_id'], keep='first')
            combined = pd.concat([deduped, combined[~mask]], ignore_index=True)
        else:
            combined.drop_duplicates(subset=['date', 'home_team', 'away_team'], keep='first', inplace=True)
        
        expected_cols = [
            'game_id', 'date', 'season', 'home_team', 'away_team', 'home_score', 
            'away_score', 'total_points', 'point_diff', 'spread', 'over_under',
            'home_ml', 'away_ml', 'venue'
        ]
        
        for col in expected_cols:
            if col not in combined.columns:
                combined[col] = None
                
        return combined[expected_cols]
    
    def merge_team_stats(self, espn_standings: pd.DataFrame, 
                         nba_api_advanced: Optional[pd.DataFrame] = None,
                         nba_api_four_factors: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Merge team standings with advanced stats into one comprehensive team profile."""
        logger.info("Merging team stats from available sources...")
        
        def standardize_team_col(df: pd.DataFrame) -> pd.DataFrame:
            if df is None or df.empty:
                return pd.DataFrame()
            df = df.copy()
            for col in df.columns:
                if str(col).lower() in ['team', 'team_name', 'teamname']:
                    df.rename(columns={col: 'team'}, inplace=True)
                    df['team'] = df['team'].apply(self.normalize_team_name)
                    break
            return df
            
        std_espn = standardize_team_col(espn_standings)
        std_adv = standardize_team_col(nba_api_advanced)
        std_ff = standardize_team_col(nba_api_four_factors)
        
        result = std_espn if not std_espn.empty else pd.DataFrame(columns=['team'])
        
        if not std_adv.empty:
            if 'team' in result.columns and 'team' in std_adv.columns:
                result = pd.merge(result, std_adv, on='team', how='outer', suffixes=('', '_adv'))
        
        if not std_ff.empty:
            if 'team' in result.columns and 'team' in std_ff.columns:
                result = pd.merge(result, std_ff, on='team', how='outer', suffixes=('', '_ff'))
                
        # Remap standard columns if they have different names in the source
        col_mapping = {}
        for col in result.columns:
            lower_col = str(col).lower()
            if lower_col in ['w', 'win', 'wins']: col_mapping[col] = 'wins'
            elif lower_col in ['l', 'loss', 'losses']: col_mapping[col] = 'losses'
            elif lower_col in ['pct', 'win_pct', 'winpct']: col_mapping[col] = 'win_pct'
            elif lower_col in ['pace', 'e_pace']: col_mapping[col] = 'pace'
            elif lower_col in ['off_rtg', 'off_rating', 'ortg']: col_mapping[col] = 'off_rating'
            elif lower_col in ['def_rtg', 'def_rating', 'drtg']: col_mapping[col] = 'def_rating'
            elif lower_col in ['net_rtg', 'net_rating', 'nrtg']: col_mapping[col] = 'net_rating'
            elif lower_col in ['efg_pct', 'efg']: col_mapping[col] = 'efg_pct'
            elif lower_col in ['ts_pct', 'ts']: col_mapping[col] = 'ts_pct'
            elif lower_col in ['tov_pct', 'tm_tov_pct']: col_mapping[col] = 'tov_pct'
            elif lower_col in ['orb_pct', 'oreb_pct']: col_mapping[col] = 'orb_pct'
            elif lower_col in ['ft_rate', 'ftr', 'fta_rate']: col_mapping[col] = 'ft_rate'
            elif lower_col in ['pts_for', 'points_for', 'pf']: col_mapping[col] = 'points_for'
            elif lower_col in ['pts_against', 'points_against', 'pa']: col_mapping[col] = 'points_against'
            
        result.rename(columns=col_mapping, inplace=True)
        result = result.loc[:, ~result.columns.duplicated()]
        
        expected_cols = [
            'team', 'conference', 'wins', 'losses', 'win_pct', 'pace', 
            'off_rating', 'def_rating', 'net_rating', 'efg_pct', 'ts_pct',
            'tov_pct', 'orb_pct', 'ft_rate', 'points_for', 'points_against'
        ]
        
        for col in expected_cols:
            if col not in result.columns:
                result[col] = None
                
        # Fill NA for duplicate/overlapping columns if needed, but for simplicity returning standard schema
        return result[expected_cols].drop_duplicates(subset=['team'])
    
    def normalize_team_name(self, name: str) -> str:
        """Normalize team names across different sources."""
        if pd.isna(name):
            return np.nan
        from config import normalize_team_name as central_normalize
        return central_normalize(name)

    def auto_sync_recent_games(self) -> None:
        """Automatically fetch and append finished games from recent days to espn_recent_games.csv."""
        import os
        from datetime import datetime, timedelta
        from config import RAW_DIR
        
        espn_path = os.path.join(RAW_DIR, 'espn_recent_games.csv')
        
        # Check staleness (if older than 2 hours or missing)
        needs_sync = True
        if os.path.exists(espn_path):
            file_age_hours = (datetime.now().timestamp() - os.path.getmtime(espn_path)) / 3600.0
            if file_age_hours < 2.0:
                needs_sync = False
                
        if not needs_sync:
            return

        try:
            from collectors.espn_collector import ESPNCollector
            collector = ESPNCollector()
            end_d = datetime.now()
            start_d = end_d - timedelta(days=3)
            new_games = collector.get_scoreboard_range(
                start_d.strftime('%Y%m%d'),
                end_d.strftime('%Y%m%d')
            )
            if not new_games.empty:
                # Keep completed games with scores > 0
                finished = new_games[
                    (new_games['status'].astype(str).str.contains('FINAL', case=False, na=False)) &
                    (pd.to_numeric(new_games['home_score'], errors='coerce') > 0)
                ].copy()
                if 'season_type' in finished.columns:
                    finished = finished[finished['season_type'] != 1]
                    
                if not finished.empty:
                    if os.path.exists(espn_path):
                        existing = pd.read_csv(espn_path)
                        combined = pd.concat([existing, finished], ignore_index=True)
                    else:
                        combined = finished
                        
                    if 'game_id' in combined.columns:
                        combined = combined.drop_duplicates(subset=['game_id'], keep='last')
                    combined.to_csv(espn_path, index=False)
                    logger.info(f"Auto-synced {len(finished)} finished games to {espn_path}")
        except Exception as e:
            logger.warning(f"auto_sync_recent_games failed gracefully: {e}")

    def load_all_games(self) -> pd.DataFrame:
        """Load and merge all game records from ESPN recent games and SDV historical games."""
        import os
        from config import RAW_DIR
        
        # Ensure latest finished games are synced before loading
        self.auto_sync_recent_games()
        
        espn_path = os.path.join(RAW_DIR, 'espn_recent_games.csv')
        sdv_path = os.path.join(RAW_DIR, 'sdv_team_box.csv')
        
        espn_df = pd.DataFrame()
        if os.path.exists(espn_path):
            try:
                espn_df = pd.read_csv(espn_path)
            except Exception as e:
                logger.warning(f"Error loading {espn_path}: {e}")
                
        sdv_df = pd.DataFrame()
        if os.path.exists(sdv_path):
            try:
                sdv_df = pd.read_csv(sdv_path)
            except Exception as e:
                logger.warning(f"Error loading {sdv_path}: {e}")
                
        merged = self.merge_game_data(espn_df, sdv_df)
        logger.info(f"DataMerger: Combined {len(espn_df)} ESPN games and {len(sdv_df)} SDV records -> {len(merged)} unified games")
        return merged

