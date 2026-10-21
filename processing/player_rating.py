import pandas as pd
import logging
import os
import time
import unicodedata
from typing import Dict, Any

from collectors.espn_collector import ESPNCollector
from collectors.advanced_stats_collector import AdvancedStatsCollector
from config import DATA_DIR

from functools import lru_cache

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_cached_draft_history() -> pd.DataFrame:
    """Load draft history from cache once with in-memory caching."""
    for fname in ['draft_history.csv', 'nba_api_draft_history.csv']:
        cache_path = os.path.join(DATA_DIR, 'raw', fname)
        if os.path.exists(cache_path):
            try:
                df = pd.read_csv(cache_path)
                if not df.empty:
                    return df
            except Exception:
                pass
    return pd.DataFrame()


class PlayerRatingEngine:
    """Evaluates team strength based on individual player ratings (EFF/PIE)."""
    
    def __init__(self, season='2025-26'):
        self.season = season
        self.espn = ESPNCollector()
        self._teams_cache = None
        self._roster_cache = {}
        self.player_stats = self._load_player_stats()
        self.draft_history = self._load_draft_history()
        self.injuries_df = self._load_injuries()

    def _load_injuries(self) -> pd.DataFrame:
        """Load injury data from ESPN collector or fallback cache."""
        try:
            return self.espn.get_injuries()
        except Exception as e:
            logger.warning(f"Could not load injuries: {e}")
            return pd.DataFrame()

    def _load_draft_history(self) -> pd.DataFrame:
        """Load draft history for evaluating rookies (in-memory cached)."""
        return _load_cached_draft_history()

    def _load_player_stats(self) -> pd.DataFrame:
        """Load player stats from cache or fetch from nba_api.
        Auto-refreshes if cache is older than 7 days."""
        # Reuse the API dump from advanced_stats_collector
        cache_path = os.path.join(DATA_DIR, 'raw', 'nba_api_players.csv')
        max_age_days = 7
        
        if os.path.exists(cache_path):
            try:
                file_age_days = (time.time() - os.path.getmtime(cache_path)) / 86400
                
                if file_age_days < max_age_days:
                    df = pd.read_csv(cache_path)
                    if not df.empty:
                        logger.info(f"Loading cached player ratings (age: {file_age_days:.1f} days)")
                        if 'EFF_PG' not in df.columns and 'PTS' in df.columns:
                            df['MISSED_FG'] = df['FGA'] - df['FGM']
                            df['MISSED_FT'] = df['FTA'] - df['FTM']
                            df['EFF_TOTAL'] = df['PTS'] + df['REB'] + df['AST'] + df['STL'] + df['BLK'] - df['MISSED_FG'] - df['MISSED_FT'] - df['TOV']
                            df['EFF_PG'] = (df['EFF_TOTAL'] / df['GP']).fillna(0.0)
                            try:
                                df.to_csv(cache_path, index=False)
                                logger.info(f"Persisted calculated EFF_PG to {cache_path}")
                            except Exception as ce:
                                logger.warning(f"Could not save EFF_PG to cache: {ce}")
                        return df
                else:
                    logger.info(f"Player ratings cache is {file_age_days:.1f} days old (max {max_age_days}d). Refreshing...")
            except Exception as e:
                logger.error(f"Error reading player ratings cache: {e}")
        
        logger.info("Fetching base player stats to calculate ratings...")
        try:
            from nba_api.stats.endpoints import leaguedashplayerstats
            # Base stats to calculate EFF
            result = leaguedashplayerstats.LeagueDashPlayerStats(
                league_id_nullable='00',
                season=self.season,
                season_type_all_star='Regular Season'
            )
            df = result.get_data_frames()[0]
            
            # Calculate EFF (Efficiency) per game
            # EFF = (PTS + REB + AST + STL + BLK - Missed FG - Missed FT - TO) / GP
            if not df.empty:
                df['MISSED_FG'] = df['FGA'] - df['FGM']
                df['MISSED_FT'] = df['FTA'] - df['FTM']
                df['EFF_TOTAL'] = df['PTS'] + df['REB'] + df['AST'] + df['STL'] + df['BLK'] - df['MISSED_FG'] - df['MISSED_FT'] - df['TOV']
                df['EFF_PG'] = df['EFF_TOTAL'] / df['GP']
                df['EFF_PG'] = df['EFF_PG'].fillna(0.0)
                
                # Save cache
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                df.to_csv(cache_path, index=False)
                
            return df
        except Exception as e:
            logger.error(f"Failed to fetch player stats for ratings: {e}")
            if os.path.exists(cache_path):
                try:
                    df = pd.read_csv(cache_path)
                    if not df.empty:
                        logger.warning(f"Falling back to existing player ratings cache: {cache_path}")
                        if 'EFF_PG' not in df.columns and 'PTS' in df.columns:
                            df['MISSED_FG'] = df['FGA'] - df['FGM']
                            df['MISSED_FT'] = df['FTA'] - df['FTM']
                            df['EFF_TOTAL'] = df['PTS'] + df['REB'] + df['AST'] + df['STL'] + df['BLK'] - df['MISSED_FG'] - df['MISSED_FT'] - df['TOV']
                            df['EFF_PG'] = (df['EFF_TOTAL'] / df['GP']).fillna(0.0)
                        return df
                except Exception:
                    pass
            return pd.DataFrame()

    def calculate_team_roster_rating(self, team_name: str) -> Dict[str, Any]:
        """
        Calculates a team's power rating by aggregating the EFF_PG of its top 8 
        active players based on current ESPN roster.
        """
        # Default rating for fallback
        result = {
            'team_name': team_name,
            'roster_rating': 0.0,
            'top_players': [],
            'status': 'fallback'
        }
        
        stats_df = self.player_stats if self.player_stats is not None else pd.DataFrame()
            
        # 1. Fetch team ID from ESPN
        team_id = None
        try:
            from config import resolve_team_info
            info = resolve_team_info(team_name)
            if info and 'espn_id' in info:
                team_id = info['espn_id']
        except Exception:
            pass

        if team_id is None:
            if self._teams_cache is None or self._teams_cache.empty:
                self._teams_cache = self.espn.get_all_teams()
            teams_df = self._teams_cache
            
            if teams_df.empty:
                return result
                
            match = teams_df[
                teams_df['display_name'].str.contains(team_name, case=False, na=False) |
                teams_df['display_name'].apply(lambda x: str(x).lower() in str(team_name).lower()) |
                teams_df['name'].apply(lambda x: str(x).lower() in str(team_name).lower())
            ]
            if match.empty and 'abbreviation' in teams_df.columns:
                match = teams_df[teams_df['abbreviation'].str.upper() == str(team_name).upper()]
                
            if match.empty:
                return result
                
            team_id = match.iloc[0]['team_id']
        
        # 2. Fetch active roster from ESPN
        if team_id not in self._roster_cache:
            self._roster_cache[team_id] = self.espn.get_team_roster(int(team_id))
        roster_df = self._roster_cache[team_id]
        
        if roster_df.empty:
            return result
            
        # 3. Match players and get their EFF_PG
        # Normalize names: strip diacritics (e.g. Dončić -> Doncic) for cross-source matching
        def normalize_name(name):
            """Strip unicode diacritics, optional tags, and lowercase for matching."""
            nfkd = unicodedata.normalize('NFKD', str(name))
            clean = nfkd.encode('ascii', 'ignore').decode('ascii').lower().strip()
            if clean.endswith('(base)'):
                clean = clean[:-6].strip()
            return clean
        
        active_player_map = {normalize_name(n): n for n in roster_df['name'].tolist()}
        
        # Create draft map for fast lookup: normalize_name -> overall_pick
        draft_map = {}
        if not self.draft_history.empty:
            for _, row in self.draft_history.iterrows():
                p_name = str(row.get('PLAYER_NAME', ''))
                pick = row.get('OVERALL_PICK', 999)
                if pd.notna(pick):
                    draft_map[normalize_name(p_name)] = int(pick)

        matched_ratings = []
        
        # 1. Add players who have stats in the current season
        found_players = set()
        if not stats_df.empty:
            for _, row in stats_df.iterrows():
                p_name_raw = str(row.get('PLAYER_NAME', ''))
                p_name_norm = normalize_name(p_name_raw)
                if p_name_norm in active_player_map:
                    try:
                        eff = float(row.get('EFF_PG', 0) or 0)
                    except (ValueError, TypeError):
                        eff = 0.0
                    try:
                        gp = int(row.get('GP', 0) or 0)
                    except (ValueError, TypeError):
                        gp = 0

                    # Base tier lookup for small-sample blend or baseline
                    pick = draft_map.get(p_name_norm, 999)
                    if pick <= 5:
                        base_eff = 14.0 # Top 5 pick (Solid starter)
                    elif pick <= 14:
                        base_eff = 9.0  # Lottery (Key rotation)
                    elif pick <= 30:
                        base_eff = 6.0  # First round (Rotation contributor)
                    else:
                        base_eff = 4.5  # General rotation / undrafted baseline
                    
                    # If they have enough games, use their actual EFF
                    if gp >= 5:
                        matched_ratings.append({
                            'name': active_player_map[p_name_norm],
                            'clean_name': active_player_map[p_name_norm],
                            'eff_pg': eff,
                            'gp': gp
                        })
                        found_players.add(p_name_norm)
                    elif gp > 0 and eff > 0:
                        # Small-sample blending: blend actual EFF with base rating
                        blended_eff = (eff * gp + base_eff * (5 - gp)) / 5.0
                        matched_ratings.append({
                            'name': active_player_map[p_name_norm],
                            'clean_name': active_player_map[p_name_norm],
                            'eff_pg': round(blended_eff, 1),
                            'gp': gp
                        })
                        found_players.add(p_name_norm)
                    
        # 2. Add Rookies & Cold Start players (GP < 5 or not in stats yet)
        for p_name_norm, original_name in active_player_map.items():
            if p_name_norm not in found_players:
                # Check if they are a recent draft pick
                pick = draft_map.get(p_name_norm, 999)
                
                # Assign Replacement Level / Rookie Base Rating (elevated floor)
                if pick <= 5:
                    eff = 14.0 # Top 5 pick (Solid starter)
                elif pick <= 14:
                    eff = 9.0  # Lottery (Key rotation)
                elif pick <= 30:
                    eff = 6.0  # First round (Rotation contributor)
                else:
                    eff = 4.5  # General rotation / undrafted baseline (elevated from 3.0)
                    
                matched_ratings.append({
                    'name': f"{original_name} (Base)",
                    'clean_name': original_name,
                    'eff_pg': eff,
                    'gp': 0
                })
        # 3. Match Injuries for this team
        injury_map = {}
        if self.injuries_df is not None and not self.injuries_df.empty:
            for _, inj_row in self.injuries_df.iterrows():
                p_name = str(inj_row.get('player_name', ''))
                if p_name:
                    injury_map[normalize_name(p_name)] = {
                        'player_name': p_name,
                        'status': str(inj_row.get('status', '')),
                        'detail': str(inj_row.get('detail', ''))
                    }

        # Apply injury status to player ratings
        for p in matched_ratings:
            p_clean = p.get('clean_name', p['name'])
            p_norm = normalize_name(p_clean)
            inj_info = injury_map.get(p_norm) or injury_map.get(normalize_name(p['name']))
            if inj_info:
                st_lower = inj_info['status'].lower()
                detail_str = f" - {inj_info['detail']}" if inj_info['detail'] else ""
                if any(k in st_lower for k in ['out', 'ofs', 'injured reserve', 'season']):
                    p['active_eff'] = 0.0
                    p['injury_status'] = f"OUT{detail_str}"
                elif 'doubtful' in st_lower:
                    p['active_eff'] = p['eff_pg'] * 0.25
                    p['injury_status'] = f"DOUBTFUL{detail_str}"
                elif any(k in st_lower for k in ['questionable', 'day-to-day', 'gtd']):
                    p['active_eff'] = p['eff_pg'] * 0.60
                    p['injury_status'] = f"GTD/QST{detail_str}"
                else:
                    p['active_eff'] = p['eff_pg']
                    p['injury_status'] = None
            else:
                p['active_eff'] = p['eff_pg']
                p['injury_status'] = None

        # Full health power: Top 8 sorted by original EFF_PG
        full_top_8 = sorted(matched_ratings, key=lambda x: x['eff_pg'], reverse=True)[:8]
        if not full_top_8:
            return result
        full_roster_rating = sum(p['eff_pg'] for p in full_top_8)

        # Active power: Top 8 sorted by active EFF_PG (injured players drop down, bench steps up)
        active_top_8 = sorted(matched_ratings, key=lambda x: x['active_eff'], reverse=True)[:8]
        active_roster_rating = sum(p['active_eff'] for p in active_top_8)

        # Injuries list for this team
        team_injuries = [
            f"{p['name']} [{p['injury_status']}]"
            for p in matched_ratings if p.get('injury_status')
        ]

        result['roster_rating'] = float(active_roster_rating)
        result['full_roster_rating'] = float(full_roster_rating)
        result['injuries'] = team_injuries
        result['top_players'] = [
            f"{p['name']} ({p['active_eff']:.1f}" + (f" - {p['injury_status']})" if p.get('injury_status') else ")")
            for p in active_top_8
        ]
        result['status'] = 'success'

        return result

