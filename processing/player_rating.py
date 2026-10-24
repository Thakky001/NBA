import pandas as pd
import logging
import os
import time
import unicodedata
from typing import Dict, Any

from collectors.espn_collector import ESPNCollector
from collectors.advanced_stats_collector import AdvancedStatsCollector
from collectors.rotowire_collector import RotoWireCollector
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
    
    def __init__(self, season='2025-26', rotowire_collector=None):
        self.season = season
        self.espn = ESPNCollector()
        self.rotowire = rotowire_collector or RotoWireCollector()
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
            'full_roster_rating': 0.0,
            'off_loss': 0.0,
            'def_loss': 0.0,
            'full_off_rating': 0.0,
            'full_def_rating': 0.0,
            'active_off_rating': 0.0,
            'active_def_rating': 0.0,
            'top_players': [],
            'starters': [],
            'lineup_status': 'Unconfirmed',
            'is_lineup_confirmed': False,
            'injuries': [],
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

                    # Calculate separate offensive and defensive efficiency if detailed stats present
                    pts = float(row.get('PTS', 0) or 0) / gp if gp > 0 else 0.0
                    ast = float(row.get('AST', 0) or 0) / gp if gp > 0 else 0.0
                    oreb = float(row.get('OREB', 0) or 0) / gp if gp > 0 else 0.0
                    stl = float(row.get('STL', 0) or 0) / gp if gp > 0 else 0.0
                    blk = float(row.get('BLK', 0) or 0) / gp if gp > 0 else 0.0
                    dreb = float(row.get('DREB', 0) or 0) / gp if gp > 0 else 0.0
                    
                    if pts > 0 or ast > 0:
                        off_eff = round(pts + 1.5 * ast + oreb, 1)
                        def_eff = round(stl * 2.0 + blk * 2.0 + dreb, 1)
                    else:
                        off_eff = round(eff * 0.65, 1)
                        def_eff = round(eff * 0.35, 1)

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
                            'off_eff': off_eff,
                            'def_eff': def_eff,
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
                            'off_eff': round(off_eff * gp / 5.0 + (base_eff * 0.65) * (5 - gp) / 5.0, 1),
                            'def_eff': round(def_eff * gp / 5.0 + (base_eff * 0.35) * (5 - gp) / 5.0, 1),
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
                    'off_eff': round(eff * 0.65, 1),
                    'def_eff': round(eff * 0.35, 1),
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
            p_off = p.get('off_eff', round(p['eff_pg'] * 0.65, 1))
            p_def = p.get('def_eff', round(p['eff_pg'] * 0.35, 1))
            inj_info = injury_map.get(p_norm) or injury_map.get(normalize_name(p['name']))
            if inj_info:
                st_lower = inj_info['status'].lower()
                detail_str = f" - {inj_info['detail']}" if inj_info['detail'] else ""
                if any(k in st_lower for k in ['out', 'ofs', 'injured reserve', 'season']):
                    p['active_eff'] = 0.0
                    p['active_off_eff'] = 0.0
                    p['active_def_eff'] = 0.0
                    p['injury_status'] = f"OUT{detail_str}"
                elif 'doubtful' in st_lower:
                    p['active_eff'] = p['eff_pg'] * 0.25
                    p['active_off_eff'] = round(p_off * 0.25, 1)
                    p['active_def_eff'] = round(p_def * 0.25, 1)
                    p['injury_status'] = f"DOUBTFUL{detail_str}"
                elif any(k in st_lower for k in ['questionable', 'day-to-day', 'gtd']):
                    p['active_eff'] = p['eff_pg'] * 0.60
                    p['active_off_eff'] = round(p_off * 0.60, 1)
                    p['active_def_eff'] = round(p_def * 0.60, 1)
                    p['injury_status'] = f"GTD/QST{detail_str}"
                else:
                    p['active_eff'] = p['eff_pg']
                    p['active_off_eff'] = p_off
                    p['active_def_eff'] = p_def
                    p['injury_status'] = None
            else:
                p['active_eff'] = p['eff_pg']
                p['active_off_eff'] = p_off
                p['active_def_eff'] = p_def
                p['injury_status'] = None

        # Check RotoWire starting 5
        rw_lineup = None
        if hasattr(self, 'rotowire') and self.rotowire:
            try:
                rw_lineup = self.rotowire.get_lineup_for_team(team_name)
            except Exception as e:
                logger.warning(f"Error fetching RotoWire lineup for {team_name}: {e}")

        # Map candidate players by normalized name
        candidates_map = {}
        for p in matched_ratings:
            p_clean = p.get('clean_name', p['name'])
            candidates_map[normalize_name(p_clean)] = p

        starter_players = []
        starter_ids = set()
        lineup_status = "Unconfirmed"
        is_confirmed = False

        if rw_lineup and rw_lineup.get('starters'):
            lineup_status = rw_lineup.get('lineup_status', 'Expected')
            is_confirmed = rw_lineup.get('is_confirmed', False)
            for s_info in rw_lineup.get('starters', []):
                s_name = s_info.get('name', '')
                s_pos = s_info.get('position', '')
                matched_player = None
                norm_s = normalize_name(s_name)
                if norm_s in candidates_map:
                    matched_player = candidates_map[norm_s]
                else:
                    # Try matching first initial + last name (e.g. "D. Mitchell" -> "Davion Mitchell")
                    s_parts = norm_s.replace('.', ' ').split()
                    if len(s_parts) >= 2:
                        s_init = s_parts[0][0]
                        s_last = s_parts[-1]
                        for c_norm, c_obj in candidates_map.items():
                            c_parts = c_norm.split()
                            if len(c_parts) >= 2 and c_parts[-1] == s_last and c_parts[0][0] == s_init:
                                matched_player = c_obj
                                break

                if matched_player and id(matched_player) not in starter_ids:
                    sp_copy = dict(matched_player)
                    sp_copy['position'] = s_pos
                    starter_players.append(sp_copy)
                    starter_ids.add(id(matched_player))

        # Bench players = remaining players not in starters sorted by active_eff
        bench_candidates = [p for p in matched_ratings if id(p) not in starter_ids]
        bench_top_3 = sorted(bench_candidates, key=lambda x: x['active_eff'], reverse=True)[:3]
        bench_full_top_3 = sorted(bench_candidates, key=lambda x: x['eff_pg'], reverse=True)[:3]

        if not matched_ratings:
            return result

        # Use 70/30 weighting if we matched at least 4 starters and at least 1 bench
        if len(starter_players) >= 4 and len(bench_top_3) >= 1:
            s_n = len(starter_players)
            b_n = len(bench_top_3)
            fb_n = len(bench_full_top_3) if bench_full_top_3 else b_n

            # Full health power (70% starters, 30% bench, scaled to 8-man level)
            s_eff_avg = sum(s['eff_pg'] for s in starter_players) / s_n
            b_eff_avg = sum(b['eff_pg'] for b in (bench_full_top_3 or bench_top_3)) / fb_n
            full_roster_rating = 8.0 * (0.70 * s_eff_avg + 0.30 * b_eff_avg)

            s_off_avg = sum(s.get('off_eff', s['eff_pg'] * 0.65) for s in starter_players) / s_n
            b_off_avg = sum(b.get('off_eff', b['eff_pg'] * 0.65) for b in (bench_full_top_3 or bench_top_3)) / fb_n
            full_off_rating = 8.0 * (0.70 * s_off_avg + 0.30 * b_off_avg)

            s_def_avg = sum(s.get('def_eff', s['eff_pg'] * 0.35) for s in starter_players) / s_n
            b_def_avg = sum(b.get('def_eff', b['eff_pg'] * 0.35) for b in (bench_full_top_3 or bench_top_3)) / fb_n
            full_def_rating = 8.0 * (0.70 * s_def_avg + 0.30 * b_def_avg)

            # Active power (70% starters, 30% bench, scaled to 8-man level)
            s_act_eff_avg = sum(s['active_eff'] for s in starter_players) / s_n
            b_act_eff_avg = sum(b['active_eff'] for b in bench_top_3) / b_n
            active_roster_rating = 8.0 * (0.70 * s_act_eff_avg + 0.30 * b_act_eff_avg)

            s_act_off_avg = sum(s.get('active_off_eff', s['active_eff'] * 0.65) for s in starter_players) / s_n
            b_act_off_avg = sum(b.get('active_off_eff', b['active_eff'] * 0.65) for b in bench_top_3) / b_n
            active_off_rating = 8.0 * (0.70 * s_act_off_avg + 0.30 * b_act_off_avg)

            s_act_def_avg = sum(s.get('active_def_eff', s['active_eff'] * 0.35) for s in starter_players) / s_n
            b_act_def_avg = sum(b.get('active_def_eff', b['active_eff'] * 0.35) for b in bench_top_3) / b_n
            active_def_rating = 8.0 * (0.70 * s_act_def_avg + 0.30 * b_act_def_avg)

            rotation_display = starter_players + bench_top_3
        else:
            # Fallback to legacy Top 8 calculation
            full_top_8 = sorted(matched_ratings, key=lambda x: x['eff_pg'], reverse=True)[:8]
            if not full_top_8:
                return result
            full_roster_rating = sum(p['eff_pg'] for p in full_top_8)
            full_off_rating = sum(p.get('off_eff', p['eff_pg'] * 0.65) for p in full_top_8)
            full_def_rating = sum(p.get('def_eff', p['eff_pg'] * 0.35) for p in full_top_8)

            active_top_8 = sorted(matched_ratings, key=lambda x: x['active_eff'], reverse=True)[:8]
            active_roster_rating = sum(p['active_eff'] for p in active_top_8)
            active_off_rating = sum(p.get('active_off_eff', p['active_eff'] * 0.65) for p in active_top_8)
            active_def_rating = sum(p.get('active_def_eff', p['active_eff'] * 0.35) for p in active_top_8)
            rotation_display = active_top_8
            if not starter_players:
                starter_players = active_top_8[:5]

        off_loss = max(0.0, full_off_rating - active_off_rating)
        def_loss = max(0.0, full_def_rating - active_def_rating)

        # Injuries list for this team
        team_injuries = [
            f"{p['name']} [{p['injury_status']}]"
            for p in matched_ratings if p.get('injury_status')
        ]

        result['roster_rating'] = float(round(active_roster_rating, 2))
        result['full_roster_rating'] = float(round(full_roster_rating, 2))
        result['off_loss'] = float(round(off_loss, 2))
        result['def_loss'] = float(round(def_loss, 2))
        result['full_off_rating'] = float(round(full_off_rating, 2))
        result['full_def_rating'] = float(round(full_def_rating, 2))
        result['active_off_rating'] = float(round(active_off_rating, 2))
        result['active_def_rating'] = float(round(active_def_rating, 2))
        result['injuries'] = team_injuries
        result['starters'] = [
            {
                'name': p.get('clean_name', p['name']),
                'position': p.get('position', ''),
                'eff_pg': round(float(p.get('eff_pg', 0.0)), 1),
                'active_eff': round(float(p.get('active_eff', 0.0)), 1),
                'injury_status': p.get('injury_status')
            }
            for p in starter_players
        ]
        result['lineup_status'] = lineup_status
        result['is_lineup_confirmed'] = is_confirmed
        result['top_players'] = [
            f"{p.get('position', '') + ': ' if p.get('position') else ''}{p['name']} ({p['active_eff']:.1f}" + (f" - {p['injury_status']})" if p.get('injury_status') else ")")
            for p in rotation_display[:8]
        ]
        result['status'] = 'success'

        return result

