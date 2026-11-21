import re
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


@lru_cache(maxsize=8)
def _load_prev_season_eff(path: str, mtime: float, season: int) -> Dict[str, Dict[str, float]]:
    """Per-game EFF for one completed season, keyed by normalised player name.

    ``season`` follows the ESPN/SDV convention of labelling a season by the
    calendar year it ends in, so 2026 means 2025-26.
    """
    try:
        if path.endswith('.parquet'):
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
    except Exception as exc:
        logger.warning(f"Could not read previous-season box scores from {path}: {exc}")
        return {}
    if df.empty:
        return {}

    required = {
        'athlete_display_name', 'season', 'minutes', 'points', 'rebounds', 'assists',
        'steals', 'blocks', 'field_goals_made', 'field_goals_attempted',
        'free_throws_made', 'free_throws_attempted', 'turnovers',
    }
    if not required.issubset(df.columns):
        return {}

    df = df[(pd.to_numeric(df['season'], errors='coerce') == season)
            & (pd.to_numeric(df['minutes'], errors='coerce') > 0)].copy()
    if df.empty:
        return {}

    if 'season_type' in df.columns:
        regular = df[pd.to_numeric(df['season_type'], errors='coerce') == 2]
        if not regular.empty:
            df = regular

    agg = df.groupby('athlete_display_name').agg(
        gp=('minutes', 'size'),
        pts=('points', 'sum'),
        reb=('rebounds', 'sum'),
        ast=('assists', 'sum'),
        stl=('steals', 'sum'),
        blk=('blocks', 'sum'),
        fgm=('field_goals_made', 'sum'),
        fga=('field_goals_attempted', 'sum'),
        ftm=('free_throws_made', 'sum'),
        fta=('free_throws_attempted', 'sum'),
        tov=('turnovers', 'sum'),
    )
    agg['eff_pg'] = (
        agg['pts'] + agg['reb'] + agg['ast'] + agg['stl'] + agg['blk']
        - (agg['fga'] - agg['fgm']) - (agg['fta'] - agg['ftm']) - agg['tov']
    ) / agg['gp']

    return {
        normalize_name(name): {'eff_pg': float(row['eff_pg']), 'gp': int(row['gp'])}
        for name, row in agg.iterrows()
    }


def normalize_name(name: str) -> str:
    """Strip unicode diacritics, generational suffixes (Jr., Sr., III), punctuation, and lowercase for matching."""
    if not name:
        return ""
    nfkd = unicodedata.normalize('NFKD', str(name))
    clean = nfkd.encode('ascii', 'ignore').decode('ascii').lower().strip()
    if clean.endswith('(base)'):
        clean = clean[:-6].strip()
    clean = re.sub(r'[-_]', ' ', clean)
    clean = re.sub(r'[^\w\s]', '', clean)
    while True:
        stripped = re.sub(r'\s+(?:jr|sr|ii|iii|iv|v)$', '', clean).strip()
        if stripped == clean:
            break
        clean = stripped
    return re.sub(r'\s+', ' ', clean)


class PlayerRatingEngine:
    """Evaluates team strength based on individual player ratings (EFF/PIE)."""

    #: A cached injury snapshot is only trustworthy for this long. The web app
    #: keeps one engine alive for the whole process, so a load-once snapshot can
    #: otherwise survive for days and drive lineups from a stale report.
    INJURY_REFRESH_TTL_SECONDS = 600

    #: Missing key players moves an NBA spread by at most ~4 points for a single
    #: star (~22 rating points). But when multiple starters/key rotation players
    #: are out simultaneously (e.g. 3-5 starters sitting out), a rigid 22.0 cap
    #: severely overvalues depleted bench lineups. The ceiling scales dynamically
    #: with the number of missing core players up to MAX_MULTI_INJURY_LOSS.
    MAX_INJURY_LOSS = 22.0
    MAX_MULTI_INJURY_LOSS = 48.0

    def __init__(self, season=None, rotowire_collector=None):
        from config import NBA_API_SEASON
        self.season = season or NBA_API_SEASON
        self.espn = ESPNCollector()
        self.rotowire = rotowire_collector or RotoWireCollector()
        self._teams_cache = None
        self._roster_cache = {}
        self.player_stats = self._load_player_stats()
        self.draft_history = self._load_draft_history()
        self.prev_season_eff = self._load_previous_season_stats()
        self.injuries_df = self._load_injuries()
        # An empty first load must not be trusted for a full TTL cycle.
        self._injuries_ts = time.time() if not self.injuries_df.empty else 0.0

    def _load_previous_season_stats(self) -> Dict[str, Dict[str, float]]:
        """Baseline ratings for players without a current-season stat line.

        Without this a returning starter who has not played yet this season is
        rated as an undrafted replacement, which is how the whole GSW rotation
        ends up valued at 4.5 EFF.
        """
        try:
            # SDV/ESPN label a season by the year it ends in, so the season
            # before "2026-27" is labelled 2026.
            prev_season = int(str(self.season).split('-')[0])
        except (TypeError, ValueError, IndexError):
            return {}

        raw_dir = os.path.join(DATA_DIR, 'raw')
        for fname in ('sdv_player_box.parquet', 'sdv_player_box.csv'):
            path = os.path.join(raw_dir, fname)
            if os.path.exists(path):
                try:
                    table = _load_prev_season_eff(path, os.path.getmtime(path), prev_season)
                    if table:
                        return table
                except Exception as e:
                    logger.warning(f"Previous-season ratings unavailable ({path}): {e}")
        return {}

    def _load_injuries(self) -> pd.DataFrame:
        """Load injury data from ESPN collector or fallback cache."""
        try:
            return self.espn.get_injuries()
        except Exception as e:
            logger.warning(f"Could not load injuries: {e}")
            return pd.DataFrame()

    def _refresh_injuries(self, force: bool = False) -> None:
        """Re-fetch the injury report once the snapshot is older than the TTL.

        Keeps the last good report when a fetch fails, so a transient network
        error cannot silently mark every player healthy.
        """
        if not force and (time.time() - self._injuries_ts) < self.INJURY_REFRESH_TTL_SECONDS:
            return
        fresh = self._load_injuries()
        if fresh is not None and not fresh.empty:
            self.injuries_df = fresh
        self._injuries_ts = time.time()

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
        Calculates a team's power rating from the EFF_PG of its eight best
        players on the current ESPN roster: 70% weight on the top five and 30%
        on the next three, scaled to an eight-man rotation. The healthy rating
        ranks by EFF; the active rating ranks by EFF after injuries are applied.
        Neither reads tonight's RotoWire lineup, which only drives the card.
        """
        self._refresh_injuries()

        injuries_as_of = (
            time.strftime('%Y-%m-%d %H:%M', time.localtime(self._injuries_ts))
            if self._injuries_ts else ""
        )

        # Default rating for fallback
        result = {
            'team_name': team_name,
            'roster_rating': 0.0,
            'full_roster_rating': 0.0,
            #: The active rating before MAX_INJURY_LOSS is applied. Without it
            #: the cap destroys the only evidence that could tune the cap.
            'uncapped_roster_rating': 0.0,
            'injury_cap_bound': False,
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
            'injuries_as_of': injuries_as_of,
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
                    # Keep one decimal so the offence/defence split below can add
                    # back up to eff exactly, at both player and team level.
                    eff = round(eff, 1)
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
                    
                    # Split EFF into offence and defence. The raw components are
                    # only used for their shape: they are rescaled so the two
                    # halves always add back up to eff_pg. The old formula
                    # ignored misses and turnovers and multiplied AST/STL/BLK,
                    # so off_loss + def_loss was ~1.85x the real EFF drop.
                    raw_off = pts + 1.5 * ast + oreb
                    raw_def = stl * 2.0 + blk * 2.0 + dreb
                    raw_total = raw_off + raw_def
                    if raw_total > 0:
                        off_eff = round(raw_off * eff / raw_total, 1)
                    else:
                        off_eff = round(eff * 0.65, 1)
                    def_eff = round(eff - off_eff, 1)

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
                        off_blend = round(off_eff * gp / 5.0 + (base_eff * 0.65) * (5 - gp) / 5.0, 1)
                        matched_ratings.append({
                            'name': active_player_map[p_name_norm],
                            'clean_name': active_player_map[p_name_norm],
                            'eff_pg': round(blended_eff, 1),
                            'off_eff': off_blend,
                            'def_eff': round(round(blended_eff, 1) - off_blend, 1),
                            'gp': gp
                        })
                        found_players.add(p_name_norm)
                    
        # 2. Add Rookies & Cold Start players (GP < 5 or not in stats yet)
        prev_table = getattr(self, 'prev_season_eff', None) or {}
        for p_name_norm, original_name in active_player_map.items():
            if p_name_norm not in found_players:
                # Check if they are a recent draft pick
                pick = draft_map.get(p_name_norm, 999)

                # A player with no current-season line is usually a returning
                # starter, not an undrafted replacement. Use last season's real
                # EFF when there is a usable sample, and only fall back to the
                # draft tier when there is nothing to go on.
                prev = prev_table.get(p_name_norm)
                if prev and prev.get('gp', 0) >= 10:
                    eff = round(float(prev['eff_pg']), 1)
                elif pick <= 5:
                    eff = 14.0 # Top 5 pick (Solid starter)
                elif pick <= 14:
                    eff = 9.0  # Lottery (Key rotation)
                elif pick <= 30:
                    eff = 6.0  # First round (Rotation contributor)
                else:
                    eff = 4.5  # General rotation / undrafted baseline (elevated from 3.0)

                off_eff = round(eff * 0.65, 1)
                matched_ratings.append({
                    'name': f"{original_name} (Base)",
                    'clean_name': original_name,
                    'eff_pg': eff,
                    'off_eff': off_eff,
                    'def_eff': round(eff - off_eff, 1),
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
                    # Scale the total first, then split it, so the offence and
                    # defence halves still add back up to the discounted value.
                    p['active_eff'] = round(p['eff_pg'] * 0.25, 1)
                    p['active_off_eff'] = round(p_off * 0.25, 1)
                    p['active_def_eff'] = round(p['active_eff'] - p['active_off_eff'], 1)
                    p['injury_status'] = f"DOUBTFUL{detail_str}"
                elif any(k in st_lower for k in ['questionable', 'day-to-day', 'gtd']):
                    p['active_eff'] = round(p['eff_pg'] * 0.60, 1)
                    p['active_off_eff'] = round(p_off * 0.60, 1)
                    p['active_def_eff'] = round(p['active_eff'] - p['active_off_eff'], 1)
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
                    # Enhanced name matching:
                    # 1. First name prefix match with same last name (e.g. "Herb" -> "Herbert", "Nic" -> "Nicolas", "Cam" -> "Cameron")
                    # 2. First initial + last name match (e.g. "D. Mitchell" -> "Davion Mitchell")
                    s_parts = norm_s.replace('.', ' ').split()
                    if len(s_parts) >= 2:
                        s_first = s_parts[0]
                        s_init = s_first[0]
                        s_last = s_parts[-1]
                        for c_norm, c_obj in candidates_map.items():
                            c_parts = c_norm.split()
                            if len(c_parts) >= 2 and c_parts[-1] == s_last:
                                c_first = c_parts[0]
                                if len(s_first) >= 3 and len(c_first) >= 3 and (c_first.startswith(s_first) or s_first.startswith(c_first)):
                                    matched_player = c_obj
                                    break
                                elif len(s_first) == 1 and c_first[0] == s_init:
                                    matched_player = c_obj
                                    break

                if matched_player and id(matched_player) not in starter_ids:
                    sp_copy = dict(matched_player)
                    sp_copy['position'] = s_pos
                    starter_players.append(sp_copy)
                    starter_ids.add(id(matched_player))
                elif not matched_player:
                    dummy_starter = {
                        'name': s_name,
                        'clean_name': s_name,
                        'position': s_pos,
                        'is_unmatched': True
                    }
                    starter_players.append(dummy_starter)
                    starter_ids.add(id(dummy_starter))

            # Dynamic Starter Baseline for unmatched starters (prevents flat 4.5 dummy trap)
            matched_effs = [s['eff_pg'] for s in starter_players if not s.get('is_unmatched') and s.get('eff_pg', 0) > 0]
            starter_baseline = round(sum(matched_effs) / len(matched_effs), 1) if matched_effs else 11.5

            for s in starter_players:
                if s.get('is_unmatched'):
                    s['eff_pg'] = starter_baseline
                    s['active_eff'] = starter_baseline
                    s['off_eff'] = round(starter_baseline * 0.65, 1)
                    s['def_eff'] = round(starter_baseline - s['off_eff'], 1)
                    s['active_off_eff'] = s['off_eff']
                    s['active_def_eff'] = s['def_eff']
                    s['gp'] = 0
                    s['injury_status'] = None
                    del s['is_unmatched']

        if not matched_ratings:
            return result

        # Full-health baseline = the five best players on the roster. RotoWire's
        # five are tonight's lineup, so when a starter is ruled out it names his
        # replacement and the baseline collapsed to bench level (CLE read 105.57
        # on 2026-11-15 instead of ~128, which also disabled the injury cap).
        full_starter_players = sorted(
            matched_ratings, key=lambda x: x.get('eff_pg', 0.0), reverse=True
        )[:5]
        full_starter_ids = {id(p) for p in full_starter_players}
        bench_full_top_3 = sorted(
            [p for p in matched_ratings if id(p) not in full_starter_ids],
            key=lambda x: x.get('eff_pg', 0.0), reverse=True
        )[:3]

        # Tonight's lineup card, as RotoWire reports it. Display only - the
        # rating below no longer reads it.
        lineup_players = []
        available_bench = sorted(
            [p for p in matched_ratings
             if id(p) not in starter_ids and p.get('active_eff', 0.0) > 0],
            key=lambda x: x.get('active_eff', 0.0),
            reverse=True,
        )
        for s in starter_players:
            status = str(s.get('injury_status') or '').upper()
            if ('OUT' in status or 'DOUBTFUL' in status) and available_bench:
                cover = dict(available_bench.pop(0))
                cover['position'] = s.get('position', cover.get('position', ''))
                cover['promoted_from_bench'] = True
                cover['replaced_player'] = s.get('clean_name', s.get('name'))
                lineup_players.append(cover)
            else:
                lineup_players.append(s)

        # Tonight's strength is the same 70/30 split over the best *available*
        # players. RotoWire's five describe the lineup, not the strength of the
        # roster, and one in three "Expected" lineups is wrong, so they cannot
        # move the number: ranking by active_eff already drops anyone ruled out
        # and scales anyone carrying a designation. It also puts the healthy and
        # active ratings on one scale, where an injury used to cost 12% less on
        # a team without a lineup because the two branches weighted players
        # differently.
        active_ranked = sorted(
            matched_ratings, key=lambda x: x.get('active_eff', 0.0), reverse=True
        )[:8]

        # A player who only enters the top five because a better one is out
        # will play starter minutes, which his bench rate understates. Ruled-out
        # bodies stay at zero - the floor is for replacements, not absences.
        active_starter_players = []
        for p in active_ranked[:5]:
            eff_now = p.get('active_eff', 0.0)
            if 0.0 < eff_now < 12.0 and id(p) not in full_starter_ids:
                p = dict(p)
                p['active_eff'] = 12.0
                p['active_off_eff'] = round(12.0 * 0.65, 1)
                p['active_def_eff'] = round(12.0 - p['active_off_eff'], 1)
                p['promoted_from_bench'] = True
            active_starter_players.append(p)

        bench_top_3 = active_ranked[5:8]
        rotation_display = active_starter_players + bench_top_3

        # One formula for every team: 70% of the value sits in the five best
        # players, 30% in the next three, scaled to an eight-man rotation. Those
        # weights are also closer to how a rotation really spends its minutes
        # (~77/23) than an unweighted top-8 sum (62.5/37.5).
        s_n = len(full_starter_players) or 1
        fb_n = len(bench_full_top_3) or 1
        s_eff_avg = sum(s['eff_pg'] for s in full_starter_players) / s_n
        b_eff_avg = sum(b['eff_pg'] for b in bench_full_top_3) / fb_n
        full_roster_rating = 8.0 * (0.70 * s_eff_avg + 0.30 * b_eff_avg)

        s_off_avg = sum(s.get('off_eff', s['eff_pg'] * 0.65) for s in full_starter_players) / s_n
        b_off_avg = sum(b.get('off_eff', b['eff_pg'] * 0.65) for b in bench_full_top_3) / fb_n
        full_off_rating = 8.0 * (0.70 * s_off_avg + 0.30 * b_off_avg)

        s_def_avg = sum(s.get('def_eff', s['eff_pg'] * 0.35) for s in full_starter_players) / s_n
        b_def_avg = sum(b.get('def_eff', b['eff_pg'] * 0.35) for b in bench_full_top_3) / fb_n
        full_def_rating = 8.0 * (0.70 * s_def_avg + 0.30 * b_def_avg)

        s_act_n = len(active_starter_players) or 1
        b_act_n = len(bench_top_3) or 1
        s_act_eff_avg = sum(s['active_eff'] for s in active_starter_players) / s_act_n
        b_act_eff_avg = sum(b['active_eff'] for b in bench_top_3) / b_act_n
        active_roster_rating = 8.0 * (0.70 * s_act_eff_avg + 0.30 * b_act_eff_avg)

        s_act_off_avg = sum(s.get('active_off_eff', s['active_eff'] * 0.65) for s in active_starter_players) / s_act_n
        b_act_off_avg = sum(b.get('active_off_eff', b['active_eff'] * 0.65) for b in bench_top_3) / b_act_n
        active_off_rating = 8.0 * (0.70 * s_act_off_avg + 0.30 * b_act_off_avg)

        s_act_def_avg = sum(s.get('active_def_eff', s['active_eff'] * 0.35) for s in active_starter_players) / s_act_n
        b_act_def_avg = sum(b.get('active_def_eff', b['active_eff'] * 0.35) for b in bench_top_3) / b_act_n
        active_def_rating = 8.0 * (0.70 * s_act_def_avg + 0.30 * b_act_def_avg)

        # Team Injury Penalty Cap:
        # A single star missing moves an NBA spread by at most ~4 points (~22 rating
        # points). For multi-star/starter absences (e.g. GSW missing 5 starters on
        # 11/20 with raw drop 45.8), a rigid 22.0 cap severely overvalues bench
        # lineups. Scale the ceiling dynamically with the fraction of missing core
        # starters up to MAX_MULTI_INJURY_LOSS (48.0).
        uncapped_active_rating = active_roster_rating
        missing_starter_impact = sum(
            max(0.0, min(1.0, 1.0 - (max(0.0, p.get('active_eff', 0.0)) / p['eff_pg'])))
            for p in full_starter_players
            if p.get('eff_pg', 0) > 0
        )
        effective_cap = (
            self.MAX_INJURY_LOSS
            if missing_starter_impact <= 1.0
            else min(
                self.MAX_MULTI_INJURY_LOSS,
                self.MAX_INJURY_LOSS + 7.5 * (missing_starter_impact - 1.0)
            )
        )
        cap_bound = (
            full_roster_rating > 0
            and (full_roster_rating - active_roster_rating) > effective_cap
        )
        if cap_bound:
            capped_active = full_roster_rating - effective_cap
            shrink = capped_active / active_roster_rating if active_roster_rating > 0 else 0.0
            active_roster_rating = capped_active
            active_off_rating *= shrink
            active_def_rating *= shrink

        # Split the total drop between offence and defence. Clamping each side
        # independently (the old way) broke additivity whenever a lineup change
        # moved one side up and the other down, so off_loss + def_loss no longer
        # matched the real EFF drop.
        total_loss = max(0.0, full_roster_rating - active_roster_rating)
        if total_loss <= 0:
            off_loss = def_loss = 0.0
        else:
            full_side_total = full_off_rating + full_def_rating
            off_share = (full_off_rating / full_side_total) if full_side_total > 0 else 0.65
            off_loss = round(total_loss * off_share, 2)
            def_loss = round(total_loss - off_loss, 2)

        # Injuries list for this team
        team_injuries = [
            f"{p['name']} [{p['injury_status']}]"
            for p in matched_ratings if p.get('injury_status')
        ]

        result['roster_rating'] = float(round(active_roster_rating, 2))
        result['full_roster_rating'] = float(round(full_roster_rating, 2))
        result['uncapped_roster_rating'] = float(round(uncapped_active_rating, 2))
        result['injury_cap_bound'] = bool(cap_bound)
        result['off_loss'] = float(round(off_loss, 2))
        result['def_loss'] = float(round(def_loss, 2))
        result['full_off_rating'] = float(round(full_off_rating, 2))
        result['full_def_rating'] = float(round(full_def_rating, 2))
        result['active_off_rating'] = float(round(active_off_rating, 2))
        result['active_def_rating'] = float(round(active_def_rating, 2))
        result['injuries'] = team_injuries
        # The lineup card keeps RotoWire's five (with positions) so the UI still
        # shows who is expected to start; the rating no longer reads it.
        result['starters'] = [
            {
                'name': p.get('clean_name', p['name']),
                'position': p.get('position', ''),
                'eff_pg': round(float(p.get('eff_pg', 0.0)), 1),
                'active_eff': round(float(p.get('active_eff', 0.0)), 1),
                'injury_status': p.get('injury_status')
            }
            for p in (lineup_players or active_starter_players)
        ]
        result['lineup_status'] = lineup_status
        result['is_lineup_confirmed'] = is_confirmed
        result['top_players'] = [
            f"{p.get('position', '') + ': ' if p.get('position') else ''}{p['name']} ({p['active_eff']:.1f}" + (f" - {p['injury_status']})" if p.get('injury_status') else ")")
            for p in rotation_display[:8]
        ]
        result['status'] = 'success'

        return result

