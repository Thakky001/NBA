import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Creates analytical features for O/U and Spread prediction from NBA game data."""

    def __init__(self, game_logs: pd.DataFrame, team_advanced: Optional[pd.DataFrame] = None):
        """
        Args:
            game_logs: DataFrame of game results
            team_advanced: DataFrame of team advanced stats
        """
        self.game_logs = game_logs.copy() if game_logs is not None else pd.DataFrame()
        self.team_advanced = team_advanced.copy() if team_advanced is not None else None
        self._preprocess()

    def _preprocess(self):
        """Prepare data: ensure date column is datetime, sort by date, 
        compute total_points = home_score + away_score,
        compute point_diff = home_score - away_score."""
        
        # Ensure we have standard column names to work with
        col_map = {}
        for col in self.game_logs.columns:
            lower = str(col).lower()
            if lower in ['date', 'game_date']: col_map[col] = 'date'
            elif lower in ['home_team', 'home']: col_map[col] = 'home_team'
            elif lower in ['away_team', 'away']: col_map[col] = 'away_team'
            elif lower in ['home_score', 'home_pts']: col_map[col] = 'home_score'
            elif lower in ['away_score', 'away_pts']: col_map[col] = 'away_score'
            elif lower in ['over_under', 'ou', 'total']: col_map[col] = 'over_under'
            elif lower in ['spread', 'line']: col_map[col] = 'spread'
            
        self.game_logs.rename(columns=col_map, inplace=True)

        for req_col in ['date', 'home_team', 'away_team', 'home_score', 'away_score']:
            if req_col not in self.game_logs.columns:
                self.game_logs[req_col] = None
        
        if 'date' in self.game_logs.columns:
            self.game_logs['date'] = pd.to_datetime(self.game_logs['date'], format='mixed', errors='coerce', utc=True).dt.tz_localize(None)
                
            self.game_logs.sort_values(by='date', inplace=True)
            self.game_logs.reset_index(drop=True, inplace=True)
            
        # Ensure numeric scores
        if 'home_score' in self.game_logs.columns and 'away_score' in self.game_logs.columns:
            self.game_logs['home_score'] = pd.to_numeric(self.game_logs['home_score'], errors='coerce')
            self.game_logs['away_score'] = pd.to_numeric(self.game_logs['away_score'], errors='coerce')
            
            # Drop games without scores (upcoming/in-progress or unplayed 0-0)
            self.game_logs = self.game_logs.dropna(subset=['home_score', 'away_score']).copy()
            self.game_logs = self.game_logs[(self.game_logs['home_score'] > 0) | (self.game_logs['away_score'] > 0)].copy()

            # Filter out Preseason (1) and Postseason/Playoffs (3) games if tagged
            # to maintain consistent regular-season pace and scoring distribution
            if 'season_type' in self.game_logs.columns:
                self.game_logs = self.game_logs[~self.game_logs['season_type'].isin([1, 1.0, '1', '1.0', 3, 3.0, '3', '3.0'])].copy()
            
            self.game_logs['total_points'] = self.game_logs['home_score'] + self.game_logs['away_score']
            self.game_logs['point_diff'] = self.game_logs['home_score'] - self.game_logs['away_score']
            
        if self.team_advanced is not None:
            # Standardize team advanced columns
            adv_map = {}
            for col in self.team_advanced.columns:
                lower = str(col).lower()
                if lower in ['team', 'team_name', 'teamname']: adv_map[col] = 'team'
                elif lower in ['pace']: adv_map[col] = 'pace'
                elif lower in ['off_rating', 'ortg', 'off_rtg']: adv_map[col] = 'off_rating'
                elif lower in ['def_rating', 'drtg', 'def_rtg']: adv_map[col] = 'def_rating'
                elif lower in ['net_rating', 'nrtg', 'net_rtg']: adv_map[col] = 'net_rating'
            self.team_advanced.rename(columns=adv_map, inplace=True)
            if 'team' in self.team_advanced.columns:
                from config import normalize_team_name
                self.team_advanced['team'] = self.team_advanced['team'].apply(lambda t: normalize_team_name(str(t)) or t)

    def _validated_games_played(self, count: Optional[int], game_date, season_start, result: dict) -> int:
        """Return the game count, or zero when the calendar says it is impossible.

        A team plays at most one game per day, so a count above that is a data
        error - it happened when a previous season's log leaked into the current
        one. Left in place the count reads as a mature sample and switches off
        both the early-season market anchor and the STRONG sample floor, which
        is the opposite of failing closed. There is no safe way to guess the
        real number, so it becomes zero and the row is flagged unreliable.
        """
        if count is None:
            return 0
        try:
            allowed = (pd.Timestamp(game_date) - pd.Timestamp(season_start)).days
        except (TypeError, ValueError):
            return int(count)
        if allowed >= 0 and int(count) > allowed:
            result['dq_games_played_invalid'] = True
            return 0
        return int(count)

    def _get_team_games(self, team: str, before_date: Optional[pd.Timestamp] = None,
                        current_season_only: bool = False,
                        regular_season_only: bool = False) -> pd.DataFrame:
        """Helper to get all games for a specific team, sorted by date."""
        from config import normalize_team_name
        norm_team = normalize_team_name(team) or team
        df = self.game_logs[
            (self.game_logs['home_team'] == norm_team) | (self.game_logs['away_team'] == norm_team)
        ].copy()
        
        if before_date is not None:
            b_date = pd.to_datetime(before_date)
            if hasattr(b_date, 'tz') and b_date.tz is not None:
                b_date = b_date.tz_convert(None)
            df = df[df['date'] < b_date]
            ref_date = b_date
        else:
            ref_date = df['date'].max() if not df.empty else None

        if current_season_only and ref_date is not None and pd.notna(ref_date):
            ref_dt = pd.to_datetime(ref_date)
            season_year = ref_dt.year if ref_dt.month >= 10 else ref_dt.year - 1
            season_start = pd.Timestamp(year=season_year, month=10, day=1)
            season_end = pd.Timestamp(year=season_year + 1, month=9, day=30)
            df = df[(df['date'] >= season_start) & (df['date'] <= season_end)]

        if regular_season_only and 'season_type' in df.columns:
            df = df[~df['season_type'].isin([1, 1.0, '1', '1.0', 3, 3.0, '3', '3.0'])]
            
        return df.sort_values('date')

    def rolling_average(self, team: str, column: str, n_games: int = 5) -> pd.Series:
        """Calculate rolling average of a stat for a team over last N games."""
        games = self._get_team_games(team)
        virtual_cols = ['score', 'points_scored', 'points_allowed', 'point_diff']
        if games.empty or (column not in games.columns and column not in virtual_cols):
            return pd.Series(dtype=float)
            
        # Extract the stat from the team's perspective
        stat_series = []
        for _, row in games.iterrows():
            if column in ['score', 'points_scored']:
                val = row['home_score'] if row['home_team'] == team else row['away_score']
            elif column in ['points_allowed']:
                val = row['away_score'] if row['home_team'] == team else row['home_score']
            elif column in ['point_diff']:
                val = row['point_diff'] if row['home_team'] == team else -row['point_diff']
            else:
                val = row.get(column, np.nan)
            stat_series.append(val)
            
        games['temp_stat'] = stat_series
        games['temp_stat'] = pd.to_numeric(games['temp_stat'], errors='coerce')
        return games['temp_stat'].rolling(window=n_games, min_periods=1).mean()

    def get_team_recent_form(self, team: str, n_games: int = 10, before_date: Optional[str] = None) -> dict:
        """Get comprehensive recent form stats for a team (strictly current regular season)."""
        if before_date:
            target_date = pd.to_datetime(before_date)
            if hasattr(target_date, 'tz') and target_date.tz is not None:
                target_date = target_date.tz_convert(None)
        else:
            target_date = None
        games = self._get_team_games(team, before_date=target_date, current_season_only=True, regular_season_only=True).tail(n_games)
        
        LEAGUE_BASELINE_PTS = 114.5
        result = {
            'wins': 0, 'losses': 0, 'win_pct': 0.5,
            'avg_points_scored': LEAGUE_BASELINE_PTS, 'avg_points_allowed': LEAGUE_BASELINE_PTS,
            'avg_total_points': LEAGUE_BASELINE_PTS * 2, 'avg_point_diff': 0.0,
            'over_count': 0, 'under_count': 0,
            'current_streak': 'None',
            'raw_points_scored': LEAGUE_BASELINE_PTS,
            'raw_points_allowed': LEAGUE_BASELINE_PTS,
            'sample_size_regressed': True
        }
        
        if games.empty:
            return result
            
        wins = 0
        losses = 0
        pts_scored = []
        pts_allowed = []
        totals = []
        diffs = []
        overs = 0
        unders = 0
        streak = 0
        streak_type = None
        
        for idx, row in games.iterrows():
            is_home = row['home_team'] == team
            scored = row['home_score'] if is_home else row['away_score']
            allowed = row['away_score'] if is_home else row['home_score']
            
            if pd.notna(scored) and pd.notna(allowed):
                diff = scored - allowed
                pts_scored.append(scored)
                pts_allowed.append(allowed)
                totals.append(scored + allowed)
                diffs.append(diff)
                
                won = diff > 0
                if won:
                    wins += 1
                    if streak_type == 'W':
                        streak += 1
                    else:
                        streak_type = 'W'
                        streak = 1
                elif diff < 0:
                    losses += 1
                    if streak_type == 'L':
                        streak += 1
                    else:
                        streak_type = 'L'
                        streak = 1
                        
                ou_line = row.get('over_under')
                if pd.notna(ou_line):
                    total = scored + allowed
                    try:
                        ou_val = float(ou_line)
                        if total > ou_val:
                            overs += 1
                        elif total < ou_val:
                            unders += 1
                    except (ValueError, TypeError):
                        pass
                    
        result['wins'] = wins
        result['losses'] = losses
        total_games = wins + losses
        result['win_pct'] = wins / total_games if total_games > 0 else 0.0
        
        raw_scored = float(np.mean(pts_scored)) if pts_scored else 114.5
        raw_allowed = float(np.mean(pts_allowed)) if pts_allowed else 114.5
        
        all_team_games = self._get_team_games(team, before_date=target_date)
        total_played = len(all_team_games)

        regressed_scored = raw_scored
        regressed_allowed = raw_allowed

        result['avg_points_scored'] = round(regressed_scored, 1)
        result['avg_points_allowed'] = round(regressed_allowed, 1)
        result['raw_points_scored'] = round(raw_scored, 1)
        result['raw_points_allowed'] = round(raw_allowed, 1)
        result['avg_total_points'] = round(regressed_scored + regressed_allowed, 1)
        result['avg_point_diff'] = round(regressed_scored - regressed_allowed, 1)
        result['sample_size_regressed'] = False
        
        result['over_count'] = overs
        result['under_count'] = unders
        result['current_streak'] = f"{streak_type}{streak}" if streak_type else 'None'
        
        return result

    def get_home_away_splits(self, team: str, before_date: Optional[str] = None) -> dict:
        """Calculate home vs away performance splits."""
        from config import normalize_team_name
        norm_team = normalize_team_name(team) or team
        games = self._get_team_games(team, before_date=pd.to_datetime(before_date) if before_date else None)
        home_games = games[games['home_team'] == norm_team]
        away_games = games[games['away_team'] == norm_team]
        
        def agg_split(df, is_home):
            scored_col = 'home_score' if is_home else 'away_score'
            allowed_col = 'away_score' if is_home else 'home_score'
            
            wins = (df[scored_col] > df[allowed_col]).sum()
            losses = (df[scored_col] < df[allowed_col]).sum()
            total = wins + losses
            
            return {
                'wins': int(wins),
                'losses': int(losses),
                'win_pct': float(wins / total) if total > 0 else 0.0,
                'avg_scored': float(df[scored_col].mean()) if not df.empty else 0.0,
                'avg_allowed': float(df[allowed_col].mean()) if not df.empty else 0.0,
                'avg_diff': float((df[scored_col] - df[allowed_col]).mean()) if not df.empty else 0.0
            }
            
        home_stats = agg_split(home_games, True)
        away_stats = agg_split(away_games, False)
        
        # Descriptive, as-of split estimate only; the forecasting margin does not add it.
        n_home_g = home_stats['wins'] + home_stats['losses']
        n_away_g = away_stats['wins'] + away_stats['losses']
        if n_home_g > 0 and n_away_g > 0:
            hca = (home_stats['avg_diff'] - away_stats['avg_diff']) / 2.0
        else:
            hca = None

        return {
            'home_wins': home_stats['wins'],
            'home_losses': home_stats['losses'],
            'home_win_pct': home_stats['win_pct'],
            'home_avg_scored': home_stats['avg_scored'],
            'home_avg_allowed': home_stats['avg_allowed'],
            
            'away_wins': away_stats['wins'],
            'away_losses': away_stats['losses'],
            'away_win_pct': away_stats['win_pct'],
            'away_avg_scored': away_stats['avg_scored'],
            'away_avg_allowed': away_stats['avg_allowed'],
            
            'home_court_advantage': round(float(hca), 2) if hca is not None else None
        }

    def get_head_to_head(self, team_a: str, team_b: str, n_games: int = 10,
                         before_date: Optional[str] = None) -> dict:
        """Get head-to-head history between two teams."""
        h2h = self.game_logs[
            ((self.game_logs['home_team'] == team_a) & (self.game_logs['away_team'] == team_b)) |
            ((self.game_logs['home_team'] == team_b) & (self.game_logs['away_team'] == team_a))
        ].sort_values('date')
        if before_date is not None:
            cutoff = pd.to_datetime(before_date, utc=True).tz_convert(None)
            h2h = h2h[h2h['date'] < cutoff]
        h2h = h2h.tail(n_games)
        
        a_wins = 0
        b_wins = 0
        totals = []
        diffs = [] # From team_a perspective
        last_results = []
        
        for _, row in h2h.iterrows():
            a_is_home = row['home_team'] == team_a
            a_score = row['home_score'] if a_is_home else row['away_score']
            b_score = row['away_score'] if a_is_home else row['home_score']
            
            if pd.notna(a_score) and pd.notna(b_score):
                if a_score > b_score: a_wins += 1
                elif b_score > a_score: b_wins += 1
                
                totals.append(a_score + b_score)
                diffs.append(a_score - b_score)
                
                last_results.append({
                    'date': row['date'].strftime('%Y-%m-%d') if pd.notna(row['date']) else None,
                    'team_a_score': a_score,
                    'team_b_score': b_score,
                    'winner': team_a if a_score > b_score else team_b if b_score > a_score else 'Tie'
                })
                
        total_games = len(last_results)
        return {
            'total_games': total_games,
            'team_a_wins': a_wins if total_games > 0 else None,
            'team_b_wins': b_wins if total_games > 0 else None,
            'avg_total_points': float(np.mean(totals)) if totals else None,
            'avg_point_diff': float(np.mean(diffs)) if diffs else None,
            'last_n_results': last_results
        }

    def get_rest_days(self, team: str, game_date: str) -> int:
        """Calculate number of rest days before a game for a team."""
        target_date = pd.to_datetime(game_date)
        if target_date.tz is not None:
            target_date = target_date.tz_convert(None)
        target_date_norm = target_date.normalize()
        past_games = self._get_team_games(team, before_date=target_date)
        
        if past_games.empty:
            return 3
            
        last_game_date = past_games.iloc[-1]['date']
        if pd.isna(last_game_date):
            return 3
            
        last_game_date_norm = pd.to_datetime(last_game_date)
        if last_game_date_norm.tz is not None:
            last_game_date_norm = last_game_date_norm.tz_convert(None)
        last_game_date_norm = last_game_date_norm.normalize()
            
        delta = target_date_norm - last_game_date_norm
        return max(0, delta.days - 1)

    def _get_team_adv_stat(self, team: str, stat: str, as_of: Optional[str] = None) -> float:
        if self.team_advanced is None or 'team' not in self.team_advanced.columns:
            return None
        from config import normalize_team_name
        norm_team = normalize_team_name(team) or team
        frame = self.team_advanced
        date_col = next((c for c in ('as_of_timestamp', 'published_at', 'snapshot_at') if c in frame.columns), None)
        if as_of is not None:
            if date_col is None:
                return None
            cutoff = pd.to_datetime(as_of, utc=True)
            if cutoff == cutoff.floor('D'):
                cutoff = cutoff + pd.Timedelta(days=1)
            available = pd.to_datetime(frame[date_col], errors='coerce', utc=True)
            frame = frame[available < cutoff]
        row = frame[(frame['team'] == norm_team) | (frame['team'] == team)]
        if row.empty and 'team' in frame.columns:
            matched_idx = frame['team'].apply(lambda x: normalize_team_name(str(x)) == norm_team)
            row = frame[matched_idx]
        if date_col is not None and not row.empty:
            row = row.sort_values(date_col).tail(1)
        if row.empty or stat not in row.columns:
            return None
        val = row.iloc[0][stat]
        return float(val) if pd.notna(val) else None

    def compute_over_under_features(self, home_team: str, away_team: str, 
                                      over_under_line: float = None, game_date: str = None) -> dict:
        """Compute all O/U related features for a matchup."""
        home_form_5 = self.get_team_recent_form(home_team, 5, before_date=game_date)
        home_form_10 = self.get_team_recent_form(home_team, 10, before_date=game_date)
        away_form_5 = self.get_team_recent_form(away_team, 5, before_date=game_date)
        away_form_10 = self.get_team_recent_form(away_team, 10, before_date=game_date)
        
        h2h = self.get_head_to_head(home_team, away_team, before_date=game_date)
        
        # Defensive Form weighting: combine 50% offensive scoring power and 50% defensive containment power
        # Expected Home = 0.5 * (home_scored + away_allowed)
        # Expected Away = 0.5 * (away_scored + home_allowed)
        # Combined = Expected Home + Expected Away
        comb_5 = 0.5 * (home_form_5['avg_points_scored'] + away_form_5['avg_points_scored']) + \
                 0.5 * (home_form_5['avg_points_allowed'] + away_form_5['avg_points_allowed'])
        comb_10 = 0.5 * (home_form_10['avg_points_scored'] + away_form_10['avg_points_scored']) + \
                  0.5 * (home_form_10['avg_points_allowed'] + away_form_10['avg_points_allowed'])
        
        # Season avg (approximate using all past games available in form with offense + defense balance)
        home_all = self.get_team_recent_form(home_team, 100, before_date=game_date)
        away_all = self.get_team_recent_form(away_team, 100, before_date=game_date)
        comb_season = 0.5 * (home_all['avg_points_scored'] + away_all['avg_points_scored']) + \
                      0.5 * (home_all['avg_points_allowed'] + away_all['avg_points_allowed'])
        
        if comb_5 > comb_10 > comb_season:
            trend = 'UP'
        elif comb_5 < comb_10 < comb_season:
            trend = 'DOWN'
        else:
            trend = 'FLAT'
            
        line_vs_avg = over_under_line - comb_10 if over_under_line is not None else None
        
        return {
            'home_avg_total_5': home_form_5['avg_total_points'],
            'home_avg_total_10': home_form_10['avg_total_points'],
            'away_avg_total_5': away_form_5['avg_total_points'],
            'away_avg_total_10': away_form_10['avg_total_points'],
            
            'combined_avg_5': comb_5,
            'combined_avg_10': comb_10,
            'combined_avg_season': comb_season,
            
            'h2h_avg_total': h2h['avg_total_points'],
            
            'home_pace': self._get_team_adv_stat(home_team, 'pace', game_date),
            'away_pace': self._get_team_adv_stat(away_team, 'pace', game_date),
            'home_ortg': self._get_team_adv_stat(home_team, 'off_rating', game_date),
            'home_drtg': self._get_team_adv_stat(home_team, 'def_rating', game_date),
            'away_ortg': self._get_team_adv_stat(away_team, 'off_rating', game_date),
            'away_drtg': self._get_team_adv_stat(away_team, 'def_rating', game_date),
            
            'home_ou_record': (home_form_10['over_count'], home_form_10['under_count']),
            'away_ou_record': (away_form_10['over_count'], away_form_10['under_count']),
            
            'trend_direction': trend,
            'line_vs_average': line_vs_avg,
            'over_under_line': over_under_line,
            'home_streak_str': home_form_10['current_streak'],
            'away_streak_str': away_form_10['current_streak']
        }

    def compute_spread_features(self, home_team: str, away_team: str,
                                  spread_line: float = None, game_date: str = None) -> dict:
        """Compute all Spread related features for a matchup."""
        home_form_5 = self.get_team_recent_form(home_team, 5, before_date=game_date)
        home_form_10 = self.get_team_recent_form(home_team, 10, before_date=game_date)
        away_form_5 = self.get_team_recent_form(away_team, 5, before_date=game_date)
        away_form_10 = self.get_team_recent_form(away_team, 10, before_date=game_date)
        
        home_splits = self.get_home_away_splits(home_team, before_date=game_date)
        away_splits = self.get_home_away_splits(away_team, before_date=game_date)
        
        h2h = self.get_head_to_head(home_team, away_team, before_date=game_date)
        
        h_net = self._get_team_adv_stat(home_team, 'net_rating', game_date)
        a_net = self._get_team_adv_stat(away_team, 'net_rating', game_date)
        
        # Fallback to current season game logs point differential if advanced net rating is missing
        if h_net is None:
            home_season_form = self.get_team_recent_form(home_team, 100, before_date=game_date)
            if home_season_form.get('wins', 0) + home_season_form.get('losses', 0) > 0:
                h_net = home_season_form.get('avg_point_diff')
                
        if a_net is None:
            away_season_form = self.get_team_recent_form(away_team, 100, before_date=game_date)
            if away_season_form.get('wins', 0) + away_season_form.get('losses', 0) > 0:
                a_net = away_season_form.get('avg_point_diff')

        power_gap = None
        if h_net is not None and a_net is not None:
            power_gap = round(float(h_net - a_net), 2)
            
        proj_margin = home_form_10['avg_point_diff'] - away_form_10['avg_point_diff']
        # Incorporate empirical home court advantage (modern NBA baseline 1.8 pts)
        # The raw point-differential features already contain home/away effects.
        
        line_vs_proj = spread_line - proj_margin if spread_line is not None else None
        
        has_h2h = h2h['total_games'] > 0 and h2h['team_a_wins'] is not None and h2h['team_b_wins'] is not None
        h2h_rec = (h2h['team_a_wins'], h2h['team_b_wins']) if has_h2h else None
        last10_h = (home_form_10['wins'], home_form_10['losses'])
        last10_a = (away_form_10['wins'], away_form_10['losses'])
        
        return {
            'home_point_diff_5': home_form_5['avg_point_diff'],
            'home_point_diff_10': home_form_10['avg_point_diff'],
            'away_point_diff_5': away_form_5['avg_point_diff'],
            'away_point_diff_10': away_form_10['avg_point_diff'],
            
            'home_net_rating': h_net,
            'away_net_rating': a_net,
            
            'home_record': last10_h, # backward-compatible alias
            'away_record': last10_a,
            'home_last10_record': last10_h,
            'away_last10_record': last10_a,
            
            'home_home_record': (home_splits['home_wins'], home_splits['home_losses']),
            'away_away_record': (away_splits['away_wins'], away_splits['away_losses']),
            
            'home_streak': home_form_10['current_streak'],
            'away_streak': away_form_10['current_streak'],
            
            'h2h_record': h2h_rec,
            'h2h_avg_margin': h2h['avg_point_diff'],
            
            'power_gap': power_gap,
            'projected_margin': proj_margin,
            'line_vs_projection': line_vs_proj,
            'spread_line': spread_line,
            'streak_str': home_form_10['current_streak']
        }

    def get_league_avg_pace(self, as_of: Optional[str] = None) -> float:
        """Calculate dynamic league average pace from team advanced stats or default 99.5."""
        if self.team_advanced is not None and not self.team_advanced.empty and 'pace' in self.team_advanced.columns:
            frame = self.team_advanced
            date_col = next((c for c in ('as_of_timestamp', 'published_at', 'snapshot_at') if c in frame.columns), None)
            if as_of is not None:
                if date_col is None:
                    return 99.5
                cutoff = pd.to_datetime(as_of, utc=True)
                if cutoff == cutoff.floor('D'):
                    cutoff = cutoff + pd.Timedelta(days=1)
                available = pd.to_datetime(frame[date_col], errors='coerce', utc=True)
                frame = frame[available < cutoff]
            if date_col is not None and not frame.empty:
                frame = frame.assign(_asof=pd.to_datetime(frame[date_col], errors='coerce', utc=True))
                frame = frame.sort_values('_asof').groupby('team', as_index=False).tail(1)
            pace_series = pd.to_numeric(frame['pace'], errors='coerce').dropna()
            if not pace_series.empty and 85.0 <= pace_series.mean() <= 115.0:
                return round(float(pace_series.mean()), 2)
        return 99.5

    def get_roster_ratings(self, home_team: str, away_team: str) -> dict:
        """Fetch roster-based ratings using PlayerRatingEngine."""
        if not hasattr(self, '_player_engine'):
            try:
                from processing.player_rating import PlayerRatingEngine
                self._player_engine = PlayerRatingEngine()
            except Exception as e:
                logger.error(f"Failed to init PlayerRatingEngine: {e}")
                self._player_engine = None
                
        if self._player_engine:
            home_data = self._player_engine.calculate_team_roster_rating(home_team)
            away_data = self._player_engine.calculate_team_roster_rating(away_team)
            return {
                'home_roster_rating': home_data.get('roster_rating', 0.0),
                'away_roster_rating': away_data.get('roster_rating', 0.0),
                'home_full_roster_rating': home_data.get('full_roster_rating', 0.0),
                'away_full_roster_rating': away_data.get('full_roster_rating', 0.0),
                # Pre-cap ratings: the difference from the fields above is how
                # much MAX_INJURY_LOSS is hiding, which is what a future cap
                # study needs.
                'home_uncapped_roster_rating': home_data.get('uncapped_roster_rating', 0.0),
                'away_uncapped_roster_rating': away_data.get('uncapped_roster_rating', 0.0),
                'home_injury_cap': home_data.get('injury_cap', 22.0),
                'away_injury_cap': away_data.get('injury_cap', 22.0),
                'home_missing_starter_impact': home_data.get('missing_starter_impact', 0.0),
                'away_missing_starter_impact': away_data.get('missing_starter_impact', 0.0),
                'dq_injury_cap_bound': bool(
                    home_data.get('injury_cap_bound', False)
                    or away_data.get('injury_cap_bound', False)
                ),
                'home_off_loss': home_data.get('off_loss', 0.0),
                'home_def_loss': home_data.get('def_loss', 0.0),
                'away_off_loss': away_data.get('off_loss', 0.0),
                'away_def_loss': away_data.get('def_loss', 0.0),
                'home_top_players': home_data.get('top_players', []),
                'away_top_players': away_data.get('top_players', []),
                'home_starters': home_data.get('starters', []),
                'away_starters': away_data.get('starters', []),
                'home_lineup_status': home_data.get('lineup_status', 'Unconfirmed'),
                'away_lineup_status': away_data.get('lineup_status', 'Unconfirmed'),
                'is_lineup_confirmed': home_data.get('is_lineup_confirmed', False) and away_data.get('is_lineup_confirmed', False),
                'home_injuries': home_data.get('injuries', []),
                'away_injuries': away_data.get('injuries', []),
                'home_injuries_as_of': home_data.get('injuries_as_of', ''),
                'away_injuries_as_of': away_data.get('injuries_as_of', '')
            }
        return {}

    def is_data_fresh(self, as_of: Optional[str] = None, max_stale_hours: float = 36.0) -> Tuple[bool, str]:
        """Check if game logs and scoreboard data are reasonably fresh.
        In the regular season, NBA games are played almost daily.
        If data has not updated in > max_stale_hours, returns (False, reason).
        """
        if self.game_logs.empty or 'date' not in self.game_logs.columns:
            return False, "Game logs empty or missing date column"
            
        latest_game_date = self.game_logs['date'].max()
        if pd.isna(latest_game_date):
            return False, "Latest game date is NaT"
            
        ref_dt = pd.to_datetime(as_of) if as_of else pd.Timestamp.now()
        if hasattr(ref_dt, 'tz') and ref_dt.tz is not None:
            ref_dt = ref_dt.tz_convert(None)
            
        # If reference date is during NBA offseason (July - September), freshness check does not apply
        if 7 <= ref_dt.month <= 9:
            return True, "Off-season period"
            
        diff_hours = (ref_dt - latest_game_date).total_seconds() / 3600.0
        if diff_hours > max_stale_hours:
            return False, f"Latest game in logs is {diff_hours:.1f} hours old (threshold: {max_stale_hours}h)"
            
        return True, "Data fresh"

    def compute_matchup_features(self, home_team: str, away_team: str,
                                   over_under_line: float = None,
                                   spread_line: float = None,
                                   game_date: str = None) -> dict:
        """Compute ALL features for a matchup (both O/U and Spread combined)."""
        if game_date is None:
            # Default to today if not provided
            game_date = pd.Timestamp.now().strftime('%Y-%m-%d')
            
        ou_feat = self.compute_over_under_features(home_team, away_team, over_under_line, game_date)
        spread_feat = self.compute_spread_features(home_team, away_team, spread_line, game_date)
        
        home_rest = self.get_rest_days(home_team, game_date)
        away_rest = self.get_rest_days(away_team, game_date)
        
        home_pace = ou_feat.get('home_pace')
        away_pace = ou_feat.get('away_pace')
        league_pace = self.get_league_avg_pace(as_of=game_date)
        
        eff_home_pace = home_pace if home_pace is not None else league_pace
        eff_away_pace = away_pace if away_pace is not None else league_pace
        
        if eff_home_pace > 100.5 and eff_away_pace > 100.5:
            style = 'FAST_VS_FAST'
        elif eff_home_pace < 98.5 and eff_away_pace < 98.5:
            style = 'SLOW_VS_SLOW'
        elif (eff_home_pace > 100.5 and eff_away_pace < 98.5) or (eff_home_pace < 98.5 and eff_away_pace > 100.5):
            style = 'FAST_VS_SLOW'
        else:
            style = 'BALANCED_PACE'
            
        fatigue = False
        if home_rest == 0 or away_rest == 0:
            fatigue = True
            
        result = {}
        result.update(ou_feat)
        result.update(spread_feat)
        
        result['home_rest_days'] = home_rest
        result['away_rest_days'] = away_rest
        result['matchup_pace_style'] = style
        result['fatigue_flag'] = fatigue
        
        # Add early season indicators (pass before_date to avoid counting future/unplayed games)
        home_form_all = self.get_team_recent_form(home_team, 100, before_date=game_date)
        away_form_all = self.get_team_recent_form(away_team, 100, before_date=game_date)
        
        # Strictly count current regular season games (from Oct 1 of current season)
        ref_date = game_date or pd.Timestamp.now()
        g_dt = pd.to_datetime(ref_date)
        if hasattr(g_dt, 'tz') and g_dt.tz is not None:
            g_dt = g_dt.tz_convert(None)
        season_year = g_dt.year if g_dt.month >= 10 else g_dt.year - 1
        season_start = pd.Timestamp(year=season_year, month=10, day=1)
        
        h_games = self._get_team_games(home_team, before_date=g_dt, current_season_only=True, regular_season_only=True)
        if not h_games.empty:
            if 'home_score' in h_games.columns:
                h_games = h_games[(h_games['home_score'].notna()) & (h_games['home_score'] > 0)]
            result['home_games_played'] = self._validated_games_played(len(h_games), g_dt, season_start, result)
        else:
            result['home_games_played'] = 0
            
        a_games = self._get_team_games(away_team, before_date=g_dt, current_season_only=True, regular_season_only=True)
        if not a_games.empty:
            if 'home_score' in a_games.columns:
                a_games = a_games[(a_games['home_score'].notna()) & (a_games['home_score'] > 0)]
            result['away_games_played'] = self._validated_games_played(len(a_games), g_dt, season_start, result)
        else:
            result['away_games_played'] = 0
        
        # Add Player-Based Valuation (Roster Ratings)
        roster_data = self.get_roster_ratings(home_team, away_team)
        result.update(roster_data)

        # Dynamic league pace
        result['league_avg_pace'] = self.get_league_avg_pace(as_of=game_date)

        # Net injury impact on total points (bidirectional: offense lost vs defense leaked)
        h_off = roster_data.get('home_off_loss', 0.0)
        h_def = roster_data.get('home_def_loss', 0.0)
        a_off = roster_data.get('away_off_loss', 0.0)
        a_def = roster_data.get('away_def_loss', 0.0)
        net_injury_impact = -(h_off + a_off) * 0.20 + (h_def + a_def) * 0.16
        result['injury_net_total_impact'] = round(net_injury_impact, 2)

        # Data-Quality Flags (Phase 1 & Phase 2 Stabilization)
        h_roster = roster_data.get('home_roster_rating', 0.0) or 0.0
        a_roster = roster_data.get('away_roster_rating', 0.0) or 0.0
        roster_gap = abs(h_roster - a_roster) if (h_roster > 0 and a_roster > 0) else 0.0

        is_fresh, freshness_reason = self.is_data_fresh(as_of=game_date)
        dq_stale_data = not is_fresh
        dq_unconfirmed_lineup = not roster_data.get('is_lineup_confirmed', False)
        dq_missing_net_rating = (result.get('home_net_rating') is None or result.get('away_net_rating') is None)
        dq_extreme_roster_gap = roster_gap >= 30.0
        dq_low_sample = min(result.get('home_games_played', 20), result.get('away_games_played', 20)) < 5
        dq_games_played_invalid = result.get('dq_games_played_invalid', False)

        # Matchup is unreliable if data is stale, or if extreme roster gap is paired with missing net rating or unconfirmed lineups
        dq_is_reliable = not (
            dq_stale_data
            or dq_games_played_invalid
            or (dq_extreme_roster_gap and (dq_unconfirmed_lineup or dq_missing_net_rating))
        )

        result['data_freshness'] = is_fresh
        result['data_freshness_detail'] = freshness_reason
        result['dq_stale_data'] = dq_stale_data
        result['dq_unconfirmed_lineup'] = dq_unconfirmed_lineup
        result['dq_missing_net_rating'] = dq_missing_net_rating
        result['dq_extreme_roster_gap'] = dq_extreme_roster_gap
        result['dq_low_sample'] = dq_low_sample
        result['dq_games_played_invalid'] = dq_games_played_invalid
        result['dq_roster_gap'] = round(roster_gap, 1)
        result['dq_is_reliable'] = dq_is_reliable
        
        return result
