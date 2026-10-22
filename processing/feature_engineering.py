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

            # Filter out Preseason games if tagged (ESPN season_type == 1)
            if 'season_type' in self.game_logs.columns:
                self.game_logs = self.game_logs[self.game_logs['season_type'] != 1].copy()
            
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

    def _get_team_games(self, team: str, before_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
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
        """Get comprehensive recent form stats for a team."""
        if before_date:
            target_date = pd.to_datetime(before_date)
            if hasattr(target_date, 'tz') and target_date.tz is not None:
                target_date = target_date.tz_convert(None)
        else:
            target_date = None
        games = self._get_team_games(team, before_date=target_date).tail(n_games)
        
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
        
        # Bayesian Shrinkage Baseline (Early-Season Prior)
        # When team has played fewer than 15 games, regress observed points toward NBA League Baseline (114.5 PTS)
        all_team_games = self._get_team_games(team, before_date=target_date)
        total_played = len(all_team_games)
        
        LEAGUE_BASELINE_PTS = 114.5
        PRIOR_WEIGHT_K = 10.0
        
        if total_played < 15 and total_played > 0:
            regressed_scored = (total_played * raw_scored + PRIOR_WEIGHT_K * LEAGUE_BASELINE_PTS) / (total_played + PRIOR_WEIGHT_K)
            regressed_allowed = (total_played * raw_allowed + PRIOR_WEIGHT_K * LEAGUE_BASELINE_PTS) / (total_played + PRIOR_WEIGHT_K)
        elif total_played == 0:
            regressed_scored = LEAGUE_BASELINE_PTS
            regressed_allowed = LEAGUE_BASELINE_PTS
        else:
            regressed_scored = raw_scored
            regressed_allowed = raw_allowed

        result['avg_points_scored'] = round(regressed_scored, 1)
        result['avg_points_allowed'] = round(regressed_allowed, 1)
        result['raw_points_scored'] = round(raw_scored, 1)
        result['raw_points_allowed'] = round(raw_allowed, 1)
        result['avg_total_points'] = round(regressed_scored + regressed_allowed, 1)
        result['avg_point_diff'] = round(regressed_scored - regressed_allowed, 1)
        result['sample_size_regressed'] = total_played < 15
        
        result['over_count'] = overs
        result['under_count'] = unders
        result['current_streak'] = f"{streak_type}{streak}" if streak_type else 'None'
        
        return result

    def get_home_away_splits(self, team: str) -> dict:
        """Calculate home vs away performance splits."""
        from config import normalize_team_name
        norm_team = normalize_team_name(team) or team
        home_games = self.game_logs[self.game_logs['home_team'] == norm_team]
        away_games = self.game_logs[self.game_logs['away_team'] == norm_team]
        
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
        
        # Empirical Home Court Advantage: (Home point diff - Away point diff) / 2.0
        # Regressed via Empirical Bayes towards modern NBA baseline (1.8 pts)
        n_home_g = home_stats['wins'] + home_stats['losses']
        n_away_g = away_stats['wins'] + away_stats['losses']
        if n_home_g > 0 and n_away_g > 0:
            raw_hca = (home_stats['avg_diff'] - away_stats['avg_diff']) / 2.0
            n_min = min(n_home_g, n_away_g)
            shrunk_hca = (n_min * raw_hca + 15.0 * 1.8) / (n_min + 15.0)
        else:
            shrunk_hca = 1.8

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
            
            'home_court_advantage': round(float(shrunk_hca), 2)
        }

    def get_head_to_head(self, team_a: str, team_b: str, n_games: int = 10) -> dict:
        """Get head-to-head history between two teams."""
        h2h = self.game_logs[
            ((self.game_logs['home_team'] == team_a) & (self.game_logs['away_team'] == team_b)) |
            ((self.game_logs['home_team'] == team_b) & (self.game_logs['away_team'] == team_a))
        ].sort_values('date').tail(n_games)
        
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
                
        return {
            'total_games': len(last_results),
            'team_a_wins': a_wins,
            'team_b_wins': b_wins,
            'avg_total_points': float(np.mean(totals)) if totals else 0.0,
            'avg_point_diff': float(np.mean(diffs)) if diffs else 0.0,
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

    def _get_team_adv_stat(self, team: str, stat: str) -> float:
        if self.team_advanced is None or 'team' not in self.team_advanced.columns:
            return None
        from config import normalize_team_name
        norm_team = normalize_team_name(team) or team
        row = self.team_advanced[self.team_advanced['team'] == norm_team]
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
        
        h2h = self.get_head_to_head(home_team, away_team)
        
        comb_5 = home_form_5['avg_points_scored'] + away_form_5['avg_points_scored']
        comb_10 = home_form_10['avg_points_scored'] + away_form_10['avg_points_scored']
        
        # Season avg (we approximate using all past games available in form)
        # For a full implementation, you'd calculate season-to-date average here.
        home_all = self.get_team_recent_form(home_team, 100, before_date=game_date)
        away_all = self.get_team_recent_form(away_team, 100, before_date=game_date)
        comb_season = home_all['avg_points_scored'] + away_all['avg_points_scored']
        
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
            
            'home_pace': self._get_team_adv_stat(home_team, 'pace'),
            'away_pace': self._get_team_adv_stat(away_team, 'pace'),
            'home_ortg': self._get_team_adv_stat(home_team, 'off_rating'),
            'home_drtg': self._get_team_adv_stat(home_team, 'def_rating'),
            'away_ortg': self._get_team_adv_stat(away_team, 'off_rating'),
            'away_drtg': self._get_team_adv_stat(away_team, 'def_rating'),
            
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
        
        home_splits = self.get_home_away_splits(home_team)
        away_splits = self.get_home_away_splits(away_team)
        
        h2h = self.get_head_to_head(home_team, away_team)
        
        h_net = self._get_team_adv_stat(home_team, 'net_rating')
        a_net = self._get_team_adv_stat(away_team, 'net_rating')
        
        power_gap = None
        if h_net is not None and a_net is not None:
            power_gap = h_net - a_net
            
        proj_margin = home_form_10['avg_point_diff'] - away_form_10['avg_point_diff']
        # Incorporate empirical home court advantage (modern NBA baseline 1.8 pts)
        proj_margin += home_splits.get('home_court_advantage', 1.8)
        
        line_vs_proj = spread_line - proj_margin if spread_line is not None else None
        
        return {
            'home_point_diff_5': home_form_5['avg_point_diff'],
            'home_point_diff_10': home_form_10['avg_point_diff'],
            'away_point_diff_5': away_form_5['avg_point_diff'],
            'away_point_diff_10': away_form_10['avg_point_diff'],
            
            'home_net_rating': h_net,
            'away_net_rating': a_net,
            
            'home_record': (home_form_10['wins'], home_form_10['losses']), # using last 10 as proxy for record
            'away_record': (away_form_10['wins'], away_form_10['losses']),
            
            'home_home_record': (home_splits['home_wins'], home_splits['home_losses']),
            'away_away_record': (away_splits['away_wins'], away_splits['away_losses']),
            
            'home_streak': home_form_10['current_streak'],
            'away_streak': away_form_10['current_streak'],
            
            'h2h_record': (h2h['team_a_wins'], h2h['team_b_wins']),
            'h2h_avg_margin': h2h['avg_point_diff'],
            
            'power_gap': power_gap,
            'projected_margin': proj_margin,
            'line_vs_projection': line_vs_proj,
            'spread_line': spread_line,
            'streak_str': home_form_10['current_streak']
        }

    def get_league_avg_pace(self) -> float:
        """Calculate dynamic league average pace from team advanced stats or default 99.5."""
        if self.team_advanced is not None and not self.team_advanced.empty and 'pace' in self.team_advanced.columns:
            pace_series = pd.to_numeric(self.team_advanced['pace'], errors='coerce').dropna()
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
                'home_off_loss': home_data.get('off_loss', 0.0),
                'home_def_loss': home_data.get('def_loss', 0.0),
                'away_off_loss': away_data.get('off_loss', 0.0),
                'away_def_loss': away_data.get('def_loss', 0.0),
                'home_top_players': home_data.get('top_players', []),
                'away_top_players': away_data.get('top_players', []),
                'home_injuries': home_data.get('injuries', []),
                'away_injuries': away_data.get('injuries', [])
            }
        return {}

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
        
        style = 'UNKNOWN'
        if home_pace is not None and away_pace is not None:
            if home_pace > 100 and away_pace > 100: style = 'FAST_VS_FAST'
            elif home_pace < 97 and away_pace < 97: style = 'SLOW_VS_SLOW'
            else: style = 'FAST_VS_SLOW' # Or some mixed variant
            
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
        
        # Add early season indicators
        home_form_all = self.get_team_recent_form(home_team, 100)
        away_form_all = self.get_team_recent_form(away_team, 100)
        result['home_games_played'] = home_form_all.get('wins', 0) + home_form_all.get('losses', 0)
        result['away_games_played'] = away_form_all.get('wins', 0) + away_form_all.get('losses', 0)
        
        # Add Player-Based Valuation (Roster Ratings)
        roster_data = self.get_roster_ratings(home_team, away_team)
        result.update(roster_data)

        # Dynamic league pace
        result['league_avg_pace'] = self.get_league_avg_pace()

        # Net injury impact on total points (bidirectional: offense lost vs defense leaked)
        h_off = roster_data.get('home_off_loss', 0.0)
        h_def = roster_data.get('home_def_loss', 0.0)
        a_off = roster_data.get('away_off_loss', 0.0)
        a_def = roster_data.get('away_def_loss', 0.0)
        net_injury_impact = -(h_off + a_off) * 0.20 + (h_def + a_def) * 0.16
        result['injury_net_total_impact'] = round(net_injury_impact, 2)
        
        return result
