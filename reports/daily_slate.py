import pandas as pd
import logging
import re
from datetime import datetime
from typing import Optional, Tuple
from config import parse_spread, calculate_home_spread

logger = logging.getLogger(__name__)


def _parse_spread(spread_text) -> Tuple[Optional[str], Optional[float]]:
    """Parse spread text using centralized config.parse_spread."""
    return parse_spread(spread_text)


class DailySlate:
    """Generates daily analysis for all NBA games.
    
    Orchestrates data collection and analysis for a full day's slate of games.
    """

    def __init__(self, espn_collector, feature_engineer, trend_analyzer, game_reporter):
        self.espn = espn_collector
        self.fe = feature_engineer
        self.analyzer = trend_analyzer
        self.reporter = game_reporter

    def analyze_today(self) -> str:
        """Analyze all of today's games (in Thai timezone UTC+7) and return formatted report."""
        from datetime import timezone, timedelta
        thai_now = datetime.now(timezone(timedelta(hours=7)))
        return self.analyze_date(thai_now.strftime('%Y%m%d'))

    def analyze_date(self, date: str) -> str:
        """Analyze all games for a specific date (supports Thai date).
        
        Args:
            date: Date in YYYYMMDD format
        
        Steps:
        1. Fetch scoreboard from ESPN for the date (checking Thai date first)
        2. For each game, run full_matchup_analysis
        3. Format all results using GameReporter
        4. Return formatted daily slate report
        """
        logger.info(f"Analyzing games for {date}")
        
        try:
            # Query using Thai date converter if available
            scoreboard = pd.DataFrame()
            if hasattr(self.espn, 'get_scoreboard_by_thai_date'):
                scoreboard = self.espn.get_scoreboard_by_thai_date(date)
            if scoreboard.empty:
                scoreboard = self.espn.get_scoreboard(date)
        except Exception as e:
            logger.error(f"Error fetching scoreboard: {e}")
            return f"Failed to fetch games for {date}"
        
        if scoreboard.empty:
            return f"No NBA games found for {date} (เวลาไทย / Thai Time)"
        
        games = []
        for _, game in scoreboard.iterrows():
            try:
                home_team = game.get('home_team', '')
                away_team = game.get('away_team', '')
                raw_spread = game.get('spread', None)
                ou = game.get('over_under', None)
                
                # Parse spread text (e.g., "ATL -1.5" -> -1.5)
                spread_value = calculate_home_spread(raw_spread, home_team, away_team)
                ou_value = float(ou) if pd.notna(ou) else None
                
                # Run analysis
                analysis = self.analyzer.full_matchup_analysis(
                    home_team=home_team,
                    away_team=away_team,
                    spread_line=spread_value,
                    over_under_line=ou_value
                )
                
                game_info = {
                    'game_id': game.get('game_id', ''),
                    'date': date,
                    'venue': game.get('venue', 'TBD'),
                    'status': game.get('status', ''),
                    'home_ml': game.get('home_ml', None),
                    'away_ml': game.get('away_ml', None),
                    'spread_text': raw_spread,
                    'over_under': ou_value,
                    'thai_time': game.get('thai_time'),
                    'thai_time_display': game.get('thai_time_display'),
                }
                
                games.append({'analysis': analysis, 'game_info': game_info})
                
            except Exception as e:
                logger.error(f"Error analyzing {home_team} vs {away_team}: {e}")
                continue
        
        if not games:
            return f"Could not analyze any games for {date}"
        
        return self.reporter.format_daily_slate(games)

    def get_best_picks(self, date: str = None, min_confidence: float = 0.55, min_ev: float = 0.0) -> str:
        """Get only the highest confidence picks for the day.
        Filters to games where O/U or Spread confidence >= min_confidence and ev >= min_ev.
        Returns formatted summary.
        """
        if not date:
            from datetime import timezone, timedelta
            thai_now = datetime.now(timezone(timedelta(hours=7)))
            date = thai_now.strftime('%Y%m%d')
            
        logger.info(f"Getting best picks for {date} with min_confidence={min_confidence}, min_ev={min_ev}")
        
        try:
            scoreboard = pd.DataFrame()
            if hasattr(self.espn, 'get_scoreboard_by_thai_date'):
                scoreboard = self.espn.get_scoreboard_by_thai_date(date)
            if scoreboard.empty:
                scoreboard = self.espn.get_scoreboard(date)
        except Exception as e:
            logger.error(f"Error fetching scoreboard: {e}")
            return f"Failed to fetch games for {date}"
            
        if scoreboard.empty:
            return f"No NBA games found for {date}"
            
        games = []
        for _, game in scoreboard.iterrows():
            try:
                home_team = game.get('home_team', '')
                away_team = game.get('away_team', '')
                raw_spread = game.get('spread', None)
                ou = game.get('over_under', None)
                
                spread_value = calculate_home_spread(raw_spread, home_team, away_team)
                ou_value = float(ou) if pd.notna(ou) else None
                
                analysis = self.analyzer.full_matchup_analysis(
                    home_team=home_team,
                    away_team=away_team,
                    spread_line=spread_value,
                    over_under_line=ou_value
                )
                
                include = False
                ou_an = analysis.get('over_under')
                sp_an = analysis.get('spread')
                if ou_an and ou_an.get('confidence', 0) >= min_confidence and ou_an.get('ev', 0) >= min_ev and ou_an.get('recommendation') != 'NEUTRAL':
                    include = True
                if sp_an and sp_an.get('confidence', 0) >= min_confidence and sp_an.get('ev', 0) >= min_ev and sp_an.get('recommendation') != 'NEUTRAL':
                    include = True
                    
                if include:
                    game_info = {
                        'game_id': game.get('game_id', ''),
                        'date': date,
                        'venue': game.get('venue', 'TBD'),
                        'status': game.get('status', ''),
                        'home_ml': game.get('home_ml', None),
                        'away_ml': game.get('away_ml', None),
                    }
                    games.append({'analysis': analysis, 'game_info': game_info})
                    
            except Exception as e:
                logger.error(f"Error analyzing {home_team} vs {away_team}: {e}")
                continue
                
        if not games:
            return f"No games met the minimum confidence threshold of {min_confidence:.0%} for {date}"
            
        return self.reporter.format_daily_slate(games)
