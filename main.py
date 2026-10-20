"""NBA Data Analysis & Betting Trends System.

CLI entry point that orchestrates data collection, processing, and analysis
for NBA Over/Under and Spread betting trends.

Usage:
    python main.py today                # Analyze all of today's games
    python main.py today --date 20260325 # Analyze games for a specific date
    python main.py teams                # List all NBA teams
    python main.py roster "Boston Celtics"  # Show team roster
    python main.py standings            # View current standings
    python main.py matchup "Boston Celtics" "Los Angeles Lakers"
    python main.py matchup "Boston Celtics" "Los Angeles Lakers" --spread -5.5 --ou 225.5
    python main.py refresh              # Refresh all cached data
"""

import argparse
import logging
import sys
import os
from datetime import datetime

# Fix Windows console encoding for emoji/unicode
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from config import ensure_dirs, NBA_SEASON, NBA_API_SEASON

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('nba')


def cmd_today(args):
    """Analyze all of today's games."""
    from collectors.espn_collector import ESPNCollector
    from analysis.trend_analyzer import TrendAnalyzer
    from processing.feature_engineering import FeatureEngineer
    from reports.game_report import GameReporter
    from reports.daily_slate import DailySlate

    logger.info("Fetching today's NBA games...")
    espn = ESPNCollector()

    # Get game logs for feature engineering
    logger.info("Loading game history for analysis...")
    scoreboard_history = _load_game_history(espn)

    # Try to load advanced stats
    team_advanced = _load_advanced_stats()

    # Setup analysis pipeline
    fe = FeatureEngineer(scoreboard_history, team_advanced)
    analyzer = TrendAnalyzer(fe)
    reporter = GameReporter()
    slate = DailySlate(espn, fe, analyzer, reporter)

    # Analyze
    date_str = args.date if hasattr(args, 'date') and args.date else None
    if date_str:
        report = slate.analyze_date(date_str)
    else:
        report = slate.analyze_today()

    print(report)


def cmd_teams(args):
    """List all NBA teams."""
    from collectors.espn_collector import ESPNCollector

    espn = ESPNCollector()
    teams = espn.get_all_teams()

    if teams.empty:
        print("Could not fetch teams data.")
        return

    print(f"\n{'='*60}")
    print(f"  🏀 NBA {NBA_SEASON} — All Teams")
    print(f"{'='*60}")
    for _, row in teams.iterrows():
        print(f"  [{row.get('abbreviation', 'N/A'):>4}] {row.get('display_name', 'Unknown')}")
    print(f"{'='*60}")
    print(f"  Total: {len(teams)} teams\n")


def cmd_roster(args):
    """Show team roster."""
    from collectors.espn_collector import ESPNCollector

    espn = ESPNCollector()
    teams = espn.get_all_teams()

    if teams.empty:
        print("Could not fetch teams data.")
        return

    # Find team by name (fuzzy match)
    team_name = args.team_name
    match = teams[teams['display_name'].str.contains(team_name, case=False, na=False)]

    if match.empty:
        print(f"Team '{team_name}' not found. Available teams:")
        for _, row in teams.iterrows():
            print(f"  - {row['display_name']}")
        return

    team_row = match.iloc[0]
    team_id = team_row['team_id']
    roster = espn.get_team_roster(int(team_id))

    if roster.empty:
        print(f"Could not fetch roster for {team_row['display_name']}")
        return

    print(f"\n{'='*70}")
    print(f"  🏀 {team_row['display_name']} — {NBA_SEASON} Roster")
    print(f"{'='*70}")
    print(f"  {'#':<4} {'Name':<25} {'Pos':<8} {'Ht':<8} {'Exp':<5} {'College':<20}")
    print(f"  {'-'*66}")
    for _, p in roster.iterrows():
        jersey = str(p.get('jersey', '')).ljust(4) if p.get('jersey') else '  - '
        name = str(p.get('name', 'Unknown'))[:24].ljust(25)
        pos = str(p.get('position', 'N/A'))[:7].ljust(8)
        ht = str(p.get('height', 'N/A'))[:7].ljust(8)
        exp = str(p.get('experience', 'R'))[:4].ljust(5)
        college = str(p.get('college', '-'))[:19].ljust(20)
        print(f"  {jersey}{name}{pos}{ht}{exp}{college}")
    print(f"{'='*70}")
    print(f"  Total: {len(roster)} players\n")


def cmd_standings(args):
    """Show current standings."""
    from collectors.espn_collector import ESPNCollector

    espn = ESPNCollector()
    standings = espn.get_standings()

    if standings.empty:
        print("Could not fetch standings.")
        return

    print(f"\n{'='*75}")
    print(f"  🏀 NBA {NBA_SEASON} Standings")
    print(f"{'='*75}")

    for conf in standings['conference'].unique():
        conf_teams = standings[standings['conference'] == conf].copy()
        print(f"\n  📋 {conf}")
        print(f"  {'Team':<25} {'W':>3} {'L':>3} {'PCT':>6} {'GB':>5} {'Streak':>7} {'Home':>7} {'Away':>7}")
        print(f"  {'-'*70}")
        for _, row in conf_teams.iterrows():
            team = str(row.get('team', ''))[:24].ljust(25)
            w = str(int(row.get('wins', 0))).rjust(3)
            l = str(int(row.get('losses', 0))).rjust(3)
            pct = f"{row.get('win_pct', 0):.3f}".rjust(6)
            gb = str(row.get('games_behind', '-')).rjust(5)
            streak = str(row.get('streak', '-')).rjust(7)
            home = str(row.get('home_record', '-')).rjust(7)
            away = str(row.get('away_record', '-')).rjust(7)
            print(f"  {team}{w}{l}{pct}{gb}{streak}{home}{away}")

    print(f"\n{'='*75}\n")


def cmd_matchup(args):
    """Analyze a specific matchup."""
    from collectors.espn_collector import ESPNCollector
    from analysis.trend_analyzer import TrendAnalyzer
    from processing.feature_engineering import FeatureEngineer
    from reports.game_report import GameReporter

    espn = ESPNCollector()
    logger.info("Loading game history for analysis...")
    game_history = _load_game_history(espn)
    team_advanced = _load_advanced_stats()

    fe = FeatureEngineer(game_history, team_advanced)
    analyzer = TrendAnalyzer(fe)
    reporter = GameReporter()

    home_team = args.home_team
    away_team = args.away_team
    spread = args.spread if hasattr(args, 'spread') and args.spread else None
    ou = args.ou if hasattr(args, 'ou') and args.ou else None

    # Try to get current odds from The Odds API
    odds_data = None
    if spread is None or ou is None:
        try:
            from collectors.odds_collector import OddsCollector
            logger.info("Fetching live Vegas lines from The Odds API...")
            odds_coll = OddsCollector()
            odds_data = odds_coll.get_game_odds(home_team, away_team)
            c_spread, c_total = odds_coll.get_consensus_lines(odds_data)
            
            if spread is None and c_spread is not None:
                spread = c_spread
                logger.info(f"Auto-detected Consensus Spread: {home_team} {c_spread}")
            
            if ou is None and c_total is not None:
                ou = c_total
                logger.info(f"Auto-detected Consensus O/U: {c_total}")
        except Exception as e:
            logger.warning(f"Could not fetch live odds: {e}")

    # Fallback to ESPN scoreboard if needed
    if spread is None or ou is None:
        logger.info("Checking ESPN for current odds...")
        scoreboard = espn.get_scoreboard()
        if not scoreboard.empty:
            from config import normalize_team_name
            norm_h = normalize_team_name(home_team)
            norm_a = normalize_team_name(away_team)
            for _, game in scoreboard.iterrows():
                gh = normalize_team_name(game.get('home_team', ''))
                ga = normalize_team_name(game.get('away_team', ''))
                if (norm_h == gh and norm_a == ga) or (norm_h in gh and norm_a in ga):
                    if ou is None and game.get('over_under'):
                        ou = float(game['over_under'])
                    if spread is None and game.get('spread'):
                        from config import calculate_home_spread
                        spread = calculate_home_spread(game.get('spread'), home_team, away_team)

    analysis = analyzer.full_matchup_analysis(
        home_team=home_team,
        away_team=away_team,
        spread_line=spread,
        over_under_line=ou
    )

    game_info = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'venue': 'TBD',
        'spread_text': f"{home_team} {spread}" if spread else "N/A",
        'over_under': ou,
        'odds_data': odds_data
    }

    report = reporter.format_matchup_report(analysis, game_info)
    print(report)


def cmd_refresh(args):
    """Refresh all cached data."""
    from collectors.espn_collector import ESPNCollector

    ensure_dirs()

    logger.info("Refreshing data from all sources...")

    # ESPN data
    logger.info("=== ESPN Data ===")
    espn = ESPNCollector()

    teams = espn.get_all_teams()
    if not teams.empty:
        teams.to_csv('data/raw/espn_teams.csv', index=False)
        logger.info(f"Saved {len(teams)} teams")

    standings = espn.get_standings()
    if not standings.empty:
        standings.to_csv('data/raw/espn_standings.csv', index=False)
        logger.info(f"Saved standings for {len(standings)} teams")

    rosters = espn.get_all_rosters()
    if not rosters.empty:
        rosters.to_csv('data/raw/espn_rosters.csv', index=False)
        logger.info(f"Saved {len(rosters)} player records")

    # Recent games (last 180 days for full season)
    from datetime import timedelta
    end = datetime.now()
    start = end - timedelta(days=180)
    recent = espn.get_scoreboard_range(
        start.strftime('%Y%m%d'),
        end.strftime('%Y%m%d')
    )
    if not recent.empty:
        recent.to_csv('data/raw/espn_recent_games.csv', index=False)
        logger.info(f"Saved {len(recent)} recent games")

    # SDV historical data
    logger.info("=== SportsDataverse Historical Data ===")
    try:
        from collectors.sdv_collector import SDVCollector
        sdv = SDVCollector()
        sdv.save_raw_data()
    except Exception as e:
        logger.warning(f"SDV data refresh failed (may need pip install sportsdataverse): {e}")

    # Advanced stats
    logger.info("=== Advanced Stats (nba_api) ===")
    try:
        from collectors.advanced_stats_collector import AdvancedStatsCollector
        adv = AdvancedStatsCollector(season=NBA_API_SEASON)
        adv.save_raw_data()
    except Exception as e:
        logger.warning(f"Advanced stats refresh failed (may need pip install nba_api): {e}")

    logger.info("✅ Data refresh complete!")


def _load_game_history(espn):
    """Load game history from cached files or fetch from ESPN.
    Uses incremental update: only fetches new games since last cache date."""
    import pandas as pd
    import os
    from datetime import timedelta

    cache_path = 'data/raw/espn_recent_games.csv'
    max_age_hours = 6

    espn_games = pd.DataFrame()

    # Case 1: No cache exists → full fetch 180 days
    if not os.path.exists(cache_path):
        logger.info("No cached data found, fetching last 180 days from ESPN...")
        end = datetime.now()
        start = end - timedelta(days=180)
        games = espn.get_scoreboard_range(
            start.strftime('%Y%m%d'),
            end.strftime('%Y%m%d')
        )
        if not games.empty:
            # Filter only completed games (scores > 0) and exclude preseason (season_type == 1)
            games = games[(pd.to_numeric(games['home_score'], errors='coerce') > 0) | (pd.to_numeric(games['away_score'], errors='coerce') > 0)]
            if 'season_type' in games.columns:
                games = games[games['season_type'] != 1]
            ensure_dirs()
            games.to_csv(cache_path, index=False)
            logger.info(f"Cached {len(games)} games to {cache_path}")
        espn_games = games

    elif os.path.exists(cache_path):
        # Case 2: Cache exists but still fresh → use as-is
        file_age_seconds = (datetime.now().timestamp() - os.path.getmtime(cache_path))
        file_age_hours = file_age_seconds / 3600

        if file_age_hours < max_age_hours:
            logger.info(f"Loading cached game history ({file_age_hours:.1f}h old, fresh)")
            espn_games = pd.read_csv(cache_path)
        else:
            # Case 3: Cache exists but stale → incremental update (fetch only new days)
            logger.info(f"Game history cache is {file_age_hours:.1f}h old. Running incremental update...")
            existing = pd.read_csv(cache_path)

            if 'date' in existing.columns and not existing.empty:
                existing['date'] = pd.to_datetime(existing['date'], errors='coerce')
                latest_cached = existing['date'].max()
                
                if pd.notna(latest_cached):
                    fetch_start = (latest_cached - timedelta(days=1)).strftime('%Y%m%d')
                    fetch_end = datetime.now().strftime('%Y%m%d')
                    
                    logger.info(f"Fetching games from {fetch_start} to {fetch_end}...")
                    new_games = espn.get_scoreboard_range(fetch_start, fetch_end)
                    
                    if not new_games.empty:
                        new_games['date'] = pd.to_datetime(new_games['date'], errors='coerce')
                        # Filter only completed games (scores > 0) and exclude preseason
                        new_games = new_games[(pd.to_numeric(new_games['home_score'], errors='coerce') > 0) | (pd.to_numeric(new_games['away_score'], errors='coerce') > 0)]
                        if 'season_type' in new_games.columns:
                            new_games = new_games[new_games['season_type'] != 1]
                        cutoff = latest_cached - timedelta(days=1)
                        old_kept = existing[existing['date'] < cutoff]
                        
                        combined = pd.concat([old_kept, new_games], ignore_index=True)
                        combined = combined.drop_duplicates(subset=['game_id'], keep='last') if 'game_id' in combined.columns else combined
                        combined = combined.sort_values('date').reset_index(drop=True)
                        
                        combined.to_csv(cache_path, index=False)
                        logger.info(f"Incremental update: {len(new_games)} new/updated games → total {len(combined)} games cached")
                        espn_games = combined
                    else:
                        logger.info("No new games found. Using existing cache.")
                        os.utime(cache_path, None)
                        espn_games = existing
            
            if espn_games.empty:
                end = datetime.now()
                start = end - timedelta(days=180)
                games = espn.get_scoreboard_range(
                    start.strftime('%Y%m%d'),
                    end.strftime('%Y%m%d')
                )
                if not games.empty:
                    games.to_csv(cache_path, index=False)
                espn_games = games

    # Merge ESPN recent games with SDV historical games
    from processing.data_merger import DataMerger
    merger = DataMerger()
    sdv_path = 'data/raw/sdv_team_box.csv'
    sdv_df = pd.DataFrame()
    if os.path.exists(sdv_path):
        try:
            sdv_df = pd.read_csv(sdv_path)
        except Exception as e:
            logger.warning(f"Error reading SDV data: {e}")

    unified_games = merger.merge_game_data(espn_games, sdv_df)
    logger.info(f"Loaded {len(unified_games)} unified games for analysis")
    return unified_games



def _load_advanced_stats():
    """Load advanced stats from cache if available."""
    import pandas as pd
    import os

    cache_path = 'data/raw/nba_api_advanced.csv'
    if os.path.exists(cache_path):
        logger.info(f"Loading cached advanced stats from {cache_path}")
        return pd.read_csv(cache_path)

    logger.info("No cached advanced stats found. Run --refresh to fetch.")
    return None


def main():
    parser = argparse.ArgumentParser(
        description='NBA Data Analysis & Betting Trends System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    subparsers = parser.add_subparsers(dest='command')

    # --today / --date
    today_parser = subparsers.add_parser('today', help="Analyze today's games")
    today_parser.add_argument('--date', '-d', type=str, default=None,
                              help='Specific date in YYYYMMDD format')
    today_parser.set_defaults(func=cmd_today)

    # --teams
    teams_parser = subparsers.add_parser('teams', help='List all NBA teams')
    teams_parser.set_defaults(func=cmd_teams)

    # --roster
    roster_parser = subparsers.add_parser('roster', help='Show team roster')
    roster_parser.add_argument('team_name', type=str, help='Team name (partial match OK)')
    roster_parser.set_defaults(func=cmd_roster)

    # --standings
    standings_parser = subparsers.add_parser('standings', help='Show current standings')
    standings_parser.set_defaults(func=cmd_standings)

    # --matchup
    matchup_parser = subparsers.add_parser('matchup', help='Analyze a matchup')
    matchup_parser.add_argument('home_team', type=str, help='Home team name')
    matchup_parser.add_argument('away_team', type=str, help='Away team name')
    matchup_parser.add_argument('--spread', type=float, default=None,
                                help='Spread line (e.g., -5.5)')
    matchup_parser.add_argument('--ou', type=float, default=None,
                                help='Over/Under line (e.g., 225.5)')
    matchup_parser.set_defaults(func=cmd_matchup)

    # --refresh
    refresh_parser = subparsers.add_parser('refresh', help='Refresh all cached data')
    refresh_parser.set_defaults(func=cmd_refresh)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nAborted.")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
