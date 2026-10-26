import os
import re
import time
import logging
from datetime import datetime, date, timedelta, timezone
from typing import Optional, Dict, List, Any, Union
import requests
import pandas as pd

logger = logging.getLogger(__name__)


class ESPNCollector:
    """Collects NBA data from ESPN's public API with retry logic and local cache fallback."""

    BASE_URL = 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba'
    STANDINGS_URL = 'https://site.api.espn.com/apis/v2/sports/basketball/nba/standings'

    def __init__(self, max_retries: int = 3, retry_delay: float = 0.1, backoff_factor: float = 2.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.backoff_factor = backoff_factor
        self._summary_cache = {}
        self._circuit_breaker_until = 0.0
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
        })

    EXPECTED_SCHEMAS = {
        'espn_teams.csv': ['team_id', 'abbreviation', 'name', 'display_name', 'location', 'conference', 'logo_url'],
        'espn_rosters.csv': ['player_id', 'name', 'jersey', 'position', 'height', 'weight', 'age', 'experience', 'college', 'team_id', 'team_name'],
        'espn_recent_games.csv': ['game_id', 'date', 'status', 'season_type', 'home_team', 'home_id', 'home_score', 'away_team', 'away_id', 'away_score', 'spread', 'over_under', 'home_ml', 'away_ml', 'venue'],
        'espn_standings.csv': ['team', 'conference', 'wins', 'losses', 'win_pct', 'games_behind', 'streak', 'home_record', 'away_record', 'points_for', 'points_against', 'point_diff'],
        'espn_injuries.csv': ['team_name', 'player_name', 'status', 'detail', 'side', 'type', 'return_date'],
    }

    def _resolve_raw_dir(self) -> str:
        """Resolve path to data/raw directory."""
        try:
            from config import RAW_DIR
            if os.path.isdir(RAW_DIR):
                return RAW_DIR
        except Exception:
            pass
        # Fallback based on file location
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        raw_path = os.path.join(base, 'data', 'raw')
        if os.path.isdir(raw_path):
            return raw_path
        # Current working directory fallback
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
                    f"Live request failed; activating fallback data from {path} ({len(df)} rows)"
                )
                return df
            except Exception as e:
                logger.error(f"Failed to read fallback CSV {path}: {e}")
        else:
            logger.warning(f"Fallback CSV {path} not found.")
        return pd.DataFrame(columns=expected_cols if expected_cols else None)

    def _get(self, url: str, params: dict = None) -> Optional[dict]:
        """Make GET request with retry backoff, fast timeout, and circuit breaker."""
        now_ts = time.time()
        if now_ts < getattr(self, '_circuit_breaker_until', 0.0):
            logger.info(
                f"ESPN circuit breaker active ({getattr(self, '_circuit_breaker_until', 0.0) - now_ts:.0f}s left); "
                f"activating fast fallback without blocking."
            )
            return None

        last_error = None
        for attempt in range(self.max_retries):
            try:
                # Fast timeout: 3.05s connect, 5.0s read to prevent application freeze
                response = self.session.get(url, params=params, timeout=(3.05, 5.0))
                if response.status_code in [429, 500, 502, 503, 504]:
                    raise requests.exceptions.HTTPError(
                        f"HTTP {response.status_code}: {response.text[:200]}",
                        response=response
                    )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.Timeout as e:
                last_error = e
                # Trip 60s circuit breaker on timeout; fail fast without waiting through 3 slow retries
                self._circuit_breaker_until = time.time() + 60.0
                logger.warning(
                    f"ESPN request timed out for {url}: {e}. Activating 60s circuit breaker to prevent UI freeze."
                )
                break
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries} failed for URL {url}: {e}"
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

        logger.warning(
            f"All {self.max_retries} attempts failed for URL {url}: {last_error}"
        )
        return None

    def get_all_teams(self) -> pd.DataFrame:
        """Fetch all NBA teams.
        Returns DataFrame with columns: team_id, abbreviation, name, display_name,
        location, conference, logo_url
        """
        url = f"{self.BASE_URL}/teams"
        try:
            data = self._get(url)
            if data is None or not data:
                return self._load_fallback_csv('espn_teams.csv')
            leagues = data.get('sports', [{}])[0].get('leagues', [])
            teams_data = leagues[0].get('teams', []) if leagues else []
            if not teams_data:
                return self._load_fallback_csv('espn_teams.csv')

            records = []
            for t in teams_data:
                team = t['team']
                records.append({
                    'team_id': team.get('id'),
                    'abbreviation': team.get('abbreviation'),
                    'name': team.get('name'),
                    'display_name': team.get('displayName'),
                    'location': team.get('location'),
                    'conference': None,
                    'logo_url': team.get('logos', [{}])[0].get('href') if team.get('logos') else None
                })
            df = pd.DataFrame(records)
            if df.empty:
                return self._load_fallback_csv('espn_teams.csv')
            return df
        except Exception as e:
            logger.warning(f"Error fetching/parsing teams data: {e}; activating fallback data.")
            return self._load_fallback_csv('espn_teams.csv')

    def _get_fallback_roster(self, team_id: Optional[int] = None) -> pd.DataFrame:
        """Helper to get fallback roster adhering to schema."""
        df = self._load_fallback_csv('espn_rosters.csv')
        cols = ['player_id', 'name', 'jersey', 'position', 'height', 'weight', 'age', 'experience', 'college']
        if df.empty:
            return pd.DataFrame(columns=cols)
        if team_id is not None and 'team_id' in df.columns:
            filtered = df[df['team_id'].astype(str) == str(team_id)]
            if not filtered.empty:
                avail_cols = [c for c in cols if c in filtered.columns]
                return filtered[avail_cols].copy()
        avail_cols = [c for c in cols if c in df.columns]
        return df[avail_cols].copy()

    def get_team_roster(self, team_id: int) -> pd.DataFrame:
        """Fetch current roster for a team.
        Returns DataFrame with columns: player_id, name, jersey, position,
        height, weight, age, experience, college
        """
        url = f"{self.BASE_URL}/teams/{team_id}/roster"
        try:
            data = self._get(url)
            if data is None or not data:
                return self._get_fallback_roster(team_id)
            athletes = data.get('athletes', [])
            if not athletes:
                return self._get_fallback_roster(team_id)

            records = []
            for athlete in athletes:
                records.append({
                    'player_id': athlete.get('id'),
                    'name': athlete.get('fullName'),
                    'jersey': athlete.get('jersey'),
                    'position': athlete.get('position', {}).get('name'),
                    'height': athlete.get('displayHeight'),
                    'weight': athlete.get('displayWeight'),
                    'age': athlete.get('age'),
                    'experience': athlete.get('experience', {}).get('years'),
                    'college': athlete.get('college', {}).get('name') if athlete.get('college') else None
                })
            df = pd.DataFrame(records)
            if df.empty:
                return self._get_fallback_roster(team_id)
            return df
        except Exception as e:
            logger.warning(f"Error fetching/parsing roster for team {team_id}: {e}; activating fallback data.")
            return self._get_fallback_roster(team_id)

    def get_all_rosters(self) -> pd.DataFrame:
        """Fetch rosters for ALL teams. Calls get_all_teams() first to get IDs,
        then calls get_team_roster() for each. Adds team_name column.
        Includes 0.5s sleep between requests to be respectful."""
        try:
            teams_df = self.get_all_teams()
            if teams_df.empty:
                return self._load_fallback_csv('espn_rosters.csv')

            all_rosters = []
            for _, row in teams_df.iterrows():
                team_id = row['team_id']
                team_name = row['display_name']

                roster_df = self.get_team_roster(team_id)
                if not roster_df.empty:
                    roster_df = roster_df.copy()
                    roster_df['team_id'] = team_id
                    roster_df['team_name'] = team_name
                    all_rosters.append(roster_df)

                time.sleep(0.1)

            if all_rosters:
                combined = pd.concat(all_rosters, ignore_index=True)
                if not combined.empty:
                    return combined
            return self._load_fallback_csv('espn_rosters.csv')
        except Exception as e:
            logger.warning(f"Error in get_all_rosters: {e}; activating fallback data.")
            return self._load_fallback_csv('espn_rosters.csv')

    def _get_fallback_scoreboard(self, date: Optional[str] = None) -> pd.DataFrame:
        """Helper to get fallback scoreboard adhering to schema."""
        df = self._load_fallback_csv('espn_recent_games.csv')
        if df.empty:
            return df
        if date:
            clean_date = str(date).replace('-', '')
            matching = df[df['date'].astype(str).str.replace('-', '').str.contains(clean_date[:8])]
            if not matching.empty:
                return matching.copy()
        return df.copy()

    def get_scoreboard(self, date: Optional[str] = None) -> pd.DataFrame:
        """Fetch scoreboard for a specific date (or today).
        Args:
            date: Date string in YYYYMMDD format. None = today.
        Returns DataFrame with columns: game_id, date, status,
        home_team, home_id, home_score, away_team, away_id, away_score,
        spread, over_under, home_ml, away_ml, venue
        """
        url = f"{self.BASE_URL}/scoreboard"
        params = {}
        if date:
            params['dates'] = date

        try:
            data = self._get(url, params=params)
            if data is None:
                return self._get_fallback_scoreboard(date)

            events = data.get('events', [])
            if not events:
                return pd.DataFrame(columns=self.EXPECTED_SCHEMAS['espn_recent_games.csv'])

            records = []
            for event in events:
                game_id = event.get('id')
                date_str = event.get('date')
                status = event.get('status', {}).get('type', {}).get('name')

                competitions = event.get('competitions', [])
                comp = competitions[0] if competitions else {}
                venue = comp.get('venue', {}).get('fullName')
                competitors = comp.get('competitors', [])

                home_team = {}
                away_team = {}
                for c in competitors:
                    if c.get('homeAway') == 'home':
                        home_team = c
                    else:
                        away_team = c

                home_id = home_team.get('id')
                home_name = home_team.get('team', {}).get('displayName')
                home_score = home_team.get('score')

                away_id = away_team.get('id')
                away_name = away_team.get('team', {}).get('displayName')
                away_score = away_team.get('score')

                # Odds
                odds = comp.get('odds', [])
                odds_dict = odds[0] if odds else {}
                spread = odds_dict.get('details')
                over_under = odds_dict.get('overUnder')
                home_ml = odds_dict.get('homeTeamOdds', {}).get('moneyLine')
                away_ml = odds_dict.get('awayTeamOdds', {}).get('moneyLine')

                status_obj = event.get('status', {})
                period = status_obj.get('period')
                clock = status_obj.get('displayClock')
                clock_detail = status_obj.get('type', {}).get('detail')

                records.append({
                    'game_id': game_id,
                    'date': date_str,
                    'status': status,
                    'period': period,
                    'clock': clock,
                    'clock_detail': clock_detail,
                    'season_type': event.get('season', {}).get('type'),
                    'home_team': home_name,
                    'home_id': home_id,
                    'home_score': home_score,
                    'away_team': away_name,
                    'away_id': away_id,
                    'away_score': away_score,
                    'spread': spread,
                    'over_under': over_under,
                    'home_ml': home_ml,
                    'away_ml': away_ml,
                    'venue': venue
                })

            df = pd.DataFrame(records)
            if df.empty:
                return self._get_fallback_scoreboard(date)
            return df
        except Exception as e:
            logger.warning(f"Error fetching/parsing scoreboard: {e}; activating fallback data.")
            return self._get_fallback_scoreboard(date)

    def get_scoreboard_by_thai_date(self, date_val: Optional[Union[str, datetime, date]] = None) -> pd.DataFrame:
        """Fetch games that take place on a specific Thai calendar date (UTC+7 / Asia/Bangkok).
        
        Because NBA games in the US (UTC-4/UTC-5) take place 11-14 hours behind Thailand,
        games playing on Thai morning/day D are scheduled on US dates (D-1) and D.
        
        Args:
            date_val: Target date as YYYYMMDD string, YYYY-MM-DD string, datetime.date, or datetime.
                      If None, defaults to current date in Thailand.
                      
        Returns DataFrame with games sorted by Thai game time, including extra columns:
            'thai_datetime', 'thai_time', 'thai_date', 'thai_time_display'
        """
        thai_tz = timezone(timedelta(hours=7))

        if date_val is None:
            target_date = datetime.now(thai_tz).date()
        elif isinstance(date_val, datetime):
            target_date = date_val.date()
        elif isinstance(date_val, date):
            target_date = date_val
        elif isinstance(date_val, str):
            clean = date_val.strip().replace('-', '').replace('/', '')
            try:
                target_date = datetime.strptime(clean[:8], '%Y%m%d').date()
            except Exception:
                target_date = datetime.now(thai_tz).date()
        else:
            target_date = datetime.now(thai_tz).date()

        # Query ESPN for US dates D-1 and D
        us_d1 = (target_date - timedelta(days=1)).strftime('%Y%m%d')
        us_d2 = target_date.strftime('%Y%m%d')

        df1 = self.get_scoreboard(us_d1)
        df2 = self.get_scoreboard(us_d2)

        dfs = [df for df in [df1, df2] if df is not None and not df.empty]
        cols = list(self.EXPECTED_SCHEMAS['espn_recent_games.csv']) + ['thai_datetime', 'thai_date', 'thai_time', 'thai_time_display']
        if not dfs:
            return pd.DataFrame(columns=cols)

        combined = pd.concat(dfs, ignore_index=True)
        if 'game_id' in combined.columns:
            combined = combined.drop_duplicates(subset=['game_id'], keep='first')

        if combined.empty or 'date' not in combined.columns:
            return pd.DataFrame(columns=cols)

        thai_dts = pd.to_datetime(combined['date'], utc=True, errors='coerce').dt.tz_convert('Asia/Bangkok')
        combined['thai_datetime'] = thai_dts
        combined['thai_date'] = thai_dts.dt.strftime('%Y-%m-%d')
        combined['thai_time'] = thai_dts.dt.strftime('%H:%M')
        combined['thai_time_display'] = thai_dts.dt.strftime('%d/%m/%Y เวลา %H:%M น.')

        matching = combined[thai_dts.dt.date == target_date].copy()
        if not matching.empty:
            return matching.sort_values('thai_datetime').reset_index(drop=True)

        return pd.DataFrame(columns=cols)

    def get_scoreboard_range(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch scoreboard for a date range.
        Uses ESPN's dynamic calendar to skip dates with no scheduled games (e.g. offseason).
        Args:
            start_date: YYYYMMDD
            end_date: YYYYMMDD
        Returns combined DataFrame of all games in range.
        """
        try:
            start = datetime.strptime(start_date, '%Y%m%d')
            end = datetime.strptime(end_date, '%Y%m%d')

            valid_dates = set()

            def fetch_calendar(d_str):
                try:
                    url = f"{self.BASE_URL}/scoreboard"
                    data = self._get(url, params={'dates': d_str})
                    if data:
                        leagues = data.get('leagues', [])
                        cal = leagues[0].get('calendar', []) if leagues else []
                        for c in cal:
                            valid_dates.add(c[:10].replace('-', ''))
                except Exception as ex:
                    logger.warning(f"Failed to fetch calendar for {d_str}: {ex}")

            fetch_calendar(start_date)
            if end_date not in valid_dates:
                fetch_calendar(end_date)

            all_games = []
            current = start
            while current <= end:
                date_str = current.strftime('%Y%m%d')
                if valid_dates and date_str not in valid_dates:
                    current += timedelta(days=1)
                    continue

                df = self.get_scoreboard(date=date_str)
                if not df.empty:
                    all_games.append(df)
                current += timedelta(days=1)
                time.sleep(0.1)

            if all_games:
                combined = pd.concat(all_games, ignore_index=True)
                if 'game_id' in combined.columns:
                    combined = combined.drop_duplicates(subset=['game_id'], keep='first')
                if not combined.empty:
                    return combined
            return self._get_fallback_scoreboard()
        except Exception as e:
            logger.warning(f"Error in get_scoreboard_range: {e}; activating fallback data.")
            return self._get_fallback_scoreboard()

    def get_recent_games(self, days: int = 180) -> pd.DataFrame:
        """Fetch recent games for the past N days."""
        end = datetime.now()
        start = end - timedelta(days=days)
        return self.get_scoreboard_range(start.strftime('%Y%m%d'), end.strftime('%Y%m%d'))

    def get_game_summary(self, game_id: str) -> Dict[str, Any]:
        """Fetch detailed game summary including boxscore, odds, win probability.
        Returns dict with keys: 'boxscore', 'odds', 'game_info', 'leaders'
        """
        gid_str = str(game_id)
        now = time.time()
        if hasattr(self, '_summary_cache') and gid_str in self._summary_cache:
            cached_data, cached_time = self._summary_cache[gid_str]
            if now - cached_time < 10:
                return cached_data

        url = f"{self.BASE_URL}/summary"
        params = {'event': game_id}
        try:
            data = self._get(url, params=params)
            if not data:
                logger.warning(f"Failed to fetch game summary for {game_id}; returning fallback dict.")
                return {
                    'boxscore': {},
                    'odds': [],
                    'game_info': {'id': str(game_id)},
                    'leaders': []
                }

            res = {
                'boxscore': data.get('boxscore', {}),
                'odds': data.get('pickcenter', []),
                'game_info': data.get('gameInfo', {}),
                'leaders': data.get('leaders', [])
            }
            if hasattr(self, '_summary_cache'):
                self._summary_cache[gid_str] = (res, now)
            return res
        except Exception as e:
            logger.warning(f"Error fetching game summary for {game_id}: {e}; returning fallback dict.")
            return {
                'boxscore': {},
                'odds': [],
                'game_info': {'id': str(game_id)},
                'leaders': []
            }

    def get_game_line_movement(self, game_id: str) -> Dict[str, Any]:
        """Extract open and current/close line movement data from game summary pickcenter.

        Returns dict with:
          - open_total: float or None
          - current_total: float or None
          - open_spread: float or None
          - current_spread: float or None
          - provider: str
        """
        if not game_id:
            return {'open_total': None, 'current_total': None, 'live_total': None, 'open_spread': None, 'current_spread': None, 'live_spread': None, 'provider': 'N/A'}

        summary = self.get_game_summary(str(game_id))
        odds_list = summary.get('odds', [])
        if not odds_list:
            return {'open_total': None, 'current_total': None, 'live_total': None, 'open_spread': None, 'current_spread': None, 'live_spread': None, 'provider': 'N/A'}

        item = odds_list[0]
        total_dict = item.get('total', {})
        spread_dict = item.get('pointSpread', {})

        def _clean_num(val):
            if val is None:
                return None
            s = str(val).strip()
            # Remove leading o, u, +, etc.
            s = re.sub(r'^[ouOU\+]+', '', s)
            try:
                return float(s)
            except (ValueError, TypeError):
                return None

        open_t = _clean_num(total_dict.get('over', {}).get('open', {}).get('line'))
        curr_t = _clean_num(total_dict.get('over', {}).get('close', {}).get('line')) or _clean_num(item.get('overUnder'))
        live_t = _clean_num(total_dict.get('over', {}).get('live', {}).get('line'))

        open_s = _clean_num(spread_dict.get('home', {}).get('open', {}).get('line'))
        curr_s = _clean_num(spread_dict.get('home', {}).get('close', {}).get('line')) or _clean_num(item.get('spread'))
        live_s = _clean_num(spread_dict.get('home', {}).get('live', {}).get('line'))

        provider = item.get('provider', {}).get('name', 'Consensus')

        return {
            'open_total': open_t,
            'current_total': curr_t,
            'live_total': live_t,
            'open_spread': open_s,
            'current_spread': curr_s,
            'live_spread': live_s,
            'provider': provider
        }

    def get_standings(self) -> pd.DataFrame:
        """Fetch current NBA standings.
        Returns DataFrame with: team, conference, wins, losses, win_pct,
        games_behind, streak, home_record, away_record,
        points_for, points_against, point_diff
        """
        try:
            data = self._get(self.STANDINGS_URL)
            if data is None or not data:
                return self._load_fallback_csv('espn_standings.csv')

            standings_data = data.get('children', [])
            if not standings_data:
                return self._load_fallback_csv('espn_standings.csv')

            records = []
            for conf in standings_data:
                for team_standing in conf.get('standings', {}).get('entries', []):
                    team = team_standing.get('team', {})
                    stats = {s.get('name'): s.get('value') for s in team_standing.get('stats', [])}

                    records.append({
                        'team': team.get('displayName'),
                        'conference': conf.get('name'),
                        'wins': stats.get('wins'),
                        'losses': stats.get('losses'),
                        'win_pct': stats.get('winPercent'),
                        'games_behind': stats.get('gamesBehind'),
                        'streak': stats.get('streak'),
                        'home_record': f"{int(stats.get('homeWins') or 0)}-{int(stats.get('homeLosses') or 0)}",
                        'away_record': f"{int(stats.get('awayWins') or 0)}-{int(stats.get('awayLosses') or 0)}",
                        'points_for': stats.get('pointsFor'),
                        'points_against': stats.get('pointsAgainst'),
                        'point_diff': stats.get('pointDifferential')
                    })

            df = pd.DataFrame(records)
            if df.empty:
                return self._load_fallback_csv('espn_standings.csv')
            return df
        except Exception as e:
            logger.warning(f"Error fetching/parsing standings: {e}; activating fallback data.")
            return self._load_fallback_csv('espn_standings.csv')

    def _get_fallback_schedule(self, team_id: Optional[int] = None) -> pd.DataFrame:
        """Helper to build fallback schedule adhering to expected schema."""
        df = self._load_fallback_csv('espn_recent_games.csv')
        cols = ['game_id', 'date', 'home_away', 'opponent', 'result', 'score', 'opp_score', 'point_diff']
        if df.empty:
            return pd.DataFrame(columns=cols)

        matches = df
        if team_id is not None:
            team_matches = df[(df['home_id'].astype(str) == str(team_id)) | (df['away_id'].astype(str) == str(team_id))]
            if not team_matches.empty:
                matches = team_matches

        records = []
        for _, row in matches.iterrows():
            is_home = (str(row.get('home_id')) == str(team_id)) if team_id is not None else True
            home_s = row.get('home_score')
            away_s = row.get('away_score')
            score = home_s if is_home else away_s
            opp_score = away_s if is_home else home_s
            diff = None
            result = None
            if pd.notna(score) and pd.notna(opp_score):
                try:
                    s_val = float(score)
                    o_val = float(opp_score)
                    diff = s_val - o_val
                    result = 'W' if diff > 0 else ('L' if diff < 0 else None)
                except (ValueError, TypeError):
                    pass
            records.append({
                'game_id': row.get('game_id'),
                'date': row.get('date'),
                'home_away': 'home' if is_home else 'away',
                'opponent': row.get('away_team') if is_home else row.get('home_team'),
                'result': result,
                'score': score,
                'opp_score': opp_score,
                'point_diff': diff
            })
        return pd.DataFrame(records)

    def get_team_schedule(self, team_id: int) -> pd.DataFrame:
        """Fetch full season schedule + results for a team.
        Returns DataFrame with: game_id, date, home_away, opponent,
        result (W/L), score, opp_score, point_diff
        """
        url = f"{self.BASE_URL}/teams/{team_id}/schedule"
        try:
            data = self._get(url)
            if data is None or not data:
                return self._get_fallback_schedule(team_id)

            events = data.get('events', [])
            if not events:
                return self._get_fallback_schedule(team_id)

            records = []
            for event in events:
                game_id = event.get('id')
                date_str = event.get('date')

                competitions = event.get('competitions', [])
                comp = competitions[0] if competitions else {}
                competitors = comp.get('competitors', [])

                team_c = None
                opp_c = None

                for c in competitors:
                    if str(c.get('id')) == str(team_id):
                        team_c = c
                    else:
                        opp_c = c

                if not team_c or not opp_c:
                    continue

                home_away = team_c.get('homeAway')
                opponent = opp_c.get('team', {}).get('displayName')
                score = team_c.get('score', {}).get('value')
                opp_score = opp_c.get('score', {}).get('value')

                result = None
                point_diff = None
                if team_c.get('winner') is True:
                    result = 'W'
                elif team_c.get('winner') is False:
                    result = 'L'

                if score is not None and opp_score is not None:
                    point_diff = score - opp_score

                records.append({
                    'game_id': game_id,
                    'date': date_str,
                    'home_away': home_away,
                    'opponent': opponent,
                    'result': result,
                    'score': score,
                    'opp_score': opp_score,
                    'point_diff': point_diff
                })

            df = pd.DataFrame(records)
            if df.empty:
                return self._get_fallback_schedule(team_id)
            return df
        except Exception as e:
            logger.warning(f"Error fetching/parsing team schedule for {team_id}: {e}; activating fallback data.")
            return self._get_fallback_schedule(team_id)

    def get_todays_games_with_odds(self) -> pd.DataFrame:
        """Convenience method: get today's games with full odds info.
        Calls get_scoreboard() and formats nicely for analysis."""
        return self.get_scoreboard()

    def get_injuries(self) -> pd.DataFrame:
        """Fetch current NBA injury report from ESPN API.
        
        Returns DataFrame with columns:
        ['team_name', 'player_name', 'status', 'detail', 'side', 'type', 'return_date']
        """
        logger.info("Fetching NBA injury report from ESPN...")
        url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"

        try:
            data = self._get(url)
            if not data or 'injuries' not in data:
                return self._load_fallback_csv('espn_injuries.csv')

            records = []
            for team in data.get('injuries', []):
                team_name = team.get('displayName', '')
                for inj in team.get('injuries', []):
                    athlete = inj.get('athlete', {})
                    player_name = athlete.get('displayName', athlete.get('fullName', ''))
                    status = inj.get('status', '')
                    details = inj.get('details', {})
                    injury_type = details.get('type', '')
                    side = details.get('side', '')
                    return_date = details.get('returnDate', '')
                    detail = details.get('detail', '')
                    if not detail and inj.get('notes', {}).get('items'):
                        detail = inj['notes']['items'][0].get('headline', '')
                    
                    records.append({
                        'team_name': team_name,
                        'player_name': player_name,
                        'status': status,
                        'detail': detail,
                        'side': side,
                        'type': injury_type,
                        'return_date': return_date
                    })

            df = pd.DataFrame(records)
            if not df.empty:
                raw_dir = self._resolve_raw_dir()
                cache_path = os.path.join(raw_dir, 'espn_injuries.csv')
                try:
                    os.makedirs(raw_dir, exist_ok=True)
                    df.to_csv(cache_path, index=False)
                except Exception:
                    pass
                return df
            return self._load_fallback_csv('espn_injuries.csv')
        except Exception as e:
            logger.warning(f"Error fetching injuries from ESPN: {e}; activating fallback data.")
            return self._load_fallback_csv('espn_injuries.csv')

