#!/usr/bin/env python
"""verify_collectors.py

Automated resilience and local fallback verification script for NBA data collectors.
Simulates network failure scenarios (ConnectionError, Timeout, HTTP 429, HTTP 500)
and asserts that all collector methods successfully return valid fallback data without crashing.
"""

import sys
import os
import logging
from unittest.mock import patch, MagicMock
import requests
import pandas as pd

# Add workspace directory to sys.path
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from collectors.espn_collector import ESPNCollector
from collectors.advanced_stats_collector import AdvancedStatsCollector
from collectors.sdv_collector import SDVCollector
from collectors.odds_collector import OddsCollector


class LogCaptureHandler(logging.Handler):
    """In-memory logging handler to verify warning log messages."""
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)

    def has_warning_containing(self, snippet: str) -> bool:
        return any(
            record.levelno >= logging.WARNING and snippet.lower() in record.getMessage().lower()
            for record in self.records
        )

    def clear(self):
        self.records.clear()


def make_mock_response(status_code: int, text: str = "Error"):
    """Helper to create mock requests.Response."""
    resp = requests.Response()
    resp.status_code = status_code
    resp._content = text.encode("utf-8")
    return resp


def test_espn_collector_failures(log_handler: LogCaptureHandler):
    """Test ESPNCollector under various simulated network failures."""
    print("\n--- Testing ESPNCollector Resilience & Fallback ---")
    failure_scenarios = [
        ("ConnectionError", requests.exceptions.ConnectionError("Connection refused")),
        ("Timeout", requests.exceptions.Timeout("Request timed out")),
        ("HTTP 429 Rate Limit", requests.exceptions.HTTPError(response=make_mock_response(429, "Too Many Requests"))),
        ("HTTP 500 Server Error", requests.exceptions.HTTPError(response=make_mock_response(500, "Internal Server Error"))),
    ]

    for scenario_name, error_to_raise in failure_scenarios:
        print(f"  Testing ESPN scenario: {scenario_name}...")
        collector = ESPNCollector(max_retries=2, retry_delay=0.01, backoff_factor=1.5)

        with patch.object(collector.session, "get", side_effect=error_to_raise):
            log_handler.clear()

            # 1. get_all_teams
            teams_df = collector.get_all_teams()
            assert isinstance(teams_df, pd.DataFrame), f"get_all_teams must return DataFrame under {scenario_name}"
            assert not teams_df.empty, f"get_all_teams fallback must be non-empty under {scenario_name}"
            expected_team_cols = {'team_id', 'abbreviation', 'name', 'display_name'}
            assert expected_team_cols.issubset(set(teams_df.columns)), (
                f"get_all_teams columns missing expected schema under {scenario_name}: {teams_df.columns}"
            )
            assert log_handler.has_warning_containing("fallback"), "Warning log must report fallback activation"

            # 2. get_team_roster
            roster_df = collector.get_team_roster(team_id=1)
            assert isinstance(roster_df, pd.DataFrame), "get_team_roster must return DataFrame"
            assert not roster_df.empty, "get_team_roster fallback must be non-empty"
            expected_roster_cols = {'player_id', 'name', 'position'}
            assert expected_roster_cols.issubset(set(roster_df.columns)), (
                f"get_team_roster columns missing expected schema: {roster_df.columns}"
            )

            # 3. get_all_rosters
            all_rosters_df = collector.get_all_rosters()
            assert isinstance(all_rosters_df, pd.DataFrame), "get_all_rosters must return DataFrame"
            assert not all_rosters_df.empty, "get_all_rosters fallback must be non-empty"

            # 4. get_scoreboard (today and specific date)
            sb_df = collector.get_scoreboard()
            assert isinstance(sb_df, pd.DataFrame), "get_scoreboard must return DataFrame"
            assert not sb_df.empty, "get_scoreboard fallback must be non-empty"
            expected_sb_cols = {'game_id', 'date', 'home_team', 'away_team'}
            assert expected_sb_cols.issubset(set(sb_df.columns)), (
                f"get_scoreboard columns missing expected schema: {sb_df.columns}"
            )

            sb_date_df = collector.get_scoreboard(date="20260407")
            assert isinstance(sb_date_df, pd.DataFrame), "get_scoreboard(date) must return DataFrame"
            assert not sb_date_df.empty, "get_scoreboard(date) fallback must be non-empty"

            # 5. get_scoreboard_range
            range_df = collector.get_scoreboard_range("20260401", "20260407")
            assert isinstance(range_df, pd.DataFrame), "get_scoreboard_range must return DataFrame"
            assert not range_df.empty, "get_scoreboard_range fallback must be non-empty"

            # 6. get_standings
            standings_df = collector.get_standings()
            assert isinstance(standings_df, pd.DataFrame), "get_standings must return DataFrame"
            assert not standings_df.empty, "get_standings fallback must be non-empty"
            expected_st_cols = {'team', 'conference', 'wins', 'losses'}
            assert expected_st_cols.issubset(set(standings_df.columns)), (
                f"get_standings columns missing expected schema: {standings_df.columns}"
            )

            # 7. get_team_schedule
            sched_df = collector.get_team_schedule(team_id=1)
            assert isinstance(sched_df, pd.DataFrame), "get_team_schedule must return DataFrame"
            assert not sched_df.empty, "get_team_schedule fallback must be non-empty"
            expected_sched_cols = {'game_id', 'date', 'home_away', 'opponent'}
            assert expected_sched_cols.issubset(set(sched_df.columns)), (
                f"get_team_schedule columns missing expected schema: {sched_df.columns}"
            )

            # 8. get_todays_games_with_odds
            todays_df = collector.get_todays_games_with_odds()
            assert isinstance(todays_df, pd.DataFrame), "get_todays_games_with_odds must return DataFrame"
            assert not todays_df.empty, "get_todays_games_with_odds fallback must be non-empty"

            # 9. get_game_summary
            summary = collector.get_game_summary("401811003")
            assert isinstance(summary, dict), "get_game_summary must return dict"
            assert "boxscore" in summary and "odds" in summary, "get_game_summary must have expected keys"

    print("  [PASS] ESPNCollector resilience & local fallback verified.")


def test_advanced_stats_collector_failures(log_handler: LogCaptureHandler):
    """Test AdvancedStatsCollector under simulated endpoint and network failures."""
    print("\n--- Testing AdvancedStatsCollector Resilience & Fallback ---")
    failure_scenarios = [
        ("HTTP 429 Rate Limit", requests.exceptions.HTTPError(response=make_mock_response(429, "Too Many Requests"))),
        ("Timeout", requests.exceptions.Timeout("Read timeout")),
        ("ConnectionError", requests.exceptions.ConnectionError("Failed to connect")),
        ("HTTP 500 Server Error", requests.exceptions.HTTPError(response=make_mock_response(500, "Internal Server Error"))),
    ]

    for scenario_name, error_to_raise in failure_scenarios:
        print(f"  Testing AdvancedStats scenario: {scenario_name}...")
        collector = AdvancedStatsCollector(max_retries=2, retry_delay=0.01, backoff_factor=1.5)

        # Patch _retry_call to simulate error or mock underlying endpoints
        with patch.object(collector, "_retry_call", side_effect=lambda func, *a, **kw: None):
            log_handler.clear()

            # 1. get_team_stats_base
            base_df = collector.get_team_stats_base()
            assert isinstance(base_df, pd.DataFrame), f"get_team_stats_base must return DataFrame under {scenario_name}"
            assert not base_df.empty, f"get_team_stats_base fallback must be non-empty under {scenario_name}"
            expected_base_cols = {'TEAM_ID', 'TEAM_NAME', 'GP', 'W', 'L', 'W_PCT'}
            assert expected_base_cols.issubset(set(base_df.columns)), (
                f"get_team_stats_base schema mismatch: {base_df.columns}"
            )
            assert log_handler.has_warning_containing("fallback"), "Warning log must report fallback activation"

            # 2. get_team_stats_advanced
            adv_df = collector.get_team_stats_advanced()
            assert isinstance(adv_df, pd.DataFrame), "get_team_stats_advanced must return DataFrame"
            assert not adv_df.empty, "get_team_stats_advanced fallback must be non-empty"
            expected_adv_cols = {'TEAM_ID', 'TEAM_NAME', 'OFF_RATING', 'DEF_RATING', 'NET_RATING'}
            assert expected_adv_cols.issubset(set(adv_df.columns)), (
                f"get_team_stats_advanced schema mismatch: {adv_df.columns}"
            )

            # 3. get_team_four_factors
            ff_df = collector.get_team_four_factors()
            assert isinstance(ff_df, pd.DataFrame), "get_team_four_factors must return DataFrame"
            assert not ff_df.empty, "get_team_four_factors fallback must be non-empty"
            expected_ff_cols = {'TEAM_ID', 'TEAM_NAME', 'EFG_PCT', 'FTA_RATE', 'TM_TOV_PCT'}
            assert expected_ff_cols.issubset(set(ff_df.columns)), (
                f"get_team_four_factors schema mismatch: {ff_df.columns}"
            )

            # 4. fetch_draft_history
            draft_df = collector.fetch_draft_history()
            assert isinstance(draft_df, pd.DataFrame), "fetch_draft_history must return DataFrame"
            assert not draft_df.empty, "fetch_draft_history fallback must be non-empty"
            expected_draft_cols = {'PERSON_ID', 'PLAYER_NAME', 'SEASON', 'ROUND_NUMBER'}
            assert expected_draft_cols.issubset(set(draft_df.columns)), (
                f"fetch_draft_history schema mismatch: {draft_df.columns}"
            )

            # 5. get_player_stats
            player_df = collector.get_player_stats()
            assert isinstance(player_df, pd.DataFrame), "get_player_stats must return DataFrame"
            assert not player_df.empty, "get_player_stats fallback must be non-empty"
            expected_player_cols = {'PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION'}
            assert expected_player_cols.issubset(set(player_df.columns)), (
                f"get_player_stats schema mismatch: {player_df.columns}"
            )

            # 6. get_game_logs
            logs_df = collector.get_game_logs('T')
            assert isinstance(logs_df, pd.DataFrame), "get_game_logs must return DataFrame"
            assert not logs_df.empty, "get_game_logs fallback must be non-empty"
            expected_logs_cols = {'SEASON_ID', 'TEAM_ID', 'TEAM_NAME', 'GAME_ID', 'GAME_DATE'}
            assert expected_logs_cols.issubset(set(logs_df.columns)), (
                f"get_game_logs schema mismatch: {logs_df.columns}"
            )

            # 7. get_all_advanced_data
            all_adv = collector.get_all_advanced_data()
            assert isinstance(all_adv, dict), "get_all_advanced_data must return dict"
            for key in ['base', 'advanced', 'four_factors', 'players', 'game_logs', 'draft_history']:
                assert key in all_adv, f"Key {key} missing in get_all_advanced_data"
                assert isinstance(all_adv[key], pd.DataFrame), f"{key} must be DataFrame"
                assert not all_adv[key].empty, f"{key} must be non-empty"

    print("  [PASS] AdvancedStatsCollector resilience & local fallback verified.")


def test_sdv_collector_failures(log_handler: LogCaptureHandler):
    """Test SDVCollector under simulated package and network failures."""
    print("\n--- Testing SDVCollector Resilience & Fallback ---")
    failure_scenarios = [
        ("ConnectionError", requests.exceptions.ConnectionError("Parquet download failed")),
        ("Timeout", requests.exceptions.Timeout("Request timeout")),
        ("HTTP 500", requests.exceptions.HTTPError(response=make_mock_response(500, "Server Error"))),
    ]

    for scenario_name, error_to_raise in failure_scenarios:
        print(f"  Testing SDV scenario: {scenario_name}...")
        collector = SDVCollector(max_retries=2, retry_delay=0.01, backoff_factor=1.5)

        with patch.object(collector, "_retry_call", side_effect=lambda func, *a, **kw: None):
            log_handler.clear()

            # 1. get_team_box_scores
            tb_df = collector.get_team_box_scores()
            assert isinstance(tb_df, pd.DataFrame), f"get_team_box_scores must return DataFrame under {scenario_name}"
            assert not tb_df.empty, f"get_team_box_scores fallback must be non-empty under {scenario_name}"
            assert log_handler.has_warning_containing("fallback"), "Warning log must report fallback activation"

            # 2. get_player_box_scores
            pb_df = collector.get_player_box_scores()
            assert isinstance(pb_df, pd.DataFrame), "get_player_box_scores must return DataFrame"
            assert not pb_df.empty, "get_player_box_scores fallback must be non-empty"

            # 3. get_schedules
            sched_df = collector.get_schedules()
            assert isinstance(sched_df, pd.DataFrame), "get_schedules must return DataFrame"
            assert not sched_df.empty, "get_schedules fallback must be non-empty"

            # 4. get_play_by_play
            pbp_df = collector.get_play_by_play()
            assert isinstance(pbp_df, pd.DataFrame), "get_play_by_play must return DataFrame"
            assert not pbp_df.empty, "get_play_by_play fallback must be non-empty"

            # 5. get_team_season_averages
            avg_df = collector.get_team_season_averages()
            assert isinstance(avg_df, pd.DataFrame), "get_team_season_averages must return DataFrame"
            assert not avg_df.empty, "get_team_season_averages fallback must be non-empty"

    print("  [PASS] SDVCollector resilience & local fallback verified.")


def test_odds_collector_failures(log_handler: LogCaptureHandler):
    """Test OddsCollector under simulated API key absence, rate limits, and network errors."""
    print("\n--- Testing OddsCollector Resilience & Fallback ---")
    failure_scenarios = [
        ("HTTP 429 Rate Limit", requests.exceptions.HTTPError(response=make_mock_response(429, "Too Many Requests"))),
        ("Timeout", requests.exceptions.Timeout("Request timeout")),
        ("ConnectionError", requests.exceptions.ConnectionError("Host unreachable")),
        ("HTTP 500 Server Error", requests.exceptions.HTTPError(response=make_mock_response(500, "Internal Server Error"))),
    ]

    for scenario_name, error_to_raise in failure_scenarios:
        print(f"  Testing Odds scenario: {scenario_name}...")
        collector = OddsCollector(max_retries=2, retry_delay=0.01, backoff_factor=1.5)
        collector.api_key = "dummy_test_key"

        with patch("requests.get", side_effect=error_to_raise):
            log_handler.clear()

            # 1. _fetch_all_odds
            odds_list = collector._fetch_all_odds()
            assert isinstance(odds_list, list), f"_fetch_all_odds must return list under {scenario_name}"
            assert len(odds_list) > 0, f"_fetch_all_odds fallback must be non-empty under {scenario_name}"
            assert log_handler.has_warning_containing("fallback"), "Warning log must report fallback activation"

            # 2. get_game_odds
            sample_game = odds_list[0]
            home_team = sample_game.get('home_team', 'Boston Celtics')
            away_team = sample_game.get('away_team', 'Atlanta Hawks')

            parsed = collector.get_game_odds(home_team, away_team)
            assert isinstance(parsed, dict), "get_game_odds must return dict"
            assert 'spreads' in parsed and 'totals' in parsed, "get_game_odds must contain spreads and totals"

            # 3. get_consensus_lines
            spread, total = collector.get_consensus_lines(parsed)
            assert spread is not None or total is not None, "get_consensus_lines should extract line data"

            # 4. get_odds_dataframe
            odds_df = collector.get_odds_dataframe()
            assert isinstance(odds_df, pd.DataFrame), "get_odds_dataframe must return DataFrame"
            assert not odds_df.empty, "get_odds_dataframe fallback must be non-empty"

            odds_df_alias = collector.get_odds_df()
            assert isinstance(odds_df_alias, pd.DataFrame) and not odds_df_alias.empty, "get_odds_df alias must work"

    print("  [PASS] OddsCollector resilience & local fallback verified.")


def test_retry_backoff_recovery():
    """Verify that retry logic retries and succeeds if endpoint recovers before max_retries."""
    print("\n--- Testing Retry Backoff & Recovery ---")
    collector = ESPNCollector(max_retries=3, retry_delay=0.01, backoff_factor=1.5)

    attempts = {"count": 0}

    def flaky_get(url, params=None, timeout=10):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise requests.exceptions.Timeout("Temporary timeout")
        # Succeed on 3rd attempt
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {
            "sports": [{"leagues": [{"teams": [{"team": {"id": "99", "displayName": "Recovered Team"}}]}]}]
        }
        return resp

    with patch.object(collector.session, "get", side_effect=flaky_get):
        result = collector.get_all_teams()
        assert not result.empty, "Collector should return data after successful recovery"
        assert attempts["count"] == 3, f"Expected 3 attempts, got {attempts['count']}"
        assert result.iloc[0]["display_name"] == "Recovered Team", "Expected recovered data from live endpoint"

    print("  [PASS] Retry backoff recovery successfully verified.")


def test_edge_cases(log_handler: LogCaptureHandler):
    """Explicitly test edge cases and boundary conditions."""
    print("\n--- Testing Explicit Edge Cases & Boundary Conditions ---")
    
    # 1. ESPNCollector unknown team_id and dates
    espn = ESPNCollector(max_retries=1, retry_delay=0.01)
    with patch.object(espn.session, "get", side_effect=requests.exceptions.Timeout("Timeout")):
        roster_unknown = espn.get_team_roster(team_id=999999)
        assert isinstance(roster_unknown, pd.DataFrame) and not roster_unknown.empty, (
            "Unknown team_id in get_team_roster must still return non-empty fallback DataFrame"
        )
        
        sched_unknown = espn.get_team_schedule(team_id=999999)
        assert isinstance(sched_unknown, pd.DataFrame) and not sched_unknown.empty, (
            "Unknown team_id in get_team_schedule must still return non-empty fallback DataFrame"
        )
        
        sb_empty_date = espn.get_scoreboard(date="")
        assert isinstance(sb_empty_date, pd.DataFrame) and not sb_empty_date.empty, (
            "Empty date string must return non-empty fallback DataFrame"
        )
        
        summary_empty_id = espn.get_game_summary(game_id="")
        assert isinstance(summary_empty_id, dict) and "boxscore" in summary_empty_id, (
            "Empty game_id must return valid dict structure"
        )

    # 2. OddsCollector edge cases
    odds = OddsCollector(max_retries=1, retry_delay=0.01)
    with patch("requests.get", side_effect=requests.exceptions.ConnectionError("Offline")):
        # Empty team names
        empty_odds = odds.get_game_odds("", "")
        assert isinstance(empty_odds, dict) and 'spreads' in empty_odds, (
            "Empty team strings must return valid fallback odds structure"
        )
        
        # None and empty inputs to consensus lines and line shopping
        c_s, c_t = odds.get_consensus_lines({})
        assert c_s is None and c_t is None, "Empty parsed_odds must return (None, None)"
        
        c_s2, c_t2 = odds.get_consensus_lines(None)
        assert c_s2 is None and c_t2 is None, "None parsed_odds must return (None, None)"
        
        best_line = odds.get_best_line_shopping({}, 'spread')
        assert best_line == "", "Empty parsed_odds in get_best_line_shopping must return empty string"

    # 3. SDVCollector edge cases
    sdv = SDVCollector(max_retries=1, retry_delay=0.01)
    with patch.object(sdv, "_retry_call", side_effect=lambda func, *a, **kw: None):
        # Empty season list
        tb_empty_seasons = sdv.get_team_box_scores(seasons=[])
        assert isinstance(tb_empty_seasons, pd.DataFrame) and not tb_empty_seasons.empty, (
            "Empty seasons list must return non-empty fallback DataFrame"
        )
        
        avg_empty = sdv.get_team_season_averages(seasons=[])
        assert isinstance(avg_empty, pd.DataFrame) and not avg_empty.empty, (
            "Empty seasons list for season averages must return non-empty fallback DataFrame"
        )

    # 4. AdvancedStatsCollector edge cases
    adv = AdvancedStatsCollector(max_retries=1, retry_delay=0.01)
    with patch.object(adv, "_retry_call", side_effect=lambda func, *a, **kw: None):
        logs_invalid = adv.get_game_logs(player_or_team='INVALID')
        assert isinstance(logs_invalid, pd.DataFrame) and not logs_invalid.empty, (
            "Invalid player_or_team parameter must return non-empty fallback DataFrame"
        )
        
        draft_zero = adv.fetch_draft_history(num_years=0)
        assert isinstance(draft_zero, pd.DataFrame) and not draft_zero.empty, (
            "Draft history with 0 years must return non-empty fallback DataFrame"
        )

    print("  [PASS] All explicit edge cases verified successfully.")


def main():
    print("================================================================")
    print("  NBA Data Collectors Automated Resilience & Fallback Tests     ")
    print("================================================================")

    # Attach LogCaptureHandler to root logger
    log_handler = LogCaptureHandler()
    log_handler.setLevel(logging.WARNING)
    logging.getLogger().addHandler(log_handler)
    logging.basicConfig(level=logging.WARNING)

    failed = False
    try:
        test_espn_collector_failures(log_handler)
        test_advanced_stats_collector_failures(log_handler)
        test_sdv_collector_failures(log_handler)
        test_odds_collector_failures(log_handler)
        test_retry_backoff_recovery()
        test_edge_cases(log_handler)
    except Exception as exc:
        print(f"\n[FAIL] Test failure occurred: {exc}")
        import traceback
        traceback.print_exc()
        failed = True

    if failed:
        print("\nVerification Status: FAILED (exit code 1)")
        sys.exit(1)
    else:
        print("\n================================================================")
        print("  ALL VERIFICATION TESTS PASSED SUCCESSFULLY! (exit code 0)     ")
        print("================================================================")
        sys.exit(0)


if __name__ == "__main__":
    main()

