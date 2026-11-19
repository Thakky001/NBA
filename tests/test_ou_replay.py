"""Tests for the O/U replay harness.

The harness is only useful if its arithmetic is boring: signs are the part most
likely to be wrong, and a wrong sign would flatter every variant equally.
"""

import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

from analysis.ou_replay import (
    closing_line_value,
    decide,
    early_season_shrinkage,
    games_played_as_of,
    implied_market_weight,
    main,
    oliver_raw_total,
    plays_impossible_count,
    rating_edge_delta,
    ratings_for,
    raw_projection,
    replay,
    reprice,
    resolve_games_played,
    summarize,
    us_game_date,
)


class TestClosingLineValue(unittest.TestCase):
    def test_over_gains_when_the_close_falls(self):
        # Bet OVER 220.5, closed 219.5 -> the market made our number better,
        # which means we took the worse side of it.
        self.assertAlmostEqual(closing_line_value("OVER", 220.5, 219.5), -1.0)
        self.assertAlmostEqual(closing_line_value("OVER", 219.5, 220.5), +1.0)

    def test_under_gains_when_the_close_falls(self):
        # For UNDER a higher line is better, so CLV is bet_line - close_line.
        # Betting UNDER 221.5 and closing 220.5 means we held the better number.
        self.assertAlmostEqual(closing_line_value("UNDER", 221.5, 220.5), +1.0)
        # Betting UNDER 220.5 and closing 221.5 means we took the worse one.
        self.assertAlmostEqual(closing_line_value("UNDER", 220.5, 221.5), -1.0)

    def test_unchanged_line_is_zero(self):
        self.assertEqual(closing_line_value("OVER", 220.5, 220.5), 0.0)
        self.assertEqual(closing_line_value("UNDER", 220.5, 220.5), 0.0)

    def test_missing_or_neutral_side_is_nan(self):
        self.assertTrue(np.isnan(closing_line_value("OVER", None, 220.5)))
        self.assertTrue(np.isnan(closing_line_value("OVER", 220.5, np.nan)))
        self.assertTrue(np.isnan(closing_line_value(None, 220.5, 221.5)))


class TestRepricing(unittest.TestCase):
    def test_raw_projection_inverts_the_anchor(self):
        line, edge, weight = 225.0, 3.0, 0.65
        raw = raw_projection(line, edge, weight)
        # Re-anchoring at the same weight must return the original edge.
        self.assertAlmostEqual((raw - line) * (1.0 - weight), edge)

    def test_reprice_is_identity_when_nothing_changes(self):
        edge = reprice(225.0, 3.0, stored_weight=0.65, new_weight=0.65)
        self.assertAlmostEqual(edge, 3.0, places=6)

    def test_more_market_weight_shrinks_the_edge(self):
        edge = reprice(225.0, 3.0, stored_weight=0.25, new_weight=0.65)
        # 3.0 was 75% of the raw gap; 35% of the same gap is 1.4.
        self.assertAlmostEqual(edge, 1.4, places=2)

    def test_extra_shrinkage_pulls_the_total_down(self):
        edge = reprice(225.0, 3.0, stored_weight=0.25, stored_shrinkage=0.0, new_shrinkage=2.0)
        # Raw gap 4.0 points, minus 2.0 of shrinkage, then 75% weight.
        self.assertAlmostEqual(edge, 1.5, places=2)

    def test_raw_shift_moves_the_projection_before_anchoring(self):
        # A 1.0 point injury term at 75% weight is worth 0.75 of edge.
        edge = reprice(225.0, 3.0, stored_weight=0.25, raw_shift=-1.0)
        self.assertAlmostEqual(edge, 2.25, places=2)

    def test_full_market_weight_collapses_the_edge(self):
        self.assertEqual(raw_projection(225.0, 5.0, 1.0), 225.0)


class TestEarlySeasonShrinkage(unittest.TestCase):
    def test_zero_once_the_sample_is_mature(self):
        self.assertEqual(early_season_shrinkage(20), 0.0)
        self.assertEqual(early_season_shrinkage(45), 0.0)
        self.assertEqual(early_season_shrinkage(None), 0.0)

    def test_caps_at_the_opening_night_value(self):
        self.assertAlmostEqual(early_season_shrinkage(0), 2.5)
        self.assertAlmostEqual(early_season_shrinkage(10), 1.25)


class TestGamesPlayed(unittest.TestCase):
    def setUp(self):
        self.logs = pd.DataFrame({
            "TEAM": ["Boston Celtics"] * 4,
            "GAME_ID": ["1", "2", "3", "4"],
            "GAME_DATE": pd.to_datetime(["2026-10-20", "2026-10-22", "2026-11-10", "2026-11-18"]),
            "FGA": [90, 90, 90, 90], "FTA": [20, 20, 20, 20],
            "OREB": [10, 10, 10, 10], "TOV": [12, 12, 12, 12], "PTS": [100, 100, 100, 100],
        })

    def test_counts_only_games_before_the_us_game_date(self):
        # The helper takes a US game date, so a game on 2026-11-10 counts
        # everything before it and not itself.
        self.assertEqual(games_played_as_of("Boston Celtics", "2026-11-10", self.logs), 2)
        self.assertEqual(games_played_as_of("Boston Celtics", "2026-11-11", self.logs), 3)

    def test_ledger_stamp_maps_to_the_previous_us_date(self):
        # Rows are written in Bangkok the morning after the US slate.
        self.assertEqual(us_game_date("2026-11-19"), pd.Timestamp("2026-11-18"))
        self.assertEqual(us_game_date(pd.Timestamp("2026-10-21 00:00")), pd.Timestamp("2026-10-20"))

    def test_unknown_team_returns_zero_not_none(self):
        self.assertEqual(games_played_as_of("Nowhere Kings", "2026-11-10", self.logs), 0)

    def test_impossible_counts_are_flagged(self):
        # 45 games cannot exist six days into the season.
        self.assertTrue(plays_impossible_count(45, "2026-10-26"))
        self.assertFalse(plays_impossible_count(5, "2026-10-26"))
        self.assertFalse(plays_impossible_count(None, "2026-10-26"))

    def test_impossible_count_is_replaced_by_the_real_one(self):
        row = pd.Series({
            "min_games": 46.0, "date": pd.Timestamp("2026-10-23"),
            "home_team": "Boston Celtics", "away_team": "Boston Celtics",
        })
        # The stamp 10-23 prices the US game on 10-22, which is after only one
        # completed Boston game (10-20).
        self.assertEqual(resolve_games_played(row, self.logs), 1.0)

    def test_plausible_count_is_left_alone(self):
        # The live app may lag a slate behind; that is its own input, not an error.
        row = pd.Series({
            "min_games": 2.0, "date": pd.Timestamp("2026-10-23"),
            "home_team": "Boston Celtics", "away_team": "Boston Celtics",
        })
        # Two games by the US date 10-22 is possible, so it is not rewritten.
        self.assertEqual(resolve_games_played(row, self.logs), 2.0)


class TestGate(unittest.TestCase):
    def test_large_edge_on_a_mature_sample_is_strong(self):
        decision = decide(225.0, 6.0, min_games=20.0)
        self.assertEqual(decision["recommendation"], "STRONG OVER")

    def test_big_negative_edge_is_strong_under(self):
        decision = decide(225.0, -4.0, min_games=20.0)
        self.assertEqual(decision["recommendation"], "STRONG UNDER")

    def test_strong_is_downgraded_below_the_sample_floor(self):
        decision = decide(225.0, 6.0, min_games=10.0)
        self.assertEqual(decision["recommendation"], "LEAN OVER")
        self.assertEqual(decision["suppressed_reason"], "EARLY_SEASON_SAMPLE_RESTRICTION")

    def test_tiny_sample_is_vetoed_outright(self):
        decision = decide(225.0, 6.0, min_games=3.0)
        self.assertEqual(decision["recommendation"], "NEUTRAL")
        self.assertEqual(decision["suppressed_reason"], "VETO_LOW_SAMPLE_SIZE")

    def test_anomalous_edge_is_vetoed(self):
        decision = decide(225.0, 12.0, min_games=20.0)
        self.assertEqual(decision["recommendation"], "NEUTRAL")
        self.assertEqual(decision["suppressed_reason"], "EXTREME_EDGE_MODEL_ANOMALY")

    def test_probability_is_not_the_raw_gaussian(self):
        # Calibration must pull conviction toward 0.5, otherwise the gate is
        # being run on an uncalibrated number.
        decision = decide(225.0, 5.0, min_games=20.0)
        self.assertLess(decision["prob_over"], 0.65)


def _frame(rows):
    base = {
        "date": pd.Timestamp("2026-11-10"),
        "game_id": "1",
        "total_line": 225.0,
        "ou_edge": 3.0,
        "min_games": 20.0,
        "total_odds_parsed": None,
        "close_total": 225.0,
        "actual_total": 220.0,
        "ou_result": "LOSS",
    }
    return pd.DataFrame([{**base, **r} for r in rows])


class TestReplay(unittest.TestCase):
    def test_rows_the_variant_cannot_price_are_dropped(self):
        frame = _frame([{"game_id": "a"}, {"game_id": "b"}])
        out = replay(frame, edge_fn=lambda r: None if r["game_id"] == "a" else 3.0)
        self.assertEqual(list(out["game_id"]), ["b"])

    def test_identity_variant_keeps_the_stored_edge(self):
        frame = _frame([{}])
        out = replay(frame)
        self.assertAlmostEqual(out.loc[0, "edge"], 3.0)

    def test_clv_is_attached_per_pick(self):
        # OVER 225.0 closing at 226.0 means the market moved above our number.
        frame = _frame([{"ou_edge": 6.0, "total_line": 225.0, "close_total": 226.0}])
        out = replay(frame)
        self.assertEqual(out.loc[0, "pick_side"], "OVER")
        self.assertAlmostEqual(out.loc[0, "clv"], 1.0)

    def test_neutral_rows_have_no_clv(self):
        frame = _frame([{"ou_edge": 0.0}])
        out = replay(frame)
        self.assertIsNone(out.loc[0, "pick_side"])
        self.assertTrue(np.isnan(out.loc[0, "clv"]))


class TestSummary(unittest.TestCase):
    def test_roi_counts_only_decided_bets(self):
        picks = pd.DataFrame({
            "pick_side": ["OVER", "OVER", None],
            "ou_result": ["WIN", "LOSS", "SKIPPED"],
            "clv": [1.0, -1.0, np.nan],
        })
        summary = summarize(picks)
        self.assertEqual(summary["picks"], 2)
        self.assertAlmostEqual(summary["roi"], (100 / 110 - 1.0) / 2)
        self.assertEqual(summary["clv_n"], 2)

    def test_clv_interval_reports_whether_it_clears_zero(self):
        up = summarize(pd.DataFrame({
            "pick_side": ["OVER"] * 6,
            "ou_result": ["WIN"] * 6,
            "clv": [1.0, 1.0, 1.0, 1.0, 1.0, 2.0],
        }))
        self.assertTrue(up["clv_above_zero"])

        flat = summarize(pd.DataFrame({
            "pick_side": ["OVER"] * 6,
            "ou_result": ["WIN"] * 6,
            "clv": [1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
        }))
        self.assertFalse(flat["clv_above_zero"])


class TestCommandLine(unittest.TestCase):
    """A broken CLI is invisible until someone runs it, so run it here."""

    def test_prints_a_table_and_exits_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = os.path.join(tmp, "ledger.csv")
            closes = os.path.join(tmp, "closes.csv")
            pd.DataFrame([
                {
                    "game_id": "g1", "date": "2026-11-10", "status": "GRADED",
                    "home_team": "Boston Celtics", "away_team": "Miami Heat",
                    "total_line": 220.5, "ou_edge": 3.0, "projected_total": 223.5,
                    "actual_total": 218.0, "ou_result": "LOSS", "total_odds": "[-110, -110]",
                    "ou_recommendation": "LEAN OVER", "ou_win_prob": 0.55,
                    "features_snapshot": '{"home_games_played": 20, "away_games_played": 20}',
                },
                {
                    "game_id": "g2", "date": "2026-11-11", "status": "GRADED",
                    "home_team": "Utah Jazz", "away_team": "Orlando Magic",
                    "total_line": 225.0, "ou_edge": -4.0, "projected_total": 221.0,
                    "actual_total": 230.0, "ou_result": "WIN", "total_odds": "[-110, -110]",
                    "ou_recommendation": "STRONG UNDER", "ou_win_prob": 0.6,
                    "features_snapshot": '{"home_games_played": 20, "away_games_played": 20}',
                },
            ]).to_csv(ledger, index=False)
            pd.DataFrame({"game_id": ["g1", "g2"], "close_total": [221.5, 224.0],
                          "open_total": [220.5, 225.0], "close_spread": [1.0, -1.0]}).to_csv(closes, index=False)

            out = io.StringIO()
            with redirect_stdout(out):
                code = main(["--ledger", ledger, "--closes", closes])
            printed = out.getvalue()

        self.assertEqual(code, 0)
        self.assertIn("baseline", printed)
        self.assertIn("games-played corrected", printed)


class TestOliverRawTotal(unittest.TestCase):
    def test_reproduces_the_analyzer_formula(self):
        # pace 100+100-100 = 100; efficiencies 110 and 120 -> 1.10 + 1.20
        raw = oliver_raw_total(
            home_pace=100.0, away_pace=100.0, league_pace=100.0,
            home_ortg=110.0, home_drtg=115.0,
            away_ortg=120.0, away_drtg=105.0,
        )
        # home_eff = (110+105)/2 = 107.5, away_eff = (120+115)/2 = 117.5
        self.assertAlmostEqual(raw, 100.0 * (1.075 + 1.175))

    def test_early_season_shrinkage_is_subtracted(self):
        kwargs = dict(home_pace=100.0, away_pace=100.0, league_pace=100.0,
                      home_ortg=110.0, home_drtg=115.0,
                      away_ortg=120.0, away_drtg=105.0)
        full = oliver_raw_total(**kwargs, min_games=25)
        early = oliver_raw_total(**kwargs, min_games=0)
        self.assertAlmostEqual(full - early, 2.5)


class TestMarketWeightRecovery(unittest.TestCase):
    def test_inverts_the_anchoring(self):
        self.assertAlmostEqual(implied_market_weight(220.0, 3.0, 230.0), 0.70)
        self.assertAlmostEqual(implied_market_weight(220.0, -4.0, 210.0), 0.60)

    def test_degenerate_gap_is_nan(self):
        self.assertTrue(np.isnan(implied_market_weight(220.0, 3.0, 220.0)))

    def test_edge_delta_is_scaled_by_the_unanchored_share(self):
        # +2 points of raw projection at weight 0.75 is +0.5 of edge.
        self.assertAlmostEqual(rating_edge_delta(228.0, 230.0, 0.75), 0.5)


class TestRatingsFor(unittest.TestCase):
    def setUp(self):
        self.ratings = pd.DataFrame({
            "team": ["Boston Celtics", "Boston Celtics", "Boston Celtics"],
            "snapshot_at": pd.to_datetime([
                "2026-11-10 00:00", "2026-11-15 00:00", "2026-11-20 00:00"]
            ).tz_localize("UTC"),
            "gp": [9, 12, 15],
            "pace": [99.0, 100.0, 101.0],
            "off_rating": [110.0, 112.0, 114.0],
            "def_rating": [108.0, 109.0, 110.0],
            "net_rating": [2.0, 3.0, 4.0],
        })

    def test_uses_the_same_day_snapshot_and_never_a_later_one(self):
        row = ratings_for(self.ratings, "Boston Celtics", "2026-11-15")
        self.assertEqual(int(row["gp"]), 12)

    def test_falls_back_to_the_last_usable_snapshot(self):
        row = ratings_for(self.ratings, "Boston Celtics", "2026-11-17")
        self.assertEqual(int(row["gp"]), 12)

    def test_returns_none_before_the_first_snapshot(self):
        self.assertIsNone(ratings_for(self.ratings, "Boston Celtics", "2026-11-09"))
        self.assertIsNone(ratings_for(self.ratings, "Nowhere Kings", "2026-11-15"))


if __name__ == "__main__":
    unittest.main()
