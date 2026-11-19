"""Tests for the point-in-time rating rebuild.

The only thing that matters here is the cut-off: a rating used to price a game
must never have seen that game's result.
"""

import unittest

import pandas as pd

from scripts.backfill_ratings import _with_opponent, ratings_as_of


def _games():
    rows = [
        # (game_id, team, date, pts, poss)
        ("1", "Boston Celtics", "2026-10-20", 110, 100),
        ("1", "New York Knicks", "2026-10-20", 100, 100),
        ("2", "Boston Celtics", "2026-10-22", 120, 100),
        ("2", "New York Knicks", "2026-10-22", 105, 100),
    ]
    return pd.DataFrame(rows, columns=["GAME_ID", "team", "GAME_DATE", "PTS", "poss"])


def _paired():
    games = _games()
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])
    return _with_opponent(games)


class TestRatingsAsOf(unittest.TestCase):
    def test_a_day_sees_only_games_strictly_before_it(self):
        games = _paired()

        before = ratings_as_of(games, "2026-10-20")
        self.assertNotIn("Boston Celtics", before.index)  # first game not yet played

        opening = ratings_as_of(games, "2026-10-21")
        self.assertEqual(int(opening.loc["Boston Celtics", "gp"]), 1)
        self.assertAlmostEqual(opening.loc["Boston Celtics", "off_rating"], 110.0)

        second = ratings_as_of(games, "2026-10-23")
        self.assertEqual(int(second.loc["Boston Celtics", "gp"]), 2)
        # 230 points on 200 possessions.
        self.assertAlmostEqual(second.loc["Boston Celtics", "off_rating"], 115.0)

    def test_pace_averages_both_sides_of_the_ball(self):
        table = ratings_as_of(_paired(), "2026-10-21")
        # One game, 100 possessions each way.
        self.assertAlmostEqual(table.loc["Boston Celtics", "pace"], 100.0)

    def test_def_rating_uses_opponent_points(self):
        table = ratings_as_of(_paired(), "2026-10-21")
        self.assertAlmostEqual(table.loc["Boston Celtics", "def_rating"], 100.0)
        self.assertAlmostEqual(table.loc["Boston Celtics", "net_rating"], 10.0)


if __name__ == "__main__":
    unittest.main()
