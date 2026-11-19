"""Tests for the shadow CLV report.

The report only has value if the sign handling is right: an inverted UNDER
would make every losing side look profitable.
"""

import unittest

import numpy as np
import pandas as pd

from analysis.ou_shadow import (
    permutation_pvalue,
    period_table,
    shadow_rows,
    summarize,
    verdict,
)


def _frame(edges, lines, closes, dates=None):
    n = len(edges)
    return pd.DataFrame({
        "game_id": [str(i) for i in range(n)],
        "total_line": lines,
        "ou_edge": edges,
        "close_total": closes,
        "date": pd.to_datetime(dates or ["2026-11-12"] * n),
    })


class TestShadowRows(unittest.TestCase):
    def test_side_follows_the_edge_sign(self):
        rows = shadow_rows(_frame([2.0, -2.0], [220.0, 220.0], [221.0, 221.0]))
        self.assertEqual(list(rows["side"]), ["OVER", "UNDER"])

    def test_under_clv_flips(self):
        # Line 220, close 221: the OVER backer holds the better number by a
        # point, the UNDER backer is a point worse off.
        rows = shadow_rows(_frame([2.0, -2.0], [220.0, 220.0], [221.0, 221.0]))
        self.assertEqual(list(rows["clv"]), [1.0, -1.0])

    def test_rows_without_a_close_are_dropped(self):
        rows = shadow_rows(_frame([2.0, -2.0], [220.0, 220.0], [np.nan, 221.0]))
        self.assertEqual(len(rows), 1)

    def test_bucket_edges_are_left_closed(self):
        rows = shadow_rows(_frame([0.5, 1.0, 2.0, 3.0], [220.0] * 4, [221.0] * 4))
        self.assertEqual([str(b) for b in rows["bucket"]],
                         ["0.5-1", "1-2", "2-3", "3+"])


class TestSummarize(unittest.TestCase):
    def test_flat_close_is_neither_a_win_nor_a_loss(self):
        rows = shadow_rows(_frame([2.0, -2.0], [220.0, 220.0], [220.0, 220.0]))
        table = summarize(rows).set_index("group").loc["all"]
        self.assertEqual(table["n"], 2)
        self.assertEqual(table["no_move"], 1.0)
        self.assertEqual(table["mean_clv"], 0.0)


class TestPermutation(unittest.TestCase):
    def test_shuffling_sides_kills_a_perfect_signal(self):
        # Every OVER was right by a point: no random reassignment matches it.
        rows = shadow_rows(_frame([2.0] * 12, [220.0] * 12, [221.0] * 12))
        self.assertLess(permutation_pvalue(rows, reps=200), 0.05)

    def test_no_movement_makes_the_side_irrelevant(self):
        rows = shadow_rows(_frame([2.0] * 10, [220.0] * 10, [220.0] * 10))
        self.assertAlmostEqual(permutation_pvalue(rows, reps=200), 1.0)


class TestPeriodsAndVerdict(unittest.TestCase):
    def test_periods_split_on_the_fix_windows(self):
        dates = ["2026-11-02", "2026-11-07", "2026-11-12"]
        rows = shadow_rows(_frame([2.0] * 3, [220.0] * 3, [221.0] * 3, dates))
        table = period_table(rows).set_index("period")
        self.assertEqual(table.loc["leak era (<11-06)", "n"], 1)
        self.assertEqual(table.loc["11-06..11-09", "n"], 1)
        self.assertEqual(table.loc["11-10 onward", "n"], 1)

    def test_verdict_reports_a_clear_negative(self):
        rows = shadow_rows(_frame([2.0] * 20, [220.0] * 20, [219.0] * 20))
        self.assertIn("negative", verdict(rows))

    def test_verdict_does_not_pretend_when_flat(self):
        rows = shadow_rows(_frame([2.0] * 10, [220.0] * 10, [220.0] * 10))
        self.assertIn("indistinguishable", verdict(rows))


if __name__ == "__main__":
    unittest.main()
