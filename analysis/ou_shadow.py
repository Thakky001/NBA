"""Shadow closing-line value over the whole settled ledger.

The gate is deliberately strict, which leaves almost no real bets to grade.
CLV does not need the gate: the model still forms a side on every game, and a
model whose side beats the closing number is informative even when its stakes
are zero. Nothing here places a bet or changes ``BETTING_PAUSED``.

Usage:
    python -m analysis.ou_shadow
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from analysis.ou_replay import (
    LEDGER_DATE_OFFSET_DAYS,
    closing_line_value,
    load_closes,
    load_replay_frame,
)

#: Edge buckets used to check that conviction and CLV move together. A model
#: whose biggest edges do no better than its smallest is not ranking games.
EDGE_BUCKETS = (0.0, 0.5, 1.0, 2.0, 3.0, float("inf"))

#: Boundaries between regimes the ledger shows: pre-fix games-played counts,
#: the short window after the fix, and the refreshed-ratings era.
PERIODS = (
    ("leak era (<11-06)", None, "2026-11-06"),
    ("11-06..11-09", "2026-11-06", "2026-11-10"),
    ("11-10 onward", "2026-11-10", None),
)

#: Below this the edge is noise and the side it implies is arbitrary.
MIN_ABS_EDGE = 0.25


def shadow_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per priced game, with the side the model would have taken.

    The shadow side is the sign of the recorded edge, so it exists for every
    game the model had an opinion on, not only those that cleared the gate.
    """
    rows = frame[
        frame["total_line"].notna()
        & frame["ou_edge"].notna()
        & frame["close_total"].notna()
    ].copy()
    rows["side"] = np.where(rows["ou_edge"] >= 0, "OVER", "UNDER")
    rows["abs_edge"] = rows["ou_edge"].abs()
    # CLV if the side were OVER; a shadow UNDER simply flips the sign.
    rows["over_clv"] = rows["close_total"] - rows["total_line"]
    rows["clv"] = np.where(rows["side"] == "OVER", rows["over_clv"], -rows["over_clv"])
    rows["bucket"] = pd.cut(rows["abs_edge"], EDGE_BUCKETS, right=False,
                            labels=["0-0.5", "0.5-1", "1-2", "2-3", "3+"])
    return rows.reset_index(drop=True)


def _mean_ci(values) -> tuple[float, float, int]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if not len(arr):
        return float("nan"), float("nan"), 0
    mean = float(arr.mean())
    if len(arr) < 2:
        return mean, float("nan"), len(arr)
    se = float(arr.std(ddof=1) / np.sqrt(len(arr)))
    return mean, 1.96 * se, len(arr)


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    """Mean CLV, its 95% interval and the share of rows that beat the close."""
    out = []
    for label, part in (("all", rows),
                        *[(str(k), g) for k, g in rows.groupby("bucket", observed=True)]):
        mean, half, n = _mean_ci(part["clv"])
        out.append({
            "group": label, "n": n, "mean_clv": mean, "ci95": half,
            "beats_close": float((part["clv"] > 0).mean()) if len(part) else float("nan"),
            "loses_to_close": float((part["clv"] < 0).mean()) if len(part) else float("nan"),
            "no_move": float((part["clv"] == 0).mean()) if len(part) else float("nan"),
        })
    return pd.DataFrame(out)


def period_table(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for label, start, end in PERIODS:
        part = rows
        if start:
            part = part[part["date"] >= pd.Timestamp(start)]
        if end:
            part = part[part["date"] < pd.Timestamp(end)]
        mean, half, n = _mean_ci(part["clv"])
        out.append({"period": label, "n": n, "mean_clv": mean, "ci95": half})
    return pd.DataFrame(out)


def permutation_pvalue(rows: pd.DataFrame, reps: int = 2000,
                       seed: int = 20261119) -> float:
    """Would shuffled sides do as well?

    Each row's CLV is a signed distance from the line to the close. Reassigning
    the side at random flips that sign, which is the null of "the model's side
    carries no information".
    """
    values = rows["over_clv"].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return float("nan")
    observed = abs(float(rows["clv"].mean()))
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(reps, len(values)))
    null = np.abs((signs * values).mean(axis=1))
    return float((null >= observed).mean())


def verdict(rows: pd.DataFrame) -> str:
    mean, half, n = _mean_ci(rows["clv"])
    if n == 0:
        return "no priced rows"
    if not np.isfinite(half):
        return f"n={n}; too few rows for an interval"
    if mean - half > 0:
        return f"CLV clears zero (n={n}, {mean:+.2f} ± {half:.2f})"
    if mean + half < 0:
        return f"CLV is negative and clear of zero (n={n}, {mean:+.2f} ± {half:.2f})"
    return f"CLV indistinguishable from zero (n={n}, {mean:+.2f} ± {half:.2f})"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", default=None)
    parser.add_argument("--closes", default=None)
    parser.add_argument("--reps", type=int, default=2000)
    args = parser.parse_args(argv)

    frame = load_replay_frame(ledger_path=args.ledger, closes=load_closes(args.closes))
    rows = shadow_rows(frame)
    if rows.empty:
        print("no rows with both a line and a closing line")
        return 0

    print(f"priced rows: {len(rows)} of {len(frame)} settled "
          f"({len(frame) - len(rows)} have no closing line)")
    print(f"ledger stamps are US date + {LEDGER_DATE_OFFSET_DAYS}\n")

    table = summarize(rows)
    print(f"{'|edge| bucket':<14} {'n':>4} {'mean CLV':>9} {'±95%':>7} "
          f"{'beat':>6} {'lost':>6} {'flat':>6}")
    for r in table.itertuples():
        half = "" if not np.isfinite(r.ci95) else f"{r.ci95:.2f}"
        print(f"{r.group:<14} {r.n:>4} {r.mean_clv:>+9.2f} {half:>7} "
              f"{r.beats_close:>6.1%} {r.loses_to_close:>6.1%} {r.no_move:>6.1%}")
    print("beat/lost/flat are shares of rows whose close moved our way, away")
    print("from us, or not at all. A flat close is a real zero, not a tie-break.")

    print()
    print("by period:")
    for r in period_table(rows).itertuples():
        half = "" if not np.isfinite(r.ci95) else f"{r.ci95:.2f}"
        print(f"  {r.period:<18} n={r.n:>3} mean {r.mean_clv:+.2f} ± {half}")

    print()
    strong = rows[rows["abs_edge"] >= 1.0]
    mean, half, n = _mean_ci(strong["clv"])
    print(f"rows with |edge| >= 1.0: n={n}  mean CLV {mean:+.2f} ± {half:.2f}")

    moved = rows[rows["close_total"] != rows["total_line"]]
    mean, half, n = _mean_ci(moved["clv"])
    print(f"rows where the close actually moved: n={n}  mean CLV {mean:+.2f} ± {half:.2f}")
    if len(rows) and len(moved) / len(rows) > 0.5:
        print("  -> most closes equal the bet line, so the sample cannot tell a")
        print("     true close from the number the app already had. Treat the")
        print("     overall mean as a lower bound on what CLV can show.")

    print()
    print("controls (same rows, ignoring the model's side):")
    always_over, _, n = _mean_ci(rows["over_clv"])
    print(f"  always OVER : {always_over:+.2f}  (n={n})")
    print(f"  always UNDER: {-always_over:+.2f}  (n={n})")
    print("  -> if the model's mean sits between these, it is not adding a side,")
    print("     it is only riding a market drift.")

    p = permutation_pvalue(rows, reps=args.reps)
    print()
    print(f"permutation test ({args.reps} shuffles): p = {p:.3f}")
    print(verdict(rows))
    print()
    print("shadow only: no bets placed, gate untouched, BETTING_PAUSED unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
