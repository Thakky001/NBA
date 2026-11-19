"""Ablation: does replacing stale ratings with point-in-time ratings help?

Three projections per ledger row, all compared against the settled total:

    recorded     the edge the app actually logged (mixes code versions)
    code+stored  today's possession formula with the ratings stored in the
                 snapshot (stale, but the app's own inputs)
    code+PIT     today's formula with ratings rebuilt from the game log

The last two share every line of code, so their difference is the rating swap
alone. ``recorded`` is only a reference: rows written by ``nba-baseline-v2``
came from an older pricing path and cannot be reproduced exactly, which the
script reports before using anything.

Run:
    python scripts/ou_rating_ablation.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from analysis import ou_replay as R
from analysis.betting_math import dynamic_early_season_market_weight

RATING_KEYS = ("home_pace", "away_pace", "league_avg_pace",
               "home_ortg", "away_ortg", "home_drtg", "away_drtg")


def _num(snapshot, key: str) -> float:
    value = snapshot.get(key) if isinstance(snapshot, dict) else None
    try:
        return float(value) if value is not None else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _raw_from_snapshot(snapshot, min_games: float) -> float:
    if any(pd.isna(_num(snapshot, k)) for k in RATING_KEYS):
        return float("nan")
    return R.oliver_raw_total(
        home_pace=_num(snapshot, "home_pace"),
        away_pace=_num(snapshot, "away_pace"),
        league_pace=_num(snapshot, "league_avg_pace"),
        home_ortg=_num(snapshot, "home_ortg"),
        home_drtg=_num(snapshot, "home_drtg"),
        away_ortg=_num(snapshot, "away_ortg"),
        away_drtg=_num(snapshot, "away_drtg"),
        min_games=min_games,
    )


def _raw_from_pit(ratings, snapshot, home_team, away_team, us_date, min_games) -> float:
    home = R.ratings_for(ratings, home_team, us_date)
    away = R.ratings_for(ratings, away_team, us_date)
    if home is None or away is None:
        return float("nan")
    return R.oliver_raw_total(
        home_pace=home["pace"], away_pace=away["pace"],
        # League pace stays a league-wide number, so it comes from the snapshot.
        league_pace=_num(snapshot, "league_avg_pace"),
        home_ortg=home["off_rating"], home_drtg=home["def_rating"],
        away_ortg=away["off_rating"], away_drtg=away["def_rating"],
        min_games=min_games,
    )


def _metrics(edges: pd.Series, frame: pd.DataFrame) -> dict:
    err = (frame["total_line"] + edges) - frame["actual_total"]
    clv = []
    picks = 0
    for edge_value, row in zip(edges, frame.itertuples()):
        if pd.isna(edge_value) or pd.isna(row.min_gp_fixed):
            continue
        decision = R.decide(float(row.total_line), float(edge_value),
                            min_games=float(row.min_gp_fixed),
                            odds=row.odds)
        if not decision["is_actionable"]:
            continue
        picks += 1
        value = R.closing_line_value(decision["pick_side"], row.total_line,
                                     row.close_total)
        if np.isfinite(value):
            clv.append(value)
    mean_clv = float(np.mean(clv)) if clv else float("nan")
    se = float(np.std(clv, ddof=1) / np.sqrt(len(clv))) if len(clv) > 1 else float("nan")
    return {
        "n": int(err.notna().sum()),
        "mae": float(err.abs().mean()), "bias": float(err.mean()),
        "picks": picks, "clv": mean_clv, "clv_se": se, "clv_n": len(clv),
    }


def main() -> int:
    frame = R.load_replay_frame(closes=R.load_closes())
    ratings = R.load_ratings_pit()
    logs = R.load_team_game_log()

    rows = []
    for _, row in frame.iterrows():
        snapshot = row["features"]
        if not isinstance(snapshot, dict) or pd.isna(row["min_games"]):
            continue
        stored_raw = _raw_from_snapshot(snapshot, row["min_games"])
        if not np.isfinite(stored_raw):
            continue
        us_date = R.us_game_date(row["date"])
        pit_raw = _raw_from_pit(ratings, snapshot, row["home_team"],
                                row["away_team"], us_date, row["min_games"])
        weight = dynamic_early_season_market_weight(row["min_games"])
        rows.append({
            "date": row["date"], "model": row["model_version"],
            "total_line": row["total_line"], "actual_total": row["actual_total"],
            "close_total": row["close_total"], "odds": row["total_odds_parsed"],
            "min_gp": row["min_games"],
            "min_gp_fixed": R.resolve_games_played(row, logs),
            "recorded": row["ou_edge"],
            "code_stored": (stored_raw - row["total_line"]) * (1.0 - weight),
            "code_pit": (pit_raw - row["total_line"]) * (1.0 - weight),
            "raw_stored": stored_raw, "raw_pit": pit_raw, "weight": weight,
        })

    df = pd.DataFrame(rows).dropna(subset=["actual_total"])
    print(f"rows usable: {len(df)} / {len(frame)}  "
          f"(the rest lack a full rating snapshot)")
    print()

    # Is the recorded edge reproducible from the stored snapshot at all?
    df["repro_err"] = (df["recorded"] - df["code_stored"]).abs()
    print("reproduction check by model version (recorded vs code+stored):")
    for version, part in df.groupby("model"):
        ok = int((part["repro_err"] <= 0.5).sum())
        print(f"  {version:<16} n={len(part):>3}  within 0.5: {ok:>3}  "
              f"median |err| {part['repro_err'].median():.2f}")
    print("  -> only exact rows can be treated as the same pricing path; the")
    print("     other version's rows are counterfactual, not reproduction.")
    print()

    table = pd.DataFrame([
        {"variant": "recorded (as shipped)", **_metrics(df["recorded"], df)},
        {"variant": "code + stale ratings", **_metrics(df["code_stored"], df)},
        {"variant": "code + point-in-time", **_metrics(df["code_pit"], df)},
    ])
    print(f"{'variant':<24} {'n':>4} {'MAE':>7} {'bias':>7} {'picks':>6} "
          f"{'CLV':>8}")
    for r in table.itertuples():
        clv = "n/a" if not np.isfinite(r.clv) else (
            f"{r.clv:+.2f}" + ("" if not np.isfinite(r.clv_se) else f"±{r.clv_se:.2f}"))
        print(f"{r.variant:<24} {r.n:>4} {r.mae:>7.2f} {r.bias:>+7.2f} "
              f"{r.picks:>6} {clv:>8}")

    market_mae = float((df["total_line"] - df["actual_total"]).abs().mean())
    print(f"\nmarket line MAE on the same rows: {market_mae:.2f}")

    swap = (df["code_pit"] - df["code_stored"]).abs()
    print(f"rating swap moves the edge: median {swap.median():.2f}, "
          f"p90 {swap.quantile(0.9):.2f}, max {swap.max():.2f}")
    print(f"rows whose edge sign flips: "
          f"{int((np.sign(df.code_pit) != np.sign(df.code_stored)).sum())}")

    v3 = df[df["model"] == "nba-baseline-v3"]
    if len(v3) > 2:
        print(f"\nsame table on the reproducible subset (n={len(v3)}):")
        for name, col in (("recorded", "recorded"),
                          ("code + stale", "code_stored"),
                          ("code + PIT", "code_pit")):
            m = _metrics(v3[col], v3)
            print(f"  {name:<14} MAE {m['mae']:.2f}  bias {m['bias']:+.2f}  "
                  f"picks {m['picks']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
