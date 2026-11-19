"""Rebuild point-in-time team ratings from the per-game team logs.

The advanced-stats collector overwrites ``nba_api_advanced.csv`` with whatever
it last fetched, so the app has no way to ask "what did this team's rating look
like before game D". The team game log does carry the box-score columns needed
to reconstruct it, so this script replays the season and writes one row per
(team, US game date).

Usage:
    python scripts/backfill_ratings.py            # write the CSV
    python scripts/backfill_ratings.py --as-of 2026-11-10

The output is deterministic: same input log, same bytes.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from config import RAW_DIR, normalize_team_name

LOGS_FILENAME = "nba_api_game_logs.csv"
OUTPUT_FILENAME = "nba_api_ratings_pit.csv"

#: Box-score columns that must be present for a game row to be usable. The
#: collector's tail is sometimes written with points only.
REQUIRED = ["FGA", "FTA", "OREB", "TOV", "PTS"]

COLUMNS = ["team", "snapshot_at", "gp", "pace", "off_rating", "def_rating", "net_rating"]


def _load_games(path: str) -> pd.DataFrame:
    logs = pd.read_csv(path, low_memory=False)
    logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"], errors="coerce")
    logs = logs.dropna(subset=["GAME_DATE"] + REQUIRED).copy()
    logs["team"] = logs["TEAM_NAME"].map(lambda t: normalize_team_name(str(t)) or str(t))
    # Standard possession estimate; the NBA's official figure also folds in
    # team rebounds and technical free throws, which the log does not carry.
    logs["poss"] = logs["FGA"] + 0.4 * logs["FTA"] - logs["OREB"] + logs["TOV"]
    return logs[["GAME_ID", "team", "GAME_DATE", "PTS", "poss"]]


def _with_opponent(games: pd.DataFrame) -> pd.DataFrame:
    opp = games.rename(columns={"team": "opp", "PTS": "opp_pts", "poss": "opp_poss"})
    merged = games.merge(opp[["GAME_ID", "opp", "opp_pts", "opp_poss"]], on="GAME_ID")
    return merged[merged["team"] != merged["opp"]]


def ratings_as_of(games: pd.DataFrame, game_date) -> pd.DataFrame:
    """Cumulative ratings using only games strictly before ``game_date``."""
    cutoff = pd.Timestamp(game_date).normalize()
    played = games[games["GAME_DATE"] < cutoff]
    if played.empty:
        return pd.DataFrame(columns=COLUMNS[2:])
    out = played.groupby("team").agg(
        gp=("GAME_ID", "count"),
        pts=("PTS", "sum"),
        poss=("poss", "sum"),
        opp_pts=("opp_pts", "sum"),
        opp_poss=("opp_poss", "sum"),
    )
    out["off_rating"] = 100.0 * out["pts"] / out["poss"]
    out["def_rating"] = 100.0 * out["opp_pts"] / out["opp_poss"]
    out["pace"] = (out["poss"] + out["opp_poss"]) / 2.0 / out["gp"]
    out["net_rating"] = out["off_rating"] - out["def_rating"]
    # League-average pace is a useful cross-check when a game log is thin.
    league_pace = float(out["pace"].mean())
    out["league_pace"] = league_pace
    return out[["gp", "pace", "off_rating", "def_rating", "net_rating", "league_pace"]]


def build(games: pd.DataFrame, first: pd.Timestamp, last: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for day in pd.date_range(first, last, freq="D"):
        table = ratings_as_of(games, day)
        if table.empty:
            continue
        table = table.reset_index()
        table["snapshot_at"] = day.tz_localize("UTC")
        rows.append(table[COLUMNS])
    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    out = pd.concat(rows, ignore_index=True)
    for col in ("pace", "off_rating", "def_rating", "net_rating"):
        out[col] = out[col].round(4)
    out["gp"] = out["gp"].astype(int)
    return out.sort_values(["snapshot_at", "team"]).reset_index(drop=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", default=os.path.join(RAW_DIR, LOGS_FILENAME))
    parser.add_argument("--output", default=os.path.join(RAW_DIR, OUTPUT_FILENAME))
    parser.add_argument("--as-of", default=None,
                        help="print one date's table instead of writing the file")
    args = parser.parse_args(argv)

    games = _with_opponent(_load_games(args.logs))
    first_day = games["GAME_DATE"].min().normalize() + pd.Timedelta(days=1)
    last_day = games["GAME_DATE"].max().normalize() + pd.Timedelta(days=2)
    print(f"log: {len(games)} team-games, {first_day.date()} .. {games['GAME_DATE'].max().date()}")

    if args.as_of:
        table = ratings_as_of(games, args.as_of)
        print(table.round(2).to_string())
        return 0

    out = build(games, first_day, last_day)
    out.to_csv(args.output, index=False, float_format="%.4f")
    print(f"wrote {len(out)} rows to {args.output}")
    print(f"  dates {out['snapshot_at'].min().date()} .. {out['snapshot_at'].max().date()}")
    print(f"  teams per date: {out.groupby('snapshot_at').size().describe().to_dict()}")
    last = out[out["snapshot_at"] == out["snapshot_at"].max()]
    print(f"  last snapshot gp range: {last['gp'].min()}-{last['gp'].max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
