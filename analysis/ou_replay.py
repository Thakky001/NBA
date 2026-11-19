"""Walk-forward replay of the O/U decision gate over the prediction ledger.

Why this exists
---------------
Testing a model change used to mean waiting weeks for new graded bets. It does
not have to: every ledger row already stores the exact inputs its decision was
made from - the line, the model's edge, the odds, and the data-quality flags.
Rebuilding the decision from those values needs no feature reconstruction, no
network access and, crucially, no lookahead, so a candidate change can be
scored against every game the live app has already played.

Why CLV and not ROI
-------------------
The raw O/U residual has a standard deviation near 19 points. The injury effect
being tested is worth 1-3 points, so separating it from that noise needs roughly
700+ games. Closing-line value on the same bets has a standard deviation near
0.6 points, which is why ~100 bets can already say something. The gate is built
on CLV for that reason, and so is this harness.

What it deliberately does not do
--------------------------------
It does not recompute features. The stored edge is the anchor, and a variant is
expressed as a repricing of that edge. That keeps the baseline identical to what
the live app actually produced, which is the only fair comparison.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd

from analysis.betting_math import (
    calculate_ev_with_push,
    calculate_spread_probabilities,
    calibrate_win_probability,
    dynamic_early_season_market_weight,
)
from analysis.prediction_tracker import LEDGER_PATH
from analysis.recommendation import STRONG_MIN_GAMES, decide_pick
from config import RAW_DIR, normalize_team_name

logger = logging.getLogger("ou_replay")

#: Assumed sigma when a row has no stored residual sigma. Matches the value the
#: analyzer falls back to when no training residuals are loaded.
SIGMA_TOTAL = 12.5

#: Total deflation applied when both teams are early in the season, per point of
#: missing sample. Mirrors trend_analyzer.analyze_over_under.
EARLY_SEASON_SHRINK_MAX = 2.5
EARLY_SEASON_GAMES = 20.0

#: First day of the 2026-27 regular season, used only to sanity-check the
#: games-played values a row was scored with.
SEASON_START = "2026-10-20"

#: The ledger stamps each row with the date it was written, in Bangkok time,
#: which is always one day after the US game date: a 7pm ET tip is 00:00Z the
#: next day and the app records it that morning. Verified on all 221 ledger
#: rows (170 tips 00:00-05:00Z, 51 tips 19:00-23:00Z).
LEDGER_DATE_OFFSET_DAYS = 1

CLOSES_FILENAME = "espn_closing_lines.csv"

#: Point-in-time team ratings rebuilt from the per-game log by
#: ``scripts/backfill_ratings.py``.
RATINGS_PIT_FILENAME = "nba_api_ratings_pit.csv"

_PICK_OVER = "OVER"
_PICK_UNDER = "UNDER"


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def _parse_json(value: Any) -> Any:
    if isinstance(value, str) and value.strip():
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return None
    return None


def _parse_odds(value: Any) -> Optional[list]:
    """Ledger stores ``[-110, -110]`` as JSON text."""
    parsed = _parse_json(value)
    if isinstance(parsed, (list, tuple)) and len(parsed) == 2:
        try:
            return [float(parsed[0]), float(parsed[1])]
        except (TypeError, ValueError):
            return None
    if isinstance(parsed, (int, float)):
        return [float(parsed), float(parsed)]
    return None


def load_closes(path: Optional[str] = None) -> pd.DataFrame:
    """Load backfilled ESPN closing lines, empty frame when absent."""
    path = path or os.path.join(RAW_DIR, CLOSES_FILENAME)
    if not os.path.exists(path):
        logger.warning(f"closing lines not found at {path}; CLV will be unavailable")
        return pd.DataFrame(columns=["game_id", "close_total", "open_total", "close_spread"])
    df = pd.read_csv(path, dtype={"game_id": str})
    return df


def load_replay_frame(
    ledger_path: Optional[str] = None,
    closes: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build the replay table: one row per graded O/U prediction.

    Rows lacking a line or a model edge are dropped rather than imputed - an
    unpriceable bet cannot be a pick, and guessing here would quietly change the
    baseline every variant is compared against.
    """
    ledger = pd.read_csv(ledger_path or LEDGER_PATH, low_memory=False)
    ledger["game_id"] = ledger["game_id"].astype(str)

    numeric = ["total_line", "ou_edge", "projected_total", "actual_total"]
    for col in numeric:
        if col in ledger.columns:
            ledger[col] = pd.to_numeric(ledger[col], errors="coerce")

    frame = ledger[
        (ledger["status"].astype(str) == "GRADED")
        & ledger["total_line"].notna()
        & ledger["ou_edge"].notna()
    ].copy()

    snapshots = frame["features_snapshot"].apply(_parse_json)
    frame["features"] = snapshots.apply(lambda s: s if isinstance(s, dict) else {})

    def _games(snap: Dict[str, Any], side: str) -> Optional[float]:
        value = snap.get(f"{side}_games_played")
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    frame["home_games_played"] = frame["features"].apply(lambda s: _games(s, "home"))
    frame["away_games_played"] = frame["features"].apply(lambda s: _games(s, "away"))
    frame["min_games"] = frame[["home_games_played", "away_games_played"]].min(axis=1)
    frame["total_odds_parsed"] = frame["total_odds"].apply(_parse_odds) if "total_odds" in frame.columns else None
    frame["pick_side"] = frame["ou_recommendation"].astype(str).str.extract(
        r"(OVER|UNDER)", expand=False)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")

    if closes is not None and not closes.empty:
        closes = closes.copy()
        closes["game_id"] = closes["game_id"].astype(str)
        keep = [c for c in ("game_id", "close_total", "open_total", "close_spread") if c in closes.columns]
        frame = frame.merge(closes[keep], on="game_id", how="left")
    else:
        for col in ("close_total", "open_total", "close_spread"):
            frame[col] = np.nan

    return frame.reset_index(drop=True)


# ---------------------------------------------------------------------------
# pricing
# ---------------------------------------------------------------------------

def closing_line_value(side: str, bet_line: float, close_line: float) -> float:
    """Points gained against the closing line. Positive is better than the close.

    OVER improves when the number falls, UNDER improves when it rises, so the
    sign flips with the side. A push-free half-point market has no tie to split.
    """
    if bet_line is None or close_line is None:
        return float("nan")
    if np.isnan(bet_line) or np.isnan(close_line):
        return float("nan")
    side = str(side).upper()
    if side == _PICK_OVER:
        return float(close_line) - float(bet_line)
    if side == _PICK_UNDER:
        return float(bet_line) - float(close_line)
    return float("nan")


def raw_projection(line: float, edge: float, market_weight: float) -> float:
    """Undo the market anchoring to recover the pre-anchor projection.

    ``edge = (raw - line) * (1 - market_weight)`` is the last thing that happens
    to a total before it reaches the gate, so this inverts it exactly.
    """
    if market_weight >= 1.0:
        return float(line)
    return float(line) + float(edge) / (1.0 - float(market_weight))


def early_season_shrinkage(min_games: Optional[float]) -> float:
    """Deflation applied to the raw total before anchoring."""
    if min_games is None or min_games >= EARLY_SEASON_GAMES:
        return 0.0
    return EARLY_SEASON_SHRINK_MAX * max(0.0, (EARLY_SEASON_GAMES - min_games) / EARLY_SEASON_GAMES)


def reprice(
    line: float,
    edge: float,
    *,
    stored_weight: float,
    stored_shrinkage: float = 0.0,
    new_weight: Optional[float] = None,
    new_shrinkage: Optional[float] = None,
    raw_shift: float = 0.0,
) -> float:
    """Re-derive the model edge under a different anchoring / shrinkage regime.

    Both corrections are applied to the same underlying projection, so a variant
    that changes only the market weight cannot accidentally inherit a different
    shrinkage assumption. ``raw_shift`` is added to the projection itself, in
    points, which is where an injury term or a bias correction belongs.
    """
    raw = raw_projection(line, edge, stored_weight) + float(stored_shrinkage)
    shrink = float(stored_shrinkage) if new_shrinkage is None else float(new_shrinkage)
    raw = raw - shrink + float(raw_shift)
    weight = stored_weight if new_weight is None else float(new_weight)
    return round((raw - float(line)) * (1.0 - weight), 4)


def oliver_raw_total(
    *,
    home_pace: float,
    away_pace: float,
    league_pace: float,
    home_ortg: float,
    home_drtg: float,
    away_ortg: float,
    away_drtg: float,
    min_games: Optional[float] = None,
) -> float:
    """Pre-anchor total from the possession model, mirrors the analyzer.

    Kept in one place so a re-priced row differs from the recorded one only
    where the inputs changed, never because two copies of the formula drifted.
    """
    expected_pace = float(home_pace) + float(away_pace) - float(league_pace)
    home_eff = (float(home_ortg) + float(away_drtg)) / 2.0
    away_eff = (float(away_ortg) + float(home_drtg)) / 2.0
    raw = expected_pace * (home_eff / 100.0) + expected_pace * (away_eff / 100.0)
    return raw - early_season_shrinkage(min_games)


def implied_market_weight(line: float, edge: float, raw_stored: float) -> float:
    """Recover the anchoring weight a row was scored with.

    ``edge = (raw - line) * (1 - weight)`` is invertible, which lets a variant
    reuse the weight production actually applied instead of assuming one.
    """
    gap = float(raw_stored) - float(line)
    if not np.isfinite(gap) or abs(gap) < 1e-9:
        return float("nan")
    return 1.0 - float(edge) / gap


def rating_edge_delta(raw_stored: float, raw_new: float, weight: float) -> float:
    """Edge change from swapping ratings, holding the anchoring weight fixed."""
    return round((float(raw_new) - float(raw_stored)) * (1.0 - float(weight)), 4)


def load_ratings_pit(path: Optional[str] = None) -> pd.DataFrame:
    """Load the point-in-time ratings table, empty frame when absent."""
    path = path or os.path.join(RAW_DIR, RATINGS_PIT_FILENAME)
    if not os.path.exists(path):
        logger.warning(f"point-in-time ratings not found at {path}")
        return pd.DataFrame(columns=["team", "snapshot_at", "gp", "pace",
                                     "off_rating", "def_rating", "net_rating"])
    frame = pd.read_csv(path)
    frame["snapshot_at"] = pd.to_datetime(frame["snapshot_at"], errors="coerce", utc=True)
    return frame


def ratings_for(ratings: pd.DataFrame, team: str, game_date: Any) -> Optional[pd.Series]:
    """Latest rating snapshot usable for a game on ``game_date`` (a US date).

    Snapshots are stamped at 00:00 UTC of the US date and hold only games
    played before it, so the same date is eligible and the next one is not.
    """
    if ratings is None or ratings.empty:
        return None
    norm = normalize_team_name(str(team)) or str(team)
    cutoff = pd.Timestamp(game_date)
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    cutoff = cutoff.normalize()
    rows = ratings[(ratings["team"] == norm) & (ratings["snapshot_at"] <= cutoff)]
    if rows.empty:
        return None
    return rows.sort_values("snapshot_at").iloc[-1]


# ---------------------------------------------------------------------------
# gate
# ---------------------------------------------------------------------------

def decide(
    line: float,
    edge: float,
    *,
    min_games: Optional[float],
    odds: Optional[list] = None,
    sigma: float = SIGMA_TOTAL,
) -> Dict[str, Any]:
    """Run the live gate on a (possibly repriced) edge.

    The probability maths is the analyzer's own, so a variant differs from the
    baseline only where it is meant to.
    """
    projected = float(line) + float(edge)
    probs = calculate_spread_probabilities(-float(line), projected, float(sigma))
    push = probs["push"]
    cal_over = calibrate_win_probability(probs["win"], push_prob=push)
    cal_under = calibrate_win_probability(probs["loss"], push_prob=push)

    if edge > 0:
        candidate_prob, odds_idx = cal_over, 0
    elif edge < 0:
        candidate_prob, odds_idx = cal_under, 1
    else:
        candidate_prob, odds_idx = 0.5, 0

    ev = None
    if odds:
        ev = calculate_ev_with_push(candidate_prob, odds[odds_idx], push_prob=push)

    flags = {
        "home_games_played": min_games,
        "away_games_played": min_games,
        "dq_is_reliable": True,
        "dq_low_sample": bool(min_games is not None and min_games < 5),
        "dq_stale_data": False,
    }
    is_early = min_games is not None and min_games < EARLY_SEASON_GAMES

    decision = decide_pick(
        market_type="total",
        current_line=float(line),
        projected_val=projected,
        edge=float(edge),
        fair_prob=cal_over,
        line_movement=None,
        open_line=None,
        data_quality_flags=flags,
        is_early_season=is_early,
        push_prob=push,
        ev=ev,
    )
    decision["projected"] = projected
    decision["prob_over"] = cal_over
    decision["prob_under"] = cal_under
    decision["push_prob"] = push
    return decision


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------

def replay(
    frame: pd.DataFrame,
    edge_fn: Optional[Callable[[pd.Series], Optional[float]]] = None,
    min_games_fn: Optional[Callable[[pd.Series], Optional[float]]] = None,
) -> pd.DataFrame:
    """Re-decide every row, optionally repricing its edge first.

    ``edge_fn`` and ``min_games_fn`` receive the row and return the variant's
    values. Returning ``None`` from ``edge_fn`` leaves that row unpriceable, and
    it is dropped instead of being silently kept at its original edge.
    """
    out = []
    for _, row in frame.iterrows():
        line = row["total_line"]
        edge = row["ou_edge"]
        min_games = row["min_games"]

        if edge_fn is not None:
            candidate = edge_fn(row)
            if candidate is None or not np.isfinite(candidate):
                continue
            edge = float(candidate)
        if min_games_fn is not None:
            min_games = min_games_fn(row)

        decision = decide(
            float(line),
            float(edge),
            min_games=None if min_games is None or pd.isna(min_games) else float(min_games),
            odds=row.get("total_odds_parsed"),
        )
        out.append({
            "date": row["date"],
            "game_id": row["game_id"],
            "line": float(line),
            "edge": float(edge),
            "recommendation": decision["recommendation"],
            "pick_side": decision["pick_side"],
            "pick_prob": decision["pick_prob"],
            "is_actionable": decision["is_actionable"],
            "suppressed_reason": decision["suppressed_reason"],
            "close_total": row.get("close_total"),
            "actual_total": row.get("actual_total"),
            "ou_result": row.get("ou_result"),
        })
    result = pd.DataFrame(out)
    if result.empty:
        return result
    result["clv"] = [
        closing_line_value(side, line, close)
        for side, line, close in zip(result["pick_side"], result["line"], result["close_total"])
    ]
    return result


def summarize(picks: pd.DataFrame, label: str = "all") -> Dict[str, Any]:
    """Pick count, win rate, ROI at -110, and the CLV t-interval."""
    graded = picks[picks["pick_side"].notna()].copy()
    wins = (graded["ou_result"].astype(str) == "WIN").sum()
    losses = (graded["ou_result"].astype(str) == "LOSS").sum()
    decided = wins + losses

    pnl = np.where(graded["ou_result"].astype(str) == "WIN", 100.0 / 110.0,
                   np.where(graded["ou_result"].astype(str) == "LOSS", -1.0, 0.0))
    roi = float(pnl.sum() / decided) if decided else float("nan")

    clv = graded["clv"].dropna()
    if len(clv) >= 2:
        se = clv.std(ddof=1) / np.sqrt(len(clv))
        lo, hi = clv.mean() - 1.96 * se, clv.mean() + 1.96 * se
    else:
        se = lo = hi = float("nan")

    return {
        "label": label,
        "picks": int(len(graded)),
        "wins": int(wins),
        "losses": int(losses),
        "win_rate": float(wins / decided) if decided else float("nan"),
        "roi": roi,
        "clv_n": int(len(clv)),
        "clv_mean": float(clv.mean()) if len(clv) else float("nan"),
        "clv_se": float(se),
        "clv_lo": float(lo),
        "clv_hi": float(hi),
        "clv_above_zero": bool(len(clv) >= 2 and lo > 0),
    }


def format_summary(row: Dict[str, Any]) -> str:
    """One aligned line, so variants line up in a terminal for comparison."""
    roi = row["roi"]
    roi_txt = f"{roi * 100:+.1f}%" if np.isfinite(roi) else "n/a"
    if row["clv_n"] >= 2 and np.isfinite(row["clv_mean"]):
        clv_txt = (f"{row['clv_mean']:+.3f} [{row['clv_lo']:+.3f},{row['clv_hi']:+.3f}]"
                   f"{'  PASS' if row['clv_above_zero'] else ''}")
    elif row["picks"] == 0:
        clv_txt = "n/a (no picks)"
    else:
        clv_txt = "n/a (no closing lines)"
    return (f"{row['label']:<28} {row['picks']:>4}  {row['win_rate']:>6.1%}  "
            f"{roi_txt:>7}  {clv_txt}")


# ---------------------------------------------------------------------------
# games-played correction
# ---------------------------------------------------------------------------

_GAME_LOG_CACHE: Dict[str, Any] = {}


def load_team_game_log() -> Optional[pd.DataFrame]:
    """Current-season team game logs, complete box rows only.

    The tail of the collector's file is written with points only, so rows missing
    the counting stats are dropped: a half-populated game would distort both the
    possession estimate and the game count.
    """
    path = os.path.join(RAW_DIR, "nba_api_game_logs.csv")
    if path in _GAME_LOG_CACHE:
        return _GAME_LOG_CACHE[path]
    if not os.path.exists(path):
        logger.warning(f"team game log not found at {path}")
        _GAME_LOG_CACHE[path] = None
        return None
    logs = pd.read_csv(path)
    logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"], errors="coerce")
    logs["TEAM"] = logs["TEAM_NAME"].map(lambda t: normalize_team_name(t) or t)
    complete = ["FGA", "FTA", "OREB", "TOV", "PTS"]
    logs = logs.dropna(subset=complete).drop_duplicates(subset=["TEAM", "GAME_ID"], keep="first")
    _GAME_LOG_CACHE[path] = logs
    return logs


def games_played_as_of(team: str, game_date: Any, logs: Optional[pd.DataFrame] = None) -> Optional[int]:
    """Games a team had completed before its game on ``game_date``.

    ``game_date`` must be the **US game date** (see :func:`us_game_date`), so the
    filter is strictly before that date and the game being priced is never
    counted.
    """
    logs = load_team_game_log() if logs is None else logs
    if logs is None or logs.empty:
        return None
    team = normalize_team_name(team) or team
    as_of = pd.Timestamp(game_date)
    played = logs[(logs["TEAM"] == team) & (logs["GAME_DATE"] < as_of)]
    return int(len(played))


def plays_impossible_count(min_games: Optional[float], game_date: Any) -> bool:
    """True when a row claimed more games than the calendar allows.

    A team cannot have played more than one game per day since opening night, so
    anything above that is a collection error, not a fast start.
    """
    if min_games is None or pd.isna(min_games):
        return False
    days = (pd.Timestamp(game_date) - pd.Timestamp(SEASON_START)).days
    return min_games > days


def us_game_date(ledger_date: Any) -> pd.Timestamp:
    """US game date for a ledger row stamped with the recording date.

    Every ledger row carries the Bangkok date it was written, which is one day
    after the US slate it prices. Using the stamp directly counts the previous
    night's game and excludes the one being priced, so the offset is applied here
    instead of at each call site.
    """
    return pd.Timestamp(ledger_date).normalize() - pd.Timedelta(days=LEDGER_DATE_OFFSET_DAYS)


def resolve_games_played(row: pd.Series, logs: Optional[pd.DataFrame] = None) -> Optional[float]:
    """The games count a row should be scored with.

    A count the calendar cannot explain is replaced by the real one. A count
    that is merely *low* is left alone: the live app can legitimately lag by a
    slate, and "correcting" that would swap the app's own input for one it never
    had, which is not a counterfactual anyone can act on.
    """
    stored = row.get("min_games")
    if stored is None or pd.isna(stored):
        return None
    game_date = us_game_date(row.get("date"))
    if not plays_impossible_count(stored, game_date):
        return float(stored)
    home_gp = games_played_as_of(row.get("home_team"), game_date, logs)
    away_gp = games_played_as_of(row.get("away_team"), game_date, logs)
    candidates = [g for g in (home_gp, away_gp) if g is not None]
    return float(min(candidates)) if candidates else None


__all__ = [
    "SIGMA_TOTAL",
    "closing_line_value",
    "decide",
    "early_season_shrinkage",
    "format_summary",
    "games_played_as_of",
    "implied_market_weight",
    "load_closes",
    "load_ratings_pit",
    "load_replay_frame",
    "load_team_game_log",
    "oliver_raw_total",
    "plays_impossible_count",
    "rating_edge_delta",
    "ratings_for",
    "raw_projection",
    "replay",
    "reprice",
    "resolve_games_played",
    "summarize",
    "us_game_date",
]


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------

def _prepare(frame: pd.DataFrame, logs: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Attach the anchoring state each row was scored with, plus true games."""
    frame = frame.copy()
    frame["stored_weight"] = frame["min_games"].apply(
        lambda g: dynamic_early_season_market_weight(None if pd.isna(g) else int(g)))
    frame["stored_shrink"] = frame["min_games"].apply(
        lambda g: early_season_shrinkage(None if pd.isna(g) else float(g)))
    frame["true_gp"] = frame.apply(lambda r: resolve_games_played(r, logs), axis=1)
    frame["impact"] = pd.to_numeric(
        frame["features"].apply(lambda s: s.get("injury_net_total_impact") or 0.0),
        errors="coerce").fillna(0.0)
    return frame


def variant_fn(*, fix_games: bool, injury_k: float = 0.0,
               bump: float = 0.0) -> Callable[[pd.Series], float]:
    """Edge repricing for the command-line variants."""

    def fn(row: pd.Series) -> float:
        gp = row["true_gp"] if fix_games else row["min_games"]
        gp = None if gp is None or pd.isna(gp) else float(gp)
        return reprice(
            row["total_line"],
            row["ou_edge"],
            stored_weight=row["stored_weight"],
            stored_shrinkage=row["stored_shrink"],
            new_weight=dynamic_early_season_market_weight(None if gp is None else int(gp)),
            new_shrinkage=early_season_shrinkage(gp),
            raw_shift=injury_k * row["impact"] - bump,
        )

    return fn


def main(argv: Optional[list] = None) -> int:
    """Print the baseline against the corrected-input variant.

    ``--injury-k`` and ``--bump`` are applied on top of the corrected variant.
    Nothing here writes to the ledger or changes production behaviour.
    """
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", default=None)
    parser.add_argument("--closes", default=None)
    parser.add_argument("--injury-k", type=float, default=0.0,
                       help="points of total per point of injury_net_total_impact")
    parser.add_argument("--bump", type=float, default=0.0, help="subtract from the raw total")
    parser.add_argument("--strong", action="store_true", help="score actionable picks only")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    frame = load_replay_frame(args.ledger, load_closes(args.closes))
    if frame.empty:
        print("no graded rows with a stored edge")
        return 1
    frame = _prepare(frame, load_team_game_log())

    corrected = int((frame["min_games"].notna() & frame["true_gp"].ne(frame["min_games"])).sum())
    missing = int(frame["min_games"].isna().sum())
    print(f"graded rows {len(frame)}   wrong games-played {corrected}   no games count {missing}")
    print()
    print(f"{'variant':<30} {'n':>4}  {'win%':>6}  {'ROI':>7}  CLV (95% CI)")
    print("-" * 88)

    scenarios = [
        ("baseline (as recorded)", dict(fix_games=False)),
        ("games-played corrected", dict(fix_games=True)),
    ]
    if args.injury_k or args.bump:
        scenarios.append((
            f"corrected k={args.injury_k:.2f} bump={args.bump:+.2f}",
            dict(fix_games=True, injury_k=args.injury_k, bump=args.bump),
        ))

    for label, kw in scenarios:
        fix = kw["fix_games"]
        picks = replay(
            frame,
            edge_fn=variant_fn(**kw),
            min_games_fn=(lambda r: r["true_gp"]) if fix else None,
        )
        if args.strong:
            picks = picks[picks["is_actionable"]]
        print(format_summary(summarize(picks, label)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
