"""Backfill closing totals/spreads for every game in the prediction ledger.

The ledger records what the line was when the pick was made, but never what it
closed at, so closing-line value could not be measured after the fact. ESPN
keeps the pickcenter open/close pair for completed games, so this script
recovers it: one row per ``game_id``, written to ``data/raw/espn_closing_lines.csv``.

The file is the input for ``analysis/ou_replay.py``. Run it again whenever the
ledger grows; rows that already have a closing total are kept and only the new
games are fetched.

Usage
-----
    python scripts/backfill_closing_lines.py
    python scripts/backfill_closing_lines.py --force     # refetch everything
    python scripts/backfill_closing_lines.py --limit 20  # smoke test
"""

import argparse
import logging
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import RAW_DIR  # noqa: E402

logger = logging.getLogger("backfill_closing_lines")

OUTPUT_PATH = os.path.join(RAW_DIR, "espn_closing_lines.csv")
COLUMNS = ["game_id", "open_total", "close_total", "open_spread", "close_spread", "provider"]


def ledger_game_ids(ledger_path: str) -> list:
    """Distinct game ids that have a stored total line, in date order."""
    ledger = pd.read_csv(ledger_path, low_memory=False)
    played = ledger[ledger["total_line"].notna()]
    played = played.drop_duplicates(subset=["game_id"])
    return [str(g) for g in played["game_id"].tolist()]


def load_existing(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(path, dtype={"game_id": str})
    return df


def fetch(collector, game_id: str) -> dict:
    """One game's open/close pair, normalised to the output schema."""
    try:
        movement = collector.get_game_line_movement(game_id)
    except Exception as exc:  # network failures must not poison the file
        logger.warning(f"{game_id}: fetch failed ({exc})")
        return {"game_id": game_id}
    return {
        "game_id": game_id,
        "open_total": movement.get("open_total"),
        "close_total": movement.get("current_total"),
        "open_spread": movement.get("open_spread"),
        "close_spread": movement.get("current_spread"),
        "provider": movement.get("provider"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", default=os.path.join(RAW_DIR, "predictions_ledger.csv"))
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--force", action="store_true", help="refetch rows that already have a close")
    parser.add_argument("--limit", type=int, default=0, help="only fetch the first N missing games")
    parser.add_argument("--delay", type=float, default=0.25, help="seconds between requests")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    game_ids = ledger_game_ids(args.ledger)
    existing = load_existing(args.output)
    logger.info(f"ledger games: {len(game_ids)}   already backfilled: {len(existing)}")

    done = set(existing["game_id"]) if not args.force else set()
    if not args.force:
        # A row without a close is retried: ESPN may not have had odds at the time.
        have_close = existing[existing["close_total"].notna()]["game_id"]
        done = set(have_close)

    todo = [g for g in game_ids if g not in done]
    if args.limit:
        todo = todo[: args.limit]
    logger.info(f"to fetch: {len(todo)}")
    if not todo:
        return 0

    from collectors.espn_collector import ESPNCollector

    collector = ESPNCollector()
    fresh = []
    for i, game_id in enumerate(todo, 1):
        fresh.append(fetch(collector, game_id))
        if i % 25 == 0 or i == len(todo):
            logger.info(f"  fetched {i}/{len(todo)}")
        time.sleep(args.delay)

    new_rows = pd.DataFrame(fresh, columns=COLUMNS)
    new_rows = new_rows[new_rows["close_total"].notna()]

    merged = pd.concat([existing[existing["game_id"].isin(set(game_ids))], new_rows], ignore_index=True)
    merged = merged.drop_duplicates(subset=["game_id"], keep="last")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    merged.to_csv(args.output, index=False)

    covered = int(merged["close_total"].notna().sum())
    logger.info(f"wrote {len(merged)} rows to {args.output}")
    logger.info(f"closing totals available: {covered}/{len(game_ids)}")
    if covered < len(game_ids):
        logger.warning("some games still lack a closing total; rerun later to retry them")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
