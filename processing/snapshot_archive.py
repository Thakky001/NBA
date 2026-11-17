"""Daily lookahead-safe snapshots of lineup/injury/roster inputs.

The replay backtest reconstructs historical features as-of game date for game
logs, but roster ratings, injuries and confirmed lineups are still loaded from
the *current* files. Archiving those files every day lets a future replay use
the data that was actually available on that date instead of future knowledge.

Snapshots are best-effort: a failed copy must never break an alert cycle.
"""

import logging
import os
import shutil
from datetime import datetime
from typing import Any, Dict, Iterable, Optional

from config import DATA_DIR, RAW_DIR

logger = logging.getLogger(__name__)

SNAPSHOT_ROOT = os.path.join(DATA_DIR, 'snapshots')

# Files whose contents are lookahead-prone when replayed later.
SNAPSHOT_FILES = (
    'espn_injuries.csv',
    'rotowire_lineups.json',
    'nba_api_players.csv',
)


def _as_date_str(as_of: Any = None) -> str:
    """Normalize a date-like value to YYYY-MM-DD."""
    if as_of is None:
        return datetime.now().strftime('%Y-%m-%d')
    if hasattr(as_of, 'strftime'):
        return as_of.strftime('%Y-%m-%d')
    return str(as_of)[:10]


def snapshot_dir(as_of: Any = None, root: Optional[str] = None) -> str:
    """Return the snapshot directory path for a given date."""
    return os.path.join(root or SNAPSHOT_ROOT, _as_date_str(as_of))


def archive_daily_snapshot(
    as_of: Any = None,
    raw_dir: Optional[str] = None,
    root: Optional[str] = None,
) -> Dict[str, str]:
    """Copy lookahead-prone inputs into snapshots/<YYYY-MM-DD>/.

    Args:
        as_of: Date of the snapshot (defaults to today).
        raw_dir: Source directory (defaults to config.RAW_DIR).
        root: Snapshot root (defaults to data/snapshots).

    Returns:
        Mapping of archived filename -> destination path. Empty on failure.
    """
    dest_dir = snapshot_dir(as_of, root)
    src_dir = raw_dir or RAW_DIR
    archived: Dict[str, str] = {}

    try:
        os.makedirs(dest_dir, exist_ok=True)
    except Exception as exc:
        logger.warning(f"Could not create snapshot directory {dest_dir}: {exc}")
        return archived

    for filename in SNAPSHOT_FILES:
        src = os.path.join(src_dir, filename)
        if not os.path.exists(src):
            continue
        dst = os.path.join(dest_dir, filename)
        try:
            shutil.copy2(src, dst)
            archived[filename] = dst
        except Exception as exc:
            logger.warning(f"Could not archive {filename} to {dest_dir}: {exc}")

    if archived:
        logger.info(f"Archived {len(archived)} daily snapshot file(s) to {dest_dir}")
    return archived


def has_snapshot(as_of: Any = None, root: Optional[str] = None) -> bool:
    """Return True if at least one archived input exists for the given date."""
    directory = snapshot_dir(as_of, root)
    return any(os.path.exists(os.path.join(directory, f)) for f in SNAPSHOT_FILES)


def snapshot_coverage(dates: Iterable[Any], root: Optional[str] = None) -> Dict[str, Any]:
    """Summarize how many replayed dates have an archived snapshot available."""
    unique = sorted({_as_date_str(d) for d in dates if d is not None})
    covered = [d for d in unique if has_snapshot(d, root)]
    return {
        'dates_replayed': len(unique),
        'dates_with_snapshot': len(covered),
        'covered_dates': covered,
    }
