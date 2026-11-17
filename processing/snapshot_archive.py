"""Daily lookahead-safe snapshots of lineup/injury/roster inputs.

The replay backtest reconstructs historical features as-of game date for game
logs, but roster ratings, injuries and confirmed lineups are still loaded from
the *current* files. Archiving those files every day lets a future replay use
the data that was actually available on that date instead of future knowledge.

Snapshots are best-effort: a failed copy must never break an alert cycle.
"""

import json
import logging
import os
import shutil
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from config import DATA_DIR, RAW_DIR

logger = logging.getLogger(__name__)

SNAPSHOT_ROOT = os.path.join(DATA_DIR, 'snapshots')
EVENT_LOG_FILENAME = 'events.jsonl'

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


# ---------------------------------------------------------------------------
# Append-only event log
# ---------------------------------------------------------------------------
# Daily file copies answer "what did the files look like at the end of the
# day". They cannot answer "when did the line move" or "how late was the injury
# report". This append-only log records timestamped observations, so the
# injury/lineup latency research can later measure how quickly the pipeline
# reacts instead of reconstructing timing from daily snapshots.
#
# Every helper is best-effort: a failed write must never break an alert cycle.


def event_log_path(root: Optional[str] = None) -> str:
    """Return the append-only event log path."""
    return os.path.join(root or SNAPSHOT_ROOT, EVENT_LOG_FILENAME)


def _as_iso_utc(value: Any = None) -> str:
    """Normalize a timestamp to a UTC ISO-8601 string."""
    if value is None:
        return datetime.now(timezone.utc).isoformat()
    if isinstance(value, datetime):
        moment = value
    else:
        text = str(value).strip()
        try:
            moment = datetime.fromisoformat(text.replace('Z', '+00:00'))
        except ValueError:
            return text
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).isoformat()


def record_event(
    event_type: str,
    game_id: Optional[Any] = None,
    payload: Optional[Dict[str, Any]] = None,
    captured_at: Any = None,
    source: Optional[str] = None,
    root: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Append one timestamped event to ``snapshots/events.jsonl``.

    Returns the written record, or ``None`` when the write failed. Callers must
    treat ``None`` as "logging unavailable", never as an alert failure.
    """
    if not event_type:
        logger.warning("record_event ignored an empty event_type")
        return None

    record = {
        'event_type': str(event_type).strip().lower(),
        'game_id': str(game_id).strip() if game_id not in (None, '') else None,
        'captured_at': _as_iso_utc(captured_at),
        'source': source,
        'payload': payload or {},
    }

    path = event_log_path(root)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')
        return record
    except Exception as exc:
        logger.warning(f"Could not append snapshot event to {path}: {exc}")
        return None


def record_odds_event(
    game_id: Optional[Any],
    total_line: Optional[float] = None,
    previous_total: Optional[float] = None,
    open_total: Optional[float] = None,
    spread_line: Optional[float] = None,
    source: Optional[str] = None,
    commence_time: Any = None,
    captured_at: Any = None,
    root: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Record one odds observation for a game.

    ``total_line`` is the observed total at ``captured_at``. ``previous_total``
    and ``open_total`` are optional, but latency research needs them to separate
    a fresh line from a stale re-observation.
    """
    payload: Dict[str, Any] = {
        'total_line': total_line,
        'previous_total': previous_total,
        'open_total': open_total,
        'spread_line': spread_line,
        'commence_time': _as_iso_utc(commence_time) if commence_time is not None else None,
    }
    if extra:
        payload.update(extra)
    return record_event(
        'odds',
        game_id=game_id,
        payload=payload,
        captured_at=captured_at,
        source=source,
        root=root,
    )


def record_injury_event(
    game_id: Optional[Any],
    team: Optional[str] = None,
    player: Optional[str] = None,
    status: Optional[str] = None,
    previous_status: Optional[str] = None,
    impact: Optional[float] = None,
    source: Optional[str] = None,
    commence_time: Any = None,
    captured_at: Any = None,
    root: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Record one injury-report observation for a game."""
    payload: Dict[str, Any] = {
        'team': team,
        'player': player,
        'status': status,
        'previous_status': previous_status,
        'impact': impact,
        'commence_time': _as_iso_utc(commence_time) if commence_time is not None else None,
    }
    if extra:
        payload.update(extra)
    return record_event(
        'injury',
        game_id=game_id,
        payload=payload,
        captured_at=captured_at,
        source=source,
        root=root,
    )


def record_lineup_event(
    game_id: Optional[Any],
    team: Optional[str] = None,
    starters: Optional[Iterable[Any]] = None,
    status: Optional[str] = None,
    source: Optional[str] = None,
    commence_time: Any = None,
    captured_at: Any = None,
    root: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Record one lineup observation for a game."""
    payload: Dict[str, Any] = {
        'team': team,
        'starters': list(starters or []),
        'status': status,
        'commence_time': _as_iso_utc(commence_time) if commence_time is not None else None,
    }
    if extra:
        payload.update(extra)
    return record_event(
        'lineup',
        game_id=game_id,
        payload=payload,
        captured_at=captured_at,
        source=source,
        root=root,
    )


def read_events(
    event_type: Optional[str] = None,
    game_id: Optional[Any] = None,
    since: Any = None,
    until: Any = None,
    root: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Read the event log, skipping malformed lines instead of failing.

    Filters are applied after parsing. ``since``/``until`` compare the
    ``captured_at`` timestamp as a UTC string, which is safe because every
    written timestamp is normalized to UTC.
    """
    path = event_log_path(root)
    if not os.path.exists(path):
        return []

    wanted_type = str(event_type).strip().lower() if event_type else None
    wanted_gid = str(game_id).strip() if game_id not in (None, '') else None
    since_iso = _as_iso_utc(since) if since is not None else None
    until_iso = _as_iso_utc(until) if until is not None else None

    events: List[Dict[str, Any]] = []
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(f"Skipping malformed event line in {path}")
                    continue
                if not isinstance(event, dict):
                    continue
                if wanted_type and str(event.get('event_type', '')).lower() != wanted_type:
                    continue
                if wanted_gid and str(event.get('game_id')) != wanted_gid:
                    continue
                captured = str(event.get('captured_at', ''))
                if since_iso and captured < since_iso:
                    continue
                if until_iso and captured > until_iso:
                    continue
                events.append(event)
    except Exception as exc:
        logger.warning(f"Could not read snapshot events from {path}: {exc}")
        return []

    return events
