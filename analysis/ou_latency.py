"""Injury and lineup event latency research.

The production pipeline already archives daily copies of the injury and lineup
inputs, but a daily copy cannot tell us *when* a line moved or how late a
questionable tag became official.  This module reads the append-only event log
written by :mod:`processing.snapshot_archive` and reports:

- how many games have timestamped odds, injury and lineup observations;
- how close to tip-off each observation arrived;
- how large the observed total-line movement was;
- whether late injury news coincides with worse or better O/U outcomes.

Every number is descriptive.  The report never enables betting, and it fails
closed when the event log has too little history.  A prospective latency edge
cannot be established from a handful of events, no matter how promising the
descriptive split looks.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from analysis.ou_evaluation import ou_side, ou_tier
from analysis.prediction_tracker import load_ledger
from processing.snapshot_archive import read_events

WIN_PAYOUT = 100.0 / 110.0
LATE_WINDOW_HOURS = 2.0
MIN_GAMES_FOR_DESCRIPTIVE = 20
MIN_JOINED_GAMES = 10


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_datetime(value: Any) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    try:
        moment = pd.to_datetime(value, errors='coerce', utc=True)
    except (TypeError, ValueError):
        return None
    if moment is None or pd.isna(moment):
        return None
    return moment


def _median(values: List[float]) -> Optional[float]:
    clean = [v for v in values if v is not None and np.isfinite(v)]
    if not clean:
        return None
    return round(float(np.median(clean)), 3)


def _hours_before_tipoff(event: Dict[str, Any]) -> Optional[float]:
    payload = event.get('payload') or {}
    captured = _to_datetime(event.get('captured_at'))
    commence = _to_datetime(payload.get('commence_time'))
    if captured is None or commence is None:
        return None
    return float((commence - captured).total_seconds() / 3600.0)


def _group_events(events: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for event in events:
        gid = event.get('game_id')
        if not gid:
            continue
        grouped.setdefault(str(gid), []).append(event)
    return grouped


def _event_summary(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_type = Counter(str(event.get('event_type', '')) for event in events)
    sources = Counter(
        str(event.get('source')) for event in events if event.get('source')
    )
    captured = [
        _to_datetime(event.get('captured_at'))
        for event in events
        if event.get('captured_at')
    ]
    captured = [value for value in captured if value is not None]
    grouped = _group_events(events)
    return {
        'total_events': len(events),
        'by_type': dict(by_type),
        'sources': dict(sources),
        'unique_games': len(grouped),
        'date_range': (
            [str(min(captured).date()), str(max(captured).date())]
            if captured else None
        ),
    }


def _odds_coverage(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    odds_events = [e for e in events if str(e.get('event_type')) == 'odds']
    grouped = _group_events(odds_events)
    hours = [_hours_before_tipoff(e) for e in odds_events]
    hours = [h for h in hours if h is not None]

    per_game_moves: List[float] = []
    late_moves = 0
    games_with_multiple = 0
    for game_events in grouped.values():
        ordered = sorted(
            game_events,
            key=lambda e: str(e.get('captured_at', '')),
        )
        lines = [
            _to_float((e.get('payload') or {}).get('total_line'))
            for e in ordered
        ]
        lines = [line for line in lines if line is not None and line > 0]
        if len(lines) < 2:
            continue
        games_with_multiple += 1
        move = lines[-1] - lines[0]
        per_game_moves.append(move)
        if abs(move) >= 1.0:
            last_hours = _hours_before_tipoff(ordered[-1])
            if last_hours is not None and last_hours <= LATE_WINDOW_HOURS:
                late_moves += 1

    return {
        'games_with_odds_events': len(grouped),
        'games_with_multiple_observations': games_with_multiple,
        'median_hours_before_tipoff': _median(hours),
        'median_abs_total_move': (
            round(float(np.median(np.abs(per_game_moves))), 3)
            if per_game_moves else None
        ),
        'late_line_moves_ge_1pt': late_moves,
        'games_with_commence_time': len(
            [
                e for e in odds_events
                if (e.get('payload') or {}).get('commence_time')
            ]
        ),
    }


def _injury_coverage(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    injury_events = [e for e in events if str(e.get('event_type')) == 'injury']
    grouped = _group_events(injury_events)
    hours = [_hours_before_tipoff(e) for e in injury_events]
    hours = [h for h in hours if h is not None]
    late_events = [h for h in hours if h <= LATE_WINDOW_HOURS]
    statuses = Counter(
        str((e.get('payload') or {}).get('status') or 'UNKNOWN')
        for e in injury_events
    )
    changed = 0
    for event in injury_events:
        payload = event.get('payload') or {}
        if payload.get('previous_status') and payload.get('status'):
            if str(payload['previous_status']) != str(payload['status']):
                changed += 1
    return {
        'games_with_injury_events': len(grouped),
        'injury_events': len(injury_events),
        'status_changes': changed,
        'median_hours_before_tipoff': _median(hours),
        'late_injury_events_le_2h': len(late_events),
        'games_with_late_injury': len(
            {
                str(e.get('game_id'))
                for e in injury_events
                if (_hours_before_tipoff(e) is not None
                    and _hours_before_tipoff(e) <= LATE_WINDOW_HOURS)
            }
        ),
        'status_counts': dict(statuses),
    }


def _lineup_coverage(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    lineup_events = [e for e in events if str(e.get('event_type')) == 'lineup']
    grouped = _group_events(lineup_events)
    hours = [_hours_before_tipoff(e) for e in lineup_events]
    hours = [h for h in hours if h is not None]
    return {
        'games_with_lineup_events': len(grouped),
        'lineup_events': len(lineup_events),
        'median_hours_before_tipoff': _median(hours),
        'late_lineup_events_le_2h': len(
            [h for h in hours if h <= LATE_WINDOW_HOURS]
        ),
    }


def _ledger_frame(ledger: Optional[pd.DataFrame]) -> pd.DataFrame:
    if ledger is None:
        ledger = load_ledger()
    if ledger is None or ledger.empty:
        return pd.DataFrame()
    required = ['game_id', 'total_line', 'actual_total', 'ou_recommendation']
    if any(col not in ledger.columns for col in required):
        return pd.DataFrame()

    rows = []
    for _, row in ledger.iterrows():
        line = _to_float(row.get('total_line'))
        actual = _to_float(row.get('actual_total'))
        side = ou_side(row)
        rec = str(row.get('ou_recommendation', 'NEUTRAL') or 'NEUTRAL').upper()
        if line is None or line <= 0 or actual is None or actual <= 0 or side is None:
            continue
        if not ('OVER' in rec or 'UNDER' in rec):
            continue
        result = (
            'WIN'
            if (side == 'OVER' and actual > line) or (side == 'UNDER' and actual < line)
            else ('PUSH' if abs(actual - line) < 1e-9 else 'LOSS')
        )
        rows.append({
            'game_id': str(row.get('game_id')),
            'date': row.get('date'),
            'total_line': line,
            'actual_total': actual,
            'side': side,
            'tier': ou_tier(row),
            'recommendation': rec,
            'edge': _to_float(row.get('ou_edge')),
            'result': result,
            'pnl': WIN_PAYOUT if result == 'WIN' else (-1.0 if result == 'LOSS' else 0.0),
        })
    return pd.DataFrame(rows)


def _join_outcomes(
    frame: pd.DataFrame,
    injury_events: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if frame.empty:
        return {'joined_games': 0, 'status': 'NO_LEDGER_PICKS'}

    late_gids = {
        str(e.get('game_id'))
        for e in injury_events
        if (_hours_before_tipoff(e) is not None
            and _hours_before_tipoff(e) <= LATE_WINDOW_HOURS)
    }
    any_injury_gids = {str(e.get('game_id')) for e in injury_events if e.get('game_id')}

    joined = frame[frame['game_id'].isin(any_injury_gids)].copy()
    if joined.empty:
        return {
            'joined_games': 0,
            'status': 'NO_JOINED_GAMES',
            'ledger_picks': int(len(frame)),
            'games_with_injury_events': len(any_injury_gids),
        }

    joined['late_injury'] = joined['game_id'].isin(late_gids)

    def _summary(subset: pd.DataFrame) -> Dict[str, Any]:
        if subset.empty:
            return {'n_picks': 0}
        pnl = subset['pnl'].to_numpy(dtype=float)
        wins = int((subset['result'] == 'WIN').sum())
        losses = int((subset['result'] == 'LOSS').sum())
        return {
            'n_picks': int(len(subset)),
            'wins': wins,
            'losses': losses,
            'win_rate': round(wins / (wins + losses) * 100.0, 2) if wins + losses else None,
            'units': round(float(pnl.sum()), 3),
            'roi_pct': round(float(pnl.mean()) * 100.0, 2) if len(pnl) else None,
            'mean_edge': (
                round(float(subset['edge'].dropna().mean()), 3)
                if subset['edge'].notna().any() else None
            ),
        }

    late = joined[joined['late_injury']]
    early = joined[~joined['late_injury']]
    return {
        'status': (
            'DESCRIPTIVE_ONLY'
            if len(joined) >= MIN_JOINED_GAMES
            else 'INSUFFICIENT_JOINED_GAMES'
        ),
        'ledger_picks': int(len(frame)),
        'games_with_injury_events': len(any_injury_gids),
        'joined_games': int(len(joined)),
        'late_injury_games': int(len(late)),
        'late_injury': _summary(late),
        'no_late_injury': _summary(early),
        'note': (
            'This split is confounded by opponent, line movement and selection; '
            'it is not evidence of a latency edge.'
        ),
    }


def _decision(
    events_summary: Dict[str, Any],
    injury: Dict[str, Any],
    joined: Dict[str, Any],
) -> Dict[str, Any]:
    if events_summary.get('total_events', 0) == 0:
        status = 'NO_EVENTS'
        reason = (
            'the event log is empty; timestamped odds/injury/lineup collection '
            'must run prospectively for several weeks before latency can be measured'
        )
    elif injury.get('injury_events', 0) == 0:
        status = 'INSUFFICIENT_EVENTS'
        reason = 'no injury events have been recorded yet'
    elif injury.get('games_with_injury_events', 0) < MIN_GAMES_FOR_DESCRIPTIVE:
        status = 'INSUFFICIENT_EVENTS'
        reason = (
            f"only {injury.get('games_with_injury_events')} games have injury events; "
            f"need at least {MIN_GAMES_FOR_DESCRIPTIVE} for a descriptive report"
        )
    elif joined.get('status') == 'INSUFFICIENT_JOINED_GAMES':
        status = 'INSUFFICIENT_EVENTS'
        reason = (
            f"only {joined.get('joined_games', 0)} injury-logged games join to a "
            'graded ledger pick'
        )
    else:
        status = 'DESCRIPTIVE_ONLY'
        reason = 'enough events for description, but far too few for a latency edge claim'

    return {
        'status': status,
        'reason': reason,
        'production_enabled': False,
        'paper_only': True,
        'latency_edge_claimed': False,
        'requires_prospective_events': True,
        'minimum_games_for_descriptive_report': MIN_GAMES_FOR_DESCRIPTIVE,
        'late_window_hours': LATE_WINDOW_HOURS,
    }


def build_latency_report(
    events: Optional[List[Dict[str, Any]]] = None,
    ledger: Optional[pd.DataFrame] = None,
    root: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the injury/lineup latency report from the event log."""
    if events is None:
        events = read_events(root=root)

    events_summary = _event_summary(events)
    odds = _odds_coverage(events)
    injury = _injury_coverage(events)
    lineup = _lineup_coverage(events)
    frame = _ledger_frame(ledger)
    joined = _join_outcomes(
        frame,
        [e for e in events if str(e.get('event_type')) == 'injury'],
    )
    decision = _decision(events_summary, injury, joined)

    return {
        'report_version': 'ou-latency-v1.0',
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'status': decision['status'],
        'data': {
            'events': events_summary,
            'odds': odds,
            'injury': injury,
            'lineup': lineup,
            'ledger_join': joined,
        },
        'decision': decision,
        'warnings': [
            'Latency cannot be validated retrospectively: the log starts empty '
            'and only accumulates from forward runs.',
            'The event log captures what this pipeline observed, not the '
            'sportsbook or injury-report timestamps themselves.',
            'Any late-injury split is confounded by line movement, opponent '
            'quality and selection; it must not be treated as an edge.',
            'Production betting stays paused. This report never enables a bet.',
        ],
    }
