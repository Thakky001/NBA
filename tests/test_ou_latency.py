"""Tests for timestamped snapshot events and latency research."""

import json
import os
import shutil
import tempfile
import unittest

import pandas as pd

from analysis.ou_latency import build_latency_report
from processing.snapshot_archive import (
    event_log_path,
    read_events,
    record_event,
    record_injury_event,
    record_lineup_event,
    record_odds_event,
)


class TestEventLog(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix='nba-events-')

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_record_and_read_roundtrip(self):
        written = record_odds_event(
            game_id='g1',
            total_line=220.5,
            previous_total=221.0,
            source='test',
            commence_time='2026-11-17T23:00:00Z',
            captured_at='2026-11-17T22:00:00Z',
            root=self.root,
        )
        self.assertIsNotNone(written)
        events = read_events(root=self.root)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['event_type'], 'odds')
        self.assertEqual(events[0]['game_id'], 'g1')
        self.assertTrue(os.path.exists(event_log_path(self.root)))

    def test_filters_by_type_game_and_time(self):
        record_odds_event('g1', total_line=220.0, captured_at='2026-11-17T20:00:00Z', root=self.root)
        record_injury_event('g1', player='P1', status='OUT', captured_at='2026-11-17T21:00:00Z', root=self.root)
        record_odds_event('g2', total_line=230.0, captured_at='2026-11-18T20:00:00Z', root=self.root)
        self.assertEqual(len(read_events(event_type='injury', root=self.root)), 1)
        self.assertEqual(len(read_events(game_id='g2', root=self.root)), 1)
        self.assertEqual(
            len(read_events(since='2026-11-17T20:30:00Z', until='2026-11-17T22:00:00Z', root=self.root)),
            1,
        )

    def test_malformed_lines_are_skipped(self):
        record_event('odds', game_id='g1', payload={'total_line': 220}, root=self.root)
        path = event_log_path(self.root)
        with open(path, 'a', encoding='utf-8') as handle:
            handle.write('{not json}\n')
        events = read_events(root=self.root)
        self.assertEqual(len(events), 1)

    def test_record_failure_returns_none(self):
        # A file where a directory is expected makes the append fail safely.
        bad_root = os.path.join(self.root, 'blocked')
        with open(bad_root, 'w', encoding='utf-8') as handle:
            handle.write('x')
        result = record_event('odds', game_id='g1', root=bad_root)
        self.assertIsNone(result)

    def test_empty_log_reads_empty(self):
        self.assertEqual(read_events(root=self.root), [])


class TestLatencyReport(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix='nba-latency-')

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _ledger(self, gids):
        return pd.DataFrame({
            'game_id': gids,
            'date': ['2026-11-17'] * len(gids),
            'total_line': [225.0] * len(gids),
            'actual_total': [230.0] * len(gids),
            'projected_total': [227.0] * len(gids),
            'ou_recommendation': ['STRONG OVER'] * len(gids),
            'ou_edge': [2.0] * len(gids),
            'ou_win_prob': [0.55] * len(gids),
            'features_snapshot': [None] * len(gids),
        })

    def test_no_events_fails_closed(self):
        report = build_latency_report(events=[], ledger=self._ledger([]), root=self.root)
        self.assertEqual(report['status'], 'NO_EVENTS')
        self.assertFalse(report['decision']['production_enabled'])
        self.assertFalse(report['decision']['latency_edge_claimed'])

    def test_odds_movement_is_measured(self):
        for index, line in enumerate((220.0, 222.0)):
            record_odds_event(
                'g1',
                total_line=line,
                source='test',
                commence_time='2026-11-17T23:00:00Z',
                captured_at=f'2026-11-17T2{1 + index}:00:00Z',
                root=self.root,
            )
        report = build_latency_report(events=None, ledger=self._ledger([]), root=self.root)
        odds = report['data']['odds']
        self.assertEqual(odds['games_with_odds_events'], 1)
        self.assertEqual(odds['games_with_multiple_observations'], 1)
        self.assertEqual(odds['median_abs_total_move'], 2.0)
        self.assertEqual(odds['late_line_moves_ge_1pt'], 1)

    def test_injury_report_joins_to_ledger(self):
        gids = [f'g{i:02d}' for i in range(25)]
        for gid in gids:
            record_injury_event(
                gid,
                team='Home',
                player='Player X',
                status='OUT',
                previous_status='QUESTIONABLE',
                source='test',
                commence_time='2026-11-17T23:00:00Z',
                captured_at='2026-11-17T22:00:00Z',
                root=self.root,
            )
        report = build_latency_report(
            events=None,
            ledger=self._ledger(gids),
            root=self.root,
        )
        self.assertEqual(report['status'], 'DESCRIPTIVE_ONLY')
        self.assertEqual(report['data']['injury']['games_with_injury_events'], 25)
        self.assertEqual(report['data']['injury']['games_with_late_injury'], 25)
        self.assertEqual(report['data']['injury']['status_changes'], 25)
        self.assertEqual(report['data']['ledger_join']['joined_games'], 25)
        self.assertEqual(report['data']['ledger_join']['late_injury_games'], 25)
        self.assertFalse(report['decision']['production_enabled'])
        self.assertFalse(report['decision']['latency_edge_claimed'])

    def test_too_few_events_still_fails_closed(self):
        for i in range(5):
            record_injury_event(
                f'g{i}',
                status='OUT',
                commence_time='2026-11-17T23:00:00Z',
                captured_at='2026-11-17T22:00:00Z',
                root=self.root,
            )
        report = build_latency_report(events=None, ledger=self._ledger([]), root=self.root)
        self.assertEqual(report['status'], 'INSUFFICIENT_EVENTS')
        self.assertFalse(report['decision']['production_enabled'])

    def test_lineup_events_are_counted(self):
        record_lineup_event(
            'g1',
            team='Home',
            starters=['A', 'B'],
            status='Confirmed',
            commence_time='2026-11-17T23:00:00Z',
            captured_at='2026-11-17T22:00:00Z',
            root=self.root,
        )
        report = build_latency_report(events=None, ledger=self._ledger([]), root=self.root)
        self.assertEqual(report['data']['lineup']['games_with_lineup_events'], 1)
        self.assertEqual(report['data']['lineup']['lineup_events'], 1)


if __name__ == '__main__':
    unittest.main()
