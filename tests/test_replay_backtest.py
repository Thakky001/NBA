"""Tests for replay metrics and go-live gate evaluation."""

import os
import shutil
import tempfile
import unittest

from analysis.replay_backtest import (
    MIN_METRIC_SAMPLE,
    REPLAY_LIMITATIONS,
    compute_go_live_checks,
    evaluate_slice,
)
from processing.snapshot_archive import archive_daily_snapshot, snapshot_coverage


def _record(result='WIN', pnl=None, cal_prob=0.55, raw_prob=0.60):
    if pnl is None:
        pnl = 0.909 if result == 'WIN' else (-1.0 if result == 'LOSS' else 0.0)
    return {
        'result': result,
        'pnl': pnl,
        'cal_prob': cal_prob,
        'raw_prob': raw_prob,
        'binary_win': 1 if result == 'WIN' else (0 if result == 'LOSS' else None),
    }


def _well_calibrated_slice(n=100):
    """67 wins / 33 losses at a constant 67% calibrated probability.

    Single probability bin -> observed rate equals mean probability (ECE 0),
    interleaved results keep drawdown small, and ROI is comfortably positive.
    """
    records = []
    for i in range(n):
        result = 'WIN' if i % 3 in (0, 1) else 'LOSS'
        records.append(_record(result, cal_prob=0.67, raw_prob=0.70))
    return evaluate_slice(records)


class TestEvaluateSlice(unittest.TestCase):

    def test_empty_slice_has_no_metrics(self):
        res = evaluate_slice([])
        self.assertEqual(res['total_picks'], 0)
        self.assertEqual(res['graded_picks'], 0)
        self.assertFalse(res['sufficient_sample'])
        self.assertIsNone(res['win_rate'])
        self.assertIsNone(res['roi_pct'])
        self.assertIsNone(res['max_drawdown_units'])
        self.assertIsNone(res['calibration_error_pct'])
        self.assertIsNone(res['raw_calibration_error_pct'])

    def test_small_slice_has_no_metrics(self):
        res = evaluate_slice([_record('WIN') for _ in range(5)])
        self.assertEqual(res['graded_picks'], 5)
        self.assertFalse(res['sufficient_sample'])
        self.assertIsNone(res['roi_pct'])
        self.assertIsNone(res['calibration_error_pct'])

    def test_sufficient_slice_populates_metrics(self):
        res = evaluate_slice([_record('WIN') for _ in range(MIN_METRIC_SAMPLE)])
        self.assertTrue(res['sufficient_sample'])
        self.assertAlmostEqual(res['win_rate'], 100.0, places=2)
        self.assertAlmostEqual(res['roi_pct'], 90.9, places=2)
        self.assertIsNotNone(res['calibration_error_pct'])


class TestGoLiveChecks(unittest.TestCase):

    def test_empty_strong_slice_cannot_pass(self):
        checks = compute_go_live_checks(evaluate_slice([]))
        self.assertFalse(checks['criterion_1_min_100_strong_picks'])
        self.assertIsNone(checks['criterion_2_strong_roi_above_2pct'])
        self.assertIsNone(checks['criterion_3_strong_calibration_under_5pct'])
        self.assertIsNone(checks['criterion_4_strong_max_dd_under_10u'])
        self.assertFalse(checks['all_passed'])

    def test_small_perfect_slice_cannot_pass_metrics(self):
        small = evaluate_slice([_record('WIN') for _ in range(10)])
        checks = compute_go_live_checks(small)
        self.assertFalse(checks['criterion_1_min_100_strong_picks'])
        self.assertIsNone(checks['criterion_2_strong_roi_above_2pct'])
        self.assertFalse(checks['all_passed'])

    def test_100_well_calibrated_strong_picks_pass(self):
        checks = compute_go_live_checks(_well_calibrated_slice(100))
        self.assertTrue(checks['criterion_1_min_100_strong_picks'])
        self.assertTrue(checks['criterion_2_strong_roi_above_2pct'])
        self.assertTrue(checks['criterion_3_strong_calibration_under_5pct'])
        self.assertTrue(checks['criterion_4_strong_max_dd_under_10u'])
        self.assertTrue(checks['all_passed'])

    def test_100_picks_with_bad_roi_fail(self):
        records = [_record('WIN', cal_prob=0.58) for _ in range(45)]
        records += [_record('LOSS', cal_prob=0.58) for _ in range(55)]
        checks = compute_go_live_checks(evaluate_slice(records))
        self.assertTrue(checks['criterion_1_min_100_strong_picks'])
        self.assertFalse(checks['criterion_2_strong_roi_above_2pct'])
        self.assertFalse(checks['all_passed'])

    def test_100_picks_with_bad_calibration_fail(self):
        records = [_record('WIN', cal_prob=0.90) for _ in range(50)]
        records += [_record('LOSS', cal_prob=0.90) for _ in range(50)]
        checks = compute_go_live_checks(evaluate_slice(records))
        self.assertTrue(checks['criterion_1_min_100_strong_picks'])
        self.assertFalse(checks['criterion_3_strong_calibration_under_5pct'])
        self.assertFalse(checks['all_passed'])

    def test_limitations_are_reported(self):
        self.assertTrue(REPLAY_LIMITATIONS)
        joined = ' '.join(REPLAY_LIMITATIONS).lower()
        self.assertIn('lookahead', joined)
        self.assertIn('snapshot', joined)


class TestSnapshotArchive(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.raw = os.path.join(self.tmp, 'raw')
        self.root = os.path.join(self.tmp, 'snapshots')
        os.makedirs(self.raw)
        with open(os.path.join(self.raw, 'espn_injuries.csv'), 'w', encoding='utf-8') as f:
            f.write('team_name,player_name,status\nBOS,Jayson Tatum,Questionable\n')
        with open(os.path.join(self.raw, 'rotowire_lineups.json'), 'w', encoding='utf-8') as f:
            f.write('{"BOS": {"lineup_status": "Projected"}}')

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_archive_copies_expected_files(self):
        archived = archive_daily_snapshot('2026-11-16', raw_dir=self.raw, root=self.root)
        self.assertIn('espn_injuries.csv', archived)
        self.assertIn('rotowire_lineups.json', archived)
        self.assertTrue(os.path.exists(os.path.join(self.root, '2026-11-16', 'espn_injuries.csv')))

    def test_missing_source_files_are_skipped(self):
        archived = archive_daily_snapshot('2026-11-16', raw_dir=self.raw, root=self.root)
        self.assertNotIn('nba_api_players.csv', archived)

    def test_snapshot_coverage_counts_dates(self):
        archive_daily_snapshot('2026-11-16', raw_dir=self.raw, root=self.root)
        coverage = snapshot_coverage(['2026-11-16', '2026-11-16', '2026-11-15'], root=self.root)
        self.assertEqual(coverage['dates_replayed'], 2)
        self.assertEqual(coverage['dates_with_snapshot'], 1)
        self.assertEqual(coverage['covered_dates'], ['2026-11-16'])


if __name__ == '__main__':
    unittest.main()
