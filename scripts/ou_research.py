#!/usr/bin/env python3
"""O/U research CLI.

Subcommands map to the five workstreams in the O/U hardening plan:

- baseline    : describe the current ledger and reproduce the known metrics
- walkforward : market-anchored residual walk-forward evaluation
- calibrate   : walk-forward probability calibration comparison
- latency     : injury/lineup snapshot and latency report
- subsets     : pre-registered situational subset research
- all         : run every report that has enough data

Reports are written to ``reports/ou/`` as JSON. The command is read-only with
respect to the predictions ledger.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.ou_evaluation import evaluate_ou_frame
from analysis.prediction_tracker import load_ledger


REPORT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'reports', 'ou'
)


def _json_default(value: Any) -> Any:
    if hasattr(value, 'item'):
        return value.item()
    return str(value)


def _write_report(name: str, payload: Dict[str, Any]) -> str:
    os.makedirs(REPORT_DIR, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    path = os.path.join(REPORT_DIR, f'{name}_{stamp}.json')
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=_json_default)
    return path


def _print_overall(report: Dict[str, Any]) -> None:
    overall = report.get('overall', {})
    market = report.get('market_baseline', {})
    print('=' * 72)
    print('O/U BASELINE')
    print('=' * 72)
    print(f"picks={overall.get('n_picks')}  W-L-P={overall.get('wins')}-{overall.get('losses')}-{overall.get('pushes')}")
    print(f"win%={overall.get('win_rate')}  units={overall.get('units')}  ROI={overall.get('roi_pct')}%")
    print(f"model MAE={overall.get('model_mae')}  market MAE={market.get('market_mae')}")
    print(f"bias(actual-line)={market.get('bias_actual_minus_line')}  corr(edge, miss)={overall.get('corr_edge_miss')}")
    print(f"Brier={overall.get('brier')}  logloss={overall.get('log_loss')}  ECE={overall.get('calibration_error')}")
    print(f"CLV points={overall.get('mean_clv_points')} (n={overall.get('clv_sample')})")
    for label in ('by_side', 'by_tier', 'by_edge_bucket', 'by_gp_bucket'):
        groups = report.get(label) or {}
        if groups:
            print(f"\n{label}:")
            for key, value in groups.items():
                print(
                    f"  {key:12s} n={value.get('n_picks'):3d} "
                    f"W-L={value.get('wins')}-{value.get('losses')} "
                    f"win%={value.get('win_rate')} ROI={value.get('roi_pct')}%"
                )
    print('=' * 72)


def run_baseline(ledger_path: str | None = None, min_sample: int = 1) -> Dict[str, Any]:
    ledger = load_ledger()
    if ledger_path:
        import pandas as pd

        ledger = pd.read_csv(ledger_path, dtype={'game_id': str})
    else:
        # Use a local copy so the CLI never mutates the ledger.
        ledger = ledger.copy()

    report = evaluate_ou_frame(ledger, min_sample=min_sample)
    report['generated_at'] = datetime.now(timezone.utc).isoformat()
    report['ledger_rows'] = int(len(ledger))
    if 'date' in ledger.columns and len(ledger):
        report['date_range'] = [str(ledger['date'].min()), str(ledger['date'].max())]
    path = _write_report('baseline', report)
    report['_report_path'] = path
    _print_overall(report)
    return report


def _load_walkforward():
    from analysis.ou_walkforward import run_market_residual_walkforward

    return run_market_residual_walkforward()


def _load_calibration():
    from analysis.ou_calibration_research import run_calibration_research

    return run_calibration_research()


def _load_latency():
    from analysis.ou_latency import build_latency_report

    return build_latency_report()


def _load_subsets():
    from analysis.ou_subsets import run_subset_research

    ledger = load_ledger().copy()
    return run_subset_research(ledger)


def main() -> int:
    parser = argparse.ArgumentParser(description='O/U research harness')
    parser.add_argument(
        'command',
        choices=['baseline', 'walkforward', 'calibrate', 'latency', 'subsets', 'all'],
        nargs='?',
        default='baseline',
    )
    parser.add_argument('--ledger', type=str, default=None, help='Optional ledger CSV path')
    parser.add_argument('--min-sample', type=int, default=1, help='Minimum decided picks for metrics')
    args = parser.parse_args()

    commands = [args.command]
    if args.command == 'all':
        commands = ['baseline', 'walkforward', 'calibrate', 'latency', 'subsets']

    for command in commands:
        if command == 'baseline':
            run_baseline(args.ledger, args.min_sample)
            continue
        loader = {
            'walkforward': _load_walkforward,
            'calibrate': _load_calibration,
            'latency': _load_latency,
            'subsets': _load_subsets,
        }[command]
        report = loader()
        path = _write_report(command, report)
        print(f"\n[{command}] report written to {path}")
        print(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default)[:4000])

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
