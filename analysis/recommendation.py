"""Single unified recommendation gate and pick decision engine for NBA betting intelligence.

Eliminates duplicate threshold logic across run_alerts.py, trend_analyzer.py,
and prediction_tracker.py.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

GATE_VERSION = "gate-v2.1"


def decide_pick(
    market_type: str,
    current_line: Optional[float],
    projected_val: Optional[float],
    edge: Optional[float],
    fair_prob: Optional[float],
    line_movement: Optional[Dict[str, Any]] = None,
    open_line: Optional[float] = None,
    home_team: Optional[str] = None,
    away_team: Optional[str] = None,
    data_quality_flags: Optional[Dict[str, Any]] = None,
    is_early_season: bool = False,
    push_prob: float = 0.0,
    ev: Optional[float] = None
) -> Dict[str, Any]:
    """Unified decision gate for Spread and Over/Under bets.
    
    Returns a standardized dictionary containing:
        - recommendation: e.g. 'STRONG HOME', 'LEAN AWAY', 'STRONG OVER', 'LEAN UNDER', 'NEUTRAL'
        - pick_side: 'HOME', 'AWAY', 'OVER', 'UNDER', or None
        - pick_prob: float probability of the recommended side (never 0.5 for actionable picks)
        - anchor_line: current line used as anchor
        - current_line: current line
        - open_line: open line
        - edge_vs_current: edge against current line
        - edge_vs_open: edge against open line
        - is_actionable: bool
        - suppressed_reason: None or str reason for suppression/veto/cap
        - gate_version: GATE_VERSION
        - data_quality_flags: dict
    """
    m_type = str(market_type).lower().strip()
    dq = data_quality_flags or {}
    
    # Base fallback response for unanalyzable matchups
    fallback = {
        'recommendation': 'NEUTRAL',
        'pick_side': None,
        'pick_prob': 0.50,
        'anchor_line': current_line,
        'current_line': current_line,
        'open_line': open_line,
        'edge_vs_current': 0.0,
        'edge_vs_open': 0.0,
        'is_actionable': False,
        'suppressed_reason': None,
        'gate_version': GATE_VERSION,
        'data_quality_flags': dq
    }

    if current_line is None or projected_val is None or edge is None or fair_prob is None:
        fallback['suppressed_reason'] = 'MISSING_LINES_OR_PROJECTION'
        return fallback

    # Compute edge against current line and open line
    edge_vs_curr = round(float(edge), 2)
    if open_line is not None:
        try:
            o_val = float(open_line)
            if m_type in ['spread', 'sp']:
                edge_vs_op = round(float(projected_val) + o_val, 2)
            else:
                edge_vs_op = round(float(projected_val) - o_val, 2)
        except (ValueError, TypeError):
            edge_vs_op = edge_vs_curr
    else:
        edge_vs_op = edge_vs_curr

    # Guard 1: Data Freshness Check (Phase 1.4 Safe Veto)
    if dq.get('dq_stale_data') is True:
        fallback.update({
            'edge_vs_current': edge_vs_curr,
            'edge_vs_open': edge_vs_op,
            'suppressed_reason': 'SUPPRESSED_STALE_DATA'
        })
        return fallback

    # Guard 2: Data Quality Reliability Veto (Phase 5 of Stabilization)
    if dq.get('dq_is_reliable') is False and abs(edge_vs_curr) >= 6.0:
        fallback.update({
            'edge_vs_current': edge_vs_curr,
            'edge_vs_open': edge_vs_op,
            'suppressed_reason': 'VETO_UNRELIABLE_DATA_QUALITY'
        })
        return fallback

    # Guard 3: Explicit Market Clash (only when flagged as true clash)
    lm = line_movement or {}
    if lm.get('is_market_clash', False) is True:
        fallback.update({
            'edge_vs_current': edge_vs_curr,
            'edge_vs_open': edge_vs_op,
            'suppressed_reason': f"MARKET_CLASH_{lm.get('verdict', 'CONFLICT')}"
        })
        return fallback

    suppressed_reason = None
    rec = 'NEUTRAL'
    pick_side = None
    pick_prob = 0.50

    if m_type in ['spread', 'sp']:
        # edge = projected_margin + current_spread
        # fair_prob is probability of HOME covering
        # When edge > 0: model favors HOME
        # When edge < 0: model favors AWAY
        if edge_vs_curr > 0:
            cand_side = 'HOME'
            cand_prob = float(fair_prob)
        else:
            cand_side = 'AWAY'
            cand_prob = round(1.0 - float(fair_prob) - float(push_prob), 6)

        # Spread Gates with Early-Season Asymmetry
        is_road_heavy_fav = (current_line is not None and current_line >= 9.5)
        is_road_underdog = (current_line is not None and current_line < 0.0)

        if is_early_season:
            strong_edge = 3.0
            strong_prob = 0.58
            lean_edge = 2.0
            lean_prob = 0.535

            if cand_side == 'AWAY':
                if is_road_heavy_fav:
                    strong_edge = 6.5
                    strong_prob = 0.62
                    lean_edge = 4.5
                    lean_prob = 0.58
                elif is_road_underdog:
                    strong_edge = 4.0
                    strong_prob = 0.58
                    lean_edge = 2.0
                    lean_prob = 0.535
        else:
            strong_edge = 3.0
            strong_prob = 0.58
            lean_edge = 1.5
            lean_prob = 0.535

        abs_edge = abs(edge_vs_curr)
        if abs_edge >= strong_edge and cand_prob >= strong_prob:
            rec = f"STRONG {cand_side}"
            pick_side = cand_side
            pick_prob = cand_prob
        elif abs_edge >= lean_edge and cand_prob >= lean_prob:
            rec = f"LEAN {cand_side}"
            pick_side = cand_side
            pick_prob = cand_prob

        # Phase 2.5: Extreme Edge Soft Cap (|edge| > 8.0 capped to LEAN)
        if abs_edge > 8.0 and 'STRONG' in rec:
            rec = f"LEAN {cand_side}"
            suppressed_reason = 'EXTREME_EDGE_SOFT_CAPPED_TO_LEAN'

    else:
        # Over/Under
        # edge = projected_total - current_total
        # fair_prob is probability of OVER
        if edge_vs_curr > 0:
            cand_side = 'OVER'
            cand_prob = float(fair_prob)
            if edge_vs_curr >= 5.0 and cand_prob >= 0.565 and (ev is None or ev >= 0.05):
                rec = 'STRONG OVER'
                pick_side = 'OVER'
                pick_prob = cand_prob
            elif edge_vs_curr >= 3.0 and cand_prob >= 0.535 and (ev is None or ev >= 0.015):
                rec = 'LEAN OVER'
                pick_side = 'OVER'
                pick_prob = cand_prob
        else:
            cand_side = 'UNDER'
            cand_prob = round(1.0 - float(fair_prob) - float(push_prob), 6)
            abs_edge = abs(edge_vs_curr)
            if abs_edge >= 2.0 and cand_prob >= 0.565 and (ev is None or ev >= 0.05):
                rec = 'STRONG UNDER'
                pick_side = 'UNDER'
                pick_prob = cand_prob
            elif abs_edge >= (1.5 if is_early_season else 1.0) and cand_prob >= 0.535 and (ev is None or ev >= 0.015):
                rec = 'LEAN UNDER'
                pick_side = 'UNDER'
                pick_prob = cand_prob

        # Phase 2.5: Extreme Edge Soft Cap (|edge| > 8.0 capped to LEAN)
        if abs(edge_vs_curr) > 8.0 and 'STRONG' in rec:
            rec = f"LEAN {cand_side}"
            suppressed_reason = 'EXTREME_EDGE_SOFT_CAPPED_TO_LEAN'

    is_act = (rec != 'NEUTRAL')
    if not is_act:
        pick_side = None
        pick_prob = 0.50
    else:
        # Bug 2.3 Assertion: Actionable picks MUST NOT have pick_prob == 0.50
        if abs(pick_prob - 0.50) < 0.001:
            logger.error(f"Bug 2.3 detected: {rec} produced ambiguous pick_prob=0.50. Reverting to NEUTRAL.")
            rec = 'NEUTRAL'
            pick_side = None
            pick_prob = 0.50
            is_act = False
            suppressed_reason = 'BUG_AMBIGUOUS_PROBABILITY'

    return {
        'recommendation': rec,
        'pick_side': pick_side,
        'pick_prob': round(pick_prob, 4),
        'anchor_line': current_line,
        'current_line': current_line,
        'open_line': open_line,
        'edge_vs_current': edge_vs_curr,
        'edge_vs_open': edge_vs_op,
        'is_actionable': is_act,
        'suppressed_reason': suppressed_reason,
        'gate_version': GATE_VERSION,
        'data_quality_flags': dq
    }
