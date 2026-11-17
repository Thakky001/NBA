"""Single unified recommendation gate and pick decision engine for NBA betting intelligence.

Eliminates duplicate threshold logic across run_alerts.py, trend_analyzer.py,
and prediction_tracker.py.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

GATE_VERSION = "gate-v2.4-strict-sample"
SELECTIVE_GATE_VERSION = "ou-selective-v1.0"
SELECTIVE_GATE_ENV = "NBA_OU_SELECTIVE_GATE"
SELECTIVE_PROFILE_ENV = "NBA_OU_SELECTIVE_PROFILE"
_TRUTHY = {"true", "1", "yes", "on"}

# Minimum current-season games per team before a STRONG pick becomes actionable.
# Graduated 8-14 band was reverted (v2.4) because there is no out-of-sample
# evidence that early-season conviction is well calibrated.
STRONG_MIN_GAMES = 15


def selective_gate_enabled(environ: Optional[Dict[str, str]] = None) -> bool:
    """Return whether the O/U selective gate is enabled (default: off)."""
    env = os.environ if environ is None else environ
    return str(env.get(SELECTIVE_GATE_ENV, "false")).strip().lower() in _TRUTHY


def load_selective_profile(
    environ: Optional[Dict[str, str]] = None,
) -> Optional[Dict[str, Any]]:
    """Load the selective profile from ``NBA_OU_SELECTIVE_PROFILE``.

    The value is inline JSON or a path to a JSON file. Any malformed value
    returns ``None`` so the gate fails closed instead of guessing a profile.
    """
    env = os.environ if environ is None else environ
    raw = env.get(SELECTIVE_PROFILE_ENV)
    if not raw:
        return None
    text = str(raw).strip()
    try:
        if text.startswith("{"):
            profile = json.loads(text)
        else:
            with open(text, "r", encoding="utf-8") as handle:
                profile = json.load(handle)
    except Exception as exc:
        logger.warning(f"Could not load selective O/U profile: {exc}")
        return None
    if not isinstance(profile, dict):
        logger.warning("Selective O/U profile must be a JSON object")
        return None
    return profile


def _selective_rule_match(
    profile: Optional[Dict[str, Any]],
    recommendation: str,
    abs_edge: float,
    pick_prob: float,
    min_games: Optional[int],
) -> Optional[str]:
    """Return the first matching pre-registered rule, or ``None``.

    The selector only downgrades. It can never upgrade a LEAN to STRONG and it
    never invents a pick that the base gate rejected.
    """
    if not isinstance(profile, dict):
        return None
    allowed = profile.get("allowed_rules")
    if not isinstance(allowed, list) or not allowed:
        return None

    if "STRONG" in recommendation:
        tier = "STRONG"
    elif "LEAN" in recommendation:
        tier = "LEAN"
    else:
        return None
    side = "OVER" if "OVER" in recommendation else ("UNDER" if "UNDER" in recommendation else None)
    if side is None:
        return None

    min_edge = profile.get("min_edge")
    if min_edge is not None and abs_edge < float(min_edge):
        return None
    min_prob = profile.get("min_prob")
    if min_prob is not None and pick_prob < float(min_prob):
        return None
    required_games = profile.get("min_games")
    if required_games is not None:
        if min_games is None or int(min_games) < int(required_games):
            return None
    gp15 = min_games is not None and int(min_games) >= STRONG_MIN_GAMES
    prob = pick_prob

    for rule in allowed:
        rule = str(rule)
        if rule == "all_picks":
            return rule
        if rule == "strong_only" and tier == "STRONG":
            return rule
        if rule == "strong_gp15" and tier == "STRONG" and gp15:
            return rule
        if rule == "under_only" and side == "UNDER":
            return rule
        if rule == "strong_under" and tier == "STRONG" and side == "UNDER":
            return rule
        if rule == "edge_ge_2" and abs_edge >= 2.0:
            return rule
        if rule == "prob_ge_053" and prob >= 0.53:
            return rule
        if rule == "edge_ge_2_prob_ge_053" and abs_edge >= 2.0 and prob >= 0.53:
            return rule
    return None


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
    ev: Optional[float] = None,
    selective_profile: Optional[Dict[str, Any]] = None,
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
        - is_actionable: bool (True ONLY for STRONG on mature samples)
        - suppressed_reason: None or str reason for suppression/veto/cap
        - gate_version: GATE_VERSION
        - data_quality_flags: dict
    """
    m_type = str(market_type).lower().strip()
    dq = data_quality_flags or {}
    gate_on = selective_gate_enabled()
    
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
        'selective_gate_version': SELECTIVE_GATE_VERSION if gate_on else None,
        'selective_gate_enabled': gate_on,
        'selective_rule': None,
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

    # Extract sample size
    h_g = dq.get('home_games_played')
    a_g = dq.get('away_games_played')
    min_games = None
    if h_g is not None and a_g is not None:
        try:
            min_games = min(int(h_g), int(a_g))
        except (ValueError, TypeError):
            pass

    # Guard 1: Data Freshness Check (Phase 1.4 Safe Veto)
    if dq.get('dq_stale_data') is True:
        fallback.update({
            'edge_vs_current': edge_vs_curr,
            'edge_vs_open': edge_vs_op,
            'suppressed_reason': 'SUPPRESSED_STALE_DATA'
        })
        return fallback

    # Guard 2: Unconditional Data Quality & Low Sample Veto
    is_reliable = dq.get('dq_is_reliable', True)
    low_sample = dq.get('dq_low_sample', False) or (min_games is not None and min_games < 5)
    if is_reliable is False or low_sample is True:
        fallback.update({
            'edge_vs_current': edge_vs_curr,
            'edge_vs_open': edge_vs_op,
            'suppressed_reason': 'VETO_UNRELIABLE_DATA_QUALITY' if is_reliable is False else 'VETO_LOW_SAMPLE_SIZE'
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

    abs_edge = abs(edge_vs_curr)

    # Guard 4: Hard Cap on Model Anomalies (|edge| > 8.0 pts on Spread or Total)
    if abs_edge > 8.0:
        fallback.update({
            'edge_vs_current': edge_vs_curr,
            'edge_vs_open': edge_vs_op,
            'suppressed_reason': 'EXTREME_EDGE_MODEL_ANOMALY'
        })
        return fallback

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

        # Spread Gates with Calibrated Probabilities & Early-Season Asymmetry
        is_road_heavy_fav = (current_line is not None and current_line >= 9.5)
        is_road_underdog = (current_line is not None and current_line < 0.0)

        # Baseline calibrated thresholds (empirically grounded to post-calibration scale)
        # In sports betting (-110 line), breakeven is 52.38%.
        # A calibrated prob >= 53.0% (raw ~56.5-58%) yields +2.5% to +6% ROI.
        strong_edge = 3.0
        strong_prob = 0.530
        lean_edge = 1.5
        lean_prob = 0.515

        if is_early_season:
            lean_edge = 2.0
            if cand_side == 'AWAY':
                if is_road_heavy_fav:
                    strong_edge = 6.5
                    strong_prob = 0.560
                    lean_edge = 4.5
                    lean_prob = 0.530
                elif is_road_underdog:
                    strong_edge = 4.0
                    strong_prob = 0.540
                    lean_edge = 2.0
                    lean_prob = 0.515

        abs_edge = abs(edge_vs_curr)
        if abs_edge >= strong_edge and cand_prob >= strong_prob:
            rec = f"STRONG {cand_side}"
            pick_side = cand_side
            pick_prob = cand_prob
        elif abs_edge >= lean_edge and cand_prob >= lean_prob:
            rec = f"LEAN {cand_side}"
            pick_side = cand_side
            pick_prob = cand_prob

    else:
        # Over/Under Gates with Calibrated Probabilities
        # edge = projected_total - current_total
        # fair_prob is calibrated probability of OVER
        if edge_vs_curr > 0:
            cand_side = 'OVER'
            cand_prob = float(fair_prob)
            if edge_vs_curr >= 5.0 and cand_prob >= 0.530 and (ev is None or ev >= 0.015):
                rec = 'STRONG OVER'
                pick_side = 'OVER'
                pick_prob = cand_prob
            elif edge_vs_curr >= 3.0 and cand_prob >= 0.515 and (ev is None or ev >= 0.005):
                rec = 'LEAN OVER'
                pick_side = 'OVER'
                pick_prob = cand_prob
        else:
            cand_side = 'UNDER'
            cand_prob = round(1.0 - float(fair_prob) - float(push_prob), 6)
            abs_edge = abs(edge_vs_curr)
            if abs_edge >= 2.0 and cand_prob >= 0.530 and (ev is None or ev >= 0.015):
                rec = 'STRONG UNDER'
                pick_side = 'UNDER'
                pick_prob = cand_prob
            elif abs_edge >= (1.5 if is_early_season else 1.0) and cand_prob >= 0.515 and (ev is None or ev >= 0.005):
                rec = 'LEAN UNDER'
                pick_side = 'UNDER'
                pick_prob = cand_prob

    # Strict Sample Gating (gate v2.4):
    # STRONG requires a verified current-season sample of >= STRONG_MIN_GAMES for
    # BOTH teams. The graduated 8-14 band was reverted because there is no
    # out-of-sample evidence it is safe; anything below the floor is LEAN only.
    # Unknown sample size fails closed (never actionable).
    if 'STRONG' in rec:
        if min_games is None:
            rec = f"LEAN {pick_side}"
            suppressed_reason = 'SAMPLE_SIZE_UNVERIFIED'
        elif min_games < STRONG_MIN_GAMES:
            rec = f"LEAN {pick_side}"
            suppressed_reason = 'EARLY_SEASON_SAMPLE_RESTRICTION'

    # Selective betting gate (method 4). Disabled by default. When enabled it
    # can only suppress picks that fail the pre-registered profile; it can never
    # upgrade a pick or rescue a NEUTRAL decision.
    selective_rule = None
    if gate_on and m_type not in ('spread', 'sp') and ('OVER' in rec or 'UNDER' in rec):
        selective_rule = _selective_rule_match(
            selective_profile,
            rec,
            abs_edge=abs(edge_vs_curr),
            pick_prob=pick_prob,
            min_games=min_games,
        )
        if selective_rule is None:
            rec = 'NEUTRAL'
            pick_side = None
            pick_prob = 0.50
            suppressed_reason = 'SELECTIVE_GATE_NO_MATCH'

    # LEAN is strictly paper/monitoring; only STRONG on mature sample is actionable
    is_act = ('STRONG' in rec)
    if rec == 'NEUTRAL':
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
        'selective_gate_version': SELECTIVE_GATE_VERSION if gate_on else None,
        'selective_gate_enabled': gate_on,
        'selective_rule': selective_rule,
        'data_quality_flags': dq
    }
