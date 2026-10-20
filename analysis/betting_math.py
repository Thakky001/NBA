"""Quantitative Betting Mathematics & Risk Optimization Engine.

Provides pure-computation functions (zero external dependencies) for:
1. Odds format conversion (American, Decimal, Probability)
2. De-vigging (removing sportsbook vig/juice to obtain fair probabilities)
3. Edge calculation (model probability vs fair market probability)
4. Expected Value (+EV % ROI per unit staked)
5. Kelly Criterion Bankroll Management (Full Kelly and Fractional/Quarter Kelly)
"""

from __future__ import annotations
import math
from typing import Dict, Any, List, Optional, Union, Tuple

_MIN_AMERICAN = 100.0


def american_to_prob(odds: float) -> float:
    """Convert American odds to implied probability (0.0 to 1.0)."""
    if odds == 0:
        return 0.50
    if odds < 0:
        return -odds / (-odds + 100.0)
    return 100.0 / (odds + 100.0)


def prob_to_american(prob: float) -> float:
    """Convert probability (0.0 to 1.0) to American odds."""
    if prob <= 0.0 or prob >= 1.0:
        return 0.0
    if prob >= 0.50:
        return -(prob / (1.0 - prob)) * 100.0
    return ((1.0 - prob) / prob) * 100.0


def decimal_to_prob(odds: float) -> float:
    """Convert decimal odds (e.g., 1.91, 2.50) to implied probability."""
    if odds <= 1.0:
        return 0.0
    return 1.0 / odds


def prob_to_decimal(prob: float) -> float:
    """Convert probability to decimal odds."""
    if prob <= 0.0:
        return 0.0
    return 1.0 / prob


def convert_odds(odds: float, from_format: str = "american") -> Dict[str, float]:
    """Convert a single odds value across American, Decimal, and Probability formats."""
    fmt = from_format.lower()
    if fmt == "american":
        prob = american_to_prob(odds)
        dec = prob_to_decimal(prob)
        ame = odds
    elif fmt == "decimal":
        prob = decimal_to_prob(odds)
        ame = prob_to_american(prob)
        dec = odds
    elif fmt == "probability":
        prob = max(0.0001, min(0.9999, odds))
        ame = prob_to_american(prob)
        dec = prob_to_decimal(prob)
    else:
        raise ValueError(f"Unknown odds format '{from_format}'. Expected 'american', 'decimal', or 'probability'.")

    return {
        "probability": round(prob, 6),
        "american": round(ame, 1),
        "decimal": round(dec, 4)
    }


def devig(odds: Union[str, List[float]], format: str = "american") -> Dict[str, Any]:
    """Remove vig/juice from sportsbook odds using the multiplicative method.

    fair_prob[i] = raw_prob[i] / sum(raw_probs)

    Args:
        odds: Comma-separated odds string (e.g., "-110,-110" or "-150,+130")
              or list of numeric odds.
        format: Format of input odds ("american", "decimal", or "probability").

    Returns:
        Dict with status, fair_probs, raw_probs, vig_pct, and overround.
    """
    if isinstance(odds, str):
        odds_list = [float(x.strip()) for x in odds.split(",") if x.strip()]
    elif isinstance(odds, (list, tuple)):
        odds_list = [float(x) for x in odds]
    else:
        raise TypeError("odds must be a comma-separated string or list/tuple of numbers")

    if len(odds_list) < 2:
        raise ValueError("At least 2 outcome odds are required to de-vig a market.")

    fmt = format.lower()
    if fmt == "american":
        raw_probs = [american_to_prob(o) for o in odds_list]
    elif fmt == "decimal":
        raw_probs = [decimal_to_prob(o) for o in odds_list]
    elif fmt == "probability":
        raw_probs = [float(o) for o in odds_list]
    else:
        raise ValueError(f"Unknown format '{format}'. Use 'american', 'decimal', or 'probability'.")

    overround = sum(raw_probs)
    if overround <= 0:
        fair_probs = [1.0 / len(raw_probs)] * len(raw_probs)
        vig_pct = 0.0
    else:
        fair_probs = [p / overround for p in raw_probs]
        vig_pct = (overround - 1.0) * 100.0

    return {
        "fair_probs": [round(p, 6) for p in fair_probs],
        "raw_probs": [round(p, 6) for p in raw_probs],
        "vig_pct": round(vig_pct, 2),
        "overround": round(overround, 6),
        "odds": odds_list,
        "format": fmt
    }


def calculate_edge(model_prob: float, market_prob: float) -> float:
    """Calculate absolute edge: model_prob - market_prob."""
    return model_prob - market_prob


def calculate_ev(model_prob: float, market_odds: float, odds_format: str = "american") -> float:
    """Calculate Expected Value (EV) ROI as a decimal per unit staked.

    EV = (Win Prob * Net Payout) - (Loss Prob * Stake)
    For a $1 stake: EV = model_prob * b - (1 - model_prob)
    where b is the decimal profit per $1 bet.

    Returns:
        float: Expected ROI (e.g., 0.05 for +5.0% EV).
    """
    fmt = odds_format.lower()
    if fmt == "american":
        if market_odds >= 100:
            b = market_odds / 100.0
        elif market_odds <= -100:
            b = 100.0 / abs(market_odds)
        else:
            b = 1.0
    elif fmt == "decimal":
        b = market_odds - 1.0
    elif fmt == "probability":
        b = (1.0 / market_odds) - 1.0 if market_odds > 0 else 1.0
    else:
        b = 1.0

    b = max(0.0001, b)
    ev = (model_prob * b) - (1.0 - model_prob)
    return round(ev, 6)


def calculate_kelly(model_prob: float,
                    market_odds: Optional[float] = None,
                    market_prob: Optional[float] = None,
                    fraction: float = 0.25,
                    max_cap: Optional[float] = None) -> Dict[str, float]:
    """Compute optimal Kelly Criterion stake fraction (Pure raw or with optional safety cap).

    Full Kelly formula:
    f* = (p * b - q) / b
    where:
      p = model probability of winning
      q = 1 - p (model probability of losing)
      b = net decimal odds (profit per 1 unit wagered)

    Args:
        model_prob: Estimated win probability (0.0 to 1.0).
        market_odds: American or decimal market odds (optional if market_prob provided).
        market_prob: Implied probability if market odds not given.
        fraction: Kelly multiplier (0.25 for Quarter Kelly, 0.50 for Half Kelly).
        max_cap: Optional maximum bankroll allocation safety cap (e.g. 0.05 = 5.0%). Default None (unconstrained).

    Returns:
        Dict with full_kelly, fractional_kelly, capped_kelly, and recommended_stake_pct.
    """
    if not (0.0 < model_prob < 1.0):
        return {
            "full_kelly": 0.0,
            "fractional_kelly": 0.0,
            "capped_kelly": 0.0,
            "recommended_stake_pct": 0.0
        }

    # Determine net odds b
    if market_odds is not None:
        if market_odds >= 100:
            b = market_odds / 100.0
        elif market_odds <= -100:
            b = 100.0 / abs(market_odds)
        elif market_odds > 1.0:  # Decimal odds
            b = market_odds - 1.0
        else:
            b = 1.0
    elif market_prob is not None and 0.0 < market_prob < 1.0:
        b = (1.0 / market_prob) - 1.0
    else:
        # Default to standard -110 odds
        b = 100.0 / 110.0

    b = max(0.0001, b)
    q = 1.0 - model_prob
    full_f = (model_prob * b - q) / b

    # If negative edge, stake is 0
    full_f = max(0.0, min(1.0, full_f))
    frac_f = full_f * fraction
    # Apply safety cap if specified; otherwise return unconstrained fractional Kelly
    capped_f = min(max_cap, frac_f) if max_cap is not None else frac_f

    return {
        "full_kelly": round(full_f, 4),
        "fractional_kelly": round(frac_f, 4),
        "capped_kelly": round(capped_f, 4),
        "recommended_stake_pct": round(capped_f * 100.0, 2)
    }


def evaluate_market_pick(model_prob: float,
                         book_odds: Union[str, List[float]] = "-110,-110",
                         pick_idx: int = 0,
                         kelly_fraction: float = 0.25,
                         max_stake_cap: Optional[float] = None) -> Dict[str, Any]:
    """All-in-one evaluation: de-vig market, compute edge, EV, and Kelly stake.

    Args:
        model_prob: Model's estimated win probability for the selected outcome (0.0 to 1.0).
        book_odds: Bookmaker odds for all outcomes (e.g., "-110,-110").
        pick_idx: Index of the outcome being evaluated (0 for first outcome, 1 for second).
        kelly_fraction: Fractional Kelly divisor (default 0.25 for quarter-Kelly).
        max_stake_cap: Optional safety cap on bankroll stake (default None = unconstrained).

    Returns:
        Dict with fair_prob, market_prob, edge, ev, kelly_fraction, and summary text.
    """
    devig_res = devig(book_odds)
    fair_probs = devig_res["fair_probs"]
    raw_probs = devig_res["raw_probs"]
    odds_list = devig_res["odds"]

    idx = min(pick_idx, len(fair_probs) - 1)
    fair_p = fair_probs[idx]
    market_p = raw_probs[idx]
    chosen_odds = odds_list[idx]

    edge = calculate_edge(model_prob, fair_p)
    ev = calculate_ev(model_prob, chosen_odds)
    kelly_res = calculate_kelly(model_prob, market_odds=chosen_odds, fraction=kelly_fraction, max_cap=max_stake_cap)

    has_value = ev > 0.0 and edge > 0.0
    rating = "+EV Opportunity" if has_value else ("Neutral / Fair" if abs(edge) < 0.01 else "-EV (Negative Edge)")

    return {
        "model_prob": round(model_prob, 4),
        "fair_prob": round(fair_p, 4),
        "market_prob": round(market_p, 4),
        "chosen_odds": chosen_odds,
        "edge": round(edge, 4),
        "edge_pct": round(edge * 100.0, 2),
        "ev": round(ev, 4),
        "ev_pct": round(ev * 100.0, 2),
        "full_kelly": kelly_res["full_kelly"],
        "quarter_kelly": kelly_res["fractional_kelly"],
        "capped_kelly": kelly_res["capped_kelly"],
        "recommended_stake_pct": kelly_res["recommended_stake_pct"],
        "has_value": has_value,
        "rating": rating,
        "vig_pct": devig_res["vig_pct"]
    }


def market_anchored_projection(raw_model_val: float,
                               market_line: Optional[float],
                               market_weight: float = 0.65) -> float:
    """Blend raw statistical projection with the consensus market line (Bayesian Shrinkage).

    In sports quantitative handicapping, consensus lines represent vast aggregated information.
    Anchoring to market prevents extreme overconfidence from small samples or model divergence.

    Args:
        raw_model_val: Model's unconstrained projection based on team stats/ratings.
        market_line: Consensus sportsbook line (e.g., FanDuel / Vegas line).
        market_weight: Weight assigned to market consensus line (default 0.65 = 65%).

    Returns:
        Anchored projection value rounded to 2 decimal places.
    """
    if market_line is None:
        return round(raw_model_val, 2)
    try:
        m_line = float(market_line)
    except (ValueError, TypeError):
        return round(raw_model_val, 2)

    if m_line <= 0 and raw_model_val > 50:  # Safety check for totals
        return round(raw_model_val, 2)

    weight = max(0.0, min(1.0, float(market_weight)))
    anchored = (weight * m_line) + ((1.0 - weight) * raw_model_val)
    return round(anchored, 2)


def analyze_line_movement(open_line: Optional[float],
                          current_line: Optional[float],
                          market_type: str,
                          recommendation: str) -> Dict[str, Any]:
    """Detect sharp line movement vs market steam against the model pick.

    Analyzes shift between opening line and current/closing line (24h before game).
    Classifies movement into:
      - SHARP_CONFIRMATION: Market money is moving in the same direction as model pick.
      - STEAM_WARNING: Heavy market action is driving the line against model pick.
      - STABLE: Line has barely shifted (< 0.5 pt difference).
      - NO_DATA: Opening line not available.

    Args:
        open_line: Opening spread or total.
        current_line: Current consensus spread or total.
        market_type: 'total' (Over/Under) or 'spread' (Against The Spread).
        recommendation: Current model recommendation string (e.g. 'STRONG UNDER', 'LEAN HOME').

    Returns:
        Dict with diff, status, badge, and human-readable detail explanation.
    """
    if open_line is None or current_line is None:
        return {
            'has_movement': False,
            'diff': 0.0,
            'status': 'NO_DATA',
            'badge': '',
            'detail': 'ไม่มีข้อมูลราคาเปิดเปรียบเทียบ'
        }

    try:
        o_val = float(open_line)
        c_val = float(current_line)
    except (ValueError, TypeError):
        return {
            'has_movement': False,
            'diff': 0.0,
            'status': 'NO_DATA',
            'badge': '',
            'detail': 'ข้อมูลราคาเปิดไม่ถูกต้อง'
        }

    diff = round(c_val - o_val, 2)
    m_type = str(market_type).strip().lower()
    rec_upper = str(recommendation).strip().upper()

    # Stable line check (< 0.5 pt shift)
    if abs(diff) < 0.5:
        return {
            'has_movement': True,
            'diff': diff,
            'status': 'STABLE',
            'badge': '➖ Stable Line',
            'detail': f'ราคานิ่งที่ {c_val:.1f} (ไม่มีการเคลื่อนไหวของเงินก้อนใหญ่)'
        }

    if m_type == 'total':
        # Line went UP (e.g. 229.5 -> 232.5, diff = +3.0)
        if diff > 0:
            if 'OVER' in rec_upper:
                return {
                    'has_movement': True,
                    'diff': diff,
                    'status': 'SHARP_CONFIRMATION',
                    'badge': '🔥 Sharp Move Confirmed',
                    'detail': f'ตลาดไหลขึ้น +{diff:.1f} แต้ม (เปิด {o_val:.1f} ➔ ปัจจุบัน {c_val:.1f}) สอดคล้องกับโมเดลสูง (เงินเซียนเทสูง)'
                }
            elif 'UNDER' in rec_upper:
                return {
                    'has_movement': True,
                    'diff': diff,
                    'status': 'STEAM_WARNING',
                    'badge': '⚠️ Market Steam Warning',
                    'detail': f'ตลาดไหลขึ้น +{diff:.1f} แต้ม (เปิด {o_val:.1f} ➔ ปัจจุบัน {c_val:.1f}) สวนทางกับโมเดลต่ำ (มีแรงซื้อแต้มสูงดันขึ้น)'
                }
            else:
                return {
                    'has_movement': True,
                    'diff': diff,
                    'status': 'LINE_MOVE',
                    'badge': '📈 Total Rose',
                    'detail': f'ราคาเปิดขยับขึ้น +{diff:.1f} แต้ม (เปิด {o_val:.1f} ➔ ปัจจุบัน {c_val:.1f})'
                }
        else:
            # Line went DOWN (e.g. 235.0 -> 232.5, diff = -2.5)
            if 'UNDER' in rec_upper:
                return {
                    'has_movement': True,
                    'diff': diff,
                    'status': 'SHARP_CONFIRMATION',
                    'badge': '🔥 Sharp Move Confirmed',
                    'detail': f'ตลาดไหลลง {diff:.1f} แต้ม (เปิด {o_val:.1f} ➔ ปัจจุบัน {c_val:.1f}) สอดคล้องกับโมเดลต่ำ (เงินเซียนเทต่ำ)'
                }
            elif 'OVER' in rec_upper:
                return {
                    'has_movement': True,
                    'diff': diff,
                    'status': 'STEAM_WARNING',
                    'badge': '⚠️ Market Steam Warning',
                    'detail': f'ตลาดไหลลง {diff:.1f} แต้ม (เปิด {o_val:.1f} ➔ ปัจจุบัน {c_val:.1f}) สวนทางกับโมเดลสูง (มีแรงซื้อแต้มต่ำกดลง)'
                }
            else:
                return {
                    'has_movement': True,
                    'diff': diff,
                    'status': 'LINE_MOVE',
                    'badge': '📉 Total Dropped',
                    'detail': f'ราคาเปิดขยับลง {diff:.1f} แต้ม (เปิด {o_val:.1f} ➔ ปัจจุบัน {c_val:.1f})'
                }

    elif m_type == 'spread':
        # Home spread shift: spread is from home team's perspective.
        # If open was -3.5 and current is -4.5 (diff = -1.0), home is favored MORE (moved toward home).
        # If open was +1.5 and current is +3.5 (diff = +2.0), home gets MORE points (moved toward away).
        moved_toward_home = diff < 0
        abs_diff = abs(diff)

        if moved_toward_home:
            if 'HOME' in rec_upper:
                return {
                    'has_movement': True,
                    'diff': diff,
                    'status': 'SHARP_CONFIRMATION',
                    'badge': '🔥 Sharp Move Confirmed',
                    'detail': f'แต้มต่อไหลเข้าข้างเจ้าบ้าน {abs_diff:.1f} แต้ม (เปิด {o_val:+.1f} ➔ ปัจจุบัน {c_val:+.1f}) สอดคล้องกับโมเดล'
                }
            elif 'AWAY' in rec_upper:
                return {
                    'has_movement': True,
                    'diff': diff,
                    'status': 'STEAM_WARNING',
                    'badge': '⚠️ Market Steam Warning',
                    'detail': f'แต้มต่อไหลเข้าข้างเจ้าบ้าน {abs_diff:.1f} แต้ม (เปิด {o_val:+.1f} ➔ ปัจจุบัน {c_val:+.1f}) สวนทางกับโมเดลทีมเยือน'
                }
            else:
                return {
                    'has_movement': True,
                    'diff': diff,
                    'status': 'LINE_MOVE',
                    'badge': '🏠 Home Money Move',
                    'detail': f'แต้มต่อขยับเข้าเจ้าบ้าน {abs_diff:.1f} แต้ม (เปิด {o_val:+.1f} ➔ ปัจจุบัน {c_val:+.1f})'
                }
        else:
            if 'AWAY' in rec_upper:
                return {
                    'has_movement': True,
                    'diff': diff,
                    'status': 'SHARP_CONFIRMATION',
                    'badge': '🔥 Sharp Move Confirmed',
                    'detail': f'แต้มต่อไหลเข้าข้างทีมเยือน {abs_diff:.1f} แต้ม (เปิด {o_val:+.1f} ➔ ปัจจุบัน {c_val:+.1f}) สอดคล้องกับโมเดล'
                }
            elif 'HOME' in rec_upper:
                return {
                    'has_movement': True,
                    'diff': diff,
                    'status': 'STEAM_WARNING',
                    'badge': '⚠️ Market Steam Warning',
                    'detail': f'แต้มต่อไหลเข้าข้างทีมเยือน {abs_diff:.1f} แต้ม (เปิด {o_val:+.1f} ➔ ปัจจุบัน {c_val:+.1f}) สวนทางกับโมเดลเจ้าบ้าน'
                }
            else:
                return {
                    'has_movement': True,
                    'diff': diff,
                    'status': 'LINE_MOVE',
                    'badge': '✈️ Away Money Move',
                    'detail': f'แต้มต่อขยับเข้าทีมเยือน {abs_diff:.1f} แต้ม (เปิด {o_val:+.1f} ➔ ปัจจุบัน {c_val:+.1f})'
                }

    return {
        'has_movement': True,
        'diff': diff,
        'status': 'LINE_MOVE',
        'badge': '📊 Line Moved',
        'detail': f'ราคาเปิด {o_val} ➔ ปัจจุบัน {c_val} (ขยับ {diff:+.1f})'
    }
