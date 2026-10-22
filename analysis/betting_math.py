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


def devig_multiplicative(raw_probs: List[float]) -> Tuple[List[float], float, float]:
    """Multiplicative (proportional) normalization method.
    
    fair_prob[i] = raw_prob[i] / overround
    """
    overround = sum(raw_probs)
    if overround <= 0:
        return [1.0 / len(raw_probs)] * len(raw_probs), 0.0, 0.0
    fair_probs = [p / overround for p in raw_probs]
    vig_pct = (overround - 1.0) * 100.0
    hold_pct = (1.0 - (1.0 / overround)) * 100.0
    return fair_probs, vig_pct, hold_pct


def devig_power(raw_probs: List[float], tol: float = 1e-12, max_iter: int = 1000) -> Tuple[List[float], float, float, float]:
    """Power (Geometric) de-vigging method: solves for exponent k such that sum(q_i^k) = 1.0.
    
    Effectively corrects favorite-longshot bias by penalizing longshots more than favorites.
    """
    overround = sum(raw_probs)
    if overround <= 1.0:
        s = overround if overround > 0 else 1.0
        return [p / s for p in raw_probs], max(0.0, (overround - 1.0) * 100.0), 0.0, 1.0
    
    low, high = 1.0, 10.0
    mid = 1.0
    for _ in range(max_iter):
        mid = (low + high) / 2.0
        sum_p = sum(p ** mid for p in raw_probs)
        if abs(sum_p - 1.0) < tol:
            break
        if sum_p > 1.0:
            low = mid
        else:
            high = mid
            
    p_unnorm = [p ** mid for p in raw_probs]
    s = sum(p_unnorm)
    fair_probs = [p / s for p in p_unnorm]
    vig_pct = (overround - 1.0) * 100.0
    hold_pct = (1.0 - (1.0 / overround)) * 100.0
    return fair_probs, vig_pct, hold_pct, round(mid, 6)


def devig_shin(raw_probs: List[float], tol: float = 1e-12, max_iter: int = 1000) -> Tuple[List[float], float, float, float]:
    """Shin's (1992, 1993) market-microstructure de-vigging method.
    
    Solves for latent parameter z under the Shin informed-trader market model:
        p_i = (sqrt(z^2 + 4*(1 - z)*(q_i^2 / overround)) - z) / (2*(1 - z))
    where sum(p_i) = 1.0.
    """
    overround = sum(raw_probs)
    if overround <= 1.0:
        s = overround if overround > 0 else 1.0
        return [p / s for p in raw_probs], max(0.0, (overround - 1.0) * 100.0), 0.0, 0.0

    low, high = 0.0, 1.0 - 1e-9
    mid = 0.0
    for _ in range(max_iter):
        mid = (low + high) / 2.0
        p_trial = []
        for qi in raw_probs:
            term = (mid ** 2) + 4.0 * (1.0 - mid) * (qi ** 2 / overround)
            p_val = (math.sqrt(max(0.0, term)) - mid) / (2.0 * (1.0 - mid))
            p_trial.append(p_val)
        sum_p = sum(p_trial)
        if abs(sum_p - 1.0) < tol:
            break
        if sum_p > 1.0:
            low = mid
        else:
            high = mid

    p_final = []
    for qi in raw_probs:
        term = (mid ** 2) + 4.0 * (1.0 - mid) * (qi ** 2 / overround)
        p_val = (math.sqrt(max(0.0, term)) - mid) / (2.0 * (1.0 - mid))
        p_final.append(p_val)
    s = sum(p_final)
    fair_probs = [p / s for p in p_final]
    vig_pct = (overround - 1.0) * 100.0
    hold_pct = (1.0 - (1.0 / overround)) * 100.0
    return fair_probs, vig_pct, hold_pct, round(mid, 6)


def devig(odds: Union[str, List[float]],
          format: str = "american",
          method: str = "multiplicative") -> Dict[str, Any]:
    """Remove vig/juice from sportsbook odds to obtain fair probabilities.

    Supports three methods:
    1. 'multiplicative': Proportional scaling: fair_prob[i] = raw_prob[i] / overround
    2. 'power': Geometric/power exponentiation p_i = q_i^k (corrects favorite-longshot bias)
    3. 'shin': Market-microstructure model estimating a latent parameter z

    Args:
        odds: Comma-separated odds string (e.g., "-110,-110" or "-150,+130")
              or list of numeric odds.
        format: Format of input odds ("american", "decimal", or "probability").
        method: De-vigging model ("multiplicative", "power", or "shin").

    Returns:
        Dict with fair_probs, raw_probs, vig_pct (overround-1), hold_pct (theoretical hold),
        overround, method, and extra parameters (z for shin, k for power).
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
    m = method.lower()
    extra_param = None

    if m == "multiplicative":
        fair_probs, vig_pct, hold_pct = devig_multiplicative(raw_probs)
    elif m == "power":
        fair_probs, vig_pct, hold_pct, k = devig_power(raw_probs)
        extra_param = {"k": k}
    elif m == "shin":
        fair_probs, vig_pct, hold_pct, z = devig_shin(raw_probs)
        extra_param = {"z": z}
    else:
        raise ValueError(f"Unknown de-vigging method '{method}'. Choose 'multiplicative', 'power', or 'shin'.")

    res = {
        "fair_probs": [round(p, 6) for p in fair_probs],
        "raw_probs": [round(p, 6) for p in raw_probs],
        "vig_pct": round(vig_pct, 2),
        "hold_pct": round(hold_pct, 2),
        "overround": round(overround, 6),
        "odds": odds_list,
        "format": fmt,
        "method": m
    }
    if extra_param:
        res.update(extra_param)
    return res


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


def calculate_ev_with_push(model_prob: float, market_odds: float, push_prob: float = 0.0,
                           odds_format: str = "american") -> float:
    """Return unit-stake expected profit when a push returns the stake.

    ``model_prob`` is the unconditional win probability; loss probability is
    ``1 - model_prob - push_prob``.
    """
    if not 0.0 <= model_prob <= 1.0 or not 0.0 <= push_prob <= 1.0 or model_prob + push_prob > 1.0:
        raise ValueError("win and push probabilities must be in [0, 1] and sum to at most 1")
    fmt = odds_format.lower()
    if fmt == "american":
        if market_odds == 0 or -100 < market_odds < 100:
            raise ValueError("American odds must be <= -100 or >= +100")
        profit = 100.0 / abs(market_odds) if market_odds < 0 else market_odds / 100.0
    elif fmt == "decimal":
        if market_odds <= 1.0:
            raise ValueError("Decimal odds must be greater than 1")
        profit = market_odds - 1.0
    else:
        raise ValueError("odds_format must be 'american' or 'decimal'")
    loss_prob = 1.0 - model_prob - push_prob
    return round(model_prob * profit - loss_prob, 6)


def empirical_posterior_sigma(min_games: int, base_sigma: float = 12.5, prior_sigma: float = 14.0) -> float:
    """Legacy heuristic retained for compatibility; not an empirical posterior.

    New forecasts estimate sigma from prior saved forecast residuals instead.
    """
    n = max(1, int(min_games))
    var = (base_sigma ** 2) + ((prior_sigma ** 2) / n)
    return round(math.sqrt(var), 2)


def calculate_kelly(model_prob: float,
                    market_odds: Optional[float] = None,
                    market_prob: Optional[float] = None,
                    fraction: float = 0.25,
                    max_cap: Optional[float] = 0.020,
                    push_prob: float = 0.0) -> Dict[str, float]:
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
        max_cap: Optional maximum bankroll allocation safety cap (default 0.020 = 2.0% max bankroll).

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
    if not 0.0 <= push_prob < 1.0 or model_prob + push_prob > 1.0:
        raise ValueError("win and push probabilities must sum to at most 1")
    q = 1.0 - model_prob - push_prob
    full_f = (model_prob * b - q) / (b * (model_prob + q)) if model_prob + q > 0 else 0.0

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
                         book_odds: Optional[Union[str, List[float]]] = "-110,-110",
                         pick_idx: int = 0,
                         kelly_fraction: float = 0.25,
                         max_stake_cap: Optional[float] = 0.020,
                         push_prob: float = 0.0) -> Dict[str, Any]:
    """All-in-one evaluation: de-vig market, compute edge, EV, and Kelly stake.

    Args:
        model_prob: Model's estimated win probability for the selected outcome (0.0 to 1.0).
        book_odds: Bookmaker odds for all outcomes (e.g., "-110,-110").
        pick_idx: Index of the outcome being evaluated (0 for first outcome, 1 for second).
        kelly_fraction: Fractional Kelly divisor (default 0.25 for quarter-Kelly).
        max_stake_cap: Optional safety cap on bankroll stake (default 0.020 = 2.0%).

    Returns:
        Dict with fair_prob, market_prob, edge, ev, kelly_fraction, and summary text.
    """
    if book_odds is None:
        return {
            "model_prob": round(model_prob, 4), "fair_prob": None, "market_prob": None,
            "chosen_odds": None, "edge": None, "edge_pct": None, "ev": None,
            "ev_pct": None, "full_kelly": 0.0, "quarter_kelly": 0.0,
            "capped_kelly": 0.0, "recommended_stake_pct": 0.0,
            "has_value": False, "rating": "NO_ODDS", "vig_pct": None
        }
    devig_res = devig(book_odds)
    fair_probs = devig_res["fair_probs"]
    raw_probs = devig_res["raw_probs"]
    odds_list = devig_res["odds"]

    idx = min(pick_idx, len(fair_probs) - 1)
    fair_p = fair_probs[idx]
    market_p = raw_probs[idx]
    chosen_odds = odds_list[idx]

    conditional_win = model_prob / (1.0 - push_prob) if push_prob < 1.0 else 0.0
    edge = calculate_edge(conditional_win, fair_p)
    ev = calculate_ev_with_push(model_prob, chosen_odds, push_prob=push_prob)
    kelly_res = calculate_kelly(model_prob, market_odds=chosen_odds, fraction=kelly_fraction,
                                max_cap=max_stake_cap, push_prob=push_prob)

    has_value = ev > 0.0 and edge > 0.0
    rating = "+EV Opportunity" if has_value else ("Neutral / Fair" if abs(edge) < 0.01 else "-EV (Negative Edge)")

    return {
        "model_prob": round(model_prob, 4),
        "push_prob": round(push_prob, 4),
        "conditional_model_prob": round(conditional_win, 4),
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
                               market_weight: float = 0.0) -> float:
    """Blend a raw projection with a market line using a caller-supplied weight.

    In sports quantitative handicapping, consensus lines represent vast aggregated information.
    Anchoring to market prevents extreme overconfidence from small samples or model divergence.

    Args:
        raw_model_val: Model's unconstrained projection based on team stats/ratings.
        market_line: Consensus sportsbook line (e.g., FanDuel / Vegas line).
        market_weight: Market weight, defaulting to a raw-model baseline (0.0).

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
                          recommendation: str,
                          home_team: Optional[str] = None,
                          away_team: Optional[str] = None,
                          is_live: bool = False) -> Dict[str, Any]:
    """Describe price movement relative to a model pick; no handle data is inferred.

    Analyzes shift between opening line and current/closing line (24h before game).
    Classifies movement into:
      - MOVE_WITH_MODEL: The line moved in the model's direction.
      - MOVE_AGAINST_MODEL: The line moved against the model's direction.
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
            'detail': 'ไม่มีข้อมูลราคาเปิดเปรียบเทียบ',
            'market_flow_prob': None,
            'market_flow_direction': '',
            'verdict': 'NO_DATA',
            'verdict_title': '',
            'verdict_detail': ''
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
            'detail': 'ข้อมูลราคาเปิดไม่ถูกต้อง',
            'market_flow_prob': None,
            'market_flow_direction': '',
            'verdict': 'NO_DATA',
            'verdict_title': '',
            'verdict_detail': ''
        }

    diff = round(c_val - o_val, 2)
    abs_diff = abs(diff)
    m_type = str(market_type).strip().lower()
    rec_upper = str(recommendation).strip().upper()

    # Determine model direction label
    if 'OVER' in rec_upper:
        model_dir = 'สูง (OVER)'
    elif 'UNDER' in rec_upper:
        model_dir = 'ต่ำ (UNDER)'
    elif 'HOME' in rec_upper:
        model_dir = 'เจ้าบ้าน (HOME)'
    elif 'AWAY' in rec_upper:
        model_dir = 'ทีมเยือน (AWAY)'
    else:
        model_dir = 'สูสี'

    # Determine market flow direction & implied probability
    if m_type == 'total':
        if diff > 0:
            flow_dir = 'สูง (OVER)'
            flow_prob = None
        elif diff < 0:
            flow_dir = 'ต่ำ (UNDER)'
            flow_prob = None
        else:
            flow_dir = 'สมดุล (NEUTRAL)'
            flow_prob = None
    else: # spread
        if diff < 0:
            flow_dir = f'{home_team} (เจ้าบ้าน)' if home_team else 'เจ้าบ้าน (HOME)'
            flow_prob = None
        elif diff > 0:
            flow_dir = f'{away_team} (ทีมเยือน)' if away_team else 'ทีมเยือน (AWAY)'
            flow_prob = None
        else:
            flow_dir = 'สมดุล (NEUTRAL)'
            flow_prob = None

    # Live in-game movement is not directly comparable to pre-game movement.
    if is_live:
        return {
            'has_movement': True,
            'diff': diff,
            'status': 'LIVE_IN_PLAY',
            'badge': '🔴 Live Line Shift',
            'detail': f'เรตสดในเกมขยับตามสถานการณ์จริงในสนาม ({o_val:.1f} ➔ {c_val:.1f})',
            'market_flow_prob': flow_prob,
            'market_flow_direction': flow_dir,
            'verdict': 'LIVE_TRACKING',
            'verdict_title': 'ติดตามแต้มสด (Live Tracking)',
            'verdict_detail': 'ราคาในเกมเปลี่ยนตามสถานการณ์สด และใช้สรุปแรงเดิมพันไม่ได้'
        }

    # Stable line check (< 0.5 pt shift)
    if abs_diff < 0.5:
        return {
            'has_movement': True,
            'diff': diff,
            'status': 'STABLE',
            'badge': '➖ Stable Line',
            'detail': f'ราคาแทบไม่เปลี่ยนจาก {o_val:.1f} เป็น {c_val:.1f}',
            'market_flow_prob': None,
            'market_flow_direction': flow_dir,
            'verdict': 'STABLE_MODEL',
            'verdict_title': f'ตามสถิติ ({model_dir})',
            'verdict_detail': 'ราคาแทบไม่เปลี่ยนแปลง'
        }

    status = 'LINE_MOVE'
    badge = '📊 Line Moved'
    detail = f'ราคาเปิด {o_val} ➔ ปัจจุบัน {c_val} (ขยับ {diff:+.1f})'

    if m_type == 'total':
        if diff > 0:
            if 'OVER' in rec_upper:
                status = 'MOVE_WITH_MODEL'
                badge = '📊 Move With Model'
                detail = f'เส้นสูงขึ้น {diff:+.1f} แต้ม สอดคล้องกับทิศทางโมเดล Over'
            elif 'UNDER' in rec_upper:
                status = 'MOVE_AGAINST_MODEL'
                badge = '⚠️ Move Against Model'
                detail = f'เส้นสูงขึ้น {diff:+.1f} แต้ม สวนทางกับทิศทางโมเดล Under'
            else:
                badge = '📈 Total Rose'
                detail = f'ราคาเปิดขยับขึ้น +{diff:.1f} แต้ม (เปิด {o_val:.1f} ➔ ปัจจุบัน {c_val:.1f})'
        else:
            if 'UNDER' in rec_upper:
                status = 'MOVE_WITH_MODEL'
                badge = '📊 Move With Model'
                detail = f'เส้นลดลง {diff:.1f} แต้ม สอดคล้องกับทิศทางโมเดล Under'
            elif 'OVER' in rec_upper:
                status = 'MOVE_AGAINST_MODEL'
                badge = '⚠️ Move Against Model'
                detail = f'เส้นลดลง {diff:.1f} แต้ม สวนทางกับทิศทางโมเดล Over'
            else:
                badge = '📉 Total Dropped'
                detail = f'ราคาเปิดขยับลง {diff:.1f} แต้ม (เปิด {o_val:.1f} ➔ ปัจจุบัน {c_val:.1f})'
    else: # spread
        h_name = f'{home_team} (เจ้าบ้าน)' if home_team else 'เจ้าบ้าน'
        a_name = f'{away_team} (ทีมเยือน)' if away_team else 'ทีมเยือน'
        moved_toward_home = diff < 0
        if moved_toward_home:
            if 'HOME' in rec_upper:
                status = 'MOVE_WITH_MODEL'
                badge = '📊 Move With Model'
                detail = f'แต้มต่อไหลเข้าข้าง {h_name} {abs_diff:.1f} แต้ม (เปิด {o_val:+.1f} ➔ ปัจจุบัน {c_val:+.1f}) สอดคล้องกับโมเดล'
            elif 'AWAY' in rec_upper:
                status = 'MOVE_AGAINST_MODEL'
                badge = '⚠️ Move Against Model'
                detail = f'แต้มต่อไหลเข้าข้าง {h_name} {abs_diff:.1f} แต้ม (เปิด {o_val:+.1f} ➔ ปัจจุบัน {c_val:+.1f}) สวนทางกับโมเดล {a_name}'
            else:
                badge = '🏠 Home Money Move'
                detail = f'แต้มต่อขยับเข้า {h_name} {abs_diff:.1f} แต้ม (เปิด {o_val:+.1f} ➔ ปัจจุบัน {c_val:+.1f})'
        else:
            if 'AWAY' in rec_upper:
                status = 'MOVE_WITH_MODEL'
                badge = '📊 Move With Model'
                detail = f'แต้มต่อไหลเข้าข้าง {a_name} {abs_diff:.1f} แต้ม (เปิด {o_val:+.1f} ➔ ปัจจุบัน {c_val:+.1f}) สอดคล้องกับโมเดล'
            elif 'HOME' in rec_upper:
                status = 'MOVE_AGAINST_MODEL'
                badge = '⚠️ Move Against Model'
                detail = f'แต้มต่อไหลเข้าข้าง {a_name} {abs_diff:.1f} แต้ม (เปิด {o_val:+.1f} ➔ ปัจจุบัน {c_val:+.1f}) สวนทางกับโมเดล {h_name}'
            else:
                badge = '✈️ Away Money Move'
                detail = f'แต้มต่อขยับเข้า {a_name} {abs_diff:.1f} แต้ม (เปิด {o_val:+.1f} ➔ ปัจจุบัน {c_val:+.1f})'

    # Movement is descriptive; it cannot identify bettors or set a probability.
    if status == 'MOVE_AGAINST_MODEL':
        verdict = 'MOVE_CONFLICT'
        verdict_title = 'ราคาเคลื่อนไหวสวนทางโมเดล'
        verdict_detail = f'เส้นขยับ {abs_diff:.1f} แต้มสวนทางกับโมเดล; ข้อมูลนี้อย่างเดียวระบุผู้เดิมพันหรือเหตุผลไม่ได้'
    elif status == 'MOVE_WITH_MODEL':
        verdict = 'MOVE_ALIGNED'
        verdict_title = f'ราคาเคลื่อนไหวสอดคล้องกับโมเดล ({model_dir})'
        verdict_detail = f'ทิศทางราคาและโมเดลสอดคล้องกัน (ไหล {diff:+.1f} แต้ม); line movement ไม่ระบุผู้วางเดิมพันหรือ handle'
    else:
        # Neutral or mild moves with no model edge: strict capital preservation (PASS)
        verdict = 'PASS'
        verdict_title = 'ข้าม (PASS) — สถิติสูสี'
        verdict_detail = f'โมเดลสถิติสูสี ไม่มี Edge ที่คุ้มค่าต่อการลงทุน แนะนำรักษาเงินทุน'

    return {
        'has_movement': True,
        'diff': diff,
        'status': status,
        'badge': badge,
        'detail': detail,
        'market_flow_prob': flow_prob,
        'market_flow_direction': flow_dir,
        'verdict': verdict,
        'verdict_title': verdict_title,
        'verdict_detail': verdict_detail
    }


def calculate_live_pace_projection(
    period: Any,
    clock_str: Any,
    home_score: Any,
    away_score: Any,
    pregame_total: Optional[float] = None,
    pregame_spread: Optional[float] = None
) -> Optional[Dict[str, Any]]:
    """Calculate in-game live scoring pace and projected final totals and margins using Bayesian shrinkage.

    Handles regular quarters (Q1-Q4), Halftime, Quarter breaks (End of Q1/Q3),
    and Overtime periods (OT1+). Forecasts remaining minutes by regressing live tempo
    towards pre-game baseline priors, preventing early Q1 blowout artifacts.

    Args:
        period: Current period/quarter (1-4 for regulation, 5+ for OT).
        clock_str: Display clock string (e.g. '07:20', 'Halftime', 'End of 1st', '0.0').
        home_score: Current home team score.
        away_score: Current away team score.
        pregame_total: Optional pre-game consensus total line (e.g. 237.5).
        pregame_spread: Optional pre-game consensus home spread line (e.g. -2.5).

    Returns:
        Dict with elapsed_mins, proj_total, proj_margin, proj_home, proj_away,
        scoring_rate, phase, and status_label, or None if inputs invalid.
    """
    try:
        if home_score is None or away_score is None:
            return None
        h_pts = float(home_score)
        a_pts = float(away_score)
        total_pts = h_pts + a_pts
        if total_pts <= 0:
            return None

        p_int = int(period) if period else 1
        raw_clock = str(clock_str).strip() if clock_str is not None else ""
        clock_lower = raw_clock.lower()

        phase = "IN_PLAY"
        status_label = f"Q{p_int} {raw_clock}" if p_int <= 4 else f"OT{p_int - 4} {raw_clock}"
        elapsed_total = 0.0

        # Check special phases: Halftime, Quarter breaks
        if "half" in clock_lower or (p_int == 2 and raw_clock in ["0.0", "0:00", "0", ""]):
            phase = "HALFTIME"
            status_label = "ช่วงพักครึ่ง (Halftime)"
            elapsed_total = 24.0
        elif "end of 1" in clock_lower or (p_int == 1 and raw_clock in ["0.0", "0:00", "0"]):
            phase = "QUARTER_BREAK"
            status_label = "จบควอเตอร์ 1 (End of Q1)"
            elapsed_total = 12.0
        elif "end of 2" in clock_lower:
            phase = "HALFTIME"
            status_label = "ช่วงพักครึ่ง (Halftime)"
            elapsed_total = 24.0
        elif "end of 3" in clock_lower or (p_int == 3 and raw_clock in ["0.0", "0:00", "0"]):
            phase = "QUARTER_BREAK"
            status_label = "จบควอเตอร์ 3 (End of Q3)"
            elapsed_total = 36.0
        elif "end of 4" in clock_lower or (p_int == 4 and raw_clock in ["0.0", "0:00", "0"]):
            phase = "REGULATION_END"
            status_label = "จบควอเตอร์ 4 (End of Q4)"
            elapsed_total = 48.0
        else:
            # Parse remaining time in period
            if ":" in raw_clock:
                mins, secs = map(float, raw_clock.split(":"))
                rem_in_period = mins + secs / 60.0
            else:
                rem_in_period = float(raw_clock) / 60.0

            period_length = 12.0 if p_int <= 4 else 5.0
            elapsed_in_period = max(0.0, min(period_length, period_length - rem_in_period))

            if p_int <= 4:
                elapsed_total = (p_int - 1) * 12.0 + elapsed_in_period
            else:
                phase = "OVERTIME"
                elapsed_total = 48.0 + (p_int - 5) * 5.0 + elapsed_in_period

        if elapsed_total <= 0.5:
            return None

        # Target regulation/OT minutes
        expected_total_mins = 48.0 if p_int <= 4 else 48.0 + (p_int - 4) * 5.0
        rem_mins = max(0.0, expected_total_mins - elapsed_total)

        # Baseline prior totals and margins
        prior_total = float(pregame_total) if (pregame_total is not None and float(pregame_total) > 150.0) else 228.0
        # In NBA betting, spread_line is home margin relative to away. (e.g. -4.5 means home favored by 4.5 pts)
        prior_margin = -float(pregame_spread) if (pregame_spread is not None) else 0.0

        # Current observed live rates (points / minute) for UI display
        live_total_rate = total_pts / elapsed_total if elapsed_total > 0 else 0.0
        live_margin_rate = (h_pts - a_pts) / elapsed_total if elapsed_total > 0 else 0.0

        # Remaining minutes projection purely anchored to pre-game baseline (zero live pace weight: w = 0)
        # Prevents early quarter shooting anomalies from distorting full game projections
        rem_fraction = rem_mins / 48.0
        proj_total = total_pts + (prior_total * rem_fraction)
        proj_margin = (h_pts - a_pts) + (prior_margin * rem_fraction)

        proj_home = (proj_total + proj_margin) / 2.0
        proj_away = (proj_total - proj_margin) / 2.0

        return {
            'elapsed_mins': round(elapsed_total, 1),
            'target_mins': round(expected_total_mins, 1),
            'rem_mins': round(rem_mins, 1),
            'proj_total': round(proj_total, 1),
            'proj_margin': round(proj_margin, 1),
            'proj_home': round(proj_home, 1),
            'proj_away': round(proj_away, 1),
            'scoring_rate': round(live_total_rate, 2),
            'phase': phase,
            'status_label': status_label
        }
    except Exception:
        return None


def evaluate_live_pick(projected_val: float,
                       current_line: Optional[float],
                       bet_type: str = 'total',
                       rem_mins: float = 48.0,
                       home_team: str = "Home",
                       away_team: str = "Away",
                       market_odds_str: str = "-110,-110") -> Optional[Dict[str, Any]]:
    """Evaluate live in-game pick comparing live projected final vs current live line.

    Args:
        projected_val: Live projected total (if bet_type=='total') or projected home margin (if bet_type=='spread').
        current_line: Live consensus line (e.g. 224.5 for total, or -4.5 for home spread).
        bet_type: 'total' (Over/Under) or 'spread' (Point Spread).
        rem_mins: Game minutes remaining (out of 48.0+).
        home_team: Home team abbreviation or name.
        away_team: Away team abbreviation or name.
        market_odds_str: American odds string for de-vigging (default "-110,-110").

    Returns:
        Dict with edge, win_prob, recommendation, ev, edge_pct, quarter_kelly, or None if line is missing.
    """
    if current_line is None:
        return None

    # Time decay factor for remaining game variance
    decay = math.sqrt(max(0.01, rem_mins / 48.0))

    if bet_type == 'total':
        # Edge = Projected Total - Current Live Total Line
        edge = projected_val - current_line
        sigma = max(3.0, 13.5 * decay)
        z = edge / (sigma * math.sqrt(2.0))
        prob_over = 0.5 * (1.0 + math.erf(z))
        prob_under = 1.0 - prob_over

        if edge >= 2.0:
            rec = "STRONG_OVER"
            rec_text = "🔥 สปีดสดชี้สูง (Strong Over)"
            pick_prob = prob_over
            pick_idx = 0
        elif edge >= 0.8:
            rec = "LEAN_OVER"
            rec_text = "📈 สปีดสดชี้สูง (Lean Over)"
            pick_prob = prob_over
            pick_idx = 0
        elif edge <= -2.0:
            rec = "STRONG_UNDER"
            rec_text = "🧊 สปีดสดชี้ต่ำ (Strong Under)"
            pick_prob = prob_under
            pick_idx = 1
        elif edge <= -0.8:
            rec = "LEAN_UNDER"
            rec_text = "📉 สปีดสดชี้ต่ำ (Lean Under)"
            pick_prob = prob_under
            pick_idx = 1
        else:
            rec = "NEUTRAL"
            rec_text = "⚖️ สปีดสดสูสีเรตตลาด (Neutral / Pass)"
            pick_prob = 0.50
            pick_idx = 0

    else:  # 'spread'
        # In NBA betting convention, current_line is Home spread.
        # Home covers if (Home Score - Away Score) + Spread > 0
        # Cover Edge = Projected Home Margin + Live Spread Line
        edge = projected_val + current_line
        sigma = max(2.5, 11.5 * decay)
        z = edge / (sigma * math.sqrt(2.0))
        prob_home = 0.5 * (1.0 + math.erf(z))
        prob_away = 1.0 - prob_home

        if edge >= 2.0:
            rec = "STRONG_HOME"
            rec_text = f"🔥 สปีดสดชี้ {home_team} (Strong Cover)"
            pick_prob = prob_home
            pick_idx = 0
        elif edge >= 0.8:
            rec = "LEAN_HOME"
            rec_text = f"📈 สปีดสดชี้ {home_team} (Lean Cover)"
            pick_prob = prob_home
            pick_idx = 0
        elif edge <= -2.0:
            rec = "STRONG_AWAY"
            rec_text = f"✈️ สปีดสดชี้ {away_team} (Strong Cover)"
            pick_prob = prob_away
            pick_idx = 1
        elif edge <= -0.8:
            rec = "LEAN_AWAY"
            rec_text = f"📉 สปีดสดชี้ {away_team} (Lean Cover)"
            pick_prob = prob_away
            pick_idx = 1
        else:
            rec = "NEUTRAL"
            rec_text = "⚖️ สปีดสดสูสีเรตตลาด (Neutral / Pass)"
            pick_prob = 0.50
            pick_idx = 0

    eval_metrics = evaluate_market_pick(pick_prob, book_odds=market_odds_str, pick_idx=pick_idx, kelly_fraction=0.25)
    is_bet = rec != "NEUTRAL" and eval_metrics["ev"] > 0

    return {
        'edge': round(edge, 1),
        'win_prob': round(pick_prob, 4),
        'recommendation': rec,
        'rec_text': rec_text,
        'ev': eval_metrics['ev'] if is_bet else 0.0,
        'edge_pct': eval_metrics['edge_pct'] if is_bet else 0.0,
        'quarter_kelly': eval_metrics['quarter_kelly'] if is_bet else 0.0,
        'fair_prob': eval_metrics['fair_prob'],
        'market_prob': eval_metrics['market_prob'],
        'is_bet': is_bet
    }


# ==============================================================================
# SECTION: QUANTITATIVE SPORTS FINANCE & ACCURATE FORMULATIONS
# ==============================================================================

def calculate_mov_elo_multiplier(margin: float, delta_elo_winner: float) -> float:
    """Calculate FiveThirtyEight's Margin of Victory (MOV) Elo multiplier.

    Formula:
        MOV_mult = (|margin| + 3)^0.8 / (7.5 + 0.006 * delta_elo_winner)

    Args:
        margin: Absolute score differential (|winner_score - loser_score|).
        delta_elo_winner: Winner's Elo minus Loser's Elo (including HCA if winner was home).

    Returns:
        float: Multiplier scaling the K-factor Elo rating change.
    """
    m = abs(float(margin))
    denom = 7.5 + 0.006 * float(delta_elo_winner)
    if denom <= 0:
        raise ValueError("winner Elo difference produces a non-positive MOV denominator")
    return round(((m + 3.0) ** 0.8) / denom, 6)


def elo_expected_probability(rating_a: float, rating_b: float, home_advantage: float = 0.0) -> float:
    """Elo logistic expectation for A; ratings and home advantage use Elo points."""
    return 1.0 / (1.0 + 10.0 ** (-((rating_a - rating_b + home_advantage) / 400.0)))


def update_elo(rating_a: float, rating_b: float, score_a: float, k_factor: float = 20.0,
               home_advantage: float = 0.0, mov_multiplier: float = 1.0) -> Tuple[float, float]:
    """Zero-sum Elo update; score_a is 1 (win), 0.5 (draw), or 0 (loss)."""
    if score_a not in (0.0, 0.5, 1.0) or k_factor < 0 or mov_multiplier < 0:
        raise ValueError("invalid Elo score, K-factor, or MOV multiplier")
    expected = elo_expected_probability(rating_a, rating_b, home_advantage)
    delta = k_factor * mov_multiplier * (score_a - expected)
    return rating_a + delta, rating_b - delta


def calculate_win_probability(projected_margin: float, sigma: float = 12.0) -> float:
    """Calculate outright win probability (Moneyline) via Normal CDF.

    P(Win) = P(Margin > 0) = Phi(projected_margin / sigma)

    The default sigma is a convenience fallback and is not calibrated by this project.

    Args:
        projected_margin: Expected margin of victory for team (positive = favored, negative = underdog).
        sigma: Standard deviation of point differential, estimated from training residuals.

    Returns:
        float: Outright win probability (0.0 to 1.0).
    """
    if sigma <= 0.0:
        return 1.0 if projected_margin > 0 else (0.5 if projected_margin == 0 else 0.0)
    z = projected_margin / (sigma * math.sqrt(2.0))
    return round(0.5 * (1.0 + math.erf(z)), 6)


def calculate_cover_probability(spread_line: float,
                                projected_margin: float,
                                sigma: float = 12.0) -> float:
    """Calculate probability of covering the point spread (Against The Spread - ATS).

    Convention:
        spread_line: Team's spread line (e.g., -6.5 for favorite, +6.5 for underdog).
        projected_margin: Model expected margin of victory for this team (e.g., +6.5).
        sigma: Standard deviation of point differential.

    Cover condition: Actual Margin + Spread Line > 0
    P(Cover) = Phi((projected_margin + spread_line) / sigma)

    CRITICAL DISTINCTION:
    - If projected_margin == 6.5 and spread_line == -6.5 (Fair Spread):
      Cover probability = Phi((6.5 - 6.5) / sigma) = Phi(0) = 50.0% exactly!
    - Outright win probability = Phi(6.5 / 12) = 70.6%.
    """
    if sigma <= 0.0:
        edge = projected_margin + spread_line
        return 1.0 if edge > 0 else (0.5 if edge == 0 else 0.0)
    edge = projected_margin + spread_line
    z = edge / (sigma * math.sqrt(2.0))
    return round(0.5 * (1.0 + math.erf(z)), 6)


def calculate_spread_probabilities(spread_line: float, projected_margin: float,
                                   sigma: float) -> Dict[str, float]:
    """Discrete Normal-continuity approximation with explicit integer-line push."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    edge = float(projected_margin) + float(spread_line)
    if not float(spread_line).is_integer():
        win = calculate_cover_probability(spread_line, projected_margin, sigma)
        return {"win": win, "push": 0.0, "loss": round(1.0 - win, 6)}
    # A tie at the adjusted margin is represented by the probability mass in
    # the half-point interval around the integer threshold.
    threshold = -float(spread_line)
    cdf = lambda x: 0.5 * (1.0 + math.erf((x - projected_margin) / (sigma * math.sqrt(2.0))))
    push = max(0.0, cdf(threshold + 0.5) - cdf(threshold - 0.5))
    loss = max(0.0, cdf(threshold - 0.5))
    win = max(0.0, 1.0 - cdf(threshold + 0.5))
    total = win + push + loss
    return {"win": round(win / total, 6), "push": round(push / total, 6), "loss": round(loss / total, 6)}


def estimate_residual_sigma(residuals: List[float]) -> float:
    """Estimate prediction-error SD from training-fold residuals only."""
    if len(residuals) < 2:
        raise ValueError("at least two training residuals are required")
    mean = sum(float(x) for x in residuals) / len(residuals)
    return math.sqrt(sum((float(x) - mean) ** 2 for x in residuals) / (len(residuals) - 1))


def empirical_line_probabilities(line: float, projection: float,
                                 training_residuals: List[float]) -> Dict[str, float]:
    """Estimate win/push/loss from held-back training residuals (actual - forecast)."""
    if not training_residuals:
        raise ValueError("training residuals are required")
    outcomes = []
    for residual in training_residuals:
        adjusted = float(projection) + float(residual) + float(line)
        tied = float(line).is_integer() and math.isclose(adjusted, 0.0, abs_tol=1e-8)
        outcomes.append("push" if tied else "win" if adjusted > 0 else "loss")
    count = len(outcomes)
    return {key: outcomes.count(key) / count for key in ("win", "push", "loss")}


def calculate_expected_pace(pace_home: float,
                            pace_away: float,
                            league_pace: float = 99.5,
                            method: str = "additive") -> float:
    """Calculate projected game pace using additive or multiplicative modeling.

    Additive model (Recommended baseline):
        Pace = League + (Pace_home - League) + (Pace_away - League) = Pace_home + Pace_away - League

    Multiplicative model:
        Pace = (Pace_home * Pace_away) / League
    """
    m = method.lower()
    if m == "additive":
        return round(float(pace_home) + float(pace_away) - float(league_pace), 2)
    elif m == "multiplicative":
        if league_pace <= 0:
            return round((float(pace_home) + float(pace_away)) / 2.0, 2)
        return round((float(pace_home) * float(pace_away)) / float(league_pace), 2)
    else:
        raise ValueError(f"Unknown pace method: {method}. Choose 'additive' or 'multiplicative'.")


def calculate_expected_total(expected_pace: float,
                             home_adj_ortg: float,
                             away_adj_ortg: float) -> float:
    """Calculate expected total game points from pace and adjusted offensive ratings.

    Formula:
        Expected Total = (Expected Pace / 100) * (Home Adj ORtg + Away Adj ORtg)
    """
    total = (float(expected_pace) / 100.0) * (float(home_adj_ortg) + float(away_adj_ortg))
    return round(total, 2)


def calculate_brier_score(predictions: List[float], outcomes: List[int]) -> float:
    """Calculate Brier Score: Mean squared error of probabilistic forecasts.

    Score = (1/N) * sum((prob_i - outcome_i)^2)
    Range: [0.0 (perfect) to 1.0 (completely wrong)].
    Reference: For a 50/50 binary market, random guessing yields 0.25.
    """
    if len(predictions) != len(outcomes) or len(predictions) == 0:
        raise ValueError("predictions and outcomes must be non-empty lists of identical length.")
    sq_err = sum((float(p) - float(y)) ** 2 for p, y in zip(predictions, outcomes))
    return round(sq_err / len(predictions), 6)


def calculate_log_loss(predictions: List[float], outcomes: List[int], eps: float = 1e-15) -> float:
    """Calculate Binary Logarithmic Loss (Cross-Entropy).

    Loss = - (1/N) * sum( y_i * ln(p_i) + (1 - y_i) * ln(1 - p_i) )
    """
    if len(predictions) != len(outcomes) or len(predictions) == 0:
        raise ValueError("predictions and outcomes must be non-empty lists of identical length.")
    loss_sum = 0.0
    for p, y in zip(predictions, outcomes):
        p_val = max(eps, min(1.0 - eps, float(p)))
        y_val = float(y)
        loss_sum += y_val * math.log(p_val) + (1.0 - y_val) * math.log(1.0 - p_val)
    return round(-loss_sum / len(predictions), 6)


def calculate_clv(bet_odds: float,
                  closing_odds: float,
                  odds_format: str = "american") -> Dict[str, float]:
    """Calculate Closing Line Value (CLV).

    CLV is the definitive quantitative test of sports betting edge:
    It measures the price advantage achieved relative to the recorded closing price.

    CLV % = (Decimal Bet Odds / Decimal Closing Odds - 1.0) * 100.0
    """
    fmt = odds_format.lower()
    if fmt == "american":
        bet_dec = prob_to_decimal(american_to_prob(bet_odds))
        close_dec = prob_to_decimal(american_to_prob(closing_odds))
    elif fmt == "decimal":
        bet_dec = float(bet_odds)
        close_dec = float(closing_odds)
    elif fmt == "probability":
        bet_dec = 1.0 / float(bet_odds) if float(bet_odds) > 0 else 1.0
        close_dec = 1.0 / float(closing_odds) if float(closing_odds) > 0 else 1.0
    else:
        raise ValueError(f"Unknown format: {odds_format}")

    if close_dec <= 0:
        clv_pct = 0.0
    else:
        clv_pct = ((bet_dec / close_dec) - 1.0) * 100.0

    return {
        "bet_decimal": round(bet_dec, 4),
        "closing_decimal": round(close_dec, 4),
        "clv_pct": round(clv_pct, 2)
    }


