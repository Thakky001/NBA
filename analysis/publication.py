"""Central publication gate.

Separates two questions that used to be conflated in the ledger:

1. What does the model think?  (``*_recommendation`` - raw, kept for research)
2. What did we actually send to users?  (published fields - used for display,
   grading and public statistics)

Every surface (Telegram card, web card, ledger writer) must call the helpers in
this module so that a pick is never shown as a bet in one place while being
silently skipped in another.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

#: Line-movement verdicts that make a model pick un-publishable.
CLASH_VERDICTS = frozenset({'CAPITAL_PRESERVATION', 'PASS', 'MOVE_CONFLICT'})

_PICK_WORDS = ('STRONG', 'LEAN')
_BLANK_TOKENS = {'', 'nan', 'none', 'nat', '<na>'}


def _text(value: Any) -> str:
    """Return a trimmed string, mapping pandas-style nulls to ''."""
    if value is None:
        return ''
    text = str(value).strip()
    return '' if text.lower() in _BLANK_TOKENS else text


def is_pick_recommendation(recommendation: Any) -> bool:
    """True when a recommendation asks for a side (STRONG/LEAN ...)."""
    upper = _text(recommendation).upper()
    return any(word in upper for word in _PICK_WORDS)


def market_verdict(market_res: Optional[Dict[str, Any]]) -> Optional[str]:
    """Extract the line-movement verdict from a market result."""
    if not market_res:
        return None
    line_movement = market_res.get('line_movement') or {}
    return _text(line_movement.get('verdict')) or None


def analysis_global_clash(analysis: Optional[Dict[str, Any]]) -> bool:
    """True when the whole analysis is flagged as a market clash."""
    if not analysis:
        return False
    if analysis.get('is_market_clash') is True:
        return True
    line_movement = analysis.get('line_movement') or {}
    return _text(line_movement.get('verdict')).upper() in CLASH_VERDICTS


def resolve_market(
    market_res: Optional[Dict[str, Any]],
    global_clash: bool = False,
    verdict: Optional[str] = None
) -> Dict[str, Any]:
    """Decide whether one market pick was published.

    Returns a dict with:
        is_published: bool
        published_recommendation: the recommendation users saw ('PASS' when
            suppressed by a market clash, 'NEUTRAL' when the model had no pick)
        publish_reason: None when published, otherwise a stable reason code
        verdict: the line-movement verdict used for the decision
        model_recommendation: the raw model recommendation
        is_market_clash: bool
    """
    market_res = market_res or {}
    rec = _text(market_res.get('recommendation')) or 'NEUTRAL'
    v = _text(verdict) or market_verdict(market_res) or ''
    verdict_upper = v.upper()
    clash = bool(
        global_clash
        or market_res.get('is_market_clash') is True
        or verdict_upper in CLASH_VERDICTS
    )

    if not is_pick_recommendation(rec):
        return {
            'is_published': False,
            'published_recommendation': 'NEUTRAL',
            'publish_reason': 'NEUTRAL',
            'verdict': v or None,
            'model_recommendation': rec,
            'is_market_clash': clash,
        }

    if clash:
        return {
            'is_published': False,
            'published_recommendation': 'PASS',
            'publish_reason': f'MARKET_CLASH_{verdict_upper or "CONFLICT"}',
            'verdict': v or None,
            'model_recommendation': rec,
            'is_market_clash': True,
        }

    return {
        'is_published': True,
        'published_recommendation': rec,
        'publish_reason': None,
        'verdict': v or None,
        'model_recommendation': rec,
        'is_market_clash': False,
    }


def resolve_analysis(analysis: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Resolve publication for both markets of an analysis object."""
    analysis = analysis or {}
    global_clash = analysis_global_clash(analysis)
    return {
        'global_clash': global_clash,
        'spread': resolve_market(analysis.get('spread') or {}, global_clash),
        'over_under': resolve_market(analysis.get('over_under') or {}, global_clash),
    }


def resolve_row(row: Any, prefix: str) -> Dict[str, Any]:
    """Derive publication fields from an existing ledger row.

    Used to backfill/display rows written before published columns existed.
    """
    rec = _text(row.get(f'{prefix}_recommendation')) or 'NEUTRAL'
    verdict = _text(row.get(f'{prefix}_verdict'))
    return resolve_market(
        {'recommendation': rec},
        global_clash=verdict.upper() in CLASH_VERDICTS,
        verdict=verdict or None,
    )
