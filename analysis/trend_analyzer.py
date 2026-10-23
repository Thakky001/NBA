import pandas as pd
import numpy as np
import logging
import math
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class TrendAnalyzer:
    """Analyzes Over/Under and Spread trends for NBA matchups.
    
    Uses features from FeatureEngineer to produce actionable analysis
    with confidence levels and supporting evidence.
    """

    # Confidence thresholds
    STRONG_THRESHOLD = 0.7
    LEAN_THRESHOLD = 0.55

    @staticmethod
    def parse_streak(s):
        if s is None or s == 'None' or pd.isna(s):
            return 0
        if isinstance(s, (int, float)):
            return int(s)
        s_str = str(s).strip()
        if s_str.startswith('W'):
            return int(s_str[1:]) if len(s_str) > 1 and s_str[1:].isdigit() else 1
        if s_str.startswith('L'):
            return -int(s_str[1:]) if len(s_str) > 1 and s_str[1:].isdigit() else -1
        return 0

    def __init__(self, feature_engineer):
        """
        Args:
            feature_engineer: FeatureEngineer instance with loaded data
        """
        self.fe = feature_engineer

    def analyze_over_under(self, home_team: str, away_team: str,
                           over_under_line: float = None,
                           open_line: float = None,
                           market_weight: float = 0.0,
                           is_live: bool = False,
                           game_date: str = None,
                           market_odds: Any = None,
                           training_residuals: Any = None,
                           **kwargs) -> Dict[str, Any]:
        """Full Over/Under analysis with Market-Anchored Projection and Line Movement."""
        
        # Use compute_matchup_features
        features = self.fe.compute_matchup_features(home_team, away_team, over_under_line=over_under_line, game_date=game_date) if hasattr(self.fe, 'compute_matchup_features') else {}
        
        factors = []
        score = 0.0
        
        # Dynamic Weighting setup
        weights = {
            'avg': 0.30,
            'pace': 0.20,
            'ou': 0.20,
            'h2h': 0.15,
            'trend': 0.15
        }
        
        home_pace = features.get('home_pace')
        away_pace = features.get('away_pace')
        if home_pace is None or away_pace is None:
            # Redistribute 0.20 weight: +0.10 to Avg, +0.10 to OU Record
            weights['pace'] = 0.0
            weights['avg'] += 0.10
            weights['ou'] += 0.10
            
        h2h_total = features.get('h2h_avg_total')
        if not h2h_total or h2h_total == 0.0:
            # Redistribute 0.15 weight: +0.15 to Trend
            weights['h2h'] = 0.0
            weights['trend'] += 0.15

        # Check if historical O/U records exist in cache; if not, safely redistribute weight
        home_ou = features.get('home_ou_record', (0, 0))
        away_ou = features.get('away_ou_record', (0, 0))
        total_games = sum(home_ou) + sum(away_ou)
        if total_games == 0 and weights['ou'] > 0:
            ou_w = weights['ou']
            weights['ou'] = 0.0
            weights['avg'] += ou_w * 0.60
            weights['pace'] += ou_w * 0.40
        
        # Factor 1: Combined average vs line
        combined_avg = features.get('combined_avg_10') or features.get('combined_avg_season', 0)
        if combined_avg and over_under_line:
            diff = combined_avg - over_under_line
            factor_score = min(max(diff / 10, -1), 1)
            direction = 'over' if factor_score > 0 else ('under' if factor_score < 0 else 'neutral')
            factors.append({
                'name': 'Combined Avg vs Line',
                'value': f'{combined_avg:.1f} vs {over_under_line:.1f} (diff: {diff:+.1f})',
                'direction': direction,
                'weight': weights['avg'],
                'score': factor_score
            })
            score += factor_score * weights['avg']
        
        # Factor 2: Pace factor
        if weights['pace'] > 0:
            avg_pace = (home_pace + away_pace) / 2 if (home_pace and away_pace) else 99.5
            league_avg_pace = 99.5
            pace_diff = avg_pace - league_avg_pace
            factor_score = min(max(pace_diff / 5, -1), 1)
            direction = 'over' if factor_score > 0 else ('under' if factor_score < 0 else 'neutral')
            factors.append({
                'name': 'Pace Factor',
                'value': f'Home: {home_pace or 99.5:.1f} | Away: {away_pace or 99.5:.1f} | Combined avg: {avg_pace:.1f}',
                'direction': direction,
                'weight': weights['pace'],
                'score': factor_score
            })
            score += factor_score * weights['pace']
        
        # Factor 3: O/U Record (if available in historical data)
        if weights['ou'] > 0 and total_games > 0:
            total_over = home_ou[0] + away_ou[0]
            over_pct = total_over / total_games
            factor_score = (over_pct - 0.5) * 2
            direction = 'over' if factor_score > 0 else ('under' if factor_score < 0 else 'neutral')
            factors.append({
                'name': 'O/U Record (Last 10)',
                'value': f'Home: {home_ou[0]}-{home_ou[1]} O/U | Away: {away_ou[0]}-{away_ou[1]} O/U',
                'direction': direction,
                'weight': weights['ou'],
                'score': factor_score
            })
            score += factor_score * weights['ou']
            
        # Factor 4: H2H avg total
        if weights['h2h'] > 0 and over_under_line:
            diff = h2h_total - over_under_line
            factor_score = min(max(diff / 10, -1), 1)
            direction = 'over' if factor_score > 0 else ('under' if factor_score < 0 else 'neutral')
            factors.append({
                'name': 'H2H Average Total',
                'value': f'{h2h_total:.1f} vs line {over_under_line:.1f}',
                'direction': direction,
                'weight': weights['h2h'],
                'score': factor_score
            })
            score += factor_score * weights['h2h']
            
        # Factor 5: Trend direction
        trend = features.get('trend_direction', 'FLAT')
        if trend == 'UP':
            factor_score = 0.5
            direction = 'over'
        elif trend == 'DOWN':
            factor_score = -0.5
            direction = 'under'
        else:
            factor_score = 0.0
            direction = 'neutral'
            
        factors.append({
            'name': 'Scoring Trend',
            'value': trend,
            'direction': direction,
            'weight': weights['trend'],
            'score': factor_score
        })
        score += factor_score * weights['trend']
        
        # Projected total calculation (Possession & Scoring baseline)
        home_ortg = features.get('home_ortg')
        home_drtg = features.get('home_drtg')
        away_ortg = features.get('away_ortg')
        away_drtg = features.get('away_drtg')
        league_avg_pace = features.get('league_avg_pace') or 99.5

        if home_ortg is not None and home_drtg is not None and away_ortg is not None and away_drtg is not None and home_pace and away_pace:
            # Dean Oliver Possession & Efficiency Model with dynamic league pace & home court baseline
            expected_pace = home_pace + away_pace - league_avg_pace
            home_exp_eff = (home_ortg + away_drtg) / 2.0
            away_exp_eff = (away_ortg + home_drtg) / 2.0
            
            raw_home_pts = expected_pace * (home_exp_eff / 100.0)
            raw_away_pts = expected_pace * (away_exp_eff / 100.0)
            raw_projected_total = raw_home_pts + raw_away_pts
        else:
            # Fallback when advanced ratings unavailable: use combined_avg directly without double pace
            raw_projected_total = combined_avg if combined_avg > 0 else (over_under_line or 220.0)

        # Injury and rest remain descriptive features until their effects are fit
        # on timestamped historical data.
        
        # Optional market blend; zero by default until weights are validated.
        from analysis.betting_math import calculate_ev_with_push, calculate_spread_probabilities, evaluate_market_pick, analyze_line_movement
        
        # Anchor line for tracking line movement
        anchor_line = open_line if (open_line is not None and open_line > 0) else over_under_line

        # Check early season status
        home_games = features.get('home_games_played', 0)
        away_games = features.get('away_games_played', 0)
        min_games = min(home_games, away_games)
        is_early_season = min_games < 10

        effective_market_weight = max(0.0, min(1.0, float(market_weight)))

        if over_under_line is not None and over_under_line > 0:
            raw_edge = raw_projected_total - over_under_line
            shrunk_edge = raw_edge * (1.0 - effective_market_weight)
            projected_total = round(over_under_line + shrunk_edge, 2)
            total_edge = round(shrunk_edge, 2)
        else:
            projected_total = round(raw_projected_total, 2)
            total_edge = 0.0
            raw_edge = 0.0
            
        from analysis.betting_math import estimate_residual_sigma, empirical_line_probabilities
        sigma_ou = estimate_residual_sigma(training_residuals) if training_residuals and len(training_residuals) >= 2 else 12.5
        probability_method = 'empirical_training_residuals' if training_residuals is not None and len(training_residuals) >= 2 else 'normal_unvalidated_sigma'

        # Calculate realistic win probability and EV-based conviction filter
        if over_under_line and over_under_line > 0:
            total_probs = (empirical_line_probabilities(-over_under_line, projected_total, training_residuals)
                           if training_residuals else None)
            if total_probs:
                win_prob_over = total_probs['win']
                win_prob_under = total_probs['loss']
                push_prob = total_probs['push']
            else:
                total_probs = calculate_spread_probabilities(-over_under_line, projected_total, sigma_ou)
                win_prob_over = total_probs['win']
                win_prob_under = total_probs['loss']
                push_prob = total_probs['push']

            if total_edge > 0:
                cand_dir = 'OVER'
                cand_prob = win_prob_over
            elif total_edge < 0:
                cand_dir = 'UNDER'
                cand_prob = win_prob_under
            else:
                cand_dir = 'NEUTRAL'
                cand_prob = 0.50

            # Candidate EV against standard -110 bookmaker line (breakeven 52.38%)
            cand_ev = calculate_ev_with_push(cand_prob, market_odds[0 if cand_dir == 'OVER' else 1], push_prob=push_prob) if market_odds else None

            # Dual Conviction Gate:
            # STRONG: |total_edge| >= 2.0 pts AND cand_prob >= 56.5% AND cand_ev >= +5.0%
            # LEAN: |total_edge| >= 1.0 pts AND cand_prob >= 53.8% AND cand_ev >= +1.5% (Only mid-season min_games >= 10)
            if abs(total_edge) >= 2.0 and cand_prob >= 0.565 and (cand_ev is None or cand_ev >= 0.05):
                recommendation = f"STRONG {cand_dir}"
            elif not is_early_season and abs(total_edge) >= 1.0 and cand_prob >= 0.538 and (cand_ev is None or cand_ev >= 0.015):
                recommendation = f"LEAN {cand_dir}"
            else:
                recommendation = 'NEUTRAL'

            win_prob = win_prob_over
            win_prob_under = total_probs['loss']
        else:
            push_prob = 0.0
            win_prob = 0.50
            recommendation = 'NEUTRAL'

        # Probability for the recommended side
        pick_win_prob = win_prob if 'OVER' in recommendation else (win_prob_under if 'UNDER' in recommendation else 0.50)
        
        # Line movement is descriptive and does not identify betting flow.
        line_mv = analyze_line_movement(open_line, over_under_line, market_type='total', recommendation=recommendation, is_live=is_live)
        
        # Anomaly Detection (flagging extreme discrepancies without clamping)
        is_anomaly = abs(total_edge) > 8.0 if over_under_line else False
        anomaly_warning = "⚠️ Extreme Discrepancy (>8 pts): Potential data anomaly / volatile line" if is_anomaly else None
        
        # Pure Raw Mode: Use pure win_prob directly without artificial clamping
        raw_win_prob = pick_win_prob
            
        confidence = abs(pick_win_prob - 0.5) * 2.0
        
        # Value and sizing are unavailable without actual market odds.
        market_odds_str = market_odds
        pick_idx = 0 if 'OVER' in recommendation else 1
        eval_metrics = evaluate_market_pick(pick_win_prob, book_odds=market_odds_str, pick_idx=pick_idx, kelly_fraction=0.25, max_stake_cap=0.020, push_prob=push_prob if over_under_line else 0.0)
        
        # Line movement is not validated as a bet-selection rule.
        is_market_clash = False

        is_bet = recommendation != 'NEUTRAL' and not is_market_clash and eval_metrics['ev'] is not None
        ev_val = eval_metrics['ev'] if is_bet else None
        edge_pct_val = eval_metrics['edge_pct'] if is_bet else None
        full_kelly = eval_metrics['full_kelly'] if is_bet else 0.0
        quarter_kelly = (eval_metrics['recommended_stake_pct'] / 100.0) if is_bet else 0.0
        if is_market_clash:
            kelly_rec = "0.0% (ข้าม - สัญญาณขัดแย้งกับกระแสเงินตลาด)"
        elif quarter_kelly > 0:
            kelly_rec = f"{quarter_kelly * 100:.1f}% of bankroll (1/4 Kelly)"
        else:
            kelly_rec = "0% (No Bet)"

        # Build summary
        summary = f"O/U {over_under_line}: {recommendation} (Edge: {total_edge:+.1f} pts | Win Prob: {pick_win_prob:.1%})\n"
        if is_market_clash:
            summary += f"  ⚠️ Market Clash: ตลาดไหลสวนทางโมเดล {line_mv.get('diff', 0):+.1f} แต้ม — แนะนำ ข้าม (PASS) เพื่อรักษาทุน\n"
        elif is_bet:
            summary += f"  ⚡ Value & Sizing: +EV: {ev_val:+.1%} | Kelly: {quarter_kelly:.1%} (1/4 Kelly)\n"
        if line_mv.get('badge'):
            summary += f"  {line_mv['badge']}: {line_mv['detail']}\n"
        if is_anomaly:
            summary += f"  {anomaly_warning}\n"
        for f in factors:
            arrow = '↑' if f['direction'] == 'over' else ('↓' if f['direction'] == 'under' else '→')
            summary += f"  {arrow} {f['name']}: {f['value']}\n"
        
        return {
            'recommendation': recommendation,
            'confidence': confidence,
            'win_prob': pick_win_prob,
            'raw_win_prob': raw_win_prob,
            'is_anomaly': is_anomaly,
            'anomaly_warning': anomaly_warning,
            'is_market_clash': is_market_clash,
            'total_edge': total_edge,
            'raw_edge': raw_edge,
            'projected_total': projected_total,
            'raw_projected_total': raw_projected_total,
            'anchor_line': anchor_line,
            'market_weight': market_weight,
            'open_line': open_line,
            'line_movement': line_mv,
            'ev': ev_val,
            'edge_pct': edge_pct_val,
            'fair_prob': eval_metrics['fair_prob'],
            'market_prob': eval_metrics['market_prob'],
            'kelly_fraction': full_kelly,
            'kelly_quarter': quarter_kelly,
            'kelly_recommendation': kelly_rec,
            'score': score,
            'factors': factors,
            'summary': summary,
            'features': features,
            'sigma_total': sigma_ou,
            'probability_method': probability_method,
            'push_prob': push_prob if over_under_line else 0.0
        }

    def analyze_spread(self, home_team: str, away_team: str,
                       spread_line: float = None,
                       open_spread: float = None,
                       market_weight: float = 0.0,
                       is_live: bool = False,
                       game_date: str = None,
                       market_odds: Any = None,
                       training_residuals: Any = None,
                       **kwargs) -> Dict[str, Any]:
        """Full Spread analysis with Market-Anchored Projection and Line Movement."""
        
        # Use compute_matchup_features to ensure we get rest days as well
        features = self.fe.compute_matchup_features(home_team, away_team, spread_line=spread_line, game_date=game_date) if hasattr(self.fe, 'compute_matchup_features') else {}
        
        factors = []
        score = 0.0
        
        # Dynamic Weighting setup
        weights = {
            'nr': 0.20,
            'pd': 0.15,
            'sp': 0.15,
            'str': 0.10,
            'h2h': 0.10,
            'rest': 0.10,
            'roster': 0.20
        }
        
        home_games = features.get('home_games_played', 10)
        away_games = features.get('away_games_played', 10)
        min_games = min(home_games, away_games)
        is_early_season = min_games < 10
        
        home_roster = features.get('home_roster_rating', 0)
        away_roster = features.get('away_roster_rating', 0)
        
        # 1. Early Season Override
        if is_early_season:
            weights['nr'] = 0.0
            weights['pd'] = 0.0
            weights['str'] = 0.0
            weights['h2h'] = 0.10
            weights['rest'] = 0.10
            weights['sp'] = 0.20
            weights['roster'] = 0.60
            
        # 2. Dynamic Star-Impact (Crisis Mode) Override
        # Assuming feature engineering provides 'home_full_roster_rating' which is 100% health
        home_full = features.get('home_full_roster_rating', home_roster)
        away_full = features.get('away_full_roster_rating', away_roster)
        
        home_drop_pct = (home_full - home_roster) / home_full if home_full > 0 else 0.0
        away_drop_pct = (away_full - away_roster) / away_full if away_full > 0 else 0.0
        
        crisis_threshold = 0.15 # 15% drop in roster power
        
        is_crisis = False
        if home_drop_pct >= crisis_threshold or away_drop_pct >= crisis_threshold:
            is_crisis = True
            # Star player is out, past 10 games are irrelevant. Shift weights.
            weights['nr'] = 0.05
            weights['pd'] = 0.05
            weights['str'] = 0.0
            weights['sp'] = 0.10
            weights['h2h'] = 0.10
            weights['rest'] = 0.10
            weights['roster'] = 0.60
        
        home_net_rtg = features.get('home_net_rating')
        away_net_rtg = features.get('away_net_rating')
        if home_net_rtg is None or away_net_rtg is None:
            if weights['nr'] > 0:
                weights['pd'] += weights['nr'] * 0.6
                weights['sp'] += weights['nr'] * 0.4
                weights['nr'] = 0.0
            home_net_rtg = 0.0
            away_net_rtg = 0.0
        
        # Factor 1: net_rating gap
        net_rtg_diff = home_net_rtg - away_net_rtg
        factor_score_nr = min(max(net_rtg_diff / 10, -1), 1)
        direction_nr = 'home' if factor_score_nr > 0 else ('away' if factor_score_nr < 0 else 'neutral')
        factors.append({
            'name': 'Net Rating Gap',
            'value': f'Home {home_net_rtg:.1f} vs Away {away_net_rtg:.1f} (Diff: {net_rtg_diff:+.1f})',
            'direction': direction_nr,
            'weight': weights['nr'],
            'score': factor_score_nr
        })
        score += factor_score_nr * weights['nr']

        # Factor 2: point differential comparison
        home_pt_diff = features.get('home_point_diff_10') or 0.0
        away_pt_diff = features.get('away_point_diff_10') or 0.0
        pt_diff_gap = home_pt_diff - away_pt_diff
        factor_score_pd = min(max(pt_diff_gap / 10, -1), 1)
        direction_pd = 'home' if factor_score_pd > 0 else ('away' if factor_score_pd < 0 else 'neutral')
        factors.append({
            'name': 'Point Differential',
            'value': f'Home {home_pt_diff:+.1f} vs Away {away_pt_diff:+.1f}',
            'direction': direction_pd,
            'weight': weights['pd'],
            'score': factor_score_pd
        })
        score += factor_score_pd * weights['pd']

        # Factor 3: home/away splits
        home_rec = features.get('home_home_record', (0,0))
        away_rec = features.get('away_away_record', (0,0))
        home_home_win_pct = home_rec[0] / (home_rec[0] + home_rec[1]) if sum(home_rec) > 0 else 0.5
        away_away_win_pct = away_rec[0] / (away_rec[0] + away_rec[1]) if sum(away_rec) > 0 else 0.5
        split_diff = home_home_win_pct - away_away_win_pct
        factor_score_sp = min(max(split_diff * 2, -1), 1)
        direction_sp = 'home' if factor_score_sp > 0 else ('away' if factor_score_sp < 0 else 'neutral')
        factors.append({
            'name': 'Home/Away Splits',
            'value': f'Home Win% at Home: {home_home_win_pct:.1%} vs Away Win% at Away: {away_away_win_pct:.1%}',
            'direction': direction_sp,
            'weight': weights['sp'],
            'score': factor_score_sp
        })
        score += factor_score_sp * weights['sp']

        # Factor 4: recent form / streak
        parse_streak = self.parse_streak
        home_streak = parse_streak(features.get('home_streak'))
        away_streak = parse_streak(features.get('away_streak'))
        streak_diff = home_streak - away_streak
        factor_score_str = min(max(streak_diff / 5, -1), 1)
        direction_str = 'home' if factor_score_str > 0 else ('away' if factor_score_str < 0 else 'neutral')
        factors.append({
            'name': 'Recent Streak',
            'value': f'Home: {home_streak:+} vs Away: {away_streak:+}',
            'direction': direction_str,
            'weight': weights['str'],
            'score': factor_score_str
        })
        score += factor_score_str * weights['str']

        # Factor 5: h2h record
        h2h_rec = features.get('h2h_record', (0,0))
        h2h_home_wins = h2h_rec[0]
        h2h_games = sum(h2h_rec)
        if h2h_games > 0:
            h2h_pct = h2h_home_wins / h2h_games
            factor_score_h2h = (h2h_pct - 0.5) * 2
        else:
            factor_score_h2h = 0.0
        direction_h2h = 'home' if factor_score_h2h > 0 else ('away' if factor_score_h2h < 0 else 'neutral')
        factors.append({
            'name': 'Head-to-Head Record',
            'value': f'Home won {h2h_home_wins}/{h2h_games}',
            'direction': direction_h2h,
            'weight': weights['h2h'],
            'score': factor_score_h2h
        })
        score += factor_score_h2h * weights['h2h']

        # Factor 6: rest days advantage (weight 0.10)
        home_rest = features.get('home_rest_days')
        if home_rest is None or home_rest < 0:
            home_rest = 2
        away_rest = features.get('away_rest_days')
        if away_rest is None or away_rest < 0:
            away_rest = 2
        rest_diff = home_rest - away_rest
        factor_score_rest = min(max(rest_diff / 3, -1), 1)
        direction_rest = 'home' if factor_score_rest > 0 else ('away' if factor_score_rest < 0 else 'neutral')
        factors.append({
            'name': 'Rest Days Advantage',
            'value': f'Home rest: {home_rest} days vs Away rest: {away_rest} days',
            'direction': direction_rest,
            'weight': weights['rest'],
            'score': factor_score_rest
        })
        score += factor_score_rest * weights['rest']

        # Factor 7: Roster Talent Gap
        if weights['roster'] > 0:
            home_roster = features.get('home_roster_rating', 0)
            away_roster = features.get('away_roster_rating', 0)
            
            roster_diff = home_roster - away_roster
            factor_score_roster = min(max(roster_diff / 15, -1), 1)
            direction_roster = 'home' if factor_score_roster > 0 else ('away' if factor_score_roster < 0 else 'neutral')
            
            factors.append({
                'name': 'Roster Talent Gap',
                'value': f'Home Rating {home_roster:.1f} vs Away Rating {away_roster:.1f}',
                'direction': direction_roster,
                'weight': weights['roster'],
                'score': factor_score_roster
            })
            score += factor_score_roster * weights['roster']

        # Projected margin (Home win margin)
        # Use one strength signal to avoid combining collinear ratings and score margins.
        # Avoids collinear double counting of pt_diff and net_rtg
        home_pace = features.get('home_pace')
        away_pace = features.get('away_pace')
        avg_pace = (home_pace + away_pace) / 2 if (home_pace and away_pace) else 99.5
        pace_factor = avg_pace / 100.0

        prior_weight = 1.0

        if (home_net_rtg != 0.0 or away_net_rtg != 0.0):
            base_margin = (net_rtg_diff * pace_factor) * prior_weight
        else:
            base_margin = pt_diff_gap * prior_weight

        # Net/point differentials already reflect venue and rest in their samples;
        # avoid adding unvalidated HCA, rest and roster coefficients a second time.
        raw_projected_margin = base_margin

        cover_edge = 0.0
        raw_cover_edge = 0.0
        win_prob = 0.50
        spread_probs = None
        anchor_spread = open_spread if open_spread is not None else spread_line
        # Evaluate against spread_line (Against The Spread - ATS)
        if spread_line is not None:
            try:
                s_line = float(spread_line)
            except (ValueError, TypeError):
                s_line = 0.0
            
            raw_cover_edge = raw_projected_margin + s_line
            
            # Optional market blend; zero by default until weights are validated.
            from analysis.betting_math import market_anchored_projection, analyze_line_movement
            anchor_spread = open_spread if (open_spread is not None) else s_line
            market_expected_margin = -float(anchor_spread)
            market_weight_spread = max(0.0, min(1.0, float(market_weight)))
            projected_margin = market_anchored_projection(raw_projected_margin, market_expected_margin, market_weight=market_weight_spread)
            cover_edge = projected_margin + s_line

            from analysis.betting_math import estimate_residual_sigma, empirical_line_probabilities, calculate_spread_probabilities
            sigma = estimate_residual_sigma(training_residuals) if training_residuals and len(training_residuals) >= 2 else 12.5
            probability_method = 'empirical_training_residuals' if training_residuals is not None and len(training_residuals) >= 2 else 'normal_unvalidated_sigma'
            spread_probs = (empirical_line_probabilities(s_line, projected_margin, training_residuals)
                            if training_residuals else calculate_spread_probabilities(s_line, projected_margin, sigma))
            win_prob = spread_probs['win']
            push_prob = spread_probs['push']
            factor_score_cover = min(max(cover_edge / 5.0, -1.0), 1.0)
            direction_cover = 'home' if factor_score_cover > 0 else ('away' if factor_score_cover < 0 else 'neutral')
            factors.append({
                'name': 'Projected Margin vs Spread Line',
                'value': f'Proj Margin {projected_margin:+.1f} vs Line {s_line:+.1f} (Cover Edge: {cover_edge:+.1f} pts | Win Prob: {win_prob:.1%})',
                'direction': direction_cover,
                'weight': 0.60,
                'score': factor_score_cover
            })
            # Combine team dominance (20%) with ATS cover edge (80%)
            score = (score * 0.20) + (factor_score_cover * 0.80)

            # Recommendations calibrated for Market-Anchored Cover Edge:
            # Conviction Filter: In early season (min_games < 10), high variance wipes out thin edges.
            # Suppress LEAN picks (< 2.0 pts cover edge) to NEUTRAL / PASS to protect bankroll.
            if is_early_season:
                if cover_edge >= 2.0:
                    recommendation = 'STRONG HOME'
                elif cover_edge <= -2.0:
                    recommendation = 'STRONG AWAY'
                else:
                    recommendation = 'NEUTRAL'
            else:
                if cover_edge >= 2.0:
                    recommendation = 'STRONG HOME'
                elif cover_edge >= 0.8:
                    recommendation = 'LEAN HOME'
                elif cover_edge <= -2.0:
                    recommendation = 'STRONG AWAY'
                elif cover_edge <= -0.8:
                    recommendation = 'LEAN AWAY'
                else:
                    recommendation = 'NEUTRAL'
        else:
            sigma = None
            push_prob = 0.0
            probability_method = 'not_calculated'
            projected_margin = raw_projected_margin
            recommendation = 'NEUTRAL'

        # Probability for the recommended side
        pick_win_prob = win_prob if 'HOME' in recommendation else (spread_probs['loss'] if spread_probs and 'AWAY' in recommendation else (1.0 - win_prob if 'AWAY' in recommendation else 0.50))
        
        # Line movement is descriptive and does not identify betting flow.
        from analysis.betting_math import analyze_line_movement
        line_mv = analyze_line_movement(open_spread, spread_line, market_type='spread', recommendation=recommendation, home_team=home_team, away_team=away_team, is_live=is_live)
        
        # Anomaly Detection (flagging extreme discrepancies without clamping)
        is_anomaly = abs(cover_edge) > 6.0 if spread_line is not None else False
        anomaly_warning = "⚠️ Extreme Spread Discrepancy (>6 pts): Potential data anomaly / early season distortion" if is_anomaly else None
        
        # Pure Raw Mode: Use pure win_prob directly without artificial clamping
        raw_win_prob = pick_win_prob
            
        confidence = abs(pick_win_prob - 0.5) * 2.0

        # Value and sizing are unavailable without actual market odds.
        from analysis.betting_math import evaluate_market_pick
        
        market_odds_str = market_odds
        pick_idx = 0 if 'HOME' in recommendation else 1
        eval_metrics = evaluate_market_pick(pick_win_prob, book_odds=market_odds_str, pick_idx=pick_idx, kelly_fraction=0.25, push_prob=push_prob if spread_line is not None else 0.0)
        
        # Line movement is not validated as a bet-selection rule.
        is_market_clash = False

        is_bet = recommendation != 'NEUTRAL' and not is_market_clash and eval_metrics['ev'] is not None
        ev_val = eval_metrics['ev'] if is_bet else None
        edge_pct_val = eval_metrics['edge_pct'] if is_bet else None
        full_kelly = eval_metrics['full_kelly'] if is_bet else 0.0
        quarter_kelly = (eval_metrics['recommended_stake_pct'] / 100.0) if is_bet else 0.0
        if is_market_clash:
            kelly_rec = "0.0% (ข้าม - สัญญาณขัดแย้งกับกระแสเงินตลาด)"
        elif quarter_kelly > 0:
            kelly_rec = f"{quarter_kelly * 100:.1f}% of bankroll (1/4 Kelly)"
        else:
            kelly_rec = "0% (No Bet)"

        summary = f"Spread {spread_line}: {recommendation} (Edge: {cover_edge:+.1f} pts | Win Prob: {pick_win_prob:.1%})\n"
        if is_market_clash:
            summary += f"  ⚠️ Market Clash: แต้มต่อไหลสวนทางโมเดล {line_mv.get('diff', 0):+.1f} แต้ม — แนะนำ ข้าม (PASS) เพื่อรักษาทุน\n"
        elif is_bet:
            summary += f"  ⚡ Value & Sizing: +EV: {ev_val:+.1%} | Kelly: {quarter_kelly:.1%} (1/4 Kelly)\n"
        if line_mv.get('badge'):
            summary += f"  {line_mv['badge']}: {line_mv['detail']}\n"
        if is_anomaly:
            summary += f"  {anomaly_warning}\n"
        for f in factors:
            arrow = '↑' if f['direction'] == 'home' else ('↓' if f['direction'] == 'away' else '→')
            summary += f"  {arrow} {f['name']}: {f['value']}\n"

        return {
            'recommendation': recommendation,
            'confidence': confidence,
            'win_prob': pick_win_prob,
            'raw_win_prob': raw_win_prob,
            'is_anomaly': is_anomaly,
            'anomaly_warning': anomaly_warning,
            'is_market_clash': is_market_clash,
            'cover_edge': cover_edge,
            'raw_cover_edge': raw_cover_edge,
            'projected_margin': projected_margin,
            'raw_projected_margin': raw_projected_margin,
            'anchor_spread': anchor_spread,
            'market_weight': market_weight,
            'open_spread': open_spread,
            'line_movement': line_mv,
            'ev': ev_val,
            'edge_pct': edge_pct_val,
            'fair_prob': eval_metrics['fair_prob'],
            'market_prob': eval_metrics['market_prob'],
            'kelly_fraction': full_kelly,
            'kelly_quarter': quarter_kelly,
            'kelly_recommendation': kelly_rec,
            'score': score,
            'factors': factors,
            'summary': summary,
            'features': features,
            'sigma_margin': sigma if spread_line is not None else None,
            'probability_method': probability_method if spread_line is not None else 'not_calculated',
            'push_prob': push_prob if spread_line is not None else 0.0
        }

    def full_matchup_analysis(self, home_team: str, away_team: str,
                               spread_line: float = None,
                               over_under_line: float = None,
                               open_spread: float = None,
                               open_line: float = None,
                               market_weight: float = 0.0,
                               is_live: bool = False,
                               game_date: str = None,
                               spread_odds: Any = None,
                               total_odds: Any = None,
                               margin_residuals: Any = None,
                               total_residuals: Any = None,
                               **kwargs) -> Dict[str, Any]:
        """Complete matchup analysis combining O/U and Spread with Market Anchoring and Line Movement.
        Returns dict with:
        - home_team, away_team
        - over_under: result from analyze_over_under (or None if no line)
        - spread: result from analyze_spread (or None if no line)
        - home_profile: team recent form
        - away_profile: team recent form
        - h2h: head-to-head history
        """
        result = {
            'home_team': home_team,
            'away_team': away_team,
        }
        
        # Home and away profiles
        if hasattr(self.fe, 'get_team_recent_form'):
            result['home_profile'] = self.fe.get_team_recent_form(home_team, before_date=game_date)
            result['away_profile'] = self.fe.get_team_recent_form(away_team, before_date=game_date)
        else:
            result['home_profile'] = {}
            result['away_profile'] = {}
        
        # Head to head
        if hasattr(self.fe, 'get_head_to_head'):
            result['h2h'] = self.fe.get_head_to_head(home_team, away_team, before_date=game_date)
        else:
            result['h2h'] = {}
        
        # Always compute features so injuries and team matchup indicators exist even if lines are not out yet
        features = self.fe.compute_matchup_features(
            home_team, away_team,
            over_under_line=over_under_line,
            spread_line=spread_line,
            game_date=game_date
        ) if hasattr(self.fe, 'compute_matchup_features') else {}
        result['features'] = features

        # O/U analysis
        if over_under_line is not None:
            result['over_under'] = self.analyze_over_under(
                home_team, away_team,
                over_under_line=over_under_line,
                open_line=open_line,
                market_weight=market_weight,
                is_live=is_live,
                game_date=game_date,
                market_odds=total_odds,
                training_residuals=total_residuals
            )
        else:
            result['over_under'] = None
            
        # Spread analysis
        if spread_line is not None:
            result['spread'] = self.analyze_spread(
                home_team, away_team,
                spread_line=spread_line,
                open_spread=open_spread,
                market_weight=market_weight,
                is_live=is_live,
                game_date=game_date,
                market_odds=spread_odds,
                training_residuals=margin_residuals
            )
        else:
            result['spread'] = None

        # AI CatBoost Residual Adjustments (Unified across Web Dashboard, Alerts & Ledger)
        try:
            from analysis.ai_residual_model import OverUnderResidualModel, SpreadResidualModel
            if not hasattr(self, '_ai_ou_model'):
                self._ai_ou_model = OverUnderResidualModel()
            if not hasattr(self, '_ai_sp_model'):
                self._ai_sp_model = SpreadResidualModel()

            # 1. Over/Under AI Adjustment
            if self._ai_ou_model and self._ai_ou_model.is_fitted and result.get('over_under'):
                ou_res = result['over_under']
                adj_ou = self._ai_ou_model.get_adjusted_projection(
                    home_team, away_team, self.fe, self, game_date=game_date, over_under_line=over_under_line
                )
                ou_res['ai_residual'] = adj_ou['ai_residual']
                ou_res['base_math_total'] = adj_ou['base_math_total']
                ou_res['adjusted_total'] = adj_ou['adjusted_total']
                if adj_ou['ai_residual'] != 0.0 and over_under_line is not None:
                    ou_res['projected_total'] = adj_ou['adjusted_total']
                    ou_res['total_edge'] = round(adj_ou['adjusted_total'] - over_under_line, 1)
                    if ou_res['total_edge'] >= 2.0:
                        ou_res['recommendation'] = 'STRONG OVER'
                    elif ou_res['total_edge'] <= -2.0:
                        ou_res['recommendation'] = 'STRONG UNDER'
                    elif ou_res['total_edge'] >= 1.0:
                        ou_res['recommendation'] = 'LEAN OVER'
                    elif ou_res['total_edge'] <= -1.0:
                        ou_res['recommendation'] = 'LEAN UNDER'
                    else:
                        ou_res['recommendation'] = 'NEUTRAL'

            # 2. Spread AI Adjustment
            if self._ai_sp_model and self._ai_sp_model.is_fitted and result.get('spread'):
                sp_res = result['spread']
                adj_sp = self._ai_sp_model.get_adjusted_spread_projection(
                    home_team, away_team, self.fe, self, game_date=game_date, spread_line=spread_line
                )
                sp_res['ai_residual'] = adj_sp['ai_residual']
                sp_res['base_math_margin'] = adj_sp['base_math_margin']
                sp_res['adjusted_margin'] = adj_sp['adjusted_margin']
                if adj_sp['ai_residual'] != 0.0 and spread_line is not None:
                    sp_res['projected_margin'] = adj_sp['adjusted_margin']
                    sp_res['cover_edge'] = round(adj_sp['adjusted_margin'] + spread_line, 1)
                    if sp_res['cover_edge'] >= 2.0:
                        sp_res['recommendation'] = 'STRONG HOME'
                    elif sp_res['cover_edge'] <= -2.0:
                        sp_res['recommendation'] = 'STRONG AWAY'
                    elif sp_res['cover_edge'] >= 1.0:
                        sp_res['recommendation'] = 'LEAN HOME'
                    elif sp_res['cover_edge'] <= -1.0:
                        sp_res['recommendation'] = 'LEAN AWAY'
                    else:
                        sp_res['recommendation'] = 'NEUTRAL'
        except Exception as e:
            logger.warning(f"AI Residual adjustment failed gracefully in full_matchup_analysis: {e}")
            
        return result

    def generate_team_profile(self, team: str) -> Dict[str, Any]:
        """Generate comprehensive team profile.
        Returns:
        - recent_form: from feature_engineer.get_team_recent_form()
        - home_away_splits: from feature_engineer.get_home_away_splits()
        - advanced_stats: pace, ortg, drtg, net_rating (if available)
        - ou_tendency: over/under tendency description
        - spread_tendency: ATS tendency description
        """
        recent_form = self.fe.get_team_recent_form(team) if hasattr(self.fe, 'get_team_recent_form') else {}
        home_away_splits = self.fe.get_home_away_splits(team) if hasattr(self.fe, 'get_home_away_splits') else {}
        
        # Default advanced stats
        advanced_stats = {
            'pace': recent_form.get('pace', 99.5),
            'ortg': recent_form.get('ortg', 100.0),
            'drtg': recent_form.get('drtg', 100.0),
            'net_rating': recent_form.get('net_rating', 0.0)
        }
        
        ou_tendency = 'Neutral O/U tendency'
        spread_tendency = 'Neutral ATS tendency'
        
        return {
            'team': team,
            'recent_form': recent_form,
            'home_away_splits': home_away_splits,
            'advanced_stats': advanced_stats,
            'ou_tendency': ou_tendency,
            'spread_tendency': spread_tendency
        }
