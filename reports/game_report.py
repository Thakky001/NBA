import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class GameReporter:
    """Generates formatted text reports for NBA game analysis."""

    SEPARATOR = '═' * 56
    THIN_SEP = '─' * 56

    def format_matchup_report(self, analysis: Dict[str, Any],
                                game_info: Dict[str, Any] = None) -> str:
        """Format a complete matchup analysis into a readable text report.
        
        Args:
            analysis: Output from TrendAnalyzer.full_matchup_analysis()
            game_info: Optional dict with date, venue, game_id, status
        """
        game_info = game_info or {}
        home_team = analysis.get('home_team', 'Home')
        away_team = analysis.get('away_team', 'Away')
        date = game_info.get('date', 'Unknown Date')
        venue = game_info.get('venue', 'Unknown Venue')
        
        spread_line = analysis.get('spread', {}).get('features', {}).get('spread_line', 'N/A') if analysis.get('spread') else 'N/A'
        ou_line = analysis.get('over_under', {}).get('features', {}).get('over_under_line', 'N/A') if analysis.get('over_under') else 'N/A'
        home_ml = game_info.get('home_ml', 'N/A')
        away_ml = game_info.get('away_ml', 'N/A')

        home_profile = analysis.get('home_profile') or {}
        away_profile = analysis.get('away_profile') or {}
        
        home_wins = home_profile.get('wins', 0)
        home_losses = home_profile.get('losses', 0)
        home_win_pct = home_wins / (home_wins + home_losses) if (home_wins + home_losses) > 0 else 0.0
        home_pts = home_profile.get('avg_points_scored', 0.0)
        home_streak = home_profile.get('current_streak', home_profile.get('streak_str', 'N/A'))
        
        away_wins = away_profile.get('wins', 0)
        away_losses = away_profile.get('losses', 0)
        away_win_pct = away_wins / (away_wins + away_losses) if (away_wins + away_losses) > 0 else 0.0
        away_pts = away_profile.get('avg_points_scored', 0.0)
        away_streak = away_profile.get('current_streak', away_profile.get('streak_str', 'N/A'))

        lines = []
        lines.append(self.SEPARATOR)
        lines.append(f"  🏀 {home_team} vs {away_team}")
        thai_time_str = game_info.get('thai_time_display') or (f"เวลา {game_info['thai_time']} น." if game_info.get('thai_time') else None)
        if thai_time_str:
            lines.append(f"  📅 {thai_time_str} (เวลาไทย) | 📍 {venue}")
        else:
            lines.append(f"  📅 {date} | 📍 {venue} (Home: {home_team})")
        lines.append(self.SEPARATOR)
        lines.append("  CURRENT LINES:")
        lines.append(f"  │ Spread:  {home_team} {spread_line}")
        lines.append(f"  │ O/U:     {ou_line}")
        lines.append(f"  │ ML:      {home_team} {home_ml} / {away_team} {away_ml}")
        lines.append(self.THIN_SEP)
        lines.append("  TEAM PROFILES:")
        lines.append(f"  │ Home: {home_wins}-{home_losses} ({home_win_pct:.3f}) | Avg PTS: {home_pts:.1f} | Streak: {home_streak}")
        lines.append(f"  │ Away: {away_wins}-{away_losses} ({away_win_pct:.3f}) | Avg PTS: {away_pts:.1f} | Streak: {away_streak}")
        
        # Check for injuries in features
        ou_analysis = analysis.get('over_under')
        spread_analysis = analysis.get('spread')
        feat = (spread_analysis.get('features') if spread_analysis else None) or (ou_analysis.get('features') if ou_analysis else {}) or {}
        home_inj = feat.get('home_injuries', [])
        away_inj = feat.get('away_injuries', [])
        if home_inj or away_inj:
            lines.append(self.THIN_SEP)
            lines.append("  INJURY REPORT:")
            lines.append(f"  │ Home: {', '.join(home_inj) if home_inj else 'Healthy / No Key Out'}")
            lines.append(f"  │ Away: {', '.join(away_inj) if away_inj else 'Healthy / No Key Out'}")
        
        if ou_analysis:
            lines.append(self.THIN_SEP)
            lines.append(f"  OVER/UNDER ANALYSIS (Line: {ou_line}):")
            for factor in ou_analysis.get('factors', []):
                lines.append(f"  │ {factor['name']}: {factor['value']}")
            edge_str = f" | Edge: {ou_analysis.get('total_edge', 0.0):+.1f} pts" if 'total_edge' in ou_analysis else ""
            prob_str = f" | Win Prob: {ou_analysis.get('win_prob', 0.5):.1%}" if 'win_prob' in ou_analysis else ""
            lines.append(f"  │ ▶ {ou_analysis.get('recommendation', 'N/A')}{edge_str}{prob_str}")
            if ou_analysis.get('ev') is not None and ou_analysis.get('recommendation') != 'NEUTRAL':
                ev_pct = ou_analysis.get('ev', 0.0) * 100.0
                kelly_pct = ou_analysis.get('kelly_quarter', 0.0) * 100.0
                lines.append(f"  │ ⚡ VALUE & SIZING: +EV: {ev_pct:+.1f}% | Kelly Stake: {kelly_pct:.1f}% (1/4 Kelly)")
            lm_ou = ou_analysis.get('line_movement')
            if lm_ou and lm_ou.get('badge'):
                lines.append(f"  │ {lm_ou['badge']}: {lm_ou['detail']}")
            if ou_analysis.get('is_anomaly'):
                lines.append("  │ ⚠️ ANOMALY WARNING: Line discrepancy exceeds 8.0 pts")
            
        if spread_analysis:
            lines.append(self.THIN_SEP)
            lines.append(f"  SPREAD ANALYSIS (Line: {spread_line}):")
            for factor in spread_analysis.get('factors', []):
                lines.append(f"  │ {factor['name']}: {factor['value']}")
            edge_str = f" | Edge: {spread_analysis.get('cover_edge', 0.0):+.1f} pts" if 'cover_edge' in spread_analysis else ""
            prob_str = f" | Win Prob: {spread_analysis.get('win_prob', 0.5):.1%}" if 'win_prob' in spread_analysis else ""
            lines.append(f"  │ ▶ {spread_analysis.get('recommendation', 'N/A')}{edge_str}{prob_str}")
            if spread_analysis.get('ev') is not None and spread_analysis.get('recommendation') != 'NEUTRAL':
                ev_pct = spread_analysis.get('ev', 0.0) * 100.0
                kelly_pct = spread_analysis.get('kelly_quarter', 0.0) * 100.0
                lines.append(f"  │ ⚡ VALUE & SIZING: +EV: {ev_pct:+.1f}% | Kelly Stake: {kelly_pct:.1f}% (1/4 Kelly)")
            lm_sp = spread_analysis.get('line_movement')
            if lm_sp and lm_sp.get('badge'):
                lines.append(f"  │ {lm_sp['badge']}: {lm_sp['detail']}")
            if spread_analysis.get('is_anomaly'):
                lines.append("  │ ⚠️ ANOMALY WARNING: Spread discrepancy exceeds 6.0 pts")
            
        odds_data = game_info.get('odds_data')
        if odds_data:
            lines.append(self.THIN_SEP)
            lines.append(f"  🛒 LINE SHOPPING (Best Odds):")
            shopping_items = 0
            
            # Print best spread for the recommended pick
            if spread_analysis and ('STRONG' in spread_analysis.get('recommendation', '') or 'LEAN' in spread_analysis.get('recommendation', '')):
                pick_team = spread_analysis.get('recommendation').replace(' (Spread)', '')
                is_home = 'HOME' in spread_analysis.get('recommendation', '').upper()
                pick_team = home_team if is_home else away_team
                
                best_bookie = "N/A"
                best_val = -999  # Always want highest spread
                best_price = ""
                
                # If picking home, we want highest home spread point. (e.g. +6 > +5, -4 > -5)
                # If picking away, we want highest away spread point.
                for bookie, spreads in odds_data.get('spreads', {}).items():
                    if is_home and 'home' in spreads:
                        h_val = spreads['home']
                        if h_val is not None and h_val > best_val:
                            best_val = h_val
                            best_bookie = bookie
                            best_price = spreads.get('home_price', '')
                    elif not is_home and 'away' in spreads:
                        a_val = spreads['away']
                        if a_val is not None and a_val > best_val:
                            best_val = a_val
                            best_bookie = bookie
                            best_price = spreads.get('away_price', '')
                            
                if best_bookie != "N/A":
                    lines.append(f"  │ Spread: Bet {pick_team} {best_val} ({best_price}) at 📍 {best_bookie.upper()}")
                    shopping_items += 1

            # Print best O/U for recommended pick
            if ou_analysis and ('STRONG' in ou_analysis.get('recommendation', '') or 'LEAN' in ou_analysis.get('recommendation', '')):
                rec = ou_analysis.get('recommendation')
                is_over = 'OVER' in rec.upper()
                
                best_bookie = "N/A"
                best_val = -999 if is_over else 999
                best_price = ""
                
                for bookie, totals in odds_data.get('totals', {}).items():
                    # For Over, we want LOWEST total point
                    if is_over and 'over' in totals:
                        o_val = totals['over']
                        if o_val is not None and (best_val == -999 or o_val < best_val):
                            best_val = o_val
                            best_bookie = bookie
                            best_price = totals.get('over_price', '')
                    # For Under, we want HIGHEST total point
                    elif not is_over and 'under' in totals:
                        u_val = totals['under']
                        if u_val is not None and (best_val == 999 or u_val > best_val):
                            best_val = u_val
                            best_bookie = bookie
                            best_price = totals.get('under_price', '')
                            
                if best_bookie != "N/A":
                    dir_str = "OVER" if is_over else "UNDER"
                    lines.append(f"  │ Total: Bet {dir_str} {best_val} ({best_price}) at 📍 {best_bookie.upper()}")
                    shopping_items += 1

            if shopping_items == 0:
                lines.append("  │ Live Vegas sportsbook comparison unavailable for this matchup.")

        lines.append(self.SEPARATOR)
        
        return '\n'.join(lines)

    def format_team_profile_report(self, profile: Dict[str, Any], team_name: str) -> str:
        """Format a team profile into readable text."""
        lines = []
        lines.append(self.SEPARATOR)
        lines.append(f"  🏀 TEAM PROFILE: {team_name}")
        lines.append(self.SEPARATOR)
        profile = profile or {}
        adv = profile.get('advanced_stats') or {}
        lines.append(f"  │ Pace: {adv.get('pace', 'N/A')}")
        lines.append(f"  │ ORTG: {adv.get('ortg', 'N/A')}")
        lines.append(f"  │ DRTG: {adv.get('drtg', 'N/A')}")
        lines.append(f"  │ Net Rating: {adv.get('net_rating', 'N/A')}")
        lines.append(f"  │ O/U Tendency: {profile.get('ou_tendency', 'N/A')}")
        lines.append(f"  │ Spread Tendency: {profile.get('spread_tendency', 'N/A')}")
        lines.append(self.SEPARATOR)
        return '\n'.join(lines)

    def format_daily_slate(self, games: list) -> str:
        """Format all games for a day into a summary report.
        
        Args:
            games: list of dicts, each with 'analysis' and 'game_info' keys
        
        Returns formatted string with:
        - Header with date
        - Quick picks summary (best O/U and Spread plays)
        - Individual game reports
        """
        if not games:
            return "No games available for daily slate."
            
        lines = []
        lines.append(self.SEPARATOR)
        lines.append("  DAILY SLATE SUMMARY")
        lines.append(self.SEPARATOR)
        lines.append("  QUICK PICKS:")
        for game in games:
            lines.append("  " + self.format_quick_pick(game.get('analysis', {})))
        
        lines.append("\n  DETAILED REPORTS:\n")
        for game in games:
            lines.append(self.format_matchup_report(game.get('analysis', {}), game.get('game_info', {})))
            lines.append("")
            
        return '\n'.join(lines)

    def format_quick_pick(self, analysis: Dict[str, Any]) -> str:
        """One-line summary of a game pick.
        Example: '🔥 LVA vs NYL | O 157.5 (65% | +EV +8.2%) | LVA -5.5 (58% | +EV +11.1%)'
        """
        home = analysis.get('home_team', 'Home')
        away = analysis.get('away_team', 'Away')
        ou = analysis.get('over_under') or {}
        spread = analysis.get('spread') or {}
        
        ou_str = "N/A"
        if ou:
            ou_line = ou.get('features', {}).get('over_under_line', '')
            ev_ou = f" | +EV {ou['ev']*100:+.1f}%" if ou.get('ev') and ou.get('recommendation') != 'NEUTRAL' else ""
            ou_str = f"{ou.get('recommendation', 'N/A')} {ou_line} ({ou.get('confidence', 0):.0%}{ev_ou})"
            
        spread_str = "N/A"
        if spread:
            spread_line = spread.get('features', {}).get('spread_line', '')
            ev_sp = f" | +EV {spread['ev']*100:+.1f}%" if spread.get('ev') and spread.get('recommendation') != 'NEUTRAL' else ""
            spread_str = f"{spread.get('recommendation', 'N/A')} {spread_line} ({spread.get('confidence', 0):.0%}{ev_sp})"
            
        return f"🔥 {home} vs {away} | {ou_str} | {spread_str}"
