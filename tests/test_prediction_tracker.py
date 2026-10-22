import unittest
import pandas as pd
from unittest.mock import MagicMock
from analysis.prediction_tracker import (
    grade_over_under_pick,
    grade_spread_pick,
    grade_verdict_pick,
    calculate_track_record,
    record_slate_predictions,
    grade_finished_games
)


class TestPredictionTracker(unittest.TestCase):
    """Unit tests for prediction grading, deduplication, and track record calculation."""

    def test_grade_over_under_outcomes(self):
        # OVER tests
        self.assertEqual(grade_over_under_pick("STRONG OVER", 220.0, 225.0), 'WIN')
        self.assertEqual(grade_over_under_pick("LEAN OVER", 220.0, 215.0), 'LOSS')
        self.assertEqual(grade_over_under_pick("STRONG OVER", 220.0, 220.0), 'PUSH')

        # UNDER tests
        self.assertEqual(grade_over_under_pick("STRONG UNDER", 220.0, 215.0), 'WIN')
        self.assertEqual(grade_over_under_pick("LEAN UNDER", 220.0, 225.0), 'LOSS')
        self.assertEqual(grade_over_under_pick("STRONG UNDER", 220.0, 220.0), 'PUSH')

        # Neutral / Invalid
        self.assertEqual(grade_over_under_pick("NEUTRAL", 220.0, 225.0), 'SKIPPED')
        self.assertEqual(grade_over_under_pick(None, 220.0, 225.0), 'SKIPPED')

    def test_grade_spread_outcomes(self):
        # HOME tests (Home favored by 3.5 pts -> line is -3.5)
        # Home wins by 4: margin = +4, margin + (-3.5) = +0.5 > 0 -> WIN
        self.assertEqual(grade_spread_pick("STRONG HOME", -3.5, 4.0), 'WIN')
        # Home wins by 3: margin = +3, margin + (-3.5) = -0.5 < 0 -> LOSS
        self.assertEqual(grade_spread_pick("LEAN HOME", -3.5, 3.0), 'LOSS')
        # Home underdog by 3.5 pts -> line is +3.5
        # Home loses by 3: margin = -3, margin + 3.5 = +0.5 > 0 -> WIN
        self.assertEqual(grade_spread_pick("STRONG HOME", 3.5, -3.0), 'WIN')

        # AWAY tests
        # Home favored by 3.5 -> Away covers if home wins by <= 3
        self.assertEqual(grade_spread_pick("STRONG AWAY", -3.5, 3.0), 'WIN')
        self.assertEqual(grade_spread_pick("LEAN AWAY", -3.5, 4.0), 'LOSS')

        # Push test (Home favored by 3, wins by 3)
        self.assertEqual(grade_spread_pick("STRONG HOME", -3.0, 3.0), 'PUSH')

        # Neutral / Skipped
        self.assertEqual(grade_spread_pick("NEUTRAL", -3.5, 5.0), 'SKIPPED')
        self.assertEqual(grade_spread_pick(None, -3.5, 5.0), 'SKIPPED')

    def test_grade_verdict_outcomes(self):
        # FOLLOW_MODEL follows the recommendation
        self.assertEqual(grade_verdict_pick("FOLLOW_MODEL", "STRONG UNDER", 220.0, 210.0), 'WIN')
        self.assertEqual(grade_verdict_pick("FOLLOW_MODEL", "STRONG UNDER", 220.0, 225.0), 'LOSS')

        # CAPITAL_PRESERVATION and PASS safely skip betting
        self.assertEqual(grade_verdict_pick("CAPITAL_PRESERVATION", "STRONG UNDER", 220.0, 225.0), 'SKIPPED')
        self.assertEqual(grade_verdict_pick("PASS", "STRONG UNDER", 220.0, 215.0), 'SKIPPED')

    def test_calculate_track_record_metrics(self):
        df = pd.DataFrame([
            {'status': 'GRADED', 'ou_recommendation': 'STRONG UNDER', 'ou_result': 'WIN', 'sp_recommendation': 'LEAN HOME', 'sp_result': 'LOSS', 'verdict_result': 'WIN'},
            {'status': 'GRADED', 'ou_recommendation': 'STRONG OVER', 'ou_result': 'WIN', 'sp_recommendation': 'STRONG AWAY', 'sp_result': 'WIN', 'verdict_result': 'WIN'},
            {'status': 'GRADED', 'ou_recommendation': 'LEAN UNDER', 'ou_result': 'LOSS', 'sp_recommendation': 'NEUTRAL', 'sp_result': 'SKIPPED', 'verdict_result': 'LOSS'},
            {'status': 'GRADED', 'ou_recommendation': 'STRONG OVER', 'ou_result': 'PUSH', 'sp_recommendation': 'STRONG HOME', 'sp_result': 'WIN', 'verdict_result': 'PUSH'},
            {'status': 'PENDING', 'ou_recommendation': 'STRONG UNDER', 'ou_result': None, 'sp_recommendation': 'LEAN AWAY', 'sp_result': None, 'verdict_result': None},
        ])

        stats = calculate_track_record(df)
        self.assertEqual(stats['total_predictions'], 5)
        self.assertEqual(stats['graded_games'], 4)
        self.assertEqual(stats['pending_games'], 1)

        # Over/Under: WIN, WIN, LOSS, PUSH -> 2 wins, 1 loss, 1 push -> 2/3 = 66.7%
        ou_s = stats['over_under']
        self.assertEqual(ou_s['wins'], 2)
        self.assertEqual(ou_s['losses'], 1)
        self.assertEqual(ou_s['pushes'], 1)
        self.assertAlmostEqual(ou_s['win_rate'], 66.7, places=1)

        # Spread: LOSS, WIN, SKIPPED, WIN -> 2 wins, 1 loss, 0 pushes -> 2/3 = 66.7%
        sp_s = stats['spread']
        self.assertEqual(sp_s['wins'], 2)
        self.assertEqual(sp_s['losses'], 1)
        self.assertAlmostEqual(sp_s['win_rate'], 66.7, places=1)

        # Strong tier:
        # OU row 0: WIN (Strong)
        # OU row 1: WIN (Strong), SP row 1: WIN (Strong)
        # SP row 3: WIN (Strong), OU row 3: PUSH (Strong)
        # Total Strong: 4 wins, 0 losses, 1 push -> 100% win rate
        str_s = stats['strong']
        self.assertEqual(str_s['wins'], 4)
        self.assertEqual(str_s['losses'], 0)
        self.assertEqual(str_s['win_rate'], 100.0)

    def test_grade_finished_games_mock(self):
        # Test grading with mocked final scoreboard
        import tempfile
        import shutil
        from unittest.mock import patch

        scoreboard_mock = pd.DataFrame([
            {'game_id': '99901', 'status': 'STATUS_FINAL', 'home_score': 110, 'away_score': 100},
            {'game_id': '99902', 'status': 'STATUS_SCHEDULED', 'home_score': 0, 'away_score': 0}
        ])

        ledger_mock = pd.DataFrame([
            {
                'game_id': '99901',
                'date': '2026-10-21',
                'home_team': 'Team A',
                'away_team': 'Team B',
                'spread_line': -5.5,
                'total_line': 205.0,
                'ou_recommendation': 'STRONG OVER',
                'sp_recommendation': 'STRONG HOME',
                'ou_verdict': 'FOLLOW_MODEL',
                'home_score': None,
                'away_score': None,
                'actual_total': None,
                'actual_margin': None,
                'ou_result': None,
                'sp_result': None,
                'verdict_result': None,
                'status': 'PENDING'
            }
        ])

        with patch('analysis.prediction_tracker.load_ledger', return_value=ledger_mock):
            with patch('analysis.prediction_tracker.save_ledger') as mock_save:
                graded_count = grade_finished_games(scoreboard_mock)
                self.assertEqual(graded_count, 1)
                mock_save.assert_called_once()
                saved_df = mock_save.call_args[0][0]
                self.assertEqual(saved_df.at[0, 'status'], 'GRADED')
                self.assertEqual(saved_df.at[0, 'ou_result'], 'WIN')  # 210 > 205
                self.assertEqual(saved_df.at[0, 'sp_result'], 'WIN')  # Home margin +10 covers -5.5
                self.assertEqual(saved_df.at[0, 'verdict_result'], 'WIN')


if __name__ == '__main__':
    unittest.main()
