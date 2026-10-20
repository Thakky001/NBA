"""Unit tests for Telegram notification bot and pre-game alert runner."""

import os
import unittest
from unittest.mock import patch, MagicMock

from notifications.telegram_bot import TelegramNotifier
from run_alerts import (
    load_alerted_ledger,
    save_alerted_ledger,
    mark_games_alerted,
    is_game_already_alerted,
    format_game_card,
)


class TestTelegramAlerts(unittest.TestCase):
    """Test suite verifying Telegram notifier and pre-game card formatting."""

    def test_split_message_short(self):
        """Short message below 3,800 chars remains a single chunk."""
        text = "Hello NBA!"
        chunks = TelegramNotifier.split_message(text, max_chunk_size=3800)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], text)

    def test_split_message_exceeding_telegram_limit(self):
        """Long message exceeding 4,096 chars splits into valid chunks <= 3800 chars."""
        # Create a message of 5,000 chars with multiple lines
        lines = [f"Game line {i:03d} - Detailed analytical stats and projection" for i in range(100)]
        long_text = "\n".join(lines)
        self.assertGreater(len(long_text), 4096)

        chunks = TelegramNotifier.split_message(long_text, max_chunk_size=3800)
        self.assertGreater(len(chunks), 1)

        # Every chunk must be strictly <= 3800 characters
        for idx, chunk in enumerate(chunks):
            self.assertLessEqual(
                len(chunk),
                3800,
                f"Chunk {idx} length {len(chunk)} exceeds safe limit 3800"
            )

        # Recombined text must contain all lines
        recombined = "\n".join(chunks)
        self.assertEqual(len(recombined.split("\n")), len(lines))

    def test_is_configured(self):
        """is_configured checks both token and chat_id."""
        bot_empty = TelegramNotifier(bot_token="", chat_id="")
        self.assertFalse(bot_empty.is_configured)

        bot_ready = TelegramNotifier(bot_token="123456:ABC-DEF", chat_id="987654321")
        self.assertTrue(bot_ready.is_configured)

    def test_send_message_unconfigured_fails_safely(self):
        """send_message returns False safely without crashing if credentials missing."""
        bot = TelegramNotifier(bot_token="", chat_id="")
        result = bot.send_message("Test")
        self.assertFalse(result)

    @patch('requests.post')
    def test_send_message_mock_success(self, mock_post):
        """send_message makes expected POST request to Telegram API."""
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.status_code = 200
        mock_post.return_value = mock_resp

        bot = TelegramNotifier(bot_token="fake_token", chat_id="fake_chat")
        result = bot.send_message("<b>NBA Matchup Alert</b>")
        self.assertTrue(result)
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertIn("https://api.telegram.org/botfake_token/sendMessage", args[0])
        self.assertEqual(kwargs['json']['chat_id'], "fake_chat")
        self.assertIn("NBA Matchup Alert", kwargs['json']['text'])

    def test_format_game_card_output(self):
        """format_game_card builds clean HTML with expected badges and info."""
        mock_analysis = {
            'spread': {
                'recommendation': 'STRONG HOME',
                'cover_edge': 3.5,
                'win_prob': 0.65,
                'ev': 0.12,
                'kelly_quarter': 0.03,
                'features': {
                    'spread_line': -4.5,
                    'home_injuries': ['Player A (OUT)'],
                    'away_injuries': ['Player B (GTD)']
                }
            },
            'over_under': {
                'recommendation': 'NEUTRAL',
                'features': {
                    'over_under_line': 225.0
                }
            }
        }
        mock_game_info = {
            'home_team': 'Boston Celtics',
            'away_team': 'Miami Heat',
            'thai_time': '07:30',
            'venue': 'TD Garden'
        }
        mock_odds = {
            'totals': {
                'draftkings': {'over': 224.5, 'under': 225.5}
            }
        }

        card = format_game_card(mock_analysis, mock_game_info, mock_odds)
        self.assertIn("Boston Celtics", card)
        self.assertIn("Miami Heat", card)
        self.assertIn("TD Garden", card)
        self.assertIn("07:30", card)
        self.assertIn("STRONG HOME", card)
        self.assertIn("+EV:", card)
        self.assertIn("Kelly:", card)

    def test_alerted_ledger_roundtrip(self):
        """Marking games as alerted persists and is_game_already_alerted checks accurately."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with patch('run_alerts.ALERT_LEDGER_PATH', tmp_path):
                date_str = "2099-01-01"
                test_gid = "test_game_999"

                # Initially false in fresh ledger
                self.assertFalse(is_game_already_alerted(date_str, test_gid))

                # Mark alerted
                mark_games_alerted(date_str, [test_gid])
                self.assertTrue(is_game_already_alerted(date_str, test_gid))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == '__main__':
    unittest.main()
