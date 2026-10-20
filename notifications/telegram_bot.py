"""Telegram Notification Service for NBA Analytics System."""

import os
import requests
import logging
from typing import Optional, List
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Client for dispatching rich HTML notifications to Telegram channels or users."""

    def __init__(self, bot_token: Optional[str] = None, chat_id: Optional[str] = None):
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN', '') if bot_token is None else bot_token
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID', '') if chat_id is None else chat_id
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"

    @property
    def is_configured(self) -> bool:
        """Check if bot token and chat ID are configured."""
        return bool(self.bot_token and self.chat_id)

    @staticmethod
    def split_message(text: str, max_chunk_size: int = 3800) -> List[str]:
        """Split a long message into safe chunks preserving line boundaries."""
        if len(text) <= max_chunk_size:
            return [text]

        chunks = []
        lines = text.split('\n')
        current_chunk = []
        current_length = 0

        for line in lines:
            line_len = len(line) + 1
            if current_length + line_len > max_chunk_size:
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = [line]
                    current_length = line_len
                else:
                    # Single extremely long line, forced split
                    chunks.append(line[:max_chunk_size])
                    current_chunk = [line[max_chunk_size:]]
                    current_length = len(current_chunk[0]) + 1
            else:
                current_chunk.append(line)
                current_length += line_len

        if current_chunk:
            chunks.append('\n'.join(current_chunk))

        return chunks

    def send_message(
        self,
        text: str,
        parse_mode: str = 'HTML',
        disable_notification: bool = False
    ) -> bool:
        """Send formatted message to Telegram with automatic chunking.

        Args:
            text: Message body (HTML formatted)
            parse_mode: 'HTML' or 'MarkdownV2'
            disable_notification: If True, delivers silently without sound

        Returns:
            bool: True if all message chunks were delivered successfully.
        """
        if not self.is_configured:
            logger.warning("Telegram Bot Token or Chat ID not configured. Message not sent.")
            return False

        chunks = self.split_message(text, max_chunk_size=3800)
        overall_success = True

        for chunk in chunks:
            payload = {
                'chat_id': self.chat_id,
                'text': chunk,
                'parse_mode': parse_mode,
                'disable_notification': disable_notification,
                'disable_web_page_preview': True
            }
            try:
                endpoint = f"{self.base_url}/sendMessage"
                resp = requests.post(endpoint, json=payload, timeout=15)
                if not resp.ok:
                    logger.error(f"Telegram API error ({resp.status_code}): {resp.text}")
                    overall_success = False
                else:
                    logger.info("Telegram message chunk dispatched successfully.")
            except Exception as exc:
                logger.error(f"Network error sending Telegram message: {exc}")
                overall_success = False

        return overall_success
