"""
Conversation History for Session Continuation
Tracks USER and ASSISTANT messages for replay into new Nova Sonic sessions.
"""

import json
import uuid
from typing import List, Tuple


# Limits
MAX_MESSAGE_BYTES = 1024
MAX_TOTAL_BYTES = 40 * 1024  # 40KB


class ConversationHistory:
    """
    Stores conversation messages and serializes them as Bedrock event JSON
    for replaying into a new Nova Sonic session.
    """

    def __init__(self):
        self._messages: List[Tuple[str, str]] = []  # (role, content)
        self._total_bytes = 0

    def add_message(self, role: str, content: str):
        """
        Add a message to history.

        Args:
            role: "USER" or "ASSISTANT"
            content: The message text (truncated to 1024 bytes)
        """
        # Truncate content to MAX_MESSAGE_BYTES
        encoded = content.encode('utf-8')
        if len(encoded) > MAX_MESSAGE_BYTES:
            # Truncate at byte boundary, decode safely
            encoded = encoded[:MAX_MESSAGE_BYTES]
            content = encoded.decode('utf-8', errors='ignore')

        msg_bytes = len(content.encode('utf-8'))

        # Evict oldest messages if total would exceed limit
        while self._messages and (self._total_bytes + msg_bytes) > MAX_TOTAL_BYTES:
            _, old_content = self._messages.pop(0)
            self._total_bytes -= len(old_content.encode('utf-8'))

        self._messages.append((role, content))
        self._total_bytes += msg_bytes

    def get_history_events(self, prompt_name: str) -> List[str]:
        """
        Serialize all messages to Bedrock event JSON strings for replay.

        Each message becomes: contentStart -> textInput -> contentEnd
        All are non-interactive with unique contentName UUIDs.

        Args:
            prompt_name: The prompt name for the new session

        Returns:
            List of JSON event strings ready to send via _send_raw_event
        """
        events = []

        for role, content in self._messages:
            content_name = str(uuid.uuid4())

            # contentStart (non-interactive)
            content_start = {
                "event": {
                    "contentStart": {
                        "promptName": prompt_name,
                        "contentName": content_name,
                        "role": role,
                        "type": "TEXT",
                        "interactive": False,
                        "textInputConfiguration": {
                            "mediaType": "text/plain"
                        }
                    }
                }
            }
            events.append(json.dumps(content_start))

            # textInput
            text_input = {
                "event": {
                    "textInput": {
                        "promptName": prompt_name,
                        "contentName": content_name,
                        "content": content
                    }
                }
            }
            events.append(json.dumps(text_input))

            # contentEnd
            content_end = {
                "event": {
                    "contentEnd": {
                        "promptName": prompt_name,
                        "contentName": content_name
                    }
                }
            }
            events.append(json.dumps(content_end))

        return events

    def get_messages(self) -> List[Tuple[str, str]]:
        """Return a copy of all stored messages as (role, content) tuples."""
        return list(self._messages)

    def clear(self):
        """Wipe all history."""
        self._messages.clear()
        self._total_bytes = 0

    @property
    def message_count(self) -> int:
        """Number of messages in history."""
        return len(self._messages)

    @property
    def total_bytes(self) -> int:
        """Total bytes of stored content."""
        return self._total_bytes
