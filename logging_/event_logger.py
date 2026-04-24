"""
Event Logger for Debug Mode
Logs all streaming events (input and output) to a JSONL file.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class EventLogger:
    """
    Logs streaming events to a JSONL file for debugging.
    Each line contains metadata (timestamp, elapsed, direction, type) plus
    the raw event data exactly as sent/received — no content modification.
    """

    def __init__(self, output_path: str):
        """
        Initialize event logger.

        Args:
            output_path: Path to the JSONL output file
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.output_path, 'w')
        self._start_time = time.time()
        print(f"📋 Event logger initialized: {self.output_path}")

    def log_event(self, direction: str, event_type: str, data: Optional[Dict[str, Any]] = None):
        """
        Write a timestamped event line with the raw event data.

        Args:
            direction: 'input' or 'output'
            event_type: Event type (e.g. 'textInput', 'audioOutput', 'contentStart', etc.)
            data: Raw event data dict — logged as-is without modification
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_s": round(time.time() - self._start_time, 3),
            "direction": direction,
            "type": event_type,
            "data": data or {},
        }

        try:
            self._file.write(json.dumps(entry, default=str) + "\n")
            self._file.flush()
        except Exception as e:
            print(f"⚠️  Event logger write error: {e}")

    def close(self):
        """Flush and close the file."""
        if self._file and not self._file.closed:
            self._file.close()
