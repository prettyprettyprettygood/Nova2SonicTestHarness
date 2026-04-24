"""
Interaction Logger for Live Mode
Records all interactions between Claude (user) and Nova Sonic.
"""

import json
import wave
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class Turn:
    """Represents a single turn in the conversation."""
    turn_number: int
    timestamp: str
    user_message: str
    sonic_response: str
    sonic_user_transcription: Optional[str] = None  # Nova Sonic's transcription of audio input
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    audio_recorded: bool = False
    audio_file: Optional[str] = None
    sonic_session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class SessionContinuationEvent:
    """Represents a session continuation event."""
    timestamp: str
    event_type: str  # "transition" or "emergency_recovery"
    old_session_number: int
    new_session_number: int
    old_session_duration_seconds: float
    messages_replayed: int
    total_bytes_replayed: int
    conversation_history: List[Dict[str, str]]  # list of {role, content}
    after_turn_number: int = 0  # Turn number at the end of which this event occurred
    audio_chunks_sent: int = 0
    success: bool = True
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class InteractionLog:
    """Complete log of an interaction session."""
    session_id: str
    test_name: str
    start_time: str
    end_time: Optional[str] = None
    configuration: Dict[str, Any] = field(default_factory=dict)
    turns: List[Turn] = field(default_factory=list)
    total_turns: int = 0
    total_tool_calls: int = 0
    errors: List[str] = field(default_factory=list)
    session_events: List[SessionContinuationEvent] = field(default_factory=list)
    conversation_audio_file: Optional[str] = None
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "test_name": self.test_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "configuration": self.configuration,
            "turns": [turn.to_dict() for turn in self.turns],
            "total_turns": self.total_turns,
            "total_tool_calls": self.total_tool_calls,
            "errors": self.errors,
            "session_events": [e.to_dict() for e in self.session_events],
            "conversation_audio_file": self.conversation_audio_file,
            "summary": self.summary
        }


class InteractionLogger:
    """
    Logs all interactions in a conversation session.
    Supports text logging, audio recording, and structured data capture.
    """

    def __init__(
        self,
        test_name: str,
        output_dir: str = "logs",
        log_audio: bool = False,
        log_text: bool = True,
        log_tools: bool = True,
        session_id: Optional[str] = None
    ):
        """
        Initialize interaction logger.

        Args:
            test_name: Name of the test
            output_dir: Directory to save logs
            log_audio: Whether to record audio
            log_text: Whether to log text
            log_tools: Whether to log tool calls
            session_id: Optional custom session ID
        """
        self.test_name = test_name
        self.output_dir = Path(output_dir)
        self.log_audio = log_audio
        self.log_text = log_text
        self.log_tools = log_tools

        # Generate session ID
        self.session_id = session_id or self._generate_session_id()

        # Create session directory under sessions/ subdirectory
        self.session_dir = self.output_dir / "sessions" / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Create structured subdirectories
        self.audio_dir = self.session_dir / "audio"
        self._logs_dir = self.session_dir / "logs"
        self._chat_dir = self.session_dir / "chat"
        self._config_dir = self.session_dir / "config"
        self._evaluation_dir = self.session_dir / "evaluation"

        for subdir in [self.audio_dir, self._logs_dir, self._chat_dir,
                        self._config_dir, self._evaluation_dir]:
            subdir.mkdir(parents=True, exist_ok=True)

        # Initialize log
        self.log = InteractionLog(
            session_id=self.session_id,
            test_name=test_name,
            start_time=datetime.now().isoformat()
        )

        # Current turn tracking
        self.current_turn_number = 0
        self.current_turn: Optional[Turn] = None

        # Audio recording
        self.current_audio_chunks: List[bytes] = []

        print(f"📝 Logger initialized - Session: {self.session_id}")

    def _generate_session_id(self) -> str:
        """Generate a unique session ID (safe for parallel execution)."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uuid = uuid.uuid4().hex[:6]
        return f"{self.test_name}_{timestamp}_{short_uuid}"

    def set_configuration(self, config: Dict[str, Any]):
        """
        Set the test configuration.

        Args:
            config: Configuration dictionary
        """
        self.log.configuration = config

    def start_turn(self, user_message: str):
        """
        Start a new conversation turn.

        Args:
            user_message: The user's message
        """
        self.current_turn_number += 1
        self.current_turn = Turn(
            turn_number=self.current_turn_number,
            timestamp=datetime.now().isoformat(),
            user_message=user_message,
            sonic_response=""
        )

        if self.log_text:
            print(f"\n{'='*60}")
            print(f"Turn {self.current_turn_number}")
            print(f"{'='*60}")
            print(f"👤 User: {user_message}")

    def add_sonic_response(self, response: str):
        """
        Add Sonic's response to the current turn.

        Args:
            response: Sonic's response text
        """
        if self.current_turn:
            self.current_turn.sonic_response += response

            if self.log_text:
                print(f"🤖 Sonic: {response}")

    def add_tool_call(self, tool_name: str, tool_use_id: str, tool_input: Dict[str, Any], tool_result: Optional[Dict[str, Any]] = None):
        """
        Add a tool call to the current turn.

        Args:
            tool_name: Name of the tool
            tool_use_id: Tool use ID
            tool_input: Tool input parameters
            tool_result: Tool result (optional)
        """
        if self.current_turn:
            tool_call = {
                "tool_name": tool_name,
                "tool_use_id": tool_use_id,
                "tool_input": tool_input,
                "tool_result": tool_result,
                "timestamp": datetime.now().isoformat()
            }
            self.current_turn.tool_calls.append(tool_call)
            self.log.total_tool_calls += 1

            if self.log_tools:
                print(f"🔧 Tool Call: {tool_name} (ID: {tool_use_id})")
                print(f"   Input: {json.dumps(tool_input, indent=2)}")
                if tool_result:
                    print(f"   Result: {json.dumps(tool_result, indent=2)}")

    def update_tool_result(self, tool_use_id: str, tool_result: Dict[str, Any]):
        """
        Update the result of a previously logged tool call.

        Args:
            tool_use_id: Tool use ID to update
            tool_result: Tool execution result
        """
        if self.current_turn:
            for tool_call in self.current_turn.tool_calls:
                if tool_call["tool_use_id"] == tool_use_id:
                    tool_call["tool_result"] = tool_result
                    if self.log_tools:
                        print(f"✅ Tool Result ({tool_call['tool_name']}): {json.dumps(tool_result, indent=2)}")
                    break

    def add_audio_chunk(self, audio_bytes: bytes):
        """
        Add an audio chunk to the current turn.

        Args:
            audio_bytes: Audio data bytes
        """
        if self.log_audio and self.current_turn:
            self.current_audio_chunks.append(audio_bytes)

    def end_turn(self, metadata: Optional[Dict[str, Any]] = None):
        """
        End the current turn and save it.

        Args:
            metadata: Optional metadata for the turn
        """
        if not self.current_turn:
            return

        # Add metadata
        if metadata:
            self.current_turn.metadata = metadata

        # Save audio if collected
        if self.log_audio and self.current_audio_chunks:
            audio_file = self._save_audio_chunks()
            if audio_file:
                self.current_turn.audio_recorded = True
                self.current_turn.audio_file = audio_file
            self.current_audio_chunks = []

        # Add turn to log
        self.log.turns.append(self.current_turn)
        self.log.total_turns += 1

        # Reset current turn
        self.current_turn = None

    def _save_audio_chunks(self) -> Optional[str]:
        """
        Save collected audio chunks to a WAV file.

        Returns:
            Path to saved audio file, or None if no audio
        """
        if not self.current_audio_chunks:
            return None

        audio_file = self.audio_dir / f"turn_{self.current_turn_number}.wav"

        try:
            # Combine all audio chunks
            audio_data = b''.join(self.current_audio_chunks)

            # Save as WAV
            with wave.open(str(audio_file), 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(16000)  # 16kHz
                wav_file.writeframes(audio_data)

            return str(audio_file.relative_to(self.session_dir))

        except Exception as e:
            print(f"❌ Error saving audio: {e}")
            return None

    def add_event(self, event_type: str, data: Dict[str, Any]):
        """
        Add a generic event to the current turn's metadata.

        Args:
            event_type: Type of event (e.g. "hangup")
            data: Event data dictionary
        """
        event = {
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            **data
        }
        if self.current_turn:
            self.current_turn.metadata.setdefault("events", []).append(event)
        else:
            # No active turn — store as a session-level event (not an error)
            self.log.summary.setdefault("events", []).append(event)

        if self.log_text:
            print(f"📋 Event ({event_type}): {data}")

    def add_error(self, error: str):
        """
        Add an error to the log.

        Args:
            error: Error message
        """
        self.log.errors.append({
            "timestamp": datetime.now().isoformat(),
            "error": error
        })
        print(f"❌ Error: {error}")

    def add_session_continuation_event(
        self,
        event_type: str,
        old_session_number: int,
        new_session_number: int,
        old_session_duration_seconds: float,
        messages_replayed: int,
        total_bytes_replayed: int,
        conversation_history: List[Dict[str, str]],
        audio_chunks_sent: int = 0,
        success: bool = True,
        error_message: Optional[str] = None
    ):
        """
        Record a session continuation event.

        Args:
            event_type: "transition" or "emergency_recovery"
            old_session_number: Session number before continuation
            new_session_number: Session number after continuation
            old_session_duration_seconds: Duration of the old session in seconds
            messages_replayed: Number of conversation messages replayed
            total_bytes_replayed: Total bytes of replayed content
            conversation_history: List of {role, content} dicts
            audio_chunks_sent: Number of silent audio chunks sent to warm the stream
            success: Whether the continuation succeeded
            error_message: Error message if the continuation failed
        """
        event = SessionContinuationEvent(
            timestamp=datetime.now().isoformat(),
            event_type=event_type,
            old_session_number=old_session_number,
            new_session_number=new_session_number,
            old_session_duration_seconds=old_session_duration_seconds,
            messages_replayed=messages_replayed,
            total_bytes_replayed=total_bytes_replayed,
            conversation_history=conversation_history,
            after_turn_number=self.current_turn_number,
            audio_chunks_sent=audio_chunks_sent,
            success=success,
            error_message=error_message
        )
        self.log.session_events.append(event)

        if self.log_text:
            status = "SUCCESS" if success else f"FAILED: {error_message}"
            print(f"\n{'='*60}")
            print(f"Session Continuation ({event_type})")
            print(f"{'='*60}")
            print(f"  After turn: {self.current_turn_number}")
            print(f"  Session #{old_session_number} -> #{new_session_number}")
            print(f"  Old session duration: {old_session_duration_seconds:.1f}s")
            print(f"  Messages replayed: {messages_replayed}")
            print(f"  Bytes replayed: {total_bytes_replayed}")
            print(f"  Audio chunks sent: {audio_chunks_sent}")
            print(f"  Status: {status}")

    def finalize(self):
        """Finalize the log and save to file."""
        self.log.end_time = datetime.now().isoformat()

        # Generate summary
        self._generate_summary()

        # Save log
        self._save_log()

        print(f"\n{'='*60}")
        print(f"Session Complete - {self.session_id}")
        print(f"{'='*60}")
        print(f"Total Turns: {self.log.total_turns}")
        print(f"Total Tool Calls: {self.log.total_tool_calls}")
        print(f"Errors: {len(self.log.errors)}")
        print(f"Log saved to: {self.session_dir}")
        print(f"{'='*60}\n")

    def _generate_summary(self):
        """Generate summary statistics."""
        total_user_chars = sum(len(turn.user_message) for turn in self.log.turns)
        total_sonic_chars = sum(len(turn.sonic_response) for turn in self.log.turns)
        turns_with_tools = sum(1 for turn in self.log.turns if turn.tool_calls)

        # Session continuation stats
        session_events = self.log.session_events
        total_session_continuations = len(session_events)
        session_transitions = sum(1 for e in session_events if e.event_type == "transition")
        emergency_recoveries = sum(1 for e in session_events if e.event_type == "emergency_recovery")
        failed_continuations = sum(1 for e in session_events if not e.success)

        self.log.summary = {
            "total_turns": self.log.total_turns,
            "total_tool_calls": self.log.total_tool_calls,
            "total_errors": len(self.log.errors),
            "turns_with_tools": turns_with_tools,
            "avg_user_message_length": total_user_chars / self.log.total_turns if self.log.total_turns > 0 else 0,
            "avg_sonic_response_length": total_sonic_chars / self.log.total_turns if self.log.total_turns > 0 else 0,
            "tool_usage_rate": turns_with_tools / self.log.total_turns if self.log.total_turns > 0 else 0,
            "total_session_continuations": total_session_continuations,
            "session_transitions": session_transitions,
            "emergency_recoveries": emergency_recoveries,
            "failed_continuations": failed_continuations
        }

    def _save_log(self):
        """Save the log to a JSON file."""
        log_file = self._logs_dir / "interaction_log.json"

        with open(log_file, 'w') as f:
            json.dump(self.log.to_dict(), f, indent=2)

        # Also save a human-readable text log
        self._save_text_log()

    def _save_text_log(self):
        """Save a human-readable text version of the log."""
        text_file = self._chat_dir / "conversation.txt"

        # Index session events by the turn number they occurred after.
        # An event's timestamp falls between turn N and turn N+1; we place it
        # after the last turn whose timestamp <= the event's timestamp.
        session_events_by_turn: Dict[int, List[SessionContinuationEvent]] = {}
        for event in self.log.session_events:
            after_turn = 0
            for turn in self.log.turns:
                if turn.timestamp <= event.timestamp:
                    after_turn = turn.turn_number
                else:
                    break
            session_events_by_turn.setdefault(after_turn, []).append(event)

        with open(text_file, 'w') as f:
            f.write(f"{'='*60}\n")
            f.write(f"Interaction Log: {self.test_name}\n")
            f.write(f"Session ID: {self.session_id}\n")
            f.write(f"Started: {self.log.start_time}\n")
            f.write(f"Ended: {self.log.end_time}\n")
            f.write(f"{'='*60}\n\n")

            # Write any session events that occurred before the first turn
            for event in session_events_by_turn.get(0, []):
                self._write_session_event(f, event)

            for turn in self.log.turns:
                f.write(f"\n{'='*60}\n")
                f.write(f"Turn {turn.turn_number} - {turn.timestamp}\n")
                f.write(f"{'='*60}\n")
                f.write(f"User: {turn.user_message}\n")
                if turn.sonic_user_transcription:
                    f.write(f"User (transcribed): {turn.sonic_user_transcription}\n")
                f.write(f"\nSonic: {turn.sonic_response}\n")

                if turn.tool_calls:
                    f.write(f"\nTool Calls ({len(turn.tool_calls)}):\n")
                    for tool_call in turn.tool_calls:
                        f.write(f"  - {tool_call['tool_name']} (ID: {tool_call['tool_use_id']})\n")
                        f.write(f"    Input: {json.dumps(tool_call['tool_input'])}\n")
                        if tool_call['tool_result']:
                            f.write(f"    Result: {json.dumps(tool_call['tool_result'])}\n")

                if turn.audio_recorded:
                    f.write(f"\nAudio: {turn.audio_file}\n")

                # Write session events that occurred after this turn
                for event in session_events_by_turn.get(turn.turn_number, []):
                    self._write_session_event(f, event)

            # Session Continuations summary section
            if self.log.session_events:
                f.write(f"\n{'='*60}\n")
                f.write(f"Session Continuations ({len(self.log.session_events)})\n")
                f.write(f"{'='*60}\n")
                for event in self.log.session_events:
                    status = "SUCCESS" if event.success else f"FAILED: {event.error_message}"
                    f.write(f"  [{event.timestamp}] {event.event_type} (after turn {event.after_turn_number}): "
                            f"#{event.old_session_number} -> #{event.new_session_number} "
                            f"(duration: {event.old_session_duration_seconds:.1f}s, "
                            f"messages: {event.messages_replayed}, "
                            f"status: {status})\n")

            # Summary
            f.write(f"\n{'='*60}\n")
            f.write(f"Summary\n")
            f.write(f"{'='*60}\n")
            for key, value in self.log.summary.items():
                f.write(f"{key}: {value}\n")

            if self.log.errors:
                f.write(f"\n{'='*60}\n")
                f.write(f"Errors ({len(self.log.errors)})\n")
                f.write(f"{'='*60}\n")
                for error in self.log.errors:
                    f.write(f"{error['timestamp']}: {error['error']}\n")

    @staticmethod
    def _write_session_event(f, event: 'SessionContinuationEvent'):
        """Write a session continuation event to the text log."""
        status = "SUCCESS" if event.success else f"FAILED: {event.error_message}"
        f.write(f"\n{'- '*30}\n")
        f.write(f"SESSION CONTINUATION ({event.event_type})\n")
        f.write(f"  Timestamp: {event.timestamp}\n")
        f.write(f"  After turn: {event.after_turn_number}\n")
        f.write(f"  Session #{event.old_session_number} -> #{event.new_session_number}\n")
        f.write(f"  Old session duration: {event.old_session_duration_seconds:.1f}s\n")
        f.write(f"  Messages replayed: {event.messages_replayed}\n")
        f.write(f"  Bytes replayed: {event.total_bytes_replayed}\n")
        f.write(f"  Audio chunks sent: {event.audio_chunks_sent}\n")
        f.write(f"  Status: {status}\n")
        f.write(f"  Conversation history:\n")
        max_content_len = 200
        for msg in event.conversation_history:
            content = msg.get("content", "")
            truncated = content[:max_content_len] + "..." if len(content) > max_content_len else content
            f.write(f"    [{msg.get('role', '?')}]: {truncated}\n")
        f.write(f"{'- '*30}\n")

    def set_conversation_audio_file(self, path: str):
        """Set the path to the full conversation stereo WAV recording."""
        self.log.conversation_audio_file = path

    @property
    def evaluation_dir(self) -> Path:
        """Path to the evaluation subdirectory."""
        return self._evaluation_dir

    @property
    def logs_dir(self) -> Path:
        """Path to the logs subdirectory."""
        return self._logs_dir

    def save_config(self, config_dict: Dict[str, Any]):
        """Save test configuration to the config subdirectory."""
        config_file = self._config_dir / "test_config.json"
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)

    def get_log(self) -> InteractionLog:
        """Get the current log."""
        return self.log
