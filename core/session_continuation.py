"""
Session Continuation Manager for Nova Sonic
Wraps SonicStreamManager to provide transparent session continuation
when the AWS Bedrock ~8-minute session timeout approaches.
"""

import asyncio
import base64
import json
import time
from collections import deque
from typing import Optional, Callable, Dict, Any

from core.sonic_stream_manager import SonicStreamManager, CHUNK_SIZE, INPUT_SAMPLE_RATE
from core.conversation_history import ConversationHistory


class AudioBuffer:
    """
    Circular buffer holding silent audio chunks for warming new sessions.
    """

    def __init__(self, duration_seconds: float = 3.0):
        """
        Args:
            duration_seconds: How many seconds of audio to buffer
        """
        self.duration_seconds = duration_seconds
        # Chunks per second: sample_rate / chunk_size
        chunks_per_second = INPUT_SAMPLE_RATE / CHUNK_SIZE
        max_chunks = int(chunks_per_second * duration_seconds)
        self._buffer: deque = deque(maxlen=max_chunks)

    def add_chunk(self, audio_bytes: bytes):
        """Add an audio chunk to the buffer."""
        self._buffer.append(audio_bytes)

    def get_all_chunks(self) -> list:
        """Get all buffered chunks."""
        return list(self._buffer)

    def clear(self):
        """Clear the buffer."""
        self._buffer.clear()


class SessionContinuationManager:
    """
    Wraps SonicStreamManager to provide transparent session continuation.

    Same external API as SonicStreamManager. When the session duration
    exceeds the transition threshold, the next call to mark_turn_complete()
    will seamlessly create a new session and replay conversation history.
    """

    def __init__(
        self,
        model_id: str = 'amazon.nova-2-sonic-v1:0',
        region: str = 'us-east-1',
        endpoint_uri: Optional[str] = None,
        system_prompt: Optional[str] = None,
        voice_id: str = 'matthew',
        inference_config: Optional[Dict[str, Any]] = None,
        tool_config: Optional[Dict[str, Any]] = None,
        tool_handler: Optional[Callable] = None,
        event_callback: Optional[Callable] = None,
        transition_threshold_seconds: float = 360.0,
        audio_buffer_duration_seconds: float = 3.0,
        enable_continuation: bool = True,
        continuation_callback: Optional[Callable] = None
    ):
        # Store construction args for creating new sessions
        self._model_id = model_id
        self._region = region
        self._endpoint_uri = endpoint_uri
        self._system_prompt = system_prompt
        self._voice_id = voice_id
        self._inference_config = inference_config
        self._tool_config = tool_config
        self._tool_handler = tool_handler
        self._event_callback = event_callback

        # Continuation settings
        self._transition_threshold = transition_threshold_seconds
        self._enable_continuation = enable_continuation
        self._continuation_callback = continuation_callback

        # State
        self._current_session: Optional[SonicStreamManager] = None
        self._session_start_time: Optional[float] = None
        self._session_number = 1
        self._history = ConversationHistory()
        self._audio_buffer = AudioBuffer(duration_seconds=audio_buffer_duration_seconds)
        self._input_audio_callback: Optional[Callable] = None
        self._event_logger = None

    def _create_session(self) -> SonicStreamManager:
        """Create a new SonicStreamManager with the stored config."""
        session = SonicStreamManager(
            model_id=self._model_id,
            region=self._region,
            endpoint_uri=self._endpoint_uri,
            system_prompt=self._system_prompt,
            voice_id=self._voice_id,
            inference_config=self._inference_config,
            tool_config=self._tool_config,
            tool_handler=self._tool_handler,
            event_callback=self._event_callback
        )
        # Propagate input audio callback and event logger to the new session
        if self._input_audio_callback:
            session.input_audio_callback = self._input_audio_callback
        if self._event_logger:
            session.event_logger = self._event_logger
        return session

    async def initialize_stream(self):
        """Create and initialize the first session."""
        self._current_session = self._create_session()
        await self._current_session.initialize_stream()
        self._session_start_time = time.time()
        self._session_number = 1
        print(f"[SessionContinuation] Session #{self._session_number} initialized")

    async def send_text_message(self, text: str):
        """
        Send a text message to Nova Sonic. Records in history.
        Falls back to emergency recovery if current session is dead.

        Args:
            text: The text message to send
        """
        # Record user message in history
        self._history.add_message("USER", text)

        # Check if current session is dead — emergency recover
        if self._current_session and not self._current_session.is_active:
            print("[SessionContinuation] Current session is dead, attempting emergency recovery...")
            await self._emergency_recover()

        if self._current_session:
            await self._current_session.send_text_message(text)

    async def mark_turn_complete(self, assistant_response: str):
        """
        Called by main.py after each turn completes (Sonic has finished responding).

        Records assistant response in history. If session duration exceeds
        the threshold, performs a full synchronous transition before returning.

        Args:
            assistant_response: The full assistant response text from this turn
        """
        # Record assistant response in history
        self._history.add_message("ASSISTANT", assistant_response)

        if not self._enable_continuation:
            return

        # Check if session duration exceeds threshold
        elapsed = time.time() - self._session_start_time
        if elapsed > self._transition_threshold:
            print(f"\n[SessionContinuation] Session #{self._session_number} "
                  f"elapsed {elapsed:.1f}s > threshold {self._transition_threshold}s")
            print(f"[SessionContinuation] Starting transition...")
            await self._perform_transition()

    def _notify_continuation(
        self,
        event_type: str,
        old_session_number: int,
        new_session_number: int,
        old_session_duration_seconds: float,
        audio_chunks_sent: int = 0,
        success: bool = True,
        error_message: Optional[str] = None
    ):
        """Notify the continuation callback with event data. Errors are caught
        so a callback failure never breaks the transition."""
        if not self._continuation_callback:
            return
        try:
            messages = self._history.get_messages()
            conversation_history = [
                {"role": role, "content": content} for role, content in messages
            ]
            self._continuation_callback(
                event_type=event_type,
                old_session_number=old_session_number,
                new_session_number=new_session_number,
                old_session_duration_seconds=old_session_duration_seconds,
                messages_replayed=len(messages),
                total_bytes_replayed=self._history.total_bytes,
                conversation_history=conversation_history,
                audio_chunks_sent=audio_chunks_sent,
                success=success,
                error_message=error_message
            )
        except Exception as cb_err:
            print(f"[SessionContinuation] Warning: continuation callback error: {cb_err}")

    async def _perform_transition(self):
        """
        Deterministic session transition sequence:
        1. Close old session
        2. Create new SonicStreamManager
        3. Initialize it
        4. Replay conversation history
        5. Send silent audio to warm the stream
        6. Promote as current session
        """
        old_session = self._current_session
        old_number = self._session_number
        old_duration = time.time() - self._session_start_time if self._session_start_time else 0.0

        # 1. Close old session gracefully
        print(f"[SessionContinuation] Closing session #{old_number}...")
        try:
            if old_session:
                await old_session.close()
        except Exception as e:
            print(f"[SessionContinuation] Warning: error closing old session: {e}")

        # 2. Create new session
        new_session = self._create_session()

        # 3. Initialize the new session
        print(f"[SessionContinuation] Initializing new session...")
        await new_session.initialize_stream()

        # 4. Replay conversation history
        history_events = self._history.get_history_events(new_session.prompt_name)
        print(f"[SessionContinuation] Replaying {self._history.message_count} messages "
              f"({len(history_events)} events, {self._history.total_bytes} bytes)...")
        for event_json in history_events:
            await new_session._send_raw_event(event_json, log_content=False)
            await asyncio.sleep(0.05)

        # 5. Send buffered silent audio to warm the new session's audio stream
        silent_chunks = self._audio_buffer.get_all_chunks()
        if silent_chunks:
            print(f"[SessionContinuation] Sending {len(silent_chunks)} silent audio chunks to warm stream...")
            for chunk in silent_chunks:
                audio_blob = base64.b64encode(chunk).decode('utf-8')
                audio_event = {
                    "event": {
                        "audioInput": {
                            "promptName": new_session.prompt_name,
                            "contentName": new_session.audio_content_name,
                            "content": audio_blob
                        }
                    }
                }
                await new_session._send_raw_event(json.dumps(audio_event), log_content=False)
            # Brief pause to let audio settle
            await asyncio.sleep(0.2)

        # 6. Promote new session
        self._current_session = new_session
        self._session_start_time = time.time()
        self._session_number += 1
        self._audio_buffer.clear()

        print(f"[SessionContinuation] Session transition complete. "
              f"Now on session #{self._session_number} "
              f"(history: {self._history.message_count} messages, "
              f"{self._history.total_bytes} bytes)")

        self._notify_continuation(
            event_type="transition",
            old_session_number=old_number,
            new_session_number=self._session_number,
            old_session_duration_seconds=old_duration,
            audio_chunks_sent=len(silent_chunks) if silent_chunks else 0
        )

    async def _emergency_recover(self):
        """
        Emergency recovery when current session is unexpectedly dead.
        Creates a new session and replays history.
        """
        old_number = self._session_number
        old_duration = time.time() - self._session_start_time if self._session_start_time else 0.0

        print(f"[SessionContinuation] Emergency recovery: creating new session...")

        try:
            # Try to close the dead session
            if self._current_session:
                try:
                    await self._current_session.close()
                except Exception:
                    pass

            # Create and initialize new session
            new_session = self._create_session()
            await new_session.initialize_stream()

            # Replay history
            history_events = self._history.get_history_events(new_session.prompt_name)
            print(f"[SessionContinuation] Emergency replay: {self._history.message_count} messages...")
            for event_json in history_events:
                await new_session._send_raw_event(event_json, log_content=False)
                await asyncio.sleep(0.05)

            # Promote
            self._current_session = new_session
            self._session_start_time = time.time()
            self._session_number += 1

            print(f"[SessionContinuation] Emergency recovery complete. "
                  f"Now on session #{self._session_number}")

            self._notify_continuation(
                event_type="emergency_recovery",
                old_session_number=old_number,
                new_session_number=self._session_number,
                old_session_duration_seconds=old_duration
            )

        except Exception as e:
            print(f"[SessionContinuation] Emergency recovery FAILED: {e}")
            import traceback
            traceback.print_exc()

            self._notify_continuation(
                event_type="emergency_recovery",
                old_session_number=old_number,
                new_session_number=old_number,
                old_session_duration_seconds=old_duration,
                success=False,
                error_message=str(e)
            )

    async def close(self):
        """Close the current session."""
        if self._current_session:
            await self._current_session.close()
            self._current_session = None

    @property
    def is_active(self) -> bool:
        """Whether the current session is active."""
        if self._current_session:
            return self._current_session.is_active
        return False

    @property
    def audio_output_queue(self):
        """The audio output queue from the current session."""
        if self._current_session:
            return self._current_session.audio_output_queue
        return asyncio.Queue()

    @property
    def prompt_name(self):
        """The prompt name from the current session."""
        if self._current_session:
            return self._current_session.prompt_name
        return None

    @property
    def sonic_session_id(self):
        """The Nova Sonic session ID from the current session."""
        if self._current_session:
            return self._current_session.sonic_session_id
        return None

    @property
    def input_audio_callback(self):
        """The input audio callback on the current session."""
        if self._current_session:
            return self._current_session.input_audio_callback
        return None

    @input_audio_callback.setter
    def input_audio_callback(self, value):
        """Set the input audio callback, storing it for session transitions."""
        self._input_audio_callback = value
        if self._current_session:
            self._current_session.input_audio_callback = value

    @property
    def event_logger(self):
        """The event logger on the current session."""
        if self._current_session:
            return self._current_session.event_logger
        return None

    @event_logger.setter
    def event_logger(self, value):
        """Set the event logger, storing it for session transitions."""
        self._event_logger = value
        if self._current_session:
            self._current_session.event_logger = value

    async def send_audio_input(self, pcm_bytes: bytes, text_for_history: str = None):
        """
        Stream real audio to Nova Sonic. Optionally records the original
        user text in conversation history so session replay uses text events.

        Args:
            pcm_bytes: Raw 16 kHz 16-bit mono PCM audio.
            text_for_history: If provided, added to history for replay on session transition.
        """
        if text_for_history:
            self._history.add_message("USER", text_for_history)

        # Check if current session is dead — emergency recover
        if self._current_session and not self._current_session.is_active:
            print("[SessionContinuation] Current session is dead, attempting emergency recovery...")
            await self._emergency_recover()

        if self._current_session:
            await self._current_session.send_audio_input(pcm_bytes)
