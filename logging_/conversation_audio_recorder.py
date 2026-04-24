"""
Conversation Audio Recorder for Nova Sonic Test Harness

Records input and output audio as separate raw LPCM files.
- Input: bytes appended continuously (silence + Polly audio), no timer needed.
- Output: silence-padded based on elapsed time from stream start so the
  output track stays time-aligned with the input track.

Both files are 16 kHz, 16-bit, mono, little-endian PCM.
"""

import time
from typing import Optional, Tuple

SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2  # bytes per sample (16-bit)
BYTES_PER_SECOND = SAMPLE_RATE * SAMPLE_WIDTH


class ConversationAudioRecorder:
    """
    Collects input and output PCM audio into two separate raw LPCM buffers.
    """

    def __init__(self):
        self._start_time: Optional[float] = None
        self._input_buf = bytearray()
        self._output_buf = bytearray()

    def start(self):
        """Mark the moment streaming begins (silence starts flowing)."""
        self._start_time = time.time()

    def add_input_chunk(self, pcm_bytes: bytes):
        """Append input audio bytes (silence or Polly audio). No timer needed."""
        if self._start_time is None:
            self._start_time = time.time()
        self._input_buf.extend(pcm_bytes)

    def add_output_chunk(self, pcm_bytes: bytes):
        """
        Append output audio, filling any silence gap first.

        Uses the input buffer length as the time reference — it's a continuous
        real-time stream (silence + Polly), so its byte count is the true
        timeline. This avoids inflated silence from wall-clock drift during
        setup, API calls, or Polly synthesis.
        """
        # Pad output to match the input timeline
        expected_bytes = len(self._input_buf)
        expected_bytes -= expected_bytes % SAMPLE_WIDTH  # align to sample boundary

        if expected_bytes > len(self._output_buf):
            self._output_buf.extend(b'\x00' * (expected_bytes - len(self._output_buf)))

        # Append actual audio
        self._output_buf.extend(pcm_bytes)

    def save(self, output_dir: str) -> Optional[Tuple[str, str, str]]:
        """
        Write input.lpcm, output.lpcm, and conversation.wav (stereo)
        into *output_dir*.

        Returns (input_path, output_path, wav_path) on success, None if no audio.
        """
        import array
        import os
        import wave

        if not self._input_buf and not self._output_buf:
            return None

        os.makedirs(output_dir, exist_ok=True)

        # Save raw LPCM files
        input_path = os.path.join(output_dir, 'input.lpcm')
        output_path = os.path.join(output_dir, 'output.lpcm')

        with open(input_path, 'wb') as f:
            f.write(self._input_buf)
        with open(output_path, 'wb') as f:
            f.write(self._output_buf)

        # Merge into stereo WAV (L=input, R=output)
        left = bytes(self._input_buf)
        right = bytes(self._output_buf)

        # Truncate to sample boundary (even byte count for 16-bit)
        left = left[:len(left) - len(left) % SAMPLE_WIDTH]
        right = right[:len(right) - len(right) % SAMPLE_WIDTH]

        # Pad shorter to match longer
        max_len = max(len(left), len(right))
        left += b'\x00' * (max_len - len(left))
        right += b'\x00' * (max_len - len(right))

        left_arr = array.array('h')
        left_arr.frombytes(left)
        right_arr = array.array('h')
        right_arr.frombytes(right)

        stereo = array.array('h', [0]) * (len(left_arr) * 2)
        stereo[0::2] = left_arr
        stereo[1::2] = right_arr

        wav_path = os.path.join(output_dir, 'conversation.wav')
        with wave.open(wav_path, 'wb') as wf:
            wf.setnchannels(2)
            wf.setsampwidth(SAMPLE_WIDTH)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(stereo.tobytes())

        return input_path, output_path, wav_path
