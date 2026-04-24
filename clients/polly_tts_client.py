"""
Polly TTS Client for Nova Sonic Test Harness
Synthesizes text to raw PCM audio using Amazon Polly.
"""

import boto3
from typing import Optional


class PollyTTSClient:
    """
    Wrapper around Amazon Polly synthesize_speech.
    Returns raw 16kHz 16-bit mono PCM bytes (no WAV header).
    """

    def __init__(
        self,
        region: str = "us-east-1",
        voice_id: str = "Matthew",
        engine: str = "neural",
        sample_rate: int = 16000,
    ):
        self.voice_id = voice_id
        self.engine = engine
        self.sample_rate = sample_rate
        self._client = boto3.client("polly", region_name=region)

    def synthesize(self, text: str) -> bytes:
        """
        Synthesize text to raw PCM audio.

        Args:
            text: The text to synthesize.

        Returns:
            Raw 16-bit mono PCM bytes at self.sample_rate Hz.
        """
        response = self._client.synthesize_speech(
            Text=text,
            OutputFormat="pcm",
            VoiceId=self.voice_id,
            Engine=self.engine,
            SampleRate=str(self.sample_rate),
        )
        return response["AudioStream"].read()
