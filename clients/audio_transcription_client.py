"""
Amazon Transcribe wrapper for transcribing Nova Sonic audio output.

Uploads WAV files to S3, runs transcription jobs, fetches results, and cleans up.
"""

import boto3
import json
import time
import uuid
import wave
from pathlib import Path
from botocore.exceptions import ClientError


def read_wav_sample_rate(audio_path: Path) -> int:
    """Read the sample rate from a WAV file header."""
    with wave.open(str(audio_path), 'rb') as wf:
        return wf.getframerate()


class TranscriptionClient:
    """Transcribes audio files using Amazon Transcribe via S3."""

    def __init__(self, s3_bucket: str, region: str = "us-east-1", s3_prefix: str = "nova-sonic-eval-temp/"):
        self.s3_bucket = s3_bucket
        self.region = region
        self.s3_prefix = s3_prefix
        self.s3_client = boto3.client("s3", region_name=region)
        self.transcribe_client = boto3.client("transcribe", region_name=region)

    def transcribe(self, audio_path: Path, sample_rate: int, session_id: str = None) -> str:
        """
        Transcribe a WAV audio file to text.

        Args:
            audio_path: Path to the WAV file
            sample_rate: Audio sample rate in Hz
            session_id: Optional session ID for S3 key prefix (falls back to UUID)

        Returns:
            Transcribed text string

        Raises:
            RuntimeError: If transcription fails
        """
        self._ensure_bucket_exists()

        prefix = session_id if session_id else uuid.uuid4().hex
        s3_key = f"{self.s3_prefix}{prefix}/{audio_path.name}"
        s3_uri = f"s3://{self.s3_bucket}/{s3_key}"
        job_name = f"nova-sonic-eval-{uuid.uuid4().hex}"
        result_s3_key = f"{self.s3_prefix}{job_name}.json"

        try:
            self._upload_to_s3(audio_path, s3_key)
            self._start_transcription_job(job_name, s3_uri, sample_rate)
            self._poll_transcription_job(job_name)
            transcript = self._fetch_transcript(result_s3_key)
            return transcript
        finally:
            # Cleanup: delete uploaded audio and result JSON from S3
            for key in [s3_key, result_s3_key]:
                try:
                    self.s3_client.delete_object(Bucket=self.s3_bucket, Key=key)
                except Exception:
                    pass

    def _ensure_bucket_exists(self):
        """Create the S3 bucket if it doesn't exist."""
        try:
            self.s3_client.head_bucket(Bucket=self.s3_bucket)
        except ClientError as e:
            error_code = int(e.response["Error"]["Code"])
            if error_code == 404:
                try:
                    if self.region == "us-east-1":
                        # us-east-1 does not accept LocationConstraint
                        self.s3_client.create_bucket(Bucket=self.s3_bucket)
                    else:
                        self.s3_client.create_bucket(
                            Bucket=self.s3_bucket,
                            CreateBucketConfiguration={"LocationConstraint": self.region}
                        )
                    print(f"Created S3 bucket: {self.s3_bucket}")
                except ClientError as create_err:
                    raise RuntimeError(f"Failed to create S3 bucket {self.s3_bucket}: {create_err}") from create_err
            else:
                raise RuntimeError(f"Failed to access S3 bucket {self.s3_bucket}: {e}") from e

    def _upload_to_s3(self, audio_path: Path, s3_key: str):
        """Upload audio file to S3."""
        self.s3_client.upload_file(str(audio_path), self.s3_bucket, s3_key)

    def _start_transcription_job(self, job_name: str, s3_uri: str, sample_rate: int):
        """Start an Amazon Transcribe job."""
        self.transcribe_client.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={"MediaFileUri": s3_uri},
            MediaFormat="wav",
            MediaSampleRateHertz=sample_rate,
            LanguageCode="en-US",
            OutputBucketName=self.s3_bucket,
            OutputKey=f"{self.s3_prefix}{job_name}.json"
        )

    def _poll_transcription_job(self, job_name: str, timeout: int = 300, interval: int = 5):
        """Poll until transcription job completes or fails."""
        elapsed = 0
        while elapsed < timeout:
            response = self.transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
            status = response["TranscriptionJob"]["TranscriptionJobStatus"]

            if status == "COMPLETED":
                return
            elif status == "FAILED":
                reason = response["TranscriptionJob"].get("FailureReason", "Unknown")
                raise RuntimeError(f"Transcription job {job_name} failed: {reason}")

            time.sleep(interval)
            elapsed += interval

        raise RuntimeError(f"Transcription job {job_name} timed out after {timeout}s")

    def _fetch_transcript(self, result_s3_key: str) -> str:
        """Fetch and parse the transcript from the S3 result JSON."""
        response = self.s3_client.get_object(Bucket=self.s3_bucket, Key=result_s3_key)
        result_data = json.loads(response["Body"].read().decode("utf-8"))
        return result_data["results"]["transcripts"][0]["transcript"]
