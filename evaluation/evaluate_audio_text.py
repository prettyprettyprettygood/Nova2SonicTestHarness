"""
Audio-Text Consistency Evaluation (Hallucination Detector)

Compares Nova Sonic's text output against its audio output (via Amazon Transcribe)
to detect hallucinations — cases where the text and audio diverge.

Usage:
    python main.py evaluate-audio --log results/sessions/<session_id>/ --s3-bucket my-bucket

Optional arguments:
    --region us-east-1
    --judge-model claude-opus
"""

import boto3
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from clients.audio_transcription_client import TranscriptionClient, read_wav_sample_rate
from evaluation.evaluation_types import (
    AudioTextConsistencyReport,
    FactualDiscrepancy,
    TurnConsistencyResult,
)
from utils.model_registry import get_model_registry


class AudioTextConsistencyJudge:
    """Compares text output vs. audio transcription using an LLM judge."""

    VALID_VERDICTS = {"CONSISTENT", "MINOR_DIFFERENCES", "HALLUCINATION"}

    def __init__(self, judge_model: str = "claude-opus", region: str = None):
        registry = get_model_registry()
        resolved = registry.resolve_judge_model(judge_model)
        self.judge_model = resolved["model_id"]
        resolved_region = region or resolved["region"]
        self.client = boto3.client("bedrock-runtime", region_name=resolved_region)

    def evaluate_turn(self, turn_number: int, text_output: str, audio_transcript: str) -> TurnConsistencyResult:
        """
        Compare text output against audio transcription for a single turn.

        Returns:
            TurnConsistencyResult with verdict and categorized differences
        """
        prompt = self._build_prompt(turn_number, text_output, audio_transcript)
        response_text = self._call_judge(prompt)
        return self._parse_response(turn_number, text_output, audio_transcript, response_text)

    def _build_prompt(self, turn_number: int, text_output: str, audio_transcript: str) -> str:
        return f"""You are an expert evaluator comparing a speech AI's text output against a transcription of its audio output. These should be semantically identical — any factual divergence is a hallucination.

## Turn {turn_number}

### Text Output (what the model produced as text)
{text_output}

### Audio Transcription (what the model actually said, transcribed from audio)
{audio_transcript}

## Your Task

Compare the text output and audio transcription. Classify every difference into exactly one of three categories:

1. **Factual discrepancies**: Different numbers, names, facts, dates, or claims. Missing or added substantive information. Contradictions between text and audio. These indicate a hallucination.

2. **Phrasing differences**: Synonym substitutions, reworded sentences that preserve the same meaning, different sentence structure with identical content. These are minor and acceptable.

3. **Filler words**: Words like "um", "uh", "you know", "like", "so", hesitation markers, or verbal tics present in the audio transcription but not in text. These are expected in speech and should be ignored.

## Important: Ignore These
- Punctuation, capitalization, and formatting differences
- Likely transcription errors (homophones like "there"/"their", "to"/"too"/"two")
- Minor word order changes that don't affect meaning

## Verdict Rules
- If ANY factual discrepancy exists → verdict is HALLUCINATION
- If only phrasing differences exist (no factual discrepancies) → verdict is MINOR_DIFFERENCES
- If only filler words or no differences at all → verdict is CONSISTENT

## Output Format
Return ONLY a JSON object with this exact structure:

```json
{{
  "verdict": "CONSISTENT",
  "factual_discrepancies": [
    {{"text_says": "exact quote from text", "audio_says": "exact quote from audio", "description": "what is different and why it matters"}}
  ],
  "phrasing_differences": [
    "Text says 'utilize' while audio says 'use'"
  ],
  "filler_words": ["um", "you know"]
}}
```

Return ONLY the JSON object, no additional text."""

    def _call_judge(self, prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [{"text": prompt}]
            }
        ]

        response = self.client.converse(
            modelId=self.judge_model,
            messages=messages,
            inferenceConfig={
                "maxTokens": 4096,
                "temperature": 0.1
            }
        )

        response_text = ""
        if "output" in response and "message" in response["output"]:
            for content in response["output"]["message"]["content"]:
                if "text" in content:
                    response_text += content["text"]

        return response_text

    def _parse_response(
        self,
        turn_number: int,
        text_output: str,
        audio_transcript: str,
        response_text: str
    ) -> TurnConsistencyResult:
        try:
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1

            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")

            data = json.loads(response_text[json_start:json_end])

            verdict = data.get("verdict", "HALLUCINATION")
            if verdict not in self.VALID_VERDICTS:
                verdict = "HALLUCINATION"

            factual_discrepancies = [
                FactualDiscrepancy(
                    text_says=d.get("text_says", ""),
                    audio_says=d.get("audio_says", ""),
                    description=d.get("description", "")
                )
                for d in data.get("factual_discrepancies", [])
            ]

            phrasing_differences = data.get("phrasing_differences", [])
            filler_words = data.get("filler_words", [])

            return TurnConsistencyResult(
                turn_number=turn_number,
                verdict=verdict,
                sonic_response_text=text_output,
                audio_transcript=audio_transcript,
                factual_discrepancies=factual_discrepancies,
                phrasing_differences=phrasing_differences,
                filler_words=filler_words,
            )

        except Exception as e:
            print(f"  Error parsing judge response for turn {turn_number}: {e}")
            return TurnConsistencyResult(
                turn_number=turn_number,
                verdict="HALLUCINATION",
                sonic_response_text=text_output,
                audio_transcript=audio_transcript,
                error=f"Judge parse failure: {str(e)}"
            )


def run_evaluation(log_dir: Path, s3_bucket: str, region: str, judge_model: str):
    """Orchestrate audio-text consistency evaluation for a session."""

    # Find interaction_log.json (new layout: logs/ subdir; legacy: root)
    log_path = log_dir / "logs" / "interaction_log.json"
    if not log_path.exists():
        log_path = log_dir / "interaction_log.json"
    if not log_path.exists():
        print(f"Error: interaction_log.json not found in {log_dir}")
        sys.exit(1)

    with open(log_path, 'r') as f:
        conversation_log = json.load(f)

    session_id = conversation_log.get("session_id", log_dir.name)
    turns = conversation_log.get("turns", [])

    print(f"\n{'='*60}")
    print(f"Audio-Text Consistency Evaluation")
    print(f"{'='*60}")
    print(f"Session: {session_id}")
    print(f"Total turns: {len(turns)}")
    print(f"Judge model: {judge_model}")
    print(f"S3 bucket: {s3_bucket}")
    print(f"{'='*60}\n")

    # Initialize clients
    try:
        transcription_client = TranscriptionClient(s3_bucket=s3_bucket, region=region)
    except Exception as e:
        print(f"Error: Failed to initialize transcription client: {e}")
        sys.exit(1)

    judge = AudioTextConsistencyJudge(judge_model=judge_model, region=region)

    turn_results = []

    for turn in turns:
        turn_number = turn.get("turn_number", 0)
        audio_recorded = turn.get("audio_recorded", False)
        audio_file = turn.get("audio_file")
        sonic_response = turn.get("sonic_response", "")
        turn_sonic_session_id = turn.get("sonic_session_id")

        print(f"Turn {turn_number}: ", end="", flush=True)

        # Skip turns without audio
        if not audio_recorded or not audio_file:
            reason = "no audio recorded" if not audio_recorded else "audio_file is null"
            print(f"skipped ({reason})")
            turn_results.append(TurnConsistencyResult(
                turn_number=turn_number,
                verdict="CONSISTENT",
                sonic_response_text=sonic_response,
                audio_transcript="",
                skipped=True,
                skip_reason=reason,
            ))
            continue

        # Resolve WAV path
        wav_path = log_dir / audio_file
        if not wav_path.exists():
            print(f"skipped (WAV file not found: {audio_file})")
            turn_results.append(TurnConsistencyResult(
                turn_number=turn_number,
                verdict="CONSISTENT",
                sonic_response_text=sonic_response,
                audio_transcript="",
                skipped=True,
                skip_reason=f"WAV file not found: {audio_file}",
            ))
            continue

        # Check file is non-empty
        if wav_path.stat().st_size == 0:
            print(f"skipped (empty audio file)")
            turn_results.append(TurnConsistencyResult(
                turn_number=turn_number,
                verdict="CONSISTENT",
                sonic_response_text=sonic_response,
                audio_transcript="",
                skipped=True,
                skip_reason="empty audio file",
            ))
            continue

        # Transcribe
        try:
            sample_rate = read_wav_sample_rate(wav_path)
            print(f"transcribing ({sample_rate}Hz)... ", end="", flush=True)
            # Use per-turn Nova Sonic session ID for S3 organization, fall back to log session ID
            s3_session_id = turn_sonic_session_id or session_id
            transcript = transcription_client.transcribe(wav_path, sample_rate, session_id=s3_session_id)
            print(f"done. ", end="", flush=True)
        except Exception as e:
            print(f"transcription error: {e}")
            turn_results.append(TurnConsistencyResult(
                turn_number=turn_number,
                verdict="CONSISTENT",
                sonic_response_text=sonic_response,
                audio_transcript="",
                error=f"Transcription failed: {str(e)}",
            ))
            continue

        # Compare via judge
        try:
            print(f"judging... ", end="", flush=True)
            result = judge.evaluate_turn(turn_number, sonic_response, transcript)
            print(f"{result.verdict}")
            turn_results.append(result)
        except Exception as e:
            print(f"judge error: {e}")
            turn_results.append(TurnConsistencyResult(
                turn_number=turn_number,
                verdict="HALLUCINATION",
                sonic_response_text=sonic_response,
                audio_transcript=transcript,
                error=f"Judge evaluation failed: {str(e)}",
            ))

    # Aggregate results
    turns_evaluated = sum(1 for r in turn_results if not r.skipped and not r.error)
    turns_skipped = sum(1 for r in turn_results if r.skipped)
    turns_with_errors = sum(1 for r in turn_results if r.error and not r.skipped)
    turns_consistent = sum(1 for r in turn_results if r.verdict == "CONSISTENT" and not r.skipped)
    turns_minor = sum(1 for r in turn_results if r.verdict == "MINOR_DIFFERENCES")
    turns_hallucinated = sum(1 for r in turn_results if r.verdict == "HALLUCINATION")
    hallucination_rate = turns_hallucinated / turns_evaluated if turns_evaluated > 0 else 0.0

    summary_parts = []
    summary_parts.append(f"{turns_evaluated}/{len(turns)} turns evaluated")
    if turns_skipped:
        summary_parts.append(f"{turns_skipped} skipped")
    if turns_with_errors:
        summary_parts.append(f"{turns_with_errors} errors")
    summary_parts.append(f"{turns_consistent} consistent")
    if turns_minor:
        summary_parts.append(f"{turns_minor} minor differences")
    if turns_hallucinated:
        summary_parts.append(f"{turns_hallucinated} hallucinations")
    summary_parts.append(f"hallucination rate: {hallucination_rate:.1%}")

    report = AudioTextConsistencyReport(
        session_id=session_id,
        evaluated_at=datetime.now().isoformat(),
        judge_model=judge_model,
        total_turns=len(turns),
        turns_evaluated=turns_evaluated,
        turns_skipped=turns_skipped,
        turns_with_errors=turns_with_errors,
        turns_consistent=turns_consistent,
        turns_minor_differences=turns_minor,
        turns_hallucinated=turns_hallucinated,
        hallucination_rate=hallucination_rate,
        turn_results=turn_results,
        summary="; ".join(summary_parts),
    )

    # Save report (new layout: evaluation/ subdir; legacy: root)
    eval_dir = log_dir / "evaluation"
    if eval_dir.exists() and eval_dir.is_dir():
        report_path = eval_dir / "audio_text_consistency.json"
    else:
        report_path = log_dir / "audio_text_consistency.json"
    with open(report_path, 'w') as f:
        json.dump(asdict(report), f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Results Summary")
    print(f"{'='*60}")
    print(f"Turns evaluated: {turns_evaluated}/{len(turns)}")
    print(f"Turns skipped:   {turns_skipped}")
    print(f"Turns with errors: {turns_with_errors}")
    print(f"Consistent:      {turns_consistent}")
    print(f"Minor differences: {turns_minor}")
    print(f"Hallucinations:  {turns_hallucinated}")
    print(f"Hallucination rate: {hallucination_rate:.1%}")
    print(f"\nReport saved to: {report_path}")

    if not turns_evaluated:
        print("\nWarning: No turns with audio were evaluated.")


def main():
    if len(sys.argv) < 2:
        print("Usage: python evaluate_audio_text.py --log <log_dir> --s3-bucket <bucket>")
        print("\nRequired arguments:")
        print("  --log <dir>         Path to session log directory")
        print("  --s3-bucket <name>  S3 bucket for temp audio uploads")
        print("\nOptional arguments:")
        print("  --region <region>   AWS region (default: us-east-1)")
        print("  --judge-model <model>  Judge model alias (default: claude-opus)")
        sys.exit(1)

    log_dir = None
    s3_bucket = None
    region = "us-east-1"
    judge_model = "claude-opus"

    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--log" and i + 1 < len(sys.argv):
            log_dir = Path(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--s3-bucket" and i + 1 < len(sys.argv):
            s3_bucket = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--region" and i + 1 < len(sys.argv):
            region = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--judge-model" and i + 1 < len(sys.argv):
            judge_model = sys.argv[i + 1]
            i += 2
        else:
            print(f"Unknown argument: {sys.argv[i]}")
            i += 1

    if not log_dir:
        print("Error: --log is required")
        sys.exit(1)

    if not s3_bucket:
        print("Error: --s3-bucket is required")
        sys.exit(1)

    if not log_dir.exists():
        print(f"Error: Directory not found: {log_dir}")
        sys.exit(1)

    run_evaluation(log_dir, s3_bucket, region, judge_model)


if __name__ == "__main__":
    main()
