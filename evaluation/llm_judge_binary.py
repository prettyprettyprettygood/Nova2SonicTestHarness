"""
LLM-as-Judge Evaluation Module with Binary (YES/NO) Metrics
Evaluates conversation quality using strict binary verdicts per metric.
"""

import boto3
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

from utils.model_registry import get_model_registry
from evaluation.evaluation_types import (
    BinaryEvaluationCriteria,
    BinaryEvaluationResult,
    BinaryMetricVerdict,
    BinaryRubricVerdict,
)


class BinaryMetricDefinition:
    """Defines a binary evaluation metric."""

    # Metric tiers control how they affect the overall pass/fail verdict
    TIER_CRITICAL = "critical"    # Must pass for overall PASS
    TIER_IMPORTANT = "important"  # Contributes to pass rate but doesn't auto-fail
    TIER_ADVISORY = "advisory"    # Reported only, doesn't affect pass/fail or pass rate

    def __init__(
        self,
        name: str,
        description: str,
        default_rubric: str | list[str],
        requires_tools: bool = False,
        tier: str = "critical",
    ):
        self.name = name
        self.description = description
        self.default_rubric = default_rubric if isinstance(default_rubric, list) else [default_rubric]
        self.requires_tools = requires_tools
        self.tier = tier


BUILTIN_METRICS: Dict[str, BinaryMetricDefinition] = {
    # --- Tier 1: Critical (must pass for overall PASS) ---
    "Goal Achievement": BinaryMetricDefinition(
        name="Goal Achievement",
        description="Did the assistant accomplish the user's stated goal?",
        default_rubric=[
            "Did the assistant correctly identify and address the user's primary goal?",
            "Did the assistant provide the specific information or outcome the user requested?",
            "Was the goal resolved by the end of the conversation (not left open or deferred)?",
        ],
        tier=BinaryMetricDefinition.TIER_CRITICAL,
    ),
    "Response Accuracy": BinaryMetricDefinition(
        name="Response Accuracy",
        description="Were the assistant's responses factually correct and relevant?",
        default_rubric=[
            "Were all facts, numbers, and claims in the responses accurate (no hallucinated information)?",
            "Were responses relevant to the user's question without unnecessary tangents?",
            "Did the assistant correctly interpret and relay tool results without distortion?",
        ],
        tier=BinaryMetricDefinition.TIER_CRITICAL,
    ),
    # --- Tier 2: Important (contributes to score, doesn't auto-fail) ---
    "Tool Usage": BinaryMetricDefinition(
        name="Tool Usage",
        description="Did the assistant use the correct tools with appropriate parameters?",
        default_rubric=[
            "Did the assistant call the right tool(s) for the user's request?",
            "Were the tool parameters accurate and correctly formatted (matching user-provided values)?",
            "Did the assistant correctly interpret and relay the tool results to the user?",
            "Did the assistant avoid unnecessary or redundant tool calls?",
        ],
        requires_tools=True,
        tier=BinaryMetricDefinition.TIER_IMPORTANT,
    ),
    "Conversation Flow": BinaryMetricDefinition(
        name="Conversation Flow",
        description="Did the conversation read like a natural exchange between two people?",
        default_rubric=[
            "Did the assistant's responses follow logically from what the user said (no non-sequiturs or ignored questions)?",
            "Did the conversation flow naturally without robotic or formulaic patterns (e.g., repeating the same greeting structure, over-confirming every statement)?",
            "Did the assistant avoid unnecessary repetition? (Note: In voice conversations, briefly restating key numbers or details when the user asks for confirmation is acceptable and expected.)",
        ],
        tier=BinaryMetricDefinition.TIER_IMPORTANT,
    ),
    # --- Tier 3: Advisory (reported, doesn't affect pass/fail) ---
    "Voice Formatting": BinaryMetricDefinition(
        name="Voice Formatting",
        description="Are the responses suitable for spoken delivery?",
        default_rubric=[
            "Would the response sound natural if read aloud, without raw markup symbols or code blocks? (Ignore minor structural patterns like numbered options that are common in spoken language.)",
            "Were responses concise enough for a voice interaction (not excessively long or dense with information)?",
        ],
        tier=BinaryMetricDefinition.TIER_ADVISORY,
    ),
    "System Prompt Compliance": BinaryMetricDefinition(
        name="System Prompt Compliance",
        description="Did the assistant follow the role, rules, and constraints in its system prompt?",
        default_rubric=[
            "Did the assistant stay in character with the role defined in the system prompt?",
            "Did the assistant follow explicit constraints and rules from the system prompt (e.g., scope limits, response style)?",
            "Did the assistant refuse or redirect appropriately when asked about topics outside its defined scope?",
        ],
        tier=BinaryMetricDefinition.TIER_IMPORTANT,
    ),
    # --- Legacy alias: maps old "Response Quality" to new metrics ---
}

# Keep backward compatibility: "Response Quality" maps to "Response Accuracy"
# so existing configs that reference it still work
LEGACY_METRIC_ALIASES: Dict[str, str] = {
    "Response Quality": "Response Accuracy",
}


class BinaryLLMJudge:
    """
    LLM-based judge for evaluating conversations using binary (YES/NO) verdicts.
    Each metric is evaluated independently with a strict pass/fail outcome.
    """

    def __init__(
        self,
        judge_model: str = "claude-opus",
        region: str = None,
        log_prompts: bool = False,
        custom_metrics: Optional[Dict[str, BinaryMetricDefinition]] = None,
    ):
        registry = get_model_registry()
        resolved = registry.resolve_judge_model(judge_model)
        self.judge_model = resolved["model_id"]
        resolved_region = region or resolved["region"]
        self.client = boto3.client("bedrock-runtime", region_name=resolved_region)
        self.log_prompts = log_prompts
        self._last_prompt = None
        self._last_response = None

        # Merge built-in + custom metrics
        self.metrics = dict(BUILTIN_METRICS)
        if custom_metrics:
            self.metrics.update(custom_metrics)

    def evaluate_conversation(
        self,
        conversation_log: Dict[str, Any],
        criteria: BinaryEvaluationCriteria,
        config: Optional[Dict[str, Any]] = None,
        expected_responses: Optional[List[str]] = None,
        expected_tool_calls: Optional[List[Optional[str]]] = None,
        rubric: Optional[str] = None,
    ) -> BinaryEvaluationResult:
        turns = conversation_log.get("turns", [])
        transcript = self._build_transcript(turns)

        # Determine active metrics: use criteria.evaluation_aspects, auto-skip Tool Usage when no tools
        has_tools = bool(config and config.get("sonic_tool_config"))
        active_metrics = []
        for aspect in criteria.evaluation_aspects:
            # Resolve legacy aliases (e.g., "Response Quality" → "Response Accuracy")
            resolved = LEGACY_METRIC_ALIASES.get(aspect, aspect)
            metric_def = self.metrics.get(resolved)
            if metric_def and metric_def.requires_tools and not has_tools:
                continue
            active_metrics.append(resolved)

        # Merge rubrics: config rubrics override defaults
        merged_rubrics: Dict[str, List[str]] = {}
        for metric_name in active_metrics:
            if metric_name in criteria.rubrics and criteria.rubrics[metric_name]:
                merged_rubrics[metric_name] = criteria.rubrics[metric_name]
            else:
                metric_def = self.metrics.get(metric_name)
                if metric_def:
                    merged_rubrics[metric_name] = metric_def.default_rubric
                else:
                    merged_rubrics[metric_name] = [f"Does the conversation satisfy '{metric_name}'?"]

        prompt = self._build_evaluation_prompt(
            transcript=transcript,
            criteria=criteria,
            active_metrics=active_metrics,
            merged_rubrics=merged_rubrics,
            config=config,
            expected_responses=expected_responses,
            expected_tool_calls=expected_tool_calls,
            rubric=rubric,
        )

        response = self._call_judge_model(prompt)

        if self.log_prompts:
            self._last_prompt = prompt
            self._last_response = response

        result = self._parse_evaluation_response(response, active_metrics)
        return result

    def _build_transcript(self, turns: List[Dict[str, Any]]) -> str:
        """Build a readable transcript from turns."""
        transcript_lines = []

        for turn in turns:
            turn_num = turn.get("turn_number", 0)
            user_msg = turn.get("user_message", "")
            sonic_response = turn.get("sonic_response", "")
            tool_calls = turn.get("tool_calls", [])

            transcript_lines.append(f"Turn {turn_num}")
            transcript_lines.append(f"User: {user_msg}")

            if tool_calls:
                for tool_call in tool_calls:
                    tool_name = tool_call.get("tool_name", "unknown")
                    tool_input = tool_call.get("tool_input", {})
                    tool_result = tool_call.get("tool_result", {})
                    transcript_lines.append(f"  [Tool Call: {tool_name}({json.dumps(tool_input)})]")
                    transcript_lines.append(f"  [Tool Result: {json.dumps(tool_result)}]")

            transcript_lines.append(f"Assistant: {sonic_response}")
            transcript_lines.append("")

        return "\n".join(transcript_lines)

    def _build_evaluation_prompt(
        self,
        transcript: str,
        criteria: BinaryEvaluationCriteria,
        active_metrics: List[str],
        merged_rubrics: Dict[str, List[str]],
        config: Optional[Dict[str, Any]],
        expected_responses: Optional[List[str]] = None,
        expected_tool_calls: Optional[List[Optional[str]]] = None,
        rubric: Optional[str] = None,
    ) -> str:
        # Build system prompt section
        system_prompt_section = ""
        if config and config.get("sonic_system_prompt"):
            system_prompt_section = f"""
## Assistant System Prompt
The following system prompt was used to configure the assistant (Nova Sonic):

> {config['sonic_system_prompt']}

Evaluate whether the assistant's behavior is consistent with these instructions.
"""

        # Build tool definitions section
        tool_section = ""
        if config:
            tool_config = config.get("sonic_tool_config")
            if tool_config:
                tools = tool_config.get("tools", [])
                if tools:
                    tool_lines = []
                    for tool in tools:
                        spec = tool.get("toolSpec", {})
                        name = spec.get("name", "unknown")
                        desc = spec.get("description", "")
                        tool_lines.append(f"- **{name}**: {desc}")
                    tool_section = f"""
## Available Tools
{chr(10).join(tool_lines)}
"""

        # Build ground truth section
        ground_truth_section = ""
        if expected_responses or expected_tool_calls or rubric:
            gt_lines = []
            max_turns = max(
                len(expected_responses) if expected_responses else 0,
                len(expected_tool_calls) if expected_tool_calls else 0,
            )
            if max_turns > 0:
                for i in range(max_turns):
                    if expected_responses and i < len(expected_responses):
                        gt_lines.append(f"Turn {i + 1} expected response: \"{expected_responses[i]}\"")
                    if expected_tool_calls and i < len(expected_tool_calls) and expected_tool_calls[i]:
                        gt_lines.append(f"Turn {i + 1} expected tool call: {expected_tool_calls[i]}")

            rubric_text = ""
            if rubric:
                rubric_text = f"\n### Rubric\n{rubric}"

            if gt_lines or rubric_text:
                ground_truth_section = f"""
## Ground Truth
{chr(10).join(gt_lines)}
{rubric_text}
"""

        # Build metrics section with rubric questions
        metrics_section_lines = []
        for metric_name in active_metrics:
            metric_def = self.metrics.get(metric_name)
            desc = metric_def.description if metric_def else metric_name
            rubric_questions = merged_rubrics.get(metric_name, [])

            metrics_section_lines.append(f"### {metric_name}")
            metrics_section_lines.append(f"{desc}")
            metrics_section_lines.append("")
            metrics_section_lines.append("Rubric questions (each must be answered YES or NO):")
            for q in rubric_questions:
                metrics_section_lines.append(f"  - {q}")
            metrics_section_lines.append("")

        metrics_section = "\n".join(metrics_section_lines)

        # Build JSON output example
        example_metric = active_metrics[0] if active_metrics else "Goal Achievement"
        example_rubrics = merged_rubrics.get(example_metric, ["Did the assistant achieve the goal?"])

        prompt = f"""You are an expert evaluator judging a conversation between a user and an AI voice assistant. You are evaluating the text transcript only — you do not have access to audio, timing, tone, or prosody.

## Evaluation Context

### User Goal
{criteria.user_goal}

### Assistant Objective
{criteria.assistant_objective}
{system_prompt_section}{tool_section}{ground_truth_section}
## Conversation Transcript
```
{transcript}
```

## Evaluation Instructions

Judge each metric independently using ONLY the text evidence in the transcript above. For each rubric question, provide a strict YES or NO verdict.

Key principles:
- **Binary strictness**: Partial success is NO. "Almost correct" is NO.
- **Evidence-based**: Cite specific turns or quotes in your reasoning.
- **Transcript only**: You cannot judge audio quality, pronunciation, timing, or interruptions. Evaluate only what is visible in the text.
- **Voice conversation context**: This is a voice assistant. The user is on a phone call and cannot see text. Keep this in mind:
  - Briefly restating key numbers or details when the user asks for confirmation is normal and expected in voice — do not penalize this as repetition.
  - Numbered options ("I can do one of two things...") are natural in spoken language.
  - Only flag formatting that would be genuinely confusing when spoken aloud (raw markdown symbols like asterisks, code blocks, deeply nested structures).
- **Tool evaluation**: When evaluating tool calls, check both the parameters sent AND how the results were communicated to the user.

## Metrics to Evaluate

{metrics_section}
## Output Format

Return a JSON object with this exact structure:

```json
{{
  "metrics": {{
    "{example_metric}": {{
      "rubrics": [
        {{"question": "{example_rubrics[0]}", "verdict": "YES", "reasoning": "..."}}
      ],
      "verdict": "YES",
      "reasoning": "Overall reasoning for this metric..."
    }}
  }},
  "summary": "X of Y metrics passed."
}}
```

Rules:
- Each metric MUST have a "verdict" field with exactly "YES" or "NO"
- Each metric MUST have a "rubrics" array with per-question verdicts
- Each rubric entry MUST have "question", "verdict" (YES/NO), and "reasoning"
- The metric-level verdict is YES only if ALL its rubric questions are YES
- Return ONLY the JSON object, no additional text
"""

        return prompt

    def _call_judge_model(self, prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [{"text": prompt}],
            }
        ]

        response = self.client.converse(
            modelId=self.judge_model,
            messages=messages,
            inferenceConfig={
                "maxTokens": 4096,
                "temperature": 0.1,
            },
        )

        response_text = ""
        if "output" in response and "message" in response["output"]:
            for content in response["output"]["message"]["content"]:
                if "text" in content:
                    response_text += content["text"]

        return response_text

    def _parse_evaluation_response(
        self,
        response_text: str,
        active_metrics: List[str],
    ) -> BinaryEvaluationResult:
        try:
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1

            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")

            json_str = response_text[json_start:json_end]
            data = json.loads(json_str)

            metrics_data = data.get("metrics", {})
            metric_verdicts: Dict[str, BinaryMetricVerdict] = {}
            strengths = []
            weaknesses = []

            for metric_name in active_metrics:
                metric_result = metrics_data.get(metric_name)

                if not metric_result:
                    # Missing metric treated as FAIL
                    metric_verdicts[metric_name] = BinaryMetricVerdict(
                        metric_name=metric_name,
                        verdict=False,
                        reasoning="Metric not found in judge response",
                    )
                    weaknesses.append(f"{metric_name}: not evaluated")
                    continue

                # Parse rubric verdicts
                rubric_verdicts = []
                for rv in metric_result.get("rubrics", []):
                    raw_verdict = str(rv.get("verdict", "NO")).strip().upper()
                    rubric_verdicts.append(BinaryRubricVerdict(
                        question=rv.get("question", ""),
                        verdict=raw_verdict == "YES",
                        reasoning=rv.get("reasoning", ""),
                    ))

                # Parse metric-level verdict
                raw_verdict = str(metric_result.get("verdict", "NO")).strip().upper()
                verdict = raw_verdict == "YES"

                metric_verdicts[metric_name] = BinaryMetricVerdict(
                    metric_name=metric_name,
                    verdict=verdict,
                    reasoning=metric_result.get("reasoning", ""),
                    rubric_verdicts=rubric_verdicts,
                )

                if verdict:
                    strengths.append(f"{metric_name}: {metric_result.get('reasoning', 'PASS')}")
                else:
                    weaknesses.append(f"{metric_name}: {metric_result.get('reasoning', 'FAIL')}")

            # Compute aggregate using metric tiers
            # Critical metrics must all pass for overall PASS
            # Important metrics contribute to pass rate
            # Advisory metrics are reported but don't affect pass/fail or rate
            critical_metrics = []
            important_metrics = []
            advisory_metrics = []

            for name, mv in metric_verdicts.items():
                metric_def = self.metrics.get(name)
                tier = metric_def.tier if metric_def else BinaryMetricDefinition.TIER_CRITICAL
                if tier == BinaryMetricDefinition.TIER_CRITICAL:
                    critical_metrics.append(mv)
                elif tier == BinaryMetricDefinition.TIER_IMPORTANT:
                    important_metrics.append(mv)
                else:
                    advisory_metrics.append(mv)

            # Overall pass requires all critical metrics to pass
            critical_all_pass = all(mv.verdict for mv in critical_metrics) if critical_metrics else True

            # Pass rate = (critical passed + important passed) / (critical + important)
            scored_metrics = critical_metrics + important_metrics
            total_scored = len(scored_metrics)
            passed_scored = sum(1 for mv in scored_metrics if mv.verdict)
            pass_rate = passed_scored / total_scored if total_scored > 0 else 0.0
            pass_fail = critical_all_pass and total_scored > 0

            summary = data.get("summary", f"{passed_scored} of {total_scored} scored metrics passed.")

            # Build compatibility fields
            aspect_ratings = {name: ("PASS" if mv.verdict else "FAIL") for name, mv in metric_verdicts.items()}

            return BinaryEvaluationResult(
                metric_verdicts=metric_verdicts,
                pass_fail=pass_fail,
                pass_rate=pass_rate,
                summary=summary,
                overall_rating="PASS" if pass_fail else "FAIL",
                aspect_ratings=aspect_ratings,
                strengths=strengths,
                weaknesses=weaknesses,
            )

        except Exception as e:
            print(f"Error parsing binary evaluation response: {e}")
            print(f"Response was: {response_text[:500]}...")

            return BinaryEvaluationResult(
                metric_verdicts={},
                pass_fail=False,
                pass_rate=0.0,
                summary=f"Evaluation parsing failed: {e}",
                overall_rating="FAIL",
                aspect_ratings={},
                strengths=[],
                weaknesses=["Evaluation parsing failed"],
            )

    def save_evaluation(
        self,
        result: BinaryEvaluationResult,
        output_path: Path,
        criteria: BinaryEvaluationCriteria,
        conversation_log: Dict[str, Any],
    ):
        # Serialize metric_verdicts
        metrics_data = {}
        for name, mv in result.metric_verdicts.items():
            metric_def = self.metrics.get(name)
            tier = metric_def.tier if metric_def else "critical"
            metrics_data[name] = {
                "verdict": "PASS" if mv.verdict else "FAIL",
                "tier": tier,
                "reasoning": mv.reasoning,
                "rubric_verdicts": [
                    {
                        "question": rv.question,
                        "verdict": "YES" if rv.verdict else "NO",
                        "reasoning": rv.reasoning,
                    }
                    for rv in mv.rubric_verdicts
                ],
            }

        evaluation_data = {
            "evaluated_at": datetime.now().isoformat(),
            "judge_model": self.judge_model,
            "rating_type": "binary",
            "criteria": {
                "user_goal": criteria.user_goal,
                "assistant_objective": criteria.assistant_objective,
                "evaluation_aspects": criteria.evaluation_aspects,
                "rubrics": criteria.rubrics,
            },
            "results": {
                "overall_rating": result.overall_rating,
                "pass_fail": result.pass_fail,
                "pass_rate": result.pass_rate,
                "summary": result.summary,
                "metric_verdicts": metrics_data,
                "aspect_ratings": result.aspect_ratings,
                "strengths": result.strengths,
                "weaknesses": result.weaknesses,
            },
            "conversation_summary": {
                "total_turns": conversation_log.get("total_turns", 0),
                "total_tool_calls": conversation_log.get("total_tool_calls", 0),
                "session_id": conversation_log.get("session_id", "unknown"),
            },
        }

        if self.log_prompts and self._last_prompt:
            evaluation_data["judge_log"] = {
                "prompt": self._last_prompt,
                "raw_response": self._last_response,
            }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(evaluation_data, f, indent=2)

        print(f"Evaluation saved: {output_path}")

        # Human-readable version
        self._save_readable_evaluation(result, output_path.with_suffix(".txt"), criteria)

    def _save_readable_evaluation(
        self,
        result: BinaryEvaluationResult,
        output_path: Path,
        criteria: BinaryEvaluationCriteria,
    ):
        pass_fail_str = "PASS" if result.pass_fail else "FAIL"

        content = f"""# Conversation Evaluation (Binary Metrics)

## Criteria
- **User Goal**: {criteria.user_goal}
- **Assistant Objective**: {criteria.assistant_objective}

## Overall: {pass_fail_str} ({result.pass_rate:.0%} metrics passed)

{result.summary}

## Metric Verdicts
"""
        for name, mv in result.metric_verdicts.items():
            verdict_str = "YES" if mv.verdict else "NO"
            content += f"\n### {name}: {verdict_str}\n"
            content += f"{mv.reasoning}\n"

            if mv.rubric_verdicts:
                content += "\nRubric Details:\n"
                for rv in mv.rubric_verdicts:
                    rv_str = "YES" if rv.verdict else "NO"
                    content += f"  - [{rv_str}] {rv.question}\n"
                    content += f"    {rv.reasoning}\n"

        with open(output_path, "w") as f:
            f.write(content)
