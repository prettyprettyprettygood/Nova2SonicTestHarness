"""
Shared evaluation types used by the LLM judges.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class BinaryRubricVerdict:
    """Result of evaluating a single rubric question (YES/NO)."""
    question: str         # "Did the agent help find flights?"
    verdict: bool         # True = YES
    reasoning: str


@dataclass
class BinaryMetricVerdict:
    """Result of evaluating a single binary metric."""
    metric_name: str
    verdict: bool
    reasoning: str
    rubric_verdicts: List[BinaryRubricVerdict] = field(default_factory=list)


@dataclass
class BinaryEvaluationCriteria:
    """Criteria for binary (YES/NO) evaluation."""
    user_goal: str
    assistant_objective: str
    evaluation_aspects: List[str]
    rubrics: Dict[str, List[str]] = field(default_factory=dict)  # metric_name -> rubric questions


@dataclass
class BinaryEvaluationResult:
    """Result of a binary LLM judge evaluation."""
    metric_verdicts: Dict[str, BinaryMetricVerdict]
    pass_fail: bool           # ALL metrics must pass
    pass_rate: float          # fraction passed (0.0-1.0)
    summary: str
    # Compatibility fields for downstream consumers (multi_session_runner, results_manager)
    overall_rating: str       # "PASS" or "FAIL"
    aspect_ratings: Dict[str, str]   # {name: "PASS"/"FAIL"}
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)


@dataclass
class FactualDiscrepancy:
    """A factual difference between text output and audio transcription."""
    text_says: str
    audio_says: str
    description: str


@dataclass
class TurnConsistencyResult:
    """Result of comparing text vs. audio transcription for a single turn."""
    turn_number: int
    verdict: str  # CONSISTENT | MINOR_DIFFERENCES | HALLUCINATION
    sonic_response_text: str
    audio_transcript: str
    factual_discrepancies: List[FactualDiscrepancy] = field(default_factory=list)
    phrasing_differences: List[str] = field(default_factory=list)
    filler_words: List[str] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str = ""
    error: str = ""


@dataclass
class AudioTextConsistencyReport:
    """Aggregate report of audio-text consistency across all turns."""
    session_id: str
    evaluated_at: str
    judge_model: str
    total_turns: int
    turns_evaluated: int
    turns_skipped: int
    turns_with_errors: int
    turns_consistent: int
    turns_minor_differences: int
    turns_hallucinated: int
    hallucination_rate: float
    turn_results: List[TurnConsistencyResult] = field(default_factory=list)
    summary: str = ""
