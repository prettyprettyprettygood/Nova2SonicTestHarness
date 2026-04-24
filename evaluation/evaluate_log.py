"""
Evaluate an existing conversation log using the Binary LLM Judge.

Usage:
    python main.py evaluate <session_directory>
    python main.py evaluate results/sessions/conversation_with_tools_20250101_120000_abc123/

Optional arguments:
    --user-goal "Custom user goal"
    --assistant-objective "Custom assistant objective"
    --aspects "Goal Achievement,Tool Usage,Response Quality"
    --rubrics '{"Goal Achievement": ["Did the agent help?"]}'
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional, List, Dict

from evaluation.llm_judge_binary import BinaryLLMJudge
from evaluation.evaluation_types import BinaryEvaluationCriteria


async def evaluate_log(
    log_dir: Path,
    user_goal: Optional[str] = None,
    assistant_objective: Optional[str] = None,
    evaluation_aspects: Optional[List[str]] = None,
    rubrics: Optional[Dict[str, List[str]]] = None,
):
    """Evaluate a conversation log."""

    # Find interaction_log.json (new layout: logs/ subdir; legacy: root)
    log_path = log_dir / "logs" / "interaction_log.json"
    if not log_path.exists():
        log_path = log_dir / "interaction_log.json"
    if not log_path.exists():
        print(f"Error: interaction_log.json not found in {log_dir}")
        return

    print(f"\n{'='*60}")
    print(f"Evaluating: {log_dir.name}")
    print(f"{'='*60}\n")

    # Load conversation log
    with open(log_path, 'r') as f:
        conversation_log = json.load(f)

    print(f"Loaded conversation with {len(conversation_log.get('turns', []))} turns\n")

    # Use defaults if not provided
    if not user_goal:
        config = conversation_log.get('config', {})
        user_goal = config.get('user_system_prompt', config.get('claude_system_prompt', 'Complete the conversation successfully'))

    if not assistant_objective:
        assistant_objective = conversation_log.get('config', {}).get('sonic_system_prompt', 'Provide helpful assistance')

    if not evaluation_aspects:
        evaluation_aspects = ['Goal Achievement', 'Response Accuracy', 'Conversation Flow', 'Voice Formatting']
        if conversation_log.get('config', {}).get('sonic_tool_config'):
            evaluation_aspects.append('Tool Usage')
        if conversation_log.get('config', {}).get('sonic_system_prompt'):
            evaluation_aspects.append('System Prompt Compliance')

    print(f"Evaluation Criteria:")
    print(f"   User Goal: {user_goal}")
    print(f"   Assistant Objective: {assistant_objective}")
    print(f"   Aspects: {', '.join(evaluation_aspects)}\n")

    # Determine eval output path (new layout: evaluation/ subdir; legacy: root)
    eval_dir = log_dir / "evaluation"
    if eval_dir.exists() and eval_dir.is_dir():
        eval_path = eval_dir / 'llm_judge_evaluation.json'
    else:
        eval_path = log_dir / 'llm_judge_evaluation.json'

    judge = BinaryLLMJudge(judge_model="claude-opus")

    criteria = BinaryEvaluationCriteria(
        user_goal=user_goal,
        assistant_objective=assistant_objective,
        evaluation_aspects=evaluation_aspects,
        rubrics=rubrics or {},
    )

    print("Running binary evaluation...\n")
    result = judge.evaluate_conversation(
        conversation_log=conversation_log,
        criteria=criteria,
    )

    # Print results
    print(f"{'='*60}")
    print("Evaluation Results")
    print(f"{'='*60}\n")
    print(f"Overall: {'PASS' if result.pass_fail else 'FAIL'} ({result.pass_rate:.0%} metrics passed)")
    print(f"{result.summary}\n")

    print("Metric Verdicts:")
    for name, mv in result.metric_verdicts.items():
        verdict_str = "YES" if mv.verdict else "NO"
        print(f"  {'PASS' if mv.verdict else 'FAIL'} {name}: {verdict_str}")
        if mv.rubric_verdicts:
            for rv in mv.rubric_verdicts:
                rv_str = "YES" if rv.verdict else "NO"
                print(f"      [{rv_str}] {rv.question}")

    judge.save_evaluation(
        result=result,
        output_path=eval_path,
        criteria=criteria,
        conversation_log=conversation_log,
    )
    print(f"\nEvaluation saved to: {eval_path}")


def main():
    """Parse arguments and run evaluation."""
    if len(sys.argv) < 2:
        print("Usage: python main.py evaluate <session_directory>")
        print("\nExample:")
        print("  python main.py evaluate results/sessions/conversation_with_tools_20250101_120000_abc123/")
        print("\nOptional arguments:")
        print("  --user-goal \"Custom user goal\"")
        print("  --assistant-objective \"Custom assistant objective\"")
        print("  --aspects \"Goal Achievement,Tool Usage,Response Quality\"")
        print("  --rubrics '{\"Goal Achievement\": [\"Did the agent help?\"]}'")
        sys.exit(1)

    log_dir = Path(sys.argv[1])

    if not log_dir.exists():
        print(f"Error: Directory not found: {log_dir}")
        sys.exit(1)

    # Parse optional arguments
    user_goal = None
    assistant_objective = None
    evaluation_aspects = None
    rubrics = None

    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--user-goal" and i + 1 < len(sys.argv):
            user_goal = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--assistant-objective" and i + 1 < len(sys.argv):
            assistant_objective = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--aspects" and i + 1 < len(sys.argv):
            evaluation_aspects = [a.strip() for a in sys.argv[i + 1].split(',')]
            i += 2
        elif sys.argv[i] == "--rubrics" and i + 1 < len(sys.argv):
            rubrics = json.loads(sys.argv[i + 1])
            i += 2
        else:
            print(f"Unknown argument: {sys.argv[i]}")
            i += 1

    # Run evaluation
    asyncio.run(evaluate_log(
        log_dir,
        user_goal=user_goal,
        assistant_objective=assistant_objective,
        evaluation_aspects=evaluation_aspects,
        rubrics=rubrics,
    ))


if __name__ == "__main__":
    main()
