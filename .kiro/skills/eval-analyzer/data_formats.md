# Evaluation Data Formats

## Batch Index (`results/batches/batch_index.json`)

```json
{
  "batches": [
    {
      "batch_id": "batch_20260225_144223",
      "timestamp": "2026-02-25T14:42:23",
      "total_sessions": 5,
      "completed": 5,
      "failed": 0,
      "pass_rate": 0.8
    }
  ]
}
```

## Batch Summary (`results/batches/<batch_id>/batch_summary.json`)

```json
{
  "batch_id": "batch_20260225_144223",
  "totals": {
    "total_sessions": 5,
    "completed": 5,
    "failed": 0,
    "evaluated": 5
  },
  "evaluation_summary": {
    "pass_rate": 0.8,
    "pass_count": 4,
    "fail_count": 1,
    "metric_pass_rates": {
      "Goal Achievement": { "passed": 4, "total": 5, "rate": 0.8 },
      "Response Quality": { "passed": 5, "total": 5, "rate": 1.0 },
      "Conversation Flow": { "passed": 3, "total": 5, "rate": 0.6 }
    }
  },
  "sessions": [
    {
      "session_id": "order_status_normal_20260225_144223_a1b2c3",
      "config_name": "order_status_normal",
      "status": "completed",
      "evaluation": {
        "overall_rating": "PASS",
        "pass_fail": true,
        "pass_rate": 1.0,
        "aspect_ratings": {
          "Goal Achievement": "PASS",
          "Response Quality": "PASS"
        },
        "strengths": ["Correctly used lookupOrder tool"],
        "weaknesses": []
      }
    }
  ]
}
```

## Session Evaluation (`results/sessions/<id>/evaluation/llm_judge_evaluation.json`)

```json
{
  "evaluation_type": "binary",
  "criteria": {
    "user_goal": "Check order status",
    "assistant_objective": "Help with order inquiries using tools",
    "evaluation_aspects": ["Goal Achievement", "Tool Usage", "Response Quality"]
  },
  "results": {
    "overall_rating": "FAIL",
    "pass_fail": false,
    "pass_rate": 0.67,
    "metric_verdicts": {
      "Goal Achievement": {
        "verdict": "PASS",
        "reasoning": "The agent successfully looked up the order...",
        "rubric_verdicts": []
      },
      "Tool Usage": {
        "verdict": "FAIL",
        "reasoning": "The agent called lookupOrder but with wrong parameter format...",
        "rubric_verdicts": [
          {
            "question": "Did the agent use the correct tool?",
            "verdict": "YES",
            "reasoning": "Used lookupOrder as expected"
          },
          {
            "question": "Were tool parameters correct?",
            "verdict": "NO",
            "reasoning": "Passed order_id without the ORD- prefix"
          }
        ]
      }
    },
    "aspect_ratings": {
      "Goal Achievement": "PASS",
      "Tool Usage": "FAIL",
      "Response Quality": "PASS"
    },
    "strengths": ["Natural conversation flow", "Empathetic tone"],
    "weaknesses": ["Tool Usage: incorrect parameter format for lookupOrder"]
  }
}
```

## Interaction Log (`results/sessions/<id>/logs/interaction_log.json`)

```json
{
  "session_id": "order_status_normal_20260225_144223_a1b2c3",
  "test_name": "order_status_normal",
  "start_time": "2026-02-25T14:42:23.261709",
  "end_time": "2026-02-25T14:47:17.065399",
  "configuration": { "...full TestConfig as dict..." },
  "turns": [
    {
      "turn_number": 1,
      "timestamp": "2026-02-25T14:42:25.000000",
      "user_message": "Hi, I'd like to check on my order ORD-2024-33100.",
      "sonic_response": "Of course! Let me look that up for you.",
      "tool_calls": [
        {
          "tool_name": "lookupOrder",
          "tool_use_id": "uuid",
          "tool_input": { "order_id": "ORD-2024-33100" },
          "tool_result": { "status": "shipped" },
          "timestamp": "2026-02-25T14:42:30.000000"
        }
      ],
      "audio_recorded": true,
      "audio_file": "turn_1.wav",
      "sonic_session_id": "uuid"
    }
  ],
  "total_turns": 4,
  "total_tool_calls": 1,
  "errors": [],
  "session_events": [],
  "summary": {
    "total_turns": 4,
    "total_tool_calls": 1,
    "tool_usage_rate": "25.00%",
    "avg_user_message_length": 236.5,
    "avg_sonic_response_length": 286.0,
    "session_continuations": 0
  }
}
```

## Key Analysis Patterns

- **High failure rate on a single metric**: Usually indicates a system prompt issue or missing tool
- **Co-failures** (multiple metrics fail together): Often indicates a fundamental conversation breakdown
- **Failures only on specific configs**: The scenario or persona may be too adversarial
- **Tool Usage failures**: Check tool definitions, parameter schemas, and system prompt tool instructions
- **Conversation Flow failures**: Often caused by too-long responses, repetition, or ignoring user input
- **System Prompt Compliance failures**: The agent broke character, ignored constraints, or failed to redirect out-of-scope requests
- **Response Quality failures**: Check for hallucinated facts, markdown in spoken responses, or irrelevant tangents
- **Regressions between batches**: Compare the configs and system prompts that changed
