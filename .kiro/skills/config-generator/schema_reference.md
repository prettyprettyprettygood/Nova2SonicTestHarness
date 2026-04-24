# TestConfig Schema Reference

All fields with their types, defaults, and descriptions. Source: `core/config_manager.py`

## Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `test_name` | string | Snake_case identifier for the test |

## Nova Sonic Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `description` | string | `""` | Human-readable test description |
| `sonic_model_id` | string | `"nova-sonic"` | Model alias (see model_aliases.md) |
| `sonic_region` | string | `"us-east-1"` | AWS region for Nova Sonic |
| `sonic_endpoint_uri` | string? | `null` | Custom endpoint URI (rare) |
| `sonic_system_prompt` | string | `"You are a helpful assistant."` | System prompt for the voice agent |
| `sonic_voice_id` | string | `"matthew"` | Output voice: `matthew`, `tiffany`, `amy` |
| `sonic_inference_config` | object | see below | Inference parameters |
| `sonic_tool_config` | object? | `null` | Tool definitions (see Tool Config below) |

### sonic_inference_config

```json
{
  "max_tokens": 1024,
  "temperature": 0.7,
  "top_p": 0.9
}
```

## User Simulator Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `user_model_id` | string | `"claude-haiku"` | Model alias for user simulator |
| `user_system_prompt` | string? | `null` | Persona prompt for simulated caller |
| `user_max_tokens` | int | `1024` | Max tokens for user responses |
| `user_temperature` | float | `1.0` | Temperature for user model |

## Interaction Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `interaction_mode` | string | `"bedrock"` | `"bedrock"` (LLM user) or `"scripted"` |
| `scripted_messages` | list[str] | `[]` | Pre-defined user messages (scripted mode) |
| `max_turns` | int | `10` | Maximum conversation turns |
| `turn_delay` | float | `0` | Seconds between turns (increase if throttled) |

## Logging Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `log_audio` | bool | `true` | Record audio per turn |
| `log_text` | bool | `true` | Log text transcripts |
| `log_tools` | bool | `true` | Log tool calls |
| `log_events` | bool | `true` | Log streaming events (debug) |
| `output_directory` | string | `"results"` | Base output directory |

## Evaluation Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `auto_evaluate` | bool | `false` | Run LLM judge after conversation |
| `evaluation_criteria` | object? | `null` | See Evaluation Criteria below |
| `log_judge_prompts` | bool | `false` | Include judge prompts in output |

### evaluation_criteria

```json
{
  "user_goal": "What the caller wants to achieve",
  "assistant_objective": "What the agent should do",
  "evaluation_aspects": ["Goal Achievement", "Conversation Flow", "Response Quality"],
  "rubrics": {
    "Goal Achievement": ["Custom rubric question 1?", "Custom rubric question 2?"]
  }
}
```

- `evaluation_aspects`: List of metric names to evaluate. Only use the built-in names below.
- `rubrics` (optional): Override default rubric questions for any metric. Each key is a metric name, value is a list of YES/NO questions. The metric passes only if ALL rubric questions pass. If omitted, the default rubrics below are used.

### Built-in Evaluation Metrics and Default Rubrics

Metrics are organized into tiers that control how they affect the overall pass/fail verdict.

#### Tier 1: Critical (must pass for overall PASS)

**Goal Achievement**
- Did the assistant correctly identify and address the user's primary goal?
- Did the assistant provide the specific information or outcome the user requested?
- Was the goal resolved by the end of the conversation (not left open or deferred)?

**Response Accuracy**
- Were all facts, numbers, and claims in the responses accurate (no hallucinated information)?
- Were responses relevant to the user's question without unnecessary tangents?
- Did the assistant correctly interpret and relay tool results without distortion?

#### Tier 2: Important (contributes to pass rate, doesn't auto-fail)

**Tool Usage** *(auto-skipped when no tools are configured)*
- Did the assistant call the right tool(s) for the user's request?
- Were the tool parameters accurate and correctly formatted (matching user-provided values)?
- Did the assistant correctly interpret and relay the tool results to the user?
- Did the assistant avoid unnecessary or redundant tool calls?

**Conversation Flow**
- Did the assistant's responses follow logically from what the user said?
- Did the conversation flow naturally without robotic or formulaic patterns?
- Did the assistant avoid unnecessary repetition? (Restating key details for user confirmation in voice is acceptable.)

**System Prompt Compliance**
- Did the assistant stay in character with the role defined in the system prompt?
- Did the assistant follow explicit constraints and rules from the system prompt (e.g., scope limits, response style)?
- Did the assistant refuse or redirect appropriately when asked about topics outside its defined scope?

#### Tier 3: Advisory (reported, doesn't affect pass/fail)

**Voice Formatting**
- Would the response sound natural if read aloud, without raw markup symbols or code blocks?
- Were responses concise enough for a voice interaction?

#### Legacy Alias
- `Response Quality` is accepted as an alias for `Response Accuracy` for backward compatibility.

## Audio Input Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `input_mode` | string | `"text"` | `"text"` (default) or `"polly"` (TTS audio) |
| `polly_voice_id` | string | `"Matthew"` | Polly voice for audio input |
| `polly_engine` | string | `"neural"` | `"neural"` or `"standard"` |
| `polly_region` | string? | `null` | Defaults to sonic_region |
| `record_conversation` | bool | `false` | Record full conversation audio |

## Tool Registry

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `tool_registry_module` | string? | `null` | Python module path with tool handlers |

## Dataset Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dataset_path` | string? | `null` | Path to JSONL dataset or `hf://org/dataset` |

## Session Continuation

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_session_continuation` | bool | `true` | Auto-reconnect on 8-min timeout |
| `transition_threshold_seconds` | float | `360.0` | Seconds before proactive reconnect |
| `audio_buffer_duration_seconds` | float | `3.0` | Audio buffer for transitions |

## Tool Config Format (sonic_tool_config)

```json
{
  "tools": [
    {
      "toolSpec": {
        "name": "toolName",
        "description": "What the tool does",
        "inputSchema": {
          "json": {
            "type": "object",
            "properties": {
              "param1": {
                "type": "string",
                "description": "Parameter description"
              }
            },
            "required": ["param1"]
          }
        }
      }
    }
  ],
  "tool_choice": {
    "auto": {}
  }
}
```
