# Model Aliases

Source: `configs/models.yaml`

## User Simulator Models

| Alias | Bedrock Model ID | Region |
|-------|-----------------|--------|
| `claude-haiku` | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | us-west-2 |
| `claude-sonnet` | `us.anthropic.claude-sonnet-4-20250514-v1:0` | us-west-2 |
| `gpt-oss` | `openai.gpt-oss-20b-1:0` | us-west-2 |
| `qwen` | `qwen.qwen3-32b-v1:0` | us-west-2 |

## Nova Sonic Models

| Alias | Bedrock Model ID | Region |
|-------|-----------------|--------|
| `nova-sonic` | `amazon.nova-2-sonic-v1:0` | us-east-1 |

## Judge Models

| Alias | Bedrock Model ID | Region |
|-------|-----------------|--------|
| `claude-opus` | `global.anthropic.claude-opus-4-5-20251101-v1:0` | us-east-1 |
| `claude-sonnet` | `us.anthropic.claude-sonnet-4-20250514-v1:0` | us-east-1 |

## Usage in Configs

- `sonic_model_id`: Use sonic model aliases (typically `"nova-sonic"`)
- `user_model_id`: Use user model aliases (typically `"claude-haiku"`)
- Unrecognized aliases are treated as literal Bedrock model IDs
