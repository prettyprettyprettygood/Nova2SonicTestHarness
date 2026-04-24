# Tool Handler Pattern

Reference: `tools/order_status_tools.py`

## Module Structure

```python
"""
<Description> Tools for <Scenario>

Tool handlers for the <scenario> test scenario.
Provides <tool_name> tool(s) with mock data.
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any

from tools.tool_registry import ToolRegistry

# Module-level registry (loaded by main.py via tool_registry_module config)
registry = ToolRegistry()

# Mock data
MOCK_DATA = {
    "item_001": {
        "status": "success",
        "field1": "value1",
        # ... realistic mock fields
    },
}


@registry.tool("toolName")
async def tool_handler(tool_input: Dict[str, Any]) -> Dict[str, Any]:
    """Description of what this tool does."""
    param = tool_input.get("param_name", "")
    
    # Simulate API latency
    await asyncio.sleep(0.5)
    
    if param in MOCK_DATA:
        return MOCK_DATA[param]
    
    return {
        "status": "error",
        "error": "Not found",
        "message": "No data found for this input."
    }


async def main():
    """Standalone test for the tool handlers."""
    print("Tool Module Test")
    print(f"Registered tools: {', '.join(registry.list_tools())}")
    
    # Test each mock entry
    for key in MOCK_DATA:
        result = await registry.execute("toolName", {"param_name": key})
        print(f"  {key} -> {result.get('status', 'unknown')}")
    
    # Test error case
    result = await registry.execute("toolName", {"param_name": "INVALID"})
    print(f"  INVALID -> {result.get('error', 'unknown')}")


if __name__ == "__main__":
    asyncio.run(main())
```

## Matching sonic_tool_config JSON

For each tool handler, generate the corresponding config block:

```json
{
  "tools": [
    {
      "toolSpec": {
        "name": "toolName",
        "description": "What this tool does — be specific so the model knows when to call it",
        "inputSchema": {
          "json": {
            "type": "object",
            "properties": {
              "param_name": {
                "type": "string",
                "description": "Description of this parameter"
              }
            },
            "required": ["param_name"]
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

## Key Rules

- Tool names in the handler (`@registry.tool("name")`) must exactly match `toolSpec.name` in the config
- Parameter names in `tool_input.get()` must match `inputSchema.properties` keys
- Always include error handling for missing/invalid inputs
- Mock data should be realistic and varied (multiple entries with different states)
- The `registry` variable must be at module level — `main.py` imports it via `importlib`
- Config field `tool_registry_module` uses Python dotted path: `"tools.my_tools"` or `"examples.my_tools"`
