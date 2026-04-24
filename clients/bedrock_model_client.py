"""
Bedrock Model Client for Multi-Turn Conversations
Supports multiple Bedrock models as user simulators.
"""

import boto3
from typing import Optional, List, Dict, Any

from utils.model_registry import get_model_registry


class BedrockModelClient:
    """
    Unified Bedrock client for generating conversational inputs.
    Model aliases are resolved via configs/models.yaml through ModelRegistry.
    """

    def __init__(
        self,
        model: str = "claude-haiku",
        region: str = None,
        aws_access_key: Optional[str] = None,
        aws_secret_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 1.0
    ):
        """
        Initialize Bedrock model client.

        Args:
            model: Model alias ("claude-haiku", "gpt-oss", "qwen") or full Bedrock model ID
            region: AWS region (overrides models.yaml if provided)
            aws_access_key: AWS access key (or use env var)
            aws_secret_key: AWS secret key (or use env var)
            system_prompt: System prompt for the model's role
            max_tokens: Maximum tokens for responses
            temperature: Temperature for responses
        """
        # Resolve model ID via registry
        registry = get_model_registry()
        resolved = registry.resolve_user_model(model)
        self.model_id = resolved["model_id"]
        self.model_short_name = model
        resolved_region = region or resolved["region"]

        # Initialize Bedrock client
        self.client = boto3.client(
            "bedrock-runtime",
            region_name=resolved_region,
        )

        self.system_prompt = system_prompt or self._get_default_system_prompt()
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Conversation history
        self.conversation_history: List[Dict[str, Any]] = []

    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for conversational user."""
        return """You are a user having a conversation with an AI assistant. Respond directly as the user would speak.

CRITICAL: Output ONLY the user's actual words - do not add phrases like "I would say", "Here's what I'd ask", or any meta-commentary.

Your behavior:
- Speak naturally and conversationally (1-3 sentences typically)
- React naturally to the assistant's responses
- Ask follow-up questions when appropriate
- Show realistic user behavior (clarifications, confirmations, new requests)

Examples of CORRECT responses:
✓ "I'm planning a trip to Japan next spring. Do you have any ideas for a 7-day itinerary?"
✓ "That sounds great! What about food recommendations?"
✓ "Thanks! Can you tell me more about the temples?"

Examples of INCORRECT responses:
✗ "Sure thing—here's what I would say: 'I'm planning a trip...'"
✗ "I would ask: 'Do you have recommendations?'"
✗ "My response would be: 'That sounds interesting.'"

Remember: You ARE the user. Speak directly without framing or meta-commentary."""

    def generate_response(
        self,
        context: str,
        assistant_message: Optional[str] = None
    ) -> str:
        """
        Generate a user response based on conversation context.

        Args:
            context: Context or instruction for what the user should say
            assistant_message: The latest message from the assistant (optional)

        Returns:
            Generated user message
        """
        print(f"🤔 Generating user message with {self.model_short_name}...")

        # Build messages
        messages = []

        # Add conversation history
        for msg in self.conversation_history:
            messages.append(msg)

        # Add the new context/instruction
        user_instruction = context
        if assistant_message:
            user_instruction = f"The assistant just said: \"{assistant_message}\"\n\nYour instruction: {context}"

        messages.append({
            "role": "user",
            "content": [{"text": user_instruction}]
        })

        # Call Bedrock with converse API (synchronous)
        try:
            response = self.client.converse(
                modelId=self.model_id,
                messages=messages,
                system=[{"text": self.system_prompt}],
                inferenceConfig={
                    "maxTokens": self.max_tokens,
                    "temperature": self.temperature
                }
            )
        except Exception as e:
            print(f"❌ Error calling Bedrock: {e}")
            raise

        # Extract text response
        user_message = ""
        if "output" in response and "message" in response["output"]:
            for content in response["output"]["message"]["content"]:
                if "text" in content:
                    user_message += content["text"]

        # Update conversation history (store with proper format)
        self.conversation_history.append({
            "role": "user",
            "content": [{"text": user_instruction}]
        })
        self.conversation_history.append({
            "role": "assistant",
            "content": [{"text": user_message}]
        })

        return user_message.strip()

    def add_to_history(self, role: str, content: str):
        """
        Manually add a message to conversation history.

        Args:
            role: "user" or "assistant"
            content: Message content
        """
        self.conversation_history.append({
            "role": role,
            "content": [{"text": content}]
        })

    def get_history(self) -> List[Dict[str, Any]]:
        """Get conversation history."""
        return self.conversation_history.copy()

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []

    def set_system_prompt(self, prompt: str):
        """Update system prompt."""
        self.system_prompt = prompt


class ScriptedUser:
    """
    A scripted user that follows a predefined conversation script.
    Useful for repeatable testing scenarios.
    """

    def __init__(self, script: List[str]):
        """
        Initialize with a conversation script.

        Args:
            script: List of user messages to send in sequence
        """
        self.script = script
        self.current_index = 0

    def get_next_message(self) -> Optional[str]:
        """
        Get the next message from the script.

        Returns:
            Next user message, or None if script is complete
        """
        if self.current_index >= len(self.script):
            return None

        message = self.script[self.current_index]
        self.current_index += 1
        return message

    def has_more_messages(self) -> bool:
        """Check if there are more messages in the script."""
        return self.current_index < len(self.script)

    def reset(self):
        """Reset to the beginning of the script."""
        self.current_index = 0


# Example scripted conversations for testing
EXAMPLE_SCRIPTS = {
    "booking_inquiry": [
        "Hi, I need to check on my hotel booking",
        "The confirmation number is ABC123",
        "When is the check-in date?",
        "Perfect, thank you!"
    ],
    "travel_planning": [
        "I'm planning a trip to Paris",
        "What are the best attractions to visit?",
        "Tell me more about the Eiffel Tower",
        "Great, I'd like to book a hotel nearby"
    ],
    "rewards_info": [
        "What are my Chase credit card benefits?",
        "Tell me about travel rewards",
        "How do I redeem my points?",
        "Thanks for the information!"
    ],
    "multi_turn_complex": [
        "I need help planning a trip",
        "I want to go to Tokyo in March",
        "What's the weather like then?",
        "Should I book hotels in advance?",
        "What about my Chase travel benefits?",
        "Can I use my points for the hotel?",
        "Perfect, let me check my current bookings first",
        "Do I have any upcoming reservations?",
        "Great, now I'm ready to book Tokyo!"
    ]
}
