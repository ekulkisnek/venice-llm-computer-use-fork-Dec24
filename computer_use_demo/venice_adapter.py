import json
import requests
import uuid
from typing import List, Dict, Any, cast
from anthropic.types.beta import (
    BetaMessage,
    BetaMessageParam,
    BetaContentBlockParam,
    BetaTextBlockParam,
    BetaUsage,
)
from anthropic import APIResponse

class VeniceClient:
    """
    Venice adapter that mimics the Anthropic API interface.
    Given Anthropic-style inputs (system, messages, tools), it returns an Anthropic-compatible APIResponse.
    """

    # A hardcoded API key for Venice. In practice, this should be set elsewhere or passed in.
    API_KEY = "es-6Kh8w7VEnnX7rpCzm7lVlnGb-J8AgGX1tcOC0d9"
    VENICE_API_URL = "https://api.venice.ai/api/v1/chat/completions"

    def __init__(self, model: str, max_tokens: int = 4096):
        self.model = model
        self.max_tokens = max_tokens
        self.headers = {
            "Authorization": f"Bearer {self.API_KEY}",
            "Content-Type": "application/json",
        }

    def create(
        self,
        messages: List[BetaMessageParam],
        system: str,
        tools: List[Any] = None
    ) -> APIResponse[BetaMessage]:
        """
        Send Anthropic-style messages to Venice and get an Anthropic-compatible response.

        Args:
            messages (List[BetaMessageParam]): Anthropic-style input messages.
            system (str): The Anthropic system prompt string.
            tools (List[Any]): Not currently used by Venice, but required by the interface.

        Returns:
            APIResponse[BetaMessage]: A response that looks like what Anthropic returns.
        """
        # Convert Anthropic messages into Venice-compatible format
        venice_messages = self._convert_messages(messages, system)

        # Prepare request payload
        request_payload = {
            "model": self.model,
            "messages": venice_messages,
            "max_tokens": self.max_tokens,
            "venice_parameters": {
                "include_venice_system_prompt": False
            },
        }

        # Make the Venice API request
        try:
            resp = requests.post(
                self.VENICE_API_URL, headers=self.headers, json=request_payload
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(f"Venice API request failed: {e}")

        venice_response = resp.json()

        # Convert Venice response to Anthropic-compatible APIResponse
        return self._convert_response(venice_response, request_payload, resp)

    def _convert_messages(
        self,
        messages: List[BetaMessageParam],
        system: str
    ) -> List[Dict[str, str]]:
        """
        Convert Anthropic messages to Venice's OpenAI-like format.

        Anthropic messages are a list of dicts like:
        {
          "role": "user" | "assistant" | "system",
          "content": [BetaTextBlockParam or similar...]
        }

        Venice expects:
        [
          {"role": "system", "content": "<system prompt>"},
          {"role": "user", "content": "<user text>"},
          {"role": "assistant", "content": "<assistant text>"}
        ]

        We prepend the Anthropic system prompt as a "system" message.
        """
        venice_messages = []
        if system:
            venice_messages.append({"role": "system", "content": system})

        for msg in messages:
            # Extract all text from the message content
            msg_text = self._extract_text_from_content(msg["content"])
            # Venice roles match Anthropic ("system", "user", "assistant"),
            # so we can reuse them directly.
            venice_messages.append({"role": msg["role"], "content": msg_text})

        return venice_messages

    def _extract_text_from_content(self, content_blocks: Any) -> str:
        """
        Extract text from the Anthropic message content blocks.

        Anthropic content blocks are usually a list of dicts with {"type": "text", "text": "..."}.
        We join them into a single string for Venice.
        """
        if isinstance(content_blocks, list):
            # Filter and join all text blocks
            return "".join(
                block["text"]
                for block in content_blocks
                if isinstance(block, dict) and block.get("type") == "text"
            )
        elif isinstance(content_blocks, dict):
            # Single block scenario
            return content_blocks.get("text", "")
        else:
            # If it's something else, fallback to str
            return str(content_blocks)

    def _convert_response(
        self,
        venice_response: Dict[str, Any],
        request_payload: Dict[str, Any],
        original_response: requests.Response
    ) -> APIResponse[BetaMessage]:
        """
        Convert the Venice JSON response into an Anthropic-compatible APIResponse[BetaMessage].

        Anthropic's BetaMessage has a structure like:
        {
          "role": "assistant",
          "content": [ { "type": "text", "text": "..."} ],
          "model": "modelname",
          "id": "some_uuid",
          "type": "message",
          "usage": BetaUsage(...),
          "system": null,
          "stop_reason": ...,
          "stop_sequence": ...
        }

        We'll mimic that here.
        """
        # Extract assistant content from Venice
        assistant_content = venice_response.get("choices", [{}])[0].get("message", {}).get("content", "")

        # Venice usage fields: Venice often provides usage similar to OpenAI style:
        # {"usage": {"prompt_tokens": X, "completion_tokens": Y, "total_tokens": Z}}
        usage_data = venice_response.get("usage", {})
        usage = BetaUsage(
            cache_creation_input_tokens=None,
            cache_read_input_tokens=None,
            input_tokens=usage_data.get("prompt_tokens", 0),
            output_tokens=usage_data.get("completion_tokens", 0)
        )

        # Construct a BetaMessage
        beta_message = BetaMessage(
            id=str(uuid.uuid4()),
            type="message",
            role="assistant",
            content=cast(List[BetaContentBlockParam], [{"type": "text", "text": assistant_content}]),
            model=request_payload["model"],
            usage=usage,
            system=None,
            stop_reason="end_turn",  # Venice doesn't provide stop_reason: choose a default, or None if unknown
            stop_sequence=None
        )

        # Build a mock request object for APIResponse
        prepared_request = requests.Request(
            method="POST",
            url=self.VENICE_API_URL,
            headers=self.headers,
            json=request_payload,
        ).prepare()

        # Build a mock response object for APIResponse
        # We already have original_response from requests, we can reuse it:
        # Just ensure content is the venice_response JSON.
        # Note: `original_response` is a requests.Response, which already has `status_code`, `headers`, etc.
        # We can simply reuse it. But we need to ensure the content is accessible, which it should be.

        # Construct the APIResponse object
        return APIResponse._create(
            response=original_response,
            parsed=beta_message,
            request=prepared_request
        )
