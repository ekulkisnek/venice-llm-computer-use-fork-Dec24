
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

class MockRequest:
    def __init__(self, method: str, url: str, headers: Dict[str, str], content: bytes):
        self.method = method
        self.url = url
        self.headers = headers
        self.content = content

    def read(self):
        return self.content

class MockResponse:
    def __init__(self, status_code: int, headers: Dict[str, str], text: str):
        self.status_code = status_code
        self.headers = headers
        self.text = text
        self._content = text.encode("utf-8")

    def read(self):
        return self._content

class VeniceAPIResponse:
    def __init__(self, venice_data: Dict[str, Any], request_data: Dict[str, Any], request_headers: Dict[str, str]):
        self.http_request = MockRequest(
            method="POST",
            url="https://api.venice.ai/api/v1/chat/completions",
            headers=request_headers,
            content=json.dumps(request_data).encode("utf-8")
        )
        self.http_response = MockResponse(
            status_code=200,
            headers={"Content-Type": "application/json"},
            text=json.dumps(venice_data)
        )
        self.venice_data = venice_data

    def parse(self) -> Dict[str, Any]:
        assistant_content = ""
        if "choices" in self.venice_data and self.venice_data["choices"]:
            assistant_content = self.venice_data["choices"][0].get("message", {}).get("content", "")
        return {
            "role": "assistant",
            "content": [{"type": "text", "text": assistant_content}],
        }

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
        
        # Create and return VeniceAPIResponse
        raw_response = VeniceAPIResponse(venice_response, request_payload, self.headers)
        return cast(APIResponse[BetaMessage], raw_response)

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

    
