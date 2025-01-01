import json
import requests
import uuid
from typing import List, Dict, Any

from anthropic.types.beta import (
    BetaMessageParam,
    BetaContentBlockParam,
    BetaMessage,
    BetaTextBlockParam,
    BetaTextBlock,
)

class MockRequest:
    def __init__(self, method: str, url: str, headers: Dict[str, str], content: bytes):
        self.method = method
        self.url = url
        self.headers = headers
        self._content = content

    def read(self) -> bytes:
        return self._content

class VeniceAPIResponse:
    """
    A mock response object that mimics Anthropic's APIResponse but provides
    attributes and methods compatible with existing code.
    """

    def __init__(self, venice_data: Dict[str, Any], request_data: Dict[str, Any], original_response: requests.Response):
        self.venice_data = venice_data
        self.http_request = MockRequest(
            method="POST",
            url=original_response.url,
            headers=dict(original_response.request.headers),
            content=json.dumps(request_data).encode("utf-8"),
        )
        self.http_response = original_response

    def parse(self) -> Dict[str, Any]:
        """
        Mimic the parse method to return a dict with 'role' and 'content'
        formatted as Anthropic expects.
        """
        assistant_content = ""
        try:
            if "choices" in self.venice_data and self.venice_data["choices"]:
                assistant_content = self.venice_data["choices"][0].get("message", {}).get("content", "")
        except (IndexError, KeyError, TypeError):
            assistant_content = "[Error: Unexpected Venice API response format]"

        return {
            "role": "assistant",
            "content": [{"type": "text", "text": assistant_content}],
        }

class VeniceClient:
    """
    Venice adapter that mimics the Anthropic API interface.

    Given Anthropic-style inputs (system, messages, tools), it returns an object
    that acts like an Anthropic APIResponse. This approach doesn't attempt to
    directly instantiate APIResponse but returns a class that looks similar.
    """

    # A hardcoded API key for Venice. In practice, pass this or store it securely.
    API_KEY = "tA_hV5tVV-SYvhqbiUoSdkOD_x3quWd-KMS5uKCOX3"
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
    ) -> VeniceAPIResponse:
        """
        Send Anthropic-style messages to Venice and return a VeniceAPIResponse object
        that mimics the Anthropic APIResponse interface.

        Args:
            messages (List[BetaMessageParam]): Anthropic-style input messages.
            system (str): The Anthropic system prompt.
            tools (List[Any]): Not used by Venice but included for interface compatibility.

        Returns:
            VeniceAPIResponse: An object that has http_request, http_response, and parse() method.
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

        # Return a VeniceAPIResponse object that looks like an APIResponse
        return VeniceAPIResponse(venice_response, request_payload, resp)

    def _convert_messages(
        self,
        messages: List[BetaMessageParam],
        system: str
    ) -> List[Dict[str, str]]:
        """
        Convert Anthropic-style messages to Venice's OpenAI-like format.

        Anthropic messages:
        [
          {"role": "user", "content": [BetaTextBlockParam, ...]},
          {"role": "assistant", "content": [BetaTextBlockParam, ...]},
          ...
        ]

        Venice expects:
        [
          {"role": "system", "content": "<system prompt>"},
          {"role": "user", "content": "User text"},
          {"role": "assistant", "content": "Assistant text"}
        ]
        """
        venice_messages = []
        if system:
            venice_messages.append({"role": "system", "content": system})

        for msg in messages:
            msg_text = self._extract_text_from_content(msg["content"])
            venice_messages.append({"role": msg["role"], "content": msg_text})

        return venice_messages

    def _extract_text_from_content(self, content_blocks: Any) -> str:
        """
        Extract text from Anthropic's content blocks.

        Typically, this is a list of dicts with {"type": "text", "text": "..."}.
        We join them into a single string for Venice.
        """
        if isinstance(content_blocks, list):
            return "".join(
                block["text"]
                for block in content_blocks
                if isinstance(block, dict) and block.get("type") == "text"
            )
        elif isinstance(content_blocks, dict):
            return content_blocks.get("text", "")
        else:
            return str(content_blocks)
