import json
import requests
from typing import List, Dict, Any, cast
from uuid import uuid4
from anthropic.types.beta import (
    BetaMessageParam,
    BetaContentBlockParam,
    BetaMessage,
    BetaTextBlockParam,
)
from anthropic import APIResponse


class VeniceClient:
    """
    Adapter for Venice API that mimics the Anthropic interface.
    """

    # Hardcoded API key
    API_KEY = "es-6Kh8w7VEnnX7rpCzm7lVlnGb-J8AgGX1tcOC0d9"

    def __init__(self, model: str, max_tokens: int = 4096):
        """
        Initialize the Venice client.

        Args:
            model (str): Venice model to use.
            max_tokens (int): Maximum tokens to return in the response.
        """
        self.model = model
        self.max_tokens = max_tokens
        self.api_url = "https://api.venice.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.API_KEY}",
            "Content-Type": "application/json",
        }

    def create(
        self,
        messages: List[BetaMessageParam],
        system: str,
        tools: List[Any] = None,  # Tools are ignored for now
    ) -> APIResponse[BetaMessage]:
        """
        Sends a Venice API request and returns a response in Anthropic-compatible format.

        Args:
            messages (List[BetaMessageParam]): Anthropic-style input messages.
            system (str): System prompt to prepend to the conversation.
            tools (List[Any]): Currently unused.

        Returns:
            APIResponse[BetaMessage]: Anthropic-compatible response object.
        """
        # Translate Anthropic messages to Venice's OpenAI-like format
        venice_messages = self._convert_messages(messages, system)

        # Prepare the request payload
        request_payload = {
            "model": self.model,
            "messages": venice_messages,
            "max_tokens": self.max_tokens,
            "venice_parameters": self._get_venice_parameters(),
        }

        # Make the HTTP request
        try:
            response = requests.post(
                self.api_url, headers=self.headers, json=request_payload
            )
            response.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(f"Venice API request failed: {e}")

        # Parse the Venice response
        venice_response = response.json()

        # Convert Venice response to Anthropic-compatible APIResponse
        return self._convert_response(venice_response, request_payload)

    def _convert_messages(
        self, messages: List[BetaMessageParam], system: str
    ) -> List[Dict[str, str]]:
        """
        Convert Anthropic-style messages into Venice's format.

        Args:
            messages (List[BetaMessageParam]): Anthropic-style input messages.
            system (str): System prompt to prepend.

        Returns:
            List[Dict[str, str]]: Venice-compatible message format.
        """
        venice_messages = [{"role": "system", "content": system}]
        for message in messages:
            content_text = self._extract_text_content(message["content"])
            venice_messages.append({"role": message["role"], "content": content_text})
        return venice_messages

    def _extract_text_content(
        self, content_blocks: Any
    ) -> str:
        """
        Extract text content from Anthropic-style content blocks.

        Args:
            content_blocks (Any): Content blocks from a message.

        Returns:
            str: Combined text content.
        """
        if isinstance(content_blocks, list):
            return "".join(
                block["text"]
                if isinstance(block, dict) and block.get("type") == "text"
                else block.text if hasattr(block, "text") else ""
                for block in content_blocks
            )
        elif hasattr(content_blocks, "text"):
            return content_blocks.text  # Handle single TextBlock
        elif isinstance(content_blocks, dict) and "text" in content_blocks:
            return content_blocks["text"]  # Handle single dictionary block
        else:
            return str(content_blocks)  # Fallback to string representation

    def _get_venice_parameters(self) -> Dict[str, Any]:
        """
        Helper method to generate Venice-specific parameters.

        Returns:
            Dict[str, Any]: Venice parameters.
        """
        return {
            "include_venice_system_prompt": False,  # System prompt already included
        }

    def _convert_response(
        self, venice_response: Dict[str, Any], request_payload: Dict[str, Any]
    ) -> APIResponse[BetaMessage]:
        """
        Convert Venice response into an Anthropic-compatible APIResponse.

        Args:
            venice_response (Dict[str, Any]): JSON response from Venice API.
            request_payload (Dict[str, Any]): Original request payload.

        Returns:
            APIResponse[BetaMessage]: Anthropic-compatible response object.
        """
        assistant_content = venice_response.get("choices", [{}])[0].get(
            "message", {}
        ).get("content", "[Error: No content]")

        # Create a mock BetaMessage
        beta_message = BetaMessage(
            id=str(uuid4()),  # Unique ID for the message
            role="assistant",
            content=[{"type": "text", "text": assistant_content}],
            model=self.model,
            type="message",
            usage={
                "input_tokens": venice_response.get("usage", {}).get("prompt_tokens", 0),
                "output_tokens": venice_response.get("usage", {}).get("completion_tokens", 0),
                "total_tokens": venice_response.get("usage", {}).get("total_tokens", 0),
            },
        )

        # Create a mock response object that matches the expected structure
        class MockResponse:
            def __init__(self):
                self.status_code = 200
                self.headers = {}
                self.text = json.dumps(venice_response)

            def json(self):
                return venice_response

        class MockRequest:
            method = 'POST'
            url = 'https://api.venice.ai/api/v1/chat/completions'
            headers = {'Content-Type': 'application/json'}
            
            def read(self):
                return json.dumps(request_payload).encode()

        response = MockResponse()
        request = MockRequest()
        
        return APIResponse(_raw_response=response, _parsed=beta_message, _raw_request=request)

