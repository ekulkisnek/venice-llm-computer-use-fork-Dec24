"""
Agentic sampling loop that calls the Anthropic API and local implenmentation of anthropic-defined computer use tools.
"""

import platform
import json
from collections.abc import Callable
from datetime import datetime
from enum import StrEnum
from typing import Any, cast
import logging
from logging.handlers import RotatingFileHandler
from logging import getLogger

from anthropic import Anthropic, AnthropicBedrock, AnthropicVertex, APIResponse

def truncate_string(s, max_length=4000):
    return s if len(s) <= max_length else s[:max_length] + "... [truncated]"

def setup_logging():
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # File handler
    file_handler = RotatingFileHandler('anthropic_api.log', maxBytes=1024*1024, backupCount=5)
    file_handler.setFormatter(formatter)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger

logger = setup_logging()
from anthropic.types import (
    MessageParam,
    ToolParam,
    ToolResultBlockParam,
)
from anthropic.types.beta import (
    BetaContentBlock,
    BetaContentBlockParam,
    BetaImageBlockParam,
    BetaMessage,
    BetaMessageParam,
    BetaTextBlockParam,
    BetaToolParam,
    BetaToolResultBlockParam,
)
from .venice_adapter import VeniceClient

from .tools import BashTool, ComputerTool, EditTool, ToolCollection, ToolResult

BETA_FLAG = "computer-use-2024-10-22"


class APIProvider(StrEnum):
    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"
    VERTEX = "vertex"
    VENICE = "venice"


PROVIDER_TO_DEFAULT_MODEL_NAME: dict[APIProvider, str] = {
    APIProvider.ANTHROPIC: "claude-3-5-sonnet-20241022",
    APIProvider.BEDROCK: "anthropic.claude-3-5-sonnet-20241022-v2:0",
    APIProvider.VERTEX: "claude-3-5-sonnet-v2@20241022",
    APIProvider.VENICE: "most_intelligent",
}


# This system prompt is optimized for the Docker environment in this repository and
# specific tool combinations enabled.
# We encourage modifying this system prompt to ensure the model has context for the
# environment it is running in, and to provide any additional information that may be
# helpful for the task at hand.
SYSTEM_PROMPT = f"""<SYSTEM_CAPABILITY>
* You are utilising an Linux computer using {platform.machine()} architecture with internet access.
* Firefox will be started for you.
* Using bash tool you can start GUI applications. GUI apps run with bash tool will appear within your desktop environment, but they may take some time to appear. Take a screenshot to confirm it did.
* When using your bash tool with commands that are expected to output very large quantities of text, redirect into a tmp file and use str_replace_editor or `grep -n -B <lines before> -A <lines after> <query> <filename>` to confirm output.
* When viewing a page it can be helpful to zoom out so that you can see everything on the page.  Either that, or make sure you scroll down to see everything before deciding something isn't available.
* When using your computer function calls, they take a while to run and send back to you.  Where possible/feasible, try to chain multiple of these calls all into one function calls request.
* The current date is {datetime.today().strftime('%A, %B %-d, %Y')}.
</SYSTEM_CAPABILITY>

<IMPORTANT>
* When using Firefox, if a startup wizard appears, IGNORE IT.  Do not even click "skip this step".  Instead, click on the address bar where it says "Search or enter address", and enter the appropriate search term or URL there.
* If the item you are looking at is a pdf, if after taking a single screenshot of the pdf it seems that you want to read the entire document instead of trying to continue to read the pdf from your screenshots + navigation, determine the URL, use curl to download the pdf, install and use pdftotext to convert it to a text file, and then read that text file directly with your StrReplaceEditTool.
* When viewing a webpage, first use your computer tool to view it and explore it.  But, if there is a lot of text on that page, instead curl the html of that page to a file on disk and then using your StrReplaceEditTool to view the contents in plain text.
</IMPORTANT>"""


logger = getLogger(__name__)

async def sampling_loop(
    *,
    model: str,
    provider: APIProvider,
    system_prompt_suffix: str,
    messages: list[BetaMessageParam],
    output_callback: Callable[[BetaContentBlock], None],
    tool_output_callback: Callable[[ToolResult, str], None],
    api_response_callback: Callable[[APIResponse[BetaMessage]], None],
    api_key: str,
    only_n_most_recent_images: int | None = None,
    max_tokens: int = 4096,
):
    """
    Agentic sampling loop for the assistant/tool interaction of computer use.
    """
    tool_collection = ToolCollection(
        ComputerTool(),
        BashTool(),
        EditTool(),
    )
    system = (
        f"{SYSTEM_PROMPT}{' ' + system_prompt_suffix if system_prompt_suffix else ''}"
    )

    while True:
        if only_n_most_recent_images:
            _maybe_filter_to_n_most_recent_images(messages, only_n_most_recent_images)

        # Log the complete API request structure
        logger.info("\n=== API Request Structure ===")
        logger.info(truncate_string(json.dumps({
            "provider": provider,
            "model": model,
            "system": system,
            "messages": messages,
            "tools": tool_collection.to_params(),
            "max_tokens": max_tokens
        }, indent=2, default=str)))

        # Call the API
        if provider == APIProvider.ANTHROPIC:
            raw_response = Anthropic(
                api_key=api_key
            ).beta.messages.with_raw_response.create(
                max_tokens=max_tokens,
                messages=messages,
                model=model,
                system=system,
                tools=cast(list[BetaToolParam], tool_collection.to_params()),
                extra_headers={"anthropic-beta": BETA_FLAG},
            )
        elif provider == APIProvider.VERTEX:
            raw_response = AnthropicVertex().messages.with_raw_response.create(
                max_tokens=max_tokens,
                messages=cast(list[MessageParam], messages),
                model=model,
                system=system,
                tools=cast(list[ToolParam], tool_collection.to_params()),
                extra_headers={"anthropic-beta": BETA_FLAG},
            )
        elif provider == APIProvider.BEDROCK:
            raw_response = AnthropicBedrock().messages.with_raw_response.create(
                max_tokens=max_tokens,
                messages=cast(list[MessageParam], messages),
                model=model,
                system=system,
                tools=cast(list[ToolParam], tool_collection.to_params()),
                extra_body={"anthropic_beta": [BETA_FLAG]},
            )
        elif provider == APIProvider.VENICE:  # Add Venice logic
            venice_client = VeniceClient(model=model, max_tokens=max_tokens)
            raw_response = venice_client.create(messages=messages, system=system, tools=tool_collection.to_params())

        api_response_callback(cast(APIResponse[BetaMessage], raw_response))

        response = raw_response.parse()

        # Log complete API response structure
        logger.info("\n=== API Response Structure ===")
        logger.info("Raw Response Headers:")
        logger.info(truncate_string(json.dumps(dict(raw_response.headers), indent=2)))
        logger.info("\nParsed Response:")
        logger.info(truncate_string(json.dumps({
            "id": response.id,
            "type": response.type,
            "role": response.role,
            "content": response.content,
            "model": response.model,
            "usage": response.usage if hasattr(response, "usage") else None,
            "system": getattr(response, "system", None),
            "stop_reason": getattr(response, "stop_reason", None),
            "stop_sequence": getattr(response, "stop_sequence", None)
        }, indent=2, default=str)))

        messages.append(
            {
                "role": "assistant",
                "content": cast(list[BetaContentBlockParam], response.content),
            }
        )

        tool_result_content: list[BetaToolResultBlockParam] = []
        for content_block in cast(list[BetaContentBlock], response.content):
            output_callback(content_block)
            if content_block.type == "tool_use":
                result = await tool_collection.run(
                    name=content_block.name,
                    tool_input=cast(dict[str, Any], content_block.input),
                )
                tool_result_content.append(
                    _make_api_tool_result(result, content_block.id)
                )
                tool_output_callback(result, content_block.id)

        if not tool_result_content:
            return messages

        messages.append({"content": tool_result_content, "role": "user"})


def _maybe_filter_to_n_most_recent_images(
    messages: list[BetaMessageParam],
    images_to_keep: int,
    min_removal_threshold: int = 10,
):
    """
    With the assumption that images are screenshots that are of diminishing value as
    the conversation progresses, remove all but the final `images_to_keep` tool_result
    images in place, with a chunk of min_removal_threshold to reduce the amount we
    break the implicit prompt cache.
    """
    if images_to_keep is None:
        return messages

    tool_result_blocks = cast(
        list[ToolResultBlockParam],
        [
            item
            for message in messages
            for item in (
                message["content"] if isinstance(message["content"], list) else []
            )
            if isinstance(item, dict) and item.get("type") == "tool_result"
        ],
    )

    total_images = sum(
        1
        for tool_result in tool_result_blocks
        for content in tool_result.get("content", [])
        if isinstance(content, dict) and content.get("type") == "image"
    )

    images_to_remove = total_images - images_to_keep
    # for better cache behavior, we want to remove in chunks
    images_to_remove -= images_to_remove % min_removal_threshold

    for tool_result in tool_result_blocks:
        if isinstance(tool_result.get("content"), list):
            new_content = []
            for content in tool_result.get("content", []):
                if isinstance(content, dict) and content.get("type") == "image":
                    if images_to_remove > 0:
                        images_to_remove -= 1
                        continue
                new_content.append(content)
            tool_result["content"] = new_content


def _make_api_tool_result(
    result: ToolResult, tool_use_id: str
) -> BetaToolResultBlockParam:
    """Convert an agent ToolResult to an API ToolResultBlockParam."""
    tool_result_content: list[BetaTextBlockParam | BetaImageBlockParam] | str = []
    is_error = False
    if result.error:
        is_error = True
        tool_result_content = _maybe_prepend_system_tool_result(result, result.error)
    else:
        if result.output:
            tool_result_content.append(
                {
                    "type": "text",
                    "text": _maybe_prepend_system_tool_result(result, result.output),
                }
            )
        if result.base64_image:
            tool_result_content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": result.base64_image,
                    },
                }
            )
    return {
        "type": "tool_result",
        "content": tool_result_content,
        "tool_use_id": tool_use_id,
        "is_error": is_error,
    }


def _maybe_prepend_system_tool_result(result: ToolResult, result_text: str):
    if result.system:
        result_text = f"<system>{result.system}</system>\n{result_text}"
    return truncate_string(result_text)