"""Module to handle OpenAI-compatible API queries (e.g., ramalama)."""

import logging
from http import HTTPStatus
from typing import Any, Optional

from requests import RequestException, Response

from command_line_assistant.config import Config
from command_line_assistant.daemon.http.session import get_session
from command_line_assistant.dbus.exceptions import RequestFailedError

logger = logging.getLogger(__name__)


def submit_openai(
    messages: list[dict[str, str]],
    config: Config,
    tools: Optional[list[dict]] = None,
) -> dict[str, Any]:
    """Submit a query to an OpenAI-compatible API endpoint.

    Args:
        messages: List of message dictionaries with 'role' and 'content'
        config: Configuration object with backend endpoint information
        tools: Optional list of tool definitions in OpenAI format

    Raises:
        RequestFailedError: If the request fails due to network issues,
                           authentication problems, or server errors

    Returns:
        dict: The complete response from the API including choices and tool calls
    """
    query_endpoint = f"{config.backend.endpoint}/v1/chat/completions"

    payload = {
        "model": config.backend.model,
        "messages": messages,
    }

    # Add tools if provided and MCP is enabled
    if tools and config.mcp.enabled:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    try:
        response = _send_openai_request(query_endpoint, payload, config)
        logger.info("Received response from OpenAI-compatible API")

        if response.status_code != HTTPStatus.OK:
            _handle_error_response(response)

        return _extract_openai_response(response)
    except RequestException as exc:
        logger.error("Failed to get response from AI: %s", exc)
        raise RequestFailedError(
            f"Communication error with the server: {str(exc)}. Please try again in a few minutes."
        ) from exc


def _send_openai_request(endpoint: str, payload: dict, config: Config) -> Response:
    """Send POST request to the OpenAI-compatible backend.

    Args:
        endpoint: Full URL endpoint
        payload: Request payload
        config: Configuration with auth settings

    Returns:
        Response object
    """
    with get_session(config) as session:
        return session.post(
            endpoint,
            json=payload,
            timeout=config.backend.timeout,
        )


def _handle_error_response(response: Response) -> None:
    """Check response for errors and raise appropriate exceptions.

    Args:
        response: Response object to check

    Raises:
        RequestFailedError: If response status code indicates an error
    """
    try:
        error_data = response.json()
        error_message = error_data.get("error", {}).get("message", response.reason)
    except Exception:
        error_message = response.reason

    full_error = (
        f"OpenAI API error (status {response.status_code}): {error_message}"
    )
    logger.error(full_error)
    raise RequestFailedError(full_error)


def _extract_openai_response(response: Response) -> dict[str, Any]:
    """Extract data from successful OpenAI API response.

    Args:
        response: Response object with JSON data

    Returns:
        dict containing the full response data

    Raises:
        RequestFailedError: If response doesn't contain valid data
    """
    try:
        data = response.json()
        if "choices" not in data or not data["choices"]:
            raise RequestFailedError("Invalid response: no choices found")

        return data
    except ValueError as e:
        logger.error("Response didn't contain valid JSON")
        raise RequestFailedError("Invalid JSON response from server") from e


def convert_mcp_tools_to_openai_format(mcp_tools: list) -> list[dict]:
    """Convert MCP tool definitions to OpenAI function calling format.

    Args:
        mcp_tools: List of MCPTool objects

    Returns:
        list: Tools in OpenAI format
    """
    openai_tools = []

    for tool in mcp_tools:
        openai_tool = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            },
        }
        openai_tools.append(openai_tool)

    return openai_tools


def create_initial_messages(question: str, context: dict[str, Any]) -> list[dict]:
    """Create the initial message list for the OpenAI API.

    Args:
        question: The user's question
        context: Context information (stdin, attachments, terminal, systeminfo)

    Returns:
        list: List of message dictionaries
    """
    # Build a comprehensive context string
    context_parts = []

    if context.get("stdin"):
        context_parts.append(f"Standard input:\n{context['stdin']}")

    if context.get("attachments", {}).get("contents"):
        attachment_content = context["attachments"]["contents"]
        attachment_mimetype = context["attachments"].get("mimetype", "text/plain")
        context_parts.append(
            f"Attached file ({attachment_mimetype}):\n{attachment_content}"
        )

    if context.get("terminal", {}).get("output"):
        context_parts.append(f"Terminal output:\n{context['terminal']['output']}")

    # Add system info
    systeminfo = context.get("systeminfo", {})
    if systeminfo:
        os_info = f"{systeminfo.get('os', '')} {systeminfo.get('version', '')}".strip()
        arch = systeminfo.get("arch", "")
        if os_info or arch:
            context_parts.append(f"System: {os_info} ({arch})")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful command-line assistant for Linux systems. "
                "Provide clear, accurate, and practical advice. "
                "When suggesting commands, explain what they do."
            ),
        }
    ]

    # If we have context, add it before the user question
    if context_parts:
        context_str = "\n\n".join(context_parts)
        messages.append({"role": "user", "content": f"Context:\n{context_str}"})

    # Add the main question
    messages.append({"role": "user", "content": question})

    return messages

