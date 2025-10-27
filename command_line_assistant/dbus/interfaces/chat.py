"""D-Bus interfaces that defines and powers our commands."""

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Optional

from dasbus.server.interface import dbus_interface
from dasbus.server.template import InterfaceTemplate
from dasbus.typing import Str, Structure

from command_line_assistant.constants import VERSION
from command_line_assistant.daemon.database.manager import DatabaseManager
from command_line_assistant.daemon.database.repository.chat import ChatRepository
from command_line_assistant.daemon.http.openai_query import convert_mcp_tools_to_openai_format
from command_line_assistant.daemon.http.query import submit
from command_line_assistant.daemon.mcp.client import MCPManager
from command_line_assistant.daemon.session import UserSessionManager
from command_line_assistant.dbus.constants import CHAT_IDENTIFIER
from command_line_assistant.dbus.context import DaemonContext
from command_line_assistant.dbus.exceptions import ChatNotFoundError
from command_line_assistant.dbus.interfaces.authorization import DBusAuthorizationMixin
from command_line_assistant.dbus.sender_context import get_current_sender
from command_line_assistant.dbus.structures.chat import (
    ChatEntry,
    ChatList,
    Question,
    Response,
)

logger = logging.getLogger(__name__)


@dataclass
class InferencePayload:
    """The payload to be submitted to the backend for inference."""

    content: Question

    def to_dict(self) -> dict[str, Any]:
        """Turn content into dictionary for submission to the backend.

        Returns:
            dict[str, Any]: The content in dictionary format.
        """
        return {
            "question": self.content.message,
            "context": {
                "stdin": self.content.stdin.stdin,
                "attachments": {
                    "contents": self.content.attachment.contents,
                    "mimetype": self.content.attachment.mimetype,
                },
                "terminal": {"output": self.content.terminal.output},
                "systeminfo": {
                    "os": self.content.systeminfo.os,
                    "version": self.content.systeminfo.version,
                    "arch": self.content.systeminfo.arch,
                    "id": self.content.systeminfo.id,
                },
                "cla": {"version": VERSION},
            },
        }


@dbus_interface(CHAT_IDENTIFIER.interface_name)
class ChatInterface(InterfaceTemplate, DBusAuthorizationMixin):
    """The DBus interface of a query."""

    def __init__(self, implementation: DaemonContext):
        """Constructor of the class

        Arguments:
            implementation (DaemonContext): The implementation context to be used in an interface.
        """
        super().__init__(implementation)

        self._db_manager = DatabaseManager(implementation.config)
        self._chat_repository = ChatRepository(self._db_manager)
        self._session_manager = UserSessionManager()
        self._mcp_manager: Optional[MCPManager] = None

        # Initialize MCP manager if enabled
        if implementation.config.mcp.enabled:
            logger.info("Initializing MCP manager")
            self._mcp_manager = MCPManager(implementation.config.mcp.servers)
            # Start MCP servers asynchronously
            asyncio.run(self._mcp_manager.start_all())

    def _verify_caller_authorization(self, sender: str, requested_user_id: str) -> None:
        """Verify that the caller is authorized to access the requested user's data.

        Arguments:
            sender: The D-Bus sender.
            requested_user_id (str): The user ID being requested in the method call.

        Raises:
            PermissionError: If the caller's user ID doesn't match the requested user ID.
        """
        self._verify_internal_user_authorization(
            sender, requested_user_id, self._session_manager
        )

    def GetAllChatFromUser(self, user_id: Str) -> Structure:
        """Get all the chat session for a given user.

        Arguments:
            user_id (Str): The identifier of the user.

        Returns:
            Structure: The list of chat sessions.
        """
        # Verify caller authorization
        sender = get_current_sender()
        self._verify_caller_authorization(sender, user_id)
        result = self._chat_repository.select_all_by_user_id(user_id)

        chat_entries = []
        for entry in result:
            chat_entries.append(
                ChatEntry(
                    str(entry.id),
                    entry.name,
                    entry.description,
                    str(entry.created_at),
                    str(entry.updated_at),
                    str(entry.deleted_at),
                )
            )

        return ChatList(chat_entries).structure()

    def DeleteAllChatForUser(self, user_id: Str) -> None:
        """Delete all chats from the user.

        Arguments:
            user_id (Str): The identifier of the user.

        Raises:
            ChatNotFoundError: In case no chat was found for the current user.
        """
        # Verify caller authorization
        sender = get_current_sender()
        self._verify_caller_authorization(sender, user_id)
        all_chats = self._chat_repository.select_all_by_user_id(user_id)

        if not all_chats:
            raise ChatNotFoundError("No chat found to delete.")

        for chat in all_chats:
            logger.info(
                "Deleting chat for user.",
                extra={"audit": True, "chat_id": chat.id, "user_id": user_id},
            )
            self._chat_repository.delete(chat.id)

    def DeleteChatForUser(self, user_id: Str, name: Str) -> None:
        """Delete a specific chat for a user.

        Arguments:
            user_id (Str): The identifier of the user.
            name (Str): The name of the chat.

        Raises:
            ChatNotFoundError: In case no chat was found with the given name for the current user.
        """
        # Verify caller authorization
        sender = get_current_sender()
        self._verify_caller_authorization(sender, user_id)
        logger.info(
            "Looking for chat associated with the user.",
            extra={"audit": True, "chat_name": name, "user_id": user_id},
        )
        chat = self._chat_repository.select_by_name(user_id, name)

        if not chat:
            logger.info(
                "Couldn't find chat with name '%s' for user '%s'. Either the name is not correct or the chat does not exist.",
                name,
                user_id,
            )
            raise ChatNotFoundError(
                f"Couldn't find chat with name '{name}'. Check the name requested and try again."
            )

        logger.info(
            "Deleting the request chat for user.",
            extra={"audit": True, "chat_id": chat[0].id, "user_id": user_id},
        )
        self._chat_repository.delete(chat[0].id)

    def GetLatestChatFromUser(self, user_id: Str) -> Str:
        """Get the latest chat session for a given user.

        Arguments:
            user_id (Str): The identifier of the user.

        Returns:
            Str: The identifier of the chat session.
        """
        # Verify caller authorization
        sender = get_current_sender()
        self._verify_caller_authorization(sender, user_id)
        result = self._chat_repository.select_latest_chat(user_id)
        return str(result.id)

    def IsChatAvailable(self, user_id: Str, name: Str) -> bool:
        """Check if a chat session is available for a given user.

        Arguments:
            user_id (Str): The identifier of the user.
            name (Str): The name of the chat.

        Returns:
            bool: True if the chat session is available, False otherwise.
        """
        # Verify caller authorization
        sender = get_current_sender()
        self._verify_caller_authorization(sender, user_id)
        result = self._chat_repository.select_by_name(user_id, name)

        if not result:
            logger.info(
                "Chat session with name '%s' not found for user '%s'.",
                name,
                user_id,
            )
            return False

        logger.info(
            "Chat session with name '%s' found for user '%s'.",
            name,
            user_id,
        )
        return True

    def GetChatId(self, user_id: Str, name: Str) -> Str:
        """Get the chat id for a given user and chat name.

        Arguments:
            user_id (Str): The identifier of the user.
            name (Str): The name of the chat.

        Raises:
            ChatNotFound: If the chat is not found.

        Returns:
            Str: The identifier of the chat session.
        """
        # Verify caller authorization
        sender = get_current_sender()
        self._verify_caller_authorization(sender, user_id)

        result = self._chat_repository.select_by_name(user_id, name)

        if not result:
            raise ChatNotFoundError(
                f"No chat found with name '{name}'. Please, make sure that this chat exist first."
            )

        logger.info(
            "Found existing chat with id '%s' and name '%s' for user '%s'",
            result[0].id,
            name,
            user_id,
        )
        return str(result[0].id)

    def CreateChat(self, user_id: Str, name: Str, description: Str) -> Str:
        """Create a new chat session for a given conversation.

        Arguments:
            user_id (Str): The identifier of the user.
            name (Str): The name of the chat. If not specified, a random name will be given.
            description (Str): A description for the chat to identify the context of it.

        Returns:
            Str: The identifier of the chat session.
        """
        # Verify caller authorization
        sender = get_current_sender()
        self._verify_caller_authorization(sender, user_id)
        identifier = self._chat_repository.insert(
            {"user_id": user_id, "name": name, "description": description}
        )
        logger.info(
            "New chat session created for user.",
            extra={"audit": True, "identifier": identifier, "chat_name": name},
        )
        return str(identifier[0])

    def IsMCPEnabled(self) -> bool:
        """Check if MCP is enabled in the daemon configuration.

        Returns:
            bool: True if MCP is enabled, False otherwise.
        """
        config = self.implementation.config
        return config.mcp.enabled and config.backend.mode == "openai"

    def AskQuestion(self, user_id: Str, message_input: Structure) -> Structure:
        """This method is mainly called by the client to retrieve it's answer.

        Arguments:
            user_id (Str): The identifier of the user.
            message_input (Structure): The message input in format of a d-bus structure.

        Returns:
            Structure: The message output in format of a d-bus structure.
        """
        # Verify caller authorization
        sender = get_current_sender()
        self._verify_caller_authorization(sender, user_id)
        
        # Submit query to backend
        content = Question.from_structure(message_input)
        payload = InferencePayload(content)

        logger.info(
            "Submitting question from user.",
            extra={
                "audit": True,
                "user": user_id,
            },
        )

        # If MCP is enabled and we're using OpenAI mode, use the tool calling loop
        config = self.implementation.config
        if config.mcp.enabled and config.backend.mode == "openai" and self._mcp_manager:
            logger.info("Using MCP tool calling loop")
            llm_response_text = asyncio.run(
                self._handle_tool_calling_loop(payload.to_dict())
            )
        else:
            # Legacy mode or MCP disabled - simple query
            logger.info(f"Using {config.backend.mode} mode (MCP disabled or not available)")
            llm_response = submit(payload.to_dict(), config)
            llm_response_text = self._extract_response_text(llm_response)

        logger.info(f"Response text length: {len(llm_response_text)} chars")
        logger.debug(f"Response text preview: {llm_response_text[:200]}...")

        # Create message object
        response = Response(llm_response_text)

        # Return the data
        return response.structure()

    def _extract_response_text(self, response: dict) -> str:
        """Extract text from response based on backend mode.

        Args:
            response: Response dictionary from submit()

        Returns:
            str: The extracted text
        """
        config = self.implementation.config
        
        if config.backend.mode == "openai":
            # OpenAI format
            choices = response.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                content = message.get("content")
                # Handle None by converting to empty string
                return content if content is not None else ""
            return ""
        else:
            # Legacy format
            text = response.get("text", "")
            return text if text is not None else ""

    async def _handle_tool_calling_loop(self, payload: dict) -> str:
        """Handle the tool calling loop for MCP integration.

        This implements the agent pattern:
        1. Send query to LLM with available tools
        2. If LLM requests tool call, execute it
        3. Send tool results back to LLM
        4. Repeat until LLM provides final answer

        Args:
            payload: The original query payload

        Returns:
            str: The final response text from the LLM
        """
        config = self.implementation.config
        max_iterations = 10  # Prevent infinite loops

        # Get available MCP tools
        mcp_tools = self._mcp_manager.get_all_tools()
        openai_tools = convert_mcp_tools_to_openai_format(mcp_tools)

        logger.info(f"Starting tool calling loop with {len(openai_tools)} available tools")

        # Submit initial query with tools
        response = submit(payload, config, tools=openai_tools)

        for iteration in range(max_iterations):
            choices = response.get("choices", [])
            if not choices:
                logger.warning("No choices in response")
                return ""

            message = choices[0].get("message", {})
            finish_reason = choices[0].get("finish_reason")

            # Check if we're done
            if finish_reason == "stop":
                # LLM provided final answer
                return message.get("content", "")

            # Check if LLM wants to call tools
            if finish_reason == "tool_calls":
                tool_calls = message.get("tool_calls", [])
                if not tool_calls:
                    logger.warning("finish_reason is tool_calls but no tool_calls found")
                    return message.get("content", "")

                logger.info(f"LLM requested {len(tool_calls)} tool call(s)")

                # Execute tool calls
                tool_results = []
                for tool_call in tool_calls:
                    tool_call_id = tool_call.get("id")
                    function = tool_call.get("function", {})
                    tool_name = function.get("name")
                    arguments_str = function.get("arguments", "{}")

                    try:
                        arguments = json.loads(arguments_str)
                        logger.info(f"Calling tool: {tool_name}")

                        # Call the MCP tool
                        result = await self._mcp_manager.call_tool(tool_name, arguments)

                        tool_results.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "content": json.dumps(result),
                            }
                        )
                    except Exception as e:
                        logger.error(f"Error calling tool {tool_name}: {e}")
                        tool_results.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "content": json.dumps({"error": str(e)}),
                            }
                        )

                # Prepare next request with tool results
                # We need to reconstruct the messages list
                # For simplicity, we'll use the original question and add tool context
                question = payload.get("question", "")
                context = payload.get("context", {})
                
                # Create a summary of tool calls for context
                tool_summary = f"\nTool calls executed:\n"
                for i, tc in enumerate(tool_calls):
                    tool_summary += f"- {tc.get('function', {}).get('name')}\n"
                
                # Update the question with tool context
                enhanced_payload = payload.copy()
                enhanced_payload["question"] = f"{question}\n{tool_summary}\nResults: {json.dumps(tool_results)}"
                
                # Submit again with tool results
                response = submit(enhanced_payload, config, tools=openai_tools)
            else:
                # Unknown finish reason
                logger.warning(f"Unknown finish_reason: {finish_reason}")
                return message.get("content", "")

        # Max iterations reached
        logger.warning("Max iterations reached in tool calling loop")
        return "I apologize, but I've reached the maximum number of tool calls. Please try rephrasing your question."
