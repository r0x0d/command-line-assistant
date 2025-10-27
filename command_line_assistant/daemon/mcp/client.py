"""MCP client implementation for communicating with MCP servers."""

import asyncio
import json
import logging
import subprocess
from dataclasses import dataclass
from typing import Any, Optional

from command_line_assistant.config.schemas.mcp import MCPServerSchema

logger = logging.getLogger(__name__)


@dataclass
class MCPTool:
    """Represents a tool provided by an MCP server.

    Attributes:
        name (str): The name of the tool
        description (str): Description of what the tool does
        input_schema (dict): JSON schema for the tool's input parameters
        server_name (str): Name of the MCP server providing this tool
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    server_name: str


class MCPClient:
    """Client for communicating with MCP servers.

    This client handles starting MCP servers, discovering their tools,
    and executing tool calls via stdio communication.
    """

    def __init__(self, server_config: MCPServerSchema):
        """Initialize the MCP client.

        Args:
            server_config: Configuration for the MCP server
        """
        self.server_config = server_config
        self.process: Optional[subprocess.Popen] = None
        self.tools: list[MCPTool] = []
        self._request_id = 0

    def _get_next_request_id(self) -> int:
        """Get the next request ID.

        Returns:
            int: The next request ID
        """
        self._request_id += 1
        return self._request_id

    async def start(self) -> None:
        """Start the MCP server process.

        Raises:
            RuntimeError: If the server fails to start
        """
        try:
            logger.info(f"Starting MCP server '{self.server_config.name}'")
            env = {**subprocess.os.environ, **self.server_config.env}

            self.process = subprocess.Popen(
                [self.server_config.command] + self.server_config.args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True,
                bufsize=1,  # Line buffered
            )

            # Initialize the connection
            await self._send_initialize()
            await self._discover_tools()

            logger.info(
                f"MCP server '{self.server_config.name}' started with {len(self.tools)} tools"
            )
        except Exception as e:
            logger.error(f"Failed to start MCP server '{self.server_config.name}': {e}")
            raise RuntimeError(
                f"Failed to start MCP server '{self.server_config.name}': {e}"
            ) from e

    async def _send_initialize(self) -> None:
        """Send initialization request to the MCP server."""
        request = {
            "jsonrpc": "2.0",
            "id": self._get_next_request_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "command-line-assistant",
                    "version": "0.4.2",
                },
            },
        }

        response = await self._send_request(request)
        if "error" in response:
            raise RuntimeError(f"MCP initialization failed: {response['error']}")

        # Send initialized notification
        notification = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        }
        await self._send_notification(notification)

    async def _discover_tools(self) -> None:
        """Discover available tools from the MCP server."""
        request = {
            "jsonrpc": "2.0",
            "id": self._get_next_request_id(),
            "method": "tools/list",
        }

        response = await self._send_request(request)
        if "error" in response:
            logger.warning(
                f"Failed to list tools for '{self.server_config.name}': {response['error']}"
            )
            return

        tools_list = response.get("result", {}).get("tools", [])
        for tool_data in tools_list:
            tool = MCPTool(
                name=tool_data["name"],
                description=tool_data.get("description", ""),
                input_schema=tool_data.get("inputSchema", {}),
                server_name=self.server_config.name,
            )
            self.tools.append(tool)
            logger.debug(f"Discovered tool: {tool.name} from {self.server_config.name}")

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict:
        """Call a tool on the MCP server.

        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool

        Returns:
            dict: The result from the tool execution

        Raises:
            ValueError: If the tool is not found
            RuntimeError: If the tool call fails
        """
        # Verify the tool exists
        tool = next((t for t in self.tools if t.name == tool_name), None)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found on server '{self.server_config.name}'")

        logger.info(f"Calling tool '{tool_name}' on server '{self.server_config.name}'")

        request = {
            "jsonrpc": "2.0",
            "id": self._get_next_request_id(),
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        }

        response = await self._send_request(request)
        if "error" in response:
            error_msg = response["error"].get("message", "Unknown error")
            raise RuntimeError(f"Tool call failed: {error_msg}")

        return response.get("result", {})

    async def _send_request(self, request: dict) -> dict:
        """Send a JSON-RPC request and wait for a response.

        Args:
            request: The JSON-RPC request to send

        Returns:
            dict: The JSON-RPC response

        Raises:
            RuntimeError: If the server is not running or communication fails
        """
        if not self.process or not self.process.stdin or not self.process.stdout:
            raise RuntimeError("MCP server is not running")

        try:
            # Send request
            request_str = json.dumps(request) + "\n"
            self.process.stdin.write(request_str)
            self.process.stdin.flush()

            # Read response
            response_str = self.process.stdout.readline()
            if not response_str:
                raise RuntimeError("MCP server closed connection")

            response = json.loads(response_str)
            return response
        except Exception as e:
            logger.error(f"MCP communication error: {e}")
            raise RuntimeError(f"MCP communication error: {e}") from e

    async def _send_notification(self, notification: dict) -> None:
        """Send a JSON-RPC notification (no response expected).

        Args:
            notification: The JSON-RPC notification to send
        """
        if not self.process or not self.process.stdin:
            raise RuntimeError("MCP server is not running")

        notification_str = json.dumps(notification) + "\n"
        self.process.stdin.write(notification_str)
        self.process.stdin.flush()

    async def stop(self) -> None:
        """Stop the MCP server process."""
        if self.process:
            logger.info(f"Stopping MCP server '{self.server_config.name}'")
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning(
                    f"MCP server '{self.server_config.name}' did not terminate, killing it"
                )
                self.process.kill()
            self.process = None

    def get_tools(self) -> list[MCPTool]:
        """Get the list of available tools.

        Returns:
            list[MCPTool]: List of tools provided by this server
        """
        return self.tools

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        asyncio.run(self.stop())


class MCPManager:
    """Manager for multiple MCP clients."""

    def __init__(self, server_configs: list[MCPServerSchema]):
        """Initialize the MCP manager.

        Args:
            server_configs: List of MCP server configurations
        """
        self.clients: list[MCPClient] = []
        self.server_configs = server_configs

    async def start_all(self) -> None:
        """Start all configured MCP servers."""
        for server_config in self.server_configs:
            try:
                client = MCPClient(server_config)
                await client.start()
                self.clients.append(client)
            except Exception as e:
                logger.error(f"Failed to start MCP server '{server_config.name}': {e}")

    async def stop_all(self) -> None:
        """Stop all MCP servers."""
        for client in self.clients:
            await client.stop()
        self.clients = []

    def get_all_tools(self) -> list[MCPTool]:
        """Get all available tools from all servers.

        Returns:
            list[MCPTool]: List of all tools from all servers
        """
        all_tools = []
        for client in self.clients:
            all_tools.extend(client.get_tools())
        return all_tools

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict:
        """Call a tool by name across all servers.

        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool

        Returns:
            dict: The result from the tool execution

        Raises:
            ValueError: If the tool is not found
        """
        # Find the server that has this tool
        for client in self.clients:
            for tool in client.get_tools():
                if tool.name == tool_name:
                    return await client.call_tool(tool_name, arguments)

        raise ValueError(f"Tool '{tool_name}' not found in any MCP server")

    def __enter__(self):
        """Context manager entry."""
        asyncio.run(self.start_all())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        asyncio.run(self.stop_all())

