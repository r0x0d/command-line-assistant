"""Schemas for the MCP (Model Context Protocol) config."""

import dataclasses
import logging
from typing import Optional

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class MCPServerSchema:
    """Schema for a single MCP server configuration.

    Attributes:
        name (str): The name of the MCP server
        command (str): The command to execute to start the MCP server
        args (list[str]): Arguments to pass to the MCP server command
        env (dict[str, str]): Environment variables to set for the MCP server
    """

    name: str
    command: str
    args: list[str] = dataclasses.field(default_factory=list)
    env: dict[str, str] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        """Post initialization method to normalize values."""
        if not self.name:
            raise ValueError("MCP server name cannot be empty")
        if not self.command:
            raise ValueError("MCP server command cannot be empty")


@dataclasses.dataclass
class MCPSchema:
    """This class represents the [mcp] section of our config.toml file.

    Attributes:
        enabled (bool): Whether MCP support is enabled
        servers (list[MCPServerSchema]): List of MCP servers to make available
    """

    enabled: bool = False
    servers: list[MCPServerSchema] = dataclasses.field(default_factory=list)

    def __post_init__(self) -> None:
        """Post initialization method to normalize values."""
        # Convert dict entries to MCPServerSchema objects
        if self.servers and isinstance(self.servers, list):
            normalized_servers = []
            for server in self.servers:
                if isinstance(server, dict):
                    normalized_servers.append(MCPServerSchema(**server))
                else:
                    normalized_servers.append(server)
            self.servers = normalized_servers

        logger.info(f"MCP support is {'enabled' if self.enabled else 'disabled'}")
        if self.enabled:
            logger.info(f"Loaded {len(self.servers)} MCP server(s)")

