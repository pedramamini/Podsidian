"""
STDIO server implementation for Podsidian MCP.

This module provides functionality to run the Podsidian MCP service in STDIO mode,
allowing direct integration with AI agents like Claude Desktop.
"""

import sys
import json
import asyncio
from typing import Dict, Any, Optional, List, Callable
from fastapi import FastAPI
from fastapi.routing import APIRoute

# Standard message format for STDIO communication
class StdioMessage:
    def __init__(self, message_type: str, data: Dict[str, Any]):
        self.type = message_type
        self.data = data
    
    def to_json(self) -> str:
        return json.dumps({
            "type": self.type,
            "data": self.data
        })


class StdioServer:
    """STDIO server for Podsidian MCP.
    
    Handles communication via stdin/stdout with JSON messages.
    """
    
    def __init__(self, app: FastAPI):
        """Initialize STDIO server with FastAPI app.
        
        Args:
            app: FastAPI application with routes
        """
        self.app = app
        self.tools = {}
        self._setup_tools()
        
    def _setup_tools(self):
        """Set up tools based on FastAPI routes."""
        print("Registering tools...", file=sys.stderr)
        
        # Extract routes from FastAPI app
        for route in self.app.routes:
            if isinstance(route, APIRoute):
                # Skip the initialize endpoint as it's not a tool
                if route.path == "/initialize":
                    continue
                    
                # Create tool name from path
                # Convert /api/v1/search/semantic to search-semantic
                path_parts = route.path.strip('/').split('/')
                if path_parts and path_parts[0] == "api":
                    path_parts = path_parts[2:]  # Skip api/v1
                
                tool_name = '-'.join(path_parts)
                
                if not tool_name:
                    continue
                
                # Register tool
                self.tools[tool_name] = {
                    "path": route.path,
                    "method": route.methods.pop() if route.methods else "GET",
                    "summary": route.summary or f"Call {route.path}",
                    "description": route.description or f"Call {route.path} endpoint"
                }
                
                print(f"Registered tool: {tool_name}", file=sys.stderr)
        
        print(f"All tools registered successfully", file=sys.stderr)
    
    async def _handle_message(self, message: Dict[str, Any]):
        """Handle incoming message from stdin.
        
        Args:
            message: Parsed JSON message
        """
        if message.get("type") == "tool_call":
            await self._handle_tool_call(message.get("data", {}))
        elif message.get("type") == "initialize":
            await self._handle_initialize()
    
    async def _handle_initialize(self):
        """Handle initialize request."""
        # Get capabilities from the /initialize endpoint
        capabilities = {
            "serverInfo": {
                "name": "Podsidian MCP",
                "version": "1.0.0",
                "capabilities": ["search", "transcribe", "summarize"]
            },
            "tools": {}
        }
        
        # Add tools
        for tool_name, tool_info in self.tools.items():
            capabilities["tools"][tool_name] = {
                "description": tool_info["description"],
                "parameters": {}  # We could extract these from FastAPI but keeping it simple
            }
        
        # Send response
        response = StdioMessage("initialize_response", capabilities)
        print(response.to_json(), flush=True)
    
    async def _handle_tool_call(self, data: Dict[str, Any]):
        """Handle tool call request.
        
        Args:
            data: Tool call data including name and parameters
        """
        tool_name = data.get("name", "")
        parameters = data.get("parameters", {})
        call_id = data.get("id", "unknown")
        
        if tool_name not in self.tools:
            error_response = StdioMessage("tool_call_error", {
                "id": call_id,
                "error": f"Unknown tool: {tool_name}"
            })
            print(error_response.to_json(), flush=True)
            return
        
        tool_info = self.tools[tool_name]
        path = tool_info["path"]
        method = tool_info["method"]
        
        # TODO: Actually call the FastAPI endpoint
        # This is a simplified implementation that would need to be expanded
        # to properly call the FastAPI endpoints with the provided parameters
        
        # For now, just return a mock response
        response = StdioMessage("tool_call_response", {
            "id": call_id,
            "result": {
                "status": "success",
                "message": f"Called {tool_name} with parameters {parameters}"
            }
        })
        print(response.to_json(), flush=True)
    
    async def run(self):
        """Run the STDIO server loop."""
        print("Server starting...", file=sys.stderr)
        print("\nPodsidian MCP Server running on stdio", file=sys.stderr)
        
        # Main loop
        while True:
            try:
                # Read a line from stdin
                line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
                
                # Check for EOF
                if not line:
                    break
                
                # Parse JSON message
                try:
                    message = json.loads(line)
                    await self._handle_message(message)
                except json.JSONDecodeError:
                    error_response = StdioMessage("error", {
                        "message": "Invalid JSON message"
                    })
                    print(error_response.to_json(), flush=True)
            except Exception as e:
                error_response = StdioMessage("error", {
                    "message": f"Error processing message: {str(e)}"
                })
                print(error_response.to_json(), flush=True)


async def run_stdio_server(app: FastAPI):
    """Run the STDIO server with the given FastAPI app.
    
    Args:
        app: FastAPI application
    """
    server = StdioServer(app)
    await server.run()
