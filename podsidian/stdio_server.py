#!/usr/bin/env python
"""
MCP-compliant STDIO server for Podsidian.

This server implements the Model Context Protocol (MCP) specification using
JSON-RPC 2.0 protocol over STDIO transport. It provides tools for accessing
Podsidian podcast functionality.

The server exposes the following tools:
- search_semantic: Search podcast transcripts using semantic similarity
- search_keyword: Search podcast transcripts for exact keyword matches
- list_episodes: List processed podcast episodes
- get_episode: Get specific episode details and transcript
- list_subscriptions: List podcast subscriptions
- mute_subscription: Mute a podcast subscription
- unmute_subscription: Unmute a podcast subscription
"""

import asyncio
import json
import sys
import logging
import argparse
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session

from .models import Episode, Podcast
from .core import PodcastProcessor
from .apple_podcasts import get_subscriptions
from .config import get_database_session

# Logger will be configured in main() after parsing arguments
logger = logging.getLogger("mcp_stdio")

# Global database session
db_session: Optional[Session] = None
processor: Optional[PodcastProcessor] = None


def ensure_db_connection() -> tuple[Session, PodcastProcessor]:
    """Ensure database connection is established."""
    global db_session, processor
    if db_session is None or processor is None:
        try:
            db_session = get_database_session()
            processor = PodcastProcessor(db_session)
            logger.info("Connected to Podsidian database")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise Exception(f"Database connection error: {str(e)}")
    return db_session, processor


class MCPServer:
    """MCP Server implementation using JSON-RPC 2.0 over STDIO."""

    def __init__(self):
        self.tools = {
            "search_semantic": {
                "description": "Search podcast transcripts using semantic similarity. Uses AI embeddings to find semantically similar content even if exact words don't match. Perfect for finding episodes about specific topics or concepts.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Natural language search query"},
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results (default: 10)",
                            "default": 10,
                        },
                        "relevance": {
                            "type": "integer",
                            "description": "Minimum relevance score 0-100 (default: 25)",
                            "default": 25,
                        },
                    },
                    "required": ["query"],
                },
            },
            "search_keyword": {
                "description": "Search podcast transcripts for exact keyword matches. Performs case-insensitive exact text matching within transcripts. Use this when you need to find specific words or phrases.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "keyword": {"type": "string", "description": "Text to search for"},
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results (default: 10)",
                            "default": 10,
                        },
                    },
                    "required": ["keyword"],
                },
            },
            "list_episodes": {
                "description": "List processed podcast episodes. Returns a paginated list of episodes that have been transcribed, ordered by publication date (newest first).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of episodes to return (default: 100)",
                            "default": 100,
                        },
                        "offset": {
                            "type": "integer",
                            "description": "Number of episodes to skip (default: 0)",
                            "default": 0,
                        },
                    },
                    "required": [],
                },
            },
            "get_episode": {
                "description": "Get specific episode details and full transcript. Use this to retrieve complete episode information including the full transcript text.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "episode_id": {
                            "type": "integer",
                            "description": "ID of the episode to retrieve",
                        }
                    },
                    "required": ["episode_id"],
                },
            },
            "list_subscriptions": {
                "description": "List all podcast subscriptions with their mute state. Shows information about all podcasts you're subscribed to in Apple Podcasts, along with their processing status in Podsidian.",
                "inputSchema": {"type": "object", "properties": {}, "required": []},
            },
            "mute_subscription": {
                "description": "Mute a podcast subscription by title. Muted podcasts will not be processed during ingestion.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Title of the podcast to mute"}
                    },
                    "required": ["title"],
                },
            },
            "unmute_subscription": {
                "description": "Unmute a podcast subscription by title. Allows the podcast to be processed during ingestion.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Title of the podcast to unmute"}
                    },
                    "required": ["title"],
                },
            },
            "generate_briefing": {
                "description": (
                    "Generate a personalized news briefing from "
                    "recent podcast episodes. Searches across "
                    "interest categories using semantic search, "
                    "then synthesizes a briefing via LLM."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "days": {
                            "type": "integer",
                            "description": "Days to look back (default: 7)",
                            "default": 7,
                        },
                        "categories": {
                            "type": "string",
                            "description": ("Comma-separated custom search categories (optional)"),
                        },
                    },
                    "required": [],
                },
            },
        }

        self.resources = {
            "podsidian://episodes": {
                "name": "Podcast Episodes",
                "description": "Access to podcast episode data",
                "mimeType": "application/json",
            },
            "podsidian://subscriptions": {
                "name": "Podcast Subscriptions",
                "description": "Access to podcast subscription data",
                "mimeType": "application/json",
            },
        }

    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming JSON-RPC 2.0 request."""
        try:
            method = request.get("method")
            params = request.get("params", {})
            request_id = request.get("id")

            logger.info(f"Handling request: {method}")

            if method == "initialize":
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}, "resources": {}},
                        "serverInfo": {"name": "podsidian-mcp", "version": "1.0.0"},
                    },
                }

            elif method == "notifications/initialized":
                # No response needed for notification
                return None

            elif method == "tools/list":
                tools_list = []
                for name, tool_def in self.tools.items():
                    tools_list.append(
                        {
                            "name": name,
                            "description": tool_def["description"],
                            "inputSchema": tool_def["inputSchema"],
                        }
                    )

                return {"jsonrpc": "2.0", "id": request_id, "result": {"tools": tools_list}}

            elif method == "tools/call":
                tool_name = params.get("name")
                arguments = params.get("arguments", {})

                result = await self.call_tool(tool_name, arguments)

                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"content": [{"type": "text", "text": result}]},
                }

            elif method == "resources/list":
                resources_list = []
                for uri, resource_def in self.resources.items():
                    resources_list.append(
                        {
                            "uri": uri,
                            "name": resource_def["name"],
                            "description": resource_def["description"],
                            "mimeType": resource_def["mimeType"],
                        }
                    )

                return {"jsonrpc": "2.0", "id": request_id, "result": {"resources": resources_list}}

            elif method == "resources/read":
                uri = params.get("uri")
                result = await self.get_resource(uri)

                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "contents": [{"uri": uri, "mimeType": "text/plain", "text": result}]
                    },
                }

            else:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
                }

        except Exception as e:
            logger.error(f"Error handling request: {e}", exc_info=True)
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "error": {"code": -32603, "message": f"Internal error: {str(e)}"},
            }

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        """Handle tool calls."""
        db, proc = ensure_db_connection()

        if name == "search_semantic":
            query = arguments["query"]
            limit = arguments.get("limit", 10)
            relevance = arguments.get("relevance", 25)

            # Convert relevance to 0-1 scale
            relevance_float = relevance / 100.0
            results = proc.search(query, limit=limit, relevance_threshold=relevance_float)

            if not results:
                return f"No results found for semantic search: '{query}'"

            response_text = f"Found {len(results)} semantic search results for '{query}':\n\n"
            for i, result in enumerate(results, 1):
                similarity_pct = int(result["similarity"] * 100)
                response_text += f"{i}. {result['podcast']} - {result['episode']}\n"
                response_text += f"   Published: {result['published_at']}\n"
                response_text += f"   Relevance: {similarity_pct}%\n"
                response_text += f"   Excerpt: {result.get('excerpt', '')[:200]}...\n\n"

            return response_text

        elif name == "search_keyword":
            keyword = arguments["keyword"]
            limit = arguments.get("limit", 10)

            # Perform keyword search using database query
            episodes = (
                db.query(Episode)
                .filter(Episode.transcript.ilike(f"%{keyword}%"))
                .limit(limit)
                .all()
            )

            if not episodes:
                return f"No results found for keyword search: '{keyword}'"

            results = []
            for episode in episodes:
                # Find excerpt around keyword
                transcript = episode.transcript or ""
                keyword_lower = keyword.lower()
                transcript_lower = transcript.lower()

                # Find first occurrence of keyword
                pos = transcript_lower.find(keyword_lower)
                if pos != -1:
                    # Extract context around keyword
                    start = max(0, pos - 150)
                    end = min(len(transcript), pos + 150)
                    excerpt = transcript[start:end]

                    # Add ellipsis if truncated
                    if start > 0:
                        excerpt = "..." + excerpt
                    if end < len(transcript):
                        excerpt = excerpt + "..."

                    results.append(
                        {
                            "podcast": episode.podcast.title,
                            "episode": episode.title,
                            "published_at": episode.published_at,
                            "excerpt": excerpt,
                        }
                    )

            response_text = f"Found {len(results)} keyword search results for '{keyword}':\n\n"
            for i, result in enumerate(results, 1):
                response_text += f"{i}. {result['podcast']} - {result['episode']}\n"
                response_text += f"   Published: {result['published_at']}\n"
                response_text += f"   Excerpt: {result['excerpt']}\n\n"

            return response_text

        elif name == "list_episodes":
            limit = arguments.get("limit", 100)
            offset = arguments.get("offset", 0)

            episodes = (
                db.query(Episode)
                .order_by(Episode.published_at.desc())
                .offset(offset)
                .limit(limit)
                .all()
            )

            if not episodes:
                return "No episodes found in the database."

            response_text = f"Found {len(episodes)} episodes (showing {offset + 1}-{offset + len(episodes)}):\n\n"
            for i, episode in enumerate(episodes, offset + 1):
                has_transcript = "✓" if episode.transcript else "✗"
                response_text += f"{i}. {episode.podcast.title} - {episode.title}\n"
                response_text += f"   Published: {episode.published_at}\n"
                response_text += f"   Transcript: {has_transcript}\n"
                response_text += f"   ID: {episode.id}\n\n"

            return response_text

        elif name == "get_episode":
            episode_id = arguments["episode_id"]

            episode = db.query(Episode).filter_by(id=episode_id).first()
            if not episode:
                return f"Episode with ID {episode_id} not found."

            response_text = f"Episode Details:\n\n"
            response_text += f"Podcast: {episode.podcast.title}\n"
            response_text += f"Title: {episode.title}\n"
            response_text += f"Published: {episode.published_at}\n"
            response_text += f"Description: {episode.description}\n\n"

            if episode.transcript:
                response_text += f"Transcript:\n{episode.transcript}\n"
            else:
                response_text += "No transcript available for this episode.\n"

            return response_text

        elif name == "list_subscriptions":
            # Get subscriptions from Apple Podcasts
            subs = get_subscriptions()
            if not subs:
                return "No podcast subscriptions found in Apple Podcasts."

            # Ensure all podcasts exist in database
            for sub in subs:
                podcast = db.query(Podcast).filter_by(feed_url=sub["feed_url"]).first()
                if not podcast:
                    podcast = Podcast(
                        title=sub["title"],
                        author=sub["author"],
                        feed_url=sub["feed_url"],
                        muted=False,
                    )
                    db.add(podcast)
            db.commit()

            # Get mute states from database
            muted_feeds = {p.feed_url: p.muted for p in db.query(Podcast).all()}

            # Get episode counts for each podcast
            from sqlalchemy import func

            episode_counts = dict(
                db.query(Podcast.feed_url, func.count(Episode.id).label("count"))
                .join(Episode, isouter=True)
                .group_by(Podcast.feed_url)
                .all()
            )

            response_text = f"Found {len(subs)} podcast subscriptions:\n\n"
            for i, sub in enumerate(subs, 1):
                muted = muted_feeds.get(sub["feed_url"], False)
                episode_count = episode_counts.get(sub["feed_url"], 0)
                status = "MUTED" if muted else "ACTIVE"

                response_text += f"{i}. {sub['title']}\n"
                response_text += f"   Author: {sub['author']}\n"
                response_text += f"   Status: {status}\n"
                response_text += f"   Episodes in DB: {episode_count}\n\n"

            return response_text

        elif name == "mute_subscription":
            title = arguments["title"]

            podcast = db.query(Podcast).filter_by(title=title).first()
            if not podcast:
                return f"Podcast '{title}' not found in database."

            podcast.muted = True
            db.commit()

            return f"Successfully muted podcast: {title}"

        elif name == "unmute_subscription":
            title = arguments["title"]

            podcast = db.query(Podcast).filter_by(title=title).first()
            if not podcast:
                return f"Podcast '{title}' not found in database."

            podcast.muted = False
            db.commit()

            return f"Successfully unmuted podcast: {title}"

        elif name == "generate_briefing":
            days = arguments.get("days", 7)
            categories_str = arguments.get("categories")

            cat_queries = None
            cat_labels = None
            if categories_str:
                cat_labels = [c.strip() for c in categories_str.split(",")]
                cat_queries = cat_labels

            result = proc.generate_briefing(
                days=days,
                categories=cat_queries,
                category_labels=cat_labels,
            )
            return result

        else:
            raise ValueError(f"Unknown tool: {name}")

    async def get_resource(self, uri: str) -> str:
        """Get a resource by URI."""
        db, proc = ensure_db_connection()

        if uri == "podsidian://episodes":
            # Get recent episodes
            episodes = db.query(Episode).order_by(Episode.published_at.desc()).limit(10).all()

            return f"Recent episodes: {len(episodes)} episodes found"

        elif uri == "podsidian://subscriptions":
            # Get subscription count
            subs = get_subscriptions()
            return f"Podcast subscriptions: {len(subs) if subs else 0} subscriptions found"

        else:
            raise ValueError(f"Unknown resource URI: {uri}")

    async def run(self):
        """Run the MCP server."""
        logger.info("Starting MCP STDIO server")

        # Initialize database connection
        ensure_db_connection()

        try:
            while True:
                # Read JSON-RPC message from stdin
                line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
                if not line:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    request = json.loads(line)
                    logger.debug(f"Received request: {request}")

                    response = await self.handle_request(request)

                    if response is not None:
                        response_json = json.dumps(response)
                        print(response_json)
                        sys.stdout.flush()
                        logger.debug(f"Sent response: {response_json}")

                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON: {e}")
                    error_response = {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {"code": -32700, "message": "Parse error"},
                    }
                    print(json.dumps(error_response))
                    sys.stdout.flush()

        except KeyboardInterrupt:
            logger.info("Server interrupted")
        except Exception as e:
            logger.error(f"Server error: {e}", exc_info=True)
        finally:
            if db_session:
                db_session.close()
            logger.info("MCP STDIO server stopped")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="MCP STDIO server for Podsidian",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--log",
        metavar="FILE",
        default="/tmp/podsidian_mcp_stdio.log",
        help="Path to log file for debug outputs",
    )

    args = parser.parse_args()

    # Configure logging with the specified log file
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(args.log)],
        force=True,  # Override any existing configuration
    )

    # Set log level for our logger
    if args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    logger.info("Podsidian MCP server starting")

    # Create and run server
    server = MCPServer()
    await server.run()


async def run_stdio_server():
    """Run the STDIO server - entry point for CLI."""
    # Configure basic logging to stderr
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
        force=True,
    )

    logger.setLevel(logging.INFO)
    logger.info("Podsidian MCP server starting")

    # Create and run server
    server = MCPServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
