import json
import logging
import os
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from mcp.server import Server
from mcp.types import Resource, TextContent, Tool

from .confluence import ConfluenceFetcher

# Configure logging
logger = logging.getLogger("mcp-atlassian")


@dataclass
class AppContext:
    """Application context for MCP Atlassian."""

    confluence: ConfluenceFetcher | None = None


def get_available_services() -> dict[str, bool | None]:
    """Determine which services are available based on environment variables."""

    # Check for either cloud authentication (URL + username + API token)
    # or server/data center authentication (URL + personal token)
    confluence_url = os.getenv("CONFLUENCE_URL")
    if confluence_url:
        is_cloud = "atlassian.net" in confluence_url
        if is_cloud:
            confluence_vars = all(
                [
                    confluence_url,
                    os.getenv("CONFLUENCE_USERNAME"),
                    os.getenv("CONFLUENCE_API_TOKEN"),
                ]
            )
            logger.info("Using Confluence Cloud authentication method")
        else:
            confluence_vars = all(
                [confluence_url, os.getenv("CONFLUENCE_PERSONAL_TOKEN")]
            )
            logger.info("Using Confluence Server/Data Center authentication method")
    else:
        confluence_vars = False

    return {"confluence": confluence_vars}


@asynccontextmanager
async def server_lifespan(server: Server) -> AsyncIterator[AppContext]:
    """Initialize and clean up application resources."""
    # Get available services
    services = get_available_services()

    try:
        # Initialize services
        confluence = ConfluenceFetcher() if services["confluence"] else None

        # Log the startup information
        logger.info("Starting MCP Atlassian server")
        if confluence:
            logger.info(f"Confluence URL: {confluence.config.url}")

        # Provide context to the application
        yield AppContext(confluence=confluence)
    finally:
        # Cleanup resources if needed
        pass


# Create server instance
app = Server("mcp-atlassian", lifespan=server_lifespan)


# Implement server handlers
@app.list_resources()
async def list_resources() -> list[Resource]:
    """List Confluence spaces the user is actively interacting with."""
    resources = []

    ctx = app.request_context.lifespan_context

    # Add Confluence spaces the user has contributed to
    if ctx and ctx.confluence:
        try:
            # Get spaces the user has contributed to
            spaces = ctx.confluence.get_user_contributed_spaces(limit=250)

            # Add spaces to resources
            resources.extend(
                [
                    Resource(
                        uri=f"confluence://{space['key']}",
                        name=f"Confluence Space: {space['name']}",
                        mimeType="text/plain",
                        description=(
                            f"A Confluence space containing documentation and knowledge base articles. "
                            f"Space Key: {space['key']}. "
                            f"{space.get('description', '')} "
                            f"Access content using: confluence://{space['key']}/pages/PAGE_TITLE"
                        ).strip(),
                    )
                    for space in spaces.values()
                ]
            )
        except Exception as e:
            logger.error(f"Error fetching Confluence spaces: {str(e)}")

    return resources


@app.read_resource()
async def read_resource(uri: str) -> str: #tuple[str, str]:
    """Read content from Confluence based on the resource URI."""
    logger.info(f"read_resource uri {uri}")
    # parsed_uri = urlparse(uri)
    uri = str(uri)

    # Get application context
    ctx = app.request_context.lifespan_context

    # Handle Confluence resources
    if uri.startswith("confluence://"):
        if not ctx or not ctx.confluence:
            raise ValueError(
                "Confluence is not configured. Please provide Confluence credentials."
            )
        parts = uri.replace("confluence://", "").split("/")

        # Handle space listing
        if len(parts) == 1:
            space_key = parts[0]

            # Use CQL to find recently updated pages in this space
            cql = f'space = "{space_key}" AND contributor = currentUser() ORDER BY lastmodified DESC'
            pages = ctx.confluence.search(cql=cql, limit=20)
            logger.info(f"num pages: {len(pages)}")
            
            if not pages:
                # Fallback to regular space pages if no user-contributed pages found
                pages = ctx.confluence.get_space_pages(space_key, limit=10)

            content = []
            for page in pages:
                page_dict = page.to_simplified_dict()
                
                title = page_dict.get("title", "Untitled")
                url = page_dict.get("url", "")

                content.append(f"# [{title}]({url})\n\n{page.page_content}\n\n---")

            return "\n\n".join(content) #, "text/markdown"

        # Handle specific page
        elif len(parts) >= 3 and parts[1] == "pages":
            space_key = parts[0]
            title = parts[2]
            page = ctx.confluence.get_page_by_title(space_key, title)

            if not page:
                raise ValueError(f"Page not found: {title}")

            return page.page_content #, "text/markdown"

    raise ValueError(f"Invalid resource URI: {uri}")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available Confluence tools."""
    tools = []
    ctx = app.request_context.lifespan_context

    # Add Confluence tools if Confluence is configured
    if ctx and ctx.confluence:
        tools.extend(
            [
                Tool(
                    name="confluence_search",
                    description="Search Confluence content using simple terms or CQL",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query - can be either a simple text (e.g. 'project documentation') or a CQL query string. Examples of CQL:\n"
                                "- Basic search: 'type=page AND space=DEV'\n"
                                "- Search by title: 'title~\"Meeting Notes\"'\n"
                                "- Recent content: 'created >= \"2023-01-01\"'\n"
                                "- Content with specific label: 'label=documentation'\n"
                                "- Recently modified content: 'lastModified > startOfMonth(\"-1M\")'\n"
                                "- Content modified this year: 'creator = currentUser() AND lastModified > startOfYear()'\n"
                                "- Content you contributed to recently: 'contributor = currentUser() AND lastModified > startOfWeek()'\n"
                                "- Content watched by user: 'watcher = \"user@domain.com\" AND type = page'\n"
                                '- Exact phrase in content: \'text ~ "\\"Urgent Review Required\\"" AND label = "pending-approval"\'\n'
                                '- Title wildcards: \'title ~ "Minutes*" AND (space = "HR" OR space = "Marketing")\'\n',
                            },
                            "limit": {
                                "type": "number",
                                "description": "Maximum number of results (1-50)",
                                "default": 10,
                                "minimum": 1,
                                "maximum": 50,
                            },
                        },
                        "required": ["query"],
                    },
                ),
                Tool(
                    name="confluence_get_page",
                    description="Get content of a specific Confluence page by ID",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "page_id": {
                                "type": "string",
                                "description": "Confluence page ID (numeric ID, can be found in the page URL). "
                                "For example, in the URL 'https://example.atlassian.net/wiki/spaces/TEAM/pages/123456789/Page+Title', "
                                "the page ID is '123456789'",
                            },
                            "include_metadata": {
                                "type": "boolean",
                                "description": "Whether to include page metadata such as creation date, last update, version, and labels",
                                "default": True,
                            },
                        },
                        "required": ["page_id"],
                    },
                ),
                Tool(
                    name="confluence_get_page_children",
                    description="Get child pages of a specific Confluence page",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "parent_id": {
                                "type": "string",
                                "description": "The ID of the parent page whose children you want to retrieve",
                            },
                            "expand": {
                                "type": "string",
                                "description": "Fields to expand in the response (e.g., 'version', 'body.storage')",
                                "default": "version",
                            },
                            "limit": {
                                "type": "number",
                                "description": "Maximum number of child pages to return (1-50)",
                                "default": 25,
                                "minimum": 1,
                                "maximum": 50,
                            },
                            "include_content": {
                                "type": "boolean",
                                "description": "Whether to include the page content in the response",
                                "default": False,
                            },
                        },
                        "required": ["parent_id"],
                    },
                ),
                Tool(
                    name="confluence_get_page_ancestors",
                    description="Get ancestor (parent) pages of a specific Confluence page",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "page_id": {
                                "type": "string",
                                "description": "The ID of the page whose ancestors you want to retrieve",
                            },
                        },
                        "required": ["page_id"],
                    },
                ),
                Tool(
                    name="confluence_get_comments",
                    description="Get comments for a specific Confluence page",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "page_id": {
                                "type": "string",
                                "description": "Confluence page ID (numeric ID, can be parsed from URL, "
                                "e.g. from 'https://example.atlassian.net/wiki/spaces/TEAM/pages/123456789/Page+Title' "
                                "-> '123456789')",
                            }
                        },
                        "required": ["page_id"],
                    },
                ),
                Tool(
                    name="confluence_create_page",
                    description="Create a new Confluence page",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "space_key": {
                                "type": "string",
                                "description": "The key of the space to create the page in "
                                "(usually a short uppercase code like 'DEV', 'TEAM', or 'DOC')",
                            },
                            "title": {
                                "type": "string",
                                "description": "The title of the page",
                            },
                            "content": {
                                "type": "string",
                                "description": "The content of the page in Markdown format. "
                                "Supports headings, lists, tables, code blocks, and other "
                                "Markdown syntax",
                            },
                            "parent_id": {
                                "type": "string",
                                "description": "Optional parent page ID. If provided, this page "
                                "will be created as a child of the specified page",
                            },
                        },
                        "required": ["space_key", "title", "content"],
                    },
                ),
                Tool(
                    name="confluence_update_page",
                    description="Update an existing Confluence page",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "page_id": {
                                "type": "string",
                                "description": "The ID of the page to update",
                            },
                            "title": {
                                "type": "string",
                                "description": "The new title of the page",
                            },
                            "content": {
                                "type": "string",
                                "description": "The new content of the page in Markdown format",
                            },
                            "is_minor_edit": {
                                "type": "boolean",
                                "description": "Whether this is a minor edit",
                                "default": False,
                            },
                            "version_comment": {
                                "type": "string",
                                "description": "Optional comment for this version",
                                "default": "",
                            },
                        },
                        "required": ["page_id", "title", "content"],
                    },
                ),
                Tool(
                    name="confluence_delete_page",
                    description="Delete an existing Confluence page",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "page_id": {
                                "type": "string",
                                "description": "The ID of the page to delete",
                            },
                        },
                        "required": ["page_id"],
                    },
                ),
            ]
        )

    return tools


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> Sequence[TextContent]:
    """Handle tool calls for Confluence operations."""
    ctx = app.request_context.lifespan_context
    try:
        # Helper functions for formatting results
        def format_comment(comment: Any) -> dict:
            if hasattr(comment, "to_simplified_dict"):
                return comment.to_simplified_dict()
            return {
                "id": comment.get("id"),
                "author": comment.get("author", {}).get("displayName", "Unknown"),
                "created": comment.get("created"),
                "body": comment.get("body"),
            }

        # Confluence operations
        if name == "confluence_search":
            if not ctx or not ctx.confluence:
                raise ValueError("Confluence is not configured.")

            query = arguments.get("query", "")
            limit = min(int(arguments.get("limit", 10)), 50)

            # Check if the query is a simple search term or already a CQL query
            if query and not any(
                x in query
                for x in ["=", "~", ">", "<", " AND ", " OR ", "currentUser()"]
            ):
                # Convert simple search term to CQL text search
                # This will search in all content (title, body, etc.)
                query = f'text ~ "{query}"'
                logger.info(f"Converting simple search term to CQL: {query}")

            pages = ctx.confluence.search(query, limit=limit)

            # Format results using the to_simplified_dict method
            search_results = [page.to_simplified_dict() for page in pages]

            return [
                TextContent(
                    type="text",
                    text=json.dumps(search_results, indent=2, ensure_ascii=False),
                )
            ]

        elif name == "confluence_get_page":
            if not ctx or not ctx.confluence:
                raise ValueError("Confluence is not configured.")

            page_id = arguments.get("page_id")
            include_metadata = arguments.get("include_metadata", True)

            page = ctx.confluence.get_page_content(page_id)

            if include_metadata:
                # The to_simplified_dict method already includes the content,
                # so we don't need to include it separately at the root level
                result = {
                    "metadata": page.to_simplified_dict(),
                }
            else:
                # For backward compatibility, keep returning content directly
                result = {"content": page.content}

            return [
                TextContent(
                    type="text", text=json.dumps(result, indent=2, ensure_ascii=False)
                )
            ]

        elif name == "confluence_get_page_children":
            if not ctx or not ctx.confluence:
                raise ValueError("Confluence is not configured.")

            parent_id = arguments.get("parent_id")
            expand = arguments.get("expand", "version")
            limit = min(int(arguments.get("limit", 25)), 50)
            include_content = arguments.get("include_content", False)

            # Add body.storage to expand if content is requested
            if include_content and "body" not in expand:
                expand = f"{expand},body.storage"

            # Get the child pages
            pages = ctx.confluence.get_page_children(
                page_id=parent_id, expand=expand, limit=limit, convert_to_markdown=True
            )

            # Format results using the to_simplified_dict method
            child_pages = [page.to_simplified_dict() for page in pages]

            # Return the formatted results
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "parent_id": parent_id,
                            "total": len(child_pages),
                            "limit": limit,
                            "results": child_pages,
                        },
                        indent=2,
                        ensure_ascii=False,
                    ),
                )
            ]

        elif name == "confluence_get_page_ancestors":
            if not ctx or not ctx.confluence:
                raise ValueError("Confluence is not configured.")

            page_id = arguments.get("page_id")

            # Get the ancestor pages
            ancestors = ctx.confluence.get_page_ancestors(page_id)

            # Format results
            ancestor_pages = [page.to_simplified_dict() for page in ancestors]

            return [
                TextContent(
                    type="text",
                    text=json.dumps(ancestor_pages, indent=2, ensure_ascii=False),
                )
            ]

        elif name == "confluence_get_comments":
            if not ctx or not ctx.confluence:
                raise ValueError("Confluence is not configured.")

            page_id = arguments.get("page_id")
            comments = ctx.confluence.get_page_comments(page_id)

            # Format comments using their to_simplified_dict method if available
            formatted_comments = [format_comment(comment) for comment in comments]

            return [
                TextContent(
                    type="text",
                    text=json.dumps(formatted_comments, indent=2, ensure_ascii=False),
                )
            ]

        elif name == "confluence_create_page":
            if not ctx or not ctx.confluence:
                raise ValueError("Confluence is not configured.")

            # Extract arguments
            space_key = arguments.get("space_key")
            title = arguments.get("title")
            content = arguments.get("content")
            parent_id = arguments.get("parent_id")

            # Create the page (with automatic markdown conversion)
            page = ctx.confluence.create_page(
                space_key=space_key,
                title=title,
                body=content,
                parent_id=parent_id,
                is_markdown=True,
            )

            # Format the result
            result = page.to_simplified_dict()

            return [
                TextContent(
                    type="text",
                    text=f"Page created successfully:\n{json.dumps(result, indent=2, ensure_ascii=False)}",
                )
            ]

        elif name == "confluence_update_page":
            if not ctx or not ctx.confluence:
                raise ValueError("Confluence is not configured.")

            page_id = arguments.get("page_id")
            title = arguments.get("title")
            content = arguments.get("content")
            is_minor_edit = arguments.get("is_minor_edit", False)
            version_comment = arguments.get("version_comment", "")

            if not page_id or not title or not content:
                raise ValueError(
                    "Missing required parameters: page_id, title, and content are required."
                )

            # Update the page (with automatic markdown conversion)
            updated_page = ctx.confluence.update_page(
                page_id=page_id,
                title=title,
                body=content,
                is_minor_edit=is_minor_edit,
                version_comment=version_comment,
                is_markdown=True,
            )

            # Format results
            page_data = updated_page.to_simplified_dict()

            return [TextContent(type="text", text=json.dumps({"page": page_data}))]

        elif name == "confluence_delete_page":
            if not ctx or not ctx.confluence:
                raise ValueError("Confluence is not configured.")

            page_id = arguments.get("page_id")

            if not page_id:
                raise ValueError("Missing required parameter: page_id is required.")

            try:
                # Delete the page
                result = ctx.confluence.delete_page(page_id=page_id)

                # Format results - our fixed implementation now correctly returns True on success
                if result:
                    response = {
                        "success": True,
                        "message": f"Page {page_id} deleted successfully",
                    }
                else:
                    # This branch should rarely be hit with our updated implementation
                    # but we keep it for safety
                    response = {
                        "success": False,
                        "message": f"Unable to delete page {page_id}. The API request completed but deletion was unsuccessful.",
                    }

                return [
                    TextContent(
                        type="text",
                        text=json.dumps(response, indent=2, ensure_ascii=False),
                    )
                ]
            except Exception as e:
                # API call failed with an exception
                logger.error(f"Error deleting Confluence page {page_id}: {str(e)}")
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "success": False,
                                "message": f"Error deleting page {page_id}",
                                "error": str(e),
                            },
                            indent=2,
                            ensure_ascii=False,
                        ),
                    )
                ]

        raise ValueError(f"Unknown tool: {name}")

    except Exception as e:
        logger.error(f"Tool execution error: {str(e)}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def run_server(transport: str = "stdio", port: int = 8000) -> None:
    """Run the MCP Atlassian server with the specified transport."""
    if transport == "sse":
        from mcp.server.sse import SseServerTransport
        from starlette.applications import Starlette
        from starlette.requests import Request
        from starlette.routing import Mount, Route

        sse = SseServerTransport("/messages/")

        async def handle_sse(request: Request) -> None:
            async with sse.connect_sse(
                request.scope, request.receive, request._send
            ) as streams:
                await app.run(
                    streams[0], streams[1], app.create_initialization_options()
                )

        starlette_app = Starlette(
            debug=True,
            routes=[
                Route("/sse", endpoint=handle_sse),
                Mount("/messages/", app=sse.handle_post_message),
            ],
        )

        import uvicorn

        # Set up uvicorn config
        config = uvicorn.Config(starlette_app, host="0.0.0.0", port=port)  # noqa: S104
        server = uvicorn.Server(config)
        # Use server.serve() instead of run() to stay in the same event loop
        await server.serve()
    else:
        from mcp.server.stdio import stdio_server

        async with stdio_server() as (read_stream, write_stream):
            await app.run(
                read_stream, write_stream, app.create_initialization_options()
            )
