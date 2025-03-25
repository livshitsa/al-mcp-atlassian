import asyncio
import logging
import os
import sys

import click
from dotenv import load_dotenv

__version__ = "0.2.5"

logger = logging.getLogger("mcp-atlassian")


@click.command()
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Increase verbosity (can be used multiple times)",
)
@click.option(
    "--env-file", type=click.Path(exists=True, dir_okay=False), help="Path to .env file"
)
@click.option(
    "--transport",
    type=click.Choice(["stdio", "sse"]),
    default="stdio",
    help="Transport type (stdio or sse)",
)
@click.option(
    "--port",
    default=8000,
    help="Port to listen on for SSE transport",
)
@click.option(
    "--confluence-url",
    help="Confluence URL (e.g., https://your-domain.atlassian.net/wiki)",
)
@click.option("--confluence-username", help="Confluence username/email")
@click.option("--confluence-token", help="Confluence API token")
@click.option(
    "--confluence-personal-token",
    help="Confluence Personal Access Token (for Confluence Server/Data Center)",
)
@click.option(
    "--confluence-ssl-verify/--no-confluence-ssl-verify",
    default=True,
    help="Verify SSL certificates for Confluence Server/Data Center (default: verify)",
)

def main(
    verbose: bool,
    env_file: str | None,
    transport: str,
    port: int,
    confluence_url: str | None,
    confluence_username: str | None,
    confluence_token: str | None,
    confluence_personal_token: str | None,
    confluence_ssl_verify: bool,

) -> None:
    """MCP Atlassian Server - Confluence functionality for MCP

    Supports both Atlassian Cloud Server/Data Center deployments.
    """
    # Configure logging based on verbosity
    logging_level = logging.INFO
    if verbose == 1:
        logging_level = logging.INFO
    elif verbose >= 2:
        logging_level = logging.DEBUG

    logging.basicConfig(level=logging_level, stream=sys.stderr)

    # Load environment variables from file if specified, otherwise try default .env
    if env_file:
        logger.debug(f"Loading environment from file: {env_file}")
        load_dotenv(env_file)
    else:
        logger.debug("Attempting to load environment from default .env file")
        load_dotenv()

    # Set environment variables from command line arguments if provided
    if confluence_url:
        os.environ["CONFLUENCE_URL"] = confluence_url
    if confluence_username:
        os.environ["CONFLUENCE_USERNAME"] = confluence_username
    if confluence_token:
        os.environ["CONFLUENCE_API_TOKEN"] = confluence_token
    if confluence_personal_token:
        os.environ["CONFLUENCE_PERSONAL_TOKEN"] = confluence_personal_token

    # Set SSL verification for Confluence Server/Data Center
    os.environ["CONFLUENCE_SSL_VERIFY"] = str(confluence_ssl_verify).lower()


    from . import server

    # Run the server with specified transport
    asyncio.run(server.run_server(transport=transport, port=port))


__all__ = ["main", "server", "__version__"]

if __name__ == "__main__":
    main()
