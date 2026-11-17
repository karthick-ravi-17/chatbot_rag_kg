import os
import sys
import logging
from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import (
    AuthError,
    ServiceUnavailable,
    ConfigurationError,
    Neo4jError,
)

# Load environment variables from a .env file if present
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def validate_neo4j_connection(
    uri: str,
    username: str,
    password: str,
    database: str = "neo4j",
) -> bool:
    """
    Validates connectivity and authentication to a Neo4j database instance.

    Args:
        uri: The Neo4j connection URI (e.g., "bolt+ssc://..." or "neo4j+s://...").
        username: The Neo4j username.
        password: The Neo4j password.
        database: The target database name (default: "neo4j").

    Returns:
        True if the connection and test query succeed, False otherwise.
    """
    logger.info(f"Attempting to connect to Neo4j at {uri} (database: '{database}')...")

    try:
        with GraphDatabase.driver(uri, auth=(username, password)) as driver:
            # 1. Verify driver connectivity and credentials
            driver.verify_connectivity()
            logger.info("Driver connectivity verified successfully.")

            # 2. Execute a lightweight read query to verify database access
            records, summary, _ = driver.execute_query(
                "RETURN 1 AS status",
                database_=database,
            )

            if records and records[0]["status"] == 1:
                server_info = summary.server
                logger.info(
                    f"Connection validated! Connected to Neo4j "
                    f"({server_info.agent}) at {server_info.address}."
                )
                return True
            else:
                logger.error("Connected, but test query returned unexpected results.")
                return False

    except AuthError as e:
        logger.error(f"Authentication failed: Invalid username or password. Details: {e}")
    except ServiceUnavailable as e:
        logger.error(
            f"Service unavailable: Unable to reach Neo4j at '{uri}'. "
            f"Check if the server is running and the URI/port are correct. Details: {e}"
        )
    except ConfigurationError as e:
        logger.error(f"Configuration error in driver settings or URI: {e}")
    except Neo4jError as e:
        logger.error(f"Neo4j database error occurred: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during Neo4j validation: {e}")

    return False


if __name__ == "__main__":
    # Load credentials from environment variables (loaded via python-dotenv)
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
    NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

    if not NEO4J_PASSWORD:
        logger.warning("Warning: NEO4J_PASSWORD is empty or not set in environment/.env.")

    is_connected = validate_neo4j_connection(
        uri=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        database=NEO4J_DATABASE,
    )

sys.exit(0 if is_connected else 1)