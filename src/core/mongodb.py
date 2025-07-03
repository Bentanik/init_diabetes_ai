"""MongoDB connection manager."""

from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from core.logging_config import get_logger
from config.settings import get_mongodb_config

logger = get_logger(__name__)

# Global MongoDB client
_mongodb_client: Optional[AsyncIOMotorClient] = None
_mongodb_db: Optional[AsyncIOMotorDatabase] = None


async def get_mongodb() -> AsyncIOMotorDatabase:
    """Get MongoDB database instance."""
    global _mongodb_client, _mongodb_db

    if _mongodb_db is None:
        try:
            # Get config
            config = get_mongodb_config()

            # Create client
            _mongodb_client = AsyncIOMotorClient(
                config["uri"],
                serverSelectionTimeoutMS=5000,  # 5 seconds timeout
            )

            # Get database
            _mongodb_db = _mongodb_client[config["database"]]

            # Test connection
            await _mongodb_db.command("ping")
            logger.info(f"Connected to MongoDB at {config['uri']}")

        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            if _mongodb_client:
                _mongodb_client.close()
            raise

    return _mongodb_db


async def close_mongodb():
    """Close MongoDB connection."""
    global _mongodb_client, _mongodb_db

    if _mongodb_client:
        _mongodb_client.close()
        _mongodb_client = None
        _mongodb_db = None
        logger.info("Closed MongoDB connection")
