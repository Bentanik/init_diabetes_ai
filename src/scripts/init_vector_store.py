"""Script để khởi tạo Qdrant vector store."""

import logging
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_vector_store():
    """Khởi tạo Qdrant vector store với cấu hình cho Vietnamese documents."""
    try:
        # Connect to Qdrant
        client = QdrantClient("localhost", port=6333)
        collection_name = "default"

        # Check if collection exists
        collections = client.get_collections().collections
        exists = any(collection.name == collection_name for collection in collections)

        if exists:
            logger.info(f"Collection {collection_name} already exists. Recreating...")
            client.delete_collection(collection_name)

        # Create collection
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=768, distance=models.Distance.COSINE  # E5 embedding dimension
            ),
        )

        # Create payload index for metadata filtering
        client.create_payload_index(
            collection_name=collection_name,
            field_name="metadata.source_file",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

        client.create_payload_index(
            collection_name=collection_name,
            field_name="metadata.file_name",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

        client.create_payload_index(
            collection_name=collection_name,
            field_name="metadata.file_extension",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

        logger.info(f"Successfully created collection {collection_name} with indexes")
        return True

    except Exception as e:
        logger.error(f"Error initializing vector store: {str(e)}")
        raise


if __name__ == "__main__":
    init_vector_store()
