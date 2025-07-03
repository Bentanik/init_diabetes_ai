"""MinIO storage integration for document storage."""

import os
from datetime import datetime
from typing import Dict, Any, Optional, List

from core.minio_client import minio_client
from core.logging_config import get_logger

logger = get_logger(__name__)


class DocumentStorage:
    """Document storage using MinIO."""

    BUCKET_NAME = "documents"

    def __init__(self):
        self.client = minio_client
        self._ensure_bucket_exists()

    def _ensure_bucket_exists(self):
        """Ensure the documents bucket exists"""
        try:
            if not self.client.client.bucket_exists(self.BUCKET_NAME):
                self.client.client.make_bucket(self.BUCKET_NAME)
                logger.info(f"Created bucket {self.BUCKET_NAME}")
        except Exception as e:
            logger.error(f"Error ensuring bucket exists: {str(e)}")
            raise Exception(f"Failed to ensure bucket exists: {str(e)}")

    def store_document(
        self,
        file_data: bytes,
        filename: str,
        knowledge_name: str,
        content_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Store a document in MinIO.

        Args:
            file_data: The file content in bytes
            filename: Original filename
            knowledge_name: Knowledge base name (used as folder name)
            content_type: File content type
            metadata: Optional metadata

        Returns:
            Dict with storage information
        """
        try:
            # Generate a unique object name with folder structure
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            object_name = f"{knowledge_name}/{timestamp}_{filename}"

            # Store the file
            self.client.upload_file(
                bucket_name=self.BUCKET_NAME,
                object_name=object_name,
                file_data=file_data,
                content_type=content_type,
            )

            # Return storage info
            return {
                "storage_path": f"{self.BUCKET_NAME}/{object_name}",
                "storage_time": datetime.now().isoformat(),
                "size_bytes": len(file_data),
                "content_type": content_type,
                "metadata": metadata or {},
            }

        except Exception as e:
            logger.error(f"Error storing document in MinIO: {str(e)}")
            raise Exception(f"Failed to store document: {str(e)}")

    def get_document(self, knowledge_name: str, object_name: str):
        """
        Get a document from MinIO.

        Args:
            knowledge_name: Knowledge base name (folder name)
            object_name: Object name in the folder

        Returns:
            The document data
        """
        try:
            full_path = f"{knowledge_name}/{object_name}"
            return self.client.get_file(self.BUCKET_NAME, full_path)
        except Exception as e:
            logger.error(f"Error getting document from MinIO: {str(e)}")
            raise Exception(f"Failed to get document: {str(e)}")

    def delete_collection_folder(self, knowledge_name: str) -> bool:
        """
        Delete all documents in a collection folder.

        Args:
            knowledge_name: Knowledge base name (folder name)

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # List all objects in the collection folder
            objects = self.client.client.list_objects(
                self.BUCKET_NAME, prefix=f"{knowledge_name}/"
            )

            # Delete all objects in the folder
            for obj in objects:
                if obj.object_name:  # Check if object_name is not None
                    self.client.delete_file(self.BUCKET_NAME, obj.object_name)
                    logger.info(
                        f"Deleted object {obj.object_name} from collection {knowledge_name}"
                    )

            logger.info(f"Deleted all documents in collection {knowledge_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection {knowledge_name}: {str(e)}")
            return False

    def get_collection_stats(self, knowledge_name: str) -> Dict[str, Any]:
        """
        Get statistics about documents in a collection.

        Args:
            knowledge_name: Knowledge base name (folder name)

        Returns:
            Dict containing statistics about the documents
        """
        try:
            # List all objects in the collection folder
            objects = list(
                self.client.client.list_objects(
                    self.BUCKET_NAME, prefix=f"{knowledge_name}/"
                )
            )

            # Calculate statistics
            total_size = 0
            file_types = {}
            documents = []

            for obj in objects:
                if obj.object_name and obj.size is not None:
                    # Extract filename from path
                    filename = obj.object_name.split("/")[-1]
                    # Get file extension
                    ext = os.path.splitext(filename)[1].lower()

                    # Update file type count
                    file_types[ext] = file_types.get(ext, 0) + 1

                    # Update total size
                    total_size += obj.size

                    # Add document info
                    documents.append(
                        {
                            "filename": filename,
                            "size": obj.size,
                            "last_modified": (
                                obj.last_modified.isoformat()
                                if obj.last_modified
                                else None
                            ),
                            "content_type": ext,
                        }
                    )

            return {
                "total_documents": len(objects),
                "total_size_bytes": total_size,
                "file_types": file_types,
                "documents": documents,
                "collection_name": knowledge_name,
                "last_updated": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            raise Exception(f"Failed to get collection stats: {str(e)}")


document_storage = DocumentStorage()
