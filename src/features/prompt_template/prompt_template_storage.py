"""MongoDB implementation cho việc lưu trữ prompt templates."""

import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import ASCENDING, TEXT

from core.logging_config import get_logger
from features.interfaces.prompt_template_interface import PromptTemplateInterface
from api.rag.models import (
    PromptTemplate,
    PromptTemplateCreate,
    PromptTemplateUpdate,
)
from utils.utils import extract_variables

logger = get_logger(__name__)


class MongoPromptTemplateStorage(PromptTemplateInterface):
    """MongoDB implementation cho việc lưu trữ prompt templates."""

    def __init__(self, db: AsyncIOMotorDatabase):
        """Khởi tạo với MongoDB database instance."""
        self.db = db
        self.collection = self.db.prompt_templates

    async def setup(self):
        """Setup indexes và các cấu hình cần thiết."""
        # Tạo text index cho tìm kiếm
        await self.collection.create_index(
            [
                ("name", TEXT),
                ("description", TEXT),
                ("template", TEXT),
            ]
        )
        # Tạo index cho các trường thường query
        await self.collection.create_index([("created_at", ASCENDING)])

    async def create_template(
        self, template_data: PromptTemplateCreate
    ) -> PromptTemplate:
        """Tạo một prompt template mới."""
        try:
            # Generate ID
            template_id = str(uuid.uuid4())

            # Extract variables from template
            variables = extract_variables(template_data.template)

            # Create template object
            template = PromptTemplate(
                id=template_id,
                name=template_data.name,
                description=template_data.description,
                template=template_data.template,
                variables=variables,
                example_values=template_data.example_values or {},
                metadata=template_data.metadata,
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
            )

            # Convert to dict and save
            template_dict = template.model_dump()
            await self.collection.insert_one(template_dict)

            return template

        except Exception as e:
            logger.error(f"Error creating prompt template: {e}")
            raise

    async def get_template(self, template_id: str) -> Optional[PromptTemplate]:
        """Lấy thông tin của một template theo ID."""
        try:
            template_dict = await self.collection.find_one({"id": template_id})
            if template_dict:
                return PromptTemplate(**template_dict)
            return None

        except Exception as e:
            logger.error(f"Error getting prompt template: {e}")
            raise

    async def list_templates(
        self,
        skip: int = 0,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[PromptTemplate]:
        """Lấy danh sách templates với phân trang và lọc."""
        try:
            # Build query
            query = filters or {}

            # Get templates with pagination
            cursor = self.collection.find(query).skip(skip).limit(limit)

            # Convert to list of PromptTemplate objects
            templates = []
            async for template_dict in cursor:
                templates.append(PromptTemplate(**template_dict))

            return templates

        except Exception as e:
            logger.error(f"Error listing prompt templates: {e}")
            raise

    async def update_template(
        self, template_id: str, template_data: PromptTemplateUpdate
    ) -> Optional[PromptTemplate]:
        """Cập nhật thông tin của một template."""
        try:
            # Get current template
            current = await self.get_template(template_id)
            if not current:
                return None

            # Build update data
            update_data = {}
            if template_data.name is not None:
                update_data["name"] = template_data.name
            if template_data.description is not None:
                update_data["description"] = template_data.description
            if template_data.template is not None:
                update_data["template"] = template_data.template
                # Re-extract variables
                variables = extract_variables(template_data.template)
                update_data["variables"] = variables
            if template_data.metadata is not None:
                update_data["metadata"] = template_data.metadata
            if template_data.example_values is not None:
                update_data["example_values"] = template_data.example_values

            update_data["updated_at"] = datetime.now().isoformat()

            # Update in database
            result = await self.collection.update_one(
                {"id": template_id}, {"$set": update_data}
            )

            if result.modified_count > 0:
                return await self.get_template(template_id)
            return None

        except Exception as e:
            logger.error(f"Error updating prompt template: {e}")
            raise

    async def delete_template(self, template_id: str) -> bool:
        """Xóa một template."""
        try:
            result = await self.collection.delete_one({"id": template_id})
            return result.deleted_count > 0

        except Exception as e:
            logger.error(f"Error deleting prompt template: {e}")
            raise

    async def search_templates(
        self, query: str, limit: int = 10
    ) -> List[PromptTemplate]:
        """Tìm kiếm templates theo từ khóa."""
        try:
            # Sử dụng text search của MongoDB
            cursor = (
                self.collection.find(
                    {"$text": {"$search": query}},
                    {"score": {"$meta": "textScore"}},
                )
                .sort([("score", {"$meta": "textScore"})])
                .limit(limit)
            )

            # Convert to list of PromptTemplate objects
            templates = []
            async for template_dict in cursor:
                templates.append(PromptTemplate(**template_dict))

            return templates

        except Exception as e:
            logger.error(f"Error searching prompt templates: {e}")
            raise
