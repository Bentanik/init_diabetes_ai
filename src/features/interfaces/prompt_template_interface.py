"""Interface cho việc lưu trữ và quản lý prompt templates."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime

from api.rag.models import (
    PromptTemplate,
    PromptTemplateCreate,
    PromptTemplateUpdate,
)


class PromptTemplateInterface(ABC):
    """Interface định nghĩa các phương thức để quản lý prompt templates."""

    @abstractmethod
    async def create_template(
        self, template_data: PromptTemplateCreate
    ) -> PromptTemplate:
        """Tạo một prompt template mới."""
        pass

    @abstractmethod
    async def get_template(self, template_id: str) -> Optional[PromptTemplate]:
        """Lấy thông tin của một template theo ID."""
        pass

    @abstractmethod
    async def list_templates(
        self,
        skip: int = 0,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[PromptTemplate]:
        """Lấy danh sách templates với phân trang và lọc."""
        pass

    @abstractmethod
    async def update_template(
        self, template_id: str, template_data: PromptTemplateUpdate
    ) -> Optional[PromptTemplate]:
        """Cập nhật thông tin của một template."""
        pass

    @abstractmethod
    async def delete_template(self, template_id: str) -> bool:
        """Xóa một template."""
        pass

    @abstractmethod
    async def search_templates(
        self, query: str, limit: int = 10
    ) -> List[PromptTemplate]:
        """Tìm kiếm templates theo từ khóa."""
        pass
