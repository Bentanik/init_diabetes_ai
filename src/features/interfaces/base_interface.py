"""Interface cơ sở cho tất cả các tính năng."""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseInterface(ABC):
    """Interface cơ sở mà tất cả các tính năng phải triển khai."""

    @abstractmethod
    async def initialize(self) -> None:
        """Khởi tạo tài nguyên cho tính năng."""
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Giải phóng tài nguyên của tính năng."""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Lấy thống kê hoạt động của tính năng."""
        pass
