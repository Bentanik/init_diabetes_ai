"""Các exception tùy chỉnh."""


class ServiceError(Exception):
    """Lỗi dịch vụ cho tất cả các vấn đề."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class ConfigError(Exception):
    """Exception cho các lỗi cấu hình."""

    pass
