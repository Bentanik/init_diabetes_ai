from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Mô hình phản hồi cho endpoint kiểm tra sức khỏe.

    Thuộc tính:
        status: Trạng thái sức khỏe dịch vụ ("healthy" hoặc "unhealthy")
        service: Tên và mô tả dịch vụ
        version: Phiên bản hiện tại của dịch vụ
    """

    status: str
    service: str
    version: str

    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "service": "AI Service - CarePlan Generator",
                "version": "1.0.0",
            }
        }
