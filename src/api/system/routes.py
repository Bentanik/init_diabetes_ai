"""Routes cho system info và kiểm tra."""

from fastapi import APIRouter, HTTPException
from core.llm_client import get_llm
from core.exceptions import ServiceError
from core.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["system"])


@router.get("/llm-info")
async def get_llm_info():
    """Lấy thông tin về LLM configuration hiện tại."""
    try:
        client = get_llm()
        info = client.get_provider_info()
        logger.info(f"Trả về thông tin LLM: {info['base_url']}")
        return {
            "status": "success",
            "data": info,
            "message": f"LLM đang sử dụng: {info['model']}",
        }
    except ServiceError as e:
        logger.error(f"Lỗi lấy thông tin LLM: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Lỗi không xác định: {e}")
        raise HTTPException(status_code=500, detail="Lỗi hệ thống")


@router.get("/health")
async def health_check():
    """Kiểm tra sức khỏe hệ thống."""
    try:
        # Kiểm tra LLM client có hoạt động không
        client = get_llm()
        llm_info = client.get_provider_info()

        return {
            "status": "healthy",
            "llm_base_url": llm_info["base_url"],
            "llm_model": llm_info["model"],
            "has_api_key": llm_info["has_api_key"],
            "message": "Hệ thống hoạt động bình thường",
        }
    except Exception as e:
        logger.error(f"Health check thất bại: {e}")
        raise HTTPException(status_code=503, detail="Hệ thống không khả dụng")
