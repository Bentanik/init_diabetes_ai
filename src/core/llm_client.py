"""Client LLM đơn giản với Langchain."""

from langchain_community.chat_models import ChatOpenAI
from config.settings import get_llm_config
from core.exceptions import ServiceError
from core.logging_config import get_logger

logger = get_logger(__name__)


class LLMClient:
    """Client LLM đơn giản - chỉ cần URL, API key và model."""

    def __init__(self):
        try:
            self.config = get_llm_config()
            self.client = self._initialize_client()
            logger.info(f"Khởi tạo LLM client thành công")
            logger.info(f"Base URL: {self.config['base_url']}")
            logger.info(f"Model: {self.config['model']}")
        except Exception as e:
            logger.error(f"Khởi tạo LLM client thất bại: {e}")
            raise ServiceError(f"Không thể khởi tạo LLM client: {e}")

    def _initialize_client(self):
        """Khởi tạo client LLM đơn giản."""
        # Chuẩn bị các tham số
        client_params = {
            "model": self.config["model"],
            "temperature": self.config["temperature"],
            "max_tokens": self.config["max_tokens"],
            "base_url": self.config["base_url"],
        }

        # ChatOpenAI luôn yêu cầu api_key, dùng dummy cho localhost/ollama
        if self.config["api_key"]:
            client_params["api_key"] = self.config["api_key"]
        else:
            # Dùng dummy key cho localhost/ollama
            client_params["api_key"] = "not-needed"

        # Tạo client ChatOpenAI (tương thích với mọi API OpenAI-format)
        return ChatOpenAI(**client_params)

    async def generate(self, prompt: str) -> str:
        """Tạo phản hồi từ LLM."""
        try:
            logger.debug(f"Đang tạo phản hồi cho prompt có độ dài: {len(prompt)}")
            response = await self.client.ainvoke(prompt)

            # Trích xuất text từ response
            if hasattr(response, "content"):
                result = str(response.content)
            else:
                result = str(response)

            logger.debug(f"Độ dài phản hồi được tạo: {len(result)}")
            return result

        except Exception as e:
            logger.error(f"Tạo phản hồi LLM thất bại: {e}")
            raise ServiceError(f"Tạo phản hồi LLM thất bại: {e}")

    def get_provider_info(self) -> dict:
        """Lấy thông tin về cấu hình hiện tại."""
        return {
            "base_url": self.config["base_url"],
            "model": self.config["model"],
            "temperature": self.config["temperature"],
            "max_tokens": self.config["max_tokens"],
            "has_api_key": bool(self.config["api_key"]),
        }


# Instance toàn cục
_llm_client = None


def get_llm():
    """Lấy instance LLM client toàn cục."""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


# Các hàm để tương thích với code cũ
async def query_care_plan_llm(prompt: str) -> str:
    """Tạo phản hồi kế hoạch chăm sóc."""
    return await get_llm().generate(prompt)


async def query_note_record_llm(prompt: str) -> str:
    """Tạo phản hồi phân tích kết quả đo."""
    return await get_llm().generate(prompt)
