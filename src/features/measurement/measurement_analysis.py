"""Triển khai tính năng phân tích kết quả đo."""

from datetime import datetime
from typing import Dict, Any

from core.exceptions import ServiceError
from core.llm_client import get_llm
from core.logging_config import get_logger
from features.interfaces.measurement_interface import MeasurementAnalysisInterface
from api.analyze.models import AnalyzeMeasurementRequest, AnalyzeMeasurementResponse
from prompts.measurement_note_prompt import MEASUREMENT_NOTE_PROMPT
from config.settings import get_config

logger = get_logger(__name__)


class MeasurementAnalysis(MeasurementAnalysisInterface):
    """Triển khai tính năng phân tích kết quả đo."""

    def __init__(self):
        """Khởi tạo phân tích kết quả đo."""
        self._llm = None
        self._stats = {
            "total_analyses": 0,
            "successful_analyses": 0,
            "failed_analyses": 0,
            "last_analysis_time": None,
        }
        self._supported_types = {
            "Blood Glucose": {
                "units": ["mmol/L", "mg/dL"],
                "normal_ranges": {
                    "fasting": {"min": 4.0, "max": 7.0},
                    "post_meal": {"min": 4.0, "max": 10.0},
                },
            },
            "Blood Pressure": {
                "units": ["mmHg"],
                "normal_ranges": {
                    "systolic": {"min": 90, "max": 140},
                    "diastolic": {"min": 60, "max": 90},
                },
            },
        }

    async def initialize(self) -> None:
        """Khởi tạo tài nguyên cho phân tích kết quả đo."""
        try:
            self._llm = get_llm()
            logger.info("Khởi tạo phân tích kết quả đo thành công")
        except Exception as e:
            logger.error(f"Lỗi khởi tạo phân tích kết quả đo: {e}")
            raise

    async def shutdown(self) -> None:
        """Giải phóng tài nguyên của phân tích kết quả đo."""
        self._llm = None
        logger.info("Đã giải phóng phân tích kết quả đo")

    def get_stats(self) -> Dict[str, Any]:
        """Lấy thống kê hoạt động của phân tích kết quả đo."""
        return self._stats.copy()

    def get_supported_types(self) -> Dict[str, Any]:
        """Lấy danh sách các loại đo được hỗ trợ và quy tắc xác thực."""
        return self._supported_types.copy()

    async def analyze_measurement(
        self, request: AnalyzeMeasurementRequest
    ) -> AnalyzeMeasurementResponse:
        """Phân tích kết quả đo và đưa ra phản hồi."""
        try:
            logger.info(f"Đang phân tích kết quả đo cho bệnh nhân: {request.patientId}")
            self._stats["total_analyses"] += 1

            # Tạo prompt từ dữ liệu yêu cầu
            prompt = self._build_prompt(request)
            logger.debug(f"Đã tạo prompt cho bệnh nhân {request.patientId}")

            # Gọi LLM để phân tích
            if self._llm is None:
                logger.error("LLM chưa được khởi tạo")
                raise ServiceError("LLM chưa được khởi tạo")
            feedback_text = await self._llm.generate(prompt)
            logger.debug(f"Đã nhận phản hồi cho bệnh nhân {request.patientId}")

            # Kiểm tra độ dài phản hồi
            max_length = get_config("max_feedback_length") or 250
            if len(feedback_text) > max_length:
                logger.warning(
                    f"Phản hồi quá dài cho bệnh nhân {request.patientId}, cắt bớt"
                )
                feedback_text = feedback_text[:max_length]

            # Tạo phản hồi
            response = AnalyzeMeasurementResponse(
                patientId=request.patientId,
                recordTime=datetime.utcnow().isoformat(),
                feedback=feedback_text.strip(),
            )

            self._stats["successful_analyses"] += 1
            self._stats["last_analysis_time"] = datetime.utcnow().isoformat()
            logger.info(
                f"Phân tích kết quả đo thành công cho bệnh nhân {request.patientId}"
            )
            return response

        except Exception as e:
            self._stats["failed_analyses"] += 1
            logger.error(
                f"Lỗi phân tích kết quả đo cho bệnh nhân {request.patientId}: {e}"
            )
            raise ServiceError(f"Lỗi phân tích kết quả đo: {e}")

    def _build_prompt(self, request: AnalyzeMeasurementRequest) -> str:
        """Tạo prompt phân tích từ dữ liệu yêu cầu."""
        return MEASUREMENT_NOTE_PROMPT.format(
            measurementType=request.measurementType,
            value=request.value,
            time=request.time,
            context=request.context or "Không rõ",
            note=request.note or "Không có ghi chú.",
        )
