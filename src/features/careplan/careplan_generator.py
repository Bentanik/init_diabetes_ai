"""Triển khai tính năng tạo kế hoạch chăm sóc."""

import json
from datetime import datetime
from typing import Dict, Any, List

from core.exceptions import ServiceError
from core.llm_client import get_llm
from core.logging_config import get_logger
from features.interfaces.careplan_interface import CarePlanInterface
from api.careplan.models import CarePlanRequest, CarePlanMeasurementResponse
from prompts.careplan_prompt import build_prompt
from utils.utils import extract_json

logger = get_logger(__name__)


class CarePlanGenerator(CarePlanInterface):
    """Triển khai tính năng tạo kế hoạch chăm sóc."""

    def __init__(self):
        """Khởi tạo tạo kế hoạch chăm sóc."""
        self._llm = None
        self._stats = {
            "total_plans": 0,
            "successful_plans": 0,
            "failed_plans": 0,
            "last_plan_time": None,
        }

    async def initialize(self) -> None:
        """Khởi tạo tài nguyên cho tạo kế hoạch chăm sóc."""
        try:
            self._llm = get_llm()
            logger.info("Khởi tạo tạo kế hoạch chăm sóc thành công")
        except Exception as e:
            logger.error(f"Lỗi khởi tạo tạo kế hoạch chăm sóc: {e}")
            raise

    async def shutdown(self) -> None:
        """Giải phóng tài nguyên của tạo kế hoạch chăm sóc."""
        self._llm = None
        logger.info("Đã giải phóng tạo kế hoạch chăm sóc")

    def get_stats(self) -> Dict[str, Any]:
        """Lấy thống kê hoạt động của tạo kế hoạch chăm sóc."""
        return self._stats.copy()

    def validate_request(self, request: CarePlanRequest) -> None:
        """Xác thực dữ liệu yêu cầu kế hoạch chăm sóc."""
        try:
            # Xác thực tuổi
            if not 1 <= request.age <= 120:
                raise ValueError("Tuổi phải từ 1 đến 120")

            # Xác thực BMI
            if not 10.0 <= request.bmi <= 50.0:
                raise ValueError("BMI phải từ 10.0 đến 50.0")

            # Xác thực giới tính
            if request.gender.lower() not in ["nam", "nữ", "khác"]:
                raise ValueError("Giới tính phải là: Nam, Nữ, hoặc Khác")

            # Xác thực loại tiểu đường
            if request.diabetesType.lower() not in ["type 1", "type 2", "gestational"]:
                raise ValueError(
                    "Loại tiểu đường phải là: Type 1, Type 2, hoặc Gestational"
                )

            logger.debug(
                f"Xác thực yêu cầu thành công cho bệnh nhân {request.patientId}"
            )

        except ValueError as e:
            logger.error(f"Lỗi xác thực yêu cầu: {e}")
            raise ServiceError(f"Yêu cầu không hợp lệ: {e}")

    async def generate_plan(
        self, request: CarePlanRequest
    ) -> List[CarePlanMeasurementResponse]:
        """Tạo kế hoạch chăm sóc dựa trên dữ liệu bệnh nhân."""
        try:
            logger.info(f"Đang tạo kế hoạch cho bệnh nhân: {request.patientId}")
            self._stats["total_plans"] += 1

            # Xác thực yêu cầu
            self.validate_request(request)

            # Tạo prompt
            prompt = build_prompt(request)
            logger.debug(f"Đã tạo prompt cho bệnh nhân {request.patientId}")

            # Gọi LLM
            if self._llm is None:
                logger.error("LLM chưa được khởi tạo")
                raise ServiceError("LLM chưa được khởi tạo")
            llm_response = await self._llm.generate(prompt)
            logger.debug(f"Đã nhận phản hồi cho bệnh nhân {request.patientId}")

            # Xử lý phản hồi JSON
            try:
                json_str = extract_json(llm_response)
                raw_data = json.loads(json_str)
                logger.debug(f"Đã xử lý JSON cho bệnh nhân {request.patientId}")
            except Exception as e:
                logger.error(f"Lỗi xử lý JSON cho bệnh nhân {request.patientId}: {e}")
                raise ServiceError(f"Lỗi xử lý phản hồi LLM: {e}")

            # Chuyển đổi sang model phản hồi
            try:
                measurements = [
                    CarePlanMeasurementResponse(**measurement)
                    for measurement in raw_data
                ]
                logger.info(
                    f"Đã tạo {len(measurements)} khuyến nghị cho bệnh nhân {request.patientId}"
                )

                self._stats["successful_plans"] += 1
                self._stats["last_plan_time"] = datetime.utcnow().isoformat()
                return measurements

            except Exception as e:
                logger.error(
                    f"Lỗi chuyển đổi phản hồi cho bệnh nhân {request.patientId}: {e}"
                )
                raise ServiceError(f"Lỗi chuyển đổi kế hoạch chăm sóc: {e}")

        except ServiceError:
            # Re-raise service errors
            self._stats["failed_plans"] += 1
            raise
        except Exception as e:
            self._stats["failed_plans"] += 1
            logger.error(f"Lỗi không mong muốn khi tạo kế hoạch: {e}")
            raise ServiceError(f"Lỗi không mong muốn khi tạo kế hoạch: {e}")
