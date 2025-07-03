"""Interface cho tính năng tạo kế hoạch chăm sóc."""

from abc import abstractmethod
from typing import List

from features.interfaces.base_interface import BaseInterface
from api.careplan.models import CarePlanRequest, CarePlanMeasurementResponse


class CarePlanInterface(BaseInterface):
    """Interface cho tính năng tạo kế hoạch chăm sóc."""

    @abstractmethod
    async def generate_plan(
        self, request: CarePlanRequest
    ) -> List[CarePlanMeasurementResponse]:
        """Tạo kế hoạch chăm sóc dựa trên dữ liệu bệnh nhân."""
        pass

    @abstractmethod
    def validate_request(self, request: CarePlanRequest) -> None:
        """Xác thực dữ liệu yêu cầu kế hoạch chăm sóc."""
        pass
