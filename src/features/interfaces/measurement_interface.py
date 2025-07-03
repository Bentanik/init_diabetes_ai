"""Interface cho tính năng phân tích kết quả đo."""

from abc import abstractmethod
from typing import Dict, Any

from features.interfaces.base_interface import BaseInterface
from api.analyze.models import AnalyzeMeasurementRequest, AnalyzeMeasurementResponse


class MeasurementAnalysisInterface(BaseInterface):
    """Interface cho tính năng phân tích kết quả đo."""

    @abstractmethod
    async def analyze_measurement(
        self, request: AnalyzeMeasurementRequest
    ) -> AnalyzeMeasurementResponse:
        """Phân tích kết quả đo và đưa ra phản hồi."""
        pass

    @abstractmethod
    def get_supported_types(self) -> Dict[str, Any]:
        """Lấy danh sách các loại đo được hỗ trợ và quy tắc xác thực."""
        pass
