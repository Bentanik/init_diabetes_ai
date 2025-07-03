"""Quản lý các tính năng của ứng dụng."""

from typing import Dict, Type, Optional, TypeVar, Generic

from core.logging_config import get_logger
from features.interfaces.base_interface import BaseInterface
from features.measurement.measurement_analysis import MeasurementAnalysis
from features.careplan.careplan_generator import CarePlanGenerator

logger = get_logger(__name__)
T = TypeVar("T", bound=BaseInterface)


class FeatureManager:
    """Quản lý tất cả các tính năng của ứng dụng."""

    def __init__(self):
        """Khởi tạo feature manager."""
        self._features: Dict[str, BaseInterface] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Khởi tạo tất cả các tính năng."""
        if self._initialized:
            return

        try:
            await self._init_feature("measurement", MeasurementAnalysis())
            await self._init_feature("careplan", CarePlanGenerator())
            self._initialized = True
            logger.info("Khởi tạo tất cả tính năng thành công")
        except Exception as e:
            logger.error(f"Lỗi khởi tạo tính năng: {e}")
            await self.shutdown()
            raise

    async def shutdown(self) -> None:
        """Giải phóng tài nguyên của tất cả tính năng."""
        for name, feature in self._features.items():
            try:
                await feature.shutdown()
                logger.info(f"Đã giải phóng tính năng {name}")
            except Exception as e:
                logger.error(f"Lỗi giải phóng tính năng {name}: {e}")

        self._features.clear()
        self._initialized = False

    async def _init_feature(self, name: str, feature: BaseInterface) -> None:
        """Khởi tạo một tính năng."""
        try:
            await feature.initialize()
            self._features[name] = feature
            logger.info(f"Khởi tạo tính năng {name} thành công")
        except Exception as e:
            logger.error(f"Lỗi khởi tạo tính năng {name}: {e}")
            raise

    def get_feature(self, feature_type: Type[T]) -> T:
        """Lấy instance của một tính năng theo kiểu."""
        for feature in self._features.values():
            if isinstance(feature, feature_type):
                return feature  # type: ignore[return-value]
        raise RuntimeError(f"Tính năng {feature_type.__name__} chưa được khởi tạo")

    def get_measurement(self) -> MeasurementAnalysis:
        """Lấy instance của tính năng phân tích kết quả đo."""
        return self.get_feature(MeasurementAnalysis)

    def get_careplan(self) -> CarePlanGenerator:
        """Lấy instance của tính năng tạo kế hoạch chăm sóc."""
        return self.get_feature(CarePlanGenerator)


# Global instance
_manager = None


def get_feature_manager() -> FeatureManager:
    """Lấy global instance của feature manager."""
    global _manager
    if _manager is None:
        _manager = FeatureManager()
    return _manager
