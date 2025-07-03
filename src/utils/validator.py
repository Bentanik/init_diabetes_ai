"""Module validation với xử lý lỗi và ghi log nâng cao."""

from typing import List, Dict, Any

from constants.careplan_schema import RECORD_TYPES, PERIODS, SUBTYPES_BY_RECORD_TYPE
from core.exceptions import ServiceError
from core.logging_config import get_logger
from api.careplan.models import CarePlanMeasurementResponse
from config.settings import get_config

logger = get_logger(__name__)


def validate_careplan_output(
    data: List[Dict[str, Any]]
) -> List[CarePlanMeasurementResponse]:
    """Kiểm tra tính hợp lệ của dữ liệu kế hoạch chăm sóc với xử lý lỗi toàn diện.

    Args:
        data: Dữ liệu kế hoạch chăm sóc thô từ LLM

    Returns:
        Danh sách các kế hoạch đo lường đã được kiểm tra

    Raises:
        ServiceError: Nếu việc kiểm tra thất bại
    """
    if not data:
        logger.warning("Nhận được dữ liệu kế hoạch chăm sóc rỗng")
        return []

    if not isinstance(data, list):
        logger.error(f"Dữ liệu kế hoạch chăm sóc không phải là list: {type(data)}")
        raise ServiceError("Dữ liệu kế hoạch chăm sóc phải là một list")

    validated = []

    for idx, item in enumerate(data):
        try:
            validated_item = _validate_single_measurement(item, idx)
            validated.append(validated_item)
        except ServiceError as e:
            logger.error(f"Kiểm tra thất bại cho mục {idx}: {e.message}")
            raise ServiceError(f"Kiểm tra thất bại cho mục {idx}: {e.message}")

    logger.info(f"Đã kiểm tra thành công {len(validated)} kế hoạch đo lường")
    return validated


def _validate_single_measurement(
    item: Dict[str, Any], index: int
) -> CarePlanMeasurementResponse:
    """Kiểm tra tính hợp lệ của một mục đo lường.

    Args:
        item: Dữ liệu của một mục đo lường
        index: Chỉ số của mục để báo cáo lỗi

    Returns:
        Phản hồi đo lường đã được kiểm tra

    Raises:
        ServiceError: Nếu việc kiểm tra thất bại
    """
    if not isinstance(item, dict):
        raise ServiceError(f"Mục {index} không phải là dictionary: {type(item)}")

    # Kiểm tra các trường bắt buộc
    record_type = item.get("recordType")
    period = item.get("period")
    subtype = item.get("subtype")
    reason = item.get("reason", "")

    # Kiểm tra recordType
    if not record_type:
        raise ServiceError(f"Thiếu trường recordType cho mục {index}")

    if record_type not in RECORD_TYPES:
        raise ServiceError(f"Loại bản ghi không hợp lệ: {record_type} cho mục {index}")

    # Kiểm tra period
    if not period:
        raise ServiceError(f"Thiếu trường period cho mục {index}")

    if period not in PERIODS:
        raise ServiceError(f"Thời điểm không hợp lệ: {period} cho mục {index}")

    # Kiểm tra subtype
    allowed_subtypes = SUBTYPES_BY_RECORD_TYPE.get(record_type, [])
    if subtype is not None and subtype not in allowed_subtypes:
        raise ServiceError(
            f"Loại phụ không hợp lệ: {subtype} cho loại bản ghi {record_type} ở mục {index}"
        )

    # Kiểm tra reason
    if not reason or not isinstance(reason, str):
        raise ServiceError(f"Thiếu trường reason hoặc không hợp lệ cho mục {index}")

    max_length = get_config("max_reason_length") or 150
    if len(reason) > max_length:
        logger.warning(f"Lý do quá dài cho mục {index}, sẽ cắt bớt")
        reason = reason[:max_length]

    # Tạo và trả về phản hồi đã được kiểm tra
    try:
        return CarePlanMeasurementResponse(
            recordType=record_type,
            period=period,
            subtype=subtype,
            reason=reason.strip(),
        )
    except Exception as e:
        raise ServiceError(f"Không thể tạo model phản hồi cho mục {index}: {e}")
