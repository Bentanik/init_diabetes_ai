from typing import List, Optional

from pydantic import BaseModel


class CarePlanRequest(BaseModel):
    """Mô hình yêu cầu để tạo kế hoạch chăm sóc cá nhân.

    Thuộc tính:
        patientId: Mã định danh bệnh nhân duy nhất
        age: Tuổi bệnh nhân tính bằng năm (1-120)
        gender: Giới tính bệnh nhân
        bmi: Chỉ số khối cơ thể (10.0-50.0)
        diabetesType: Loại tiểu đường
        insulinSchedule: Lịch trình insulin hiện tại
        treatmentMethod: Phương pháp điều trị chính
        complications: Danh sách biến chứng tiểu đường
        pastDiseases: Danh sách bệnh lý đã mắc trước đây
        lifestyle: Mô tả lối sống của bệnh nhân
    """

    patientId: str
    age: int
    gender: str
    bmi: float
    diabetesType: str
    insulinSchedule: str
    treatmentMethod: str
    complications: List[str]
    pastDiseases: List[str]
    lifestyle: str

    class Config:
        schema_extra = {
            "example": {
                "patientId": "P001",
                "age": 45,
                "gender": "Nam",
                "bmi": 23.5,
                "diabetesType": "Type 2",
                "insulinSchedule": "Twice daily",
                "treatmentMethod": "Insulin + Metformin",
                "complications": ["Hypertension", "Retinopathy"],
                "pastDiseases": ["Heart disease"],
                "lifestyle": "Sedentary work, irregular meals, high stress",
            }
        }


class CarePlanMeasurementResponse(BaseModel):
    """Mô hình phản hồi cho khuyến nghị đo lường trong kế hoạch chăm sóc.

    Thuộc tính:
        recordType: Loại đo lường (ví dụ: "BloodGlucose", "BloodPressure")
        period: Thời điểm đo (ví dụ: "before_breakfast", "morning")
        subtype: Bối cảnh đo lường cụ thể (ví dụ: "fasting", "sitting")
        reason: Giải thích chi tiết bằng tiếng Việt về lý do khuyến nghị đo lường này
    """

    recordType: str
    period: str
    subtype: Optional[str]
    reason: str

    class Config:
        schema_extra = {
            "example": {
                "recordType": "BloodGlucose",
                "period": "before_breakfast",
                "subtype": "fasting",
                "reason": "Theo dõi đường huyết lúc đói để đánh giá hiệu quả điều trị insulin ban đêm và khả năng kiểm soát glucose tự nhiên của cơ thể.",
            }
        }
