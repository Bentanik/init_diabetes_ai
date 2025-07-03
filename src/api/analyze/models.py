from typing import Optional

from pydantic import BaseModel


class AnalyzeMeasurementRequest(BaseModel):
    """Mô hình yêu cầu để phân tích kết quả đo của bệnh nhân.

    Thuộc tính:
        patientId: Mã định danh bệnh nhân duy nhất
        measurementType: Loại đo lường (ví dụ: "Blood Glucose", "Blood Pressure")
        value: Giá trị đo với đơn vị (ví dụ: "7.2 mmol/L", "145/90 mmHg")
        time: Thời gian đo theo định dạng 24h (ví dụ: "07:30", "21:30")
        context: Bối cảnh đo lường (ví dụ: "fasting", "after lunch", "resting")
        note: Ghi chú của bệnh nhân về các điều kiện ảnh hưởng đến kết quả đo
    """

    patientId: str
    measurementType: str
    value: str
    time: str
    context: Optional[str] = None
    note: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "patientId": "P001",
                "measurementType": "Blood Glucose",
                "value": "7.2 mmol/L",
                "time": "07:30",
                "context": "fasting",
                "note": "Ăn tối muộn, ngủ không đủ giấc, căng thẳng công việc",
            }
        }


class AnalyzeMeasurementResponse(BaseModel):
    """Mô hình phản hồi cho phản hồi phân tích kết quả đo.

    Thuộc tính:
        patientId: Mã định danh bệnh nhân
        recordTime: Dấu thời gian ISO khi thực hiện phân tích
        feedback: Phản hồi cá nhân hóa được tạo bởi AI bằng tiếng Việt
    """

    patientId: str
    recordTime: str
    feedback: str

    class Config:
        schema_extra = {
            "example": {
                "patientId": "P001",
                "recordTime": "2024-01-15T08:30:00.000Z",
                "feedback": "Chỉ số đường huyết 7.2 mmol/L lúc đói của bạn hơi cao hơn mức bình thường (< 7.0). Có thể do bạn ăn tối muộn và căng thẳng công việc. Hãy thử ăn tối sớm hơn, tránh thức khuya và tập thể dục nhẹ buổi tối. Bạn đang cố gắng rất tốt, cứ tiếp tục theo dõi nhé!",
            }
        }
