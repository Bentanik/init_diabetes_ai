# RECORD_TYPES = ["BloodGlucose", "BloodPressure", "Weight", "Height", "HbA1c"]
RECORD_TYPES = ["BloodGlucose", "BloodPressure"]


PERIODS = [
    "before_breakfast",
    "after_breakfast",
    "before_lunch",
    "after_lunch",
    "before_dinner",
    "after_dinner",
    "before_sleep",
    "morning",
    "noon",
    "evening",
]

SUBTYPES_BY_RECORD_TYPE = {
    "BloodGlucose": ["fasting", "postprandial", "null"],  # đo lúc đói  # sau ăn
    "BloodPressure": ["resting", "sitting", "standing", "null"],
    "Weight": ["null"],
    "Height": ["null"],
    "HbA1c": ["null"],
}
