from api.careplan.models import CarePlanRequest


def build_prompt(request: CarePlanRequest) -> str:
    return f"""
You are an intelligent assistant for diabetes care.

Based on the patient's profile below, generate a personalized list of recommended measurement schedules.

Each item must include:
- What to measure (recordType)
- When to measure (period)
- Under what condition (subtype), if applicable
- Why to measure (reason)

⚠️ REQUIREMENTS for `reason`:
- Absolutely **no mention of AI, virtual assistants, suggestion models, or intelligent systems**
- Avoid casual or chat-like phrasing, don't write like you're chatting
- Use a **cause-and-effect medical explanation style**, not just descriptive or generic goals like "to monitor health"
- Reasons should be **cause–effect analysis**, not simply descriptive
- Written in **Vietnamese**
- Must be **specific and clinically justified** based on the patient's individual profile
- Do NOT use generic explanations like “to monitor health” or “to check levels”
- Consider: diabetes type, insulin usage, treatment method, complications, past diseases, BMI, age, gender, and lifestyle
- Reason must be concise, easy to understand, and no more than **150 words**

⚠️ OUTPUT FORMAT:
Respond with **only a JSON array** in this exact structure, without any extra explanation:

[
  {{
    "recordType": one of ["BloodGlucose", "BloodPressure"],
    "period": 
        For BloodGlucose:
          one of ["before_breakfast", "after_breakfast", "before_lunch", "after_lunch", "before_dinner", "after_dinner", "before_sleep"]
        For BloodPressure:
          one of ["morning", "noon", "evening", "before_sleep"]
    "subtype":
        For BloodGlucose:
          one of ["fasting", "postprandial", "null"]
        For BloodPressure:
          one of ["resting", "sitting", "standing", "null"]
        For other types: null
    "reason": The clinical reason for measurement, written in **Vietnamese**, clearly tailored to the patient's condition, concise, easy to understand, and limited to a maximum of 150 words.
  }},
  ...
]

🚫 Do NOT include any explanation before or after the JSON array.

---

🧑‍⚕️ PATIENT PROFILE:
- ID: {request.patientId}
- Age: {request.age}
- Gender: {request.gender}
- BMI: {request.bmi}
- Diabetes type: {request.diabetesType}
- Insulin schedule: {request.insulinSchedule}
- Treatment method: {request.treatmentMethod}
- Complications: {", ".join(request.complications)}
- Past diseases: {", ".join(request.pastDiseases) or "None"}
- Lifestyle: {request.lifestyle}

---
Please generate a clinically sound and personalized measurement plan based on this information.
"""
