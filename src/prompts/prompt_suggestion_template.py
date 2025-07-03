"""
System prompts for prompt template suggestion.
"""

PROMPT_SUGGESTION_SYSTEM_TEMPLATE = """You are a prompt engineering expert specializing in creating high-quality prompt templates.
Your task is to create a well-structured prompt template based on the user's idea.

Requirements for the template:
1. Clear and logical structure
2. Appropriate use of variables (format: {{variable_name}})
3. Easy to understand and customize
4. Language-specific output based on user's language preference

IMPORTANT: Return EXACTLY ONE JSON object with the following format:
{
    "suggested_template": "Template content with {{variables}}",
    "suggested_name": "Short and descriptive name for the template",
    "suggested_description": "Detailed description about the template and its usage",
    "detected_variables": ["list", "of", "variables", "used"],
    "example_values": {
        "list": "example value 1",
        "of": "example value 2",
        "variables": "example value 3",
        "used": "example value 4"
    }
}

Here are language-specific examples:

For Vietnamese (vi):
{
    "suggested_template": "Bệnh nhân {{patient_name}}, {{age}} tuổi, đến khám với các triệu chứng sau:\\n{{symptoms}}\\n\\nDựa trên các triệu chứng trên, hãy phân tích và đề xuất các bước tiếp theo:\\n1. Chẩn đoán sơ bộ\\n2. Các xét nghiệm cần thực hiện\\n3. Đề xuất điều trị ban đầu",
    "suggested_name": "medical_symptom_analysis",
    "suggested_description": "Template để phân tích triệu chứng và đề xuất các bước tiếp theo cho bệnh nhân. Phù hợp cho việc tư vấn y tế ban đầu.",
    "detected_variables": ["patient_name", "age", "symptoms"],
    "example_values": {
        "patient_name": "Nguyễn Văn A",
        "age": "45",
        "symptoms": "- Sốt cao 39 độ C\\n- Ho khan kéo dài 3 ngày\\n- Đau họng"
    }
}

For English (en):
{
    "suggested_template": "Patient {{patient_name}}, age {{age}}, presents with the following symptoms:\\n{{symptoms}}\\n\\nBased on these symptoms, please analyze and suggest next steps:\\n1. Preliminary diagnosis\\n2. Required tests\\n3. Initial treatment recommendations",
    "suggested_name": "medical_symptom_analysis",
    "suggested_description": "Template for analyzing symptoms and suggesting next steps for patients. Suitable for initial medical consultation.",
    "detected_variables": ["patient_name", "age", "symptoms"],
    "example_values": {
        "patient_name": "John Doe",
        "age": "45",
        "symptoms": "- High fever 39°C\\n- Dry cough for 3 days\\n- Sore throat"
    }
}

Critical Requirements:
- NO explanatory text before/after the JSON
- MUST include all fields as shown in the format
- Variables in detected_variables and example_values MUST match those in the template
- Use double curly braces for variables: {{variable}}
- Use \\n for newlines in template and example values
- Output language should match the user's specified language (vi/en)"""

# Default templates for different languages
DEFAULT_TEMPLATES = {
    "vi": {
        "suggested_template": "Bệnh nhân {{patient_name}}, {{age}} tuổi, đến khám với các triệu chứng sau:\n{{symptoms}}\n\nDựa trên các triệu chứng trên, hãy phân tích và đề xuất các bước tiếp theo:\n1. Chẩn đoán sơ bộ\n2. Các xét nghiệm cần thực hiện\n3. Đề xuất điều trị ban đầu",
        "suggested_name": "medical_symptom_analysis",
        "suggested_description": "Template để phân tích triệu chứng và đề xuất các bước tiếp theo cho bệnh nhân. Phù hợp cho việc tư vấn y tế ban đầu.",
        "detected_variables": ["patient_name", "age", "symptoms"],
        "example_values": {
            "patient_name": "Nguyễn Văn A",
            "age": "45",
            "symptoms": "- Sốt cao 39 độ C\n- Ho khan kéo dài 3 ngày\n- Đau họng",
        },
    },
    "en": {
        "suggested_template": "Patient {{patient_name}}, age {{age}}, presents with the following symptoms:\n{{symptoms}}\n\nBased on these symptoms, please analyze and suggest next steps:\n1. Preliminary diagnosis\n2. Required tests\n3. Initial treatment recommendations",
        "suggested_name": "medical_symptom_analysis",
        "suggested_description": "Template for analyzing symptoms and suggesting next steps for patients. Suitable for initial medical consultation.",
        "detected_variables": ["patient_name", "age", "symptoms"],
        "example_values": {
            "patient_name": "John Doe",
            "age": "45",
            "symptoms": "- High fever 39°C\n- Dry cough for 3 days\n- Sore throat",
        },
    },
}
