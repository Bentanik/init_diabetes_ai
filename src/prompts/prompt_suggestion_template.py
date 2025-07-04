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
5. IMPORTANT: The template should match the user's idea and domain

IMPORTANT: Return ONLY the template content, nothing else. The template should:
1. Use {{variable_name}} for variables
2. Use \\n for newlines
3. Match the user's language (vi/en)
4. Be specific to the user's idea
5. NO JSON, NO explanation, ONLY the template text

Example for Vietnamese (vi):
Phân tích {{topic}} sau:\\n{{content}}\\n\\nHãy đưa ra các nhận xét sau:\\n1. Điểm chính\\n2. Ưu điểm\\n3. Nhược điểm\\n4. Đề xuất cải thiện

Example for English (en):
Analyze the following {{topic}}:\\n{{content}}\\n\\nPlease provide the following analysis:\\n1. Key points\\n2. Strengths\\n3. Weaknesses\\n4. Improvement suggestions"""

# Default templates for different languages
DEFAULT_TEMPLATES = {
    "vi": {
        "suggested_template": "Phân tích {{topic}} sau:\n{{content}}\n\nHãy đưa ra các nhận xét sau:\n1. Điểm chính\n2. Ưu điểm\n3. Nhược điểm\n4. Đề xuất cải thiện",
        "suggested_name": "general_analysis",
        "suggested_description": "Template để phân tích và đánh giá một chủ đề hoặc nội dung bất kỳ.",
        "detected_variables": ["topic", "content"],
        "example_values": {
            "topic": "Báo cáo doanh thu Q3/2023",
            "content": "- Doanh thu tăng 15%\n- Chi phí giảm 5%\n- Khách hàng mới tăng 20%",
        },
    },
    "en": {
        "suggested_template": "Analyze the following {{topic}}:\n{{content}}\n\nPlease provide the following analysis:\n1. Key points\n2. Strengths\n3. Weaknesses\n4. Improvement suggestions",
        "suggested_name": "general_analysis",
        "suggested_description": "Template for analyzing and evaluating any topic or content.",
        "detected_variables": ["topic", "content"],
        "example_values": {
            "topic": "Q3/2023 Revenue Report",
            "content": "- Revenue up 15%\n- Costs down 5%\n- New customers up 20%",
        },
    },
}
