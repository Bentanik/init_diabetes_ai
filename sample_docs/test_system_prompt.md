# Test System Prompt trong Global Settings

## Hướng dẫn sử dụng System Prompt

### 1. Cập nhật System Prompt

Gọi API để set system prompt mặc định:

```bash
curl -X PATCH "http://localhost:8000/api/rag/settings/system-prompt" \
  -H "Content-Type: application/json" \
  -d "Bạn là một bác sĩ chuyên khoa nội tiết. Hãy tư vấn về bệnh tiểu đường dựa trên thông tin y khoa được cung cấp. Context: {{context}}. Câu hỏi: {{query}}. Hãy đưa ra lời khuyên chuyên nghiệp và phù hợp."
```

### 2. Kiểm tra Settings

```bash
curl -X GET "http://localhost:8000/api/rag/settings"
```

### 3. Test Chat với System Prompt

```bash
curl -X POST "http://localhost:8000/api/rag/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user_123",
    "query": "Tôi bị tiểu đường type 2, nên ăn gì?"
  }'
```

## Thứ tự ưu tiên của System Prompt

1. **Global Settings System Prompt** (cao nhất)
2. **Prompt Template** (nếu không có global system prompt)
3. **Default System Prompt** (fallback)

## Variables hỗ trợ trong System Prompt

- `{{context}}`: Kết quả tìm kiếm từ knowledge bases
- `{{query}}`: Câu hỏi của user
- `{{language}}`: Ngôn ngữ (vi/en)
- `{{user_id}}`: ID của user
- `{{chat_history}}`: Lịch sử chat gần đây
- Plus các variables từ template example values (nếu có)
