"""Models cho RAG API - Request và Response schemas."""

from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import uuid


class FileInfoModel(BaseModel):
    """Model cho thông tin file."""

    filename: str = Field(description="Tên file")
    file_size: int = Field(description="Kích thước file (bytes)")
    file_extension: str = Field(description="Extension của file")
    content_type: str = Field(description="MIME type của file")
    upload_time: str = Field(description="Thời gian upload")
    storage_path: Optional[str] = Field(description="Đường dẫn lưu trữ trong MinIO")
    storage_time: Optional[str] = Field(description="Thời gian lưu trữ trong MinIO")


class FileUploadResponse(BaseModel):
    """Response model cho file upload và processing."""

    success: bool = Field(description="Trạng thái thành công")
    message: str = Field(description="Thông điệp mô tả kết quả")
    file_info: FileInfoModel = Field(description="Thông tin file đã upload")
    document_ids: List[str] = Field(
        description="Danh sách document_ids đã được process"
    )
    statistics: Dict[str, Any] = Field(description="Thống kê processing")
    processing_time: float = Field(description="Thời gian xử lý (giây)")


class DocumentInfo(BaseModel):
    """Model cho thông tin chi tiết của một document."""

    filename: str = Field(description="Tên file")
    size: int = Field(description="Kích thước file (bytes)")
    last_modified: Optional[str] = Field(description="Thời gian chỉnh sửa cuối")
    content_type: str = Field(description="Loại file")


class CollectionStats(BaseModel):
    """Model cho thống kê documents trong một collection."""

    total_documents: int = Field(description="Tổng số documents")
    total_size_bytes: int = Field(description="Tổng dung lượng (bytes)")
    file_types: Dict[str, int] = Field(description="Số lượng file theo loại")
    documents: List[DocumentInfo] = Field(
        description="Danh sách chi tiết các documents"
    )
    collection_name: str = Field(description="Tên collection")
    last_updated: str = Field(description="Thời gian cập nhật thống kê")


class ChunkModel(BaseModel):
    """Model cho một chunk."""

    content: str = Field(description="Nội dung chunk")
    chunk_id: int = Field(description="ID của chunk")
    global_chunk_id: str = Field(description="Global unique ID của chunk")
    content_length: int = Field(description="Độ dài nội dung")
    metadata: Dict[str, Any] = Field(description="Metadata của chunk")


class ProcessingStats(BaseModel):
    """Model cho thống kê processing."""

    total_files_processed: int = Field(description="Tổng số file đã xử lý")
    total_documents_created: int = Field(description="Tổng số document đã tạo")
    total_chunks_created: int = Field(description="Tổng số chunk đã tạo")
    processing_errors: int = Field(description="Số lỗi processing")
    success_rate: float = Field(description="Tỷ lệ thành công (%)")
    avg_chunks_per_file: float = Field(description="Trung bình chunk trên file")
    last_processing_time: Optional[str] = Field(description="Thời gian processing cuối")


class ErrorResponse(BaseModel):
    """Response model cho lỗi."""

    success: bool = False
    error: str = Field(description="Thông điệp lỗi")
    error_code: str = Field(description="Mã lỗi")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Chi tiết lỗi")


class KnowledgeBaseCreate(BaseModel):
    """Schema cho request tạo knowledge base."""

    name: str = Field(
        ..., description="Tên của knowledge base", min_length=1, max_length=255
    )
    description: Optional[str] = Field(None, description="Mô tả về knowledge base")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata bổ sung")


class KnowledgeBaseUpdate(BaseModel):
    """Schema cho request cập nhật knowledge base."""

    name: Optional[str] = Field(
        None, description="Tên mới của knowledge base", min_length=1, max_length=255
    )
    description: Optional[str] = Field(None, description="Mô tả mới về knowledge base")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata mới")


class KnowledgeBaseResponse(BaseModel):
    """Schema cho response knowledge base."""

    name: str = Field(description="Tên của knowledge base")
    description: Optional[str] = Field(None, description="Mô tả về knowledge base")
    created_at: str = Field(description="Thời gian tạo")
    updated_at: Optional[str] = Field(None, description="Thời gian cập nhật mới nhất")
    document_count: int = Field(
        default=0, description="Số lượng tài liệu trong knowledge base"
    )
    total_size_mb: float = Field(
        default=0.0, description="Tổng dung lượng tài liệu (MB)"
    )

    class Config:
        from_attributes = True


class KnowledgeBaseList(BaseModel):
    """Schema cho danh sách knowledge bases."""

    knowledge_bases: List[KnowledgeBaseResponse] = Field(
        description="Danh sách knowledge bases"
    )
    total: int = Field(description="Tổng số knowledge bases")


class MultiCollectionSearchRequest(BaseModel):
    """Model cho yêu cầu tìm kiếm trên nhiều knowledge bases."""

    query: str = Field(description="Câu hỏi/query cần tìm kiếm")
    collection_names: List[str] = Field(
        description="Danh sách các knowledge bases cần tìm kiếm"
    )
    top_k: Optional[int] = Field(default=5, description="Số lượng kết quả trả về")
    score_threshold: Optional[float] = Field(
        default=0.3, description="Ngưỡng điểm tối thiểu (0-1)", ge=0.0, le=1.0
    )


class SearchResult(BaseModel):
    """Model cho một kết quả tìm kiếm."""

    content: str = Field(description="Nội dung đoạn văn bản")
    metadata: Dict[str, Any] = Field(description="Metadata của đoạn văn bản")
    score: float = Field(description="Điểm tương đồng")
    collection_name: str = Field(description="Knowledge base chứa kết quả này")


class MultiCollectionSearchResponse(BaseModel):
    """Model cho kết quả tìm kiếm từ nhiều knowledge bases."""

    results: List[SearchResult] = Field(description="Danh sách kết quả tìm kiếm")
    total_results: int = Field(description="Tổng số kết quả tìm được")
    processing_time: float = Field(description="Thời gian xử lý (giây)")
    collection_stats: Dict[str, Any] = Field(
        description="Thống kê theo từng collection"
    )


class PromptTemplateIdea(BaseModel):
    """Schema cho request gợi ý prompt template từ ý tưởng."""

    idea: str = Field(description="Ý tưởng/mục đích của prompt template")
    context: Optional[Dict[str, Any]] = Field(
        None, description="Context bổ sung cho việc gợi ý"
    )
    language: str = Field(
        default="vi", description="Ngôn ngữ muốn sinh prompt template"
    )


class PromptTemplateSuggestion(BaseModel):
    """Schema cho response gợi ý prompt template."""

    suggested_template: str = Field(description="Template được gợi ý")
    suggested_name: str = Field(description="Tên được gợi ý cho template")
    suggested_description: str = Field(description="Mô tả được gợi ý cho template")
    detected_variables: List[str] = Field(
        description="Các biến được phát hiện trong template"
    )
    example_values: Dict[str, str] = Field(description="Giá trị mẫu cho các biến")
    generation_time: str = Field(description="Thời gian sinh gợi ý")


class PromptTemplateCreate(BaseModel):
    """Schema cho request tạo prompt template."""

    name: str = Field(description="Tên của template")
    description: Optional[str] = Field(None, description="Mô tả về template")
    template: str = Field(description="Nội dung template với các placeholders")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata bổ sung")
    example_values: Optional[Dict[str, str]] = Field(
        None, description="Giá trị mẫu cho các biến"
    )


class PromptTemplateUpdate(BaseModel):
    """Schema cho request cập nhật prompt template."""

    name: Optional[str] = Field(None, description="Tên mới của template")
    description: Optional[str] = Field(None, description="Mô tả mới về template")
    template: Optional[str] = Field(None, description="Nội dung template mới")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata mới")
    example_values: Optional[Dict[str, str]] = Field(
        None, description="Giá trị mẫu mới cho các biến"
    )


class PromptTemplateList(BaseModel):
    """Schema cho danh sách prompt templates."""

    templates: List["PromptTemplate"] = Field(description="Danh sách templates")
    total: int = Field(description="Tổng số templates")


class PromptGeneration(BaseModel):
    """Schema cho request sinh prompt từ template."""

    template_id: str = Field(description="ID của template cần sử dụng")
    variables: Dict[str, Any] = Field(description="Giá trị cho các biến trong template")
    context: Optional[Dict[str, Any]] = Field(None, description="Context bổ sung")


class GeneratedPrompt(BaseModel):
    """Schema cho response prompt đã được sinh."""

    prompt: str = Field(description="Prompt đã được sinh")
    template_id: str = Field(description="ID của template đã sử dụng")
    variables_used: Dict[str, Any] = Field(description="Các biến đã sử dụng")
    generation_time: str = Field(description="Thời gian sinh prompt")


class PromptTemplate(BaseModel):
    """Model cho một prompt template."""

    id: Optional[str] = Field(None, description="ID của template")
    name: str = Field(description="Tên của template")
    description: Optional[str] = Field(None, description="Mô tả về template")
    template: str = Field(description="Nội dung template với các placeholders")
    variables: List[str] = Field(description="Danh sách các biến trong template")
    example_values: Dict[str, str] = Field(
        default_factory=dict, description="Giá trị mẫu cho các biến"
    )
    created_at: Optional[str] = Field(None, description="Thời gian tạo")
    updated_at: Optional[str] = Field(None, description="Thời gian cập nhật cuối")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata bổ sung")


class ChatSettings(BaseModel):
    """Settings cho một chat session"""

    collection_names: List[str] = Field(
        default_factory=list, description="Danh sách knowledge bases được sử dụng"
    )
    prompt_template_id: Optional[str] = Field(
        None, description="ID của prompt template được sử dụng"
    )
    temperature: float = Field(0.7, description="Temperature cho LLM")
    max_tokens: int = Field(2000, description="Số tokens tối đa cho mỗi response")
    language: str = Field("vi", description="Ngôn ngữ sử dụng trong chat")
    search_kwargs: Dict[str, Any] = Field(
        default_factory=lambda: {
            "k": 5,
            "score_threshold": 0.3,
            "bm25_weight": 0.3,
            "vector_weight": 0.7,
        },
        description="Các tham số cho việc tìm kiếm",
    )

    @classmethod
    def get_default(cls) -> "ChatSettings":
        """Get default settings instance"""
        return cls(
            prompt_template_id="",
            temperature=0.7,
            max_tokens=2000,
            language="vi",
        )


class ChatMessage(BaseModel):
    """Message trong một chat session"""

    role: str = Field(..., description="Vai trò của message (user/assistant/system)")
    content: str = Field(..., description="Nội dung của message")
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Metadata bổ sung"
    )
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class ChatSession(BaseModel):
    """Một chat session"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    settings: ChatSettings = Field(
        default_factory=ChatSettings.get_default, description="Settings cho session"
    )
    messages: List[ChatMessage] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Metadata bổ sung"
    )


class ChatRequest(BaseModel):
    """Request để gửi message trong chat"""

    session_id: str
    message: str
    metadata: Optional[Dict[str, Any]] = None


class SimpleChatRequest(BaseModel):
    """Request đơn giản cho chat - chỉ cần userId và query"""

    user_id: str = Field(description="ID của user")
    query: str = Field(description="Câu hỏi/message của user")


class ChatResponse(BaseModel):
    """Response cho một chat message"""

    message: ChatMessage
    # context: List[SearchResult] = Field(
    #     default_factory=list, description="Context được sử dụng để generate response"
    # )
    # metadata: Optional[Dict[str, Any]] = None


class PromptTemplateInfo(BaseModel):
    """Thông tin đầy đủ của một prompt template"""

    id: str
    name: str
    description: Optional[str] = None
    template: str
    variables: List[str]
    example_values: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: str
    updated_at: Optional[str] = None


class GlobalSettings(BaseModel):
    """Global settings cho toàn bộ hệ thống"""

    id: str = Field(default="global", description="ID của settings")

    # Language settings
    default_language: str = Field("vi", description="Ngôn ngữ mặc định")

    # System prompt settings
    system_prompt: Optional[str] = Field(
        None, description="System prompt mặc định cho chat"
    )

    # LLM settings
    llm_settings: Dict[str, Any] = Field(
        default_factory=lambda: {
            "temperature": 0.7,
            "max_tokens": 2000,
        },
        description="Settings cho LLM",
    )

    # Search settings
    search_settings: Dict[str, Any] = Field(
        default_factory=lambda: {
            "k": 5,
            "score_threshold": 0.3,
            "bm25_weight": 0.3,
            "vector_weight": 0.7,
        },
        description="Settings cho search",
    )

    # Knowledge base settings
    available_collections: List[str] = Field(
        default_factory=list, description="Danh sách các knowledge bases có thể sử dụng"
    )
    default_collections: List[str] = Field(
        default_factory=list,
        description="Danh sách knowledge bases mặc định cho chat mới",
    )

    # Prompt template settings
    available_prompt_templates: List[PromptTemplateInfo] = Field(
        default_factory=list,
        description="Danh sách đầy đủ các prompt templates có thể sử dụng",
    )
    default_prompt_template_id: Optional[str] = Field(
        None, description="ID của prompt template mặc định"
    )

    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class SettingsUpdate(BaseModel):
    """Model cho việc update settings"""

    default_language: Optional[str] = None
    system_prompt: Optional[str] = None
    llm_settings: Optional[Dict[str, Any]] = None
    search_settings: Optional[Dict[str, Any]] = None
    available_collections: Optional[List[str]] = None
    default_collections: Optional[List[str]] = None
    default_prompt_template_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
