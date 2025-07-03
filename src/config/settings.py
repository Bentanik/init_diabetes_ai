"""Quản lý cấu hình."""

import os
from dotenv import load_dotenv
from typing import Dict, Any

from core.exceptions import ConfigError

# Load environment variables
load_dotenv()

# Cấu hình ứng dụng
config = {
    # API Configuration
    "app_title": "AI Service - CarePlan Generator",
    "app_version": "1.0.0",
    "app_description": "AI Service for diabetes care plan generation and measurement analysis",
    # LLM Configuration - Đơn giản chỉ cần 3 thứ
    "llm_base_url": os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
    "llm_api_key": os.getenv("LLM_API_KEY")
    or os.getenv("OPENROUTER_API_KEY"),  # Backward compatibility
    "llm_model": os.getenv("LLM_MODEL", "deepseek/deepseek-r1-distill-llama-70b:free"),
    # Common LLM Settings
    "default_temperature": float(os.getenv("LLM_TEMPERATURE", "0.3")),
    "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "2048")),
    # RAG Configuration
    "rag_embedding_provider": os.getenv(
        "RAG_EMBEDDING_PROVIDER", "local"
    ),  # local, openai
    "rag_embedding_model": os.getenv("RAG_EMBEDDING_MODEL"),  # auto-select if not set
    "rag_embedding_api_key": os.getenv("RAG_EMBEDDING_API_KEY")
    or os.getenv("OPENAI_API_KEY"),
    "rag_collection_name": os.getenv("RAG_COLLECTION_NAME", "default"),
    "rag_vectorstore_dir": os.getenv("RAG_VECTORSTORE_DIR", "./data/vectorstore"),
    "rag_chunk_size": int(os.getenv("RAG_CHUNK_SIZE", "1000")),
    "rag_chunk_overlap": int(os.getenv("RAG_CHUNK_OVERLAP", "200")),
    "rag_retrieval_k": int(os.getenv("RAG_RETRIEVAL_K", "5")),
    "rag_score_threshold": float(os.getenv("RAG_SCORE_THRESHOLD", "0.3")),
    # MinIO Configuration
    "minio_endpoint": os.getenv("MINIO_ENDPOINT", "localhost:9000"),
    "minio_access_key": os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
    "minio_secret_key": os.getenv("MINIO_SECRET_KEY", "minioadmin"),
    "minio_secure": os.getenv("MINIO_SECURE", "false").lower() == "true",
    # Validation limits
    "max_reason_length": 150,
    "max_feedback_length": 250,
}


# Hàm getter
def get_config(key: str, default=None):
    """Lấy giá trị cấu hình theo key."""
    return config.get(key, default)


def get_llm_config():
    """Lấy cấu hình LLM đơn giản."""
    return {
        "base_url": config["llm_base_url"],
        "api_key": config["llm_api_key"],
        "model": config["llm_model"],
        "temperature": config["default_temperature"],
        "max_tokens": config["max_tokens"],
    }


def get_rag_config():
    """Lấy cấu hình RAG."""
    return {
        "embedding_provider": config["rag_embedding_provider"],
        "embedding_model": config["rag_embedding_model"],
        "embedding_api_key": config["rag_embedding_api_key"],
        "collection_name": config["rag_collection_name"],
        "vectorstore_dir": config["rag_vectorstore_dir"],
        "chunk_size": config["rag_chunk_size"],
        "chunk_overlap": config["rag_chunk_overlap"],
        "retrieval_k": config["rag_retrieval_k"],
        "score_threshold": config["rag_score_threshold"],
    }


def get_minio_config():
    """Lấy cấu hình MinIO."""
    return {
        "endpoint": config["minio_endpoint"],
        "access_key": config["minio_access_key"],
        "secret_key": config["minio_secret_key"],
        "secure": config["minio_secure"],
    }


def get_api_key():
    """Lấy API key (để tương thích với code cũ)."""
    api_key = config["llm_api_key"]

    # Chỉ check nếu không phải localhost hoặc ollama
    base_url = config["llm_base_url"].lower()
    is_local = any(x in base_url for x in ["localhost", "127.0.0.1", "11434"])

    if not is_local and not api_key:
        raise ValueError(
            "Missing LLM_API_KEY for remote API. For localhost/ollama, API key is optional."
        )

    return api_key


def get_mongodb_config() -> Dict[str, Any]:
    """Lấy cấu hình MongoDB."""
    try:
        return {
            "uri": os.getenv("MONGODB_URI", "mongodb://localhost:27017"),
            "database": os.getenv("MONGODB_DATABASE", "diabetes_ai"),
        }
    except Exception as e:
        raise ConfigError(f"Lỗi khi lấy cấu hình MongoDB: {e}")
