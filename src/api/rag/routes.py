"""
RAG API routes cho knowledge base management và document processing
"""

import json
import os
import tempfile
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import uuid
import string
import re

from fastapi import (
    APIRouter,
    File,
    UploadFile,
    HTTPException,
    Form,
    Query,
    status,
    Depends,
    Request,
)
from fastapi.responses import JSONResponse
from motor.motor_asyncio import AsyncIOMotorDatabase
from pydantic import BaseModel

from features.rag.rag_pipeline import RAGPipeline, RAGPipelineConfig
from features.rag.chunking import ChunkingConfig
from features.rag.embedding import EmbeddingConfig, Embedding
from features.rag.vector_store import VectorStoreConfig, VectorStore
from features.rag.storage import document_storage
from features.rag.retrieval import (
    HybridRetriever,
    MultiCollectionConfig,
    create_multi_collection_retriever,
    HybridSearchConfig,
    MultiCollectionHybridRetriever,
)
from core.logging_config import get_logger
from core.llm_client import get_llm
from core.mongodb import get_mongodb
from features.prompt_template.prompt_template_storage import MongoPromptTemplateStorage
from utils.utils import extract_variables
from .models import (
    KnowledgeBaseCreate,
    KnowledgeBaseUpdate,
    KnowledgeBaseResponse,
    KnowledgeBaseList,
    FileUploadResponse,
    FileInfoModel,
    CollectionStats,
    MultiCollectionSearchRequest,
    MultiCollectionSearchResponse,
    SearchResult,
    PromptTemplate,
    PromptTemplateCreate,
    PromptTemplateList,
    PromptTemplateUpdate,
    GeneratedPrompt,
    PromptGeneration,
    PromptTemplateIdea,
    PromptTemplateSuggestion,
    ChatSession,
    ChatSettings,
    ChatResponse,
    ChatRequest,
    ChatMessage,
    GlobalSettings,
    SettingsUpdate,
    PromptTemplateInfo,
    SimpleChatRequest,
)
from utils.utils import validate_retriever_initialization, check_retriever_type
from features.rag.document_parser import DocumentParser
from prompts.prompt_suggestion_template import (
    PROMPT_SUGGESTION_SYSTEM_TEMPLATE,
    DEFAULT_TEMPLATES,
)

router = APIRouter(tags=["RAG Knowledge Base"])
logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".html", ".htm", ".csv", ".md"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

# Singleton pipeline
rag_pipeline: RAGPipeline | None = None


# Dependency để lấy MongoDB database instance
async def get_db() -> AsyncIOMotorDatabase:
    """Get MongoDB database instance."""
    return await get_mongodb()


# Dependency để lấy prompt template storage instance
async def get_prompt_template_storage(
    db: AsyncIOMotorDatabase = Depends(get_db),
) -> MongoPromptTemplateStorage:
    """Get prompt template storage instance."""
    storage = MongoPromptTemplateStorage(db)
    await storage.setup()
    return storage


def get_rag_pipeline(collection_name: str) -> RAGPipeline:
    """Get RAG pipeline instance với collection name cụ thể."""
    global rag_pipeline
    if not rag_pipeline:
        config = RAGPipelineConfig(
            chunking_config=ChunkingConfig(),
            embedding_config=EmbeddingConfig(),
            vector_store_config=VectorStoreConfig(collection_name=collection_name),
        )
        rag_pipeline = RAGPipeline(config)
    else:
        # Kiểm tra xem collection đã tồn tại chưa
        if collection_name not in rag_pipeline.vector_stores:
            # Tạo vector store mới cho collection
            config = VectorStoreConfig(collection_name=collection_name)
            vector_store = VectorStore(
                embeddings=rag_pipeline.embeddings, config=config
            )
            vector_store.create_collection()
            rag_pipeline.vector_stores[collection_name] = vector_store

    return rag_pipeline


def validate_file(file: UploadFile) -> tuple[bool, str]:
    """Validate file type và size"""
    if not file.filename:
        return False, "Filename is required"

    ext = os.path.splitext(file.filename.lower())[1]
    if ext not in SUPPORTED_EXTENSIONS:
        return (
            False,
            f"Unsupported file type. Supported: {', '.join(SUPPORTED_EXTENSIONS)}",
        )

    return True, "Valid"


def format_file_info(
    file: UploadFile, size: int, storage_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Format file info cho response"""
    info = {
        "filename": file.filename,
        "file_size": size,
        "file_extension": (
            os.path.splitext(file.filename.lower())[1] if file.filename else ""
        ),
        "content_type": file.content_type or "application/octet-stream",
        "upload_time": datetime.now().isoformat(),
    }

    if storage_info:
        info.update(
            {
                "storage_path": storage_info["storage_path"],
                "storage_time": storage_info["storage_time"],
            }
        )

    return info


@router.post(
    "/knowledge-bases",
    response_model=KnowledgeBaseResponse,
    summary="Tạo knowledge base mới",
    description="Tạo knowledge base mới trong vector store. Mỗi knowledge base là một collection riêng biệt.",
)
async def create_knowledge_base(
    kb_data: KnowledgeBaseCreate, db: AsyncIOMotorDatabase = Depends(get_db)
):
    """
    Tạo knowledge base mới trong vector store.

    - **name**: Tên của knowledge base, sẽ được dùng làm collection name
    - **description**: Mô tả về knowledge base (optional)
    - **metadata**: Metadata bổ sung (optional)
    """
    try:
        # Khởi tạo VectorStore với collection name mới
        config = VectorStoreConfig(collection_name=kb_data.name)
        vector_store = VectorStore(config=config)
        success = vector_store.create_collection(force_recreate=True)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Không thể tạo collection trong vector store",
            )

        # Lưu thông tin cơ bản của knowledge base vào MongoDB
        kb_info = {
            "name": kb_data.name,
            "description": kb_data.description,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "document_count": 0,
            "total_size_bytes": 0,
        }
        await db["knowledge_base"].update_one(
            {"name": kb_data.name}, {"$set": kb_info}, upsert=True
        )

        return KnowledgeBaseResponse(
            name=kb_data.name,
            description=kb_data.description,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            document_count=0,
            total_size_mb=0.0,
        )

    except Exception as e:
        logger.error(f"Error creating KB: {e}")
        raise HTTPException(500, detail=f"Lỗi khi tạo knowledge base: {str(e)}")


@router.get(
    "/knowledge-bases",
    response_model=KnowledgeBaseList,
    summary="Lấy danh sách knowledge bases",
    description="Lấy danh sách tất cả knowledge bases hiện có.",
)
async def list_knowledge_bases(db: AsyncIOMotorDatabase = Depends(get_db)):
    """
    Lấy danh sách tất cả knowledge bases với thông tin đầy đủ.
    """
    try:
        # Khởi tạo VectorStore không cần embeddings vì chỉ dùng để quản lý collections
        vector_store = VectorStore()
        collections = vector_store.list_collections()

        kbs = []
        for collection in collections:
            # Lấy metadata từ MongoDB
            kb_metadata = await db["knowledge_base"].find_one({"name": collection.name})

            # Lấy thông tin collection từ vector store
            vector_store.config.collection_name = collection.name
            info = vector_store.get_collection_info()

            # Lấy document stats từ MinIO
            try:
                doc_stats = document_storage.get_collection_stats(collection.name)
                document_count = doc_stats.get("total_documents", 0)
                total_size_bytes = doc_stats.get("total_size_bytes", 0)
                total_size_mb = round(total_size_bytes / (1024 * 1024), 2)
                # Tìm thời gian cập nhật mới nhất từ documents
                latest_update = doc_stats.get("last_updated")
            except Exception as e:
                logger.warning(
                    f"Could not get document stats for {collection.name}: {e}"
                )
                document_count = 0
                total_size_mb = 0.0
                latest_update = None

            # Xác định created_at và updated_at
            created_at = info.get("created_at")
            if not created_at:
                created_at = datetime.now().isoformat()

            # Ưu tiên updated_at từ document stats, sau đó từ metadata, cuối cùng là created_at
            updated_at = latest_update
            if not updated_at and kb_metadata:
                updated_at = kb_metadata.get("updated_at")
            if not updated_at:
                updated_at = created_at

            # Cập nhật document stats trong MongoDB metadata nếu cần
            if kb_metadata:
                # Cập nhật stats mới nhất vào MongoDB
                await db["knowledge_base"].update_one(
                    {"name": collection.name},
                    {
                        "$set": {
                            "document_count": document_count,
                            "total_size_bytes": total_size_bytes,
                            "updated_at": updated_at,
                        }
                    },
                )
            else:
                # Tạo metadata mới nếu chưa có
                kb_metadata = {
                    "name": collection.name,
                    "description": None,
                    "metadata": {},
                    "created_at": created_at,
                    "updated_at": updated_at,
                    "document_count": document_count,
                    "total_size_bytes": total_size_bytes,
                }
                await db["knowledge_base"].insert_one(kb_metadata)

            kbs.append(
                KnowledgeBaseResponse(
                    name=collection.name,
                    description=kb_metadata.get("description") if kb_metadata else None,
                    created_at=created_at,
                    updated_at=updated_at,
                    document_count=document_count,
                    total_size_mb=total_size_mb,
                )
            )

        return KnowledgeBaseList(
            knowledge_bases=kbs,
            total=len(kbs),
        )

    except Exception as e:
        logger.error(f"Error listing KBs: {e}")
        raise HTTPException(
            500, detail=f"Lỗi khi lấy danh sách knowledge bases: {str(e)}"
        )


@router.put(
    "/knowledge-bases/{name}",
    response_model=KnowledgeBaseResponse,
    summary="Cập nhật thông tin knowledge base",
    description="Cập nhật thông tin knowledge base như description và metadata.",
)
async def update_knowledge_base(
    name: str,
    kb_update: KnowledgeBaseUpdate,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    """
    Cập nhật thông tin knowledge base.

    - **name**: Tên của knowledge base cần cập nhật
    - **kb_update**: Thông tin cần cập nhật
    """
    try:
        # Kiểm tra knowledge base có tồn tại không
        vector_store = VectorStore()
        collections = vector_store.list_collections()
        if not any(c.name == name for c in collections):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Không tìm thấy knowledge base {name}",
            )

        # Chuẩn bị dữ liệu cập nhật
        update_data = {
            "updated_at": datetime.now().isoformat(),
        }

        if kb_update.description is not None:
            update_data["description"] = kb_update.description

        # Cập nhật trong MongoDB
        result = await db["knowledge_base"].update_one(
            {"name": name}, {"$set": update_data}, upsert=True
        )

        if result.matched_count == 0 and result.upserted_id is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Không thể cập nhật knowledge base metadata",
            )

        # Lấy thông tin đầy đủ sau khi cập nhật
        kb_metadata = await db["knowledge_base"].find_one({"name": name})

        # Lấy collection info
        vector_store.config.collection_name = name
        info = vector_store.get_collection_info()

        # Lấy document stats
        try:
            doc_stats = document_storage.get_collection_stats(name)
            document_count = doc_stats.get("total_documents", 0)
            total_size_bytes = doc_stats.get("total_size_bytes", 0)
            total_size_mb = round(total_size_bytes / (1024 * 1024), 2)
        except Exception:
            document_count = kb_metadata.get("document_count", 0) if kb_metadata else 0
            total_size_mb = round(
                (kb_metadata.get("total_size_bytes", 0) if kb_metadata else 0)
                / (1024 * 1024),
                2,
            )

        return KnowledgeBaseResponse(
            name=name,
            description=kb_metadata.get("description") if kb_metadata else None,
            created_at=(
                kb_metadata.get("created_at", datetime.now().isoformat())
                if kb_metadata
                else datetime.now().isoformat()
            ),
            updated_at=(
                kb_metadata.get("updated_at")
                if kb_metadata
                else datetime.now().isoformat()
            ),
            document_count=document_count,
            total_size_mb=total_size_mb,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating KB: {e}")
        raise HTTPException(500, detail=f"Lỗi khi cập nhật knowledge base: {str(e)}")


@router.delete(
    "/knowledge-bases/{name}",
    response_model=Dict[str, Any],
    summary="Xóa knowledge base",
    description="Xóa knowledge base, tất cả documents và dữ liệu lưu trữ trong MinIO.",
)
async def delete_knowledge_base(name: str, db: AsyncIOMotorDatabase = Depends(get_db)):
    """
    Xóa knowledge base.

    - **name**: Tên của knowledge base cần xóa
    """
    try:
        # Xóa collection trong vector store
        config = VectorStoreConfig(collection_name=name)
        vector_store = VectorStore(config=config)
        vector_store_success = vector_store.delete_collection()

        if not vector_store_success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Không thể xóa knowledge base {name} trong vector store",
            )

        # Xóa folder và tất cả documents trong MinIO
        minio_success = document_storage.delete_collection_folder(name)
        if not minio_success:
            logger.warning(f"Không thể xóa documents của collection {name} trong MinIO")

        # Xóa metadata trong MongoDB
        metadata_result = await db["knowledge_base"].delete_one({"name": name})
        metadata_deleted = metadata_result.deleted_count > 0

        return {
            "success": True,
            "message": f"Đã xóa knowledge base {name} và tất cả dữ liệu liên quan",
            "time": datetime.now().isoformat(),
            "details": {
                "vector_store_deleted": vector_store_success,
                "minio_documents_deleted": minio_success,
                "metadata_deleted": metadata_deleted,
            },
        }

    except Exception as e:
        logger.error(f"Error deleting KB: {e}")
        raise HTTPException(500, detail=f"Lỗi khi xóa knowledge base: {str(e)}")


@router.post(
    "/knowledge-bases/{name}/documents",
    response_model=FileUploadResponse,
    summary="Upload document vào knowledge base",
    description="""Upload và process document vào knowledge base cụ thể.""",
)
async def upload_document(
    name: str,
    file: UploadFile = File(...),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
    metadata: Dict[str, Any] = Form({}),
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    """
    Upload và process document vào knowledge base.

    - **name**: Tên của knowledge base
    - **file**: File cần upload
    - **chunk_size**: Kích thước mỗi chunk (default: 1000)
    - **chunk_overlap**: Độ overlap giữa các chunks (default: 200)
    - **metadata**: Metadata bổ sung cho document
    """
    start_time = time.time()
    temp_file = None

    try:
        valid, msg = validate_file(file)
        if not valid:
            raise HTTPException(400, detail=msg)

        content = await file.read()
        size = len(content)
        if size > MAX_FILE_SIZE:
            raise HTTPException(413, detail="File too large.")

        # Store file in MinIO with folder structure
        storage_info = document_storage.store_document(
            file_data=content,
            filename=file.filename or "unknown",
            knowledge_name=name,
            content_type=file.content_type or "application/octet-stream",
            metadata=metadata,
        )

        # Add storage info to metadata
        metadata.update(
            {
                "storage_path": storage_info["storage_path"],
                "storage_time": storage_info["storage_time"],
            }
        )

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(file.filename or "")[1]
        ) as tmp:
            tmp.write(content)
            temp_file = tmp.name

        # Get pipeline với collection name cụ thể
        pipeline = get_rag_pipeline(name)

        # Update chunk config nếu khác default
        if chunk_size != 1000 or chunk_overlap != 200:
            config = RAGPipelineConfig(
                chunking_config=ChunkingConfig(
                    chunk_size=chunk_size, chunk_overlap=chunk_overlap
                ),
                embedding_config=EmbeddingConfig(),
                vector_store_config=VectorStoreConfig(collection_name=name),
            )
            global rag_pipeline
            rag_pipeline = RAGPipeline(config)
            pipeline = rag_pipeline

        # Add file info vào metadata
        metadata.update(
            {
                "uploaded_filename": file.filename,
                "upload_time": datetime.now().isoformat(),
                "file_size_bytes": size,
                "knowledge_base": name,
            }
        )

        doc_ids = pipeline.process_and_store(
            temp_file, collection_name=name, extra_metadata=metadata
        )
        stats = pipeline.get_stats()
        processing_time = round(time.time() - start_time, 2)

        logger.info(
            f"Uploaded {file.filename} to KB {name}: {len(doc_ids)} vectors in {processing_time}s"
        )

        # Cập nhật metadata thống kê trong MongoDB
        try:
            doc_stats = document_storage.get_collection_stats(name)
            document_count = doc_stats.get("total_documents", 0)
            total_size_bytes = doc_stats.get("total_size_bytes", 0)
            current_time = datetime.now().isoformat()

            await db["knowledge_base"].update_one(
                {"name": name},
                {
                    "$set": {
                        "document_count": document_count,
                        "total_size_bytes": total_size_bytes,
                        "updated_at": current_time,
                    }
                },
                upsert=True,
            )
            logger.info(f"Updated metadata for KB {name}: {document_count} documents")
        except Exception as e:
            logger.warning(f"Failed to update metadata for KB {name}: {e}")

        return FileUploadResponse(
            success=True,
            message=f"Processed {file.filename} -> {len(doc_ids)} vectors",
            file_info=FileInfoModel(**format_file_info(file, size, storage_info)),
            document_ids=doc_ids,
            statistics=stats,
            processing_time=processing_time,
        )

    finally:
        if temp_file and os.path.exists(temp_file):
            os.unlink(temp_file)


@router.get(
    "/knowledge-bases/{name}/stats",
    response_model=Dict[str, Any],
    summary="Knowledge Base Stats",
    description="Lấy thống kê của knowledge base.",
)
async def get_knowledge_base_stats(name: str):
    """
    Lấy thống kê của knowledge base.

    - **name**: Tên của knowledge base
    """
    try:
        pipeline = get_rag_pipeline(name)
        stats = pipeline.get_stats()
        info = pipeline.vector_stores[name].get_collection_info()

        return {
            "success": True,
            "name": name,
            "stats": stats,
            "collection_info": info,
            "time": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error getting KB stats: {e}")
        raise HTTPException(
            500, detail=f"Lỗi khi lấy thống kê knowledge base: {str(e)}"
        )


@router.get(
    "/knowledge-bases/{name}/documents/stats",
    response_model=CollectionStats,
    summary="Thống kê documents trong knowledge base",
    description="Lấy thông tin thống kê về các documents trong một knowledge base.",
)
async def get_documents_stats(name: str):
    """
    Lấy thống kê về documents trong knowledge base.

    - **name**: Tên của knowledge base cần thống kê
    """
    try:
        # Kiểm tra collection có tồn tại không
        vector_store = VectorStore()
        collections = vector_store.list_collections()
        if not any(c.name == name for c in collections):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Không tìm thấy knowledge base {name}",
            )

        # Lấy thống kê từ MinIO
        stats = document_storage.get_collection_stats(name)

        # Format dung lượng để dễ đọc
        total_size_mb = round(stats["total_size_bytes"] / (1024 * 1024), 2)
        logger.info(
            f"Collection {name} stats: {stats['total_documents']} documents, {total_size_mb} MB"
        )

        return CollectionStats(**stats)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting documents stats: {e}")
        raise HTTPException(500, detail=f"Lỗi khi lấy thống kê documents: {str(e)}")


@router.post(
    "/search",
    response_model=MultiCollectionSearchResponse,
    summary="Tìm kiếm trên nhiều knowledge bases",
    description="Thực hiện tìm kiếm hybrid (BM25 + Vector) trên nhiều knowledge bases.",
)
async def multi_collection_search(request: MultiCollectionSearchRequest):
    """
    Tìm kiếm trên nhiều knowledge bases.

    - **query**: Câu hỏi/query cần tìm kiếm
    - **collection_names**: Danh sách các knowledge bases cần tìm kiếm
    - **top_k**: Số lượng kết quả trả về (default: 5)
    - **score_threshold**: Ngưỡng điểm tối thiểu (default: 0.3)
    """
    try:
        start_time = time.time()

        # Kiểm tra các collections có tồn tại không
        vector_store = VectorStore()
        collections = vector_store.list_collections()
        existing_collections = {c.name for c in collections}

        invalid_collections = set(request.collection_names) - existing_collections
        if invalid_collections:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Không tìm thấy knowledge bases: {', '.join(invalid_collections)}",
            )

        # Khởi tạo embedding model
        try:
            embeddings = Embedding()
            logger.info("Successfully initialized embedding model")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Lỗi khởi tạo embedding model: {str(e)}",
            )

        # Tạo vector stores cho từng collection
        vector_stores = {}
        for name in request.collection_names:
            try:
                config = VectorStoreConfig(collection_name=name)
                store = VectorStore(embeddings=embeddings, config=config)
                store.create_collection()  # Kết nối tới collection đã tồn tại
                vector_stores[name] = store
                logger.info(f"Successfully created vector store for collection: {name}")
            except Exception as e:
                logger.error(
                    f"Failed to create vector store for collection {name}: {e}"
                )
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Lỗi tạo vector store cho collection {name}: {str(e)}",
                )

        # Validate vector stores were created properly
        if not vector_stores:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Không thể tạo được vector stores nào",
            )

        # Tạo retriever cho multi-collection search
        try:
            hybrid_config = HybridSearchConfig(
                bm25_weight=0.3,
                vector_weight=0.7,
                top_k=request.top_k or 5,
                score_threshold=(
                    request.score_threshold
                    if request.score_threshold is not None
                    else 0.3
                ),
            )

            multi_config = MultiCollectionConfig(
                normalize_scores=True,
                top_k=request.top_k or 5,
                score_threshold=(
                    request.score_threshold
                    if request.score_threshold is not None
                    else 0.3
                ),
            )

            logger.info(
                f"Creating MultiCollectionHybridRetriever with {len(vector_stores)} collections"
            )
            retriever = MultiCollectionHybridRetriever(
                vector_stores=vector_stores,
                hybrid_config=hybrid_config,
                multi_collection_config=multi_config,
            )

            # Validate retriever was created properly using utility function
            is_valid, error_msg = validate_retriever_initialization(retriever)
            if not is_valid:
                raise ValueError(f"Retriever validation failed: {error_msg}")

            # Log retriever type information for debugging
            retriever_info = check_retriever_type(retriever)
            logger.info(f"Successfully created retriever: {retriever_info}")

        except Exception as e:
            logger.error(f"Failed to create retriever: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Lỗi tạo retriever: {str(e)}",
            )

        # Thực hiện tìm kiếm
        try:
            logger.info(f"Starting search with query: {request.query[:50]}...")
            results = retriever.get_relevant_documents(request.query)
            logger.info(f"Search completed, found {len(results)} results")
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Lỗi thực hiện tìm kiếm: {str(e)}",
            )

        # Format kết quả
        search_results = []
        for doc in results:
            search_results.append(
                SearchResult(
                    content=doc.page_content,
                    metadata=doc.metadata,
                    score=float(doc.metadata.get("hybrid_score", 0.0)),
                    collection_name=doc.metadata.get("collection_name", "unknown"),
                )
            )

        # Lấy thống kê theo collection
        collection_stats = {}
        for name in request.collection_names:
            try:
                stats = document_storage.get_collection_stats(name)
                collection_stats[name] = stats
            except Exception as e:
                logger.warning(f"Không lấy được thống kê cho {name}: {e}")
                collection_stats[name] = {"error": str(e)}

        processing_time = time.time() - start_time

        return MultiCollectionSearchResponse(
            results=search_results,
            total_results=len(search_results),
            processing_time=processing_time,
            collection_stats=collection_stats,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during multi-collection search: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi thực hiện tìm kiếm: {str(e)}",
        )


# Prompte Template và Setting CRUD


@router.post(
    "/prompt-templates/suggest",
    response_model=str,
    summary="Gợi ý prompt template từ ý tưởng",
    description="Sử dụng LLM để gợi ý prompt template dựa trên ý tưởng của người dùng.",
)
async def suggest_prompt_template(request: Request):
    """
    Gợi ý prompt template từ ý tưởng.
    """
    try:
        # Get request body
        body = await request.body()
        idea = body.decode()

        # Get LLM response
        llm = get_llm()
        system_prompt = PROMPT_SUGGESTION_SYSTEM_TEMPLATE
        user_prompt = f"Ý tưởng: {idea}"

        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        logger.info(f"Sending prompt to LLM:\n{full_prompt}")

        # Get template from LLM
        template = await llm.generate(full_prompt)
        return template.strip()

    except Exception as e:
        logger.error(f"Error suggesting prompt template: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi gợi ý prompt template: {str(e)}",
        )


@router.get(
    "/settings",
    response_model=GlobalSettings,
    summary="Lấy global settings",
    description="Lấy các settings mặc định của hệ thống.",
)
async def get_settings(
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    """
    Lấy global settings của hệ thống.
    """
    try:
        # 1. Lấy global settings
        settings_doc = await db["global_settings"].find_one({"id": "global"})
        if not settings_doc:
            # Tạo default settings nếu chưa có
            default_settings = {
                "id": "global",
                "system_prompt": None,
                "available_collections": [],
                "default_language": "vi",
                "search_settings": {"k": 5, "score_threshold": 0.3},
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            }
            await db["global_settings"].insert_one(default_settings)
            settings_doc = default_settings

        # Đảm bảo tất cả fields cần thiết có mặt
        if settings_doc:
            # Thêm system_prompt nếu không có
            if "system_prompt" not in settings_doc:
                settings_doc["system_prompt"] = None

            try:
                global_settings = GlobalSettings(**settings_doc)
            except Exception as e:
                logger.warning(f"Error parsing settings, using defaults: {e}")
                # Fallback with manual creation
                global_settings = GlobalSettings.construct(
                    system_prompt=settings_doc.get("system_prompt"),
                    available_collections=settings_doc.get("available_collections", []),
                    default_language=settings_doc.get("default_language", "vi"),
                    search_settings=settings_doc.get(
                        "search_settings", {"k": 5, "score_threshold": 0.3}
                    ),
                )
        else:
            # Tạo default GlobalSettings object
            global_settings = GlobalSettings.construct(
                system_prompt=None,
                available_collections=[],
                default_language="vi",
                search_settings={"k": 5, "score_threshold": 0.3},
            )

        return global_settings

    except Exception as e:
        logger.error(f"Error getting settings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi lấy settings: {str(e)}",
        )


@router.patch(
    "/settings",
    response_model=GlobalSettings,
    summary="Cập nhật global settings",
    description="Cập nhật các settings mặc định của hệ thống.",
)
async def update_settings(
    update: SettingsUpdate,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    """
    Cập nhật global settings.

    - **update**: Các settings cần cập nhật
    """
    try:
        # Get current settings
        current = await get_settings(db)

        # Update fields
        update_data = update.dict(exclude_unset=True)

        if update_data:
            # Update timestamp
            update_data["updated_at"] = datetime.now().isoformat()

            # Update in MongoDB
            result = await db["global_settings"].update_one(
                {"id": "global"}, {"$set": update_data}
            )

            if result.modified_count == 0:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Không thể cập nhật settings",
                )

        # Get updated settings
        return await get_settings(db)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating settings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi cập nhật settings: {str(e)}",
        )


@router.patch(
    "/settings/system-prompt",
    response_model=Dict[str, Any],
    summary="Cập nhật system prompt",
    description="Cập nhật system prompt mặc định trong global settings.",
)
async def update_system_prompt(
    system_prompt: str,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    """
    Cập nhật system prompt mặc định.

    - **system_prompt**: System prompt mới
    """
    try:
        # Cập nhật system prompt trong global settings
        result = await db["global_settings"].update_one(
            {"id": "global"},
            {
                "$set": {
                    "system_prompt": system_prompt,
                    "updated_at": datetime.now().isoformat(),
                }
            },
            upsert=True,
        )

        if result.matched_count == 0 and result.upserted_id is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Không thể cập nhật system prompt",
            )

        return {
            "success": True,
            "message": "Đã cập nhật system prompt thành công",
            "system_prompt_preview": (
                system_prompt[:100] + "..."
                if len(system_prompt) > 100
                else system_prompt
            ),
            "timestamp": datetime.now().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating system prompt: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi cập nhật system prompt: {str(e)}",
        )


@router.post(
    "/prompt-templates",
    response_model=PromptTemplate,
    summary="Tạo prompt template mới",
    description="Tạo một prompt template mới với các placeholders.",
)
async def create_prompt_template(
    template_data: PromptTemplateCreate,
    storage: MongoPromptTemplateStorage = Depends(get_prompt_template_storage),
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    """
    Tạo prompt template mới.

    - **name**: Tên của template
    - **description**: Mô tả về template (optional)
    - **template**: Nội dung template với các placeholders
    - **metadata**: Metadata bổ sung (optional)
    - **example_values**: Giá trị mẫu cho các biến (optional)
    """
    try:
        template = await storage.create_template(template_data)

        # Kiểm tra xem có default template chưa, nếu chưa thì đặt template này làm default
        settings = await db["global_settings"].find_one({"id": "global"})
        if not settings or not settings.get("default_prompt_template_id"):
            # Đặt template mới tạo làm default
            await db["global_settings"].update_one(
                {"id": "global"},
                {
                    "$set": {
                        "default_prompt_template_id": template.id,
                        "updated_at": datetime.now().isoformat(),
                    }
                },
                upsert=True,
            )
            logger.info(f"Set new template '{template.name}' as default template")

        return template

    except Exception as e:
        logger.error(f"Error creating prompt template: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi tạo prompt template: {str(e)}",
        )


@router.get(
    "/prompt-templates",
    response_model=PromptTemplateList,
    summary="Lấy danh sách prompt templates",
    description="Lấy danh sách tất cả prompt templates hiện có.",
)
async def list_prompt_templates(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    storage: MongoPromptTemplateStorage = Depends(get_prompt_template_storage),
):
    """
    Lấy danh sách tất cả prompt templates.

    - **skip**: Số lượng templates bỏ qua (phân trang)
    - **limit**: Số lượng templates tối đa trả về
    """
    try:
        templates = await storage.list_templates(skip=skip, limit=limit)
        return PromptTemplateList(templates=templates, total=len(templates))

    except Exception as e:
        logger.error(f"Error listing prompt templates: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi lấy danh sách prompt templates: {str(e)}",
        )


@router.get(
    "/prompt-templates/{template_id}",
    response_model=PromptTemplate,
    summary="Lấy chi tiết prompt template",
    description="Lấy thông tin chi tiết của một prompt template.",
)
async def get_prompt_template(
    template_id: str,
    storage: MongoPromptTemplateStorage = Depends(get_prompt_template_storage),
):
    """
    Lấy thông tin chi tiết của prompt template.

    - **template_id**: ID của template cần lấy thông tin
    """
    try:
        template = await storage.get_template(template_id)
        if not template:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Không tìm thấy prompt template với ID: {template_id}",
            )

        return template

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting prompt template: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi lấy thông tin prompt template: {str(e)}",
        )


@router.put(
    "/prompt-templates/{template_id}",
    response_model=PromptTemplate,
    summary="Cập nhật prompt template",
    description="Cập nhật thông tin của một prompt template.",
)
async def update_prompt_template(
    template_id: str,
    template_data: PromptTemplateUpdate,
    storage: MongoPromptTemplateStorage = Depends(get_prompt_template_storage),
):
    """
    Cập nhật prompt template.

    - **template_id**: ID của template cần cập nhật
    - **template_data**: Dữ liệu cập nhật
    """
    try:
        template = await storage.update_template(template_id, template_data)
        if not template:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Không tìm thấy prompt template với ID: {template_id}",
            )

        return template

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating prompt template: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi cập nhật prompt template: {str(e)}",
        )


@router.delete(
    "/prompt-templates/{template_id}",
    response_model=Dict[str, Any],
    summary="Xóa prompt template",
    description="Xóa một prompt template.",
)
async def delete_prompt_template(
    template_id: str,
    storage: MongoPromptTemplateStorage = Depends(get_prompt_template_storage),
):
    """
    Xóa prompt template.

    - **template_id**: ID của template cần xóa
    """
    try:
        success = await storage.delete_template(template_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Không tìm thấy prompt template với ID: {template_id}",
            )

        return {
            "success": True,
            "message": f"Đã xóa prompt template {template_id}",
            "deleted_id": template_id,
            "timestamp": datetime.now().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting prompt template: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi xóa prompt template: {str(e)}",
        )


@router.post(
    "/prompt-templates/{template_id}/generate",
    response_model=GeneratedPrompt,
    summary="Sinh prompt từ template",
    description="Sinh prompt từ template với các biến được cung cấp.",
)
async def generate_prompt(
    template_id: str,
    generation_data: PromptGeneration,
    storage: MongoPromptTemplateStorage = Depends(get_prompt_template_storage),
):
    """
    Sinh prompt từ template.

    - **template_id**: ID của template cần sử dụng
    - **generation_data**: Dữ liệu cho việc sinh prompt
    """
    try:
        template = await storage.get_template(template_id)
        if not template:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Không tìm thấy prompt template với ID: {template_id}",
            )

        # Generate prompt
        prompt = template.template.format(**generation_data.variables)

        return GeneratedPrompt(
            prompt=prompt,
            template_id=template_id,
            variables_used=generation_data.variables,
            generation_time=datetime.now().isoformat(),
        )

    except KeyError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Thiếu biến bắt buộc: {str(e)}",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating prompt: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi sinh prompt: {str(e)}",
        )


@router.get(
    "/prompt-templates/search/{query}",
    response_model=List[PromptTemplate],
    summary="Tìm kiếm prompt templates",
    description="Tìm kiếm prompt templates theo từ khóa.",
)
async def search_prompt_templates(
    query: str,
    limit: int = Query(10, ge=1, le=100),
    storage: MongoPromptTemplateStorage = Depends(get_prompt_template_storage),
):
    """
    Tìm kiếm prompt templates theo từ khóa.

    - **query**: Từ khóa để tìm kiếm
    - **limit**: Số lượng kết quả tối đa
    """
    try:
        templates = await storage.search_templates(query, limit=limit)
        return templates

    except Exception as e:
        logger.error(f"Error searching prompt templates: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi tìm kiếm prompt templates: {str(e)}",
        )


@router.patch(
    "/prompt-templates/{template_id}/set-default",
    response_model=Dict[str, Any],
    summary="Đặt prompt template làm mặc định",
    description="Đặt một prompt template làm template mặc định trong global settings.",
)
async def set_default_prompt_template(
    template_id: str,
    storage: MongoPromptTemplateStorage = Depends(get_prompt_template_storage),
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    """
    Đặt prompt template làm mặc định.

    - **template_id**: ID của template cần đặt làm mặc định
    """
    try:
        # Kiểm tra template có tồn tại không
        template = await storage.get_template(template_id)
        if not template:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Không tìm thấy prompt template với ID: {template_id}",
            )

        # Cập nhật default_prompt_template_id trong global settings
        result = await db["global_settings"].update_one(
            {"id": "global"},
            {
                "$set": {
                    "default_prompt_template_id": template_id,
                    "updated_at": datetime.now().isoformat(),
                }
            },
            upsert=True,  # Tạo settings nếu chưa có
        )

        if result.matched_count == 0 and result.upserted_id is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Không thể cập nhật default template",
            )

        return {
            "success": True,
            "message": f"Đã đặt template '{template.name}' làm mặc định",
            "template_id": template_id,
            "template_name": template.name,
            "timestamp": datetime.now().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting default prompt template: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi đặt template mặc định: {str(e)}",
        )


def clean_llm_response(text: str) -> str:
    """Clean và format response từ LLM để loại bỏ các ký tự lỗi và format chuẩn."""
    if not text:
        return ""

    try:
        # Loại bỏ các ký tự không mong muốn
        import re

        # Loại bỏ các escape characters và special characters
        text = text.replace("\\n", "\n").replace("\\t", "\t")
        text = text.replace('\\"', '"').replace("\\'", "'")

        # Loại bỏ các markdown code blocks nếu có
        text = re.sub(r"```[\s\S]*?```", "", text)
        text = re.sub(r"`([^`]*)`", r"\1", text)

        # Loại bỏ HTML tags nếu có
        text = re.sub(r"<[^>]+>", "", text)

        # Loại bỏ các ký tự điều khiển
        text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]", "", text)

        # Chuẩn hóa whitespace
        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        # Loại bỏ các dòng trống liên tiếp
        text = re.sub(r"\n\s*\n", "\n\n", text)

        return text

    except Exception as e:
        logger.warning(f"Error cleaning LLM response: {e}")
        # Fallback: chỉ loại bỏ whitespace thừa
        return " ".join(text.split())


def format_system_prompt_for_clean_output(base_prompt: str, language: str) -> str:
    """Thêm instructions để LLM trả về text sạch."""

    clean_instructions = {
        "vi": """

QUAN TRỌNG: Hãy trả lời bằng văn bản thuần túy, sạch sẽ:
- KHÔNG sử dụng markdown, HTML, hoặc format đặc biệt
- KHÔNG sử dụng ký tự escape (\\n, \\t, etc.)
- KHÔNG đặt text trong code blocks (```)
- Chỉ sử dụng text thông thường, dễ đọc
- Trả lời ngắn gọn, rõ ràng, chính xác""",
        "en": """

IMPORTANT: Please respond with clean, plain text only:
- NO markdown, HTML, or special formatting
- NO escape characters (\\n, \\t, etc.)
- NO code blocks (```)
- Use plain, readable text only
- Keep response concise, clear, and accurate""",
    }

    instruction = clean_instructions.get(language, clean_instructions["vi"])
    return base_prompt + instruction


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Chat với AI",
    description="Chat đơn giản với AI sử dụng global settings và auto session management.",
)
async def chat(
    request: SimpleChatRequest,
    db: AsyncIOMotorDatabase = Depends(get_db),
    storage: MongoPromptTemplateStorage = Depends(get_prompt_template_storage),
):
    """
    Chat với AI.

    - **user_id**: ID của user
    - **query**: Câu hỏi của user

    Hệ thống sẽ tự động:
    - Lấy global settings từ MongoDB
    - Tạo/lấy session cho user
    - Sử dụng default collections và prompt template
    - Tìm kiếm context và generate response
    """
    try:
        start_time = time.time()

        # 1. Lấy global settings
        global_settings = await get_settings(db)

        # 2. Lấy/tạo chat session cho user
        session_id = f"user_{request.user_id}"
        session_doc = await db["chat_sessions"].find_one({"id": session_id})

        if not session_doc:
            # Tạo session mới với settings từ global
            chat_settings = ChatSettings(
                collection_names=global_settings.available_collections,
                prompt_template_id=None,
                temperature=0.7,
                max_tokens=2000,
                language=global_settings.default_language,
                search_kwargs=global_settings.search_settings,
            )

            new_session = ChatSession(
                id=session_id,
                settings=chat_settings,
                metadata={"user_id": request.user_id},
            )

            session_doc = new_session.model_dump()
            await db["chat_sessions"].insert_one(session_doc)
            session = new_session
        else:
            session = ChatSession(**session_doc)

        # 3. Thêm user message vào session
        user_message = ChatMessage(
            role="user",
            content=request.query,
            metadata={"timestamp": datetime.now().isoformat()},
        )
        session.messages.append(user_message)

        # 4. Tìm kiếm context nếu có collections
        context_results = []
        if session.settings.collection_names:
            try:
                search_request = MultiCollectionSearchRequest(
                    query=request.query,
                    collection_names=session.settings.collection_names,
                    top_k=session.settings.search_kwargs.get("k", 5),
                    score_threshold=session.settings.search_kwargs.get(
                        "score_threshold", 0.3
                    ),
                )

                search_response = await multi_collection_search(search_request)
                context_results = search_response.results

            except Exception as e:
                logger.warning(f"Search failed, continuing without context: {e}")

        # 5. Lấy system prompt với thứ tự ưu tiên: Global Settings > Template > Default
        system_prompt = ""

        # Ưu tiên 1: System prompt từ global settings
        if global_settings.system_prompt:
            system_prompt = global_settings.system_prompt
            logger.info("Using system prompt from global settings")

        # Ưu tiên 2: Template prompt nếu không có global system prompt
        elif session.settings.prompt_template_id:
            try:
                template = await storage.get_template(
                    session.settings.prompt_template_id
                )
                if template:
                    system_prompt = template.template
                    logger.info(f"Using template: {template.name}")

            except Exception as e:
                logger.warning(f"Failed to get template, using default: {e}")

        # Nếu có system prompt (từ settings hoặc template), thực hiện variable replacement
        if system_prompt:
            # Tạo context string từ search results
            context_text = "\n".join(
                [f"- {result.content}" for result in context_results[:3]]
            )

            # Tạo chat history cho template (nếu cần)
            template_history = ""
            if len(session.messages) > 1:
                recent_msgs = session.messages[-4:]  # 2 cặp hỏi-đáp gần nhất
                history_parts = []
                for msg in recent_msgs[:-1]:  # Không bao gồm message hiện tại
                    if msg.role in ["user", "assistant"]:
                        role_name = "User" if msg.role == "user" else "Assistant"
                        history_parts.append(f"{role_name}: {msg.content}")
                template_history = "\n".join(history_parts)

            # Replace các biến cơ bản
            system_prompt = system_prompt.replace("{{context}}", context_text)
            system_prompt = system_prompt.replace("{{query}}", request.query)
            system_prompt = system_prompt.replace(
                "{{language}}", session.settings.language
            )
            system_prompt = system_prompt.replace("{{user_id}}", request.user_id)
            system_prompt = system_prompt.replace("{{chat_history}}", template_history)

            # Replace additional variables nếu có template
            if session.settings.prompt_template_id:
                try:
                    template = await storage.get_template(
                        session.settings.prompt_template_id
                    )
                    if template and template.example_values:
                        for var, example in template.example_values.items():
                            system_prompt = system_prompt.replace(
                                f"{{{{{var}}}}}", str(example)
                            )
                except Exception:
                    pass  # Ignore errors for additional variables

        # Ưu tiên 3: Default system prompt nếu không có gì
        if not system_prompt:
            if session.settings.language == "vi":
                system_prompt = f"""Bạn là một trợ lý AI hữu ích. Hãy trả lời câu hỏi dựa trên thông tin được cung cấp.

Câu hỏi: {request.query}

Context: {chr(10).join([f"- {result.content}" for result in context_results[:3]])}

Hãy trả lời một cách chính xác và hữu ích."""
            else:
                system_prompt = f"""You are a helpful AI assistant. Please answer the question based on the provided information.

Question: {request.query}

Context: {chr(10).join([f"- {result.content}" for result in context_results[:3]])}

Please provide an accurate and helpful response."""

        # Thêm instructions cho clean output
        system_prompt = format_system_prompt_for_clean_output(
            system_prompt, session.settings.language
        )

        # 6. Gọi LLM để generate response
        try:
            llm = get_llm()

            # Tạo chat history string cho context
            chat_history = ""
            if len(session.messages) > 1:  # Có lịch sử chat
                recent_messages = session.messages[
                    -6:
                ]  # Lấy 6 messages gần nhất (3 cặp hỏi-đáp)
                history_parts = []
                for msg in recent_messages[:-1]:  # Không bao gồm message hiện tại
                    if msg.role == "user":
                        history_parts.append(f"Người dùng: {msg.content}")
                    elif msg.role == "assistant":
                        history_parts.append(f"Trợ lý: {msg.content}")

                if history_parts:
                    if session.settings.language == "vi":
                        chat_history = f"\n\nLịch sử cuộc trò chuyện:\n{chr(10).join(history_parts)}\n"
                    else:
                        chat_history = f"\n\nConversation history:\n{chr(10).join(history_parts)}\n"

            # Generate response với chat history
            prompt_text = f"{system_prompt}{chat_history}\n\nUser: {request.query}"
            raw_response = await llm.generate(prompt_text)

            # Clean và format response
            response_text = clean_llm_response(raw_response)

            # Kiểm tra response có hợp lệ không
            if not response_text or len(response_text.strip()) < 5:
                if session.settings.language == "vi":
                    response_text = (
                        "Xin lỗi, tôi không thể tạo phản hồi phù hợp cho câu hỏi này."
                    )
                else:
                    response_text = "Sorry, I couldn't generate an appropriate response for this question."

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            if session.settings.language == "vi":
                response_text = (
                    f"Xin lỗi, tôi gặp sự cố khi tạo phản hồi. Vui lòng thử lại sau."
                )
            else:
                response_text = f"Sorry, I encountered an issue generating a response. Please try again later."

        # 7. Tạo assistant message
        assistant_message = ChatMessage(
            role="assistant",
            content=response_text,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "context_count": len(context_results),
                "template_used": session.settings.prompt_template_id,
                "processing_time": round(time.time() - start_time, 2),
            },
        )
        session.messages.append(assistant_message)

        # 8. Cập nhật session trong database
        session.updated_at = datetime.now().isoformat()
        await db["chat_sessions"].update_one(
            {"id": session_id}, {"$set": session.model_dump()}, upsert=True
        )

        # 9. Return response
        return ChatResponse(
            message=assistant_message,
            # context=context_results,
            # metadata={
            #     "session_id": session_id,
            #     "user_id": request.user_id,
            #     "message_count": len(session.messages),
            #     "collections_used": session.settings.collection_names,
            #     "processing_time": round(time.time() - start_time, 2),
            # },
        )

    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi trong quá trình chat: {str(e)}",
        )


class ChatMessagesResponse(BaseModel):
    messages: List[ChatMessage]


@router.get(
    "/chat/sessions/{user_id}",
    response_model=ChatMessagesResponse,
    summary="Lấy chat session của user",
    description="Lấy thông tin chat session và lịch sử chat của user.",
)
async def get_user_chat_session(
    user_id: str,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    """
    Lấy **messages** của chat session user.
    """
    try:
        session_id = f"user_{user_id}"
        session_doc = await db["chat_sessions"].find_one({"id": session_id})

        if not session_doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Không tìm thấy chat session cho user {user_id}",
            )

        return {"messages": session_doc["messages"]}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting chat session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi lấy chat session: {str(e)}",
        )


@router.delete(
    "/chat/sessions/{user_id}",
    response_model=Dict[str, Any],
    summary="Xóa chat session của user",
    description="Xóa chat session và toàn bộ lịch sử chat của user.",
)
async def delete_user_chat_session(
    user_id: str,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    """
    Xóa chat session của user.

    - **user_id**: ID của user
    """
    try:
        session_id = f"user_{user_id}"
        result = await db["chat_sessions"].delete_one({"id": session_id})

        if result.deleted_count == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Không tìm thấy chat session cho user {user_id}",
            )

        return {
            "success": True,
            "message": f"Đã xóa chat session cho user {user_id}",
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting chat session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi xóa chat session: {str(e)}",
        )
