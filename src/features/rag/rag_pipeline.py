"""
RAG Pipeline - Vietnamese Document Processing

Pipeline để xử lý documents từ raw files đến chunks, 
được tối ưu hóa đặc biệt cho tiếng Việt.

Current Workflow:
1. Document Processing: Parse documents → Extract text + metadata
2. Chunking: Smart chunking với Vietnamese optimization
3. Text Processing: Add BM25 tokens và các metadata khác
4. Embedding: Multilingual E5 embeddings
5. Vector Storage: Qdrant vector store
6. Retrieval: Hybrid search (BM25 + Vector) across multiple collections
7. Ready for next phases: Generation

Hỗ trợ:
- Multiple file formats: PDF, DOCX, TXT, HTML, CSV
- Vietnamese text optimization
- Smart chunking strategy selection 
- Batch processing với progress tracking
- Rich metadata tracking
- BM25 token generation
- Multilingual embeddings
- Vector storage và retrieval
- Hybrid search (BM25 + Vector)
- Multi-collection search with weights

Future phases sẽ có:
- Embedding models integration
- Vector databases (Chroma/FAISS)  
- LLM integration cho Q&A
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
import json
from dataclasses import dataclass
import re
from underthesea import word_tokenize as vi_tokenize
import nltk

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

from .document_parser import DocumentParser
from .chunking import Chunking, ChunkingConfig
from .embedding import EmbeddingConfig, Embedding
from .vector_store import VectorStoreConfig, VectorStore
from .retrieval import (
    HybridSearchConfig,
    MultiCollectionConfig,
    create_hybrid_retriever,
    create_multi_collection_retriever,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RAGPipelineConfig:
    """Configuration for RAG Pipeline."""

    chunking_config: Optional[ChunkingConfig] = None
    embedding_config: Optional[EmbeddingConfig] = None
    vector_store_config: Optional[VectorStoreConfig] = None
    retrieval_config: Optional[HybridSearchConfig] = None
    multi_collection_config: Optional[MultiCollectionConfig] = None


class RAGPipeline:
    """
    RAG Pipeline - Single File Processing

    Features:
    - Single file document parsing (PDF, DOCX, TXT, HTML, CSV)
    - Vietnamese text optimization
    - Smart chunking strategies (Hierarchical, Semantic, Simple)
    - Multilingual embeddings (E5 model)
    - Vector storage (Qdrant)
    - Hybrid search (BM25 + Vector)
    - Multi-collection search with weights
    - Error handling và robust processing
    - Rich metadata tracking
    - Statistics monitoring
    """

    def __init__(self, config: Optional[RAGPipelineConfig] = None):
        """
        Khởi tạo RAG Pipeline.

        Args:
            config: Cấu hình cho pipeline. Nếu None, dùng config mặc định.
        """
        # Load config
        self.config = config or RAGPipelineConfig()

        # Document processing components
        self.parser = DocumentParser()
        self.chunking = Chunking(self.config.chunking_config)

        # Embedding và vector store components
        self.embeddings = Embedding(self.config.embedding_config)

        # Dictionary to store vector stores for each collection
        self.vector_stores: Dict[str, VectorStore] = {}
        self._initialize_vector_stores()

        # Statistics tracking
        self.stats = {
            "total_files_processed": 0,
            "total_documents_created": 0,
            "total_chunks_created": 0,
            "total_embeddings_created": 0,
            "total_vectors_stored": 0,
            "processing_errors": 0,
            "last_processing_time": None,
        }

        logger.info("RAG Pipeline initialized successfully")
        logger.info(
            f"Supported formats: {', '.join(sorted(self.parser.supported_formats))}"
        )
        logger.info(
            "All phases ready: Document Processing + Chunking + Embedding + Vector Storage + Retrieval"
        )

    def _initialize_vector_stores(self):
        """Initialize vector stores for each collection."""
        # Get base vector store config
        base_config = self.config.vector_store_config or VectorStoreConfig()

        # Get collection names and weights from multi-collection config
        if self.config.multi_collection_config:
            collection_weights = self.config.multi_collection_config.collection_weights
        else:
            # Use collection name from config
            collection_name = base_config.collection_name
            collection_weights = {collection_name: 1.0}

        # Create vector store for each collection
        for collection_name in collection_weights.keys():
            config = VectorStoreConfig(
                **{
                    **base_config.__dict__,
                    "collection_name": collection_name,
                }
            )
            vector_store = VectorStore(embeddings=self.embeddings, config=config)
            vector_store.create_collection()
            self.vector_stores[collection_name] = vector_store

    def process_and_store(
        self,
        file_path: str,
        collection_name: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Process một file và lưu vào vector store.

        Args:
            file_path: Đường dẫn file cần process
            collection_name: Tên collection để lưu. Nếu None, dùng collection mặc định
            extra_metadata: Metadata bổ sung

        Returns:
            List[str]: Document IDs trong vector store
        """
        # Use default collection if none specified
        if collection_name is None:
            collection_name = "default"

        # Check if collection exists
        if collection_name not in self.vector_stores:
            logger.error(f"Collection {collection_name} không tồn tại")
            return []

        # Process file thành chunks
        chunks = self._process_single_file(file_path, extra_metadata)
        if not chunks:
            return []

        # Add embedding metadata
        embedding_metadata = self.embeddings.get_metadata()
        for chunk in chunks:
            chunk.metadata.update(
                {
                    "embedding_model": embedding_metadata["model_name"],
                    "embedding_dimension": embedding_metadata["dimension"],
                    "collection_name": collection_name,
                }
            )

        # Store trong vector store
        try:
            vector_store = self.vector_stores[collection_name]
            document_ids = vector_store.add_documents(chunks)
            self.stats["total_embeddings_created"] += len(chunks)
            self.stats["total_vectors_stored"] += len(document_ids)
            logger.info(
                f"Stored {len(document_ids)} vectors in collection {collection_name}"
            )
            return document_ids
        except Exception as e:
            logger.error(f"Error storing vectors: {str(e)}")
            raise

    def search(
        self,
        query: str,
        collection_names: Optional[List[str]] = None,
        collection_weights: Optional[Dict[str, float]] = None,
        k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Tìm kiếm hybrid (BM25 + Vector) trong một hoặc nhiều collections.

        Args:
            query: Query text
            collection_names: Danh sách collections để tìm kiếm. Nếu None, tìm trong tất cả
            collection_weights: Trọng số cho mỗi collection
            k: Số lượng kết quả trả về
            score_threshold: Ngưỡng similarity score
            filter_conditions: Điều kiện lọc metadata

        Returns:
            List[Document]: Danh sách documents phù hợp nhất
        """
        # Determine which collections to search
        if collection_names is None:
            search_collections = list(self.vector_stores.keys())
        else:
            search_collections = [
                name for name in collection_names if name in self.vector_stores
            ]
            if not search_collections:
                logger.error("Không tìm thấy collection nào hợp lệ")
                return []

        # Get vector stores to search
        search_stores = {name: self.vector_stores[name] for name in search_collections}

        # Create multi-collection retriever
        retriever = create_multi_collection_retriever(
            vector_stores=search_stores,
            collection_weights=collection_weights,
            hybrid_config=self.config.retrieval_config,
            top_k=k
            or (
                self.config.multi_collection_config.top_k
                if self.config.multi_collection_config
                else 5
            ),
            score_threshold=score_threshold
            or (
                self.config.multi_collection_config.score_threshold
                if self.config.multi_collection_config
                else 0.3
            ),
        )

        # Perform search
        try:
            results = retriever.get_relevant_documents(query)
            logger.info(
                f"Found {len(results)} relevant documents from {len(search_collections)} collections"
            )
            return results
        except Exception as e:
            logger.error(f"Error during hybrid search: {str(e)}")
            raise

    def similarity_search(
        self,
        query: str,
        collection_name: Optional[str] = None,
        k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict[str, Any]] = None,
        return_scores: bool = False,
    ) -> Union[List[Document], List[Tuple[Document, float]]]:
        """
        Tìm kiếm similarity trong vector store.
        DEPRECATED: Use search() instead for hybrid search.

        Args:
            query: Query text
            collection_name: Collection để tìm kiếm. Nếu None, dùng collection mặc định
            k: Số lượng kết quả tối đa
            score_threshold: Ngưỡng similarity score
            filter_conditions: Điều kiện lọc metadata
            return_scores: Có trả về scores không

        Returns:
            Union[List[Document], List[Tuple[Document, float]]]: Danh sách documents phù hợp nhất
        """
        logger.warning(
            "similarity_search() is deprecated. Use search() instead for hybrid search."
        )

        # Use default collection if none specified
        if collection_name is None:
            collection_name = "default"

        # Check if collection exists
        if collection_name not in self.vector_stores:
            logger.error(f"Collection {collection_name} không tồn tại")
            return [] if not return_scores else []

        try:
            vector_store = self.vector_stores[collection_name]
            if return_scores:
                return vector_store.similarity_search_with_score(
                    query,
                    k=k,
                    score_threshold=score_threshold,
                    filter_conditions=filter_conditions,
                )
            else:
                return vector_store.similarity_search(
                    query,
                    k=k,
                    score_threshold=score_threshold,
                    filter_conditions=filter_conditions,
                )
        except Exception as e:
            logger.error(f"Error during similarity search: {str(e)}")
            raise

    def _process_single_file(
        self,
        file_path: str,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Process một file thành chunks."""
        try:
            # Parse file
            documents = self.parser.load_single_document(file_path)
            if not documents:
                return []
            self.stats["total_documents_created"] += len(documents)

            # Add file metadata
            file_metadata = {
                "source_file": os.path.basename(file_path),
                "file_extension": os.path.splitext(file_path)[1],
                "processing_timestamp": datetime.now().isoformat(),
            }
            if extra_metadata:
                file_metadata.update(extra_metadata)

            # Add metadata to documents
            for doc in documents:
                doc.metadata.update(file_metadata)

            # Chunk documents
            chunks = self.chunking.chunk_documents(documents)
            if not chunks:
                return []
            self.stats["total_chunks_created"] += len(chunks)

            # Update stats
            self.stats["total_files_processed"] += 1
            self.stats["last_processing_time"] = datetime.now().isoformat()

            return chunks

        except Exception as e:
            self.stats["processing_errors"] += 1
            logger.error(f"Error processing {file_path}: {str(e)}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """
        Lấy thống kê của toàn bộ pipeline.

        Returns:
            Dict với thông tin thống kê đầy đủ
        """
        stats: Dict[str, Any] = {}

        # Add pipeline stats
        pipeline_stats = self.stats.copy()

        # Add derived metrics
        if pipeline_stats["total_files_processed"] > 0:
            pipeline_stats["avg_documents_per_file"] = (
                pipeline_stats["total_documents_created"]
                / pipeline_stats["total_files_processed"]
            )
            pipeline_stats["avg_chunks_per_file"] = (
                pipeline_stats["total_chunks_created"]
                / pipeline_stats["total_files_processed"]
            )
            pipeline_stats["avg_vectors_per_file"] = (
                pipeline_stats["total_vectors_stored"]
                / pipeline_stats["total_files_processed"]
            )

        if pipeline_stats["total_documents_created"] > 0:
            pipeline_stats["avg_chunks_per_document"] = (
                pipeline_stats["total_chunks_created"]
                / pipeline_stats["total_documents_created"]
            )

        pipeline_stats["success_rate"] = (
            (
                pipeline_stats["total_files_processed"]
                - pipeline_stats["processing_errors"]
            )
            / max(pipeline_stats["total_files_processed"], 1)
        ) * 100

        # Add component stats
        for k, v in pipeline_stats.items():
            if isinstance(v, (int, float)):
                stats[str(k)] = v
            else:
                stats[str(k)] = str(v)

        for k, v in self.embeddings.get_stats().items():
            if isinstance(v, (int, float)):
                stats[f"embedding_{str(k)}"] = v
            else:
                stats[f"embedding_{str(k)}"] = str(v)

        # Add vector store stats for each collection
        for collection_name, vector_store in self.vector_stores.items():
            for k, v in vector_store.get_stats().items():
                if isinstance(v, (int, float)):
                    stats[f"vector_store_{collection_name}_{str(k)}"] = v
                else:
                    stats[f"vector_store_{collection_name}_{str(k)}"] = str(v)

        return stats

    def reset_stats(self) -> None:
        """Reset statistics về 0."""
        self.stats = {
            "total_files_processed": 0,
            "total_documents_created": 0,
            "total_chunks_created": 0,
            "total_embeddings_created": 0,
            "total_vectors_stored": 0,
            "processing_errors": 0,
            "last_processing_time": None,
        }
        self.embeddings.reset_stats()
        for vector_store in self.vector_stores.values():
            vector_store.reset_stats()
        logger.info("Pipeline statistics reset")

    def get_embedding_info(self) -> Dict[str, str]:
        """
        Lấy thông tin về embedding model.

        Returns:
            Dict với thông tin về model
        """
        return self.embeddings.get_metadata()


def process_file(
    file_path: str,
    collection_name: Optional[str] = None,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> List[str]:
    """
    Hàm tiện ích để process một file với config mặc định.

    Args:
        file_path: Đường dẫn file cần process
        collection_name: Collection để lưu document
        chunk_size: Kích thước chunk
        chunk_overlap: Độ overlap giữa các chunks

    Returns:
        List[str]: Document IDs trong vector store
    """
    config = RAGPipelineConfig(
        chunking_config=ChunkingConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    )
    pipeline = RAGPipeline(config)
    return pipeline.process_and_store(file_path, collection_name=collection_name)
