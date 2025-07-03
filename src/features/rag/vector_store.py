"""
Vector Store

Tích hợp Qdrant để lưu trữ và tìm kiếm vector.

Tính năng:
- Lưu trữ và quản lý vector trong Qdrant
- Tìm kiếm theo độ tương đồng
- Tìm kiếm BM25
- Quản lý bộ sưu tập (collection)
- Lọc theo metadata
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import uuid
from datetime import datetime
import numpy as np
from rank_bm25 import BM25Okapi
from underthesea import word_tokenize as vi_tokenize
import nltk
from nltk.tokenize import word_tokenize as en_tokenize

from qdrant_client.models import FilterSelector, PointIdsList

try:
    from langchain_qdrant import QdrantVectorStore as LangChainVectorStore
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from qdrant_client import QdrantClient, AsyncQdrantClient
    from qdrant_client.models import (
        Distance,
        VectorParams,
        PointStruct,
        Filter,
        FieldCondition,
        MatchValue,
        SearchRequest,
        Batch,
        PayloadSchemaType,
        ExtendedPointId,
        CollectionDescription,
    )
    from qdrant_client.http import models
except ImportError as e:
    logging.error(f"Thiếu thư viện Qdrant: {e}")
    raise

# Download NLTK data
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

logger = logging.getLogger(__name__)


@dataclass
class VectorStoreConfig:
    """Cấu hình cho vector store"""

    # Kết nối Qdrant
    qdrant_url: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: Optional[str] = None

    # Cài đặt collection
    collection_name: str = "default"
    vector_size: int = 768  # Kích thước vector của E5-base
    distance_metric: Distance = Distance.COSINE

    # Cài đặt tìm kiếm
    search_limit: int = 10
    search_score_threshold: float = 0.7

    # Cài đặt indexing
    batch_size: int = 100
    enable_payload_indexing: bool = True


class VectorStore:
    """
    Dịch vụ quản lý vector

    Cung cấp các chức năng:
    - Lưu trữ và quản lý vector
    - Tìm kiếm theo độ tương đồng
    - Tìm kiếm BM25
    - Theo dõi hiệu suất
    """

    def __init__(
        self,
        embeddings: Optional[Embeddings] = None,
        config: Optional[VectorStoreConfig] = None,
    ):
        self.embeddings = embeddings
        self.config = config or VectorStoreConfig()

        # Khởi tạo clients
        self.client: Optional[QdrantClient] = None
        self.async_client: Optional[AsyncQdrantClient] = None
        self.vector_store: Optional[LangChainVectorStore] = None

        # BM25 components
        self._bm25: Optional[BM25Okapi] = None
        self._documents: List[Document] = []
        self._doc_tokens: List[List[str]] = []
        self._last_update_time: Optional[float] = None

        # Theo dõi thống kê
        self._stats = {
            "total_documents_indexed": 0,
            "total_searches_performed": 0,
            "total_indexing_time": 0.0,
            "total_search_time": 0.0,
            "last_operation_time": None,
        }

        self._initialize_clients()

    def _initialize_clients(self):
        """Khởi tạo kết nối Qdrant"""
        try:
            # Client đồng bộ
            self.client = QdrantClient(
                host=self.config.qdrant_url,
                port=self.config.qdrant_port,
                api_key=self.config.qdrant_api_key,
            )

            # Client bất đồng bộ
            self.async_client = AsyncQdrantClient(
                host=self.config.qdrant_url,
                port=self.config.qdrant_port,
                api_key=self.config.qdrant_api_key,
            )

            # Vector store (chỉ khởi tạo nếu có embeddings)
            if self.embeddings is not None:
                self.vector_store = LangChainVectorStore(
                    client=self.client,
                    collection_name=self.config.collection_name,
                    embedding=self.embeddings,
                )

            logger.info(
                f"Đã khởi tạo kết nối Qdrant: {self.config.qdrant_url}:{self.config.qdrant_port}"
            )

        except Exception as e:
            logger.error(f"Lỗi khởi tạo kết nối Qdrant: {e}")
            raise

    def _check_embeddings_required(self):
        """Kiểm tra xem embeddings có được yêu cầu không"""
        if self.embeddings is None:
            raise ValueError(
                "Embeddings là bắt buộc cho chức năng này. "
                "Vui lòng khởi tạo VectorStore với embeddings."
            )

    def create_collection(self, force_recreate: bool = False) -> bool:
        """
        Tạo collection trong Qdrant

        Args:
            force_recreate: Xóa collection cũ nếu đã tồn tại
        """
        try:
            if self.client is None:
                logger.error("Chưa khởi tạo kết nối Qdrant")
                return False

            collection_name = self.config.collection_name

            # Kiểm tra collection đã tồn tại
            collections = self.client.get_collections().collections
            collection_exists = any(
                getattr(col, "name", None) == collection_name for col in collections
            )

            if collection_exists:
                if force_recreate:
                    logger.info(f"Xóa collection cũ: {collection_name}")
                    self.client.delete_collection(collection_name=collection_name)
                else:
                    logger.info(f"Collection {collection_name} đã tồn tại")
                    return True

            # Lấy kích thước vector
            vector_size = self.config.vector_size

            # Thử lấy kích thước thực tế từ model embedding
            if hasattr(self.embeddings, "__dict__"):
                for attr_name in [
                    "embedding_dimension",
                    "model_output_dim",
                    "vector_size",
                ]:
                    if hasattr(self.embeddings, attr_name):
                        attr_value = getattr(self.embeddings, attr_name)
                        if callable(attr_value):
                            try:
                                result = attr_value()
                                if isinstance(result, (int, str)):
                                    vector_size = int(result)
                                    break
                            except:
                                continue
                        elif isinstance(attr_value, (int, str)):
                            vector_size = int(attr_value)
                            break

            logger.info(f"Tạo collection: {collection_name} (kích thước={vector_size})")

            # Đảm bảo vector_size là số nguyên
            vector_size = int(vector_size)

            # Tạo collection
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=self.config.distance_metric,
                ),
            )

            # Tạo các chỉ mục cho metadata
            if self.config.enable_payload_indexing:
                self._create_payload_indices(collection_name)

            logger.info(f"Đã tạo thành công collection {collection_name}")
            return True

        except Exception as e:
            logger.error(f"Lỗi khi tạo collection: {e}")
            return False

    def _create_payload_indices(self, collection_name: str):
        """Tạo các chỉ mục cho metadata để tìm kiếm hiệu quả"""
        try:
            if self.client is None:
                logger.error("Chưa khởi tạo kết nối Qdrant")
                return

            # Các trường metadata cần đánh index
            index_fields = [
                "source_file",
                "doc_id",
                "chunk_id",
                "embedding_model",
                "file_extension",
                "processing_timestamp",
            ]

            for field in index_fields:
                try:
                    self.client.create_payload_index(
                        collection_name=collection_name,
                        field_name=field,
                        field_schema=PayloadSchemaType.KEYWORD,
                    )
                    logger.debug(f"Đã tạo index cho trường: {field}")
                except Exception as e:
                    logger.debug(f"Lỗi khi tạo index cho {field}: {e}")

        except Exception as e:
            logger.warning(f"Lỗi khi tạo các chỉ mục metadata: {e}")

    def add_documents(
        self,
        documents: List[Document],
        batch_size: Optional[int] = None,
        bm25_retriever: Optional[Any] = None,  # BM25Retriever instance
    ) -> List[str]:
        """
        Thêm documents vào vector store.

        Args:
            documents: Danh sách documents cần thêm
            batch_size: Kích thước batch, nếu None sẽ dùng giá trị từ config
            bm25_retriever: BM25Retriever instance để preprocess tokens

        Returns:
            List[str]: IDs của documents đã thêm
        """
        self._check_embeddings_required()
        if not documents:
            return []

        if self.vector_store is None:
            logger.error("Vector store chưa được khởi tạo")
            return []

        batch_size = batch_size or self.config.batch_size
        document_ids = []

        try:
            start_time = datetime.now()

            # Process documents in batches
            for i in range(0, len(documents), batch_size):
                batch = documents[i : i + batch_size]

                # Preprocess documents for BM25 if retriever provided
                if bm25_retriever is not None:
                    for doc in batch:
                        # Add BM25 tokens to metadata if not already present
                        if bm25_retriever.METADATA_TOKENS_KEY not in doc.metadata:
                            doc.metadata = bm25_retriever.preprocess_document(
                                doc.page_content, doc.metadata
                            )

                # Add to vector store
                ids = self.vector_store.add_documents(batch)
                document_ids.extend(ids)

            # Update stats
            end_time = datetime.now()
            self._stats["total_documents_indexed"] += len(documents)
            self._stats["total_indexing_time"] += (
                end_time - start_time
            ).total_seconds()
            self._stats["last_operation_time"] = end_time

            logger.info(
                f"Đã thêm {len(documents)} documents vào {self.config.collection_name}"
            )
            return document_ids

        except Exception as e:
            logger.error(f"Lỗi thêm documents: {e}")
            raise

    async def aadd_documents(
        self,
        documents: List[Document],
        batch_size: Optional[int] = None,
        bm25_retriever: Optional[Any] = None,  # BM25Retriever instance
    ) -> List[str]:
        """Async version of add_documents"""
        return self.add_documents(documents, batch_size, bm25_retriever)

    def similarity_search(
        self,
        query: str,
        k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Tìm kiếm tài liệu tương tự

        Args:
            query: Câu truy vấn
            k: Số lượng kết quả tối đa
            score_threshold: Ngưỡng điểm tối thiểu
            filter_conditions: Điều kiện lọc metadata

        Returns:
            Danh sách tài liệu tương tự
        """
        self._check_embeddings_required()
        if self.vector_store is None:
            logger.error("Chưa khởi tạo vector store")
            return []

        try:
            start_time = datetime.now()

            # Thiết lập tham số tìm kiếm
            k = k or self.config.search_limit
            score_threshold = score_threshold or self.config.search_score_threshold

            # Tạo bộ lọc nếu có
            filter_obj = (
                self._build_filter(filter_conditions) if filter_conditions else None
            )

            # Thực hiện tìm kiếm
            results = self.vector_store.similarity_search(
                query,
                k=k,
                score_threshold=score_threshold,
                filter=filter_obj,
            )

            # Cập nhật thống kê
            search_time = (datetime.now() - start_time).total_seconds()
            self._stats["total_searches_performed"] += 1
            self._stats["total_search_time"] += search_time

            logger.debug(f"Tìm thấy {len(results)} kết quả trong {search_time:.2f}s")
            return results

        except Exception as e:
            logger.error(f"Lỗi khi tìm kiếm: {e}")
            return []

    def similarity_search_with_score(
        self,
        query: str,
        k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Tìm kiếm tài liệu tương tự và trả về điểm số

        Args:
            query: Câu truy vấn
            k: Số lượng kết quả tối đa
            score_threshold: Ngưỡng điểm tối thiểu
            filter_conditions: Điều kiện lọc metadata

        Returns:
            Danh sách tuple (tài liệu, điểm số)
        """
        self._check_embeddings_required()
        if self.vector_store is None:
            logger.error("Chưa khởi tạo vector store")
            return []

        try:
            start_time = datetime.now()

            # Thiết lập tham số tìm kiếm
            k = k or self.config.search_limit
            score_threshold = score_threshold or self.config.search_score_threshold

            # Tạo bộ lọc nếu có
            filter_obj = (
                self._build_filter(filter_conditions) if filter_conditions else None
            )

            # Thực hiện tìm kiếm
            results = self.vector_store.similarity_search_with_score(
                query,
                k=k,
                score_threshold=score_threshold,
                filter=filter_obj,
            )

            # Cập nhật thống kê
            search_time = (datetime.now() - start_time).total_seconds()
            self._stats["total_searches_performed"] += 1
            self._stats["total_search_time"] += search_time
            self._stats["last_operation_time"] = datetime.now().isoformat()

            logger.debug(f"Tìm thấy {len(results)} kết quả trong {search_time:.2f}s")
            return results

        except Exception as e:
            logger.error(f"Lỗi khi tìm kiếm: {e}")
            return []

    def _build_filter(self, conditions: Dict[str, Any]) -> Optional[Filter]:
        """
        Xây dựng bộ lọc từ điều kiện

        Args:
            conditions: Điều kiện lọc dạng dict

        Returns:
            Đối tượng Filter của Qdrant hoặc None nếu không có điều kiện
        """
        if not conditions:
            return None

        try:
            filter_conditions = []
            for field, value in conditions.items():
                filter_conditions.append(
                    FieldCondition(
                        key=field,
                        match=MatchValue(value=value),
                    )
                )

            return Filter(must=filter_conditions)

        except Exception as e:
            logger.error(f"Lỗi khi tạo bộ lọc: {e}")
            return None

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Lấy thông tin về collection

        Returns:
            Dict chứa thông tin về collection
        """
        if self.client is None:
            logger.error("Chưa khởi tạo kết nối Qdrant")
            return {}

        try:
            collection_name = self.config.collection_name
            info = self.client.get_collection(collection_name)

            # Xử lý cấu hình vector
            try:
                vectors_config = info.config.params.vectors
                if isinstance(vectors_config, dict):
                    # Multiple vector configs
                    first_key = next(iter(vectors_config))
                    first_vector = vectors_config[first_key]
                    vector_size = first_vector.size
                    distance = first_vector.distance
                else:
                    # Single vector config
                    vector_size = (
                        vectors_config.size
                        if vectors_config
                        else self.config.vector_size
                    )
                    distance = (
                        vectors_config.distance
                        if vectors_config
                        else self.config.distance_metric
                    )
            except:
                # Fallback to config values
                vector_size = self.config.vector_size
                distance = self.config.distance_metric

            return {
                "collection_name": collection_name,
                "vector_size": vector_size,
                "distance_metric": distance,
                "points_count": getattr(info, "points_count", 0),
                "status": (
                    "ready" if getattr(info, "status", None) == "green" else "error"
                ),
                "created_at": getattr(info, "created_at", None),
            }

        except Exception as e:
            logger.error(f"Lỗi khi lấy thông tin collection: {e}")
            return {}

    def delete_documents(
        self,
        document_ids: Optional[List[Union[str, int]]] = None,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Xóa tài liệu khỏi vector store

        Args:
            document_ids: Danh sách ID tài liệu cần xóa
            filter_conditions: Điều kiện lọc để xóa

        Returns:
            True nếu xóa thành công, False nếu có lỗi
        """
        if self.client is None:
            logger.error("Chưa khởi tạo kết nối Qdrant")
            return False

        try:
            collection_name = self.config.collection_name

            if document_ids:
                # Xóa theo ID
                points_selector = PointIdsList(
                    points=[str(id) for id in document_ids],
                )
                self.client.delete(
                    collection_name=collection_name,
                    points_selector=points_selector,
                )
                logger.info(f"Đã xóa {len(document_ids)} tài liệu theo ID")

            if filter_conditions:
                # Xóa theo điều kiện
                filter_obj = self._build_filter(filter_conditions)
                if filter_obj:
                    self.client.delete(
                        collection_name=collection_name,
                        points_selector=FilterSelector(filter=filter_obj),
                    )
                    logger.info("Đã xóa tài liệu theo điều kiện lọc")

            return True

        except Exception as e:
            logger.error(f"Lỗi khi xóa tài liệu: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Lấy thống kê hoạt động

        Returns:
            Dict chứa các thông số thống kê
        """
        stats = self._stats.copy()

        # Tính trung bình
        if stats["total_searches_performed"] > 0:
            stats["avg_search_time_per_search"] = (
                stats["total_search_time"] / stats["total_searches_performed"]
            )

        if stats["total_documents_indexed"] > 0:
            stats["avg_indexing_time_per_doc"] = (
                stats["total_indexing_time"] / stats["total_documents_indexed"]
            )

        return stats

    def reset_stats(self):
        """Đặt lại thống kê về 0"""
        for key in self._stats:
            if key != "last_operation_time":
                self._stats[key] = 0

    def list_collections(self) -> List[CollectionDescription]:
        """
        Lấy danh sách tất cả collections

        Returns:
            List[CollectionDescription]: Danh sách collections
        """
        try:
            if self.client is None:
                logger.error("Chưa khởi tạo kết nối Qdrant")
                return []

            collections = self.client.get_collections()
            return collections.collections

        except Exception as e:
            logger.error(f"Lỗi khi lấy danh sách collections: {e}")
            return []

    def delete_collection(self) -> bool:
        """
        Xóa collection hiện tại

        Returns:
            bool: True nếu xóa thành công
        """
        try:
            if self.client is None:
                logger.error("Chưa khởi tạo kết nối Qdrant")
                return False

            collection_name = self.config.collection_name
            self.client.delete_collection(collection_name=collection_name)
            logger.info(f"Đã xóa collection: {collection_name}")
            return True

        except Exception as e:
            logger.error(f"Lỗi khi xóa collection: {e}")
            return False

    def get_tokens_for_text(self, text: str) -> List[str]:
        """
        Get tokens for text using Vietnamese/English tokenization.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        # Detect language (simple heuristic)
        has_vietnamese = any(ord(c) > 128 for c in text)

        # Tokenize based on language
        if has_vietnamese:
            tokens = vi_tokenize(text)
        else:
            tokens = en_tokenize(text)

        # Convert to lowercase and remove punctuation
        tokens = [
            token.lower()
            for token in tokens
            if token.strip() and not all(c in ".,!?;:()[]{}\"'" for c in token)
        ]

        return tokens

    def _check_for_updates(self):
        """Check if we need to update BM25 index."""
        if self.vector_store is None:
            return

        # Get all documents from vector store
        try:
            # Get documents from collection
            if self.client is None:
                logger.error("Chưa khởi tạo kết nối Qdrant")
                return

            # Get all points from collection
            collection_info = self.client.get_collection(
                collection_name=self.config.collection_name
            )
            if not collection_info:
                return

            points = self.client.scroll(
                collection_name=self.config.collection_name,
                limit=collection_info.points_count or 0,
                with_payload=True,
                with_vectors=False,
            )[0]

            if not points:
                return

            # Convert points to documents
            all_docs = []
            for point in points:
                if not point.payload:
                    continue
                doc = Document(
                    page_content=point.payload.get("page_content", ""),
                    metadata=point.payload.get("metadata", {}),
                )
                all_docs.append(doc)

            # Check if we need to update
            if (
                self._last_update_time is None
                or len(all_docs) != len(self._documents)
                or any(
                    doc.metadata.get("processing_timestamp", "")
                    > self._last_update_time
                    for doc in all_docs
                )
            ):
                # Update documents and tokens
                self._documents = all_docs
                self._doc_tokens = [
                    self.get_tokens_for_text(doc.page_content)
                    for doc in self._documents
                ]

                # Create new BM25 instance
                if self._doc_tokens:
                    self._bm25 = BM25Okapi(self._doc_tokens)
                    self._last_update_time = datetime.now().timestamp()
                    logger.debug(
                        f"Updated BM25 index with {len(self._documents)} documents"
                    )

        except Exception as e:
            logger.error(f"Error checking for BM25 updates: {e}")

    def bm25_search(
        self,
        query: str,
        k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Tìm kiếm BM25 trong vector store.

        Args:
            query: Query text
            k: Số lượng kết quả tối đa
            score_threshold: Ngưỡng similarity score
            filter_conditions: Điều kiện lọc metadata

        Returns:
            List[Tuple[Document, float]]: Danh sách (document, score)
        """
        try:
            start_time = datetime.now()

            # Update BM25 index if needed
            self._check_for_updates()

            if not self._bm25 or not self._documents:
                return []

            # Get tokens from query
            query_tokens = self.get_tokens_for_text(query)

            # Get BM25 scores
            scores = self._bm25.get_scores(query_tokens)

            # Normalize scores to [0,1]
            if len(scores) > 0:
                min_score = min(scores)
                max_score = max(scores)
                if max_score > min_score:
                    scores = (scores - min_score) / (max_score - min_score)

            # Sort by score
            doc_scores = list(zip(self._documents, scores))
            doc_scores.sort(key=lambda x: float(x[1]), reverse=True)

            # Apply score threshold
            if score_threshold is not None:
                doc_scores = [
                    (doc, score)
                    for doc, score in doc_scores
                    if score >= score_threshold
                ]

            # Apply filter conditions
            if filter_conditions:
                doc_scores = [
                    (doc, score)
                    for doc, score in doc_scores
                    if all(
                        doc.metadata.get(k) == v for k, v in filter_conditions.items()
                    )
                ]

            # Get top k results
            k = k or self.config.search_limit
            results = doc_scores[:k]

            # Update stats
            search_time = (datetime.now() - start_time).total_seconds()
            self._stats["total_searches_performed"] += 1
            self._stats["total_search_time"] += search_time
            self._stats["last_operation_time"] = datetime.now().isoformat()
            logger.debug(f"Found {len(results)} BM25 results in {search_time:.2f}s")
            return [(doc, float(score)) for doc, score in results]

        except Exception as e:
            logger.error(f"Error during BM25 search: {e}")
            return []


def create_vector_store(
    embeddings: Embeddings, config: Optional[VectorStoreConfig] = None
) -> VectorStore:
    """
    Tạo một instance của VectorStore

    Args:
        embeddings: Model embedding để tạo vector
        config: Cấu hình cho vector store

    Returns:
        Instance của VectorStore
    """
    return VectorStore(embeddings=embeddings, config=config)
