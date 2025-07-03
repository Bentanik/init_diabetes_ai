"""
Multilingual BM25 Retriever cho tìm kiếm văn bản song ngữ Việt-Anh
Sử dụng thuật toán BM25 để tìm kiếm dựa trên từ khóa (keyword-based search)
Đặc điểm:
- Hỗ trợ song ngữ Việt-Anh
- Sử dụng tokens đã được tạo sẵn trong metadata
- Kết hợp được với vector search trong hybrid search
- Chuẩn hóa điểm số để dễ dàng kết hợp với các phương pháp khác
"""

from typing import List, Optional, Dict, Tuple, Any, Set, cast
import numpy as np
from rank_bm25 import BM25Okapi
from langchain.schema import Document, BaseRetriever
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
import logging
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class BM25Config(BaseModel):
    """Cấu hình cho BM25 Retriever."""

    # Số lượng kết quả trả về
    top_k: int = Field(default=5, description="Số lượng kết quả trả về")

    # Ngưỡng điểm tối thiểu để lọc kết quả
    score_threshold: float = Field(
        default=0.3, description="Ngưỡng điểm tối thiểu để lọc kết quả"
    )

    # Tham số k1 trong công thức BM25, điều chỉnh ảnh hưởng của term frequency
    k1: float = Field(default=1.5, description="Tham số k1 trong công thức BM25")

    # Tham số b trong công thức BM25, điều chỉnh ảnh hưởng của document length
    b: float = Field(default=0.75, description="Tham số b trong công thức BM25")


class BM25(BaseRetriever):
    """BM25 Retriever hỗ trợ song ngữ Việt-Anh."""

    METADATA_TOKENS_KEY = "bm25_tokens"  # Key để lưu tokens trong metadata

    def __init__(
        self,
        vector_store,  # Qdrant client
        config: Optional[BM25Config] = None,
    ):
        """Initialize BM25 retriever.

        Args:
            vector_store: Qdrant vector store instance
            config: Configuration for BM25
        """
        self.vector_store = vector_store
        self.config = config or BM25Config()
        self._bm25: Optional[BM25Okapi] = None
        self._documents: List[Document] = []
        self._doc_tokens: List[List[str]] = []
        self._doc_id_map: Dict[str, int] = {}  # Map point_id to index
        self._last_fetch_count: int = 0

        # Build initial index
        self._build_initial_index()

    def _process_new_points(self, points: List[Any]) -> None:
        """Process new points from vector store.

        Args:
            points: List of points from vector store
        """
        for point in points:
            if point.id not in self._doc_id_map:
                # Get tokens from metadata
                tokens = point.payload.get(self.METADATA_TOKENS_KEY, [])
                if not tokens:
                    logger.warning(
                        f"No BM25 tokens found in metadata for document {point.id}"
                    )
                    continue

                # Add to index
                doc_idx = len(self._documents)
                self._doc_id_map[point.id] = doc_idx
                self._doc_tokens.append(tokens)

                # Create Document object
                doc = Document(
                    page_content=point.payload.get("content", ""),
                    metadata={"point_id": point.id, **point.payload},
                )
                self._documents.append(doc)

        # Rebuild BM25 index if we have new documents
        if self._doc_tokens:
            self._bm25 = BM25Okapi(self._doc_tokens, k1=self.config.k1, b=self.config.b)

    def _build_initial_index(self) -> None:
        """Build initial BM25 index from vector store."""
        # Get all points from vector store
        points = self.vector_store.get_points()
        if points:
            self._process_new_points(points)
            self._last_fetch_count = len(points)

    def _check_for_updates(self) -> None:
        """Check for new documents in vector store."""
        current_count = self.vector_store.get_points_count()
        if current_count > self._last_fetch_count:
            # Get only new points
            new_points = self.vector_store.get_points()[self._last_fetch_count :]
            self._process_new_points(new_points)
            self._last_fetch_count = current_count

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        """Get relevant documents for query.

        Args:
            query: Search query
            run_manager: Callback manager

        Returns:
            List of relevant documents
        """
        # Check for new documents
        self._check_for_updates()

        if not self._bm25:
            return []

        # Get tokens from query using same tokenization as in chunking
        query_tokens = self.vector_store.get_tokens_for_text(query)

        # Get BM25 scores
        scores = self._bm25.get_scores(query_tokens)

        # Normalize scores to [0,1]
        scores = self._normalize_scores(scores)

        # Get top k documents above threshold
        top_k_indices = np.argsort(scores)[::-1][: self.config.top_k]
        results = []

        for idx in top_k_indices:
            score = scores[idx]
            if score >= self.config.score_threshold:
                doc = self._documents[idx]
                doc.metadata["bm25_score"] = float(score)
                results.append(doc)

        return results

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to range [0,1].

        Args:
            scores: Raw BM25 scores

        Returns:
            Normalized scores
        """
        if len(scores) == 0:
            return scores

        min_score = np.min(scores)
        max_score = np.max(scores)

        if max_score == min_score:
            return np.ones_like(scores)

        return (scores - min_score) / (max_score - min_score)

    def add_documents(self, documents: List[Document]) -> None:
        """Add new documents to retriever.

        Args:
            documents: List of documents to add
        """
        # Add documents to vector store
        self.vector_store.add_documents(documents, bm25_retriever=self)

        # Update BM25 index
        self._check_for_updates()
