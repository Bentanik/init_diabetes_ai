"""
Hybrid Search kết hợp BM25 và Vector Search

Đặc điểm:
- Kết hợp BM25 và vector search
- Tùy chỉnh trọng số, chuẩn hóa, song ngữ Việt-Anh
"""

from typing import List, Optional, Dict
from langchain.schema import Document, BaseRetriever
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from pydantic import BaseModel, Field, ConfigDict
import logging

from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class HybridSearchConfig(BaseModel):
    bm25_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    vector_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    top_k: int = Field(default=5, ge=1)
    score_threshold: float = Field(default=0.3, ge=0.0, le=1.0)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class MultiCollectionConfig(BaseModel):
    collection_weights: Dict[str, float] = Field(
        default_factory=lambda: {"default": 1.0}
    )
    normalize_scores: bool = Field(default=True)
    top_k: int = Field(default=5, ge=1)
    score_threshold: float = Field(default=0.3, ge=0.0, le=1.0)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class HybridRetriever(BaseRetriever):
    def __init__(
        self,
        vector_store: VectorStore,
        config: Optional[HybridSearchConfig] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._vector_store = vector_store
        self._config = config or HybridSearchConfig()

    @property
    def vector_store(self) -> VectorStore:
        return self._vector_store

    @property
    def config(self) -> HybridSearchConfig:
        return self._config

    def has_vector_store(self) -> bool:
        return self._vector_store is not None

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        if not self.has_vector_store():
            logger.error("Vector store not properly initialized")
            return []

        try:
            search_k = max(self.config.top_k * 2, 10)
            bm25_results = self.vector_store.bm25_search(query, k=search_k)
            vector_results = self.vector_store.similarity_search_with_score(
                query, k=search_k
            )

            bm25_dict = {}
            for doc, score in bm25_results:
                doc_id = doc.metadata.get("point_id")
                if doc_id is None:
                    doc_id = f"bm25_{hash(doc.page_content[:100])}"
                    doc.metadata["point_id"] = doc_id
                bm25_dict[doc_id] = (doc, score)

            vector_dict = {}
            for doc, score in vector_results:
                doc_id = doc.metadata.get("point_id")
                if doc_id is None:
                    doc_id = f"vector_{hash(doc.page_content[:100])}"
                    doc.metadata["point_id"] = doc_id
                vector_dict[doc_id] = (doc, score)

            all_ids = set(bm25_dict) | set(vector_dict)
            combined_results = []

            for doc_id in all_ids:
                bm25_score = bm25_dict.get(doc_id, (None, 0))[1]
                vector_score = vector_dict.get(doc_id, (None, 0))[1]
                weighted_score = (
                    self.config.bm25_weight * bm25_score
                    + self.config.vector_weight * vector_score
                )
                if weighted_score < self.config.score_threshold:
                    continue

                # Get document from either bm25_dict or vector_dict, with fallback handling
                doc_tuple = bm25_dict.get(doc_id) or vector_dict.get(doc_id)
                if doc_tuple is None:
                    continue
                doc = doc_tuple[0]
                doc.metadata.update(
                    {
                        "bm25_score": bm25_score,
                        "vector_score": vector_score,
                        "hybrid_score": weighted_score,
                    }
                )
                combined_results.append((doc, weighted_score))

            combined_results.sort(key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in combined_results[: self.config.top_k]]

        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return []


class MultiCollectionHybridRetriever(BaseRetriever):
    def __init__(
        self,
        vector_stores: Dict[str, VectorStore],
        hybrid_config: Optional[HybridSearchConfig] = None,
        multi_collection_config: Optional[MultiCollectionConfig] = None,
    ):
        super().__init__()
        self._vector_stores = vector_stores
        self._config = hybrid_config or HybridSearchConfig()
        self._multi_config = multi_collection_config or MultiCollectionConfig()
        self._retrievers: Dict[str, HybridRetriever] = {
            name: HybridRetriever(store, self._config)
            for name, store in vector_stores.items()
        }

    @property
    def vector_stores(self) -> Dict[str, VectorStore]:
        return self._vector_stores

    @property
    def config(self) -> HybridSearchConfig:
        return self._config

    @property
    def multi_config(self) -> MultiCollectionConfig:
        return self._multi_config

    @property
    def retrievers(self) -> Dict[str, HybridRetriever]:
        return self._retrievers

    def has_vector_stores(self) -> bool:
        return bool(self._vector_stores)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        if not self.has_vector_stores():
            logger.error("Vector stores not properly initialized")
            return []

        all_results = []
        for collection_name, retriever in self.retrievers.items():
            try:
                weight = self.multi_config.collection_weights.get(collection_name, 1.0)
                results = retriever.get_relevant_documents(query)
                for doc in results:
                    doc.metadata["collection_name"] = collection_name
                    doc.metadata["hybrid_score"] *= weight
                all_results.extend(results)
            except Exception as e:
                logger.error(f"Error searching collection {collection_name}: {e}")

        if self.multi_config.normalize_scores and all_results:
            scores = [doc.metadata["hybrid_score"] for doc in all_results]
            min_score, max_score = min(scores), max(scores)
            if max_score > min_score:
                for doc in all_results:
                    doc.metadata["hybrid_score"] = (
                        doc.metadata["hybrid_score"] - min_score
                    ) / (max_score - min_score)

        all_results.sort(key=lambda x: x.metadata["hybrid_score"], reverse=True)
        return [
            doc
            for doc in all_results
            if doc.metadata["hybrid_score"] >= self.multi_config.score_threshold
        ][: self.multi_config.top_k]


def create_hybrid_retriever(
    vector_store: VectorStore,
    bm25_weight: float = 0.3,
    vector_weight: float = 0.7,
    top_k: int = 5,
    score_threshold: float = 0.0,
) -> HybridRetriever:
    config = HybridSearchConfig(
        bm25_weight=bm25_weight,
        vector_weight=vector_weight,
        top_k=top_k,
        score_threshold=score_threshold,
    )
    return HybridRetriever(vector_store=vector_store, config=config)


def create_multi_collection_retriever(
    vector_stores: Dict[str, VectorStore],
    collection_weights: Optional[Dict[str, float]] = None,
    hybrid_config: Optional[HybridSearchConfig] = None,
    normalize_scores: bool = True,
    top_k: int = 5,
    score_threshold: float = 0.3,
) -> MultiCollectionHybridRetriever:
    multi_config = MultiCollectionConfig(
        collection_weights=collection_weights or {"default": 1.0},
        normalize_scores=normalize_scores,
        top_k=top_k,
        score_threshold=score_threshold,
    )
    return MultiCollectionHybridRetriever(
        vector_stores=vector_stores,
        hybrid_config=hybrid_config,
        multi_collection_config=multi_config,
    )
