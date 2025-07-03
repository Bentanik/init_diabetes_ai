"""Tests for hybrid retrieval functionality."""

import pytest
from unittest.mock import Mock, patch
import numpy as np
from langchain.schema import Document

from src.features.rag.retrieval import (
    HybridSearchConfig,
    HybridRetriever,
    MultiCollectionHybridRetriever,
    MultiCollectionConfig,
    create_hybrid_retriever,
)
from src.utils.utils import (
    validate_retriever_initialization,
    check_retriever_type,
    safe_access_vector_store,
)


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store."""
    store = Mock()

    # Mock BM25 search results
    store.bm25_search.return_value = [
        (
            Document(
                page_content="BM25 only doc",
                metadata={"point_id": "1", "bm25_tokens": ["bm25", "only", "doc"]},
            ),
            0.8,
        ),
        (
            Document(
                page_content="Common doc",
                metadata={"point_id": "2", "bm25_tokens": ["common", "doc"]},
            ),
            0.6,
        ),
    ]

    # Mock vector search results
    store.similarity_search_with_score.return_value = [
        (
            Document(
                page_content="Vector only doc",
                metadata={"point_id": "3", "vector": [0.1, 0.2, 0.3]},
            ),
            0.9,
        ),
        (
            Document(
                page_content="Common doc",
                metadata={"point_id": "2", "vector": [0.4, 0.5, 0.6]},
            ),
            0.7,
        ),
    ]

    return store


def test_hybrid_search_config_normalization():
    """Test that weights are normalized to sum to 1."""
    config = HybridSearchConfig(bm25_weight=2, vector_weight=3)
    assert config.bm25_weight == 0.4  # 2/(2+3)
    assert config.vector_weight == 0.6  # 3/(2+3)
    assert pytest.approx(config.bm25_weight + config.vector_weight) == 1.0


def test_create_hybrid_retriever():
    """Test creation of hybrid retriever with custom config."""
    store = Mock()
    retriever = create_hybrid_retriever(
        store,
        bm25_weight=0.4,
        vector_weight=0.6,
        top_k=3,
        score_threshold=0.5,
    )

    assert isinstance(retriever, HybridRetriever)
    assert retriever.config.bm25_weight == 0.4
    assert retriever.config.vector_weight == 0.6
    assert retriever.config.top_k == 3
    assert retriever.config.score_threshold == 0.5


def test_hybrid_search_combines_results(mock_vector_store):
    """Test that hybrid search combines results from both methods."""
    retriever = create_hybrid_retriever(mock_vector_store)
    results = retriever.get_relevant_documents("test query")

    # Should find 3 unique documents (2 unique + 1 common)
    assert len(results) == 3

    # Check that scores were combined correctly
    point_ids = [doc.metadata["point_id"] for doc in results]
    assert "1" in point_ids  # BM25 only doc
    assert "2" in point_ids  # Common doc
    assert "3" in point_ids  # Vector only doc

    # Common doc should have both scores
    common_doc = next(doc for doc in results if doc.metadata["point_id"] == "2")
    assert "bm25_score" in common_doc.metadata
    assert "vector_score" in common_doc.metadata
    assert "hybrid_score" in common_doc.metadata


def test_hybrid_search_respects_threshold(mock_vector_store):
    """Test that results below threshold are filtered out."""
    # Set a high threshold that only the best results should pass
    retriever = create_hybrid_retriever(mock_vector_store, score_threshold=0.75)
    results = retriever.get_relevant_documents("test query")

    # Only highest scoring documents should remain
    assert all(doc.metadata["hybrid_score"] >= 0.75 for doc in results)


def test_hybrid_search_empty_results(mock_vector_store):
    """Test handling of empty search results."""
    # Mock empty results from both methods
    mock_vector_store.bm25_search.return_value = []
    mock_vector_store.similarity_search_with_score.return_value = []

    retriever = create_hybrid_retriever(mock_vector_store)
    results = retriever.get_relevant_documents("test query")

    assert len(results) == 0


def test_hybrid_search_weights(mock_vector_store):
    """Test that weights affect final scores correctly."""
    # Create retriever with custom weights
    retriever = create_hybrid_retriever(
        mock_vector_store,
        bm25_weight=0.7,  # Favor BM25 results
        vector_weight=0.3,
    )
    results = retriever.get_relevant_documents("test query")

    # BM25-only doc should rank higher than vector-only doc
    scores = [doc.metadata["hybrid_score"] for doc in results]
    bm25_doc_score = next(
        doc.metadata["hybrid_score"]
        for doc in results
        if doc.metadata["point_id"] == "1"
    )
    vector_doc_score = next(
        doc.metadata["hybrid_score"]
        for doc in results
        if doc.metadata["point_id"] == "3"
    )

    assert bm25_doc_score > vector_doc_score


def test_hybrid_search_score_normalization(mock_vector_store):
    """Test that scores are properly normalized."""
    retriever = create_hybrid_retriever(mock_vector_store)
    results = retriever.get_relevant_documents("test query")

    # All scores should be between 0 and 1
    for doc in results:
        assert 0 <= doc.metadata["bm25_score"] <= 1
        assert 0 <= doc.metadata["vector_score"] <= 1
        assert 0 <= doc.metadata["hybrid_score"] <= 1


class TestHybridRetrieverValidation:
    """Test retriever validation and defensive programming."""

    def test_hybrid_retriever_validation_success(self, mock_vector_store):
        """Test successful validation of HybridRetriever."""
        retriever = HybridRetriever(
            vector_store=mock_vector_store, config=HybridSearchConfig()
        )

        is_valid, error_msg = validate_retriever_initialization(retriever)
        assert is_valid
        assert error_msg == ""

    def test_hybrid_retriever_type_checking(self, mock_vector_store):
        """Test type checking for HybridRetriever."""
        retriever = HybridRetriever(
            vector_store=mock_vector_store, config=HybridSearchConfig()
        )

        info = check_retriever_type(retriever)
        assert info["is_hybrid"] is True
        assert info["is_multi_collection"] is False
        assert info["has_vector_store"] is True
        assert info["type"] == "HybridRetriever"

    def test_multi_collection_retriever_validation_success(self, mock_vector_store):
        """Test successful validation of MultiCollectionHybridRetriever."""
        vector_stores = {
            "collection1": mock_vector_store,
            "collection2": mock_vector_store,
        }

        retriever = MultiCollectionHybridRetriever(
            vector_stores=vector_stores,
            hybrid_config=HybridSearchConfig(),
            multi_collection_config=MultiCollectionConfig(),
        )

        is_valid, error_msg = validate_retriever_initialization(retriever)
        assert is_valid
        assert error_msg == ""

    def test_multi_collection_retriever_type_checking(self, mock_vector_store):
        """Test type checking for MultiCollectionHybridRetriever."""
        vector_stores = {
            "collection1": mock_vector_store,
            "collection2": mock_vector_store,
        }

        retriever = MultiCollectionHybridRetriever(
            vector_stores=vector_stores,
            hybrid_config=HybridSearchConfig(),
            multi_collection_config=MultiCollectionConfig(),
        )

        info = check_retriever_type(retriever)
        assert info["is_hybrid"] is False
        assert info["is_multi_collection"] is True
        assert info["has_vector_stores"] is True
        assert info["type"] == "MultiCollectionHybridRetriever"

    def test_safe_vector_store_access_hybrid(self, mock_vector_store):
        """Test safe access to vector_store on HybridRetriever."""
        retriever = HybridRetriever(
            vector_store=mock_vector_store, config=HybridSearchConfig()
        )

        accessed_store = safe_access_vector_store(retriever)
        assert accessed_store is mock_vector_store

    def test_safe_vector_store_access_multi_collection(self, mock_vector_store):
        """Test safe access to vector_stores on MultiCollectionHybridRetriever."""
        vector_stores = {
            "collection1": mock_vector_store,
            "collection2": mock_vector_store,
        }

        retriever = MultiCollectionHybridRetriever(
            vector_stores=vector_stores,
            hybrid_config=HybridSearchConfig(),
            multi_collection_config=MultiCollectionConfig(),
        )

        # Access specific collection
        accessed_store = safe_access_vector_store(retriever, "collection1")
        assert accessed_store is mock_vector_store

        # Access without specifying collection (should return first)
        accessed_store = safe_access_vector_store(retriever)
        assert accessed_store is mock_vector_store

    def test_hybrid_retriever_invalid_vector_store_type(self):
        """Test HybridRetriever with invalid vector store type."""
        with pytest.raises(TypeError, match="Expected VectorStore instance"):
            HybridRetriever(
                vector_store="invalid_type",  # type: ignore  # Should be VectorStore instance
                config=HybridSearchConfig(),
            )

    def test_multi_collection_retriever_invalid_vector_stores_type(
        self, mock_vector_store
    ):
        """Test MultiCollectionHybridRetriever with invalid vector stores type."""
        with pytest.raises(TypeError, match="Expected dict of VectorStore instances"):
            MultiCollectionHybridRetriever(
                vector_stores="invalid_type",  # type: ignore  # Should be dict
                hybrid_config=HybridSearchConfig(),
                multi_collection_config=MultiCollectionConfig(),
            )

    def test_multi_collection_retriever_invalid_store_type(self):
        """Test MultiCollectionHybridRetriever with invalid store type in dict."""
        with pytest.raises(TypeError, match="Expected VectorStore instance"):
            MultiCollectionHybridRetriever(
                vector_stores={"collection1": "invalid_store"},  # type: ignore  # Should be VectorStore
                hybrid_config=HybridSearchConfig(),
                multi_collection_config=MultiCollectionConfig(),
            )

    def test_hybrid_retriever_has_vector_store_property(self, mock_vector_store):
        """Test has_vector_store property."""
        retriever = HybridRetriever(
            vector_store=mock_vector_store, config=HybridSearchConfig()
        )

        assert retriever.has_vector_store is True

    def test_multi_collection_retriever_has_vector_stores_method(
        self, mock_vector_store
    ):
        """Test has_vector_stores method."""
        vector_stores = {
            "collection1": mock_vector_store,
        }

        retriever = MultiCollectionHybridRetriever(
            vector_stores=vector_stores,
            hybrid_config=HybridSearchConfig(),
            multi_collection_config=MultiCollectionConfig(),
        )

        assert retriever.has_vector_stores() is True

    def test_hybrid_retriever_error_handling(self, mock_vector_store):
        """Test error handling in HybridRetriever search."""
        # Mock an error in vector store methods
        mock_vector_store.bm25_search.side_effect = Exception("Search error")

        retriever = HybridRetriever(
            vector_store=mock_vector_store, config=HybridSearchConfig()
        )

        # Should return empty list instead of raising exception
        results = retriever.get_relevant_documents("test query")
        assert results == []

    def test_multi_collection_retriever_error_handling(self, mock_vector_store):
        """Test error handling in MultiCollectionHybridRetriever search."""
        # Create a mock that will fail
        failing_store = Mock()
        failing_store.bm25_search.side_effect = Exception("Search error")
        failing_store.similarity_search_with_score.side_effect = Exception(
            "Search error"
        )

        vector_stores = {
            "good_collection": mock_vector_store,
            "bad_collection": failing_store,
        }

        retriever = MultiCollectionHybridRetriever(
            vector_stores=vector_stores,
            hybrid_config=HybridSearchConfig(),
            multi_collection_config=MultiCollectionConfig(),
        )

        # Should handle the error gracefully and return results from good collection
        results = retriever.get_relevant_documents("test query")
        # Should not crash and may return results from the working collection
        assert isinstance(results, list)


class TestHybridRetriever:
    """Test HybridRetriever functionality."""

    def test_hybrid_retriever_initialization(self, mock_vector_store):
        """Test HybridRetriever initialization."""
        config = HybridSearchConfig(bm25_weight=0.4, vector_weight=0.6, top_k=3)
        retriever = HybridRetriever(vector_store=mock_vector_store, config=config)

        assert retriever.vector_store == mock_vector_store
        assert retriever.config.bm25_weight == 0.4
        assert retriever.config.vector_weight == 0.6
        assert retriever.config.top_k == 3

    def test_hybrid_search_combines_results(self, mock_vector_store):
        """Test that hybrid search properly combines BM25 and vector results."""
        retriever = HybridRetriever(
            vector_store=mock_vector_store,
            config=HybridSearchConfig(bm25_weight=0.3, vector_weight=0.7, top_k=5),
        )

        results = retriever.get_relevant_documents("test query")

        # Should get results from both methods
        assert len(results) >= 1
        # Results should have metadata with scores
        for doc in results:
            assert "hybrid_score" in doc.metadata

    def test_create_hybrid_retriever_function(self, mock_vector_store):
        """Test create_hybrid_retriever function."""
        retriever = create_hybrid_retriever(
            vector_store=mock_vector_store,
            bm25_weight=0.4,
            vector_weight=0.6,
            top_k=3,
            score_threshold=0.1,
        )

        assert isinstance(retriever, HybridRetriever)
        assert retriever.config.bm25_weight == 0.4
        assert retriever.config.vector_weight == 0.6
        assert retriever.config.top_k == 3
        assert retriever.config.score_threshold == 0.1
