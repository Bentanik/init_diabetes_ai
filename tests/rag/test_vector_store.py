"""
Test suite cho Vector Store Service

Test Qdrant vector store integration với LangChain
cho Vietnamese RAG pipeline.
"""

import sys
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock
from typing import List, Dict, Any

# Thêm path để import từ src
current_file = Path(__file__)
tests_rag_dir = current_file.parent  # tests/rag/
tests_dir = tests_rag_dir.parent  # tests/
aiservice_dir = tests_dir.parent  # aiservice/
src_dir = aiservice_dir / "src"
sys.path.insert(0, str(src_dir))

# Import modules to test
from src.rag.vector_store import (
    VectorStoreConfig,
    QdrantVectorService,
    create_qdrant_vector_store,
    create_vietnamese_vector_store,
)
from src.rag.embedding import EmbeddingConfig, MultilingualE5Embeddings
from langchain_core.documents import Document

# Mock Qdrant dependencies for testing without actual Qdrant server
try:
    from qdrant_client.models import Distance
except ImportError:
    # Create mock for testing
    class Distance:
        COSINE = "Cosine"
        EUCLIDEAN = "Euclidean"


class TestVectorStoreConfig:
    """Test vector store configuration"""

    def test_default_config(self):
        """Test default configuration values"""
        config = VectorStoreConfig()

        assert config.qdrant_url == "localhost"
        assert config.qdrant_port == 6333
        assert config.qdrant_api_key is None
        assert config.collection_name == "vietnamese_documents"
        assert config.vector_size == 768
        assert config.distance_metric == Distance.COSINE
        assert config.search_limit == 10
        assert config.search_score_threshold == 0.7
        assert config.batch_size == 100
        assert config.enable_payload_indexing == True

    def test_custom_config(self):
        """Test custom configuration"""
        config = VectorStoreConfig(
            qdrant_url="remote-qdrant.com",
            qdrant_port=6334,
            qdrant_api_key="test-key",
            collection_name="custom_collection",
            vector_size=1024,
            distance_metric=Distance.EUCLIDEAN,
            search_limit=20,
            search_score_threshold=0.8,
            batch_size=50,
        )

        assert config.qdrant_url == "remote-qdrant.com"
        assert config.qdrant_port == 6334
        assert config.qdrant_api_key == "test-key"
        assert config.collection_name == "custom_collection"
        assert config.vector_size == 1024
        assert config.distance_metric == Distance.EUCLIDEAN
        assert config.search_limit == 20
        assert config.search_score_threshold == 0.8
        assert config.batch_size == 50


class TestQdrantVectorServiceInitialization:
    """Test vector service initialization"""

    @pytest.fixture
    def mock_embeddings(self):
        """Mock embeddings for testing"""
        embeddings = Mock()
        embeddings.embed_documents.return_value = [[0.1] * 768, [0.2] * 768]
        embeddings.embed_query.return_value = [0.1] * 768
        return embeddings

    @patch("src.rag.vector_store.QdrantClient")
    @patch("src.rag.vector_store.AsyncQdrantClient")
    @patch("src.rag.vector_store.QdrantVectorStore")
    def test_initialization_success(
        self, mock_vector_store, mock_async_client, mock_client, mock_embeddings
    ):
        """Test successful initialization"""
        config = VectorStoreConfig()
        service = QdrantVectorService(mock_embeddings, config)

        assert service.embeddings == mock_embeddings
        assert service.config == config
        assert service.client is not None
        assert service.async_client is not None
        assert service.vector_store is not None

        # Check stats initialization
        assert service._stats["total_documents_indexed"] == 0
        assert service._stats["total_searches_performed"] == 0

    @patch("src.rag.vector_store.QdrantClient")
    def test_initialization_with_connection_error(
        self, mock_client_class, mock_embeddings
    ):
        """Test initialization with connection error"""
        mock_client_class.side_effect = Exception("Connection failed")

        config = VectorStoreConfig()

        with pytest.raises(Exception, match="Connection failed"):
            QdrantVectorService(mock_embeddings, config)


class TestCollectionManagement:
    """Test collection creation and management"""

    @pytest.fixture
    def mock_embeddings(self):
        embeddings = Mock()
        return embeddings

    @pytest.fixture
    def mock_service(self, mock_embeddings):
        """Mock vector service for testing"""
        config = VectorStoreConfig(collection_name="test_collection")

        with patch("src.rag.vector_store.QdrantClient") as mock_client_class, patch(
            "src.rag.vector_store.AsyncQdrantClient"
        ), patch("src.rag.vector_store.QdrantVectorStore"):

            service = QdrantVectorService(mock_embeddings, config)

            # Mock the client methods
            mock_client = Mock()
            mock_collections_response = Mock()
            mock_collections_response.collections = []
            mock_client.get_collections.return_value = mock_collections_response
            mock_client.create_collection = Mock()
            mock_client.delete_collection = Mock()

            service.client = mock_client

            return service

    def test_create_collection_new(self, mock_service):
        """Test creating new collection"""
        result = mock_service.create_collection()

        assert result == True
        mock_service.client.create_collection.assert_called_once()

    def test_create_collection_exists_no_recreate(self, mock_service):
        """Test when collection already exists without force recreate"""
        # Mock existing collection
        existing_collection = Mock()
        existing_collection.name = "test_collection"
        mock_collections_response = Mock()
        mock_collections_response.collections = [existing_collection]
        mock_service.client.get_collections.return_value = mock_collections_response

        result = mock_service.create_collection(force_recreate=False)

        assert result == True
        mock_service.client.create_collection.assert_not_called()

    def test_create_collection_exists_with_recreate(self, mock_service):
        """Test recreating existing collection"""
        # Mock existing collection
        existing_collection = Mock()
        existing_collection.name = "test_collection"
        mock_collections_response = Mock()
        mock_collections_response.collections = [existing_collection]
        mock_service.client.get_collections.return_value = mock_collections_response

        result = mock_service.create_collection(force_recreate=True)

        assert result == True
        mock_service.client.delete_collection.assert_called_once_with(
            collection_name="test_collection"
        )
        mock_service.client.create_collection.assert_called_once()


class TestDocumentIndexing:
    """Test document indexing functionality"""

    @pytest.fixture
    def mock_embeddings(self):
        embeddings = Mock()
        embeddings.embed_documents.return_value = [[0.1] * 768, [0.2] * 768]
        return embeddings

    @pytest.fixture
    def mock_service(self, mock_embeddings):
        """Mock vector service for testing"""
        config = VectorStoreConfig(batch_size=2)

        with patch("src.rag.vector_store.QdrantClient"), patch(
            "src.rag.vector_store.AsyncQdrantClient"
        ), patch("src.rag.vector_store.QdrantVectorStore"):

            service = QdrantVectorService(mock_embeddings, config)

            # Mock vector store
            mock_vector_store = Mock()
            mock_vector_store.add_documents.return_value = ["id1", "id2"]
            service.vector_store = mock_vector_store

            # Mock client for create_collection
            mock_client = Mock()
            mock_collections_response = Mock()
            mock_collections_response.collections = []
            mock_client.get_collections.return_value = mock_collections_response
            service.client = mock_client

            return service

    @pytest.fixture
    def sample_documents(self):
        """Sample Vietnamese documents for testing"""
        return [
            Document(
                page_content="Trí tuệ nhân tạo đang phát triển rất nhanh",
                metadata={"source": "doc1.txt", "page": 1},
            ),
            Document(
                page_content="Machine learning giúp giải quyết nhiều vấn đề",
                metadata={"source": "doc2.txt", "page": 1},
            ),
            Document(
                page_content="Deep learning sử dụng mạng neural sâu",
                metadata={"source": "doc3.txt", "page": 1},
            ),
        ]

    def test_add_documents_success(self, mock_service, sample_documents):
        """Test successful document addition"""
        document_ids = mock_service.add_documents(sample_documents)

        assert isinstance(document_ids, list)
        assert len(document_ids) > 0

        # Check stats update
        stats = mock_service.get_stats()
        assert stats["total_documents_indexed"] == 3

    def test_add_documents_batch_processing(self, mock_service, sample_documents):
        """Test batch processing with custom batch size"""
        document_ids = mock_service.add_documents(sample_documents, batch_size=1)

        assert isinstance(document_ids, list)
        # Should call add_documents multiple times due to batch size
        assert mock_service.vector_store.add_documents.call_count >= 1

    def test_add_empty_documents(self, mock_service):
        """Test adding empty document list"""
        document_ids = mock_service.add_documents([])

        assert document_ids == []

    def test_add_documents_without_vector_store(self, mock_embeddings):
        """Test adding documents when vector store is not initialized"""
        config = VectorStoreConfig()

        with patch("src.rag.vector_store.QdrantClient"), patch(
            "src.rag.vector_store.AsyncQdrantClient"
        ), patch("src.rag.vector_store.QdrantVectorStore"):

            service = QdrantVectorService(mock_embeddings, config)
            service.vector_store = None  # Simulate uninitialized

            documents = [Document(page_content="test")]
            result = service.add_documents(documents)

            assert result == []


class TestSimilaritySearch:
    """Test similarity search functionality"""

    @pytest.fixture
    def mock_embeddings(self):
        embeddings = Mock()
        embeddings.embed_query.return_value = [0.1] * 768
        return embeddings

    @pytest.fixture
    def mock_service(self, mock_embeddings):
        """Mock vector service for testing"""
        config = VectorStoreConfig()

        with patch("src.rag.vector_store.QdrantClient"), patch(
            "src.rag.vector_store.AsyncQdrantClient"
        ), patch("src.rag.vector_store.QdrantVectorStore"):

            service = QdrantVectorService(mock_embeddings, config)

            # Mock vector store search results
            mock_vector_store = Mock()
            search_results = [
                Document(
                    page_content="Trí tuệ nhân tạo là lĩnh vực khoa học",
                    metadata={"source": "doc1.txt", "score": 0.9},
                ),
                Document(
                    page_content="Machine learning là một phần của AI",
                    metadata={"source": "doc2.txt", "score": 0.8},
                ),
            ]
            mock_vector_store.similarity_search.return_value = search_results
            mock_vector_store.similarity_search_with_score.return_value = [
                (search_results[0], 0.9),
                (search_results[1], 0.8),
            ]

            service.vector_store = mock_vector_store

            return service

    def test_similarity_search_basic(self, mock_service):
        """Test basic similarity search"""
        query = "Tìm kiếm thông tin về AI"
        results = mock_service.similarity_search(query)

        assert isinstance(results, list)
        assert len(results) == 2

        for doc in results:
            assert isinstance(doc, Document)
            assert len(doc.page_content) > 0

        # Check stats update
        stats = mock_service.get_stats()
        assert stats["total_searches_performed"] == 1

    def test_similarity_search_with_parameters(self, mock_service):
        """Test similarity search with custom parameters"""
        query = "Machine learning"
        results = mock_service.similarity_search(query=query, k=5, score_threshold=0.85)

        assert isinstance(results, list)
        mock_service.vector_store.similarity_search.assert_called_once()

    def test_similarity_search_with_filter(self, mock_service):
        """Test similarity search with metadata filter"""
        query = "AI research"
        filter_conditions = {"source": "doc1.txt"}

        results = mock_service.similarity_search(
            query=query, filter_conditions=filter_conditions
        )

        assert isinstance(results, list)
        # Verify filter was built and passed
        call_args = mock_service.vector_store.similarity_search.call_args
        assert "filter" in call_args.kwargs

    def test_similarity_search_with_score(self, mock_service):
        """Test similarity search with scores"""
        query = "Deep learning"
        results = mock_service.similarity_search_with_score(query)

        assert isinstance(results, list)
        assert len(results) == 2

        for doc, score in results:
            assert isinstance(doc, Document)
            assert isinstance(score, (int, float))
            assert 0 <= score <= 1

    def test_similarity_search_without_vector_store(self, mock_embeddings):
        """Test search when vector store is not initialized"""
        config = VectorStoreConfig()

        with patch("src.rag.vector_store.QdrantClient"), patch(
            "src.rag.vector_store.AsyncQdrantClient"
        ), patch("src.rag.vector_store.QdrantVectorStore"):

            service = QdrantVectorService(mock_embeddings, config)
            service.vector_store = None  # Simulate uninitialized

            results = service.similarity_search("test query")
            assert results == []


class TestFilterBuilding:
    """Test filter building functionality"""

    @pytest.fixture
    def mock_service(self):
        """Mock vector service for testing"""
        embeddings = Mock()
        config = VectorStoreConfig()

        with patch("src.rag.vector_store.QdrantClient"), patch(
            "src.rag.vector_store.AsyncQdrantClient"
        ), patch("src.rag.vector_store.QdrantVectorStore"):

            return QdrantVectorService(embeddings, config)

    def test_build_filter_simple_conditions(self, mock_service):
        """Test building filter with simple conditions"""
        conditions = {"source": "document.pdf", "page": 1, "category": "AI"}

        filter_obj = mock_service._build_filter(conditions)

        assert filter_obj is not None
        # Filter should contain field conditions

    def test_build_filter_list_values(self, mock_service):
        """Test building filter with list values"""
        conditions = {"source": ["doc1.txt", "doc2.txt"], "category": "AI"}

        filter_obj = mock_service._build_filter(conditions)
        assert filter_obj is not None

    def test_build_filter_empty_conditions(self, mock_service):
        """Test building filter with empty conditions"""
        filter_obj = mock_service._build_filter({})
        assert filter_obj is None


class TestCollectionInfo:
    """Test collection information retrieval"""

    @pytest.fixture
    def mock_service(self):
        """Mock vector service for testing"""
        embeddings = Mock()
        config = VectorStoreConfig(collection_name="test_collection")

        with patch("src.rag.vector_store.QdrantClient"), patch(
            "src.rag.vector_store.AsyncQdrantClient"
        ), patch("src.rag.vector_store.QdrantVectorStore"):

            service = QdrantVectorService(embeddings, config)

            # Mock collection info
            mock_client = Mock()
            mock_collection_info = Mock()
            mock_collection_info.config.params.vectors = Mock()
            mock_collection_info.config.params.vectors.distance = Distance.COSINE
            mock_collection_info.config.params.vectors.size = 768
            mock_collection_info.points_count = 1000
            mock_collection_info.indexed_vectors_count = 950
            mock_collection_info.status = "green"

            mock_client.get_collection.return_value = mock_collection_info
            service.client = mock_client

            return service

    def test_get_collection_info_success(self, mock_service):
        """Test successful collection info retrieval"""
        info = mock_service.get_collection_info()

        assert isinstance(info, dict)
        assert "name" in info
        assert "vector_size" in info
        assert "distance_metric" in info
        assert "points_count" in info
        assert "indexed_count" in info
        assert "status" in info

        assert info["name"] == "test_collection"
        assert info["points_count"] == 1000

    def test_get_collection_info_without_client(self):
        """Test collection info when client is not initialized"""
        embeddings = Mock()
        config = VectorStoreConfig()

        with patch("src.rag.vector_store.QdrantClient"), patch(
            "src.rag.vector_store.AsyncQdrantClient"
        ), patch("src.rag.vector_store.QdrantVectorStore"):

            service = QdrantVectorService(embeddings, config)
            service.client = None

            info = service.get_collection_info()
            assert info == {}


class TestDocumentDeletion:
    """Test document deletion functionality"""

    @pytest.fixture
    def mock_service(self):
        """Mock vector service for testing"""
        embeddings = Mock()
        config = VectorStoreConfig()

        with patch("src.rag.vector_store.QdrantClient"), patch(
            "src.rag.vector_store.AsyncQdrantClient"
        ), patch("src.rag.vector_store.QdrantVectorStore"):

            service = QdrantVectorService(embeddings, config)

            # Mock client delete method
            mock_client = Mock()
            mock_client.delete = Mock()
            service.client = mock_client

            return service

    def test_delete_documents_by_ids(self, mock_service):
        """Test deleting documents by IDs"""
        document_ids = ["id1", "id2", "id3"]

        result = mock_service.delete_documents(document_ids=document_ids)

        assert result == True
        mock_service.client.delete.assert_called_once()

    def test_delete_documents_by_filter(self, mock_service):
        """Test deleting documents by filter"""
        filter_conditions = {"source": "old_document.txt"}

        result = mock_service.delete_documents(filter_conditions=filter_conditions)

        assert result == True
        mock_service.client.delete.assert_called_once()

    def test_delete_documents_no_parameters(self, mock_service):
        """Test delete with no parameters"""
        result = mock_service.delete_documents()

        assert result == False

    def test_delete_documents_without_client(self):
        """Test delete when client is not initialized"""
        embeddings = Mock()
        config = VectorStoreConfig()

        with patch("src.rag.vector_store.QdrantClient"), patch(
            "src.rag.vector_store.AsyncQdrantClient"
        ), patch("src.rag.vector_store.QdrantVectorStore"):

            service = QdrantVectorService(embeddings, config)
            service.client = None

            result = service.delete_documents(document_ids=["id1"])
            assert result == False


class TestStatistics:
    """Test statistics tracking"""

    @pytest.fixture
    def mock_embeddings(self):
        embeddings = Mock()
        embeddings.embed_documents.return_value = [[0.1] * 768]
        embeddings.embed_query.return_value = [0.1] * 768
        return embeddings

    @pytest.fixture
    def mock_service(self, mock_embeddings):
        """Mock vector service for testing"""
        config = VectorStoreConfig()

        with patch("src.rag.vector_store.QdrantClient"), patch(
            "src.rag.vector_store.AsyncQdrantClient"
        ), patch("src.rag.vector_store.QdrantVectorStore"):

            service = QdrantVectorService(mock_embeddings, config)

            # Mock vector store and client
            mock_vector_store = Mock()
            mock_vector_store.add_documents.return_value = ["id1"]
            mock_vector_store.similarity_search.return_value = []
            service.vector_store = mock_vector_store

            mock_client = Mock()
            mock_collections_response = Mock()
            mock_collections_response.collections = []
            mock_client.get_collections.return_value = mock_collections_response
            service.client = mock_client

            return service

    def test_initial_stats(self, mock_service):
        """Test initial statistics"""
        stats = mock_service.get_stats()

        assert stats["total_documents_indexed"] == 0
        assert stats["total_searches_performed"] == 0
        assert stats["total_indexing_time"] == 0.0
        assert stats["total_search_time"] == 0.0

    def test_stats_after_operations(self, mock_service):
        """Test statistics after operations"""
        # Perform operations
        documents = [Document(page_content="test document")]
        mock_service.add_documents(documents)
        mock_service.similarity_search("test query")

        stats = mock_service.get_stats()

        assert stats["total_documents_indexed"] == 1
        assert stats["total_searches_performed"] == 1
        assert stats["total_indexing_time"] > 0
        assert stats["total_search_time"] > 0

    def test_reset_stats(self, mock_service):
        """Test resetting statistics"""
        # Perform operations first
        documents = [Document(page_content="test document")]
        mock_service.add_documents(documents)

        # Reset stats
        mock_service.reset_stats()

        stats = mock_service.get_stats()
        assert stats["total_documents_indexed"] == 0
        assert stats["total_indexing_time"] == 0.0


class TestFactoryFunctions:
    """Test factory functions"""

    @pytest.fixture
    def mock_embeddings(self):
        return Mock()

    @patch("src.rag.vector_store.QdrantVectorService")
    def test_create_qdrant_vector_store(self, mock_service_class, mock_embeddings):
        """Test Qdrant vector store factory"""
        config = VectorStoreConfig(collection_name="test")

        service = create_qdrant_vector_store(mock_embeddings, config)

        mock_service_class.assert_called_once_with(mock_embeddings, config)

    @patch("src.rag.vector_store.QdrantVectorService")
    def test_create_vietnamese_vector_store(self, mock_service_class, mock_embeddings):
        """Test Vietnamese-optimized vector store factory"""
        service = create_vietnamese_vector_store(mock_embeddings, "vietnamese_docs")

        mock_service_class.assert_called_once()
        # Check config passed to constructor
        call_args = mock_service_class.call_args
        config = call_args[0][1]  # Second argument
        assert config.collection_name == "vietnamese_docs"
        assert config.vector_size == 768
        assert config.distance_metric == Distance.COSINE


class TestAsyncOperations:
    """Test async operations"""

    @pytest.fixture
    def mock_embeddings(self):
        embeddings = Mock()
        embeddings.embed_documents.return_value = [[0.1] * 768]
        return embeddings

    @pytest.fixture
    def mock_service(self, mock_embeddings):
        """Mock vector service for testing"""
        config = VectorStoreConfig()

        with patch("src.rag.vector_store.QdrantClient"), patch(
            "src.rag.vector_store.AsyncQdrantClient"
        ), patch("src.rag.vector_store.QdrantVectorStore"):

            service = QdrantVectorService(mock_embeddings, config)

            # Mock vector store
            mock_vector_store = Mock()
            mock_vector_store.add_documents.return_value = ["id1"]
            service.vector_store = mock_vector_store

            # Mock client
            mock_client = Mock()
            mock_collections_response = Mock()
            mock_collections_response.collections = []
            mock_client.get_collections.return_value = mock_collections_response
            service.client = mock_client

            return service

    @pytest.mark.asyncio
    async def test_async_add_documents(self, mock_service):
        """Test async document addition"""
        documents = [Document(page_content="async test document")]

        document_ids = await mock_service.aadd_documents(documents)

        assert isinstance(document_ids, list)
        assert len(document_ids) > 0


if __name__ == "__main__":
    pytest.main([__file__])
