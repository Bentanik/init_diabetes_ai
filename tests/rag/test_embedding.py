"""
Test suite cho Embedding

Test embedding với multilingual E5 model.
"""

import sys
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock

# Thêm path để import từ src
current_file = Path(__file__)
tests_rag_dir = current_file.parent  # tests/rag/
tests_dir = tests_rag_dir.parent  # tests/
aiservice_dir = tests_dir.parent  # aiservice/
src_dir = aiservice_dir / "src"
sys.path.insert(0, str(src_dir))

# Import modules to test
from src.features.rag.embedding import EmbeddingConfig, Embedding
from langchain_core.documents import Document


class TestEmbeddingConfig:
    """Test embedding configuration"""

    def test_default_config(self):
        """Test default configuration values"""
        config = EmbeddingConfig()

        assert config.model_name == "intfloat/multilingual-e5-base"
        assert config.device == "auto"
        assert config.normalize_embeddings == True
        assert config.batch_size == 16
        assert config.max_tokens == 512
        assert config.query_instruction == "query: "
        assert config.passage_instruction == "passage: "
        assert config.cache_size == 10000

    def test_custom_config(self):
        """Test custom configuration"""
        config = EmbeddingConfig(
            model_name="intfloat/multilingual-e5-large",
            batch_size=32,
            max_tokens=256,
            device="cuda",
            cache_size=5000,
        )

        assert config.model_name == "intfloat/multilingual-e5-large"
        assert config.batch_size == 32
        assert config.max_tokens == 256
        assert config.device == "cuda"
        assert config.cache_size == 5000


class TestEmbedding:
    """Test embedding implementation"""

    @pytest.fixture
    def mock_sentence_transformer(self, monkeypatch):
        """Mock SentenceTransformer để test without loading model"""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1] * 768, [0.2] * 768])
        mock_model.get_sentence_embedding_dimension.return_value = 768

        mock_transformer_class = MagicMock(return_value=mock_model)
        monkeypatch.setattr(
            "src.features.rag.embedding.SentenceTransformer", mock_transformer_class
        )

        return mock_transformer_class, mock_model

    @pytest.fixture
    def embedding_model(self, mock_sentence_transformer):
        """Create embedding model for testing"""
        mock_transformer_class, mock_model = mock_sentence_transformer

        config = EmbeddingConfig(batch_size=2)
        embeddings = Embedding(config)
        embeddings.model = mock_model  # Set the mock model
        return embeddings, mock_model

    def test_initialization(self, embedding_model):
        """Test model initialization"""
        embeddings, mock_model = embedding_model
        assert embeddings.config.model_name == "intfloat/multilingual-e5-base"
        assert embeddings.model is not None

    def test_prepare_texts_query(self, embedding_model):
        """Test text preparation for queries"""
        embeddings, mock_model = embedding_model
        texts = ["Tìm kiếm thông tin về AI"]
        prepared = embeddings._prepare_texts(texts, is_query=True)

        assert len(prepared) == 1
        assert prepared[0].startswith("query: ")
        assert "Tìm kiếm thông tin về AI" in prepared[0]

    def test_prepare_texts_documents(self, embedding_model):
        """Test text preparation for documents"""
        embeddings, mock_model = embedding_model
        texts = [
            "Trí tuệ nhân tạo là lĩnh vực khoa học",
            "Machine learning là một phần của AI",
        ]
        prepared = embeddings._prepare_texts(texts, is_query=False)

        assert len(prepared) == 2
        for text in prepared:
            assert text.startswith("passage: ")

    def test_count_tokens(self, embedding_model):
        """Test token counting functionality"""
        embeddings, mock_model = embedding_model
        texts = ["Đây là một văn bản tiếng Việt để test token counting"]
        token_count = embeddings._count_tokens(texts)

        assert isinstance(token_count, int)
        assert token_count > 0

    def test_embed_query_single(self, embedding_model):
        """Test embedding single query"""
        embeddings, mock_model = embedding_model
        query = "Trí tuệ nhân tạo là gì?"
        embedding = embeddings.embed_query(query)

        assert isinstance(embedding, list)
        assert len(embedding) == 768  # E5-base dimension
        assert all(isinstance(x, (int, float, np.floating)) for x in embedding)

    def test_embed_documents_multiple(self, embedding_model):
        """Test embedding multiple documents"""
        embeddings, mock_model = embedding_model
        documents = [
            "Trí tuệ nhân tạo (AI) là khả năng của máy tính",
            "Machine learning giúp máy tính học từ dữ liệu",
            "Deep learning sử dụng mạng neural sâu",
        ]

        # Reset mock và set side_effect cho batch processing (batch_size=2)
        mock_model.reset_mock()
        mock_model.encode.side_effect = [
            np.array([[0.1] * 768, [0.2] * 768]),  # First batch: 2 docs
            np.array([[0.3] * 768]),  # Second batch: 1 doc
        ]

        embedding_results = embeddings.embed_documents(documents)

        assert isinstance(embedding_results, list)
        assert len(embedding_results) == 3
        assert len(embedding_results[0]) == 768
        assert len(embedding_results[1]) == 768
        assert len(embedding_results[2]) == 768

    def test_embed_documents_from_chunks(self, embedding_model):
        """Test embedding Document objects"""
        embeddings, mock_model = embedding_model
        documents = [
            Document(
                page_content="Tài liệu về trí tuệ nhân tạo",
                metadata={"source": "doc1.txt"},
            ),
            Document(
                page_content="Nghiên cứu về machine learning",
                metadata={"source": "doc2.txt"},
            ),
        ]

        results = embeddings.embed_documents_from_chunks(documents)

        assert len(results) == 2
        for i, result in enumerate(results):
            assert isinstance(result, Document)
            assert result.page_content == documents[i].page_content
            assert "q_768_vec" in result.metadata
            assert "embedding_model" in result.metadata
            assert len(result.metadata["q_768_vec"]) == 768

    def test_batch_processing(self, embedding_model):
        """Test batch processing with large number of documents"""
        embeddings, mock_model = embedding_model
        # Create more documents than batch size
        documents = [f"Document số {i} về trí tuệ nhân tạo" for i in range(5)]

        # Mock batch calls
        mock_model.encode.side_effect = [
            np.array([[0.1] * 768, [0.2] * 768]),  # First batch
            np.array([[0.3] * 768, [0.4] * 768]),  # Second batch
            np.array([[0.5] * 768]),  # Third batch
        ]

        embedding_results = embeddings.embed_documents(documents)

        assert len(embedding_results) == 5
        for embedding in embedding_results:
            assert len(embedding) == 768

    def test_get_embedding_dimension(self, embedding_model):
        """Test getting embedding dimension"""
        embeddings, mock_model = embedding_model
        dim = embeddings.get_embedding_dimension()
        assert dim == 768

    def test_get_stats(self, embedding_model):
        """Test statistics tracking"""
        embeddings, mock_model = embedding_model
        # Process some documents to generate stats
        embeddings.embed_query("test query")

        stats = embeddings.get_stats()

        assert "total_texts_embedded" in stats
        assert "total_tokens_processed" in stats
        assert "total_embedding_time" in stats
        assert "batch_count" in stats
        assert "cache_hits" in stats
        assert "cache_misses" in stats
        assert "cache_hit_rate" in stats

        assert stats["total_texts_embedded"] > 0

    def test_reset_stats(self, embedding_model):
        """Test resetting statistics"""
        embeddings, mock_model = embedding_model
        # Generate some stats first
        embeddings.embed_query("test")

        # Reset
        embeddings.reset_stats()

        stats = embeddings.get_stats()
        assert stats["total_texts_embedded"] == 0
        assert stats["total_embedding_time"] == 0.0
        assert stats["cache_hits"] == 0
        assert stats["cache_misses"] == 0

    def test_get_metadata(self, embedding_model):
        """Test getting model metadata"""
        embeddings, mock_model = embedding_model
        metadata = embeddings.get_metadata()

        assert metadata["model_name"] == "intfloat/multilingual-e5-base"
        assert metadata["dimension"] == 768
        assert metadata["normalize"] == True
        assert metadata["device"] in ["cuda", "cpu"]
        assert metadata["batch_size"] == 2  # Batch size from config
        assert metadata["cache_size"] == 10000


class TestErrorHandling:
    """Test error handling scenarios"""

    def test_empty_text_handling(self, monkeypatch):
        """Test handling empty text"""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.0] * 768])
        mock_model.get_sentence_embedding_dimension.return_value = 768

        mock_transformer_class = MagicMock(return_value=mock_model)
        monkeypatch.setattr(
            "src.features.rag.embedding.SentenceTransformer", mock_transformer_class
        )

        config = EmbeddingConfig()
        embeddings = Embedding(config)
        embeddings.model = mock_model

        # Empty text should still produce embedding
        embedding = embeddings.embed_query("")
        assert len(embedding) == 768

    def test_very_long_text_handling(self, monkeypatch):
        """Test handling very long text"""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1] * 768])
        mock_model.get_sentence_embedding_dimension.return_value = 768

        mock_transformer_class = MagicMock(return_value=mock_model)
        monkeypatch.setattr(
            "src.features.rag.embedding.SentenceTransformer", mock_transformer_class
        )

        config = EmbeddingConfig(max_tokens=512)
        embeddings = Embedding(config)
        embeddings.model = mock_model

        # Create very long text
        long_text = "Đây là một văn bản rất dài. " * 100

        embedding = embeddings.embed_query(long_text)
        assert len(embedding) == 768

    def test_special_characters_handling(self, monkeypatch):
        """Test handling special characters"""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array(
            [[0.1] * 768, [0.2] * 768, [0.3] * 768, [0.4] * 768]
        )
        mock_model.get_sentence_embedding_dimension.return_value = 768

        mock_transformer_class = MagicMock(return_value=mock_model)
        monkeypatch.setattr(
            "src.features.rag.embedding.SentenceTransformer", mock_transformer_class
        )

        config = EmbeddingConfig()
        embeddings = Embedding(config)
        embeddings.model = mock_model

        special_texts = [
            "Text with @#$%^&*() symbols",
            "Emoji test 😀🇻🇳💻",
            "Numbers 123456789",
            "Mixed: text123!@# 中文 русский",
        ]

        embeddings_result = embeddings.embed_documents(special_texts)

        assert len(embeddings_result) == 4
        for embedding in embeddings_result:
            assert len(embedding) == 768


if __name__ == "__main__":
    pytest.main([__file__])
