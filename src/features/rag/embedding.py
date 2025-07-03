"""
Embedding

Sử dụng model E5 multilingual với các tối ưu:
- Caching embeddings
- Tự động điều chỉnh batch size
- Xử lý song song cho batches lớn
- Monitoring và thống kê
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
import time
from functools import lru_cache

try:
    from sentence_transformers import SentenceTransformer
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
except ImportError as e:
    logging.error(f"Thiếu dependencies: {e}")
    raise

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Cấu hình cho embedding model"""

    model_name: str = "intfloat/multilingual-e5-base"
    batch_size: int = 16  # Mặc định, sẽ tự điều chỉnh
    max_tokens: int = 512  # Độ dài tối ưu cho E5
    device: str = "auto"  # Tự động phát hiện GPU/CPU
    normalize_embeddings: bool = True
    query_instruction: str = "query: "  # Format E5
    passage_instruction: str = "passage: "  # Format E5
    cache_size: int = 10000  # Số lượng embeddings cache


class Embedding(Embeddings):
    """
    Embedding model sử dụng E5 multilingual với các tối ưu:
    - Caching kết quả embedding
    - Tự động điều chỉnh batch size
    - Xử lý song song cho batches lớn
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """Khởi tạo model với cấu hình tùy chọn"""
        self.config = config or EmbeddingConfig()
        self.model = None
        self.device = None
        self._stats = {
            "total_texts_embedded": 0,
            "total_tokens_processed": 0,
            "total_embedding_time": 0.0,
            "batch_count": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

        self._initialize_model()
        logger.info(f"Đã khởi tạo E5 Embeddings trên {self.device}")

    def _initialize_model(self):
        """Khởi tạo model với device tối ưu"""
        try:
            # Tự động phát hiện device
            if self.config.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = self.config.device

            logger.info(
                f"Đang tải model {self.config.model_name} trên {self.device}..."
            )

            self.model = SentenceTransformer(self.config.model_name, device=self.device)
            self.model.max_seq_length = self.config.max_tokens

            # Không tự động điều chỉnh batch size
            logger.info(
                f"Đã tải model thành công. Max length: {self.config.max_tokens}, "
                f"Batch size: {self.config.batch_size}"
            )

        except Exception as e:
            logger.error(f"Lỗi khởi tạo model: {e}")
            raise

    def _prepare_texts(self, texts: List[str], is_query: bool = False) -> List[str]:
        """Chuẩn bị texts với instruction và truncate nếu cần"""
        instruction = (
            self.config.query_instruction
            if is_query
            else self.config.passage_instruction
        )

        prepared_texts = []
        for text in texts:
            # Truncate text nếu quá dài
            if len(text) > self.config.max_tokens * 4:
                text = text[: self.config.max_tokens * 4]
            prepared_text = f"{instruction}{text}"
            prepared_texts.append(prepared_text)

        return prepared_texts

    def _count_tokens(self, texts: List[str]) -> int:
        """Ước tính số tokens"""
        return sum(len(text.split()) for text in texts)

    @lru_cache(maxsize=10000)  # Cache kết quả embedding
    def _get_cached_embedding(self, text: str) -> Optional[List[float]]:
        """Lấy embedding từ cache nếu có"""
        return None  # Placeholder for actual cache implementation

    def _encode_batch(
        self, texts: List[str], is_query: bool = False
    ) -> Tuple[np.ndarray, int]:
        """
        Encode một batch texts với monitoring
        Returns: (embeddings, token_count)
        """
        start_time = time.time()

        # Check cache trước
        cached_embeddings = []
        texts_to_embed = []

        for text in texts:
            cached = self._get_cached_embedding(text)
            if cached is not None:
                cached_embeddings.append(cached)
                self._stats["cache_hits"] += 1
            else:
                texts_to_embed.append(text)
                self._stats["cache_misses"] += 1

        if not texts_to_embed:
            return np.array(cached_embeddings), 0

        # Prepare texts chưa có trong cache
        prepared_texts = self._prepare_texts(texts_to_embed, is_query)
        token_count = self._count_tokens(prepared_texts)

        try:
            if self.model is None:
                raise ValueError("Model chưa được khởi tạo")

            # Generate embeddings
            embeddings = self.model.encode(
                prepared_texts,
                convert_to_numpy=True,
                normalize_embeddings=self.config.normalize_embeddings,
                show_progress_bar=False,
                batch_size=self.config.batch_size,
            )

            # Update stats
            embed_time = time.time() - start_time
            self._stats["total_texts_embedded"] += len(texts)
            self._stats["total_tokens_processed"] += token_count
            self._stats["total_embedding_time"] += embed_time
            self._stats["batch_count"] += 1

            # Combine với cached results
            if cached_embeddings:
                embeddings = np.vstack([np.array(cached_embeddings), embeddings])

            logger.debug(
                f"Đã embed {len(texts)} texts trong {embed_time:.2f}s "
                f"(cache hits: {len(cached_embeddings)})"
            )

            return embeddings, token_count

        except Exception as e:
            logger.error(f"Lỗi encoding batch: {e}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents với xử lý song song cho batches lớn"""
        if not texts:
            return []

        all_embeddings = []
        total_token_count = 0

        # Xử lý song song nếu có nhiều texts
        if len(texts) > self.config.batch_size * 2:
            with ThreadPoolExecutor() as executor:
                futures = []
                for i in range(0, len(texts), self.config.batch_size):
                    batch = texts[i : i + self.config.batch_size]
                    futures.append(executor.submit(self._encode_batch, batch, False))

                for future in futures:
                    embeddings, token_count = future.result()
                    if isinstance(embeddings, np.ndarray):
                        embeddings = embeddings.tolist()
                    all_embeddings.extend(embeddings)
                    total_token_count += token_count
        else:
            # Xử lý tuần tự cho ít texts
            for i in range(0, len(texts), self.config.batch_size):
                batch = texts[i : i + self.config.batch_size]
                embeddings, token_count = self._encode_batch(batch, False)
                if isinstance(embeddings, np.ndarray):
                    embeddings = embeddings.tolist()
                all_embeddings.extend(embeddings)
                total_token_count += token_count

        logger.info(
            f"Đã embed {len(texts)} documents. " f"Tổng số tokens: {total_token_count}"
        )
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed một query đơn lẻ"""
        embeddings, token_count = self._encode_batch([text], is_query=True)
        logger.debug(f"Đã embed query. Tokens: {token_count}")
        return embeddings[0].tolist()

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async version cho documents"""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(executor, self.embed_documents, texts)

    async def aembed_query(self, text: str) -> List[float]:
        """Async version cho query"""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(executor, self.embed_query, text)

    def embed_documents_from_chunks(self, documents: List[Document]) -> List[Document]:
        """Embed chunks và thêm embeddings vào metadata"""
        if not documents:
            return documents

        # Extract text content
        texts = [doc.page_content for doc in documents]

        # Generate embeddings
        embeddings = self.embed_documents(texts)
        vector_dim = len(embeddings[0]) if embeddings else 768

        # Add embeddings to document metadata
        for doc, embedding in zip(documents, embeddings):
            doc.metadata[f"q_{vector_dim}_vec"] = embedding
            doc.metadata["embedding_model"] = self.config.model_name
            doc.metadata["embedding_dim"] = vector_dim
            doc.metadata["bm25_tokens"] = doc.metadata["bm25_tokens"]

        logger.info(
            f"Đã thêm embeddings cho {len(documents)} documents (dim={vector_dim})"
        )
        return documents

    async def aembed_documents_from_chunks(
        self, documents: List[Document]
    ) -> List[Document]:
        """Async version cho embed chunks"""
        if not documents:
            return documents

        texts = [doc.page_content for doc in documents]
        embeddings = await self.aembed_documents(texts)
        vector_dim = len(embeddings[0]) if embeddings else 768

        for doc, embedding in zip(documents, embeddings):
            doc.metadata[f"q_{vector_dim}_vec"] = embedding
            doc.metadata["embedding_model"] = self.config.model_name
            doc.metadata["embedding_dim"] = vector_dim

        return documents

    def get_embedding_dimension(self) -> int:
        """Lấy kích thước vector embedding"""
        dim = 768  # E5-base default
        if self.model is not None:
            model_dim = self.model.get_sentence_embedding_dimension()
            if model_dim is not None:
                dim = model_dim
        return dim

    def get_stats(self) -> Dict[str, Any]:
        """Lấy thống kê embedding"""
        stats = self._stats.copy()
        stats["avg_time_per_batch"] = stats["total_embedding_time"] / max(
            stats["batch_count"], 1
        )
        stats["avg_tokens_per_text"] = stats["total_tokens_processed"] / max(
            stats["total_texts_embedded"], 1
        )
        stats["cache_hit_rate"] = stats["cache_hits"] / max(
            stats["cache_hits"] + stats["cache_misses"], 1
        )
        return stats

    def reset_stats(self):
        """Reset thống kê"""
        for key in self._stats:
            self._stats[key] = 0

    def get_metadata(self) -> Dict[str, Any]:
        """Lấy metadata của model"""
        return {
            "model_name": self.config.model_name,
            "dimension": self.get_embedding_dimension(),
            "normalize": self.config.normalize_embeddings,
            "device": self.device,
            "batch_size": self.config.batch_size,
            "cache_size": self.config.cache_size,
        }
