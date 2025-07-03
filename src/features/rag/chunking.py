"""
Chunking Strategy

Chiến lược chunking tối ưu cho văn bản đa ngôn ngữ (Việt - Anh).

Tính năng:
- Tự động phát hiện cấu trúc và ngôn ngữ
- Chunking thích ứng cho kết quả tối ưu
- Hỗ trợ văn bản hỗn hợp Việt - Anh
- API đơn giản, không cần cấu hình phức tạp
"""

import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import nltk
from nltk.corpus import stopwords
from underthesea import word_tokenize as vi_tokenize

logger = logging.getLogger(__name__)


@dataclass
class ChunkingConfig:
    """Cấu hình cho chunking"""

    chunk_size: int = 512
    chunk_overlap: int = 64
    min_chunk_size: int = 50  # Giảm từ 100 xuống 50 để xử lý text ngắn
    max_chunk_size: int = 1024
    keep_small_chunks: bool = True  # Thêm option để kiểm soát việc giữ/bỏ chunk nhỏ


class MultilingualTextProcessor:
    """Text processor cho văn bản đa ngôn ngữ"""

    # Patterns cho cả Vietnamese và English
    ALL_HEADING_PATTERNS = [
        # Vietnamese patterns - improved
        r"^(CHƯƠNG|PHẦN|MỤC|BÀI|TIẾT|PHỤLỤC|PHỤ LỤC)\s+[IVXLCDM\d]+",
        r"^\d+\.\s+[A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐĐ]",
        r"^[IVXLCDM]+\.\s+[A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐĐ]",
        # English patterns
        r"^(CHAPTER|SECTION|PART|ARTICLE|APPENDIX)\s+[IVXLCDM\d]+",
        r"^\d+\.\s+[A-Z]",
        r"^[IVXLCDM]+\.\s+[A-Z]",
        r"^#{1,6}\s+",  # Markdown
        # Common patterns
        r"^[A-Z][A-Z\s]{2,}$",  # ALL CAPS HEADINGS
        r"^[A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐĐ][A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐĐ\s]{2,}$",  # Vietnamese ALL CAPS
    ]

    LIST_PATTERNS = [
        r"^[-•\*]\s+",
        r"^\d+[\.\)]\s+",
        r"^[a-zA-Z][\.\)]\s+",
        r"^[IVXLCDM]+[\.\)]\s+",
    ]

    @staticmethod
    def detect_language(text: str) -> Dict[str, Any]:
        """Phát hiện ngôn ngữ trong text"""
        vietnamese_chars = (
            r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđĐ]"
        )

        # Đếm số ký tự tiếng Việt
        vietnamese_count = len(re.findall(vietnamese_chars, text, re.IGNORECASE))

        # Đếm tổng số ký tự a-zA-Z + tiếng Việt
        total_chars = len(
            re.findall(
                r"[a-zA-ZàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđĐ]",
                text,
            )
        )

        # Nếu không có ký tự nào, trả về unknown
        if total_chars == 0:
            return {"language": "unknown", "vietnamese_ratio": 0, "confidence": 0}

        # Xác định ngôn ngữ & độ tin cậy
        vietnamese_ratio = vietnamese_count / total_chars

        if (
            vietnamese_ratio > 0.20
        ):  # Tăng threshold lên 0.20 để phân biệt better mixed/vietnamese
            language = "vietnamese"
            confidence = min(vietnamese_ratio * 2, 1.0)
        elif vietnamese_ratio > 0.05:
            language = "mixed"
            confidence = 0.7
        else:
            language = "english"
            confidence = 1.0 - vietnamese_ratio

        return {
            "language": language,
            "vietnamese_ratio": vietnamese_ratio,
            "confidence": confidence,
        }

    @staticmethod
    def analyze_structure(text: str) -> Dict[str, Any]:
        """Phân tích cấu trúc nội dung để chọn strategy chunking tối ưu

        Return dict gồm:
        - has_headings: có tiêu đề không
        - has_lists: có danh sách không
        - has_tables: có bảng không
        - has_code: có đoạn code không
        - paragraph_count: số đoạn văn
        - structure_type: loại cấu trúc văn bản (code, hierarchical, tabular, complex, simple)
        - language_info: kết quả phân tích ngôn ngữ từ MultilingualTextProcessor
        """
        lines = text.split("\n")

        analysis = {
            "has_headings": False,
            "has_lists": False,
            "has_tables": False,
            "has_code": False,
            "paragraph_count": 0,
            "structure_type": "simple",
            "language_info": MultilingualTextProcessor.detect_language(text),
        }

        heading_count = 0
        list_count = 0
        code_count = 0
        paragraph_count = 0

        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue

            # Kiểm tra xem dòng này có phải tiêu đề không
            for pattern in MultilingualTextProcessor.ALL_HEADING_PATTERNS:
                if re.match(pattern, line_stripped, re.IGNORECASE):
                    analysis["has_headings"] = True
                    heading_count += 1
                    break

            # Kiểm tra xem dòng này có phải dòng danh sách không
            for pattern in MultilingualTextProcessor.LIST_PATTERNS:
                if re.match(pattern, line_stripped):
                    analysis["has_lists"] = True
                    list_count += 1
                    break

            # Kiểm tra bảng: dòng có ít nhất 3 dấu '|' coi như bảng
            if "|" in line and line.count("|") >= 2:
                analysis["has_tables"] = True

            # Kiểm tra đoạn code dựa trên tiền tố hoặc các từ khóa lập trình
            if (
                line_stripped.startswith("```")
                or line_stripped.startswith("def ")
                or line_stripped.startswith("class ")
                or line_stripped.startswith("import ")
                or line_stripped.startswith("function ")
            ):
                analysis["has_code"] = True
                code_count += 1

            # Đếm đoạn văn: những dòng có độ dài lớn hơn 10 ký tự tính là đoạn văn
            if len(line_stripped) > 10:
                paragraph_count += 1

        analysis["paragraph_count"] = paragraph_count

        # Quyết định loại cấu trúc văn bản dựa trên kết quả phân tích
        if analysis["has_code"]:
            analysis["structure_type"] = "code"
        elif heading_count > paragraph_count * 0.2:
            # Nếu tỷ lệ tiêu đề lớn > 20% số đoạn văn → cấu trúc phân cấp (hierarchical)
            analysis["structure_type"] = "hierarchical"
        elif analysis["has_tables"]:
            # Nếu có bảng → cấu trúc dạng bảng (tabular)
            analysis["structure_type"] = "tabular"
        elif len(text) > 5000 and paragraph_count > 10:
            # Văn bản dài và nhiều đoạn → cấu trúc phức tạp (complex)
            analysis["structure_type"] = "complex"
        else:
            # Mặc định là văn bản đơn giản (simple)
            analysis["structure_type"] = "simple"

        return analysis

    @staticmethod
    def extract_hierarchical_structure(text: str) -> List[Dict[str, Any]]:
        """
        Trích xuất cấu trúc phân cấp của văn bản dựa trên các mẫu heading.

        Quy trình:
        - Tách văn bản thành các dòng.
        - Duyệt từng dòng, loại bỏ khoảng trắng đầu cuối.
        - Với mỗi dòng, kiểm tra xem có khớp với các pattern heading không.
        - Nếu khớp, ghi nhận vị trí dòng (line_number), cấp độ heading (level) và nội dung dòng (text).
        - Cấp độ được tính bằng công thức (index_pattern % 4) + 1 để chuẩn hóa về 4 cấp.

        Trả về:
        - Danh sách dict, mỗi dict gồm:
            + line_number: vị trí dòng trong văn bản
            + level: cấp độ heading từ 1 đến 4
            + text: nội dung heading
        """

        lines = text.split("\n")
        hierarchy = []

        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                continue

            # Duyệt qua các pattern định nghĩa cho các heading
            for j, pattern in enumerate(MultilingualTextProcessor.ALL_HEADING_PATTERNS):
                # Kiểm tra xem dòng có khớp pattern này không, bỏ qua hoa thường
                match = re.match(pattern, line_stripped, re.IGNORECASE)
                if match:
                    # Nếu khớp, thêm thông tin heading vào danh sách hierarchy
                    hierarchy.append(
                        {
                            "line_number": i,
                            "level": (j % 4) + 1,  # Normalize to 1-4 levels
                            "text": line_stripped,
                        }
                    )
                    break

        return hierarchy

    @staticmethod
    def tokenize_text(text: str, lang_info: Dict[str, Any]) -> List[str]:
        """Tokenize text dựa trên ngôn ngữ đã detect.

        Args:
            text: Text cần tokenize
            lang_info: Thông tin ngôn ngữ từ detect_language()

        Returns:
            List các tokens đã được xử lý
        """
        # Stopwords tiếng Việt
        VIETNAMESE_STOPWORDS = {
            "và",
            "của",
            "cho",
            "được",
            "với",
            "các",
            "có",
            "trong",
            "đã",
            "những",
            "này",
            "về",
            "như",
            "là",
            "để",
            "theo",
            "tại",
            "từ",
            "không",
            "còn",
            "bị",
            "khi",
            "sẽ",
            "nhiều",
            "phải",
            "vì",
            "trên",
            "dưới",
            "nếu",
            "cần",
            "bởi",
            "lúc",
            "họ",
            "tôi",
            "anh",
            "chị",
            "nó",
            "một",
            "hai",
            "ba",
            "bốn",
            "năm",
            "sáu",
            "bảy",
            "tám",
            "chín",
            "mười",
        }

        # Download NLTK data if needed
        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download("stopwords")

        ENGLISH_STOPWORDS = set(stopwords.words("english"))

        # Tokenize based on language
        if lang_info["language"] == "vietnamese":
            # Vietnamese tokenization
            tokens = vi_tokenize(text)
            # Remove Vietnamese stopwords
            tokens = [t for t in tokens if t.lower() not in VIETNAMESE_STOPWORDS]
        else:
            # Default to English tokenization
            tokens = re.findall(r"\b\w+\b", text.lower())
            # Remove English stopwords
            tokens = [t for t in tokens if t not in ENGLISH_STOPWORDS]

        return tokens


class Chunking:
    """
    Chunking chọn strategy chunkin dựa trên nội dung.

    Class này tự động phân tích nội dung và chọn
    strategy phù hợp nhất (simple, hierarchical, semantic).

    Attributes:
        config: Cấu hình chunking (chunk_size, chunk_overlap, etc.)
        text_processor: Instance của MultilingualTextProcessor để phân tích text
        splitter: LangChain RecursiveCharacterTextSplitter đã được tùy chỉnh
    """

    def __init__(self, config: Optional[ChunkingConfig] = None):
        """
        Khởi tạo Chunking với cấu hình mặc định hoặc tùy chỉnh.

        Args:
            config: Cấu hình chunking. Nếu None, sử dụng ChunkingConfig mặc định.
        """
        self.config = config if config is not None else ChunkingConfig()
        self.text_processor = MultilingualTextProcessor()
        self._init_splitters()

    def _init_splitters(self):
        """
        Khởi tạo LangChain text splitters với separators tối ưu cho đa ngôn ngữ.

        Quy trình:
        - Định nghĩa danh sách separators theo thứ tự ưu tiên từ cao đến thấp
        - Bao gồm separators cho cả tiếng Việt/Trung và tiếng Anh
        - Khởi tạo RecursiveCharacterTextSplitter với config tùy chỉnh
        - Sử dụng _count_tokens() làm hàm đếm tokens thông minh
        """
        separators = [
            "\n\n\n",
            "\n\n",
            "\n",
            "。",
            "．",
            "！",
            "？",
            "；",
            ".",
            "!",
            "?",
            ";",
            ":",
            " ",
            "",
        ]

        self.splitter = RecursiveCharacterTextSplitter(
            separators=separators,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=self._count_tokens,
            is_separator_regex=False,
        )

    def _count_tokens(self, text: str) -> int:
        """
        Đếm số tokens ước lượng dựa trên ngôn ngữ được phát hiện.

        Công thức ước lượng tokens:
        - Tiếng Việt: ~3 ký tự/token (do có dấu thanh điệu)
        - Tiếng Anh: ~4 ký tự/token (từ trung bình dài hơn)
        - Hỗn hợp: ~3.5 ký tự/token (trung bình giữa 2 ngôn ngữ)

        Args:
            text: Văn bản cần đếm tokens

        Returns:
            Số tokens ước lượng (int)
        """
        lang_info = self.text_processor.detect_language(text)

        if lang_info["language"] == "vietnamese":
            return len(text) // 3  # Tiếng Việt: chia 3
        elif lang_info["language"] == "english":
            return len(text) // 4  # Tiếng Anh: chia 4
        else:  # mixed
            return int(len(text) / 3.5)  # Hỗn hợp: chia 3.5

    def _prepare_chunk_metadata(
        self, chunk_text: str, base_metadata: Dict[str, Any], strategy: str, **kwargs
    ) -> Dict[str, Any]:
        """Chuẩn bị metadata cho chunk.

        Args:
            chunk_text: Nội dung chunk
            base_metadata: Metadata cơ bản
            strategy: Strategy đang sử dụng
            **kwargs: Metadata bổ sung

        Returns:
            Dict metadata đã được cập nhật
        """
        # Get language info
        lang_info = self.text_processor.detect_language(chunk_text)

        # Generate BM25 tokens
        tokens = self.text_processor.tokenize_text(chunk_text, lang_info)

        # Build metadata
        chunk_metadata = base_metadata.copy()
        chunk_metadata.update(
            {
                "strategy": strategy,
                "token_count": self._count_tokens(chunk_text),
                "language_info": lang_info,
                "bm25_tokens": tokens,
                **kwargs,
            }
        )

        return chunk_metadata

    def _chunk_simple(self, text: str, metadata: Dict[str, Any]) -> List[Document]:
        """
        Chunking đơn giản cho nội dung không có cấu trúc đặc biệt.

        Chia text thành các chunks có kích thước tương đối đều nhau,
        có xử lý các trường hợp đặc biệt như:
        - Merge các chunks quá nhỏ
        - Tránh cắt giữa từ/câu
        - Thêm metadata phù hợp

        Args:
            text: Văn bản cần chunk
            metadata: Metadata gốc để gắn vào các chunks

        Returns:
            List Document chunks với strategy = "simple"
        """
        if not text:
            return []

        splitter = self.splitter
        chunks = splitter.split_text(text)
        documents = []
        doc_index = 0

        for chunk in chunks:
            chunk_stripped = chunk.strip()
            if not chunk_stripped:  # Skip empty chunks
                continue

            # Merge small chunks with previous one if possible
            if (
                len(chunk_stripped) < self.config.min_chunk_size
                and documents
                and len(documents[-1].page_content) + len(chunk_stripped)
                < self.config.max_chunk_size
            ):
                # Merge with previous chunk
                documents[-1].page_content += "\n" + chunk_stripped
                # Update metadata for merged chunk
                documents[-1].metadata = self._prepare_chunk_metadata(
                    documents[-1].page_content,
                    metadata,
                    "simple",
                    chunk_index=doc_index - 1,
                )
                continue
            elif (
                len(chunk_stripped) < self.config.min_chunk_size
                and not self.config.keep_small_chunks
            ):
                continue

            # Prepare metadata for new chunk
            chunk_metadata = self._prepare_chunk_metadata(
                chunk_stripped, metadata, "simple", chunk_index=doc_index
            )

            documents.append(
                Document(page_content=chunk_stripped, metadata=chunk_metadata)
            )
            doc_index += 1

        return documents

    def _chunk_hierarchical(
        self, text: str, metadata: Dict[str, Any]
    ) -> List[Document]:
        """
        Chunking phân cấp cho nội dung có cấu trúc rõ ràng (headings, sections).
        """
        structure = self.text_processor.extract_hierarchical_structure(text)

        if not structure:
            return self._chunk_simple(text, metadata)

        lines = text.split("\n")
        sections = []
        current_section = []
        current_level = float("inf")

        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                if current_section:
                    current_section.append(line)
                continue

            line_level = None
            for heading in structure:
                if heading["line_number"] == i:
                    line_level = heading["level"]
                    break

            if line_level is not None and line_level <= current_level:
                if current_section:
                    section_content = "\n".join(current_section)
                    sections.append(
                        {
                            "content": section_content,
                            "level": current_level,
                            "token_count": self._count_tokens(section_content),
                        }
                    )
                current_section = [line]
                current_level = line_level
            else:
                current_section.append(line)

        if current_section:
            section_content = "\n".join(current_section)
            sections.append(
                {
                    "content": section_content,
                    "level": current_level,
                    "token_count": self._count_tokens(section_content),
                }
            )

        documents = []
        for i, section in enumerate(sections):
            if section["token_count"] > self.config.max_chunk_size:
                sub_chunks = self._chunk_simple(section["content"], metadata)
                for chunk in sub_chunks:
                    chunk.metadata = self._prepare_chunk_metadata(
                        chunk.page_content,
                        metadata,
                        "hierarchical_split",
                        section_level=section["level"],
                        chunk_index=i,
                    )
                documents.extend(sub_chunks)
            else:
                chunk_metadata = self._prepare_chunk_metadata(
                    section["content"],
                    metadata,
                    "hierarchical",
                    section_level=section["level"],
                    chunk_index=i,
                )

                documents.append(
                    Document(
                        page_content=section["content"].strip(), metadata=chunk_metadata
                    )
                )

        return documents

    def _chunk_semantic(self, text: str, metadata: Dict[str, Any]) -> List[Document]:
        """
        Chunking ngữ nghĩa cho nội dung phức tạp dựa trên đoạn văn.
        """
        paragraphs = re.split(r"\n\s*\n", text)
        chunks = []
        current_chunk = []
        current_tokens = 0

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            para_tokens = self._count_tokens(paragraph)

            if (
                current_tokens + para_tokens > self.config.max_chunk_size
                and current_chunk
            ):
                chunk_content = "\n\n".join(current_chunk)
                chunks.append(chunk_content)
                current_chunk = [paragraph]
                current_tokens = para_tokens
            else:
                current_chunk.append(paragraph)
                current_tokens += para_tokens

        if current_chunk:
            chunk_content = "\n\n".join(current_chunk)
            chunks.append(chunk_content)

        documents = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = self._prepare_chunk_metadata(
                chunk, metadata, "semantic", chunk_index=i
            )

            documents.append(Document(page_content=chunk, metadata=chunk_metadata))

        return documents

    def chunk_text(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Chunk text với strategy tối ưu được tự động chọn dựa trên phân tích nội dung.

        Đây là method chính của Chunking, tự động:
        - Phân tích cấu trúc và ngôn ngữ của text
        - Chọn strategy phù hợp (simple/hierarchical/semantic)
        - Áp dụng strategy được chọn
        - Trả về chunks với metadata đầy đủ

        Strategy selection logic:
        - hierarchical: Nếu phát hiện cấu trúc phân cấp (headings > 20% paragraphs)
        - semantic: Nếu text phức tạp/có bảng (length > 5000 & paragraphs > 10)
        - simple: Các trường hợp còn lại (text đơn giản, code)

        Args:
            text: Văn bản cần chunk
            metadata: Optional metadata để gắn vào tất cả chunks

        Returns:
            List Document chunks với strategy tối ưu đã được chọn
        """
        if not text or not text.strip():
            return []

        # Phân tích cấu trúc nội dung
        analysis = self.text_processor.analyze_structure(text)

        # Chuẩn bị metadata
        chunk_metadata = (metadata or {}).copy()
        chunk_metadata.update(
            {"original_length": len(text), "structure_analysis": analysis}
        )

        # Chọn strategy tối ưu dựa trên phân tích
        structure_type = analysis["structure_type"]

        logger.info(
            f"Analyzing text: {structure_type} structure, language: {analysis['language_info']['language']}"
        )

        if structure_type == "hierarchical":
            chunks = self._chunk_hierarchical(text, chunk_metadata)
            logger.info(f"Used hierarchical chunking → {len(chunks)} chunks")
        elif structure_type in ["complex", "tabular"]:
            chunks = self._chunk_semantic(text, chunk_metadata)
            logger.info(f"Used semantic chunking → {len(chunks)} chunks")
        else:  # simple, code
            chunks = self._chunk_simple(text, chunk_metadata)
            logger.info(f"Used simple chunking → {len(chunks)} chunks")

        return chunks

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Chunk nhiều documents cùng lúc, mỗi document được xử lý độc lập.

        Xử lý batch documents với thông tin tracking:
        - Gắn document_index vào metadata của mỗi chunk
        - Log progress cho từng document
        - Tổng hợp tất cả chunks thành một list duy nhất
        - Mỗi document được phân tích và chọn strategy riêng biệt

        Args:
            documents: List các Document objects cần chunk

        Returns:
            List tất cả chunks từ tất cả documents đã được xử lý
        """
        all_chunks = []

        for i, doc in enumerate(documents):
            logger.info(f"Processing document {i+1}/{len(documents)}")

            doc_metadata = doc.metadata.copy()
            doc_metadata["document_index"] = i

            chunks = self.chunk_text(doc.page_content, doc_metadata)
            all_chunks.extend(chunks)

        logger.info(f"Total: {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks


# Simple API functions
def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    metadata: Optional[Dict[str, Any]] = None,
) -> List[Document]:
    """
    Simple API để chunk text với cấu hình tối ưu

    Args:
        text: Text cần chunk
        chunk_size: Kích thước chunk mong muốn
        chunk_overlap: Overlap between chunks
        metadata: Optional metadata

    Returns:
        List of optimally chunked documents
    """
    config = ChunkingConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    chunking = Chunking(config)
    return chunking.chunk_text(text, metadata)


def chunk_documents(
    documents: List[Document], chunk_size: int = 512, chunk_overlap: int = 64
) -> List[Document]:
    """
    Simple API để chunk multiple documents

    Args:
        documents: List of documents to chunk
        chunk_size: Desired chunk size
        chunk_overlap: Overlap between chunks

    Returns:
        List of optimally chunked documents
    """
    config = ChunkingConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    chunking = Chunking(config)
    return chunking.chunk_documents(documents)


# Backward compatibility
def create_multilingual_Chunking(
    chunk_size: int = 512, chunk_overlap: int = 64, **kwargs
) -> Chunking:
    config = ChunkingConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return Chunking(config)


def chunk_multilingual_text(text: str, **kwargs) -> List[Document]:
    return chunk_text(text, **kwargs)


def create_vietnamese_Chunking(*args, **kwargs) -> Chunking:
    return create_multilingual_Chunking(*args, **kwargs)


def chunk_vietnamese_text(*args, **kwargs) -> List[Document]:
    return chunk_text(*args, **kwargs)
