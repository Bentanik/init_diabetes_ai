# 🇻🇳 RAG System Vietnamese

Hệ thống RAG (Retrieval-Augmented Generation) tối ưu cho tiếng Việt với HuggingFace embeddings và RAGFlow PDF parsing.

## ✨ Features

- ✅ **Privacy First** - Không cần API key, chạy hoàn toàn local
- ✅ **Vietnamese Optimized** - Chunking & prompts tối ưu cho tiếng Việt
- ✅ **Smart PDF Parsing** - RAGFlow extraction cải tiến ~40%
- ✅ **HuggingFace Embeddings** - Models mạnh mẽ cho đa ngôn ngữ
- ✅ **FastAPI + Swagger** - REST API hiện đại với documentation
- ✅ **ChromaDB** - Vector database hiệu suất cao
- 🚀 **GPU Accelerated** - RTX 3050 enabled (236+ texts/second)

## 🚀 Quick Start

### 1. Cài Đặt

```bash
pip install -r requirements_vietnamese_rag.txt
```

### 2. Cấu Hình LLM

```bash
# Windows PowerShell
$env:LLM_BASE_URL="http://localhost:11434/v1"
$env:LLM_API_KEY=""
$env:LLM_MODEL="llama3.2"

# Linux/Mac
export LLM_BASE_URL="http://localhost:11434/v1"
export LLM_API_KEY=""
export LLM_MODEL="llama3.2"
```

### 3. Khởi Động Server

```bash
cd src
python main.py
```

### 4. Performance Check (Recommended)

```bash
# Check current E5 models performance
python test_e5_models.py

# Test final optimized config
python test_final_config.py
```

### 5. Test & Demo

```bash
# Interactive demo với menu
python quick_start.py

# Test toàn bộ hệ thống
python test_rag_system.py

# Test LLM config
python test_llm_simple.py
```

## 🌐 Web Interface

- **API Server**: http://localhost:8000
- **Swagger UI**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/rag/health

## 📚 API Usage

### Upload PDF/DOCX

```bash
curl -X POST "http://localhost:8000/rag/upload_files" \
  -F "files=@document.pdf" \
  -F "preserve_structure=true"
```

### Upload Text

```bash
curl -X POST "http://localhost:8000/rag/upload_text" \
  -H "Content-Type: application/json" \
  -d '{"text": "Nội dung văn bản tiếng Việt"}'
```

### Query

```bash
curl -X POST "http://localhost:8000/rag/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "Câu hỏi của bạn?", "k": 5}'
```

## 📖 Documentation

- 📄 **[HƯỚNG_DẪN_SỬ_DỤNG_RAG.md](HƯỚNG_DẪN_SỬ_DỤNG_RAG.md)** - Hướng dẫn chi tiết
- 🚀 **[GPU_SUCCESS_SUMMARY.md](GPU_SUCCESS_SUMMARY.md)** - GPU setup & performance optimization
- 🎯 **[E5_MIGRATION_SUMMARY.md](E5_MIGRATION_SUMMARY.md)** - E5 models migration & performance
- 🧪 **[test_rag_system.py](test_rag_system.py)** - Test script toàn diện
- 🎮 **[quick_start.py](quick_start.py)** - Interactive demo

## 🏗️ Architecture

```
aiservice/
├── src/
│   ├── api/rag/routes.py      # REST API endpoints
│   ├── rag/
│   │   ├── service.py         # Main RAG service
│   │   ├── embeddings.py      # HuggingFace embeddings
│   │   ├── chunker.py         # RAGFlow PDF + Vietnamese chunking
│   │   ├── vectorstore.py     # ChromaDB vector store
│   │   └── hybrid_retriever.py # 🔥 BM25 + Embedding hybrid search
│   └── core/llm_client.py     # LLM integration
└── data/                      # Local storage
```

### 🎯 **Innovation: Hybrid Retrieval**

**Unified Retrieval Architecture** - Single powerful retriever:

- ✅ **Hybrid Search**: BM25 (keyword) + Embedding (semantic)
- ✅ **Flexible Methods**: `hybrid`, `bm25_only`, `embedding_only`
- ✅ **Performance**: +15-30% accuracy vs embedding-only
- ✅ **Vietnamese Optimized**: Custom tokenization & fusion

## 🛠️ Development

Built for graduation thesis with academic innovation points:

1. **RAGFlow Integration** - Advanced PDF text extraction
2. **Vietnamese NLP** - Language-specific optimizations
3. **Local Privacy** - No external API dependencies
4. **Modular Design** - Easy to extend and customize

## 📞 Support

- 🔍 **Health Check**: `GET /rag/health`
- 📊 **System Info**: `GET /rag/system_info`
- 📖 **API Docs**: http://localhost:8000/docs
- 🚀 **Quick Demo**: `python quick_start.py`

---
