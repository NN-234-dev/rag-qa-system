# ============================================
# RAG REST API Server — 通用 HTTP 接口
# ============================================
# 任何语言、任何大模型应用通过 HTTP 调用即可获得 RAG 能力。
#
# 启动: python src/api_server.py
# 文档: http://localhost:8000/docs (FastAPI 自动生成 Swagger UI)
#
# 使用示例:
#   curl -X POST http://localhost:8000/ask \
#     -H "Content-Type: application/json" \
#     -d '{"question":"什么是RAG?","username":"default"}'
#
#   curl -X POST http://localhost:8000/search \
#     -H "Content-Type: application/json" \
#     -d '{"query":"微服务部署","top_k":5}'
# ============================================

import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from typing import Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
import uvicorn

import config
from src.qa_chain import ask, ask_stream
from src.vector_store import search_similar, get_collection_info, get_all_user_collections
from src.knowledge_db import list_documents, get_total_documents, get_all_users
from src.hybrid_retriever import get_hybrid_retriever, rebuild_hybrid_index
from src.document_loader import load_document
from src.text_splitter import split_documents
from src.vector_store import add_documents
from src.knowledge_db import add_document as db_add_document

# ============================================
# 数据模型
# ============================================

class SearchRequest(BaseModel):
    query: str = Field(..., description="搜索查询词")
    top_k: int = Field(default=5, ge=1, le=20, description="返回结果数量")
    username: str = Field(default="default", description="用户名")
    include_shared: bool = Field(default=True, description="是否搜索公共库")


class AskRequest(BaseModel):
    question: str = Field(..., description="用户问题")
    top_k: int = Field(default=5, ge=1, le=20, description="检索文档数量")
    temperature: float = Field(default=None, ge=0.0, le=1.0, description="生成温度")
    username: str = Field(default="default", description="用户名")
    include_shared: bool = Field(default=True, description="是否搜索公共库")


class SearchResult(BaseModel):
    query: str
    total_found: int
    results: list[dict]


class AskResult(BaseModel):
    question: str
    search_query: str
    answer: str
    sources: list[dict]
    stats: dict


class DocListResult(BaseModel):
    username: str
    total_documents: int
    total_chunks: int
    shared_chunks_available: int
    documents: list[dict]


class HealthResult(BaseModel):
    status: str
    ollama_connected: bool
    model: str
    embedding_model: str
    users: list[str]


# ============================================
# FastAPI 应用
# ============================================

app = FastAPI(
    title="RAG Knowledge Base API",
    description="""
## 通用 RAG 知识库 HTTP API

**任何大模型应用** 通过 HTTP 调用即可获得 RAG 能力。

### 核心能力

| 端点 | 功能 | 适用场景 |
|------|------|---------|
| `POST /search` | 语义搜索，返回文档片段 | Agent 查找相关知识 |
| `POST /ask` | 完整 RAG 问答，返回答案+来源 | 用户直接提问 |
| `GET /docs` | 列出知识库所有文档 | 了解有哪些资料 |
| `POST /context` | 获取完整文档上下文 | 代码生成、深度分析 |

### 使用方式

任何语言都能调用——Python requests、curl、JavaScript fetch、LangChain Tool...

```python
import requests
resp = requests.post("http://localhost:8000/ask",
    json={"question": "什么是RAG?", "username": "default"})
print(resp.json()["answer"])
```
    """,
    version="1.0.0",
)


# ============================================
# 启动事件
# ============================================

@app.on_event("startup")
async def startup():
    """启动时检查 Ollama 连接"""
    try:
        from openai import OpenAI
        client = OpenAI(base_url=config.OLLAMA_BASE_URL, api_key="ollama")
        client.models.list()
        print("[API] Ollama 连接正常")
    except Exception:
        print("[API] 警告: 无法连接 Ollama，问答功能将不可用")


# ============================================
# API 端点
# ============================================

@app.get("/health", response_model=HealthResult)
async def health_check():
    """健康检查 + 系统信息"""
    ollama_ok = False
    try:
        from openai import OpenAI
        client = OpenAI(base_url=config.OLLAMA_BASE_URL, api_key="ollama")
        client.models.list()
        ollama_ok = True
    except Exception:
        pass

    users = list(set(get_all_users() + get_all_user_collections()))

    return {
        "status": "ok" if ollama_ok else "degraded",
        "ollama_connected": ollama_ok,
        "model": config.OLLAMA_MODEL,
        "embedding_model": config.EMBEDDING_MODEL_NAME,
        "users": users,
    }


@app.post("/search", response_model=SearchResult)
async def search(req: SearchRequest):
    """语义搜索知识库，返回相关文档片段"""
    docs = search_similar(
        req.query,
        top_k=req.top_k,
        username=req.username,
        include_shared=req.include_shared,
    )

    results = []
    for i, doc in enumerate(docs, 1):
        results.append({
            "rank": i,
            "score": doc.metadata.get("similarity_score", "N/A"),
            "source": doc.metadata.get("source", "unknown"),
            "kb_source": "private" if doc.metadata.get("kb_source") == "private" else "shared",
            "content": doc.page_content,
            "content_length": len(doc.page_content),
        })

    return {
        "query": req.query,
        "total_found": len(results),
        "results": results,
    }


@app.post("/ask", response_model=AskResult)
async def ask_question(req: AskRequest):
    """完整 RAG 问答：检索 + 生成 + 来源引用"""
    result = ask(
        req.question,
        top_k=req.top_k,
        temperature=req.temperature,
        username=req.username,
        include_shared=req.include_shared,
    )

    return {
        "question": req.question,
        "search_query": result.get("search_query", req.question),
        "answer": result["answer"],
        "sources": result["sources"],
        "stats": {
            "docs_retrieved": result.get("docs_retrieved", 0),
            "docs_relevant": result.get("docs_relevant", 0),
            "retrieval_time": round(result.get("retrieval_time", 0), 3),
            "rerank_time": round(result.get("rerank_time", 0), 3),
            "generation_time": round(result.get("generation_time", 0), 3),
        },
    }


@app.get("/docs/{username}", response_model=DocListResult)
@app.get("/docs", response_model=DocListResult)
async def list_docs(username: str = "default"):
    """列出知识库文档"""
    docs = list_documents(username=username)
    info = get_collection_info(username=username)

    doc_list = []
    for doc in docs:
        doc_list.append({
            "id": doc["id"],
            "filename": doc["filename"],
            "file_type": doc["file_type"],
            "file_size_kb": round(doc["file_size"] / 1024, 1),
            "chunk_count": doc["chunk_count"],
            "upload_time": doc["upload_time"],
        })

    return {
        "username": username,
        "total_documents": len(doc_list),
        "total_chunks": info.get("private_chunks", 0),
        "shared_chunks_available": info.get("shared_chunks", 0),
        "documents": doc_list,
    }


@app.post("/context", response_model=SearchResult)
async def get_context(req: SearchRequest):
    """获取完整文档上下文（同 search，但明确语义：返回完整内容供 Agent 使用）"""
    return await search(req)


# ============================================
# 入口
# ============================================

def main():
    print("=" * 50)
    print("RAG Knowledge Base API Server")
    print(f"  Ollama: {config.OLLAMA_BASE_URL}")
    print(f"  Model:  {config.OLLAMA_MODEL}")
    print(f"  Embed:  {config.EMBEDDING_MODEL_NAME}")
    print(f"  Docs:   http://localhost:8000/docs")
    print(f"  API:    http://localhost:8000")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
