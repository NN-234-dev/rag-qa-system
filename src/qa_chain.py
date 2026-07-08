"""RAG 问答链 —— 检索 + 生成"""

import time
from typing import List, Optional
from langchain_core.documents import Document
import config
from src.vector_store import search_similar
from src.hybrid_retriever import get_hybrid_retriever, rebuild_hybrid_index
from src.llm_client import generate, generate_stream
from src.query_rewriter import rewrite_query
from src.llm_reranker import rerank


RAG_PROMPT_TEMPLATE = """你是一个基于知识库的问答助手。请根据以下参考资料回答用户问题。

## 规则
- 只使用参考资料中的信息回答，不要编造
- 如果资料中没有相关信息，请明确说"参考资料中未找到相关信息"
- 回答时引用具体的资料片段，标注来源
- 用中文回答

## 参考资料
{context}

## 用户问题
{question}

## 回答"""


def _build_context(docs: List[Document]) -> str:
    """检索到的文档拼成 prompt 上下文"""
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "未知来源")
        content = doc.page_content.strip()
        parts.append(f"[参考资料 {i}] 来源: {source}\n{content}")
    return "\n\n---\n\n".join(parts)


def _build_prompt(question: str, context: str) -> str:
    return RAG_PROMPT_TEMPLATE.format(context=context, question=question)


def ask(
    question: str,
    top_k: int = None,
    temperature: float = None,
    username: str = None,
    include_shared: bool = None,
) -> dict:
    """RAG 问答主流程（非流式）。
    链路：改写 → 混合检索 → rerank → 拼 prompt → LLM 生成"""
    if top_k is None:
        top_k = config.TOP_K
    if temperature is None:
        temperature = config.TEMPERATURE
    if include_shared is None:
        include_shared = config.ENABLE_SHARED_KB

    search_query = question
    if config.ENABLE_QUERY_REWRITING:
        search_query = rewrite_query(question, temperature=config.REWRITE_TEMPERATURE)

    fetch_k = top_k * config.RERANK_CANDIDATE_MULTIPLIER if config.ENABLE_LLM_RERANK else top_k
    t0 = time.time()
    retriever = get_hybrid_retriever()
    if retriever._bm25 is None:
        rebuild_hybrid_index(username=username, include_shared=include_shared)
    docs = retriever.search(search_query, top_k=fetch_k, username=username, include_shared=include_shared)

    relevant_docs = [
        d for d in docs
        if d.metadata.get("similarity_score", 0) < config.SIMILARITY_THRESHOLD
    ]

    if not relevant_docs:
        retrieval_time = time.time() - t0
        return {
            "answer": "参考资料中未找到与您问题相关的信息，请尝试更换问法或上传更多相关文档。",
            "sources": [],
            "retrieval_time": retrieval_time,
            "generation_time": 0,
            "docs_retrieved": len(docs),
            "docs_relevant": 0,
            "search_query": search_query,
        }

    rerank_time = 0
    docs_before_rerank = len(relevant_docs)
    if config.ENABLE_LLM_RERANK and len(relevant_docs) > top_k:
        t_rerank = time.time()
        relevant_docs = rerank(question, relevant_docs, top_k=top_k)
        rerank_time = time.time() - t_rerank

    retrieval_time = time.time() - t0

    # 用原始问题构建 prompt，不是改写后的查询
    context = _build_context(relevant_docs)
    prompt = _build_prompt(question, context)

    t1 = time.time()
    answer = generate(prompt, temperature=temperature)
    generation_time = time.time() - t1

    sources = []
    seen_sources = set()
    for doc in relevant_docs:
        source = doc.metadata.get("source", "未知")
        if source not in seen_sources:
            seen_sources.add(source)
            sources.append({
                "source": source,
                "chunk_index": doc.metadata.get("chunk_index", -1),
                "content_preview": doc.page_content[:100] + "...",
                "kb_source": doc.metadata.get("kb_source", "private"),
            })

    return {
        "answer": answer,
        "sources": sources,
        "retrieval_time": retrieval_time,
        "generation_time": generation_time,
        "docs_retrieved": len(docs),
        "docs_relevant": len(relevant_docs),
        "search_query": search_query,
        "rerank_time": rerank_time,
        "docs_before_rerank": docs_before_rerank,
    }


def ask_stream(
    question: str,
    top_k: int = None,
    temperature: float = None,
    username: str = None,
    include_shared: bool = None,
):
    """RAG 问答主流程（流式），先检索再流式输出 LLM 回答"""
    if top_k is None:
        top_k = config.TOP_K
    if temperature is None:
        temperature = config.TEMPERATURE
    if include_shared is None:
        include_shared = config.ENABLE_SHARED_KB

    search_query = question
    if config.ENABLE_QUERY_REWRITING:
        search_query = rewrite_query(question, temperature=config.REWRITE_TEMPERATURE)

    fetch_k = top_k * config.RERANK_CANDIDATE_MULTIPLIER if config.ENABLE_LLM_RERANK else top_k
    t0 = time.time()
    retriever = get_hybrid_retriever()
    if retriever._bm25 is None:
        rebuild_hybrid_index(username=username, include_shared=include_shared)
    docs = retriever.search(search_query, top_k=fetch_k, username=username, include_shared=include_shared)

    relevant_docs = [
        d for d in docs
        if d.metadata.get("similarity_score", 0) < config.SIMILARITY_THRESHOLD
    ]

    retrieval_time = time.time() - t0

    if not relevant_docs:
        yield {"type": "answer", "text": "参考资料中未找到与您问题相关的信息。"}
        yield {"type": "done", "sources": [], "retrieval_time": retrieval_time, "search_query": search_query}
        return

    rerank_time = 0
    docs_before_rerank = len(relevant_docs)
    if config.ENABLE_LLM_RERANK and len(relevant_docs) > top_k:
        t_rerank = time.time()
        relevant_docs = rerank(question, relevant_docs, top_k=top_k)
        rerank_time = time.time() - t_rerank

    context = _build_context(relevant_docs)
    prompt = _build_prompt(question, context)

    t1 = time.time()
    full_answer = ""
    for chunk in generate_stream(prompt, temperature=temperature):
        full_answer += chunk
        yield {"type": "answer", "text": chunk}

    generation_time = time.time() - t1

    sources = []
    seen_sources = set()
    for doc in relevant_docs:
        source = doc.metadata.get("source", "未知")
        if source not in seen_sources:
            seen_sources.add(source)
            sources.append({
                "source": source,
                "chunk_index": doc.metadata.get("chunk_index", -1),
                "content_preview": doc.page_content[:100] + "...",
                "kb_source": doc.metadata.get("kb_source", "private"),
            })

    yield {
        "type": "done",
        "sources": sources,
        "retrieval_time": retrieval_time,
        "generation_time": generation_time,
        "docs_retrieved": len(docs),
        "docs_relevant": len(relevant_docs),
        "search_query": search_query,
        "rerank_time": rerank_time,
        "docs_before_rerank": docs_before_rerank,
    }
