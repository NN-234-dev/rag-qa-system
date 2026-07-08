# ============================================
# RAG 问答链 — 核心流水线组装
# ============================================

import time
from typing import List, Optional
from langchain_core.documents import Document
import config
from src.vector_store import search_similar
from src.hybrid_retriever import get_hybrid_retriever, rebuild_hybrid_index
from src.llm_client import generate, generate_stream
from src.query_rewriter import rewrite_query
from src.llm_reranker import rerank


# ============================================
# Prompt 模板
# ============================================
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
    """将检索到的文档拼接为 Prompt 中的上下文"""
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "未知来源")
        content = doc.page_content.strip()
        parts.append(f"[参考资料 {i}] 来源: {source}\n{content}")

    return "\n\n---\n\n".join(parts)


def _build_prompt(question: str, context: str) -> str:
    """构建完整的 Prompt"""
    return RAG_PROMPT_TEMPLATE.format(context=context, question=question)


def ask(
    question: str,
    top_k: int = None,
    temperature: float = None,
    username: str = None,
    include_shared: bool = None,
) -> dict:
    """
    RAG 问答主流程（非流式）。

    完整链路：
    1. 查询改写 → 把口语化问题转为搜索查询
    2. 混合检索 → 语义 + BM25（同时查私人库 + 公共库）
    3. LLM Rerank → 对候选文档逐篇打分精筛
    4. 拼接 Prompt → 将检索结果嵌入模板
    5. 调用 LLM → 生成答案
    6. 返回答案 + 引用来源

    Returns:
        dict: {
            "answer": str,
            "sources": list[dict],
            "retrieval_time": float,
            "generation_time": float,
        }
    """
    if top_k is None:
        top_k = config.TOP_K
    if temperature is None:
        temperature = config.TEMPERATURE
    if include_shared is None:
        include_shared = config.ENABLE_SHARED_KB

    # Step 0: 查询改写（可选，将口语化问题转为精确搜索查询）
    search_query = question
    if config.ENABLE_QUERY_REWRITING:
        search_query = rewrite_query(question, temperature=config.REWRITE_TEMPERATURE)

    # Step 1: 混合检索（语义 + BM25）—— 用改写后的查询
    # 如果启用了 Rerank，多取一些候选文档，然后让 LLM 精筛
    fetch_k = top_k * config.RERANK_CANDIDATE_MULTIPLIER if config.ENABLE_LLM_RERANK else top_k
    t0 = time.time()
    retriever = get_hybrid_retriever()
    if retriever._bm25 is None:
        rebuild_hybrid_index(username=username, include_shared=include_shared)
    docs = retriever.search(search_query, top_k=fetch_k, username=username, include_shared=include_shared)

    # 余弦距离阈值过滤：score 越小越相似，大于阈值的文档不纳入上下文
    relevant_docs = [
        d
        for d in docs
        if d.metadata.get("similarity_score", 0) < config.SIMILARITY_THRESHOLD
    ]

    # 如果过滤后没有相关文档，直接返回
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

    # Step 1.5: LLM Rerank —— 对候选文档逐篇打分，取最相关的 top_k 篇
    rerank_time = 0
    docs_before_rerank = len(relevant_docs)
    if config.ENABLE_LLM_RERANK and len(relevant_docs) > top_k:
        t_rerank = time.time()
        relevant_docs = rerank(question, relevant_docs, top_k=top_k)
        rerank_time = time.time() - t_rerank

    retrieval_time = time.time() - t0

    # Step 2: 构建 Prompt（用原始问题，不是改写后的查询）
    context = _build_context(relevant_docs)
    prompt = _build_prompt(question, context)

    # Step 3: 调用 LLM
    t1 = time.time()
    answer = generate(prompt, temperature=temperature)
    generation_time = time.time() - t1

    # Step 4: 提取来源信息
    sources = []
    seen_sources = set()
    for doc in relevant_docs:
        source = doc.metadata.get("source", "未知")
        if source not in seen_sources:
            seen_sources.add(source)
            sources.append(
                {
                    "source": source,
                    "chunk_index": doc.metadata.get("chunk_index", -1),
                    "content_preview": doc.page_content[:100] + "...",
                    "kb_source": doc.metadata.get("kb_source", "private"),
                }
            )

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
    """
    RAG 问答主流程（流式输出）。

    先检索，再将检索结果 + 问题发给 LLM，流式返回答案。
    最后 yield 来源信息。
    """
    if top_k is None:
        top_k = config.TOP_K
    if temperature is None:
        temperature = config.TEMPERATURE
    if include_shared is None:
        include_shared = config.ENABLE_SHARED_KB

    # Step 0: 查询改写
    search_query = question
    if config.ENABLE_QUERY_REWRITING:
        search_query = rewrite_query(question, temperature=config.REWRITE_TEMPERATURE)

    # Step 1: 混合检索（语义 + BM25）—— 用改写后的查询
    fetch_k = top_k * config.RERANK_CANDIDATE_MULTIPLIER if config.ENABLE_LLM_RERANK else top_k
    t0 = time.time()
    retriever = get_hybrid_retriever()
    if retriever._bm25 is None:
        rebuild_hybrid_index(username=username, include_shared=include_shared)
    docs = retriever.search(search_query, top_k=fetch_k, username=username, include_shared=include_shared)

    relevant_docs = [
        d
        for d in docs
        if d.metadata.get("similarity_score", 0) < config.SIMILARITY_THRESHOLD
    ]

    retrieval_time = time.time() - t0

    if not relevant_docs:
        yield {"type": "answer", "text": "参考资料中未找到与您问题相关的信息。"}
        yield {"type": "done", "sources": [], "retrieval_time": retrieval_time, "search_query": search_query}
        return

    # Step 1.5: LLM Rerank
    rerank_time = 0
    docs_before_rerank = len(relevant_docs)
    if config.ENABLE_LLM_RERANK and len(relevant_docs) > top_k:
        t_rerank = time.time()
        relevant_docs = rerank(question, relevant_docs, top_k=top_k)
        rerank_time = time.time() - t_rerank

    # Step 2 & 3: 构建 Prompt + 流式生成（用原始问题）
    context = _build_context(relevant_docs)
    prompt = _build_prompt(question, context)

    t1 = time.time()
    full_answer = ""
    for chunk in generate_stream(prompt, temperature=temperature):
        full_answer += chunk
        yield {"type": "answer", "text": chunk}

    generation_time = time.time() - t1

    # Step 4: 来源
    sources = []
    seen_sources = set()
    for doc in relevant_docs:
        source = doc.metadata.get("source", "未知")
        if source not in seen_sources:
            seen_sources.add(source)
            sources.append(
                {
                    "source": source,
                    "chunk_index": doc.metadata.get("chunk_index", -1),
                    "content_preview": doc.page_content[:100] + "...",
                    "kb_source": doc.metadata.get("kb_source", "private"),
                }
            )

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
