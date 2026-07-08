# ============================================
# MCP Server — 将 RAG 系统封装为 MCP 协议服务
# ============================================
# 使用方式（Claude Code / Cursor / Continue.dev 等 MCP 客户端）:
#
#   在客户端的 MCP 配置中添加:
#   {
#     "rag-kb": {
#       "command": "python",
#       "args": ["-m", "src.mcp_server"],
#       "cwd": "/path/to/rag-qa-system"
#     }
#   }
#
# 然后就可以在对话中直接调用:
#   "搜索我的知识库里关于微服务的文档"
#   "根据知识库帮我回答: 怎么配置负载均衡?"
#   "我的知识库里有哪些文档?"
#
# 设计思路:
# - 暴露 4 个 Tool，粒度从粗到细
# - rag_search: 只检索不生成，给调用方最大灵活性
# - rag_ask: 完整 RAG 问答，适合直接回答用户问题
# - rag_list_docs: 文档发现
# - rag_get_context: 获取指定文档全文，适合代码类 Agent 使用
# ============================================

import sys
import os
from pathlib import Path

# 确保项目根目录在 sys.path 中
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import json
from typing import Optional
import config
from src.qa_chain import ask
from src.vector_store import search_similar, get_collection_info
from src.knowledge_db import list_documents, get_total_documents


# ============================================
# MCP Server 定义
# ============================================

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
except ImportError:
    print(
        "需要安装 mcp 包: pip install mcp\n"
        "如果安装失败，尝试: pip install mcp --index-url https://pypi.tuna.tsinghua.edu.cn/simple",
        file=sys.stderr,
    )
    sys.exit(1)


# 创建 MCP Server 实例
server = Server("rag-knowledge-base")


# ============================================
# Tool 实现
# ============================================

async def _tool_search(
    query: str,
    top_k: int = 5,
    username: str = "default",
    include_shared: bool = True,
) -> str:
    """
    语义搜索知识库，返回最相关的文档片段。

    适用场景：调用方只需要相关上下文，不需要 LLM 生成答案。
    比如 Claude Code 想先看看知识库里有什么相关内容，再决定怎么回答。
    """
    docs = search_similar(
        query,
        top_k=top_k,
        username=username,
        include_shared=include_shared,
    )

    if not docs:
        return json.dumps({
            "query": query,
            "results": [],
            "message": "未找到相关文档。请尝试更换关键词，或确认知识库中已上传文档。"
        }, ensure_ascii=False, indent=2)

    results = []
    for i, doc in enumerate(docs, 1):
        score = doc.metadata.get("similarity_score", "N/A")
        source = doc.metadata.get("source", "未知")
        kb_source = doc.metadata.get("kb_source", "private")
        results.append({
            "rank": i,
            "score": score,
            "source": source,
            "kb_source": "私人库" if kb_source == "private" else "公共库",
            "content": doc.page_content[:500],
            "content_length": len(doc.page_content),
        })

    return json.dumps({
        "query": query,
        "total_found": len(results),
        "results": results,
    }, ensure_ascii=False, indent=2)


async def _tool_ask(
    question: str,
    top_k: int = 5,
    username: str = "default",
    include_shared: bool = True,
) -> str:
    """
    完整 RAG 问答：检索 + 生成答案 + 引用来源。

    适用场景：用户直接提问，需要基于知识库的 AI 回答。
    内部走完整流水线：Query Rewriting → 混合检索 → LLM Rerank → LLM 生成。
    """
    result = ask(
        question,
        top_k=top_k,
        username=username,
        include_shared=include_shared,
    )

    return json.dumps({
        "question": question,
        "search_query": result.get("search_query", question),
        "answer": result["answer"],
        "sources": result["sources"],
        "stats": {
            "docs_retrieved": result.get("docs_retrieved", 0),
            "docs_relevant": result.get("docs_relevant", 0),
            "retrieval_time": round(result.get("retrieval_time", 0), 3),
            "rerank_time": round(result.get("rerank_time", 0), 3),
            "generation_time": round(result.get("generation_time", 0), 3),
        }
    }, ensure_ascii=False, indent=2)


async def _tool_list_docs(
    username: str = "default",
) -> str:
    """
    列出知识库中的所有文档。

    适用场景：调用方想知道知识库中有什么可用的资料。
    """
    docs = list_documents(username=username)

    if not docs:
        return json.dumps({
            "username": username,
            "documents": [],
            "total": 0,
            "message": f"用户 {username} 的知识库中暂无文档。"
        }, ensure_ascii=False, indent=2)

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

    info = get_collection_info(username=username)

    return json.dumps({
        "username": username,
        "total_documents": len(doc_list),
        "total_chunks": info.get("private_chunks", 0),
        "shared_chunks_available": info.get("shared_chunks", 0),
        "documents": doc_list,
    }, ensure_ascii=False, indent=2)


async def _tool_get_context(
    query: str,
    top_k: int = 3,
    username: str = "default",
    include_shared: bool = True,
) -> str:
    """
    获取知识库中与查询相关的完整上下文（不做 LLM 生成）。
    返回完整文档片段，适合作为其他 Agent 的上下文注入。

    与 search 的区别：search 返回摘要（截断500字符），
    get_context 返回完整文档内容，供 Agent 深入分析。

    适用场景：Claude Code 需要基于你的文档写代码/做分析时，
    先拉取完整上下文，再结合自身的推理能力处理。
    """
    docs = search_similar(
        query,
        top_k=top_k,
        username=username,
        include_shared=include_shared,
    )

    if not docs:
        return json.dumps({
            "query": query,
            "contexts": [],
            "message": "未找到相关文档。"
        }, ensure_ascii=False, indent=2)

    contexts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "未知")
        kb_source = doc.metadata.get("kb_source", "private")
        contexts.append({
            "index": i,
            "source": source,
            "kb_source": "私人库" if kb_source == "private" else "公共库",
            "content": doc.page_content,  # 完整内容，不截断
        })

    return json.dumps({
        "query": query,
        "total_contexts": len(contexts),
        "contexts": contexts,
    }, ensure_ascii=False, indent=2)


# ============================================
# 注册 Tools 到 MCP Server
# ============================================

@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="rag_search",
            description=(
                "语义搜索知识库，返回最相关的文档片段（不调用 LLM 生成答案）。"
                "适用场景：Agent 需要查找知识库中的相关信息作为参考上下文。"
                "返回文档片段的摘要（前500字符）、相关性分数、来源文件名和知识库类型（私人/公共）。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索查询词，可以是关键词或自然语言描述"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回结果数量，默认 5",
                        "default": 5
                    },
                    "username": {
                        "type": "string",
                        "description": "用户名，用于知识库隔离。默认 'default'",
                        "default": "default"
                    },
                    "include_shared": {
                        "type": "boolean",
                        "description": "是否同时搜索公共知识库，默认 true",
                        "default": True
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="rag_ask",
            description=(
                "完整 RAG 问答：基于知识库内容生成 AI 回答，附带引用来源。"
                "内部走完整流水线：Query Rewriting → 混合检索(语义+BM25+RRF) → LLM Rerank → LLM 生成。"
                "适用场景：用户直接提问，需要基于文档内容的权威回答。"
                "返回答案、引用来源列表、搜索统计信息（检索耗时、重排耗时、生成耗时）。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "用户问题，自然语言即可"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "检索文档数量，默认 5",
                        "default": 5
                    },
                    "username": {
                        "type": "string",
                        "description": "用户名，默认 'default'",
                        "default": "default"
                    },
                    "include_shared": {
                        "type": "boolean",
                        "description": "是否同时搜索公共知识库，默认 true",
                        "default": True
                    },
                },
                "required": ["question"],
            },
        ),
        Tool(
            name="rag_list_docs",
            description=(
                "列出指定用户知识库中的所有文档。"
                "适用场景：Agent 想了解知识库中有哪些可用的资料。"
                "返回文档列表（文件名、类型、大小、分块数、上传时间）和统计信息。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "username": {
                        "type": "string",
                        "description": "用户名，默认 'default'",
                        "default": "default"
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="rag_get_context",
            description=(
                "获取知识库中与查询相关的完整文档内容（不做 LLM 生成）。"
                "与 rag_search 的区别：返回完整内容而非摘要截断。"
                "适用场景：Agent 需要基于完整文档进行代码编写、分析或决策。"
                "建议先用 rag_list_docs 了解有哪些文档，再用本工具拉取相关内容。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索查询词"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回结果数量，默认 3",
                        "default": 3
                    },
                    "username": {
                        "type": "string",
                        "description": "用户名，默认 'default'",
                        "default": "default"
                    },
                    "include_shared": {
                        "type": "boolean",
                        "description": "是否同时搜索公共知识库，默认 true",
                        "default": True
                    },
                },
                "required": ["query"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """路由 Tool 调用到对应实现"""

    if name == "rag_search":
        result = await _tool_search(
            query=arguments["query"],
            top_k=arguments.get("top_k", 5),
            username=arguments.get("username", "default"),
            include_shared=arguments.get("include_shared", True),
        )
    elif name == "rag_ask":
        result = await _tool_ask(
            question=arguments["question"],
            top_k=arguments.get("top_k", 5),
            username=arguments.get("username", "default"),
            include_shared=arguments.get("include_shared", True),
        )
    elif name == "rag_list_docs":
        result = await _tool_list_docs(
            username=arguments.get("username", "default"),
        )
    elif name == "rag_get_context":
        result = await _tool_get_context(
            query=arguments["query"],
            top_k=arguments.get("top_k", 3),
            username=arguments.get("username", "default"),
            include_shared=arguments.get("include_shared", True),
        )
    else:
        result = json.dumps({"error": f"未知工具: {name}"}, ensure_ascii=False)

    return [TextContent(type="text", text=result)]


# ============================================
# 入口
# ============================================

def main():
    """启动 MCP Server（stdio 传输）"""
    import asyncio

    # 检查 Ollama 连接
    try:
        from openai import OpenAI
        client = OpenAI(base_url=config.OLLAMA_BASE_URL, api_key="ollama")
        client.models.list()
    except Exception:
        print(
            "[警告] 无法连接 Ollama，请确认 Ollama 已启动。"
            "知识库搜索功能可用，但问答生成功能需要 Ollama。",
            file=sys.stderr,
        )

    async def run():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )

    asyncio.run(run())


if __name__ == "__main__":
    main()
