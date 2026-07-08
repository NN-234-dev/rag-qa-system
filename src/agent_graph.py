# ============================================
# LangGraph Agent — 将 RAG 管线升级为自主决策的 Agent 系统
# ============================================
#
# 架构（状态图）：
#   START
#     │
#     ▼
#   ┌──────────┐
#   │ Planner  │  LLM 拆解问题 → N 个子问题
#   └────┬─────┘
#        │
#        ▼
#   ┌──────────┐
#   │Researcher│  每个子问题并行检索，汇总文档
#   └────┬─────┘
#        │
#        ▼
#   ┌──────────┐
#   │ Verifier │  结构性检查：缺信息？结果矛盾？
#   └────┬─────┘
#        │
#        ▼
#   ┌──────────┐
#   │Synthesizer│  综合所有检索结果，写出完整回答
#   └────┬─────┘
#        │
#        ▼
#   ┌──────────┐
#   │  Critic  │  自我反思：答案完整吗？需要补充搜索吗？
#   └────┬─────┘
#        │
#   ┌────┴─────┐
#   │ 完整?    │─── 否 + iter<MAX ──→ 回到 Planner
#   └────┬─────┘
#        │ 是
#        ▼
#   ┌──────────┐
#   │ 最终输出  │
#   └──────────┘
#
# 面试核心卖点：
# 1. Planning-Action-Observation-Reflection 完整 Agent 循环
# 2. LangGraph StateGraph + ConditionalEdge 实现自主决策
# 3. 复用现有 RAG 组件（检索器、LLM），不重写基础设施
# 4. 每个节点职责单一，可独立测试和优化
#
# 与现有 qa_chain.ask() 的区别：
# - qa_chain:  用户提问 → 改写 → 检索 → 回答    （一锤子买卖）
# - agent_graph: 提问 → 拆解 → 并行检索 → 验证 → 综合 → 反思 → 不够再搜
#   （Agent 自主决定搜什么、搜几轮、答案够不够）

import time
import re
from typing import TypedDict, Annotated, List, Dict, Any, Optional, Generator
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

import config
from src.llm_client import generate, generate_stream
from src.query_rewriter import rewrite_query
from src.hybrid_retriever import get_hybrid_retriever, rebuild_hybrid_index
from src.llm_reranker import rerank


# ============================================
# State 定义 — Agent 在节点间传递的状态
# ============================================

class AgentState(TypedDict):
    """Agent 全局状态，在节点间流转。

    面试要点：TypedDict 不是普通 dict——
    LangGraph 用 key 做 state channel，每个节点返回部分更新即可。
    """
    question: str                          # 用户原始问题
    sub_questions: List[str]               # Planner 拆解出的子问题
    research_results: Dict[str, Any]       # {sub_q: {"docs": [...], "has_info": bool}}
    verification: str                      # Verifier 的检查结论
    draft_answer: str                      # Synthesizer 的初稿
    critique: str                          # Critic 的反思
    needs_more_research: bool              # 是否需要补充搜索（条件边的判断依据）
    iteration: int                         # 当前循环次数（防无限循环）
    final_answer: str                      # 最终输出
    sources: List[Dict[str, Any]]          # 引用来源
    # 计时信息
    planner_time: float
    research_time: float
    verify_time: float
    synthesize_time: float
    critique_time: float
    total_retrieval_time: float
    total_generation_time: float


# ============================================
# Prompt 模板 — 每个节点的系统提示
# ============================================

PLANNER_PROMPT = """你是一个问题分析专家。用户提出了一个问题，你需要将其拆解为多个独立的子问题，用于从知识库中分别检索相关信息。

## 拆解规则
1. 识别问题中的多个维度/方面，每个维度拆为一个子问题
2. 如果问题涉及比较/对比，分别拆出各方的独立子问题
3. 子问题应该是独立的、可检索的，用词精确、正式
4. 子问题数量控制在 2-5 个，简单问题可以不拆解（1个子问题）
5. 子问题之间不应有重复

## 输出格式
每行一个子问题，不要编号、不要前缀、不要引号、不要额外解释。

## 示例1
用户: Python和Java在性能上有什么区别？

子问题:
Python语言性能特点与基准测试数据
Java语言JIT编译优化与性能特点
Python与Java在计算密集型任务上的性能对比

## 示例2
用户: 如何部署系统到生产环境？

子问题:
生产环境部署流程与步骤
生产环境服务器配置要求
部署安全注意事项与最佳实践

## 用户问题
{question}

## 子问题:"""


# Researcher 为每个子问题做简短摘要的 prompt（轻量，低延迟）
RESEARCHER_SUMMARY_PROMPT = """根据以下检索到的文档片段，用 2-3 句话总结与问题相关的关键信息。只陈述事实，不要编造。

## 子问题
{sub_question}

## 检索到的文档
{context}

## 关键信息总结（2-3句话）:"""


SYNTHESIZER_PROMPT = """你是一个基于知识库的问答专家。用户提出了一个问题，系统将其拆解为多个子问题并分别检索了相关信息。请综合所有研究成果，给出一份完整、准确的回答。

## 规则
- 综合所有子问题的检索结果，写出一份逻辑连贯的回答
- 只使用参考资料中的信息，不要编造
- 如果某个方面资料中没有相关信息，在回答中诚实说明
- 引用具体的资料片段并标注来源
- 用中文回答，结构清晰

## 用户原始问题
{question}

## 各子问题的研究成果
{research_summary}

## 回答:"""


CRITIC_PROMPT = """你是一个严谨的审稿人。请审查以下回答是否完整、准确。

## 审查标准
1. 用户原始问题的所有方面是否都被覆盖？
2. 回答是否引用了具体的来源？
3. 是否存在编造或推测的内容？
4. 是否有明显的信息缺口需要补充检索？

## 用户原始问题
{question}

## 系统给出的回答
{answer}

## 审查结论
请简要说明：
- 如果回答完整且准确，回复 "PASS"
- 如果存在信息缺口，回复 "NEED_MORE: <具体缺什么信息，需要搜索什么>" """


# ============================================
# 工具函数
# ============================================

def _doc_to_dict(doc) -> dict:
    """将 LangChain Document 转为可序列化的 dict（LangGraph state 要求）"""
    return {
        "content": doc.page_content,
        "metadata": dict(doc.metadata),
    }


def _build_context(docs: List[dict]) -> str:
    """将文档 dict 列表拼接为上下文文本"""
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.get("metadata", {}).get("source", "未知来源")
        content = doc.get("content", "")
        parts.append(f"[参考资料 {i}] 来源: {source}\n{content}")
    return "\n\n---\n\n".join(parts)


def _extract_sources(docs: List[dict]) -> List[Dict[str, Any]]:
    """从文档列表中提取去重的来源信息"""
    sources = []
    seen = set()
    for doc in docs:
        source = doc.get("metadata", {}).get("source", "未知")
        if source not in seen:
            seen.add(source)
            sources.append({
                "source": source,
                "content_preview": doc.get("content", "")[:100] + "...",
                "kb_source": doc.get("metadata", {}).get("kb_source", "private"),
            })
    return sources


# ============================================
# 节点实现
# ============================================

def planner_node(state: AgentState) -> dict:
    """
    节点1: 规划器 (Planner)

    用 LLM 将用户问题拆解为 N 个独立的子问题。
    每个子问题将在 Researcher 节点中独立检索。

    面试要点：这是 Agent 的 Planning 环节——
    不是简单地把一句话扔给搜索引擎，而是先思考"我需要知道什么"。
    """
    t0 = time.time()
    question = state["question"]

    prompt = PLANNER_PROMPT.format(question=question)
    raw = generate(prompt, temperature=config.AGENT_TEMPERATURE)

    # 解析：每行一个子问题，跳过空行和明显的非问题行
    sub_questions = []
    for line in raw.strip().split("\n"):
        line = line.strip()
        # 跳过空行、以及 "子问题:" "思考:" 等噪音
        if not line:
            continue
        if line.startswith("子问题") or line.startswith("思考") or line.startswith("注意"):
            continue
        # 清理可能的编号前缀 "1. " "- " "· " 等
        line = re.sub(r'^[\d]+[\.\、\)\s]+', '', line)
        line = re.sub(r'^[-·•\*\s]+', '', line)
        if len(line) >= 4:  # 有效子问题至少4个字符
            sub_questions.append(line)

    # 限制数量
    sub_questions = sub_questions[:config.AGENT_MAX_SUB_QUESTIONS]

    # Fallback: 如果解析失败，用原问题作为唯一子问题
    if not sub_questions:
        sub_questions = [question]

    return {
        "sub_questions": sub_questions,
        "iteration": state.get("iteration", 0),
        "planner_time": time.time() - t0,
    }


def researcher_node(state: AgentState) -> dict:
    """
    节点2: 研究员 (Researcher)

    对每个子问题执行完整检索管线：
    - 查询改写（可选，将子问题转为搜索查询）
    - 混合检索（语义 + BM25，复用现有 HybridRetriever）
    - LLM 重排序（可选，精筛 top_k 篇）
    - 每个子问题生成简短摘要

    面试要点：
    - 多子问题并行检索 = Agent 的 Action 环节
    - 复用现有 RAG 管线（hybrid_retriever + reranker），体现代码复用
    - 每个子问题独立执行，互不干扰
    """
    t0 = time.time()
    sub_questions = state["sub_questions"]
    username = state.get("username", config.DEFAULT_USER)
    include_shared = state.get("include_shared", config.ENABLE_SHARED_KB)
    total_retrieval_time = 0.0

    research_results = {}
    all_docs_for_sources = []

    for i, sub_q in enumerate(sub_questions):
        # Step 1: 查询改写（将子问题优化为搜索查询）
        search_query = sub_q
        if config.ENABLE_QUERY_REWRITING:
            search_query = rewrite_query(sub_q, temperature=config.REWRITE_TEMPERATURE)

        # Step 2: 混合检索
        fetch_k = config.TOP_K * config.RERANK_CANDIDATE_MULTIPLIER if config.ENABLE_LLM_RERANK else config.TOP_K
        t_ret = time.time()
        retriever = get_hybrid_retriever()
        if retriever._bm25 is None:
            rebuild_hybrid_index(username=username, include_shared=include_shared)
        docs = retriever.search(
            search_query, top_k=fetch_k,
            username=username, include_shared=include_shared,
        )

        # 相似度阈值过滤
        relevant_docs = [
            d for d in docs
            if d.metadata.get("similarity_score", 0) < config.SIMILARITY_THRESHOLD
        ]

        # Step 3: LLM 重排序
        if config.ENABLE_LLM_RERANK and len(relevant_docs) > config.TOP_K:
            relevant_docs = rerank(sub_q, relevant_docs, top_k=config.TOP_K)

        total_retrieval_time += time.time() - t_ret

        # Step 4: 生成简短摘要（轻量 LLM 调用，提炼关键信息）
        has_info = len(relevant_docs) > 0
        summary = ""
        if has_info:
            context = _build_context([_doc_to_dict(d) for d in relevant_docs])
            summary_prompt = RESEARCHER_SUMMARY_PROMPT.format(
                sub_question=sub_q, context=context
            )
            summary = generate(summary_prompt, temperature=0.2)
        else:
            summary = "[未检索到相关信息]"

        # 存储该子问题的研究成果
        docs_dicts = [_doc_to_dict(d) for d in relevant_docs]
        research_results[sub_q] = {
            "summary": summary,
            "docs": docs_dicts,
            "has_info": has_info,
            "search_query": search_query,
        }
        all_docs_for_sources.extend(docs_dicts)

    return {
        "research_results": research_results,
        "sources": _extract_sources(all_docs_for_sources),
        "total_retrieval_time": state.get("total_retrieval_time", 0) + total_retrieval_time,
        "research_time": time.time() - t0,
    }


def verifier_node(state: AgentState) -> dict:
    """
    节点3: 验证器 (Verifier)

    结构性检查（不调用 LLM，零延迟）：
    1. 是否有子问题完全没检索到信息？
    2. 检索结果之间是否有潜在冲突关键词？
    3. 覆盖度评估：有多少子问题找到了信息？

    面试要点：不是每个验证都需要 LLM——
    结构性的检查可以快速过滤明显问题，节省推理成本。
    """
    t0 = time.time()
    research_results = state["research_results"]
    sub_questions = state["sub_questions"]

    missing = []
    found = []
    for sub_q in sub_questions:
        result = research_results.get(sub_q, {})
        if result.get("has_info", False):
            found.append(sub_q)
        else:
            missing.append(sub_q)

    total = len(sub_questions)
    found_count = len(found)
    coverage = found_count / total if total > 0 else 0

    # 构建验证报告
    lines = []
    lines.append(f"子问题总数: {total}, 检索到信息: {found_count}, 覆盖率: {coverage:.0%}")

    if missing:
        lines.append(f"\n⚠️ 以下子问题未检索到相关信息:")
        for m in missing:
            lines.append(f"  - {m}")
        lines.append(f"\n建议: 尝试更宽泛的搜索词或检查知识库是否包含相关内容")

    # 检查潜在矛盾（简单关键词检测）
    all_summaries = " ".join([
        research_results.get(q, {}).get("summary", "") for q in sub_questions
    ])
    contradiction_keywords = ["但是", "然而", "相反", "矛盾", "不一致", "与...不同"]
    potential_conflicts = [kw for kw in contradiction_keywords if kw in all_summaries]

    if potential_conflicts:
        lines.append(f"\n🔍 检测到潜在矛盾信号，需在综合阶段注意: {', '.join(potential_conflicts)}")

    if coverage >= 0.5:
        lines.append("\n✅ 信息覆盖度尚可，可以进入综合阶段")
    else:
        lines.append("\n❌ 信息覆盖度不足，建议补充检索")

    return {
        "verification": "\n".join(lines),
        "verify_time": time.time() - t0,
    }


def synthesizer_node(state: AgentState) -> dict:
    """
    节点4: 综合写作器 (Synthesizer)

    将所有子问题的研究成果合并，写出一份逻辑连贯的完整回答。

    这是 Agent 的 "交付物" ——
    不是把检索结果堆在一起，而是综合、比较、组织成一篇回答。
    """
    t0 = time.time()
    question = state["question"]
    research_results = state["research_results"]

    # 构建研究成果摘要（每个子问题: 摘要 + 关键文档片段）
    parts = []
    for sub_q, result in research_results.items():
        parts.append(f"### 子问题: {sub_q}")
        parts.append(f"检索查询: {result.get('search_query', sub_q)}")
        parts.append(f"摘要: {result.get('summary', '无')}")
        # 附加最相关的文档片段（取第一个文档的前 300 字符）
        docs = result.get("docs", [])
        if docs:
            first_doc_content = docs[0].get("content", "")[:300]
            parts.append(f"关键片段: {first_doc_content}")
        parts.append("")

    research_summary = "\n".join(parts)

    prompt = SYNTHESIZER_PROMPT.format(
        question=question,
        research_summary=research_summary,
    )
    answer = generate(prompt, temperature=config.AGENT_TEMPERATURE)

    return {
        "draft_answer": answer,
        "synthesize_time": time.time() - t0,
    }


def critic_node(state: AgentState) -> dict:
    """
    节点5: 自我审查器 (Critic)

    用 LLM 审查 Synthesizer 给出的回答是否完整、准确。
    如果发现信息缺口 → 设置 needs_more_research = True
    → 条件边会路由回 Planner 进行补充搜索。

    面试要点：这是 Agent 的 Reflection 环节——
    不是"生成完就完了"，而是自我审视"我给的答案够不够好"。
    这正是 Agent 和普通管线的本质区别。
    """
    t0 = time.time()
    question = state["question"]
    answer = state["draft_answer"]

    prompt = CRITIC_PROMPT.format(question=question, answer=answer)
    critique = generate(prompt, temperature=0.2)

    # 判断是否需要更多研究
    needs_more = "NEED_MORE" in critique.upper()

    # 但如果已经达到最大循环次数，强制停止
    iteration = state.get("iteration", 0)
    if iteration >= config.MAX_AGENT_ITERATIONS:
        needs_more = False

    # 如果不需要更多研究，设置最终答案
    final_answer = answer if not needs_more else ""

    return {
        "critique": critique,
        "needs_more_research": needs_more,
        "final_answer": final_answer,
        "iteration": iteration,  # 保持不变，在条件边后递增
        "critique_time": time.time() - t0,
    }


# ============================================
# 条件边：决定是否继续循环
# ============================================

def _should_continue(state: AgentState) -> str:
    """
    Critic 之后的条件路由：
    - 如果需要更多研究 且 未达最大循环 → 回到 Planner
    - 否则 → 结束，输出最终答案

    面试要点：这是 LangGraph ConditionalEdge 的经典用法——
    Agent 基于自身状态做决策，形成自主控制循环。
    """
    if state.get("needs_more_research", False):
        return "planner"
    return "end"


# ============================================
# 循环迭代处理：回到 Planner 前递增计数器
# ============================================

def _increment_iteration(state: AgentState) -> dict:
    """在回到 Planner 之前递增循环计数，并把 Critic 的建议注入为新的搜索方向"""
    iteration = state.get("iteration", 0) + 1
    # 将 Critic 的反馈附加到原始问题上，让 Planner 知道缺什么
    critique = state.get("critique", "")
    original_question = state["question"]

    # 构建增强版问题：原文 + 缺失方向
    enhanced_question = original_question
    if "NEED_MORE" in critique.upper():
        # 提取 NEED_MORE 后面的内容
        match = re.search(r'NEED_MORE[:\s]*(.*)', critique, re.IGNORECASE)
        if match:
            missing_info = match.group(1).strip()
            enhanced_question = (
                f"{original_question}\n\n"
                f"[补充说明：之前的回答不够完整，请特别关注以下缺失信息：{missing_info}]"
            )

    return {
        "iteration": iteration,
        "question": enhanced_question,
        "needs_more_research": False,  # 重置
    }


# ============================================
# 构建 StateGraph
# ============================================

def build_agent_graph():
    """
    组装 LangGraph StateGraph。

    节点: planner → researcher → verifier → synthesizer → critic
    边:   critic → [条件判断] → planner (loop) 或 END

    Returns:
        编译后的 LangGraph graph 对象
    """
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("planner", planner_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("verifier", verifier_node)
    workflow.add_node("synthesizer", synthesizer_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("increment", _increment_iteration)

    # 设置入口
    workflow.set_entry_point("planner")

    # 添加边
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "verifier")
    workflow.add_edge("verifier", "synthesizer")
    workflow.add_edge("synthesizer", "critic")

    # 条件边：Critic 决定下一步
    workflow.add_conditional_edges(
        "critic",
        _should_continue,
        {
            "planner": "increment",  # 需要更多研究 → 先递增迭代计数
            "end": END,              # 完成 → 结束
        }
    )
    workflow.add_edge("increment", "planner")  # 递增后回到 Planner

    return workflow.compile()


# ============================================
# 公开 API — 与现有 qa_chain.ask() 兼容
# ============================================

def agent_ask(
    question: str,
    username: str = None,
    include_shared: bool = None,
) -> dict:
    """
    Agent 模式问答（非流式）。

    与 qa_chain.ask() 签名兼容，内部使用 LangGraph Agent 执行
    Planning → Research → Verify → Synthesize → Critique 循环。

    Args:
        question: 用户问题
        username: 用户名（多用户隔离）
        include_shared: 是否检索公共库

    Returns:
        dict: {
            "answer": str,
            "sources": list[dict],
            "sub_questions": list[str],
            "iterations": int,
            "critique": str,
            "timing": {...},
        }
    """
    if username is None:
        username = config.DEFAULT_USER
    if include_shared is None:
        include_shared = config.ENABLE_SHARED_KB

    graph = build_agent_graph()

    initial_state: AgentState = {
        "question": question,
        "sub_questions": [],
        "research_results": {},
        "verification": "",
        "draft_answer": "",
        "critique": "",
        "needs_more_research": False,
        "iteration": 0,
        "final_answer": "",
        "sources": [],
        "planner_time": 0,
        "research_time": 0,
        "verify_time": 0,
        "synthesize_time": 0,
        "critique_time": 0,
        "total_retrieval_time": 0,
        "total_generation_time": 0,
        "username": username,
        "include_shared": include_shared,
    }

    # 执行 graph
    final_state = graph.invoke(initial_state)

    # 计算总时间
    total_gen = (
        final_state.get("planner_time", 0)
        + final_state.get("research_time", 0)
        + final_state.get("synthesize_time", 0)
        + final_state.get("critique_time", 0)
    )

    return {
        "answer": final_state.get("final_answer") or final_state.get("draft_answer", ""),
        "sources": final_state.get("sources", []),
        "sub_questions": final_state.get("sub_questions", []),
        "iterations": final_state.get("iteration", 0),
        "critique": final_state.get("critique", ""),
        "verification": final_state.get("verification", ""),
        "research_results": final_state.get("research_results", {}),
        "retrieval_time": final_state.get("total_retrieval_time", 0),
        "generation_time": total_gen,
    }


def agent_ask_stream(
    question: str,
    username: str = None,
    include_shared: bool = None,
) -> Generator[dict, None, None]:
    """
    Agent 模式问答（流式输出）。

    逐步 yield Agent 的内部状态变化，适合 Streamlit 实时展示：
    - {"type": "phase", "phase": "planning", "text": "..."}
    - {"type": "phase", "phase": "researching", "text": "..."}
    - {"type": "phase", "phase": "verifying", "text": "..."}
    - {"type": "phase", "phase": "synthesizing", "text": "..."}
    - {"type": "phase", "phase": "critiquing", "text": "..."}
    - {"type": "answer", "text": "..."}
    - {"type": "done", "sources": [...], ...}

    面试要点：流式输出让用户看到 Agent 的"思考过程"——
    不是黑盒等待，而是透明的多步推理。这对用户体验和调试都很重要。
    """
    if username is None:
        username = config.DEFAULT_USER
    if include_shared is None:
        include_shared = config.ENABLE_SHARED_KB

    graph = build_agent_graph()

    initial_state: AgentState = {
        "question": question,
        "sub_questions": [],
        "research_results": {},
        "verification": "",
        "draft_answer": "",
        "critique": "",
        "needs_more_research": False,
        "iteration": 0,
        "final_answer": "",
        "sources": [],
        "planner_time": 0,
        "research_time": 0,
        "verify_time": 0,
        "synthesize_time": 0,
        "critique_time": 0,
        "total_retrieval_time": 0,
        "total_generation_time": 0,
        "username": username,
        "include_shared": include_shared,
    }

    # ============================================
    # 手动执行每个节点，以便在节点间 yield 进度
    # ============================================

    # Phase 1: Planner
    yield {"type": "phase", "phase": "planning", "text": "正在分析问题，拆解为子问题..."}
    state = planner_node(initial_state)
    state = {**initial_state, **state}
    yield {
        "type": "phase", "phase": "planning",
        "text": f"拆解出 {len(state['sub_questions'])} 个子问题",
        "sub_questions": state["sub_questions"],
    }

    # Phase 2: Researcher
    for i, sub_q in enumerate(state["sub_questions"]):
        yield {
            "type": "phase", "phase": "researching",
            "text": f"正在检索子问题 {i+1}/{len(state['sub_questions'])}: {sub_q[:50]}...",
        }
    state = {**state, **researcher_node(state)}
    # 报告检索结果
    found_count = sum(
        1 for r in state["research_results"].values() if r.get("has_info", False)
    )
    yield {
        "type": "phase", "phase": "researching",
        "text": f"检索完成: {found_count}/{len(state['sub_questions'])} 个子问题找到相关信息",
        "research_results": {
            q: {"summary": r["summary"], "has_info": r["has_info"]}
            for q, r in state["research_results"].items()
        },
    }

    # Phase 3: Verifier
    yield {"type": "phase", "phase": "verifying", "text": "正在交叉验证检索结果..."}
    state = {**state, **verifier_node(state)}
    yield {
        "type": "phase", "phase": "verifying",
        "text": state["verification"],
    }

    # Phase 4: Synthesizer
    yield {"type": "phase", "phase": "synthesizing", "text": "正在综合所有信息，撰写回答..."}
    state = {**state, **synthesizer_node(state)}

    # Phase 5: Critic
    yield {"type": "phase", "phase": "critiquing", "text": "正在自我审查回答质量..."}
    state = {**state, **critic_node(state)}

    # 检查是否需要循环
    iteration = 0
    while state.get("needs_more_research", False) and iteration < config.MAX_AGENT_ITERATIONS:
        iteration += 1
        yield {
            "type": "phase", "phase": "planning",
            "text": f"审查发现信息不足，第 {iteration} 轮补充搜索...",
        }
        state = {**state, **_increment_iteration(state)}
        state = {**state, **planner_node(state)}
        yield {
            "type": "phase", "phase": "planning",
            "text": f"补充拆解: {len(state['sub_questions'])} 个新子问题",
        }

        # 补充检索
        critic_research = researcher_node(state)
        # 合并研究结果
        existing = state.get("research_results", {})
        existing.update(critic_research.get("research_results", {}))
        state = {
            **state,
            "research_results": existing,
            "total_retrieval_time": state.get("total_retrieval_time", 0)
                + critic_research.get("total_retrieval_time", 0),
        }
        yield {"type": "phase", "phase": "researching", "text": "补充检索完成，重新综合..."}

        # 重新综合
        state = {**state, **synthesizer_node(state)}
        # 重新审查
        state = {**state, **critic_node(state)}
        yield {"type": "phase", "phase": "critiquing", "text": state.get("critique", "")}

    # 输出最终答案
    answer = state.get("final_answer") or state.get("draft_answer", "")
    yield {"type": "answer", "text": answer}

    # 计算总时间
    total_gen = (
        state.get("planner_time", 0)
        + state.get("research_time", 0)
        + state.get("synthesize_time", 0)
        + state.get("critique_time", 0)
    )

    # 输出元信息
    yield {
        "type": "done",
        "sources": state.get("sources", []),
        "sub_questions": state.get("sub_questions", []),
        "iterations": state.get("iteration", iteration),
        "critique": state.get("critique", ""),
        "verification": state.get("verification", ""),
        "retrieval_time": state.get("total_retrieval_time", 0),
        "generation_time": total_gen,
    }
