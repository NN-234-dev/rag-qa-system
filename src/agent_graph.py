"""
LangGraph Agent —— RAG 管线 + 多步推理

Planner → Researcher → Verifier → Synthesizer → Critic
   ↑                                                  │
   └── 补充搜索 ←── NEED_MORE ←──────────────────────┘
"""

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


class AgentState(TypedDict):
    """Agent 在各节点间传递的共享状态"""
    question: str
    sub_questions: List[str]
    research_results: Dict[str, Any]   # {sub_q: {"docs": [...], "has_info": bool}}
    verification: str
    draft_answer: str
    critique: str
    needs_more_research: bool          # Critic 的判断结果，决定条件边的路由
    iteration: int
    final_answer: str
    sources: List[Dict[str, Any]]
    # timing
    planner_time: float
    research_time: float
    verify_time: float
    synthesize_time: float
    critique_time: float
    total_retrieval_time: float
    total_generation_time: float
    username: str
    include_shared: bool


# ---- prompts ----

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


# ---- helpers ----

def _doc_to_dict(doc) -> dict:
    """LangChain Document → 可序列化 dict（LangGraph state 里不能存对象）"""
    return {
        "content": doc.page_content,
        "metadata": dict(doc.metadata),
    }


def _build_context(docs: List[dict]) -> str:
    """把文档 dict 列表拼成一段上下文文本"""
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.get("metadata", {}).get("source", "未知来源")
        content = doc.get("content", "")
        parts.append(f"[参考资料 {i}] 来源: {source}\n{content}")
    return "\n\n---\n\n".join(parts)


def _extract_sources(docs: List[dict]) -> List[Dict[str, Any]]:
    """从文档列表中提取去重的引用来源"""
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


# ---- nodes ----

def planner_node(state: AgentState) -> dict:
    """拆解用户问题 → N 个子问题"""
    t0 = time.time()
    question = state["question"]

    prompt = PLANNER_PROMPT.format(question=question)
    raw = generate(prompt, temperature=config.AGENT_TEMPERATURE)

    # 解析 LLM 输出：每行一个子问题
    sub_questions = []
    for line in raw.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("子问题") or line.startswith("思考") or line.startswith("注意"):
            continue
        line = re.sub(r'^[\d]+[\.\、\)\s]+', '', line)
        line = re.sub(r'^[-·•\*\s]+', '', line)
        if len(line) >= 4:
            sub_questions.append(line)

    sub_questions = sub_questions[:config.AGENT_MAX_SUB_QUESTIONS]

    if not sub_questions:
        sub_questions = [question]

    return {
        "sub_questions": sub_questions,
        "iteration": state.get("iteration", 0),
        "planner_time": time.time() - t0,
    }


def researcher_node(state: AgentState) -> dict:
    """每个子问题走一遍检索管线：改写 → 混合检索 → rerank → LLM 摘要"""
    t0 = time.time()
    sub_questions = state["sub_questions"]
    username = state.get("username", config.DEFAULT_USER)
    include_shared = state.get("include_shared", config.ENABLE_SHARED_KB)
    total_retrieval_time = 0.0

    research_results = {}
    all_docs_for_sources = []

    for sub_q in sub_questions:
        search_query = sub_q
        if config.ENABLE_QUERY_REWRITING:
            search_query = rewrite_query(sub_q, temperature=config.REWRITE_TEMPERATURE)

        fetch_k = config.TOP_K * config.RERANK_CANDIDATE_MULTIPLIER if config.ENABLE_LLM_RERANK else config.TOP_K
        t_ret = time.time()
        retriever = get_hybrid_retriever()
        if retriever._bm25 is None:
            rebuild_hybrid_index(username=username, include_shared=include_shared)
        docs = retriever.search(
            search_query, top_k=fetch_k,
            username=username, include_shared=include_shared,
        )

        relevant_docs = [
            d for d in docs
            if d.metadata.get("similarity_score", 0) < config.SIMILARITY_THRESHOLD
        ]

        if config.ENABLE_LLM_RERANK and len(relevant_docs) > config.TOP_K:
            relevant_docs = rerank(sub_q, relevant_docs, top_k=config.TOP_K)

        total_retrieval_time += time.time() - t_ret

        # 每个子问题做个简短摘要，减轻 synthesizer 的上下文压力
        has_info = len(relevant_docs) > 0
        if has_info:
            context = _build_context([_doc_to_dict(d) for d in relevant_docs])
            summary_prompt = RESEARCHER_SUMMARY_PROMPT.format(
                sub_question=sub_q, context=context
            )
            summary = generate(summary_prompt, temperature=0.2)
        else:
            summary = "[未检索到相关信息]"

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
    """结构性检查：覆盖率、信息缺口、矛盾信号。不调 LLM，零延迟"""
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

    lines = []
    lines.append(f"子问题总数: {total}, 检索到信息: {found_count}, 覆盖率: {coverage:.0%}")

    if missing:
        lines.append(f"\n以下子问题未检索到相关信息:")
        for m in missing:
            lines.append(f"  - {m}")

    # 简单关键词检测矛盾信号
    all_summaries = " ".join([
        research_results.get(q, {}).get("summary", "") for q in sub_questions
    ])
    contradiction_keywords = ["但是", "然而", "相反", "矛盾", "不一致"]
    potential_conflicts = [kw for kw in contradiction_keywords if kw in all_summaries]

    if potential_conflicts:
        lines.append(f"\n检测到可能的矛盾信号: {', '.join(potential_conflicts)}")

    if coverage >= 0.5:
        lines.append("\n覆盖度 ok，可以综合")
    else:
        lines.append("\n覆盖度不足，建议补充检索")

    return {
        "verification": "\n".join(lines),
        "verify_time": time.time() - t0,
    }


def synthesizer_node(state: AgentState) -> dict:
    """把各子问题的研究成果合并成一篇完整回答"""
    t0 = time.time()
    question = state["question"]
    research_results = state["research_results"]

    parts = []
    for sub_q, result in research_results.items():
        parts.append(f"### 子问题: {sub_q}")
        parts.append(f"检索查询: {result.get('search_query', sub_q)}")
        parts.append(f"摘要: {result.get('summary', '无')}")
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
    """自我审查：回答完整吗？不完整就设 needs_more_research = True 触发补充搜索"""
    t0 = time.time()
    question = state["question"]
    answer = state["draft_answer"]

    prompt = CRITIC_PROMPT.format(question=question, answer=answer)
    critique = generate(prompt, temperature=0.2)

    needs_more = "NEED_MORE" in critique.upper()

    iteration = state.get("iteration", 0)
    if iteration >= config.MAX_AGENT_ITERATIONS:
        needs_more = False

    final_answer = answer if not needs_more else ""

    return {
        "critique": critique,
        "needs_more_research": needs_more,
        "final_answer": final_answer,
        "iteration": iteration,
        "critique_time": time.time() - t0,
    }


# ---- 条件边 & 循环控制 ----

def _should_continue(state: AgentState) -> str:
    """Critic 之后的路由：needs_more → 回 planner，否则结束"""
    if state.get("needs_more_research", False):
        return "planner"
    return "end"


def _increment_iteration(state: AgentState) -> dict:
    """循环计数器 + 把 Critic 的缺失反馈注入到问题里"""
    iteration = state.get("iteration", 0) + 1
    critique = state.get("critique", "")
    original_question = state["question"]

    enhanced_question = original_question
    if "NEED_MORE" in critique.upper():
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
        "needs_more_research": False,
    }


# ---- 构建 graph ----

def build_agent_graph():
    """组装 StateGraph：planner → researcher → verifier → synthesizer → critic ⇄ planner"""
    workflow = StateGraph(AgentState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("verifier", verifier_node)
    workflow.add_node("synthesizer", synthesizer_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("increment", _increment_iteration)

    workflow.set_entry_point("planner")

    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "verifier")
    workflow.add_edge("verifier", "synthesizer")
    workflow.add_edge("synthesizer", "critic")

    workflow.add_conditional_edges(
        "critic",
        _should_continue,
        {"planner": "increment", "end": END}
    )
    workflow.add_edge("increment", "planner")

    return workflow.compile()


# ---- 公开 API ----

def agent_ask(
    question: str,
    username: str = None,
    include_shared: bool = None,
) -> dict:
    """Agent 模式问答（非流式），签名兼容 qa_chain.ask()"""
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

    final_state = graph.invoke(initial_state)

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
    """Agent 模式问答（流式），逐步 yield 每个阶段的进度，给 Streamlit 用"""
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

    # 逐步执行每个节点，方便在节点间 yield 进度

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

    # 如果有需要，循环补充搜索
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

        critic_research = researcher_node(state)
        existing = state.get("research_results", {})
        existing.update(critic_research.get("research_results", {}))
        state = {
            **state,
            "research_results": existing,
            "total_retrieval_time": state.get("total_retrieval_time", 0)
                + critic_research.get("total_retrieval_time", 0),
        }
        yield {"type": "phase", "phase": "researching", "text": "补充检索完成，重新综合..."}

        state = {**state, **synthesizer_node(state)}
        state = {**state, **critic_node(state)}
        yield {"type": "phase", "phase": "critiquing", "text": state.get("critique", "")}

    answer = state.get("final_answer") or state.get("draft_answer", "")
    yield {"type": "answer", "text": answer}

    total_gen = (
        state.get("planner_time", 0)
        + state.get("research_time", 0)
        + state.get("synthesize_time", 0)
        + state.get("critique_time", 0)
    )

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
