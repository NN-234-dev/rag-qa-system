# ============================================
# LLM 重排序器 — LLM-based Reranker
# ============================================
# 在混合检索之后，用 LLM 对候选文档逐一打分，按相关性重新排序。
#
# 为什么需要 Rerank？
# 混合检索（语义 + BM25 + RRF）只看排名位置，不"读"文档内容。
# LLM 能理解文档和问题的语义关系，过滤掉"看起来相似但其实不相关"的噪音。
#
# 面试要点：
# - 这是 Pointwise Rerank：每篇文档独立打分。
#   另一种是 Listwise：所有文档一起排序，但文档多了 LLM 容易乱。
# - 检索 top_k*2 条 → 打分 → 取 top_k，这是经典的"粗排+精排"模式。
#   类似搜索引擎的：召回(Recall) → 粗排(Rough Rank) → 精排(Fine Rank)

from typing import List
from langchain_core.documents import Document
from src.llm_client import generate


# 评分 Prompt —— 让 LLM 输出 1-5 分
RERANK_PROMPT = """你是一个文档相关性评估专家。请评估以下文档与用户问题的相关程度，给出 1-5 分的评分。

## 用户问题
{question}

## 文档内容
{content}

## 评分标准
- 5: 高度相关，直接包含问题所需的信息
- 4: 相关，提供了有用的背景或部分答案
- 3: 部分相关，有一些联系但不直接
- 2: 略微相关，只有个别词汇匹配
- 1: 不相关，与问题完全无关

## 重要规则
- 只输出一个数字（1、2、3、4、5），不要输出任何其他文字、符号或解释
- 如果文档内容为空或完全无法理解，输出 1"""


def _score_document(question: str, doc: Document) -> float:
    """
    让 LLM 对一篇文档打分。

    Args:
        question: 用户问题
        doc: 候选文档

    Returns:
        0.0~1.0 之间的归一化分数（1-5分 除以 5）
    """
    content = doc.page_content.strip()
    if not content:
        return 0.0

    # 控制文档长度，避免超出模型上下文窗口
    # deepseek-r1 有 128k 上下文，但单篇打分不需要全文
    max_len = 1500
    if len(content) > max_len:
        content = content[:max_len] + "..."

    prompt = RERANK_PROMPT.format(question=question, content=content)

    try:
        raw = generate(prompt, temperature=0.1)
        raw = raw.strip()

        # 尝试提取数字
        import re
        match = re.search(r'[1-5]', raw)
        if match:
            score = int(match.group())
            return score / 5.0  # 归一化到 0~1

        # 如果解析失败，给一个保守的中等分数
        return 0.5

    except Exception:
        # 打分失败，给中等分数，不阻断流程
        return 0.5


def rerank(question: str, docs: List[Document], top_k: int = 5) -> List[Document]:
    """
    对候选文档进行 LLM 重排序。

    流程：
    1. 每篇文档调用 LLM 打分（1-5 分）
    2. 按分数降序排列
    3. 返回 top_k 篇

    Args:
        question: 用户问题（改写后的查询或原始问题）
        docs: 候选文档列表（通常为 top_k * 2 条）
        top_k: 最终返回的文档数

    Returns:
        按相关性重新排序后的文档列表（最多 top_k 条）
    """
    if not docs:
        return []

    # 逐篇打分
    scored_docs = []
    for doc in docs:
        score = _score_document(question, doc)
        doc.metadata["rerank_score"] = score
        scored_docs.append((score, doc))

    # 按分数降序排列
    scored_docs.sort(key=lambda x: x[0], reverse=True)

    # 取 top_k
    result = [doc for _, doc in scored_docs[:top_k]]

    return result
