# ============================================
# 文本分块器 — 递归分块 + 重叠控制
# ============================================

from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    docs: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> List[Document]:
    """
    使用 RecursiveCharacterTextSplitter 对文档进行语义分块。

    面试点：为什么用 Recursive 而不是固定长度？
    → Recursive 按 段落(\\n\\n) → 句子(\\n) → 词(空格) → 字符 的优先级切分，
      尽可能保持语义完整性，避免把一句话切成两半。

    Args:
        docs: 待分块的文档列表
        chunk_size: 每块的最大字符数
        chunk_overlap: 相邻块之间的重叠字符数

    Returns:
        分块后的文档片段列表
    """
    # separators 定义了切分优先级：先按段落切，再按句子，最后按字符
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""],
        is_separator_regex=False,
    )

    chunks = text_splitter.split_documents(docs)

    # 为每个 chunk 添加序号元数据（用于后续引用定位）
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    return chunks


def get_chunk_stats(chunks: List[Document]) -> dict:
    """
    统计分块信息，用于调试和参数调优。
    """
    if not chunks:
        return {"count": 0, "avg_length": 0, "min_length": 0, "max_length": 0}

    lengths = [len(chunk.page_content) for chunk in chunks]
    return {
        "count": len(chunks),
        "avg_length": sum(lengths) / len(lengths),
        "min_length": min(lengths),
        "max_length": max(lengths),
    }
