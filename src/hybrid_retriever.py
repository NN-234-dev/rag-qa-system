# ============================================
# 混合检索器 — 语义检索 + BM25 关键词检索 + RRF 融合
# ============================================

from typing import List
import jieba
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
import config
from src.vector_store import get_vector_store, get_shared_store, search_similar
from src.multi_user import get_user_collection_name, get_shared_collection_name


class HybridRetriever:
    """
    混合检索器：语义检索 + BM25 关键词检索。

    面试点：为什么需要混合检索？
    - 语义检索（向量相似度）：擅长找"意思相近"的内容。
      比如问"怎么退款"能找到"退费流程"，但可能漏掉精确包含"退款"关键词的段落。
    - BM25 关键词检索：擅长精确匹配关键词。
      比如搜索"Q3财报"能精确命中，但对"三季度财务报告"这种同义表达无能为力。
    - 两者结合 → 互补短板，召回率更高。

    RRF (Reciprocal Rank Fusion):
    - 对两种检索结果分别排名，取排名倒数加权求和
    - score = sum(1 / (k + rank_i))，k 是平滑常数（通常为 60）
    - 不需要知道原始相似度分数，只关心排名
    """

    def __init__(self):
        self._bm25: BM25Okapi = None
        self._corpus: List[Document] = []
        self._tokenized_corpus: List[List[str]] = []

    def _tokenize(self, text: str) -> List[str]:
        """使用 jieba 分词，中文关键一步"""
        return list(jieba.cut(text))

    def build_bm25_index(self, docs: List[Document]):
        """
        对所有文档块构建 BM25 索引。
        每次文档变更（上传/删除）后需要重建。
        """
        self._corpus = docs
        self._tokenized_corpus = [
            self._tokenize(doc.page_content) for doc in docs
        ]
        if self._tokenized_corpus:
            self._bm25 = BM25Okapi(self._tokenized_corpus)

    def _bm25_search(self, query: str, top_k: int) -> List[Document]:
        """BM25 关键词检索"""
        if not self._bm25 or not self._corpus:
            return []

        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        # 按分数排序，取 top_k
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in indexed_scores[:top_k]:
            doc = self._corpus[idx]
            doc.metadata["bm25_score"] = round(float(score), 4)
            results.append(doc)

        return results

    def search(
        self,
        query: str,
        top_k: int = None,
        semantic_weight: float = 0.6,
        username: str = None,
        include_shared: bool = None,
    ) -> List[Document]:
        """
        混合检索主方法。

        Args:
            query: 查询文本
            top_k: 返回文档数
            semantic_weight: 语义检索权重（0-1），剩余为 BM25 权重
            username: 用户名（用于多用户隔离）
            include_shared: 是否包含公共库

        Returns:
            融合排序后的文档列表
        """
        if top_k is None:
            top_k = config.TOP_K

        # 每种检索多取一些候选，再融合
        fetch_k = top_k * 3

        # 语义检索（自动查私人库+公共库）
        semantic_docs = search_similar(
            query, top_k=fetch_k,
            username=username, include_shared=include_shared,
        )

        # BM25 检索
        bm25_docs = self._bm25_search(query, top_k=fetch_k)

        # RRF 融合
        merged = self._rrf_fusion(
            semantic_docs, bm25_docs, top_k
        )

        return merged

    def _rrf_fusion(
        self,
        results_a: List[Document],
        results_b: List[Document],
        top_k: int,
        k: int = 60,
    ) -> List[Document]:
        """
        RRF (Reciprocal Rank Fusion) 融合算法。

        对两个排序列表中的文档，根据排名计算 RRF 分数：
        score = 1/(k+rank_a) + 1/(k+rank_b)

        面试点：为什么用 RRF 而不是直接加权？
        → RRF 不需要归一化不同检索器的原始分数（语义相似度和 BM25 分数不可比），
          只关心排名，简单有效。
        """
        scores = {}
        doc_map = {}

        # 处理列表 A（语义检索）
        for rank, doc in enumerate(results_a, 1):
            doc_id = doc.page_content[:200]  # 用内容前 200 字符做去重 key
            rrf_score = 1.0 / (k + rank)
            scores[doc_id] = scores.get(doc_id, 0) + rrf_score
            doc_map[doc_id] = doc

        # 处理列表 B（BM25）
        for rank, doc in enumerate(results_b, 1):
            doc_id = doc.page_content[:200]
            rrf_score = 1.0 / (k + rank)
            scores[doc_id] = scores.get(doc_id, 0) + rrf_score
            doc_map[doc_id] = doc

        # 按 RRF 分数排序
        sorted_doc_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        # 返回 top_k
        results = []
        for doc_id in sorted_doc_ids[:top_k]:
            doc = doc_map[doc_id]
            doc.metadata["rrf_score"] = round(scores[doc_id], 6)
            results.append(doc)

        return results


# 全局混合检索器实例
_hybrid_retriever: HybridRetriever = None


def get_hybrid_retriever() -> HybridRetriever:
    """获取混合检索器实例（懒加载单例）"""
    global _hybrid_retriever
    if _hybrid_retriever is None:
        _hybrid_retriever = HybridRetriever()
    return _hybrid_retriever


def rebuild_hybrid_index(username: str = None, include_shared: bool = None):
    """重建 BM25 索引（文档变更后调用）。
    同时索引当前用户的私人库和公共库。"""
    if include_shared is None:
        include_shared = config.ENABLE_SHARED_KB

    all_docs = []

    # 收集私人库文档
    private_store = get_vector_store(username)
    private_collection = private_store._collection
    private_results = private_collection.get()
    if private_results["ids"]:
        for i in range(len(private_results["ids"])):
            all_docs.append(Document(
                page_content=private_results["documents"][i],
                metadata=private_results["metadatas"][i] if private_results["metadatas"] else {},
            ))

    # 收集公共库文档
    if include_shared:
        shared_store = get_shared_store()
        shared_collection = shared_store._collection
        shared_results = shared_collection.get()
        if shared_results["ids"]:
            for i in range(len(shared_results["ids"])):
                all_docs.append(Document(
                    page_content=shared_results["documents"][i],
                    metadata=shared_results["metadatas"][i] if shared_results["metadatas"] else {},
                ))

    retriever = get_hybrid_retriever()
    retriever.build_bm25_index(all_docs)
