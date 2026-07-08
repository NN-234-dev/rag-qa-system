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
    """混合检索：语义 + BM25 + RRF 融合。
    RRF 只关心排名，不需要归一化不同检索器的原始分数。"""

    def __init__(self):
        self._bm25: BM25Okapi = None
        self._corpus: List[Document] = []
        self._tokenized_corpus: List[List[str]] = []

    def _tokenize(self, text: str) -> List[str]:
        """jieba 分词"""
        return list(jieba.cut(text))

    def build_bm25_index(self, docs: List[Document]):
        """文档变更后重建 BM25 索引"""
        self._corpus = docs
        self._tokenized_corpus = [
            self._tokenize(doc.page_content) for doc in docs
        ]
        if self._tokenized_corpus:
            self._bm25 = BM25Okapi(self._tokenized_corpus)

    def _bm25_search(self, query: str, top_k: int) -> List[Document]:
        if not self._bm25 or not self._corpus:
            return []

        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

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
        """混合检索主入口"""
        if top_k is None:
            top_k = config.TOP_K

        fetch_k = top_k * 3  # 各取 3 倍候选，融合后裁到 top_k

        semantic_docs = search_similar(
            query, top_k=fetch_k,
            username=username, include_shared=include_shared,
        )

        bm25_docs = self._bm25_search(query, top_k=fetch_k)

        return self._rrf_fusion(semantic_docs, bm25_docs, top_k)

    def _rrf_fusion(
        self,
        results_a: List[Document],
        results_b: List[Document],
        top_k: int,
        k: int = 60,
    ) -> List[Document]:
        """RRF: score = 1/(k+rank_a) + 1/(k+rank_b)，用内容前 200 字符做去重 key"""
        scores = {}
        doc_map = {}

        for rank, doc in enumerate(results_a, 1):
            doc_id = doc.page_content[:200]
            rrf_score = 1.0 / (k + rank)
            scores[doc_id] = scores.get(doc_id, 0) + rrf_score
            doc_map[doc_id] = doc

        for rank, doc in enumerate(results_b, 1):
            doc_id = doc.page_content[:200]
            rrf_score = 1.0 / (k + rank)
            scores[doc_id] = scores.get(doc_id, 0) + rrf_score
            doc_map[doc_id] = doc

        sorted_doc_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

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
