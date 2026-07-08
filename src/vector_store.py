# ============================================
# 向量数据库操作 — Chroma 封装（多用户版）
# ============================================
# 支持多用户隔离：
# - 每个用户一个独立 Chroma collection（kb_user_xxx）
# - 一个公共 collection（kb_shared）全员可搜
# - 懒加载 + 缓存，避免重复创建 Chroma 实例

from typing import List, Optional, Dict
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
import config
from src.embeddings import get_embedding_model
from src.multi_user import (
    get_user_collection_name,
    get_shared_collection_name,
)


# 全局缓存：{collection_name: Chroma instance}
_stores: Dict[str, Chroma] = {}


def _get_store(collection_name: str) -> Chroma:
    """获取指定 collection 的 Chroma 实例（懒加载 + 缓存）"""
    if collection_name not in _stores:
        _stores[collection_name] = Chroma(
            collection_name=collection_name,
            embedding_function=get_embedding_model(),
            persist_directory=str(config.CHROMA_DIR),
            collection_metadata={"hnsw:space": "cosine"},
        )
    return _stores[collection_name]


# ============================================
# 向后兼容：保留原有的全局接口
# 默认操作当前用户的私人库
# ============================================

def get_vector_store(username: str = None) -> Chroma:
    """
    获取向量库实例。

    Args:
        username: 用户名。None 时使用 config.DEFAULT_USER。
    """
    if username is None:
        username = config.DEFAULT_USER
    collection_name = get_user_collection_name(username)
    return _get_store(collection_name)


def get_shared_store() -> Chroma:
    """获取公共知识库实例"""
    return _get_store(get_shared_collection_name())


# ============================================
# 写入操作 —— 始终写入当前用户的私人库
# ============================================

def add_documents(
    docs: List[Document],
    doc_id: Optional[str] = None,
    username: str = None,
    to_shared: bool = False,
) -> int:
    """
    将文档块添加到向量库。

    Args:
        docs: 文档块列表
        doc_id: 可选的文档标识符，用于后续按文档删除
        username: 目标用户。None 时使用默认用户。
        to_shared: 是否添加到公共库（True 则忽略 username）

    Returns:
        添加的文档块数量
    """
    if to_shared:
        store = get_shared_store()
    else:
        store = get_vector_store(username)

    if doc_id:
        for doc in docs:
            doc.metadata["doc_id"] = doc_id

    store.add_documents(docs)
    return len(docs)


# ============================================
# 检索操作 —— 同时查私人库 + 公共库
# ============================================

def search_similar(
    query: str,
    top_k: int = None,
    username: str = None,
    include_shared: bool = None,
) -> List[Document]:
    """
    语义相似度检索。

    同时搜索当前用户的私人库和公共库，合并结果后按分数排序。

    Args:
        query: 查询文本
        top_k: 返回结果数量
        username: 用户名
        include_shared: 是否包含公共库。None 时使用 config.ENABLE_SHARED_KB

    Returns:
        最相似的文档块列表（含相似度分数）
    """
    if top_k is None:
        top_k = config.TOP_K
    if include_shared is None:
        include_shared = config.ENABLE_SHARED_KB

    # 搜索私人库
    private_store = get_vector_store(username)
    private_results = private_store.similarity_search_with_score(query, k=top_k * 2)

    all_results = []
    for doc, score in private_results:
        doc.metadata["similarity_score"] = round(score, 4)
        doc.metadata["kb_source"] = "private"  # 标记来源
        all_results.append((score, doc))

    # 搜索公共库
    if include_shared:
        shared_store = get_shared_store()
        shared_collection = shared_store._collection
        if shared_collection.count() > 0:  # 公共库有数据才搜
            shared_results = shared_store.similarity_search_with_score(
                query, k=top_k
            )
            for doc, score in shared_results:
                doc.metadata["similarity_score"] = round(score, 4)
                doc.metadata["kb_source"] = "shared"  # 标记来源
                all_results.append((score, doc))

    # 按相似度分数排序（cosine 距离越小越相似）
    all_results.sort(key=lambda x: x[0])

    # 取 top_k
    docs_with_score = []
    for _, doc in all_results[:top_k]:
        docs_with_score.append(doc)

    return docs_with_score


# ============================================
# 删除操作
# ============================================

def delete_by_doc_id(doc_id: str, username: str = None) -> bool:
    """
    按文档 ID 删除向量数据。
    先在私人库找，找不到再去公共库找。
    """
    # 尝试私人库
    store = get_vector_store(username)
    collection = store._collection
    results = collection.get(where={"doc_id": doc_id})
    ids_to_delete = results.get("ids", [])

    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
        return True

    # 尝试公共库
    shared_store = get_shared_store()
    shared_collection = shared_store._collection
    results = shared_collection.get(where={"doc_id": doc_id})
    ids_to_delete = results.get("ids", [])

    if ids_to_delete:
        shared_collection.delete(ids=ids_to_delete)
        return True

    return False


# ============================================
# 统计与维护
# ============================================

def get_collection_info(username: str = None) -> dict:
    """获取向量库统计信息"""
    private_store = get_vector_store(username)
    shared_store = get_shared_store()

    private_count = private_store._collection.count()
    shared_count = shared_store._collection.count()

    return {
        "username": username or config.DEFAULT_USER,
        "private_chunks": private_count,
        "shared_chunks": shared_count,
        "total_chunks": private_count + shared_count,
        "persist_dir": str(config.CHROMA_DIR),
    }


def get_all_user_collections() -> List[str]:
    """
    扫描 Chroma 持久化目录，返回所有用户名列表。
    通过检查 SQLite 中的 collection 表来发现已有用户。
    """
    import sqlite3
    import os

    chroma_sqlite = config.CHROMA_DIR / "chroma.sqlite3"
    if not chroma_sqlite.exists():
        return [config.DEFAULT_USER]

    try:
        conn = sqlite3.connect(str(chroma_sqlite))
        rows = conn.execute(
            "SELECT name FROM collections WHERE name LIKE ?",
            (f"{get_user_collection_name('')}%",)
        ).fetchall()
        conn.close()

        users = []
        for (name,) in rows:
            if name.startswith("kb_user_"):
                username = name[len("kb_user_"):]
                if username:
                    users.append(username)

        return users if users else [config.DEFAULT_USER]
    except Exception:
        return [config.DEFAULT_USER]


def reset_collection(username: str = None):
    """清空指定用户的向量库"""
    if username is None:
        username = config.DEFAULT_USER
    collection_name = get_user_collection_name(username)
    if collection_name in _stores:
        _stores[collection_name].delete_collection()
        del _stores[collection_name]


def reset_shared_collection():
    """清空公共库"""
    shared_name = get_shared_collection_name()
    if shared_name in _stores:
        _stores[shared_name].delete_collection()
        del _stores[shared_name]


def reset_all():
    """清空所有用户的向量库和公共库"""
    for name in list(_stores.keys()):
        _stores[name].delete_collection()
    _stores.clear()
