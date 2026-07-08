"""Chroma 向量库操作（多用户版）—— 每个用户独立 collection，懒加载 + 缓存"""

from typing import List, Optional, Dict
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
import config
from src.embeddings import get_embedding_model
from src.multi_user import (
    get_user_collection_name,
    get_shared_collection_name,
)


_stores: Dict[str, Chroma] = {}


def _get_store(collection_name: str) -> Chroma:
    if collection_name not in _stores:
        _stores[collection_name] = Chroma(
            collection_name=collection_name,
            embedding_function=get_embedding_model(),
            persist_directory=str(config.CHROMA_DIR),
            collection_metadata={"hnsw:space": "cosine"},
        )
    return _stores[collection_name]


def get_vector_store(username: str = None) -> Chroma:
    if username is None:
        username = config.DEFAULT_USER
    return _get_store(get_user_collection_name(username))


def get_shared_store() -> Chroma:
    return _get_store(get_shared_collection_name())


def add_documents(
    docs: List[Document],
    doc_id: Optional[str] = None,
    username: str = None,
    to_shared: bool = False,
) -> int:
    if to_shared:
        store = get_shared_store()
    else:
        store = get_vector_store(username)

    if doc_id:
        for doc in docs:
            doc.metadata["doc_id"] = doc_id

    store.add_documents(docs)
    return len(docs)


def search_similar(
    query: str,
    top_k: int = None,
    username: str = None,
    include_shared: bool = None,
) -> List[Document]:
    """同时搜私人库 + 公共库，合并后按 cosine 距离排序"""
    if top_k is None:
        top_k = config.TOP_K
    if include_shared is None:
        include_shared = config.ENABLE_SHARED_KB

    private_store = get_vector_store(username)
    private_results = private_store.similarity_search_with_score(query, k=top_k * 2)

    all_results = []
    for doc, score in private_results:
        doc.metadata["similarity_score"] = round(score, 4)
        doc.metadata["kb_source"] = "private"
        all_results.append((score, doc))

    if include_shared:
        shared_store = get_shared_store()
        shared_collection = shared_store._collection
        if shared_collection.count() > 0:
            shared_results = shared_store.similarity_search_with_score(query, k=top_k)
            for doc, score in shared_results:
                doc.metadata["similarity_score"] = round(score, 4)
                doc.metadata["kb_source"] = "shared"
                all_results.append((score, doc))

    all_results.sort(key=lambda x: x[0])

    return [doc for _, doc in all_results[:top_k]]


def delete_by_doc_id(doc_id: str, username: str = None) -> bool:
    store = get_vector_store(username)
    collection = store._collection
    ids_to_delete = collection.get(where={"doc_id": doc_id}).get("ids", [])
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
        return True

    shared_store = get_shared_store()
    shared_collection = shared_store._collection
    ids_to_delete = shared_collection.get(where={"doc_id": doc_id}).get("ids", [])
    if ids_to_delete:
        shared_collection.delete(ids=ids_to_delete)
        return True

    return False


def get_collection_info(username: str = None) -> dict:
    private_store = get_vector_store(username)
    shared_store = get_shared_store()
    return {
        "username": username or config.DEFAULT_USER,
        "private_chunks": private_store._collection.count(),
        "shared_chunks": shared_store._collection.count(),
        "persist_dir": str(config.CHROMA_DIR),
    }


def get_all_user_collections() -> List[str]:
    """扫描 Chroma SQLite，返回已存在的用户名列表"""
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
    if username is None:
        username = config.DEFAULT_USER
    collection_name = get_user_collection_name(username)
    if collection_name in _stores:
        _stores[collection_name].delete_collection()
        del _stores[collection_name]


def reset_shared_collection():
    shared_name = get_shared_collection_name()
    if shared_name in _stores:
        _stores[shared_name].delete_collection()
        del _stores[shared_name]


def reset_all():
    for name in list(_stores.keys()):
        _stores[name].delete_collection()
    _stores.clear()
