# ============================================
# 知识库元数据管理 — SQLite 增删改查（多用户版）
# ============================================

import sqlite3
import time
from pathlib import Path
from typing import List, Optional
import config


def _get_connection() -> sqlite3.Connection:
    """获取数据库连接（自动创建表和目录）"""
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(config.DB_PATH))
    conn.row_factory = sqlite3.Row
    _init_table(conn)
    return conn


def _init_table(conn: sqlite3.Connection):
    """初始化数据库表 + 向后兼容的 migration"""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            file_type TEXT NOT NULL,
            file_size INTEGER NOT NULL,
            chunk_count INTEGER NOT NULL,
            upload_time REAL NOT NULL,
            file_path TEXT NOT NULL,
            username TEXT NOT NULL DEFAULT 'default'
        )
        """
    )
    # 向后兼容：旧表如果没有 username 列，手动加上
    try:
        conn.execute("SELECT username FROM documents LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute("ALTER TABLE documents ADD COLUMN username TEXT NOT NULL DEFAULT 'default'")
        conn.commit()
    conn.commit()


# ============================================
# CRUD —— 所有查询自动按 username 过滤
# ============================================

def add_document(
    filename: str,
    file_type: str,
    file_size: int,
    chunk_count: int,
    file_path: str,
    username: str = None,
) -> int:
    """
    添加文档元数据记录。
    """
    if username is None:
        username = config.DEFAULT_USER

    conn = _get_connection()
    cursor = conn.execute(
        """
        INSERT INTO documents (filename, file_type, file_size, chunk_count, upload_time, file_path, username)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (filename, file_type, file_size, chunk_count, time.time(), file_path, username),
    )
    conn.commit()
    doc_id = cursor.lastrowid
    conn.close()
    return doc_id


def list_documents(username: str = None) -> List[dict]:
    """列出指定用户的文档。None 表示所有用户。"""
    conn = _get_connection()
    if username is None:
        rows = conn.execute(
            "SELECT * FROM documents ORDER BY upload_time DESC"
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM documents WHERE username = ? ORDER BY upload_time DESC",
            (username,)
        ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_document(doc_id: int) -> Optional[dict]:
    """根据 ID 获取单个文档信息"""
    conn = _get_connection()
    row = conn.execute(
        "SELECT * FROM documents WHERE id = ?", (doc_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def delete_document(doc_id: int) -> bool:
    """删除文档记录"""
    conn = _get_connection()
    cursor = conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    conn.commit()
    deleted = cursor.rowcount > 0
    conn.close()
    return deleted


def get_total_documents(username: str = None) -> int:
    """获取文档总数"""
    conn = _get_connection()
    if username:
        count = conn.execute(
            "SELECT COUNT(*) FROM documents WHERE username = ?", (username,)
        ).fetchone()[0]
    else:
        count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    conn.close()
    return count


def get_total_chunks(username: str = None) -> int:
    """获取文档总块数"""
    conn = _get_connection()
    if username:
        total = conn.execute(
            "SELECT SUM(chunk_count) FROM documents WHERE username = ?", (username,)
        ).fetchone()[0]
    else:
        total = conn.execute("SELECT SUM(chunk_count) FROM documents").fetchone()[0]
    conn.close()
    return total or 0


def get_all_users() -> List[str]:
    """获取所有在 SQLite 中有文档的用户名"""
    conn = _get_connection()
    rows = conn.execute(
        "SELECT DISTINCT username FROM documents ORDER BY username"
    ).fetchall()
    conn.close()
    return [row["username"] for row in rows]
