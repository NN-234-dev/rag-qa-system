# ============================================
# 文档加载器 — 支持 PDF / TXT / Word / Markdown
# ============================================

from pathlib import Path
from typing import List
from langchain_core.documents import Document


def load_document(file_path: str) -> List[Document]:
    """
    根据文件扩展名自动选择加载器，解析文档为 LangChain Document 列表。

    Args:
        file_path: 文档文件的绝对路径

    Returns:
        List[Document]: 解析后的文档对象列表

    Raises:
        ValueError: 不支持的文件格式
        FileNotFoundError: 文件不存在
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    ext = file_path.suffix.lower()

    try:
        if ext == ".pdf":
            return _load_pdf(file_path)
        elif ext == ".txt":
            return _load_txt(file_path)
        elif ext in [".docx", ".doc"]:
            return _load_docx(file_path)
        elif ext == ".md":
            return _load_markdown(file_path)
        else:
            raise ValueError(
                f"不支持的文件格式: {ext}。支持的格式: PDF, TXT, DOCX, MD"
            )
    except Exception as e:
        raise RuntimeError(f"文档加载失败 [{file_path.name}]: {str(e)}")


def _load_pdf(file_path: Path) -> List[Document]:
    """加载 PDF 文档"""
    from langchain_community.document_loaders import PyPDFLoader

    loader = PyPDFLoader(str(file_path))
    docs = loader.load()

    # 为每个 page 标注来源文件名
    for doc in docs:
        doc.metadata["source"] = file_path.name
        doc.metadata["file_type"] = "pdf"

    return docs


def _load_txt(file_path: Path) -> List[Document]:
    """加载纯文本文档"""
    from langchain_community.document_loaders import TextLoader

    loader = TextLoader(str(file_path), encoding="utf-8")
    docs = loader.load()

    for doc in docs:
        doc.metadata["source"] = file_path.name
        doc.metadata["file_type"] = "txt"

    return docs


def _load_docx(file_path: Path) -> List[Document]:
    """加载 Word 文档"""
    from langchain_community.document_loaders import Docx2txtLoader

    loader = Docx2txtLoader(str(file_path))
    docs = loader.load()

    for doc in docs:
        doc.metadata["source"] = file_path.name
        doc.metadata["file_type"] = "docx"

    return docs


def _load_markdown(file_path: Path) -> List[Document]:
    """加载 Markdown 文档（用 TextLoader 避免 unstructured 的 spaCy 依赖）"""
    from langchain_community.document_loaders import TextLoader

    loader = TextLoader(str(file_path), encoding="utf-8")
    docs = loader.load()

    for doc in docs:
        doc.metadata["source"] = file_path.name
        doc.metadata["file_type"] = "md"

    return docs
