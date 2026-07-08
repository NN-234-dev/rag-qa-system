# ============================================
# 文档图片/表格处理器 — NotebookLM 风格
# ============================================
# 从 PDF 中提取图片和表格，用 Ollama 视觉模型描述内容
# 把图片描述融合进文档块，让 RAG 系统能"看懂"图片和表格

import os
from pathlib import Path
from typing import List, Tuple
import config
from src.llm_client import describe_image


def extract_images_from_pdf(pdf_path: str) -> List[str]:
    """
    从 PDF 中提取所有嵌入的图片，保存到 data/extracted_images/。

    使用 PyMuPDF (fitz) —— PDF 图片提取的工业标准库。

    Args:
        pdf_path: PDF 文件路径

    Returns:
        提取出的图片文件路径列表
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError(
            "需要安装 PyMuPDF 来提取 PDF 中的图片。\n"
            "运行: pip install pymupdf"
        )

    pdf_path = Path(pdf_path)
    doc_name = pdf_path.stem
    output_dir = config.IMAGE_DIR / doc_name
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = []
    doc = fitz.open(str(pdf_path))

    for page_num, page in enumerate(doc):
        # 获取页面中的所有图片
        image_list = page.get_images(full=True)

        for img_idx, img_info in enumerate(image_list):
            xref = img_info[0]  # 图片在 PDF 内部的引用 ID

            try:
                # 提取原始图片数据
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                ext = base_image["ext"]  # jpeg, png, bmp...

                # 保存到磁盘
                img_filename = f"page{page_num + 1}_img{img_idx + 1}.{ext}"
                img_path = output_dir / img_filename
                img_path.write_bytes(image_bytes)
                image_paths.append(str(img_path))

            except Exception as e:
                print(f"  [警告] 提取图片失败 (page {page_num + 1}, img {img_idx + 1}): {e}")

    doc.close()
    return image_paths


def extract_tables_from_pdf(pdf_path: str) -> List[str]:
    """
    从 PDF 中提取表格内容（文字形式）。

    PyMuPDF 1.23+ 内置了表格检测算法，不需要额外依赖。

    Args:
        pdf_path: PDF 文件路径

    Returns:
        表格内容列表，每个元素是格式化的表格文字
    """
    try:
        import fitz
    except ImportError:
        raise ImportError("需要安装 PyMuPDF: pip install pymupdf")

    pdf_path = Path(pdf_path)
    doc = fitz.open(str(pdf_path))

    tables = []

    for page_num, page in enumerate(doc):
        # find_tables() 返回页面中检测到的所有表格
        found_tables = page.find_tables()

        if found_tables and found_tables.tables:
            for tab_idx, table in enumerate(found_tables.tables):
                # table.extract() 返回二维列表 [[cell, cell], [cell, cell]]
                rows = table.extract()
                if not rows:
                    continue

                # 格式化为可读文本
                header = rows[0] if rows else []
                body = rows[1:] if len(rows) > 1 else []

                lines = []
                lines.append(f"[表格 页码{page_num + 1}-{tab_idx + 1}]")
                if header:
                    lines.append(" | ".join(str(c) if c else "" for c in header))
                    lines.append("-" * 40)
                for row in body:
                    lines.append(" | ".join(str(c) if c else "" for c in row))

                tables.append("\n".join(lines))

    doc.close()
    return tables


def describe_images_from_pdf(pdf_path: str) -> dict:
    """
    提取 PDF 中的图片并用视觉模型逐一描述。

    这是 NotebookLM 风格的核心功能：
    用户上传含图表的 PDF → AI 能"看到"图表内容 → 问答时能引用

    Args:
        pdf_path: PDF 文件路径

    Returns:
        {
            "image_descriptions": {图片路径: 描述文字},
            "table_contents": [表格文字列表],
            "summary": "该文档包含的所有图片和表格的汇总描述"
        }
    """
    result = {
        "image_descriptions": {},
        "table_contents": [],
        "summary": "",
    }

    print(f"\n[ImageHandler] 处理文档: {pdf_path}")

    # Step 1: 提取表格（不需要视觉模型，PyMuPDF 直接读）
    print("  [1/3] 提取表格...")
    try:
        tables = extract_tables_from_pdf(pdf_path)
        result["table_contents"] = tables
        print(f"  发现 {len(tables)} 个表格")
    except Exception as e:
        print(f"  表格提取失败: {e}")

    # Step 2: 提取图片
    print("  [2/3] 提取图片...")
    try:
        image_paths = extract_images_from_pdf(pdf_path)
        print(f"  发现 {len(image_paths)} 张图片")
    except Exception as e:
        print(f"  图片提取失败: {e}")
        image_paths = []

    # Step 3: 用视觉模型描述每张图片
    if image_paths:
        print(f"  [3/3] 识别 {len(image_paths)} 张图片...")
        for i, img_path in enumerate(image_paths, 1):
            print(f"  ({i}/{len(image_paths)}) {Path(img_path).name}...")
            try:
                desc = describe_image(img_path)
                result["image_descriptions"][img_path] = desc
            except Exception as e:
                result["image_descriptions"][img_path] = f"[识别失败] {e}"
    else:
        print("  [3/3] 无图片，跳过")

    # 生成汇总
    summary_parts = []
    if result["table_contents"]:
        summary_parts.append(f"本文档包含 {len(result['table_contents'])} 个表格。")
        for t in result["table_contents"]:
            summary_parts.append(t)

    if result["image_descriptions"]:
        summary_parts.append(f"\n本文档包含 {len(result['image_descriptions'])} 张图片。")
        for img_path, desc in result["image_descriptions"].items():
            img_name = Path(img_path).name
            summary_parts.append(f"\n[图片: {img_name}]\n{desc}")

    result["summary"] = "\n".join(summary_parts)

    return result


def process_document_with_vision(file_path: str) -> str:
    """
    一站式处理：提取文档中的图片和表格，返回可被 RAG 索引的文本描述。

    这个返回的文本会附加到原始文档内容后面，一起被分块和向量化。

    Args:
        file_path: 文档路径（目前只支持 PDF）

    Returns:
        图片和表格的文字描述，可直接拼接到文档内容中
    """
    ext = Path(file_path).suffix.lower()

    if ext != ".pdf":
        # 非 PDF 文件暂不做图片提取（Word 的图片提取需要额外处理）
        return ""

    try:
        result = describe_images_from_pdf(file_path)
        return result["summary"] if result["summary"] else ""
    except ImportError as e:
        print(f"  [跳过] 图片处理: {e}")
        return ""
    except Exception as e:
        print(f"  [警告] 图片处理异常: {e}")
        return ""
