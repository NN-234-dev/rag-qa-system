# ============================================
# 测试脚本 — 命令行验证完整 RAG 链路
# 运行方式: python test_pipeline.py
# 前提: 在项目根目录创建 .env 文件配置 DEEPSEEK_API_KEY
# ============================================

import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
sys.path.insert(0, str(Path(__file__).parent))

from config import CHUNK_SIZE, CHUNK_OVERLAP, TOP_K
from src.document_loader import load_document
from src.text_splitter import split_documents, get_chunk_stats
from src.vector_store import add_documents, search_similar, get_collection_info, reset_collection
from src.qa_chain import ask


def test_with_text():
    """使用测试文本验证完整流程"""
    print("=" * 60)
    print("RAG Pipeline Test")
    print("=" * 60)

    # 1. 创建测试文档
    test_content = """
    RAG（Retrieval-Augmented Generation，检索增强生成）是一种结合信息检索与文本生成的AI技术架构。

    在传统的LLM应用中，模型仅依赖于训练时学到的知识，存在知识截止日期和幻觉问题。
    RAG通过在生成答案前先从外部知识库中检索相关信息，将检索到的文档作为上下文提供给LLM，
    从而显著提高答案的准确性和可验证性。

    RAG系统通常包含以下核心组件：
    1. 文档加载器（Document Loader）：负责解析PDF、Word、TXT等不同格式的文档
    2. 文本分块器（Text Splitter）：将长文档切分为适合检索的短片段
    3. 嵌入模型（Embedding Model）：将文本转换为向量表示
    4. 向量数据库（Vector Database）：存储和检索向量化的文档
    5. 大语言模型（LLM）：根据检索到的上下文生成最终答案

    文本分块是RAG系统中至关重要的一步。如果分块太大，检索精度会下降；
    如果分块太小，可能丢失上下文信息。通常推荐每个块包含300-500个字符，
    并保留10%-20%的重叠区域，以确保不会在边界处丢失关键信息。

    混合检索是提升RAG效果的重要手段。纯语义检索擅长理解同义词和语义相似的表达，
    但在精确关键词匹配方面表现不佳。结合BM25等关键词检索算法，
    可以通过RRF（Reciprocal Rank Fusion）将两种检索结果融合，兼顾语义理解和精确匹配。

    在实际生产环境中，RAG系统还需要考虑：
    - 文档预处理和清洗质量
    - 嵌入模型的选择（中文场景推荐BGE系列）
    - 检索效率优化（索引策略、缓存机制）
    - 答案质量评估和监控
    """

    # 写入临时文件
    test_file = Path(__file__).parent / "data" / "uploads" / "test_rag_intro.txt"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text(test_content, encoding="utf-8")
    print(f"\n[Test] Created test document: {test_file.name}")

    # 2. 加载文档
    print("\n[1/5] Loading document...")
    docs = load_document(str(test_file))
    print(f"  [OK] Loaded {len(docs)} page(s)/segment(s)")

    # 3. 分块
    print("\n[2/5] Splitting text...")
    chunks = split_documents(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    stats = get_chunk_stats(chunks)
    print(f"  [OK] {stats['count']} chunks, avg length {stats['avg_length']:.0f} chars")

    # 4. 向量化 + 存储
    print("\n[3/5] Embedding & storing...")
    print("  [..] First run will download BGE-small-zh (~400MB), please wait...")

    # 清空旧数据
    reset_collection()

    n = add_documents(chunks, doc_id="test_doc_1")
    info = get_collection_info()
    print(f"  [OK] Stored {n} vectors, total collection: {info['total_chunks']}")

    # 5. 检索
    print("\n[4/5] Semantic search test...")
    test_questions = [
        "什么是RAG？",
        "文本分块有什么注意事项？",
        "为什么需要混合检索？",
    ]

    for q in test_questions:
        results = search_similar(q, top_k=2)
        print(f"\n  Q: {q}")
        for i, doc in enumerate(results, 1):
            preview = doc.page_content[:80].replace("\n", " ")
            score = doc.metadata.get("similarity_score", "N/A")
            print(f"    [{i}] score={score} | {preview}...")

    # 6. LLM 问答
    print("\n[5/5] RAG Q&A test...")
    api_key_configured = True
    try:
        from config import DEEPSEEK_API_KEY
        if not DEEPSEEK_API_KEY:
            api_key_configured = False
    except Exception:
        api_key_configured = False

    if not api_key_configured:
        print("  [SKIP] DeepSeek API Key not configured, skipping LLM test")
        print("  [TIP] Create .env file with: DEEPSEEK_API_KEY=sk-your-key")
    else:
        for q in test_questions[:2]:
            result = ask(q, top_k=TOP_K)
            print(f"\n  Q: {q}")
            print(f"  A: {result['answer'][:200]}...")
            print(f"  [Stats] Retrieved {result['docs_retrieved']}, relevant {result['docs_relevant']}")
            print(f"  [Time] Retrieval {result['retrieval_time']:.2f}s | Generation {result['generation_time']:.2f}s")
            if result["sources"]:
                print(f"  [Sources] {[s['source'] for s in result['sources']]}")

    print("\n" + "=" * 60)
    print("Pipeline test complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_with_text()
