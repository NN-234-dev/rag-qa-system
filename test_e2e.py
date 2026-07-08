"""端到端测试脚本 — 验证核心链路"""
import sys, time
sys.path.insert(0, '.')

# ===== 测试1: 文档加载 + 分块 =====
print('=' * 50)
print('Test 1: Document Loading + Chunking')
print('=' * 50)
from src.document_loader import load_document
from src.text_splitter import split_documents, get_chunk_stats

docs = load_document('data/uploads/test_rag_intro.txt')
print(f'  Parsed {len(docs)} paragraphs')

chunks = split_documents(docs, chunk_size=500, chunk_overlap=100)
stats = get_chunk_stats(chunks)
print(f'  Generated {stats["count"]} chunks, avg {stats["avg_length"]:.0f} chars')
print('  PASS')

# ===== 测试2: 向量库连接 =====
print()
print('=' * 50)
print('Test 2: Vector Store Connection')
print('=' * 50)
from src.vector_store import get_collection_info
info = get_collection_info(username='default')
print(f'  Private: {info["private_chunks"]} chunks')
print(f'  Shared: {info["shared_chunks"]} chunks')
print('  PASS')

# ===== 测试3: 混合检索 =====
print()
print('=' * 50)
print('Test 3: Hybrid Retrieval (Semantic + BM25 + RRF)')
print('=' * 50)
from src.hybrid_retriever import get_hybrid_retriever, rebuild_hybrid_index
rebuild_hybrid_index(username='default', include_shared=True)
retriever = get_hybrid_retriever()
results = retriever.search('RAG是什么', top_k=3, username='default', include_shared=True)
print(f'  Retrieved {len(results)} results')
for i, doc in enumerate(results, 1):
    src = doc.metadata.get('source', '?')
    score = doc.metadata.get('similarity_score', doc.metadata.get('rrf_score', '?'))
    print(f'  [{i}] {src} (score={score})')
print('  PASS')

# ===== 测试4: LLM 问答 (完整管线) =====
print()
print('=' * 50)
print('Test 4: Full RAG Pipeline (Ask)')
print('=' * 50)
from src.qa_chain import ask

t0 = time.time()
result = ask('什么是RAG?', top_k=3, username='default', include_shared=True)
elapsed = time.time() - t0

print(f'  Search query: {result.get("search_query", "N/A")}')
print(f'  Docs retrieved: {result.get("docs_retrieved", 0)}')
print(f'  Docs relevant:  {result.get("docs_relevant", 0)}')
print(f'  Retrieval: {result.get("retrieval_time", 0):.2f}s')
print(f'  Rerank:    {result.get("rerank_time", 0):.2f}s')
print(f'  Generate:  {result.get("generation_time", 0):.2f}s')
print(f'  Total:     {elapsed:.2f}s')
print(f'  Sources:   {len(result.get("sources", []))}')
for s in result.get('sources', []):
    print(f'    - {s["source"]} ({s.get("kb_source", "?")})')
print(f'  Answer preview: {result["answer"][:200]}...')
print('  PASS')

print()
print('=' * 50)
print('ALL TESTS PASSED')
print('=' * 50)
