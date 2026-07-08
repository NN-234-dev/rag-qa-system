# -*- coding: utf-8 -*-
"""混合检索 vs 纯语义检索 对比测试"""
import sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, '.')

from src.hybrid_retriever import get_hybrid_retriever, rebuild_hybrid_index
from src.vector_store import search_similar

rebuild_hybrid_index()
retriever = get_hybrid_retriever()

test_queries = [
    # 偏语义的查询
    ('语义型: "怎么退钱"', '如何退款'),
    # 偏精确关键词的查询
    ('关键词型: "HNSW"', 'HNSW'),
    # 混合型查询
    ('混合型: "RAG的核心思想"', 'RAG 核心思想'),
    # 精确匹配型
    ('精确型: "Reciprocal Rank Fusion"', 'Reciprocal Rank Fusion'),
]

print('=' * 70)
print('混合检索 (语义 + BM25 + RRF) vs 纯语义检索 对比')
print('=' * 70)

for label, query in test_queries:
    print(f'\n{"─" * 70}')
    print(f'查询: {label}')

    # 纯语义
    t0 = time.time()
    semantic_results = search_similar(query, top_k=5)
    semantic_time = time.time() - t0

    # 混合
    t0 = time.time()
    hybrid_results = retriever.search(query, top_k=5)
    hybrid_time = time.time() - t0

    print(f'\n  纯语义 (Chroma) 耗时 {semantic_time:.3f}s:')
    for i, doc in enumerate(semantic_results, 1):
        preview = doc.page_content[:80].replace('\n', ' | ')
        score = doc.metadata.get('similarity_score', 'N/A')
        print(f'    [{i}] score={score} | {preview}...')

    print(f'\n  混合检索 (语义+BM25+RRF) 耗时 {hybrid_time:.3f}s:')
    for i, doc in enumerate(hybrid_results, 1):
        preview = doc.page_content[:80].replace('\n', ' | ')
        rrf = doc.metadata.get('rrf_score', 'N/A')
        sem = doc.metadata.get('similarity_score', 'N/A')
        bm25 = doc.metadata.get('bm25_score', 'N/A')
        print(f'    [{i}] RRF={rrf} sem={sem} bm25={bm25} | {preview}...')

    # 去重: 检查混合检索是否覆盖了语义检索没覆盖的内容
    sem_set = {d.page_content[:200] for d in semantic_results[:3]}
    hyb_set = {d.page_content[:200] for d in hybrid_results[:3]}
    new_docs = hyb_set - sem_set
    if new_docs:
        print(f'\n  >>> 混合检索额外发现 {len(new_docs)} 条语义检索未命中的文档')

print('\n' + '=' * 70)
print('对比测试完成')
print('=' * 70)
