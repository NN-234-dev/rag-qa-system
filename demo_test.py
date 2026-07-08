# -*- coding: utf-8 -*-
"""临时测试脚本：入库大模型开发文档 + 5个问答验证"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, '.')

from src.document_loader import load_document
from src.text_splitter import split_documents, get_chunk_stats
from src.vector_store import add_documents, reset_collection, get_collection_info
from src.qa_chain import ask

# 清空旧数据
reset_collection()

# 入库
print('[1/3] Loading document...')
docs = load_document('data/uploads/大模型应用开发实战指南.md')
print(f'  {len(docs)} paragraphs')

print('[2/3] Splitting...')
chunks = split_documents(docs)
stats = get_chunk_stats(chunks)
print(f'  {stats["count"]} chunks, avg {stats["avg_length"]:.0f} chars')

print('[3/3] Embedding & storing...')
n = add_documents(chunks, doc_id='guide_doc_1')
info = get_collection_info()
print(f'  {n} vectors stored, total: {info["total_chunks"]}')

# 5 个测试问题
questions = [
    '什么是RAG？它解决了什么问题？',
    '文本分块需要注意什么？',
    'BGE模型和OpenAI的嵌入模型比有什么优劣？',
    'Chroma向量数据库有什么特点？',
    'RAG和微调有什么区别？',
]

print('\n' + '=' * 60)
print('RAG Q&A TEST RESULTS')
print('=' * 60)

for q in questions:
    print(f'\n{"─" * 60}')
    print(f'Q: {q}')
    result = ask(q, top_k=3)
    answer = result['answer']
    print(f'A: {answer[:400]}{"..." if len(answer) > 400 else ""}')
    print(f'  Stats: retrieved={result["docs_retrieved"]}, relevant={result["docs_relevant"]}, search={result["retrieval_time"]:.2f}s, llm={result["generation_time"]:.2f}s')
    print(f'  Sources: {[s["source"] for s in result["sources"]]}')

print('\n' + '=' * 60)
print('ALL TESTS PASSED')
print('=' * 60)
