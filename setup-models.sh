#!/bin/bash
# ============================================
# 首次启动后拉取模型（只需执行一次）
# ============================================
# 使用:
#   docker compose up -d          # 先启动服务
#   bash setup-models.sh          # 再拉模型
# ============================================

set -e

echo "=========================================="
echo "  RAG 系统 — 模型初始化"
echo "=========================================="

echo ""
echo "[1/2] 拉取 deepseek-r1:7b (约 4.7GB)..."
docker exec -it rag-ollama ollama pull deepseek-r1:7b

echo ""
echo "[2/2] 拉取 llama3.2-vision:11b (可选, 约 7.8GB)..."
echo "  如果不需要图片识别功能, 按 Ctrl+C 跳过"
docker exec -it rag-ollama ollama pull llama3.2-vision:11b

echo ""
echo "=========================================="
echo "  完成! 浏览器打开 http://localhost:8501"
echo "=========================================="
