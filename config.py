# ============================================
# RAG 智能问答系统 — 全局配置
# ============================================

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

# ============================================
# 项目路径
# ============================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma_db"
DB_PATH = DATA_DIR / "knowledge.db"
IMAGE_DIR = DATA_DIR / "extracted_images"  # 从文档中提取的图片

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================
# Ollama 本地模型配置
# ============================================
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1:7b")           # 文本生成（回答、改写、重排）
OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "llama3.2-vision:11b")  # 图片识别/描述

# ============================================
# 嵌入模型配置
# ============================================
EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
LOCAL_MODEL_PATH = str(DATA_DIR / "models" / "models" / "BAAI--bge-small-zh-v1.5" / "snapshots" / "master")
EMBEDDING_DIMENSION = 512
EMBEDDING_DEVICE = "cpu"

# ============================================
# 文档分块配置
# ============================================
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# ============================================
# 检索配置
# ============================================
TOP_K = 5
SIMILARITY_THRESHOLD = 0.5

# ============================================
# 查询改写配置
# ============================================
ENABLE_QUERY_REWRITING = True
REWRITE_TEMPERATURE = 0.2  # 改写用低温，保证稳定

# ============================================
# LLM Rerank 配置
# ============================================
ENABLE_LLM_RERANK = True
RERANK_CANDIDATE_MULTIPLIER = 2  # 检索 top_k * N 条，然后 LLM 打分筛选

# ============================================
# 多用户知识库配置
# ============================================
DEFAULT_USER = "default"
ENABLE_SHARED_KB = True  # 是否同时检索公共知识库

# ============================================
# LLM 生成配置
# ============================================
TEMPERATURE = 0.4
MAX_TOKENS = 4096  # deepseek-r1 需要留足思考 token 的空间

# ============================================
# UI 配置
# ============================================
PAGE_TITLE = "RAG 智能问答系统"
PAGE_ICON = "📚"
MAX_FILE_SIZE_MB = 200
SUPPORTED_FILE_TYPES = ["pdf", "txt", "docx", "md"]

# ============================================
# Agent 模式配置（LangGraph 多步推理）
# ============================================
ENABLE_AGENT_MODE = True          # 是否启用 Agent 模式（Streamlit 开关）
MAX_AGENT_ITERATIONS = 2          # Agent 自我反思的最大循环次数
AGENT_MAX_SUB_QUESTIONS = 5       # 单次问题拆解的最大子问题数
AGENT_TEMPERATURE = 0.3           # Agent 各节点的生成温度（略低于默认，保证稳定性）
