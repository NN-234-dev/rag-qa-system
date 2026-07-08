# ============================================
# 嵌入模型管理 — BGE-small-zh 单例加载
# ============================================

import os
from pathlib import Path
from langchain_community.embeddings import HuggingFaceEmbeddings
import config

# GFW 绕过：优先使用 ModelScope 本地下载的模型；否则走 HF 镜像
_MODEL_PATH = config.LOCAL_MODEL_PATH if Path(config.LOCAL_MODEL_PATH).exists() else config.EMBEDDING_MODEL_NAME
if not Path(config.LOCAL_MODEL_PATH).exists():
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


# 全局单例，避免重复加载模型（400MB 加载一次约 5-10 秒）
_embedding_model = None


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    获取嵌入模型实例（单例模式）。

    面试点：BGE-small-zh-v1.5
    - 维度: 512（small 版本），Base 版本是 768
    - 为什么选 small？演示场景速度优先，512 维足够区分语义
    - 为什么不用 OpenAI Embedding？免费离线、中文优化更好、不依赖外部 API
    """
    global _embedding_model
    if _embedding_model is None:
        print(f"[Embedding] 正在加载模型 {_MODEL_PATH} ...")
        _embedding_model = HuggingFaceEmbeddings(
            model_name=_MODEL_PATH,
            model_kwargs={"device": config.EMBEDDING_DEVICE},
            encode_kwargs={
                "normalize_embeddings": True,  # 归一化，余弦相似度更准确
            },
        )
        print("[Embedding] 模型加载完成")
    return _embedding_model
