# ============================================
# RAG 智能问答系统 — Docker 镜像
# ============================================

FROM python:3.10-slim

WORKDIR /app

# 系统依赖（PyMuPDF 需要 libgomp1）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Python 依赖（用清华镜像加速）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 项目代码
COPY . .

# HuggingFace 国内镜像
ENV HF_ENDPOINT=https://hf-mirror.com

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
