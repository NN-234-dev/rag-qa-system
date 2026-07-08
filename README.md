# RAG 智能问答系统

基于 RAG (Retrieval-Augmented Generation) 架构的本地知识库问答系统。上传文档 → 自动分块向量化 → 提问 → AI 根据文档内容回答并标注来源。

**完全本地运行**：使用 Ollama 部署的 `deepseek-r1:7b` + `BAAI/bge-small-zh-v1.5`，无需联网，数据不出本机。

## 架构总览

```mermaid
flowchart LR
    A[📄 文档上传] --> B[✂️ 文本分块]
    B --> C[🔢 向量化<br/>BGE-small-zh]
    C --> D[💾 Chroma 向量库<br/>私人库 / 公共库]
    
    E[❓ 用户提问] --> F[🔄 Query Rewriting<br/>口语→搜索查询]
    F --> G[🔍 混合检索<br/>语义 + BM25 + RRF]
    G --> H[🎯 LLM Rerank<br/>逐篇打分精筛]
    H --> I[📝 拼接 Prompt]
    I --> J[🤖 LLM 生成<br/>deepseek-r1:7b]
    J --> K[📎 带来源引用的回答]
    
    D --> G
```

### Agent 模式架构（LangGraph 多步推理）

开启 Agent 模式后，系统从"检索→回答"升级为自主决策的多步推理循环：

```mermaid
flowchart TD
    START(( )) --> P[🧠 Planner<br/>LLM 拆解问题→N个子问题]
    P --> R[🔍 Researcher<br/>每个子问题并行检索+摘要]
    R --> V[✅ Verifier<br/>结构性检查：覆盖率/矛盾/缺口]
    V --> S[✍️ Synthesizer<br/>综合所有发现生成完整回答]
    S --> C[🔎 Critic<br/>自我审查：回答完整吗？]
    C -->|NEED_MORE| I[➕ 补充搜索反馈]
    I --> P
    C -->|PASS| END(( ))

    style START fill:#4CAF50,color:#fff
    style END fill:#4CAF50,color:#fff
    style C fill:#FF9800,color:#fff
```

> **关键设计**：Critic 节点自主输出 PASS / NEED_MORE，触发条件边回到 Planner 重新搜索（最多 2 轮迭代）。这使系统具备了**自我反思能力**——不满足于第一次检索结果时，会自动补充搜索。

## 快速开始

### 🐳 Docker 部署（推荐，一键启动）

无需安装 Python 环境，只需 Docker Desktop。

```bash
# 1. 克隆仓库
git clone https://github.com/NN-234-dev/rag-qa-system.git
cd rag-qa-system

# 2. 一键启动（首次构建镜像约 3-5 分钟）
docker compose up -d

# 3. 拉取 AI 模型（仅首次需要，约 4.7GB）
bash setup-models.sh

# 4. 浏览器打开
# http://localhost:8501
```

**服务架构**：
```
docker compose up
├── ollama (端口 11434)    ← deepseek-r1:7b 模型推理
└── rag-app (端口 8501)    ← Streamlit UI + RAG 引擎
    ├── ChromaDB + SQLite  ← 持久化在 Docker Volume，重启不丢失
    └── BGE Embedding      ← 首次运行自动下载并缓存
```

> **给面试官**：如果不想装 Docker，可以直接看 `README.md` 中的架构图和下方设计决策了解系统设计，或要求候选人现场 Live Demo。

---

### 本地开发部署（从源码运行）

**环境要求**：Python 3.10+ / [Ollama](https://ollama.com) / 8GB+ 内存

```bash
# 1. 安装依赖
cd rag-qa-system
pip install -r requirements.txt

# 2. 拉取模型
ollama pull deepseek-r1:7b            # 文本生成 (~4.7GB)
ollama pull llama3.2-vision:11b        # 可选，图片识别 (~7.8GB)

# 3. 启动
streamlit run app.py
```

浏览器打开 `http://localhost:8501`，上传文档，开始提问。

## 技术栈

| 组件 | 选型 | 选型理由 |
|------|------|----------|
| **大模型** | deepseek-r1:7b (Ollama) | 本地部署零成本、中文强、推理链透明 |
| **嵌入模型** | BAAI/bge-small-zh-v1.5 | 中文优化、512维/速度快、离线可用 |
| **向量库** | Chroma | 轻量零配置、本地持久化、支持多 Collection |
| **关键词检索** | BM25 (rank_bm25 + jieba) | 无外部依赖、中文分词精准、与语义检索互补 |
| **融合算法** | RRF (Reciprocal Rank Fusion) | 无需归一化异构分数、k=60 平滑常数 |
| **UI 框架** | Streamlit | 聊天界面原生支持、Python 纯写、快速迭代 |
| **元数据存储** | SQLite | 零配置、SQL 标准、与 Chroma 互为备份 |
| **PDF 解析** | PyMuPDF (fitz) | 图文表格全支持、Python 最成熟的 PDF 库 |
| **分词** | jieba | 中文分词标准方案、轻量零配置 |

## 项目结构

```
rag-qa-system/
├── app.py                      # Streamlit 主界面（多用户版 + Agent 模式）
├── config.py                   # 全局配置常量
├── requirements.txt            # Python 依赖
├── Dockerfile                  # Docker 镜像构建
├── docker-compose.yml          # 一键编排 Ollama + RAG 服务
├── setup-models.sh             # 首次启动模型拉取脚本
├── .dockerignore
├── .gitignore
├── README.md                   # 本文件
├── interview_prep.md           # 面试高频问题+回答要点
├── mcp_config.json.example     # MCP 客户端配置示例
├── src/
│   ├── __init__.py
│   ├── document_loader.py      # 多格式文档解析 (PDF/Word/TXT/MD)
│   ├── text_splitter.py        # RecursiveCharacterTextSplitter 封装
│   ├── embeddings.py           # BGE 模型加载（单例模式）
│   ├── vector_store.py         # Chroma 增删查（多 Collection 支持）
│   ├── hybrid_retriever.py     # 语义 + BM25 + RRF 融合检索
│   ├── llm_client.py           # Ollama API 封装（文本+视觉）
│   ├── qa_chain.py             # RAG 问答完整流水线
│   ├── agent_graph.py           # LangGraph Agent 多步推理引擎
│   ├── query_rewriter.py       # LLM 查询改写（口语→搜索查询）
│   ├── llm_reranker.py         # Pointwise LLM 重排序
│   ├── image_handler.py        # PDF 图片/表格提取 + 视觉描述
│   ├── multi_user.py           # 多用户命名空间管理
│   ├── mcp_server.py            # MCP Server（AI Agent 原生协议）
│   ├── api_server.py            # REST API Server（通用 HTTP）
│   └── knowledge_db.py         # SQLite 文档元数据 CRUD
├── data/                       # 运行时数据（不上传 Git）
│   ├── uploads/                # 上传的原始文档
│   ├── chroma_db/              # Chroma 持久化目录
│   ├── extracted_images/       # PDF 中提取的图片
│   └── knowledge.db            # SQLite 数据库
├── test_e2e.py                 # 端到端测试
└── test_pipeline.py            # 检索效果对比测试
```

## 核心设计决策

### 1. 混合检索：语义 + BM25

**为什么不用纯语义检索？** 语义检索擅长找"意思相近"的内容（"退款"匹配"退费流程"），但对精确关键词（"Q3财报"、"v2.1版本"）可能漏掉。BM25 恰好互补。

**RRF 融合**：语义相似度分数和 BM25 分数不是一个量纲，直接加权需要归一化。RRF 只关心排名，公式简单有效：

$$RRF\ score = \sum_{i} \frac{1}{k + rank_i},\quad k=60$$

### 2. 查询改写 (Query Rewriting)

用户提问通常是口语化的（"这玩意儿怎么配置？"），直接用去检索效果很差。先让 LLM 把问题改写为更精确的搜索查询（"系统配置方法 设置步骤 参数说明"），再用改写后的查询去检索。**注意：生成答案时仍使用原始问题**，保证回答的针对性。

### 3. LLM Rerank（Pointwise）

检索（粗排）→ LLM 逐篇打分（精排）→ 取 top_k。经典的"粗排 + 精排"两阶段检索模式。

**为什么用 Pointwise 而不是 Listwise？** Pointwise 每篇独立打分，可以并行；Listwise 所有文档一起打分，受上下文窗口限制且更慢。打分标准 1-5 分，temperature=0.1 保证评分稳定性。

### 4. 多用户知识库隔离

每个用户独立 Chroma Collection（`kb_user_{username}`）+ 共享公共库（`kb_shared`）。搜索时同时查私人+公共，结果合并排序。类比 Python venv："环境隔离但共享底层引擎"。不搞密码认证——本地单机场景不需要，上生产再加 JWT 即可。

### 5. 完整流水线

```
用户问题 
  → Query Rewriting（口语→搜索查询）
  → 混合检索：语义(私人库+公共库) + BM25 
  → RRF 融合
  → 余弦距离阈值过滤
  → LLM Rerank（逐篇打分，取 top_k）
  → 拼接 Prompt（用原始问题）
  → LLM 流式生成
  → 返回答案 + 引用来源（标注🔒私人/🌐公共）
```

### 6. Agent 多步推理（LangGraph）

从线性管线升级为自主决策的 Agent 系统：

| 节点 | 职责 | LLM |
|------|------|-----|
| **Planner** | 将复杂问题拆解为 2-5 个子问题 | ✅ |
| **Researcher** | 每个子问题独立检索 + LLM 摘要 | ✅ |
| **Verifier** | 结构性检查（覆盖率/矛盾/缺口），零 LLM | ❌ |
| **Synthesizer** | 综合所有发现，生成完整回答 | ✅ |
| **Critic** | 自我审查：判断是否需要补充搜索 | ✅ |

**为什么是 5 节点而不是简单的 ReAct？** ReAct 适合工具调用类任务（"帮我发邮件"），但知识问答需要的是"拆解→检索→验证→综合"的深度推理链。Verifier 用零 LLM 的结构性检查（比 LLM 快 10 倍），Critic 用 LLM 判断是否需要补充搜索——各司其职。

### 7. 幻觉控制

- **严格 Prompt 约束**："只使用参考资料中的信息，不要编造"
- **相似度阈值过滤**：cosine 距离 > 0.5 的文档直接丢弃
- **明确拒答**：无相关文档时返回"未找到相关信息"
- **来源引用**：每个回答必须标注引用了哪些文档片段

## 功能特性

- [x] 多格式文档（PDF/Word/TXT/Markdown）
- [x] 混合检索（语义 + BM25 + RRF）
- [x] LLM 重排序（Pointwise 打分，两阶段精排）
- [x] LangGraph Agent 多步推理（5 节点 + 自我反思闭环）
- [x] 查询改写（口语→搜索查询）
- [x] 多用户知识库隔离（私人/公共）
- [x] 图片/表格识别（NotebookLM 风格）
- [x] 流式输出（逐 token 渲染 + Agent 阶段进度）
- [x] 来源引用（标注文档名 + 片段预览 + 知识库来源）
- [x] 参数可调（Chunk Size / Top-K / Temperature / Agent 开关）
- [x] Docker 一键部署（`docker compose up`）
- [x] MCP Server 集成（可被 Claude Code/Cursor 等 Agent 调用）
- [x] REST API 集成（任意语言、任意大模型 HTTP 调用）
- [x] 完全本地运行（无需联网）
- [x] 4 道幻觉防线（Prompt 约束 + 阈值过滤 + 来源引用 + Rerank 把关）

## 🔌 集成方式 — 让任何大模型应用拥有 RAG 能力

本系统最大的差异化：**不仅是一个带 UI 的问答应用，更是可被任何大模型应用调用的知识库基础设施。**

提供两种集成方式：

### 方式一：REST API（推荐，最通用）

**一行 curl 就能用，任何语言、任何模型都能集成。**

```bash
# 启动 API 服务
python src/api_server.py
# 访问 http://localhost:8000/docs 查看 Swagger 文档

# 任意大模型应用中调用
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"什么是RAG?","username":"default"}'
```

| 端点 | 方法 | 功能 |
|------|------|------|
| `/health` | GET | 健康检查 + 系统信息 |
| `/search` | POST | 语义搜索，返回文档片段 |
| `/ask` | POST | 完整 RAG 问答，返回答案+来源 |
| `/docs/{username}` | GET | 列出知识库所有文档 |
| `/context` | POST | 获取完整文档上下文 |

**Python 接入示例（3 行代码）**：

```python
import requests
resp = requests.post("http://localhost:8000/ask",
    json={"question": "怎么部署微服务?", "username": "alice"})
print(resp.json()["answer"])  # 基于 alice 知识库的 AI 回答
```

**LangChain Tool 封装**：

```python
from langchain.tools import tool
import requests

@tool
def query_knowledge_base(question: str) -> str:
    """搜索私有知识库获取答案"""
    r = requests.post("http://localhost:8000/ask",
        json={"question": question})
    return r.json()["answer"]
```

### 方式二：MCP Server（AI Agent 原生协议）

支持 [MCP 协议](https://modelcontextprotocol.io) 的客户端（Claude Code、Cursor、Continue.dev）可直接调用。

```json
{
  "mcpServers": {
    "rag-kb": {
      "command": "python",
      "args": ["src/mcp_server.py"],
      "cwd": "/path/to/rag-qa-system"
    }
  }
}
```

### 架构价值（面试要点）

> 大多数 RAG 项目的终点是一个 Streamlit/Gradio UI；我的项目把 RAG 变成了**可被集成的协议服务**。其他开发者不需要看懂我的代码，只需要一行 curl 或 3 行 Python 就能让他们的 LLM 应用拥有私有知识库检索能力。这是从"应用"到"基础设施"的思维跃迁。

## License

MIT
