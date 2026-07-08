"""RAG 智能问答系统 — Streamlit 主界面（多用户版）"""

import streamlit as st
import time
from pathlib import Path
import sys

# 确保项目根目录可导入
sys.path.insert(0, str(Path(__file__).parent))

import config
from src.document_loader import load_document
from src.text_splitter import split_documents, get_chunk_stats
from src.vector_store import (
    add_documents,
    delete_by_doc_id,
    get_collection_info,
    reset_collection,
    get_all_user_collections,
)
from src.hybrid_retriever import rebuild_hybrid_index
from src.knowledge_db import (
    add_document as db_add_document,
    list_documents,
    delete_document as db_delete_document,
    get_total_documents,
    get_total_chunks,
    get_all_users,
)
from src.qa_chain import ask_stream
from src.agent_graph import agent_ask_stream
from src.image_handler import process_document_with_vision
from src.multi_user import get_user_collection_name, get_shared_collection_name

# 页面设置
st.set_page_config(
    page_title=config.PAGE_TITLE,
    page_icon=config.PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# 会话状态初始化
if "current_user" not in st.session_state:
    st.session_state["current_user"] = config.DEFAULT_USER
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 侧边栏
with st.sidebar:
    st.title("📚 知识库管理")

    # ── Ollama 连接检查 ──
    try:
        from openai import OpenAI
        test_client = OpenAI(base_url=config.OLLAMA_BASE_URL, api_key="ollama")
        test_client.models.list()
    except Exception:
        st.error("⚠️ 无法连接 Ollama，请确认：\n1. Ollama 已安装并启动\n2. 模型已下载: ollama pull deepseek-r1:7b")
        st.stop()

    # ── 用户管理 ──
    st.subheader("👤 当前用户")

    # 获取所有已知用户（SQLite + Chroma 合并）
    db_users = get_all_users()
    chroma_users = get_all_user_collections()
    all_users = sorted(set(db_users + chroma_users + [config.DEFAULT_USER]))

    current_user = st.session_state["current_user"]

    # 用户选择下拉框 + 新建输入框
    col_sel, col_new = st.columns([3, 1])
    with col_sel:
        selected_user = st.selectbox(
            "切换用户",
            options=all_users,
            index=all_users.index(current_user) if current_user in all_users else 0,
            label_visibility="collapsed",
        )
        if selected_user != current_user:
            st.session_state["current_user"] = selected_user
            # 切换用户时清空聊天历史
            st.session_state["messages"] = []
            st.rerun()

    with col_new:
        new_user = st.text_input(
            "新建",
            placeholder="用户名",
            label_visibility="collapsed",
        )
        if new_user and new_user.strip():
            clean_name = new_user.strip().lower().replace(" ", "_")
            if clean_name not in all_users:
                st.session_state["current_user"] = clean_name
                st.session_state["messages"] = []
                st.rerun()

    username = st.session_state["current_user"]
    st.caption(f"📂 私人库: `{get_user_collection_name(username)}`")

    # 公共库开关
    include_shared = st.checkbox(
        "🔗 同时检索公共知识库",
        value=config.ENABLE_SHARED_KB,
        help="启用后在提问时会同时搜索公共库中的文档",
    )
    config.ENABLE_SHARED_KB = include_shared

    st.divider()

    # ── 文档上传 ──
    st.subheader("📤 上传文档")
    uploaded_files = st.file_uploader(
        f"上传到 {username} 的私人库",
        type=config.SUPPORTED_FILE_TYPES,
        accept_multiple_files=True,
        help=f"单个文件最大 {config.MAX_FILE_SIZE_MB}MB",
    )

    if uploaded_files:
        if st.button("🚀 处理并入库", use_container_width=True, type="primary"):
            for uploaded_file in uploaded_files:
                file_size_mb = uploaded_file.size / (1024 * 1024)
                if file_size_mb > config.MAX_FILE_SIZE_MB:
                    st.warning(f"文件 {uploaded_file.name} 超过 {config.MAX_FILE_SIZE_MB}MB 限制，已跳过")
                    continue

                with st.status(f"处理中: {uploaded_file.name}...", expanded=True) as status:
                    try:
                        # 保存文件
                        save_path = config.UPLOAD_DIR / uploaded_file.name
                        save_path.write_bytes(uploaded_file.getvalue())

                        # 加载
                        st.write("📖 解析文档...")
                        docs = load_document(str(save_path))
                        st.write(f"   解析出 {len(docs)} 个段落")

                        # 图片/表格识别（NotebookLM 风格）
                        st.write("🖼️ 识别图片和表格...")
                        vision_text = process_document_with_vision(str(save_path))
                        if vision_text:
                            from langchain_core.documents import Document
                            docs.append(
                                Document(
                                    page_content=vision_text,
                                    metadata={"source": f"{uploaded_file.name} (图片/表格)",
                                               "file_type": "vision"}
                                )
                            )
                            st.write(f"   识别完成，附加 {len(vision_text)} 字符的视觉描述")

                        # 分块
                        st.write("✂️ 文本分块...")
                        chunks = split_documents(
                            docs,
                            chunk_size=st.session_state.get("chunk_size", config.CHUNK_SIZE),
                            chunk_overlap=st.session_state.get("chunk_overlap", config.CHUNK_OVERLAP),
                        )
                        stats = get_chunk_stats(chunks)
                        st.write(f"   生成 {stats['count']} 个文本块 (平均 {stats['avg_length']:.0f} 字符)")

                        # 入库（写入当前用户的私人库）
                        st.write("📊 向量化存储...")
                        doc_id = f"doc_{int(time.time())}_{uploaded_file.name}"
                        add_documents(chunks, doc_id=doc_id, username=username)

                        # SQLite 记录（带用户名）
                        db_add_document(
                            filename=uploaded_file.name,
                            file_type=Path(uploaded_file.name).suffix.lower().lstrip("."),
                            file_size=uploaded_file.size,
                            chunk_count=stats["count"],
                            file_path=str(save_path),
                            username=username,
                        )

                        status.update(label=f"✅ {uploaded_file.name} 处理完成", state="complete")

                    except Exception as e:
                        status.update(label=f"❌ {uploaded_file.name} 处理失败", state="error")
                        st.error(str(e))

            # 重建 BM25 索引
            rebuild_hybrid_index(username=username, include_shared=include_shared)
            st.rerun()

    # ── 已入库文档列表（仅显示当前用户） ──
    st.subheader(f"📋 {username} 的文档")
    docs_in_db = list_documents(username=username)

    if docs_in_db:
        for doc in docs_in_db:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.text(f"📄 {doc['filename']} ({doc['chunk_count']}块)")
            with col2:
                if st.button("🗑️", key=f"del_{doc['id']}", help=f"删除 {doc['filename']}"):
                    delete_by_doc_id(f"doc_{doc['upload_time']}_{doc['filename']}", username=username)
                    db_delete_document(doc["id"])
                    try:
                        Path(doc["file_path"]).unlink(missing_ok=True)
                    except Exception:
                        pass
                    rebuild_hybrid_index(username=username, include_shared=include_shared)
                    st.rerun()

        st.caption(f"私人库: {len(docs_in_db)} 个文档, {get_total_chunks(username=username)} 个文本块")
    else:
        st.caption("暂无文档，请上传")

    st.divider()

    # ── 参数配置 ──
    st.subheader("⚙️ 检索参数")

    chunk_size = st.slider(
        "分块大小 (Chunk Size)",
        min_value=200, max_value=1500, value=config.CHUNK_SIZE, step=50,
        help="每个文本块的最大字符数。越小检索越精确但可能丢失上下文",
    )
    st.session_state["chunk_size"] = chunk_size

    chunk_overlap = st.slider(
        "重叠大小 (Overlap)",
        min_value=0, max_value=300, value=config.CHUNK_OVERLAP, step=20,
        help="相邻块之间的重叠字符数，防止关键信息被切断",
    )
    st.session_state["chunk_overlap"] = chunk_overlap

    top_k = st.slider(
        "检索数量 (Top-K)",
        min_value=1, max_value=10, value=config.TOP_K, step=1,
        help="每次检索返回的文档片段数量",
    )

    temperature = st.slider(
        "生成温度 (Temperature)",
        min_value=0.0, max_value=1.0, value=config.TEMPERATURE, step=0.05,
        help="越低回答越保守（事实性高），越高越有创造性",
    )

    enable_rewrite = st.checkbox(
        "启用查询改写 (Query Rewriting)",
        value=config.ENABLE_QUERY_REWRITING,
        help="用 LLM 将口语化问题改写为更精确的搜索查询，提高检索命中率",
    )
    config.ENABLE_QUERY_REWRITING = enable_rewrite

    enable_rerank = st.checkbox(
        "启用 LLM 重排序 (LLM Rerank)",
        value=config.ENABLE_LLM_RERANK,
        help="检索后让 LLM 对候选文档逐一打分，筛选最相关的文档（更准但更慢）",
    )
    config.ENABLE_LLM_RERANK = enable_rerank

    enable_agent = st.checkbox(
        "🧠 Agent 模式 (多步推理)",
        value=config.ENABLE_AGENT_MODE,
        help="启用后 Agent 会自动拆解复杂问题 → 并行检索 → 交叉验证 → 综合回答 → 自我审查。"
             "适合需要多角度分析的复杂问题。简单问题建议关闭以加快响应速度。",
    )
    config.ENABLE_AGENT_MODE = enable_agent

    st.divider()

    # ── 系统信息 ──
    st.subheader("ℹ️ 系统信息")
    try:
        info = get_collection_info(username=username)
        st.caption(f"私人库: {info['private_chunks']} 块 | 公共库: {info['shared_chunks']} 块")
    except Exception:
        st.caption("向量库: 未初始化")
    st.caption(f"嵌入模型: {config.EMBEDDING_MODEL_NAME}")
    st.caption(f"LLM: {config.OLLAMA_MODEL}")
    st.caption(f"视觉模型: {config.OLLAMA_VISION_MODEL}")

    # 重新索引按钮
    if st.button("🔄 清空当前用户知识库", use_container_width=True):
        reset_collection(username=username)
        for doc in docs_in_db:
            db_delete_document(doc["id"])
            try:
                Path(doc["file_path"]).unlink(missing_ok=True)
            except Exception:
                pass
        rebuild_hybrid_index(username=username, include_shared=include_shared)
        st.rerun()

# 主区域 — 问答
st.title("📚 RAG 智能问答系统")
st.caption(f"当前用户: **{username}** | 上传文档到左侧知识库，然后在下方提问。AI 会基于文档内容回答。")

# ── 显示历史消息 ──
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=msg.get("avatar")):
        st.markdown(msg["content"])

        # 显示引用来源（标注 KB 来源）
        if msg.get("sources"):
            with st.expander("📎 引用来源"):
                for s in msg["sources"]:
                    kb_tag = "🔒私人" if s.get("kb_source") == "private" else "🌐公共"
                    st.caption(f"📄 {s['source']} {kb_tag}")
                    st.text(s["content_preview"])

        # 显示改写后的查询
        if msg.get("search_query") and msg["search_query"] != msg.get("content", ""):
            st.caption(f"🔍 改写查询: _{msg['search_query']}_")

        # 显示耗时
        if msg.get("retrieval_time"):
            parts = [f"🔍 检索 {msg.get('retrieval_time', 0):.2f}s"]
            if msg.get("rerank_time", 0) > 0:
                parts.append(f"| 🎯 重排 {msg.get('rerank_time', 0):.2f}s "
                             f"({msg.get('docs_before_rerank', 0)}→{msg.get('docs_relevant', 0)}条)")
            parts.append(f"| ✍️ 生成 {msg.get('generation_time', 0):.2f}s")
            parts.append(f"| 📊 检索到 {msg.get('docs_retrieved', 0)} 条 "
                         f"({msg.get('docs_relevant', 0)} 条相关)")
            st.caption(" ".join(parts))

# ── 输入区域 ──
if prompt := st.chat_input(f"请输入您的问题（基于 {username} 的知识库）..."):
    # 检查是否有文档（当前用户 + 公共库）
    if get_total_documents(username=username) == 0 and not include_shared:
        st.warning(f"⚠️ {username} 的私人库中没有文档，请先上传或开启公共库检索")
        st.stop()

    # 添加用户消息
    st.session_state.messages.append(
        {"role": "user", "content": prompt, "avatar": "👤"}
    )
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # 生成回答
    with st.chat_message("assistant", avatar="🤖"):
        answer_placeholder = st.empty()
        status_placeholder = st.empty()

        full_answer = ""
        meta_info = {}

        # ── 根据 Agent 模式选择不同的流式管线 ──
        if enable_agent:
            # Agent 模式：多步推理
            stream = agent_ask_stream(
                prompt,
                username=username,
                include_shared=include_shared,
            )

            for chunk in stream:
                if chunk["type"] == "phase":
                    # 显示 Agent 当前阶段
                    phase_icons = {
                        "planning": "🧠",
                        "researching": "🔍",
                        "verifying": "✅",
                        "synthesizing": "✍️",
                        "critiquing": "🔎",
                    }
                    icon = phase_icons.get(chunk["phase"], "⏳")
                    status_placeholder.info(f"{icon} {chunk['text']}")
                elif chunk["type"] == "answer":
                    full_answer = chunk["text"]
                    answer_placeholder.markdown(full_answer)
                    status_placeholder.empty()
                elif chunk["type"] == "done":
                    meta_info = chunk
                    answer_placeholder.markdown(full_answer)
                    status_placeholder.empty()
        else:
            # 普通模式：直接检索+回答
            with st.spinner("🔍 正在检索相关文档..."):
                stream = ask_stream(
                    prompt,
                    top_k=top_k,
                    temperature=temperature,
                    username=username,
                    include_shared=include_shared,
                )

            for chunk in stream:
                if chunk["type"] == "answer":
                    full_answer += chunk["text"]
                    answer_placeholder.markdown(full_answer + "▌")
                elif chunk["type"] == "done":
                    meta_info = chunk
                    answer_placeholder.markdown(full_answer)

        # 显示引用来源（标注 KB 来源）
        if meta_info.get("sources"):
            with st.expander("📎 引用来源"):
                for s in meta_info["sources"]:
                    kb_tag = "🔒私人" if s.get("kb_source") == "private" else "🌐公共"
                    st.caption(f"📄 {s['source']} {kb_tag}")
                    st.text(s["content_preview"])

        # 显示改写后的查询（普通模式）
        search_query = meta_info.get("search_query", "")
        if search_query and search_query != prompt:
            st.caption(f"🔍 改写查询: _{search_query}_")

        # ── Agent 模式的额外展示 ──
        if enable_agent:
            # 展示子问题拆解
            sub_questions = meta_info.get("sub_questions", [])
            if sub_questions:
                with st.expander(f"🧠 问题拆解 ({len(sub_questions)} 个子问题)"):
                    for i, sq in enumerate(sub_questions, 1):
                        st.caption(f"{i}. {sq}")

            # 展示验证报告
            verification = meta_info.get("verification", "")
            if verification:
                with st.expander("✅ 交叉验证报告"):
                    st.text(verification)

            # 展示 Critic 审查
            critique = meta_info.get("critique", "")
            if critique:
                with st.expander("🔎 自我审查"):
                    st.text(critique)

            # 展示循环次数
            iterations = meta_info.get("iterations", 0)
            if iterations > 0:
                st.caption(f"🔄 Agent 进行了 {iterations} 轮补充搜索")

        # 显示耗时
        retrieval_time = meta_info.get("retrieval_time", 0)
        generation_time = meta_info.get("generation_time", 0)
        rerank_time = meta_info.get("rerank_time", 0)
        docs_retrieved = meta_info.get("docs_retrieved", 0)
        docs_relevant = meta_info.get("docs_relevant", 0)
        docs_before_rerank = meta_info.get("docs_before_rerank", 0)
        parts = [f"🔍 检索 {retrieval_time:.2f}s"]
        if rerank_time > 0:
            parts.append(f"| 🎯 重排 {rerank_time:.2f}s ({docs_before_rerank}→{docs_relevant}条)")
        parts.append(f"| ✍️ 生成 {generation_time:.2f}s")
        parts.append(f"| 📊 检索到 {docs_retrieved} 条 ({docs_relevant} 条相关)")
        st.caption(" ".join(parts))

    # 保存到历史（带 kb_source 标注）
    sources_with_kb = []
    for s in meta_info.get("sources", []):
        s_copy = dict(s)
        # hybrid_retriever 返回的 doc metadata 中有 kb_source
        s_clean = {"source": s.get("source", ""), "content_preview": s.get("content_preview", "")}
        sources_with_kb.append(s_clean)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_answer,
            "avatar": "🤖",
            "sources": meta_info.get("sources", []),
            "retrieval_time": meta_info.get("retrieval_time", 0),
            "generation_time": meta_info.get("generation_time", 0),
            "docs_retrieved": meta_info.get("docs_retrieved", 0),
            "docs_relevant": meta_info.get("docs_relevant", 0),
            "search_query": meta_info.get("search_query", ""),
            "rerank_time": meta_info.get("rerank_time", 0),
            "docs_before_rerank": meta_info.get("docs_before_rerank", 0),
        }
    )

# ── 页脚 ──
st.divider()
st.caption("💡 提示：不同用户的文档完全隔离。开启「公共库检索」可同时搜索共享资料。")
