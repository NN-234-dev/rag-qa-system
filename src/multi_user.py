# ============================================
# 多用户知识库管理 — Multi-User Knowledge Base
# ============================================
# 每个用户拥有独立的 Chroma collection（私人库），
# 同时所有用户共享一个 kb_shared collection（公共库）。
#
# 类比 Python venv：
#   kb_user_alice/  = ~/venvs/project_a/   (隔离)
#   kb_user_bob/    = ~/venvs/project_b/   (隔离)
#   kb_shared/      = /usr/lib/python3.12/ (共享，只读)
#
# 设计决策：
# - 用户之间完全隔离：Alice 上传的文档 Bob 搜不到
# - 公共库全员可搜：适合放公司制度、技术手册等共享资料
# - 用户只是字符串标签，无需密码（本地单机演示场景）


# ============================================
# 命名规则
# ============================================
USER_COLLECTION_PREFIX = "kb_user_"
SHARED_COLLECTION_NAME = "kb_shared"


def get_user_collection_name(username: str) -> str:
    """根据用户名生成 Chroma collection 名称"""
    # 清理用户名中的特殊字符
    safe_name = username.strip().lower().replace(" ", "_")
    return f"{USER_COLLECTION_PREFIX}{safe_name}"


def get_shared_collection_name() -> str:
    """公共库 collection 名称"""
    return SHARED_COLLECTION_NAME


def get_username_from_collection(collection_name: str) -> str:
    """从 collection 名称反推用户名"""
    if collection_name == SHARED_COLLECTION_NAME:
        return "shared"
    if collection_name.startswith(USER_COLLECTION_PREFIX):
        return collection_name[len(USER_COLLECTION_PREFIX):]
    return collection_name
