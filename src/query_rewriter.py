# ============================================
# 查询改写器 — Query Rewriting
# ============================================
# 用本地 Ollama 模型将用户的口语化问题改写为更精确的搜索查询。
# 改写后的查询用于向量检索 + BM25，原始问题保留用于 LLM 生成答案。
#
# 原理：
# 用户可能用口语、简称、模糊表述提问，
# LLM 能理解语义并补充同义词、纠正表述、提取关键词，
# 从而提高检索召回率。
#
# 为什么不用规则（同义词表/分词器）？
# - 规则方法只能覆盖预设的模式，遇到新表述就失效
# - LLM 能理解上下文，把"那玩意儿"映射到"系统配置界面"
# - 而且零维护：加新领域不需要手工维护词表

from src.llm_client import generate


REWRITE_PROMPT = """你是一个搜索查询优化专家。用户会提出一个问题，你需要把它改写成一个更利于语义检索的查询语句，用来从知识库中搜索相关文档。

## 改写规则
1. 纠正口语化表述，改为正式的书面语
2. 如果用户使用了简称或指代词（"这个""那个""它"），替换为完整的具体表述
3. 适当补充同义词和相关术语，提高检索覆盖面
4. 保留原问题的所有关键信息，不要遗漏
5. **只输出改写后的查询，不要加引号、解释、前缀或任何额外文字**

## 示例
用户: 这玩意儿怎么配置？
改写: 系统配置方法 设置步骤 参数说明

用户: 模型的准确率怎么样
改写: 模型准确率 评估指标 性能测试结果

## 用户问题
{question}

## 改写查询"""


def rewrite_query(question: str, temperature: float = 0.2) -> str:
    """
    将用户原始问题改写为更适合检索的查询语句。

    使用低温度（0.2）确保改写结果稳定、不偏离原意。

    Args:
        question: 用户输入的原始问题
        temperature: 改写温度，默认 0.2（越低越保守）

    Returns:
        改写后的查询字符串。
        如果 LLM 调用失败，返回原始问题（降级策略）。
    """
    try:
        prompt = REWRITE_PROMPT.format(question=question)
        rewritten = generate(prompt, temperature=temperature)
        # 清理：去掉可能残留的标签和空白
        rewritten = rewritten.strip().strip('"').strip("'").strip()
        # 如果改写结果为空或太短，返回原问题
        if not rewritten or len(rewritten) < 2:
            return question
        return rewritten
    except Exception:
        # 降级：任何异常都返回原始问题，不影响主流程
        return question
