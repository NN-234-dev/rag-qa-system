# ============================================
# LLM 客户端 — Ollama 本地模型封装
# ============================================
# 同时支持文本模型 (deepseek-r1:7b) 和视觉模型 (llama3.2-vision:11b)
# Ollama 默认地址: http://localhost:11434/v1 (OpenAI 兼容接口)
#
# deepseek-r1 特殊处理：
# deepseek-r1 是推理模型，Ollama 会将思考过程放入 message.reasoning，
# 最终回答放入 message.content。但 content 中有时也会混入  标签，
# 需要做清理。

import base64
import re
from pathlib import Path
from openai import OpenAI
import config

_client = None
_vision_client = None


def _get_client() -> OpenAI:
    """获取文本模型客户端"""
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=config.OLLAMA_BASE_URL,
            api_key="ollama",  # Ollama 不需要真 key
        )
    return _client


def _get_vision_client() -> OpenAI:
    """获取视觉模型客户端"""
    global _vision_client
    if _vision_client is None:
        _vision_client = OpenAI(
            base_url=config.OLLAMA_BASE_URL,
            api_key="ollama",
        )
    return _vision_client


def _strip_thinking(text: str) -> str:
    """
    清理 deepseek-r1 输出中可能混入的 think 标签。

    Ollama 通常将 deepseek-r1 的思考过程分离到 message.reasoning 字段，
    message.content 只包含最终回答。但某些情况下（如流式输出）标签可能
    混入 content，做 fallback 清理。
    """
    if not text:
        return ""
    # 去掉  think ... /think  之间的内容（含标签本身）
    text = re.sub(r'think[\s\S]*?/think', '', text)
    # 去掉可能残留的标签碎片
    text = text.replace('<｜end▁of▁thinking｜> ', '').replace(' /response ', '')
    return text.strip()


def generate(prompt: str, temperature: float = None) -> str:
    """
    调用 Ollama 本地模型生成回答（非流式）。

    对 deepseek-r1 推理模型：
    - 思考过程在 response.choices[0].message.reasoning
    - 最终回答在 response.choices[0].message.content
    - 如果 content 为空（所有 token 被思考消耗），尝试从 reasoning 提取
      最后一段作为回答（fallback）

    Args:
        prompt: 完整的提示词
        temperature: 生成温度

    Returns:
        模型生成的回答文本
    """
    if temperature is None:
        temperature = config.TEMPERATURE

    client = _get_client()

    try:
        response = client.chat.completions.create(
            model=config.OLLAMA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "你是一个严谨的知识库问答助手。请严格依据参考资料回答，不要编造信息。",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=config.MAX_TOKENS,
        )
        msg = response.choices[0].message
        text = msg.content or ""

        # Fallback: 如果 content 为空，尝试从 reasoning 中提取有用内容
        if not text and hasattr(msg, 'reasoning') and msg.reasoning:
            # reasoning 的最后部分通常是模型的最终回答
            reasoning = msg.reasoning
            # 尝试在 reasoning 中找 "answer" "回答" 之后的文本
            text = reasoning

        return _strip_thinking(text)

    except Exception as e:
        error_msg = str(e)
        if "Connection refused" in error_msg or "ConnectionError" in error_msg:
            return (
                "[错误] 无法连接 Ollama，请确认：\n"
                "1. Ollama 已启动（运行 ollama serve）\n"
                "2. 模型已下载（运行 ollama list 查看）"
            )
        return f"[错误] LLM 调用失败: {error_msg}"


def generate_stream(prompt: str, temperature: float = None):
    """
    调用 Ollama 本地模型生成回答（流式输出）。

    流式输出原理：
    1. stream=True 让 Ollama 逐 token 返回结果
    2. 每个 chunk 包含一小段文字
    3. 前端逐字渲染，用户不用等

    对 deepseek-r1：流式模式下思考 token 在 chunk.choices[0].delta.reasoning，
    回答 token 在 chunk.choices[0].delta.content。只 yield content 部分。

    Yields:
        每次 yield 一段新生成的文本
    """
    if temperature is None:
        temperature = config.TEMPERATURE

    client = _get_client()

    try:
        response = client.chat.completions.create(
            model=config.OLLAMA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "你是一个严谨的知识库问答助手。请严格依据参考资料回答，不要编造信息。",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=config.MAX_TOKENS,
            stream=True,
        )

        for chunk in response:
            delta = chunk.choices[0].delta
            # 只输出 content，跳过 reasoning（思考过程）
            if delta.content is not None and delta.content:
                yield delta.content

    except Exception as e:
        yield f"[错误] LLM 调用失败: {str(e)}"


# ============================================
# 视觉模型 — 图片识别/描述
# ============================================

def describe_image(image_path: str, question: str = None) -> str:
    """
    用视觉模型描述图片内容。

    Args:
        image_path: 图片文件路径（支持 jpg/png/bmp/gif）
        question: 可选，针对图片提具体问题。默认让模型详细描述。

    Returns:
        图片内容的文字描述
    """
    if question is None:
        question = "请详细描述这张图片中的所有内容，包括文字、表格数据、图表信息等。"

    client = _get_vision_client()

    # 读取图片并转 base64
    img_path = Path(image_path)
    if not img_path.exists():
        return f"[错误] 图片文件不存在: {image_path}"

    with open(img_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    # 根据扩展名确定 MIME 类型
    ext = img_path.suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".bmp": "image/bmp",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    mime_type = mime_map.get(ext, "image/jpeg")

    try:
        response = client.chat.completions.create(
            model=config.OLLAMA_VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_data}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=config.MAX_TOKENS,
        )
        return response.choices[0].message.content

    except Exception as e:
        error_msg = str(e)
        if "Connection refused" in error_msg or "ConnectionError" in error_msg:
            return (
                "[错误] 无法连接 Ollama，请确认 Ollama 已启动。"
            )
        if "model" in error_msg.lower() and ("not found" in error_msg.lower() or "not exist" in error_msg.lower()):
            return (
                f"[错误] 视觉模型 {config.OLLAMA_VISION_MODEL} 未下载。\n"
                f"请在终端运行: ollama pull {config.OLLAMA_VISION_MODEL}"
            )
        return f"[错误] 图片识别失败: {error_msg}"


def describe_images_batch(image_paths: list, question: str = None) -> dict:
    """
    批量描述多张图片。

    Args:
        image_paths: 图片路径列表
        question: 可选，统一的问题

    Returns:
        {image_path: description} 的字典
    """
    results = {}
    for img_path in image_paths:
        print(f"  [Vision] 正在识别: {img_path}")
        desc = describe_image(img_path, question)
        results[img_path] = desc
    return results
