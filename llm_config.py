"""
llm_config.py —— 后端自适应的 ChatOpenAI 工厂（全项目共享）

按 .env 里的 BACKEND 变量返回配置好的 ChatOpenAI 实例，脚本其余部分不感知后端差异：

- BACKEND="ollama"：模型取 MODEL_OLLAMA，关 thinking 用 reasoning_effort="none"
- BACKEND="vllm"  ：模型取 MODEL_VLLM，  关 thinking 用 extra_body 模板变量

两个后端共用 BASE_URL（http://localhost:8000/v1，Ollama 经 export OLLAMA_HOST=127.0.0.1:8000
启动在与 vLLM 相同的端口上），因此切换后端只需改 .env 的 BACKEND 一行，脚本零改动。

为什么关 thinking 要分后端：两个推理引擎的参数互不相认——
Ollama 认 OpenAI 标准值 reasoning_effort="none"，但不认 vLLM 的 chat_template_kwargs；
vLLM 0.10.2 的 reasoning_effort 枚举只有 low/medium/high，收到 "none" 直接 422。
详细分析见 docs/09-gguf-model-load.md 第 8、9、10 节。
"""
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


def get_chat_model(**extra) -> ChatOpenAI:
    """读取 .env 并按当前后端（BACKEND）返回配置好的 ChatOpenAI 实例。

    Args:
        **extra: 脚本自有参数（temperature、max_tokens 等），原样透传给 ChatOpenAI。
    """
    load_dotenv()

    # 1. 读取并校验后端标识
    backend = os.getenv("BACKEND", "ollama").lower()
    if backend not in ("ollama", "vllm"):
        raise ValueError(f"BACKEND 只支持 ollama / vllm，当前: {backend}")

    # 2. 读取公共配置与当前后端对应的模型名
    api_key = os.getenv("API_KEY")
    base_url = os.getenv("BASE_URL")
    model = os.getenv("MODEL_OLLAMA" if backend == "ollama" else "MODEL_VLLM")

    # 3. 配置校验（沿用项目既有模式）
    if not api_key:
        raise ValueError("未检测到 API_KEY，请检查 .env 文件是否配置正确")
    if not base_url:
        raise ValueError("未检测到 BASE_URL，请检查 .env 文件是否配置正确")
    if not model:
        raise ValueError(f"未检测到 MODEL_{backend.upper()}，请检查 .env 文件是否配置正确")

    kwargs = dict(api_key=api_key, base_url=base_url, model=model, **extra)

    # 4. 按后端选择关闭 thinking 的方式
    if backend == "ollama":
        kwargs["reasoning_effort"] = "none"   # OpenAI 标准参数，Ollama 兼容层支持
    else:
        kwargs["extra_body"] = {              # vLLM 私有约定：向 chat template 传变量
            "chat_template_kwargs": {"enable_thinking": False}
        }

    return ChatOpenAI(**kwargs)
