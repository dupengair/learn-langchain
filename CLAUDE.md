# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个 LangChain / LangGraph 的学习实践项目（跟随教程编写，非 LangChain 官方源码仓库）。每个 `test_*.py` 都是一个**独立的教学演示脚本**，不是 pytest 测试——它们在模块顶层直接执行 LLM 调用。**不要运行 pytest**（也未安装 pytest），直接用 python 运行脚本。

## 运行环境

- Python 来自 conda 环境 `ai-gpu`：`/home/dupengair/shared/conda/anaconda3/envs/ai-gpu/bin/python3`（Python 3.12）
- **没有 requirements.txt / pyproject.toml**，依赖直接装在 conda 环境里。缺少依赖时用 `python3 -m pip install <pkg>` 装入该环境
- 关键包：langchain 1.x（1.4.x）、langchain-openai 1.x、langgraph 1.x、openai 3.x、python-dotenv

## LLM 后端配置

**脚本优先在本地 Ollama 下运行**。后端适配由项目根的 **`llm_config.py`** 集中管理（这是唯一的适配层，改后端逻辑只改这一个文件，不要在脚本里直连 `ChatOpenAI`）：

- `.env` 的 `BACKEND` 显式声明当前后端（`ollama` / `vllm`，默认 ollama），**切换后端只改这一行**
- `MODEL_OLLAMA="qwen3.5-4b:q4"`（Qwen3.5-4B 的 Q4_K_M GGUF，thinking 模型）、`MODEL_VLLM="qwen3-0.6b"`（0.6B 小模型，输出质量低——不能以生成内容的质量判断脚本是否正确）按后端分开定义，新增模型只改 `.env`
- `BASE_URL="http://localhost:8000/v1"` 两后端共用：Ollama 经 `export OLLAMA_HOST=127.0.0.1:8000` 启动在与 vLLM 相同的端口上。运行前用 `curl http://localhost:8000/api/tags` 确认 8000 上跑的是目标服务

脚本内初始化的唯一写法（`load_dotenv` 和配置校验都在工厂内部完成）：

```python
from llm_config import get_chat_model

llm = get_chat_model(temperature=0.3, max_tokens=1024)
```

**两条硬约束**（详细分析见 `docs/09-gguf-model-load.md` 第 8、9、10 节）：

1. **关 thinking 的参数按后端分支，已封装进 `llm_config.py`**：Ollama 用 `reasoning_effort="none"`（OpenAI 标准参数），vLLM 0.10.2 不认 `"none"` 值会 422、须用 `extra_body={"chat_template_kwargs": {"enable_thinking": False}}`。两引擎参数互不相认，不要在脚本里手写任何一种。
2. **`max_tokens` 不要顶到 4096**。Ollama 运行时上下文默认只有 4096（`curl http://localhost:8000/api/ps` 的 `context_length` 才是真实生效值），顶满会截断在思考阶段导致 `response.content` 为空。保持 1024~2048 即可。

依赖 `.env` 的其他变量（如 `MODEL_PATH`）必须在 `get_chat_model()` 调用**之后**读取（工厂内部先执行 `load_dotenv()`）。临时切换后端测试可用环境变量覆盖：`BACKEND=vllm python3 test_xxx.py`（`load_dotenv` 默认不覆盖已有环境变量）。

## 常用命令

```bash
# 直接运行演示脚本（需 ai-gpu 环境的 python，且 Ollama 服务在线）
python3 test_langchain_hello.py    # 最小 ChatOpenAI 调用
python3 test_langgraph_hello.py    # LangGraph 两节点工作流
```

没有配置 lint / format / 测试框架，不要假设存在这些命令。

## 代码约定（新脚本应遵循的既有模式）

1. **模型初始化模式**：新脚本统一 `from llm_config import get_chat_model` 后 `llm = get_chat_model(temperature=0.3, max_tokens=1024)`，**不要**直接 `ChatOpenAI(...)`、不要手写 `load_dotenv` 后的环境校验（工厂内部已做）。后端差异（模型名、关 thinking 参数）全部由 `llm_config.py` 处理。
2. **中文编号注释**：脚本用 `# 1. 导入模块`、`# 2. 加载 .env 环境变量` 这类带步骤编号的中文注释组织结构，注释和文案全部用中文。
3. **教学优先**：单文件自包含、逻辑直白（如 Prompt 用字符串而非模板），避免为"工程化"而引入抽象层。

## 两种核心模式

- **单次调用**（`test_langchain_hello.py`）：`llm = get_chat_model(...)` 初始化后直接 `llm.invoke(prompt)`，取 `response.content`。
- **LangGraph 工作流**（`test_langgraph_hello.py`）：`TypedDict(total=False)` 定义 State → 节点函数接收 state、返回部分状态 dict → `StateGraph` + `add_node` / `add_edge(START, ..., END)` → `compile()` → `app.invoke({...})`。

## 工具与文档

- **代码分析用 CodeGraph**：本仓库有 `.codegraph/` 索引，分析 LangChain、LangGraph 内部实现时优先用 `codegraph_explore`，而不是 grep + 逐文件读。
- 回答问题使用中文
- 需要记录的文档放到 ./docs 下面
