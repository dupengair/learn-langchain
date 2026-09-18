# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个 LangChain / LangGraph 的学习实践项目（跟随教程编写，非 LangChain 官方源码仓库）。每个 `test_*.py` 都是一个**独立的教学演示脚本**，不是 pytest 测试——它们在模块顶层直接执行 LLM 调用。**不要运行 pytest**（也未安装 pytest），直接用 python 运行脚本。

## 运行环境

- Python 来自 conda 环境 `ai-gpu`：`/home/dupengair/shared/conda/anaconda3/envs/ai-gpu/bin/python3`（Python 3.12）
- **没有 requirements.txt / pyproject.toml**，依赖直接装在 conda 环境里。缺少依赖时用 `python3 -m pip install <pkg>` 装入该环境
- 关键包：langchain 1.x（1.4.x）、langchain-openai 1.x、langgraph 1.x、openai 3.x、python-dotenv

## LLM 后端配置

模型通过 `.env` 配置，指向**本地 OpenAI 兼容服务**（如 vLLM）：

- `API_KEY="EMPTY"`（本地服务不需要真实 key）
- `BASE_URL="http://localhost:8000/v1"`
- `MODEL="qwen3-0.6b"`（0.6B 的本地小模型，输出质量低、可能不遵循指令——不能以生成内容的质量来判断脚本是否正确）

运行任何脚本前需确认本地推理服务已启动在 8000 端口，否则调用会连接失败。

## 常用命令

```bash
# 直接运行演示脚本（需 ai-gpu 环境的 python，且本地 LLM 服务在线）
python3 test_langchain_hello.py    # 最小 ChatOpenAI 调用
python3 test_langgraph_hello.py    # LangGraph 两节点工作流
```

没有配置 lint / format / 测试框架，不要假设存在这些命令。

## 代码约定（新脚本应遵循的既有模式）

1. **环境加载与校验模式**：每个脚本开头都是 `load_dotenv()` → 读取 `API_KEY` / `BASE_URL` / `MODEL` → 逐项 `if not ...: raise ValueError(...)`，然后初始化 `ChatOpenAI(api_key=..., base_url=..., model=...)`。新脚本保持一致。
2. **中文编号注释**：脚本用 `# 1. 导入模块`、`# 2. 加载 .env 环境变量` 这类带步骤编号的中文注释组织结构，注释和文案全部用中文。
3. **教学优先**：单文件自包含、逻辑直白（如 Prompt 用字符串而非模板），避免为"工程化"而引入抽象层。

## 两种核心模式

- **单次调用**（`test_langchain_hello.py`）：`ChatOpenAI` 初始化后直接 `llm.invoke(prompt)`，取 `response.content`。
- **LangGraph 工作流**（`test_langgraph_hello.py`）：`TypedDict(total=False)` 定义 State → 节点函数接收 state、返回部分状态 dict → `StateGraph` + `add_node` / `add_edge(START, ..., END)` → `compile()` → `app.invoke({...})`。

## 工具与文档

- **代码分析用 CodeGraph**：本仓库有 `.codegraph/` 索引，分析 LangChain、LangGraph 内部实现时优先用 `codegraph_explore`，而不是 grep + 逐文件读。
- 回答问题使用中文
- 需要记录的文档放到 ./docs 下面
