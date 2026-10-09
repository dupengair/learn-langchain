# learn-langchain

LangChain / LangGraph 大模型应用开发学习仓库。全程基于**本地模型服务**（OpenAI 兼容接口），所有代码注释与文档均为中文，按 9 个阶段循序渐进。

当前共 **67 个教学演示脚本**（`test_*.py`，模块顶层直接执行，不是 pytest 测试），配套 **11 篇文档**（1 篇学习方案 + 10 篇踩坑/原理）。脚本默认在 **Ollama + Qwen3.5-4B(GGUF)** 下运行，可通过 `.env` 一行切换到 **vLLM + Qwen3-0.6B**。

## 学习路线

详细方案见 [docs/01-learning-plan.md](docs/01-learning-plan.md)（各阶段的知识点、实践任务、完成标准）。

| 阶段 | 主题 | 状态 |
|---|---|---|
| 1 | LangChain 与 LangGraph 框架认知（安装、环境验证、首次体验） | ✅ |
| 2 | LangChain 核心组件（模型调用、提示词模板、输出解析） | ✅ |
| 3 | LangChain 进阶组件（记忆、工具、组合实践） | ✅ |
| 4 | 应用级系统设计与 RAG（文档加载、切分、向量检索、RAG 问答） | ✅ |
| 5 | 智能体应用设计与实现（Function Calling） | ✅ |
| 6 | LangGraph 基础（有状态工作流、节点、边、状态管理、superstep） | ✅ |
| 7 | LangGraph 进阶（多智能体协作） | 🔶 基础协作、checkpoint/interrupt、并行、子图已跑通 |
| 8 | 智能体综合实战 | ⬜ |
| 9 | 项目总结与展望 | ⬜ |

## 目录结构

```
learn-langchain/
├── llm_config.py              # ★ 后端自适应的 ChatOpenAI 工厂(全项目唯一适配层)
├── docs/                      # 学习文档与踩坑记录(见文末列表)
├── CLAUDE.md                  # Claude Code 协作约定
├── .env.example               # 环境变量模板(.env 不入库)
└── test_*.py                  # 67 个教学演示脚本,按主题分组如下
```

| 主题 | 脚本 |
|---|---|
| 环境与最小调用 | `test_langchain_env` `test_langchain_hello` `test_langchain_hf`(HuggingFace 本地 pipeline) |
| 提示词工程 | `test_langchain_prompt-temp` `test_langchain_prompt-fewshot` `test_langchain_prompt-fewshot-json` |
| 输出解析 | `test_langchain_outputparser-base` `test_langchain_outputparser-str` `test_langchain_outputparser-json` `test_langchain_outputparser-pydantic` |
| 链式工作流与容错 | `test_langchain_workflow-chain` `test_langchain_workflow-multichain` `test_langchain_routerchain-base` `test_langchain_chainerr-except` `test_langchain_chainerr-retry` `test_langchain_chainerr-fallback` |
| 对话记忆 | `test_langchain_chatbot` `test_langchain_chatbot-rounds` `test_langchain_chatbot-mem` `test_langchain_chatbot-mem2` `test_langchain_manualmem` `test_langchain_fullmem` `test_langchain_winmem` `test_langchain_summem` |
| 工具(Function Calling) | `test_langchain_tool` `test_langchain_tool-def` `test_langchain_tool-file` `test_langchain_tool-mem` |
| 文档加载与切分 | `test_langchain_load-txt` `test_langchain_load-md` `test_langchain_load-pdf` `test_langchain_load-docs` `test_langchain_load-multi` `test_langchain_textsplit-CharacterTextSplitter` `test_langchain_textsplit-MarkdownTextSplitter` `test_langchain_textsplit-RecursiveCharacterTextSplitter` |
| 向量与 RAG | `test_langchain_vecemb-textvec` `test_langchain_vecemb-faissretrive` `test_langchain_vecemb-search` `test_langchain_vecemb-mmr` `test_langchain_vecemb-RAG-qa` `test_langchain_vecemb-RAG-evaluate`(ragas 评估) |
| LangGraph 基础 | `test_langgraph_hello` `test_langgraph_node` `test_langgraph_status` `test_langgraph_simple-llm` `test_langgraph_edges-fix` `test_langgraph_edges-cond` `test_langgraph_edges-loop` `test_langgraph_superstep-linear` `test_langgraph_superstep-execflow` `test_langgraph_superstep-branch` `test_langgraph_superstep-loop` |
| 多智能体协作 | `test_multiagent_simpleLLM` `test_multiagent_multiLLM` `test_multiagent_parallel`(并行节点) `test_multiagent_arch-supervisor`(supervisor 架构) `test_multiagent_subgraph`(子图) |
| 持久化与人工干预 | `test_multiagent_checkpoint`(存档恢复) `test_multiagent_interrupt-before`(执行前人工确认) `test_multiagent_interrupt-after` `test_multiagent_retry-count`(重试计数) `test_multiagent_state-undo`(状态撤销) |
| 状态管理综合实战 | `test_multiagent_state-manual`(手动状态管理) `test_multiagent_novel-agent`(小说创作全流程+进度追踪,输出见 `novel_final_output.txt`) |

## 环境准备

### 1. Python 环境

conda 环境 `ai-gpu`（Python 3.12），核心依赖：

```bash
pip install langchain langchain-openai langgraph python-dotenv
```

已验证版本：langchain 1.4.1 / langgraph 1.2.11 / openai 1.109.1（**1.x 新 API，0.x 老教程的 `LLMChain`、`Memory`、`AgentExecutor` 已移除**，详见学习方案第五节对照表）。

### 2. 本地模型服务（二选一，共用 8000 端口）

**Ollama（默认，跑 GGUF 量化模型）**：

```bash
export OLLAMA_HOST=127.0.0.1:8000   # 与 vLLM 共用 8000 端口,切换后端时 .env 的 BASE_URL 不用改
ollama serve
# 注册本地 GGUF(已有则跳过):Modelfile 内容为 FROM ./Qwen3.5-4B-q4_k_m.gguf
ollama create qwen3.5-4b:q4 -f Modelfile
```

**vLLM（备选，跑 FP16 原始权重）**：

```bash
python3 test_vllm_qwen3-0.6b.py   # 位于 ~/shared/LLM/vllm/,served-model-name qwen3-0.6b
```

确认 8000 端口上跑的是谁：`curl http://localhost:8000/api/tags` 返回 Ollama 模型列表即 Ollama 在线。

### 3. 环境变量

复制模板并按需修改（本地服务无需真实 API Key）：

```bash
cp .env.example .env
```

```dotenv
BACKEND="ollama"                      # 当前推理后端: ollama / vllm,切换只改这一行
BASE_URL="http://localhost:8000/v1"   # 两后端共用
MODEL_OLLAMA="qwen3.5-4b:q4"
MODEL_VLLM="qwen3-0.6b"
MODEL_PATH="/path/to/Fine-tuning/"    # 仅 RAG 系列 embedding 脚本使用
```

## 快速开始

```bash
python3 test_langchain_env.py     # 验证依赖版本与后端连通
python3 test_langchain_hello.py   # LangChain:生成一段学习建议
python3 test_langgraph_hello.py   # LangGraph:生成→精简 两节点工作流
```

**切换后端**：改 `.env` 的 `BACKEND="vllm"` 即可，脚本零改动；临时测试也可用环境变量覆盖（`load_dotenv` 默认不覆盖已有环境变量）：

```bash
BACKEND=vllm python3 test_langchain_hello.py
```

## 架构:`llm_config.py` 后端自适应工厂

所有脚本通过共享工厂初始化模型，**不直连 `ChatOpenAI`、不手写环境加载与校验**：

```python
from llm_config import get_chat_model

llm = get_chat_model(temperature=0.3, max_tokens=1024)
```

工厂按 `.env` 的 `BACKEND` 自动完成两件事：

1. **选模型**：读取 `MODEL_OLLAMA` 或 `MODEL_VLLM`
2. **关 thinking 的方式**（两引擎参数互不相认）：
   - Ollama:`reasoning_effort="none"`（OpenAI 标准参数）
   - vLLM:`extra_body={"chat_template_kwargs": {"enable_thinking": False}}`（模板变量；vLLM 0.10.2 不认 `"none"` 值，会 422）

注意事项：依赖 `.env` 的其他变量（如 `MODEL_PATH`）须在 `get_chat_model()` **之后**读取；`max_tokens` 不要顶到 4096（Ollama 运行时上下文默认 4096，顶满会截断在思考阶段导致输出为空）。完整分析见 [docs/09-gguf-model-load.md](docs/09-gguf-model-load.md)。

## 踩坑与原理文档

| 文档 | 主题 |
|---|---|
| [00-学习笔记](docs/00-学习笔记.md) | 各阶段知识点笔记(Memory/LCEL/RAG/Agent…) |
| [02-hardware-model-guide](docs/02-hardware-model-guide.md) | 硬件与模型选型 |
| [03-close-qwen3-thinking](docs/03-close-qwen3-thinking.md) | vLLM 下关闭 Qwen3 thinking 的四种方案实测 |
| [04-few-shot-prompt-template](docs/04-few-shot-prompt-template.md) | Few-shot 提示词模板 |
| [05-chatbot-mem-calc-bug](docs/05-chatbot-mem-calc-bug.md) | 对话记忆计算 bug 分析 |
| [06-vecemb-bert-unexpected](docs/06-vecemb-bert-unexpected.md) | 向量嵌入 BERT 意外行为 |
| [07-ragas-langchain-community-compat](docs/07-ragas-langchain-community-compat.md) | ragas 与 langchain-community 兼容性垫片 |
| [08-faiss-load-local-index-name](docs/08-faiss-load-local-index-name.md) | FAISS 本地索引加载的名称坑 |
| [09-gguf-model-load](docs/09-gguf-model-load.md) | ★ GGUF 模型加载、Ollama 后端适配、关 thinking 参数演进(8~10 节为本仓库架构定稿依据) |
| [10-interrupt-before-stream-loop](docs/10-interrupt-before-stream-loop.md) | ★ interrupt_before 中断语义六问:stream/invoke 入口、for 循环与引擎停摆、GraphInterrupt 源码、generator 惰性触发(附纯图实验) |

## 约定

- 代码：中文编号注释（`# 1. 导入模块` …），单文件自包含，教学优先；模型初始化统一走 `llm_config.get_chat_model`
- 文档：统一放 `docs/`，命名 `NN-主题.md`
- 输出质量：Ollama 4B 模型输出尚可；vLLM 的 Qwen3-0.6B 指令遵循与工具调用能力有限——判断代码对错以逻辑为准，不纠结输出文采
