# learn-langchain

LangChain / LangGraph 大模型应用开发学习仓库。全程基于**本地模型**（vLLM 部署的 Qwen3-0.6B，OpenAI 兼容接口），所有代码注释与文档均为中文，按 9 个阶段循序渐进。

## 学习路线

详细方案见 [docs/01-learning-plan.md](docs/01-learning-plan.md)（各阶段的知识点、实践任务、完成标准）。

| 阶段 | 主题 | 状态 |
|---|---|---|
| 1 | LangChain 与 LangGraph 框架认知（安装、环境验证、首次体验） | ✅ |
| 2 | LangChain 核心组件（模型调用、提示词模板、输出解析） | ⬜ |
| 3 | LangChain 进阶组件（记忆、工具、组合实践） | ⬜ |
| 4 | 应用级系统设计与 RAG（链式工作流、检索增强生成） | ⬜ |
| 5 | 智能体应用设计与实现 | ⬜ |
| 6 | LangGraph 基础（有状态工作流、节点、边、状态管理） | 🔶 最小示例已跑通 |
| 7 | LangGraph 进阶（多智能体协作、复杂流程管控） | ⬜ |
| 8 | 智能体综合实战 | ⬜ |
| 9 | 项目总结与展望 | ⬜ |

## 目录结构

```
learn-langchain/
├── docs/                      # 学习文档
│   └── 01-learning-plan.md    # 学习方案总纲（9 阶段详细计划）
├── test_langchain_env.py      # 阶段1：环境与依赖版本验证（langchain/langgraph/openai + .env）
├── test_langchain_hello.py    # 阶段1：LangChain 最小调用（ChatOpenAI 单轮对话）
├── test_langgraph_hello.py    # 阶段1/6：LangGraph 最小工作流（两节点：生成→精简）
├── CLAUDE.md                  # Claude Code 协作约定
├── .env.example               # 环境变量模板（.env 不入库）
└── .gitignore
```

> 说明：`test_*.py` 是教学演示脚本（模块顶层直接执行），不是 pytest 测试。
> 后续各阶段代码将放入 `ch02-core/` … `ch08-project/` 目录，阶段笔记放入 `docs/`。

## 环境准备

### 1. Python 环境

conda 环境 `ai-gpu`（Python 3.12），核心依赖：

```bash
pip install langchain langchain-openai langgraph python-dotenv
```

已验证版本：langchain 1.4.x / langchain-openai 1.6.x / langgraph 1.2.x（**1.x 新 API，0.x 老教程的 `LLMChain`、`Memory`、`AgentExecutor` 已移除**，详见学习方案第五节对照表）。

### 2. 本地模型服务（vLLM）

```bash
vllm serve /home/dupengair/shared/LLM/Fine-tuning/model/Qwen3-0.6B \
  --served-model-name qwen3-0.6b --max-model-len 4096
```

### 3. 环境变量

复制模板并按需修改（本地服务无需真实 API Key）：

```bash
cp .env.example .env
```

```dotenv
API_KEY="EMPTY"
BASE_URL="http://localhost:8000/v1"
MODEL="qwen3-0.6b"
```

## 快速开始

```bash
# 确认 vLLM 服务在线后，依次运行
python3 test_langchain_env.py     # 验证依赖版本与 .env 配置
python3 test_langchain_hello.py   # LangChain：生成一段学习建议
python3 test_langgraph_hello.py   # LangGraph：生成→精简 两节点工作流
```

## 约定

- 代码：中文编号注释（`# 1. 导入模块` …），单文件自包含，教学优先
- 文档：统一放 `docs/`，命名 `NN-主题.md`
- 模型约束：Qwen3-0.6B 上下文仅 4096 tokens、指令遵循与工具调用能力有限——判断代码对错以逻辑为准，不纠结输出文采（详见学习方案第二节）
