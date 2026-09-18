# LangChain / LangGraph 学习方案

> 本文档是整个仓库的学习总纲：9 个阶段的学习目标、知识点、实践任务与完成标准。
> 所有代码跑在**本地模型**（vLLM + Qwen3-0.6B）上，所有文档与注释使用中文。

## 一、学习目标

1. 理解 LangChain 的核心抽象（模型 I/O、提示词、输出解析、工具、RAG），能用 LCEL 组合出生产级链路；
2. 理解 LangGraph 的有状态工作流模型（State / Node / Edge / Checkpointer / 中断），能实现多智能体协作；
3. 最终独立设计并完成一个智能体综合项目；
4. 全程使用本地 OpenAI 兼容接口，掌握"模型可替换"的开发方式（换 API 供应商只改 `.env`）。

## 二、环境基线（已就绪）

| 项 | 值 |
|---|---|
| Python | 3.12（conda 环境 `ai-gpu`） |
| langchain | 1.4.x（**1.x 新版 API，非 0.x 老教程 API**） |
| langchain-openai | 1.6.x |
| langgraph | 1.2.x |
| 推理服务 | vLLM，OpenAI 兼容接口 `http://localhost:8000/v1` |
| 模型 | Qwen3-0.6B（`served-model-name: qwen3-0.6b`） |
| 上下文长度 | **4096 tokens（硬约束，设计 Prompt / RAG 时必须考虑）** |

vLLM 启动参考：

```bash
vllm serve /home/dupengair/shared/LLM/Fine-tuning/model/Qwen3-0.6B \
  --served-model-name qwen3-0.6b --max-model-len 4096
# 若练习工具调用（第 3、5、7 阶段），需额外启用工具解析参数，具体以所用 vLLM 版本文档为准
```

### 小模型约束与应对策略（贯穿全程）

- **指令遵循弱**：Prompt 要短、明确、单任务；不要指望一次生成复杂结构化长文；
- **工具调用可能不稳定**：涉及 Agent 的阶段若出现工具选择混乱，属于模型能力问题而非代码问题，可临时换更大的模型验证代码正确性（只改 `.env` 的 `MODEL` 与服务即可）；
- **上下文仅 4096**：RAG 分块要小、多轮对话历史要裁剪；
- **验收标准**：看代码逻辑与结构是否正确，**不以生成文本的文采/质量判断代码对错**。

## 三、总体路线（9 阶段）

| 阶段 | 主题 | 关键内容 | 代码目录 | 状态 |
|---|---|---|---|---|
| 1 | 框架认知 | 框架介绍、环境安装、框架体验 | 根目录 `test_*.py` | ✅ 已完成 |
| 2 | 核心组件实操 | 模型调用、提示词模板、输出解析 | `ch02-core/` | ⬜ |
| 3 | 进阶组件实操 | 记忆、工具、组合实践 | `ch03-tools-memory/` | ⬜ |
| 4 | 应用级系统与 RAG | 链式工作流、RAG 实践 | `ch04-rag/` | ⬜ |
| 5 | 智能体应用设计 | ReAct、create_agent、系统提示词 | `ch05-agent/` | ⬜ |
| 6 | LangGraph 基础 | 有状态工作流、节点、边、状态管理 | `ch06-langgraph-basics/` | 🔶 最小示例已跑通 |
| 7 | LangGraph 进阶 | 多智能体协作、复杂流程管控 | `ch07-multi-agent/` | ⬜ |
| 8 | 智能体综合实战 | 完整项目 | `ch08-project/` | ⬜ |
| 9 | 总结与展望 | 复盘、对比、后续方向 | `docs/` | ⬜ |

## 四、分阶段详细计划

### 阶段 1：框架认知 ✅

**已完成**：环境安装与连通性验证（`test_langchain_env.py`）、LangChain 单次调用（`test_langchain_hello.py`）、LangGraph 最小工作流（`test_langgraph_hello.py`）。

**收尾任务**：
- [ ] 把三个根目录脚本移入 `ch01-basics/`（README 同步更新）
- [ ] 用 `python3 -m pip freeze > requirements.txt` 固化当前依赖

### 阶段 2：LangChain 核心组件实操

**学习目标**：掌握模型 I/O 三件套——模型调用、提示词模板、输出解析，以及把它们串成 LCEL 链。

**知识点**：
- `ChatOpenAI` 关键参数：`temperature`、`max_tokens`、`timeout`；`invoke` / `batch` / `stream` 三种调用方式
- 提示词：`ChatPromptTemplate`（`SystemMessage` + `HumanMessage`）、`{变量}` 占位、`MessagesPlaceholder`
- 输出解析：`StrOutputParser`、`JsonOutputParser`、Pydantic 结构化输出（`with_structured_output`）
- LCEL 管道语法：`chain = prompt | llm | parser`

**实践任务**（新建 `ch02-core/`）：
- [ ] `model_basics.py`：同一 Prompt 对比 temperature=0 / 0.7 的输出；体验 `stream()` 流式打印
- [ ] `prompt_templates.py`：带变量的提示词模板 + `MessagesPlaceholder` 多轮消息
- [ ] `output_parsers.py`：让模型输出 JSON 并用解析器取字段；用 Pydantic 定义结构（字段尽量少，迁就小模型）
- [ ] `lcel_chain.py`：`prompt | llm | parser` 组合，对比手写三步的写法

**完成标准**：4 个脚本全部跑通，能说清 LCEL 管道每一步的输入输出类型。

### 阶段 3：LangChain 进阶组件实操

**学习目标**：掌握 1.x 下的"记忆"与"工具"，并组合成小应用。

**知识点**：
- **记忆 = 消息历史管理**（1.x 已移除 0.x 的 Memory 类）：手动维护 messages 列表；用 LangGraph 的 checkpointer（`InMemorySaver`）+ `thread_id` 自动管理
- 工具：`@tool` 装饰器定义工具（名称、描述、参数 schema），绑定到模型
- 组合实践：带一个本地工具（如"查询当前目录文件"）的最小 Agent

**实践任务**（新建 `ch03-tools-memory/`）：
- [ ] `chat_memory_manual.py`：手动拼接历史消息实现多轮对话（理解记忆本质）
- [ ] `define_tools.py`：用 `@tool` 定义 2~3 个简单工具并直接调用验证 schema
- [ ] `mini_agent.py`：`create_agent` + 工具 + `InMemorySaver`，多轮对话中让它使用工具并记住上下文

**完成标准**：能解释"记忆为什么在 1.x 中交给 LangGraph 管理"；mini_agent 能完成至少一次正确的工具调用。

### 阶段 4：应用级系统设计与 RAG 实践

**学习目标**：掌握复杂链的组合方式，跑通完整 RAG 链路。

**知识点**：
- LCEL 进阶：`RunnableParallel`、`RunnablePassthrough`、`RunnableLambda`，链的分支与合并
- RAG 五步：文档加载 → 分块（`RecursiveCharacterTextSplitter`，**块要小**）→ 向量化 → 检索 → 生成
- **Embedding 模型需单独准备**（Qwen3-0.6B 是对话模型，不含 embedding），二选一：
  - 简单路线：`sentence-transformers` + `BAAI/bge-small-zh-v1.5`（中文友好、体积小）本地算向量
  - 一致路线：vLLM 再起一个 embedding 服务，用 `OpenAIEmbeddings(base_url=...)` 走同一套接口
- 向量库：FAISS 或 Chroma（本地、零部署）

**依赖准备**（按需安装）：`langchain-community`、`faiss-cpu`（或 `chromadb`）、`sentence-transformers`、`pypdf`

**实践任务**（新建 `ch04-rag/`）：
- [ ] `split_demo.py`：对一篇本地 Markdown/_txt 文档分块，打印块数与长度
- [ ] `build_vectorstore.py`：文档 → 向量 → 入库，并做一次相似度检索
- [ ] `rag_chain.py`：检索结果塞进 Prompt → 回答"只依据文档"的问题；对比无 RAG 时的幻觉

**完成标准**：rag_chain 对文档外的问题能回答"文档中未提及"，不编造。

### 阶段 5：智能体应用设计与实现

**学习目标**：理解 Agent 的构成（模型 + 工具 + 策略 + 停止条件），会用 1.x 的 `create_agent`。

**知识点**：
- ReAct 循环原理：思考 → 行动 → 观察 → 再思考
- `create_agent`（langchain 1.x 的 Agent 工厂）：系统提示词、工具集、`max_iterations` 防失控
- 系统提示词设计：角色、能力边界、输出格式约定
- Agent vs Chain 的选型：确定性任务用链，开放性任务用 Agent

**实践任务**（新建 `ch05-agent/`）：
- [ ] `first_agent.py`：单工具 Agent，观察中间步骤（打印 Agent 执行轨迹）
- [ ] `multi_tool_agent.py`：2~3 个工具（如：计算器、时间查询、文件统计），测试小模型的工具选择准确率并记录现象
- [ ] `system_prompt_lab.py`：同一任务换 3 版系统提示词，对比行为差异

**完成标准**：能画出一次 Agent 运行的完整循环图；记录小模型工具调用的成功/失败案例。

### 阶段 6：LangGraph 基础——有状态工作流与核心概念 🔶

**已有基础**：`test_langgraph_hello.py` 已跑通 State → Node → Edge → compile → invoke 最小闭环。

**学习目标**：系统掌握 LangGraph 核心概念，摆脱"只会上帝视角直线图"的状态。

**知识点**：
- State 设计：`TypedDict`、`Annotated` + reducer（如 `operator.add` 做消息累加）
- 节点与边：普通边、**条件边**（`add_conditional_edges`）、`START`/`END`
- Checkpointer：`InMemorySaver` 持久化、`thread_id` 多会话、时间旅行（获取历史状态）
- 中断与人工介入：`interrupt` + `Command(resume=...)`
- 子图（subgraph）复用

**实践任务**（新建 `ch06-langgraph-basics/`）：
- [ ] `conditional_edge.py`：按状态字段走不同分支（如：问题分类 → 各自处理节点）
- [ ] `state_reducer.py`：用 reducer 实现多节点结果累加
- [ ] `checkpoint_demo.py`：同一 `thread_id` 两轮 `invoke`，第二轮能看到第一轮的痕迹
- [ ] `human_in_loop.py`：关键节点 `interrupt`，人工确认后恢复执行

**完成标准**：能把现有 `test_langgraph_hello.py` 改造成带条件分支的版本；能解释 checkpointer 解决了什么问题。

### 阶段 7：LangGraph 进阶——多智能体协作与复杂流程管控

**学习目标**：掌握多 Agent 系统的主流拓扑与实现。

**知识点**：
- 三种协作拓扑：**Supervisor（主管分发）** / **Network（自由交接 handoff）** / **顺序流水线**
- 共享 State vs 消息传递的取舍
- 复杂流程管控：并行节点、map-reduce 模式、循环与退出条件、错误兜底节点
- 每个"智能体" = 一个子图节点（内部封装自己的工具与提示词）

**实践任务**（新建 `ch07-multi-agent/`）：
- [ ] `pipeline_agents.py`：三节点流水线（如：大纲 → 写作 → 审校），共享 State 传递稿件
- [ ] `supervisor.py`：Supervisor 节点根据任务分派给"研究员/写手/校对"三个子 Agent
- [ ] `loop_guard.py`：带最大循环次数与兜底退出的迭代改进工作流（生成 → 评价不达标则重写）

**完成标准**：supervisor 示例中，一次任务能被正确分派 ≥2 个子 Agent；能说明三种拓扑各自适用场景。

### 阶段 8：智能体综合实战（三选一）

| 候选项目 | 核心技术点 | 适配本地小模型程度 |
|---|---|---|
| **A. 个人知识库问答助手**（推荐） | RAG + 工具（检索器作为工具）+ 记忆（多轮追问） | ★★★ 检索兜底弥补模型知识不足 |
| B. 多智能体写作流水线 | Supervisor + 顺序子 Agent + 人工审校中断 | ★★ 长文本受 4096 上下文限制 |
| C. 学习路径规划助手 | 条件分支 + 工具 + 结构化输出 | ★★★ 任务拆分后单步都很简单 |

**要求**（新建 `ch08-project/`）：
- 项目化结构：`main.py` + `agents/` + `tools/` + `prompts/`，不再单文件
- README 说明架构图（手画 Mermaid 即可）、运行方式、已知问题
- 至少包含：1 个 LangGraph 图、≥2 个工具、状态持久化、错误兜底
- 验收：换 `.env` 中的模型（如更大参数模型）后项目依然可运行——验证"模型可替换"设计

### 阶段 9：项目总结与展望

- [ ] 复盘文档 `docs/10-summary.md`：9 个阶段各一段"学到了什么 / 踩过什么坑"
- [ ] 小模型 vs 大模型行为差异总结（工具调用、指令遵循、长文本）
- [ ] 后续方向：LangGraph 部署（LangGraph Platform）、评估（evals）、模型微调（本机 Fine-tuning 目录已具备条件）、MCP 协议

## 五、版本注意：LangChain 1.x vs 老教程（0.x）

本仓库安装的是 **langchain 1.x**，网上大量教程基于 0.x，API 已变化（已在本机验证）：

| 0.x 老写法（教程常见） | 1.x 正确写法 |
|---|---|
| `LLMChain(llm=..., prompt=...)` | LCEL：`prompt \| llm \| parser` |
| `ConversationBufferMemory` 等 Memory 类 | 消息列表手动管理 / LangGraph checkpointer（`InMemorySaver`） |
| `initialize_agent` / `AgentExecutor` | `from langchain.agents import create_agent` |
| `langchain.prompts` / `langchain.schema` | 统一收敛到 `langchain_core.prompts` / `langchain_core.messages` |
| Hub 拉取提示词模板 | 直接在代码里写 `ChatPromptTemplate` |

**遇教程 API 报 `ImportError` 时，先查这张表，再查官方文档，不要降级安装 0.x。**

## 六、目录规划

```
learn-langchain/
├── docs/                       # 所有文档（每阶段一篇笔记：02-framework.md … 10-summary.md）
│   └── 01-learning-plan.md     # 本文档
├── test_*.py                   # 阶段 1 脚本（待移入 ch01-basics/）
├── ch02-core/ … ch08-project/  # 各阶段代码目录
├── .env                        # 本地模型配置（不提交）
├── .env.example                # 配置模板
└── CLAUDE.md                   # Claude Code 协作约定
```

## 七、进度跟踪

> 每完成一个实践任务就把 `[ ]` 改成 `[x]`，并在 `docs/` 新增一篇阶段笔记。

- 阶段 1：✅（收尾：脚本归档、固化 requirements.txt）
- 阶段 2~9：未开始

## 八、参考资料

- LangChain 官方文档（1.x）：https://docs.langchain.com/langchain-platform/overview
- LangChain 1.0 发布说明（了解 breaking changes）：https://blog.langchain.com/langchain-v1/
- LangGraph 官方文档：https://docs.langchain.com/langgraph-platform/overview
- LangChain Academy（官方免费课程，LangGraph 部分强烈推荐）：https://academy.langchain.com/
- vLLM 文档（本地推理服务）：https://docs.vllm.ai/
- Qwen3 模型卡：https://huggingface.co/Qwen/Qwen3-0.6B
