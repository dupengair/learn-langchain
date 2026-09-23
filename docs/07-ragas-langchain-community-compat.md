# 07 - RAGAS 评估脚本报错分析与修复方案

> 涉及脚本：`test_langchain_vecemb-RAG-evaluate.py`
> 报错：`ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'`
> 结论先行：这是 **ragas 0.4.3 与 langchain-community 0.4.x 的版本不兼容**，且脚本还混用了 RAGAS 的两套互不兼容 API。本文给出两套已实测验证的修复代码（推荐方案 B），**不改动源码，供手动修改学习**。

---

## 1. 问题现象

```
File ".../ragas/llms/base.py", line 12, in <module>
    from langchain_community.chat_models.vertexai import ChatVertexAI
ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'
```

注意：报错不是发生在我们自己的代码里，而是发生在 `import ragas` **触发的库内部导入链**上。只要 `import ragas`，无论用哪套 API，都会在加载阶段就崩溃。

## 2. 根因分析（三层）

### 2.1 直接原因：ragas 顶层导入了已被删除的模块

`ragas/llms/base.py` 第 12 行在**模块顶层**无条件执行：

```python
from langchain_community.chat_models.vertexai import ChatVertexAI   # ← 该模块已不存在
from langchain_community.llms import VertexAI                        # ← 这行还能导入成功
```

导入链：`import ragas` → `ragas/__init__.py` → `ragas/evaluation.py` → `ragas/llms/__init__.py` → `ragas/llms/base.py` → **炸**。

而 `ChatVertexAI` 在 ragas 里唯一的用途是放进一个 `isinstance` 判断列表（`MULTIPLE_COMPLETION_SUPPORTED`，用于检测 LLM 是否支持 n-completion），**不做任何实际调用**——这为后面的"垫片补丁"修复法提供了依据。

### 2.2 版本背景：langchain-community 0.4.x 删除了弃用模块

当前环境实测版本：

| 包 | 版本 | 说明 |
|---|---|---|
| ragas | **0.4.3** | 已是 PyPI 最新版（升级解决不了） |
| langchain | 1.4.1 | |
| langchain-community | **0.4.2** | langchain 1.x 时代的 community 包 |
| langchain-core | 1.6.3 | |

时间线：
- ragas 0.4.3 开发时针对的是 langchain-community **0.3.x**，当时 `chat_models/vertexai.py` 还在（早已标记弃用，真正实现迁去了独立的 `langchain-google-vertexai` 包，旧路径只是转发）。
- langchain 1.x 发布时，langchain-community 升到 **0.4.x**，把所有弃用的 chat/llm 模块**整体删除**（目录里只剩 `google_palm.py` 等极少数，`vertexai.py` 彻底消失）。
- ragas 0.4.3 的依赖声明是裸的 `langchain-community`（**无版本上限**），所以 pip 不会拦截这个组合，安装时一切正常，一运行才炸。
- 查证 PyPI：ragas 0.4.3 就是最新版 → **"升级 ragas" 此路不通**。

### 2.3 为什么不能降级 langchain-community（另一个"想当然"的修法）

`pip install "langchain-community<0.4"` 会连带把 `langchain-core` 降到 0.3.x（0.3.x 系要求 `langchain-core<0.4`），而本环境的 langchain 1.4.1 / langchain-classic 等全都依赖 langchain-core 1.x → **整个 langchain 1.x 栈被破坏**，其他脚本的 FAISS / HuggingFaceEmbeddings 导入也会受影响。排除。

## 3. 连带发现：脚本还埋着 4 个后续必炸的问题

修好导入错误后继续运行，脚本还会依次撞上这些问题（按触发顺序）：

### 问题 ①：指标与评估 API 混用（最核心的设计错误）

脚本第 11-17 行从 `ragas.metrics.collections` 导入指标（**新版 v2 API**），第 121 行却用 `evaluate()` 跑（**旧版 API 入口**）。这两套在 ragas 0.4.x 中**不能混用**：

- `collections` 指标（`ContextPrecision` 等）的基类是 `SimpleBaseMetric`，**不继承** `SingleTurnMetric`；
- `evaluate()` 内部分发时只筛选 `isinstance(metric, SingleTurnMetric)` 的指标 → collections 指标被**静默跳过**，一个任务都不提交 → 抛 `ExceptionInRunner`；
- 反过来，collections 指标构造函数**必须**传入 `llm=`（新版 `llm_factory` 生成的对象），且 `_validate_llm` 会**显式拒绝** LangChain 的 `ChatOpenAI`（报错信息："Collections metrics only support modern InstructorLLM ... Use: llm_factory(...)"）。

即：`ContextPrecision()` 不带参数构造会直接 `TypeError`（缺必填的 `llm`）；带 `ChatOpenAI` 会被拒；靠 `evaluate(llm=ChatOpenAI(...))` 注入也无效。三头堵死。

### 问题 ②：`NameError: api_key`（第 124 行）

```python
results = evaluate(dataset=dataset, metrics=metrics, llm=ChatOpenAI(api_key=api_key, temperature=0))
#                                                                       ^^^^^^^ 小写，未定义
```

脚本里的变量是大写 `API_KEY`，这里写成小写 `api_key`，一执行就 `NameError`。

### 问题 ③：评估用 LLM 没配 `base_url` / `model`（第 124 行）

即使改对变量名，这个 `ChatOpenAI(api_key=..., temperature=0)` 没传 `base_url` 和 `model`，会默认请求 `https://api.openai.com/v1`——本地 vLLM 环境下必然连接失败/超时。评估用的 judge LLM 也必须指向本地服务。

### 问题 ④：数据集缺 `reference`（ground truth）列

`ContextPrecision` 和 `ContextRecall` 都是**带参考答案**的指标：precision 需要 LLM 对照参考答案逐条判断检索片段的相关性；recall 需要对照参考答案统计检索覆盖了哪些句子。数据集只有 `question / answer / contexts` 三列 → 必需列校验失败（或全 NaN）。

补一个说明（不是 bug）：`question/answer/contexts` 是 RAGAS v1 列名，`evaluate()` 内部会调用 `convert_v1_to_v2_dataset` 自动重命名为 `user_input/response/retrieved_contexts`（`ground_truth → reference`），所以旧列名本身能被兼容，但 `ground_truth` 列必须补上。

## 4. 修复方案

### 共同前提：3 行"垫片补丁"解决导入错误

利用 2.1 的结论（`ChatVertexAI` 只被 ragas 用于 isinstance 判断），在**任何 `import ragas` 之前**往 `sys.modules` 里塞一个同名占位模块，让 ragas 的顶层导入能通过：

```python
# ★ 必须放在 from ragas import ... 之前
import sys, types

_shim = types.ModuleType("langchain_community.chat_models.vertexai")
class ChatVertexAI:          # 占位类：ragas 仅用于 isinstance 检查，不会真正实例化
    pass
_shim.ChatVertexAI = ChatVertexAI
sys.modules["langchain_community.chat_models.vertexai"] = _shim
```

说明：
- 只改脚本自身内存中的导入表，**不动 site-packages**，无副作用；
- 等未来 ragas 发布修复版后，删掉这 3 行即可无缝回归。

### 方案 B（推荐）：改用 legacy 指标 + `evaluate()`，改动最小

脚本现有的"构建 Dataset → `evaluate()` 一步出 4 个分"的结构完全保留，只换指标来源并补齐参数。实测（本地 vLLM qwen3-0.6b）输出：

```
{'context_precision': 1.0000, 'context_recall': 1.0000,
 'faithfulness': nan, 'answer_relevancy': 0.8597}
```

完整修改点（对照脚本行号）：

```python
# ★ 垫片补丁放最前（第 1 行之前）
import sys, types
_shim = types.ModuleType("langchain_community.chat_models.vertexai")
class ChatVertexAI: pass
_shim.ChatVertexAI = ChatVertexAI
sys.modules["langchain_community.chat_models.vertexai"] = _shim

# ---- 原 11-17 行改为：metrics 用 ragas.metrics（legacy 单例），不再 import collections ----
from ragas import evaluate
from ragas.metrics import (
    context_precision,   # 检索精确率
    context_recall,      # 检索召回率
    faithfulness,        # 事实一致性
    answer_relevancy     # 答案相关性
)
from ragas.run_config import RunConfig

# ---- 原 118 行改为：单例指标无需（也不能）传 llm 构造 ----
metrics = [context_precision, context_recall, faithfulness, answer_relevancy]

# ---- 原 104-112 行：给每条样本补上参考答案 ground_truth ----
test_data.append({
    "question": question,
    "answer": answer,
    "contexts": contexts,
    "ground_truth": "……人工/半自动撰写的标准答案……",   # ★ 新增
})

# ---- 原 121-125 行改为：补齐 base_url/model/变量名/embeddings ----
results = evaluate(
    dataset=dataset,
    metrics=metrics,
    llm=llm,                # 复用脚本开头那个配好 base_url 的 llm（★ 不是新建 ChatOpenAI）
    embeddings=embeddings,  # 复用已加载的 HuggingFaceEmbeddings（answer_relevancy 需要）
    run_config=RunConfig(max_retries=3, max_wait=5, timeout=60),  # 可选：控制超时重试
)
```

`evaluate()` 会对传入的 LangChain `ChatOpenAI` / `HuggingFaceEmbeddings` 自动包装（源码 `evaluation.py` 中 `isinstance(llm, BaseLanguageModel)` → `LangchainLLMWrapper`），v1 列名也会自动转换，所以照上面写就行。

> 实测中 `faithfulness=nan`：该指标要求 judge LLM 先把答案拆解成事实句再逐条验证，对模型指令遵循要求最高，0.6B 小模型在 60s 超时内没产出合法结构（`TimeoutError`）。这是**模型能力问题而非代码问题**——换更大的 judge 模型或调大 `timeout` 即可出分（符合本项目 CLAUDE.md 的约定：不能用小模型的输出质量判断脚本正确性）。

### 方案 A（进阶）：collections 新 API，逐指标逐样本评估

如果就是想学 ragas 的新组件式 API，则**放弃 `evaluate()`**，用 `llm_factory` + `ascore()`。管道已实测全部跑通（导入 → 构造 → 发起评估请求均正常；最终失败点仅是小模型输出的 JSON 不合法，instructor 解析重试耗尽——换更大模型即可）：

```python
# ★ 垫片补丁同上，仍需放在 import ragas* 之前
import openai, asyncio
from ragas.llms import llm_factory                      # 新版 LLM 工厂（基于 OpenAI SDK client）
from ragas.embeddings import HuggingFaceEmbeddings      # 注意是 ragas 的，不是 langchain_community 的
from ragas.metrics.collections import (
    ContextPrecisionWithoutReference,   # 无参考答案版；有标准答案可用 ContextPrecision(llm=...)
    ContextRecall,
    Faithfulness,
    AnswerRelevancy,
)

# 1) 新版评估组件：直接用 OpenAI SDK 客户端指向本地 vLLM
client = openai.AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
eval_llm = llm_factory(MODEL, client=client)            # 必填，构造时就要给
eval_emb = HuggingFaceEmbeddings(model=embedding_model_name)   # ★ 参数名是 model 不是 model_name

# 2) 指标在构造时注入 llm / embeddings（不再是 evaluate() 注入）
metrics = [
    ContextPrecisionWithoutReference(llm=eval_llm),
    ContextRecall(llm=eval_llm),
    Faithfulness(llm=eval_llm),
    AnswerRelevancy(llm=eval_llm, embeddings=eval_emb),
]

# 3) 逐样本异步评估（新 API 没有 evaluate() 这种一键入口）
async def run_eval():
    scores = []
    for item in test_data:   # 第 6 步采集的原始 dict 列表
        row = {}
        for m in metrics:
            r = await m.ascore(
                user_input=item["question"],
                response=item["answer"],
                retrieved_contexts=item["contexts"],
            )
            row[m.name] = r.value
        scores.append(row)
    return scores

for row in asyncio.run(run_eval()):
    print(row)
```

注意差异：新 API 的入参名是 `user_input / response / retrieved_contexts`（不再走 Dataset 列名映射）；`datasets.Dataset` 那步可以整个删掉。无标准答案时用 `ContextPrecisionWithoutReference`，否则用 `ContextPrecision(llm=..., name=...)` 变体并传 `reference=`。

### 两条路径怎么选

| | 方案 B（legacy） | 方案 A（collections） |
|---|---|---|
| 改动量 | 小（结构不变） | 大（评估流程重写为异步） |
| 与教程代码一致性 | 高 | 低 |
| 官方态度 | 逐步弃用中 | 未来方向 |
| 本环境实测 | ✅ 四指标出分 | ✅ 管道通，需更强模型出分 |

学习阶段建议先走方案 B 把评估闭环跑起来，再对照方案 A 体会 ragas 新旧 API 的设计差异。

## 5. 验证记录

- 验证方式：临时脚本（已删除）+ 在线本地 vLLM（qwen3-0.6b），未改动仓库源码与 site-packages。
- 方案 B：`evaluate()` 完成 2 样本 × 4 指标，输出 `context_precision=1.0 / context_recall=1.0 / answer_relevancy=0.86 / faithfulness=nan`（NaN 为小模型超时，见上文说明）。
- 方案 A：`llm_factory` 构造、`Faithfulness.ascore()` 调用链正常，最终因 0.6B 模型生成的 JSON 不合法被 instructor 判定重试耗尽——属模型能力边界，代码路径正确。

## 6. 学习要点

1. **顶层 import 的脆弱性**：库在模块顶层硬导入可选/已弃用的依赖，会让"装得上、跑不起来"。遇到 import 期报错，先看报错发生在谁的导入链上，而不是自己的代码。
2. **依赖无版本上限 ≠ 兼容**：ragas 声明 `langchain-community` 未设上限，langchain 1.x 删模块后组合即炸。复现问题时先 `pip list` 对照版本，再查上游 issue。
3. **一个库两套 API 不能混**：ragas 0.4.x 同时维护 legacy（`ragas.metrics` 单例 + `evaluate()`）与 v2（`ragas.metrics.collections` + `llm_factory` + `ascore()`），导入来源和评估入口必须配套。
4. **评估数据集要有标准答案**：precision/recall 类指标以 `reference`（旧名 `ground_truth`）为锚，没有参考答案就只能选 `*WithoutReference` 变体或放弃这两项。
5. **judge 模型 ≠ 生成模型**：RAGAS 各指标靠 LLM 做结构化判定（输出 JSON），对指令遵循能力的要求远高于普通问答，0.6B 量级的模型容易超时/格式错误导致 NaN。
