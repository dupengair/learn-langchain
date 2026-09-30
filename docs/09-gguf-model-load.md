# test_langchain_hf.py 能否直接加载 GGUF 模型 —— 结论与修改方案

> 对应脚本:`test_langchain_hf.py`(HuggingFacePipeline 加载本地 transformers 格式模型)
> 新模型:`/home/dupengair/shared/LLM/ollama/model/Qwen3.5-4B-q4/Qwen3.5-4B-q4_k_m.gguf`(Q4_K_M 量化,约 2.75GB)
> 结论先行:**原脚本不能直接运行;修改后能运行,推荐改走 Ollama 的 OpenAI 兼容接口(已实测 ✅)**
>
> **端口说明**:本文排查时的实测在 Ollama 默认端口 11434;此后实际部署改为 `export OLLAMA_HOST=127.0.0.1:8000`,让 Ollama 与 vLLM **共用 8000 端口**,`.env` 的 `BASE_URL` 固定为 `http://localhost:8000/v1`,切换后端只改 `MODEL`。文中历史实测数据保留原样,涉及端点处按 8000 理解即可。

---

## 1. 为什么不能直接运行

`test_langchain_hf.py` 的加载链路是:

```
AutoModelForCausalLM.from_pretrained(目录)
   └→ 期望目录里有 config.json + model.safetensors / *.bin
        └→ transformers 原生格式(FP16/BF16 权重)
```

而 GGUF 是 **llama.cpp 的私有格式**:单文件、量化权重、元数据内嵌在文件头部,**没有 `config.json`**。直接把路径改成 GGUF 所在目录(或文件)运行,会在第 10 行就报错:

```
OSError: .../Qwen3.5-4B-q4 does not appear to have a file named config.json
```

这不是路径写法问题,是**格式层不兼容**,换任何路径写法都绕不过去。

## 2. 实测环境事实(2026-09-30)

| 项目 | 状态 |
|------|------|
| GGUF 文件 | `ollama/model/Qwen3.5-4B-q4/Qwen3.5-4B-q4_k_m.gguf` ✅ 存在 |
| Ollama 服务 | `localhost:11434` **正在运行**,且已通过 Modelfile 把该 GGUF 注册为 `qwen3.5-4b:q4`(capabilities: tools / thinking / completion,架构 `qwen35`) |
| transformers | 4.57.6(有 GGUF 反量化能力,但**架构映射里没有 `qwen35`**,只有 qwen2 / qwen3 系列——已实测) |
| gguf(python 包) | 已安装(仅读文件用) |
| llama-cpp-python | **未安装** |

---

## 3. 方案一(推荐,已实测 ✅):走 Ollama 的 OpenAI 兼容接口

### 3.1 为什么推荐

- Ollama **已经在跑这个模型了**,模型 `qwen3.5-4b:q4` 随时可调用,零新增依赖
- Ollama 原生就是吃 GGUF 的,量化原样生效(Q4_K_M 占 2.75GB 显存/内存,而不是反量化后的 ~8.5GB)
- 写法和项目里访问 vLLM 的脚本**完全一致**——都是 `ChatOpenAI` 换个 `base_url`,体现了 OpenAI 兼容接口"换后端不改代码"的意义

### 3.2 修改后的脚本(整文件替换 `test_langchain_hf.py` 第 1~11 行的加载部分)

```python
# 导入模块（Ollama提供OpenAI兼容服务，LangChain用同一个ChatOpenAI类访问）
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 1. 加载.env环境变量（BASE_URL固定 http://localhost:8000/v1，vLLM与Ollama共用8000端口，
#    Ollama经 export OLLAMA_HOST=127.0.0.1:8000 启动；切换后端只改MODEL，本脚本零改动）
load_dotenv()
API_KEY = os.getenv("API_KEY")             # "EMPTY"，本地服务不校验key
BASE_URL = os.getenv("BASE_URL")           # http://localhost:8000/v1
MODEL = os.getenv("MODEL")                 # qwen3.5-4b:q4（ollama list显示的名字，Modelfile创建时的tag）

# 2. 校验配置（沿用项目既有模式）
if not all([API_KEY, BASE_URL, MODEL]):
    raise ValueError("缺少必要配置：API_KEY / BASE_URL / MODEL")

# 3. 初始化ChatOpenAI（和访问vLLM的写法一样，只是base_url和model不同）
llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=MODEL,
    max_tokens=2048,   # thinking模型先思考再回答，额度必须给足，否则正文被截断（见3.3）
)

# 4. 调用模型（invoke接口和HuggingFacePipeline完全一致）
prompt = "请用3句话解释什么是LangChain？"
result = llm.invoke(prompt)

print("Ollama(GGUF)模型回复：")
print(result.content)
```

### 3.3 thinking 模型的坑(实测数据)

Qwen3.5 是 thinking 模型,和 `03-close-qwen3-thinking.md` 里 vLLM 的表现**不完全一样**:

| 行为 | vLLM(qwen3-0.6b,03 文档) | Ollama(qwen3.5-4b,本次实测) |
|------|------|------|
| 思考内容去向 | 混在 `content` 里污染正文 | **自动分离到 `reasoning` 字段**,`content` 干净 ✅ |
| 思考是否耗 token | 耗 | 照样耗(实测一次"回答 ok"花了 150 token 思考;完整回答 LangChain 问题共 1241 输出 token) |
| 关闭方式 | `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` | OpenAI 兼容层用 **`reasoning_effort="none"`**(见第 8 节);原生 `/api/chat` 用 `think: false`。`chat_template_kwargs` 和 `think` 传给兼容端点都会被**静默忽略** |

**实践结论**:不用纠结关闭思考,只要 `max_tokens` 给足(建议 ≥2048),`result.content` 拿到的就是干净正文。这正好验证了 03 文档第 158 行的预判——"换推理引擎时,关闭思考的参数名会不同"。
(注:该结论适用于 ctx 够用的场景;当 `max_tokens` 顶到运行时上下文上限时,"给足额度"反而会稳定复现正文为空,见第 8 节。)

---

## 4. 方案二(本机不可行 ❌):transformers 直接读 GGUF

transformers ≥4.41 支持 `gguf_file` 参数,理论上保持原脚本结构只改两行:

```python
model = AutoModelForCausalLM.from_pretrained(model_dir, gguf_file="Qwen3.5-4B-q4_k_m.gguf")
tokenizer = AutoTokenizer.from_pretrained(model_dir, gguf_file="Qwen3.5-4B-q4_k_m.gguf")
```

但本机实测走不通,两道坎:

1. **架构不支持(硬伤)**:该 GGUF 的架构是 `qwen35`,而 transformers 4.57.6 的架构映射里只有 `qwen2` / `qwen3` 系列(`'qwen35' in CONFIG_MAPPING` 为 False,已实测)。反量化完成后构建模型类时直接报 `Unrecognized architecture`。除非升级 transformers 到支持 Qwen3.5 的版本。
2. **即使架构支持也不划算**:这套机制是把量化权重**反量化**回 FP16 加载——4.2B 参数要占 ~8.5GB 内存、加载要等完整转换,量化带来的体积/速度优势全部丢失。GGUF 的正确打开方式本来就不是 transformers。

## 5. 方案三(可选):llama-cpp-python 直接推理

不走服务、进程内加载 GGUF 的原生方案,LangChain 对应 `langchain_community.llms.LlamaCpp`。但当前环境**未安装**,且 `llama-cpp-python` 需要本地编译(要 GPU 加速还得提前设 `CMAKE_ARGS="-DGGML_CUDA=on"`),安装成本明显高于方案一。既然 Ollama 服务已在跑,没有选它的理由,仅作知识点记录。

## 6. 方案对比总表

| 方案 | 改动量 | 新增依赖 | 量化是否生效 | 本机可用性 | 推荐度 |
|------|:---:|:---:|:---:|:---:|:---:|
| Ollama OpenAI 兼容接口 + `ChatOpenAI` | 换加载部分 3 行配置 | 无(服务已在跑) | ✅ 原生 Q4_K_M | ✅ 已实测 | ⭐⭐⭐ 首选 |
| transformers `gguf_file` 参数 | 改 2 行 | 无 | ❌ 反量化回 FP16 | ❌ `qwen35` 架构不识别 | 本机不可行 |
| `llama-cpp-python` + `LlamaCpp` | 换整个加载+LLM 部分 | 需编译安装 | ✅ | ⚠️ 未安装 | 备选 |

---

## 7. 验证环境记录

- 验证方式:临时脚本(运行后已删除,未改动任何项目源码),`ChatOpenAI(base_url="http://localhost:11434/v1", model="qwen3.5-4b:q4", max_tokens=2048)` 调用"请用3句话解释什么是LangChain?"
- 结果:`result.content` 输出完整、干净的 3 句话正文;`usage_metadata` 显示 output_tokens=1241(其中约 1100 为 thinking 消耗)
- 模型信息来源:`curl http://localhost:11434/api/tags` → family `qwen35`,quantization Q4_K_M,4.2B,context 262144

---

## 8. 补充排查:`test_langchain_hello.py` 在 Ollama 下 `content` 打印为空

> 现象:`test_langchain_hello.py` 改用 Ollama + `qwen3.5-4b:q4` 后,`print(response.content)` 输出空字符串;把 `max_tokens` 从 2048 提到 **4096 反而必然复现**
> 结论先行:**`max_tokens=4096` 撞上了 Ollama 运行时上下文上限 4096,生成被截断在思考阶段;而照搬 vLLM 的 `chat_template_kwargs` 关思考参数被 Ollama 静默忽略,思考无法关闭**。修复:`reasoning_effort="none"`(已实测 ✅)

### 8.1 复现实测(与脚本完全同配置)

```
content:            ''
finish_reason:      length          ← 没生成完,是被掐断的
usage:              input=30, output=4066, total=4096   ← 正好等于 ctx 上限
```

### 8.2 根因:三个因素叠加

| # | 因素 | 证据 |
|---|------|------|
| ① | **Ollama 运行时上下文默认只有 4096**,与模型标称的 262144 无关 | `curl http://localhost:11434/api/ps` → `context_length: 4096`(Ollama 0.35.0) |
| ② | **生成空间 = min(max_tokens, num_ctx − prompt_tokens)**。请求 `max_tokens=4096` 不会扩大 ctx,只会被 clamp 到 ~4066;模型此次思考就超过 4066 token,还没开始写正文就被截断 | `finish_reason=length` + `total_tokens=4096`,思考内容全在 `reasoning` 字段,`content` 从未被写入 |
| ③ | 脚本里照搬 03 文档的 `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` 是 **vLLM 的私有参数,Ollama 兼容层不认识且不报错(静默忽略)**,thinking 全开 | 兼容层传 `think: false` 同样不生效(实测 reasoning 依旧存在) |

**反直觉点**:第 1 版文档说"max_tokens 给足就行",在 ctx 够用时成立;但 `max_tokens` 顶到 ctx 上限时,"给足"恰恰让每次生成都必然撞墙截断。截断若发生在思考阶段,`content` 就是空的。

**为什么排查困难**:LangChain 的 `AIMessage` 不保留 Ollama 响应里的非标准 `reasoning` 字段(`additional_kwargs` 里只有 `refusal: null`),所以"思考去了哪、被截断在哪"在 LangChain 侧完全不可见,看起来就像"模型什么都没输出"。排查时要用 `curl` 直连看原始响应,或看 `finish_reason`。

### 8.3 修复方案(已实测 ✅):`reasoning_effort="none"`

Ollama 0.35.0 的 OpenAI 兼容层支持 OpenAI 标准参数 `reasoning_effort`,且 langchain-openai 1.x 有同名字段,直接加一行:

```python
llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=MODEL,
    max_tokens=4096,
    temperature=0.3,
    reasoning_effort="none",   # 关闭 thinking（OpenAI标准参数，Ollama 0.35 兼容层支持）
    # 删掉原来的 extra_body={"chat_template_kwargs": ...}  ← vLLM 专用，Ollama 忽略
)
```

实测结果:

```
content: '先掌握 Python 基础，再学习机器学习框架。多动手做项目，从预测房价开始，逐步深入理论，保持实践。'
usage: output_tokens=29(对比修复前 4066), finish_reason=stop
```

`max_tokens=4096` 保留也没问题——思考不再消耗额度,29 token 远够。**同时记得删掉无效的 `extra_body`**,避免误导后续维护。

### 8.4 备选方案

| 方案 | 适用场景 | 做法 |
|------|---------|------|
| `reasoning_effort="none"`(推荐) | 要正文不要思考 | 如上,一行改动 |
| 原生 API `think: false` | 直连 `/api/chat` 或改用 `langchain-ollama` 包的 `ChatOllama` 时 | 请求体加 `"think": false`(curl 实测 29 token 出正文;`ChatOllama` 本机未安装) |
| 扩大运行时上下文 | 想保留 thinking(思考+正文都放得下) | Modelfile 里加 `PARAMETER num_ctx 16384` 后 `ollama create qwen3.5-4b:q4-16k -f Modelfile` 创建新 tag;或启动 Ollama 前设环境变量 `OLLAMA_CONTEXT_LENGTH=16384` |
| 调小 `max_tokens` | 临时兜底 | 让 `max_tokens` 明显小于 `num_ctx − prompt`(如 2048),降低撞墙概率,但思考长度不可控,不保证不复现 |

### 8.5 方法论沉淀

1. **`finish_reason=length` + `total_tokens ≈ ctx上限` 是"撞墙"特征**。LangChain 侧看不到 reasoning 时,先看这两个值,再 `curl /api/ps` 核对运行时 `context_length`。
2. **推理引擎的私有参数换引擎就失效,且多为静默忽略**(03 文档预判的第二次验证:`chat_template_kwargs` 是 vLLM 的,`think` 是 Ollama 原生 API 的,`reasoning_effort` 才是两边兼容层通吃的 OpenAI 标准)。
3. **Ollama 的 ctx 要显式配置**:模型文件的 context length 和运行时加载的是两回事,`/api/ps` 看到的才是真实生效值。

### 8.6 补充验证环境记录

- Ollama 0.35.0,`/api/ps` 显示运行时 `context_length: 4096`
- 复现与修复均用临时脚本(运行后已删除),项目源码未改动,结论已同步修正本文 3.3 节的关闭方式表格

---

## 9. 补充排查:`reasoning_effort="none"` 在 vLLM 下 422 报错 —— 按后端条件发送

> 现象:脚本切回 vLLM 后端(`qwen3-0.6b`)时,带 `reasoning_effort="none"` 的请求报 422
> 结论先行:**vLLM 0.10.2 认识该字段但枚举只接受 `low/medium/high`,`"none"` 不在其中**,必须条件发送。落地写法:`reasoning_effort='none' if 'qwen3.5' in MODEL else None`(按模型名区分后端,已批量应用于全部脚本,双后端实测 ✅)

### 9.1 报错原文(实测,vLLM 0.10.2)

```
{"error":{"message":"1 validation error:\n  {'type': 'literal_error',
  'loc': ('body', 'reasoning_effort'),
  'msg': \"Input should be 'low', 'medium' or 'high'\", 'input': 'none', ...}}
```

注意两点机理:

1. **不是"未知字段被忽略"**——vLLM 的请求模型里有 `reasoning_effort` 字段(早期为 gpt-oss 类推理模型加的),pydantic Literal 校验卡在**值**上:`none` 不在 `low/medium/high` 枚举里,直接 422
2. `reasoning_effort="none"` 是 OpenAI 较新引入的值(GPT-5.1 时代,语义"不思考");Ollama 跟进了,vLLM 0.10.2 还没跟进。**换大模型推理引擎时,标准参数的取值范围也要核对,不能只看参数名是否存在**

### 9.2 关键前提(实测):`reasoning_effort=None` 时字段完全不进请求体

```python
ChatOpenAI(model='m', reasoning_effort=None)._default_params   # → 不含 'reasoning_effort' 键
```

这让"条件发送"可以写成一行三元,而不必把构造函数改成 dict 展开重构。

### 9.3 为什么没有"两边通吃"的统一参数

| 参数 | Ollama 兼容层 | vLLM 0.10.2 |
|------|------|------|
| `reasoning_effort="none"` | ✅ 关闭 thinking | ❌ 422(值不在枚举) |
| `reasoning_effort="low"` | ✅(等效关) | ✅ 通过校验,**但对 qwen3 不起作用**(thinking 照开) |
| `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` | ❌ 静默忽略 | ✅ 关闭 thinking(03 文档方案) |

实测确认:vLLM 传 `low` 只是不报错,生成的 content 照样以 `<think>` 开头。**两个引擎关思考的参数互不相认,分支不可避免**。

### 9.4 方案选型(三个候选,均验证过技术可行性)

| 方案 | 写法 | 优点 | 缺点 |
|------|------|------|------|
| A. 服务路由探测 | 启动时 GET `{root}/api/tags`:Ollama 200 / vLLM 404 | 真环境检测,与模型名无关;服务未启动时报错清晰 | 每脚本 +6 行,依赖 requests |
| **B. 模型名判断(已采用)** | `'none' if 'qwen3.5' in MODEL else None` | 一行改动,零网络请求,符合教学脚本直白风格 | 按命名约定推断后端,非真检测 |
| C. `.env` 开关 | `os.getenv('REASONING_EFFORT') or None` | 配置层解决 | 违背"切换后端只改 MODEL"的约定,多维护一个变量 |

选 **B** 的理由:qwen3.5 系列 GGUF 模型只会在 Ollama 后端运行、qwen3-0.6b 只会在 vLLM 后端运行,模型名与后端在本项目里天然一一对应;且切换后端本来就要改 `MODEL`,判定条件随之自动切换,零额外维护成本。

### 9.5 落地写法(2026-09-30 起全部脚本的标准形态)

```python
llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=MODEL,
    max_tokens=1024,
    temperature=0.3,
    reasoning_effort='none' if 'qwen3.5' in MODEL else None,  # qwen3.5系列(GGUF)只在Ollama后端运行；vLLM不认none值会422，None=不发送该字段
)
```

### 9.6 双后端端到端实测(同一份判定逻辑)

| 场景 | MODEL | 判定结果 | 实测 |
|------|-------|---------|------|
| vLLM(8001 临时实例) | `qwen3-0.6b` | None → 不发送 | ✅ 无 422,正常返回;content 带 `<think>` 前缀(vLLM 无 reasoning-parser,thinking 混在正文里,03 文档所述既有行为,与本次改动无关) |
| Ollama(8000) | `qwen3.5-4b:q4` | 发送 `"none"` | ✅ finish=stop,content 干净 |

### 9.7 遗留说明

- vLLM 后端如需关 thinking,手动给 `ChatOpenAI` 加回 03 文档的 `extra_body={"chat_template_kwargs": {"enable_thinking": False}}`(对 qwen3-0.6b 有效)
- 若未来升级 vLLM 到支持 `"none"` 枚举的版本,三元条件可整体简化回固定 `reasoning_effort="none"`
- 备选的探测方案 A 代码已验证可用,如后续后端与模型名的对应关系被打破(如 vLLM 也部署 qwen3.5),按 9.4 表切到方案 A

---

## 10. 架构定稿:`llm_config.py` 共享工厂 + `.env` 显式声明后端(2026-09-30)

> 背景:第 8、9 节的方案按模型名条件判断,存在两个问题——新增模型要改所有脚本的判断串;vLLM 侧关 thinking 的 `extra_body` 处于注释状态,切回 vLLM 需手动改代码
> 结论先行:**新建 `llm_config.py` 集中后端适配逻辑,`.env` 用 `BACKEND` 显式声明后端、`MODEL_OLLAMA`/`MODEL_VLLM` 分开定义模型名,切换后端只改 `.env` 一行,脚本零改动**(已批量应用于全部脚本,双后端端到端实测 ✅)

### 10.1 方案演进回顾(为什么走到这一步)

| 版本 | 关 thinking 方式 | 切后端要做什么 | 缺陷 |
|------|------|------|------|
| v1(03 文档时代) | vLLM `extra_body` 写死 | 改 `.env` 的 MODEL | 换 Ollama 后参数被静默忽略,thinking 全开 |
| v2(第 8 节) | `reasoning_effort="none"` 写死 | 改 `.env` 的 MODEL | vLLM 下 422(第 9 节) |
| v3(第 9 节) | `'none' if 'qwen3.5' in MODEL else None` | 改 `.env` 的 MODEL | 新模型名要同步改全部脚本;vLLM 关 thinking 失效需手动改代码 |
| **v4(本节,定稿)** | `llm_config.py` 按后端分支 | **只改 `.env` 的 `BACKEND` 一行** | — |

演进的本质:**"后端适配"从脚本文本里的片段,收敛为集中管理的一层**。前三个版本每次逻辑变更都要批量重写 37 个脚本,v4 之后逻辑变更只动 `llm_config.py` 一个文件。

### 10.2 最终形态

**`.env`**(完整模板见 `.env.example`):

```bash
BACKEND="ollama"              # 当前推理后端: ollama / vllm(切换只改这一行)
BASE_URL="http://localhost:8000/v1"   # 两后端共用 8000 端口
MODEL_OLLAMA="qwen3.5-4b:q4"  # ollama list 里注册的 tag
MODEL_VLLM="qwen3-0.6b"       # vLLM 的 served-model-name
```

**脚本侧**(初始化从 ~15 行缩到 3 行,环境校验由工厂内部完成):

```python
from llm_config import get_chat_model

llm = get_chat_model(temperature=0.3, max_tokens=1024)
```

**`llm_config.py` 的职责**(约 60 行,完整中文注释):读 `BACKEND` → 校验合法性(非法值抛 ValueError)→ 取对应 `MODEL_XXX` → 按后端拼装关 thinking 参数(ollama 用 `reasoning_effort="none"`,vllm 用 `extra_body`)→ 返回配置好的 `ChatOpenAI`。

### 10.3 设计要点

1. **`BACKEND` 是唯一后端事实源**:比按模型名猜测语义清晰,且与"切换后端本来就改 `.env`"的操作流天然一致。默认 `ollama`。
2. **`MODEL` 派生而非散落**:模型名按后端分开定义(`MODEL_OLLAMA`/`MODEL_VLLM`),新增模型只改 `.env`,任何脚本都不用动。
3. **依赖 dotenv 的环境变量(`MODEL_PATH` 等)必须在 `get_chat_model()` 调用之后读取**——工厂内部先执行 `load_dotenv()`,调用前读取会拿到 None。本次改造中 2 个 vecemb 脚本的 `MODEL_PATH` 读取+校验已自动移植到构造行之后。
4. **环境变量可覆盖 `.env`**:`load_dotenv()` 默认不覆盖已存在的环境变量,因此 `BACKEND=vllm BASE_URL=http://localhost:8001/v1 python3 test_xxx.py` 可临时切换后端测试,不动 `.env` 文件。
5. **双构造脚本天然兼容**:`core_llm`/`fallback_llm` 等多个实例都是各自一行 `get_chat_model(...)`,参数差异照旧保留。

### 10.4 定稿验证记录(2026-09-30)

| 验证项 | 结果 |
|------|------|
| 37 个含 `ChatOpenAI` 的脚本批量改造(含此前遗漏关 thinking 参数的 4 个:chatbot / chatbot-rounds / prompt-temp / env) | ✅ 全部语法通过,无 `ChatOpenAI` 直连残留 |
| Ollama(8000):hello(单构造) / chainerr-fallback(双构造) / vecemb-RAG-qa(MODEL_PATH+embedding+FAISS) | ✅ 全部正常输出,RAG 正确引用知识库 |
| vLLM(8001 临时实例):`BACKEND=vllm` 环境变量覆盖后跑 hello | ✅ 无 422,content 无 `<think>` 前缀(extra_body 关 thinking 生效),模型为 `MODEL_VLLM` |
| 非法 `BACKEND=sglang` | ✅ 抛 `ValueError: BACKEND 只支持 ollama / vllm` |
| 环境覆盖机制:`BACKEND`/`BASE_URL` 环境变量优先于 `.env` | ✅ 利用 `load_dotenv()` 不覆盖已有环境变量的默认行为 |

测试用 vLLM 实例已关闭删除,未影响常驻的 Ollama 服务。

### 10.5 收尾清理:脚本内冗余的 `load_dotenv`(2026-09-30)

工厂内部已执行 `load_dotenv()`,脚本层的重复调用全部移除。删除前逐脚本判定过必要性:

| 类别 | 数量 | 判定与处理 |
|------|:---:|------|
| 脚本内已无任何 `os.getenv` | 35 | `load_dotenv()` + `from dotenv import load_dotenv` 纯冗余,删除;悬空的 `import os` 一并删除 |
| 有 `MODEL_PATH = os.getenv(...)` 但位于工厂调用**之后** | 2(vecemb-RAG-qa / RAG-evaluate) | dotenv 已由工厂内部加载,`load_dotenv()` 冗余删除;`import os` 保留(MODEL_PATH 仍需要) |
| `os.getenv` 在工厂调用之前(删了会拿到 None) | 0 | 无此情况 |
| 纯 embedding 脚本(无工厂调用,自行读 env) | 4(faissretrive / mmr / search / textvec) | `load_dotenv` 是必要的,**保留不动** |

清理后 `load_dotenv` 全项目仅存于 `llm_config.py` 与上述 4 个纯 embedding 脚本。实测:env.py / hello.py / vecemb-RAG-qa 均正常。
