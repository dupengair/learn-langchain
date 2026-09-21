# 关闭 Qwen3 thinking 模式 —— 解决 few-shot 脚本回答被截断问题

> 对应脚本:`test_langchain_prompt-fewshot.py`
> 现象:输出几乎全是 `<think>...</think>` 思考内容,正文被截断甚至完全没有
> 结论先行:**推荐在 `ChatOpenAI` 构造函数里加一行 `extra_body`,已在本地服务实测有效**

---

## 1. 问题根因分析

### 1.1 现象复现(实测数据)

对 `test_langchain_prompt-fewshot.py` 同款配置发起"用一句话说明什么是机器学习"的请求:

```
content: <think>
嗯，用户让我用一句话说明什么是机器学习。首先，我需要确定用户的
需求是什么。他们可能是在学习……（思考还没结束，就没了）
usage: completion_tokens=200   ← max_tokens=200 全部被思考耗尽
```

200 个 token 的生成额度**全部**花在了 `<think>` 思考上,正文一个字都没输出。

### 1.2 三个因素叠加导致

| # | 因素 | 位置 |
|---|------|------|
| ① | **Qwen3 系列默认开启 thinking 模式**。其 chat template(Jinja2)里 `enable_thinking` 默认为 `True`,生成时强制先输出 `<think>...</think>` 再输出正文 | 模型自带模板,不由你控制 |
| ② | vLLM 启动脚本**没有** `--reasoning-parser`,thinking 内容不会被分离,而是原样混在 `content` 字段里返回 | `~/shared/LLM/vllm/test_vllm_qwen3-0.6b.py` |
| ③ | 脚本限制 `max_tokens=200`,对 thinking 模型来说太小,思考过程轻易就把它耗光 | `test_langchain_prompt-fewshot.py:24` |

### 1.3 原理:thinking 是模板层的行为,不是模型权重层的

Qwen3 是"混合推理"模型——同一个权重,是否思考由 **chat template 的渲染结果**决定:

- `enable_thinking=True`(默认):模板在 assistant 回复前留空,模型自行生成 `<think>...`
- `enable_thinking=False`:模板直接在 assistant 开头**预填一个空的 `<think>\n\n</think>` 块**,模型看到"思考已结束",就直接输出正文

vLLM 的 OpenAI 兼容服务器支持在**请求体**中通过 `chat_template_kwargs` 向模板传自定义变量,因此不用重启服务、不用改模型,请求级即可切换:

```json
{"chat_template_kwargs": {"enable_thinking": false}}
```

LangChain 的 `ChatOpenAI` 通过 `extra_body` 参数把任意字段附加到请求体,两者对接就是一行配置。

---

## 2. 推荐方案:构造函数加 `extra_body`(已实测 ✅)

### 2.1 修改位置

文件:`test_langchain_prompt-fewshot.py`,第 19~25 行的 `ChatOpenAI(...)` 构造处,**加一个参数即可**:

```python
chat_model = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=MODEL,
    temperature=0.3,        # 随机性：0-1，越小越严谨，越大越有创造力
    max_tokens=200,         # 最大生成 tokens 数，避免生成过长内容
    extra_body={            # 关闭 Qwen3 的 thinking 模式：
        "chat_template_kwargs": {"enable_thinking": False}  # extra_body 里的字段会原样附加进 OpenAI 请求体
    }
)
```

### 2.2 实测效果

```
场景4: 构造函数 extra_body
content: 机器学习是通过算法让计算机从数据中学习规律并做出预测或决策的过程。
usage: completion_tokens=21   ← 相比之前 200 token 还被截断，现在 21 token 完整回答
```

选择这个位置的理由:

- **一处配置,脚本内所有 `invoke` 都生效**,符合单文件教学脚本"改一处、看全局"的习惯
- 不触碰提示词本身,few-shot 模板保持纯净
- 需要重新启用 thinking 时删掉这个参数即可,可逆性最好

---

## 3. 备选方案对比(同样实测过)

### 方案 A:调用级 `.bind(extra_body=...)` ✅

只对绑定了参数的那次调用生效,适合"同一个模型实例,有的调用要思考、有的不要"的场景:

```python
# 5. 调用模型生成结果（只对这一次调用关闭 thinking）
result = chat_model.bind(
    extra_body={"chat_template_kwargs": {"enable_thinking": False}}
).invoke([{"role": "user", "content": formatted_prompt}])
```

实测:`completion_tokens=21`,回答完整干净。注意 `bind()` 返回的是一个新对象,原 `chat_model` 不受影响。

### 方案 B:提示词软开关 `/no_think` ⚠️ 不推荐

Qwen3 训练时内置的软开关,在提示词末尾加 `/no_think`:

```python
result = chat_model.invoke([{"role": "user", "content": formatted_prompt + "\n/no_think"}])
```

实测:0.6B 模型这次"碰巧"遵循了,回答完整,但 `content` 开头**残留了空 think 标记**:

```
<think>

</think>

机器学习是一种通过数据训练模型，使系统能够自动学习并做出预测或决策的计算机科学技术。
```

缺点:可靠性依赖模型自觉(小模型经常忽略)、污染 `content`(需要后处理剥标记)、还混入了用户提示词。仅适合快速实验。

### 方案 C:服务端 `--reasoning-parser qwen3` ℹ️ 解决"污染"但不解决"截断"

在 `test_vllm_qwen3-0.6b.py` 的 `args` 里加:

```python
"--reasoning-parser", "qwen3",   # 把 <think> 内容解析到 reasoning_content 字段
```

效果:thinking 仍会发生,但 vLLM 把它分离到响应的 `reasoning_content` 字段,`content` 只剩正文——LangChain 读的 `result.content` 就是干净的。

**但注意**:thinking token 依然计入 `max_tokens=200`,截断问题依旧存在。适合"想保留思考过程、只是不想让它混进正文"的需求;和本方案(真正关掉思考)用途不同。

### 方案对比总表

| 方案 | 改动位置 | thinking 是否发生 | content 是否干净 | 是否解决截断 | 推荐度 |
|------|---------|:---:|:---:|:---:|:---:|
| 构造函数 `extra_body` | fewshot 脚本 1 处 | ❌ 关闭 | ✅ | ✅ | ⭐⭐⭐ 首选 |
| `.bind(extra_body=...)` | 单次调用 | ❌ 关闭 | ✅ | ✅ | ⭐⭐ 按调用切换时用 |
| `/no_think` 软开关 | 提示词 | ❌ 关闭 | ⚠️ 残留空标记 | ✅ | ⭐ 不推荐 |
| `--reasoning-parser qwen3` | vLLM 启动脚本(需重启服务) | ✅ 仍开启 | ✅ 分离到 reasoning_content | ❌ | 按"保留思考"需求选 |

---

## 4. 原理延伸:这条链路是怎么走通的

```
ChatOpenAI(extra_body={...})
   └→ langchain-openai 把 extra_body 合并进 OpenAI SDK 的请求体
        └→ POST /v1/chat/completions
            body: {"model": "qwen3-0.6b",
                   "chat_template_kwargs": {"enable_thinking": false}, ...}
             └→ vLLM 用这些 kwargs 渲染模型的 chat template（Jinja2）
                  └→ enable_thinking=False 时模板预填 <think>\n\n</think>
                       └→ 模型直接输出正文，不再生成思考
```

理解要点:

1. **`extra_body` 是通用逃生通道**——OpenAI SDK / LangChain 没有为每个推理引擎的私有参数都定义字段,任何非标准参数都通过它透传。以后遇到 vLLM、SGLang 等私有参数,都是这个套路。
2. **`chat_template_kwargs` 是 vLLM 的约定**,不是 OpenAI 协议标准字段;换推理引擎(如 Ollama、llama.cpp)时关闭思考的参数名会不同。
3. 如果未来换成**非混合推理模型**(如 Qwen2.5),这个参数会被模板忽略、无害;换回 thinking 模型时删掉参数即可恢复。

---

## 5. 验证环境记录

- 验证脚本:临时脚本(运行后已删除),未改动任何项目源码
- 服务:`python test_vllm_qwen3-0.6b.py`(端口 8000,`qwen3-0.6b`,max_model_len 4096)
- 参数:`temperature=0.3, max_tokens=200`,与 fewshot 脚本一致
- 结果:见上文各场景实测输出,方案 A / 推荐方案均以 21 token 完整回答
