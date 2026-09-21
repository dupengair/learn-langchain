# chatbot-mem 计算错误分析 —— 工具失效 + 表达式提取错误的双重 Bug

> 对应脚本:`test_langchain_chatbot-mem.py`
> 现象:带窗口记忆的计算工具 chatbot,计算结果有错误(如"弟弟比我大3岁"答 7 岁,应为 13 岁)
> 结论先行:**主因是代码实现问题(两层 Bug 叠加),模型能力是次要因素。计算工具实际上从头到尾没有生效过,所有"答对"的题都是 qwen3-0.6b 自己心算的。**
> 本文所有结论均来自实际运行(本地 vLLM + qwen3-0.6b 在线环境),复现脚本在文末。

## 1. 结论速览

| 编号 | 问题 | 层级 | 严重性 |
|------|------|------|--------|
| **Bug A** | `calc_tool.run(表达式)` 永远返回空字符串,工具从未产生过计算结果 | 代码 | **致命** |
| **Bug B** | 正则删字提取表达式,把不相干的数字拼接成错误算式(`3加5`→`35`) | 代码 | 严重(被 Bug A 掩盖) |
| **Bug C** | 意图检测词太宽("多少"/"="),非计算问题误触发工具路径 | 代码 | 次要 |
| Bug D | `enhanced_input` 里把 `chat_history` 以 Python repr 形式塞入,是噪音且与模板注入重复 | 代码 | 次要 |
| 因素 E | 0.6B 模型裸算:简单题对、带记忆/语义推理的题错(10-3=7 而非 10+3=13) | 模型 | 次要(工具修好后基本消除) |

**错误掩盖机制(本案最有趣的地方)**:Bug A 导致工具返回空 → Bug B 拼接出的错误算式算了个寂寞 → LLM 拿到"计算结果:(空)",只能自己心算 → 简单题(3+5×2=13)恰好蒙对,制造了"大部分正常"的假象 → 直到遇到"弟弟比我大3岁"这种需要记忆+语义推理的题,模型心算错误才暴露问题。

---

## 2. 实测复现记录

用管道喂入 5 条消息运行原脚本(`printf '...\n' | python3 test_langchain_chatbot-mem.py`):

| 输入 | 原脚本实际回答 | 判定 |
|------|--------------|------|
| 计算 3+5*2 等于多少 | "3 + 5 × 2 等于 13" | ✅ 对(但其实是模型心算的,见 Bug A) |
| 3加5等于多少 | "3加5等于8" | ✅ 对(模型心算;工具实际提取出 `35`) |
| 12÷4等于多少 | "12 ÷ 4 等于 3" | ✅ 对(模型心算;工具实际提取出 `124`) |
| 我今年10岁 | 正常回答 | ✅ |
| 我弟弟比我大3岁,弟弟今年多少岁? | "**7岁**" | ❌ **错,应为 13 岁**(10+3,模型做了 10-3) |

## 3. 逐层定位

### 3.1 Bug A(致命):`PythonREPLTool.run(表达式)` 永远返回空

直接调用工具实测:

```
run('3+5*2')      -> ''
run('35')         -> ''
run('124')        -> ''
run('print(3+5*2)') -> '13\n'     ← 只有 print 才有输出!
```

看安装的 `langchain_experimental/utilities/python.py` 里 `PythonREPL.run` 的文档字符串:

> *"Run command with own globals/locals and **returns anything printed**."*

`PythonREPL` 只捕获 **print 到 stdout 的内容**。直接传表达式 `3+5*2`,Python 会求值但什么都不打印,stdout 为空,返回 `''`。

**后果链**:第 74 行提取出 `calc_expr` → 第 78 行 `calc_tool.run(calc_expr)` 返回 `''` → `enhanced_input` 里"计算过程/结果:"后面是空的 → LLM 无结果可参考,全程裸算。

**这就是为什么"3+5×2=13"看起来是对的**——那不是工具算的,是 0.6B 恰好会这么简单的算术。换一道大数题就能让它翻车:0.6B 心算 `738 × 216` 这种题不可靠,而修复后工具给出 `159408`,它能正确复述(见第 4 节)。

### 3.2 Bug B(严重):`re.sub` 删字提取,把不相干的数字拼接起来

第 74 行:

```python
calc_expr = re.sub(r"[^\d\+\-\*\/\(\)\.]", "", user_input)
```

这个正则的含义是"删掉所有不是 数字/`+ - * /`/括号/点 的字符",**保留集里没有 `×`、`÷`,更没有中文运算词**。删掉字符后,两边的数字会直接**拼接成一个多位数**:

| 用户输入 | 提取结果 | 应该是 | 工具若生效会算出 |
|---------|---------|--------|----------------|
| `3加5等于多少` | `35` | `3+5` | 35 ❌ |
| `12÷4等于多少` | `124` | `12÷4` | 124 ❌ |
| `100 除以 4 是多少` | `1004` | `100/4` | 1004 ❌ |
| `帮我求 25 的平方` | `25` | `25**2` | 25 ❌(平方语义丢失) |
| `我弟弟比我大3岁...` | `3` | (10+3,语义计算) | 3 ❌ |

实测确认这些错误表达式确实被送进了 `calc_tool.run()`——只是因为 Bug A,它们都返回了空字符串,**两个 Bug 恰好互相掩盖**。只修 Bug A 不修 Bug B 的话,上面所有题都会变成"理直气壮的错答"(比如"3加5等于35"),比现在更糟。**两个必须一起修。**

### 3.3 Bug C(次要):意图检测误触发

第 67 行的 `calc_pattern` 含 `多少`、`=`、`计算` 等宽泛词。"我弟弟今年多少岁?"会被判定为"需要计算",提取出 `3` 走工具路径。即使 A、B 修好,这句也会算出"弟弟 3 岁"再交给模型——错误结果反而会带偏模型。**修复思路:提取不出"含运算符的合法表达式"就不走工具路径,降级为普通问答**(校验逻辑顺带解决了这个问题,见 4.1 的 `has_op` 判断)。

### 3.4 Bug D(次要):`enhanced_input` 里的历史噪音

第 86 行把 `chat_history`(Python 消息对象列表)直接 f-string 进提示词,产生 `[HumanMessage(content='...', additional_kwargs={}, ...)]` 这种 repr 噪音;而且 prompt 模板的 `MessagesPlaceholder` 已经注入过格式化好的历史了,**重复且有害**——对 0.6B 这种小模型,噪音会挤占注意力。直接删掉 `同时参考历史对话:{chat_history}` 这段即可,历史由模板负责。

### 3.5 因素 E(模型能力,次要但真实)

- 修复工具后实测:0.6B **复述工具结果完全没问题**,连 `738 × 216 = 159,408` 都能正确表达(见 4.2)。所以"模型不行"不是主因。
- 真正属于模型能力问题的只有一类:**需要结合记忆做语义推理的计算**("我今年10岁" + "比我大3岁" → 10+3)。这种题工具救不了(表达式里没有"10",它在上一轮记忆里),正则也救不了,0.6B 心算不可靠——这正是第 5 节 Function Calling 方案要接管的场景。

---

## 4. 修复方案(请手动修改源码,对照学习)

### 4.1 方案 A:最小修复(保持现有"正则判断 + 工具"架构)

只需改 `judge_and_calc` 函数内部,共三处:

```python
def judge_and_calc(inputs):
    user_input = inputs["input"]
    chat_history = inputs["chat_history"]

    # 【修复 B】中文运算词/符号归一化,再按 token 提取(不再用 re.sub 删字拼接)
    def extract_expression(text):
        for cn, sym in [("乘以", "*"), ("除以", "/"), ("加上", "+"), ("减去", "-"),
                        ("乘", "*"), ("除", "/"), ("加", "+"), ("减", "-"),
                        ("×", "*"), ("÷", "/")]:
            text = text.replace(cn, f" {sym} ")
        tokens = re.findall(r"\d+\.?\d*|[+\-*/()]", text)
        return " ".join(tokens)

    calc_expr = extract_expression(user_input)
    # 【修复 C】提取结果必须含运算符才算可计算,否则直答(顺带解决误触发)
    has_op = bool(re.search(r"[+\-*/]", calc_expr))

    if has_op:
        try:
            # 【修复 A】用 print() 包装,PythonREPL 只捕获 print 出来的内容
            calc_result = calc_tool.run(f"print({calc_expr})").strip()
        except Exception as e:
            calc_result = f"计算出错:{str(e)}"

        # 【修复 D】不再把 chat_history 塞进来,历史由 MessagesPlaceholder 负责
        enhanced_input = f"""用户问题:{user_input}
计算过程/结果:{calc_expr} = {calc_result}
请结合计算结果,用简单易懂的语言回答用户问题。"""
        inputs["input"] = enhanced_input
    return inputs
```

同时第 67 行的 `calc_pattern` 判断可以简化甚至删掉——`has_op` 已经承担了"是否需要计算"的判断。

**修复后实测效果**(在同一环境验证过):

| 输入 | 原逻辑 | 修复后 |
|------|--------|--------|
| `3+5*2` | expr=`3+5*2`,run()→`''`(空) | expr=`3 + 5 * 2`,run()→`13` ✅ |
| `3加5` | expr=`35`,run()→`''` | expr=`3 + 5`,run()→`8` ✅ |
| `12÷4` | expr=`124`,run()→`''` | expr=`12 / 4`,run()→`3.0` ✅ |
| `100 除以 4` | expr=`1004`,run()→`''` | expr=`100 / 4`,run()→`25.0` ✅ |
| `738 * 216`(端到端) | 模型裸算,大数不可靠 | 模型答"159,408" ✅(正确复述工具结果) |
| `25 的平方` | expr=`25` | 无运算符→不调用工具,直答(幂运算语义仍需方案 B) |

### 4.2 方案 B(进阶,正解):Function Calling

方案 A 修得好大数题,但**正则提取有天花板**:`25 的平方`、`弟弟比我大3岁` 这类语义级计算,再精细的正则也写不出表达式。正确方向是让 **LLM 自己生成表达式参数**,代码只负责执行。完整实现见第 5 节(含实测调优过程)。

---

## 5. 方案 B 完整实现:Function Calling 版计算助手(推荐)

### 5.1 设计思想:分工反转

| | 方案 A(正则) | 方案 B(Function Calling) |
|---|---|---|
| 表达式从哪来 | 代码用正则从原句里"抠" | **模型理解语义后自己写**(`"100 除以 4"` → `"100 / 4"`,连 `10+3` 这种跨轮推理都能写) |
| 谁决定要不要算 | 正则关键词匹配 | 模型自主决定(生成 `tool_calls` 就是"要算") |
| 谁执行 | PythonREPL | 我们提供的 `@tool` 函数(eval) |
| 瓶颈 | 正则写不全(语义题无解) | 模型太小会"懒得"调工具(见 5.3,有对策) |

核心消息协议(tool calling 的完整一轮,值得记住):

```
Human("帮我算 100 除以 4")
  → AIMessage(tool_calls=[calculator("100/4")])   # 模型不再输出文字,而是"点菜"
  → ToolMessage("25.0", tool_call_id=...)          # 代码执行后回填,靠 id 配对
  → AIMessage("100 除以 4 等于 25。")               # 模型拿着结果组织自然语言
```

另外本版**不再使用 `RunnableWithMessageHistory`**:它包装的是"单次 LLM 调用链",而工具循环里的消息由我们自己管理更直白;窗口记忆退化为一个普通列表,每轮只存"干净"的两条消息(见 5.4 第 5 段的坑)。

### 5.2 环境实测:`tool_choice` 三种取值(决定架构的关键发现)

先确认"能不能强制模型必须调工具":

| `bind_tools` 的 tool_choice | 实测结果 |
|---|---|
| `"auto"`(或不传) | ✅ 正常工作,模型能生成 `calculator({'expression': '738 * 216'})` |
| `"required"` / `"any"` | ❌ 当前 vLLM 直接 400 报错(JSON 校验错误,不支持) |
| `"calculator"`(指定函数名) | ❌ 被静默忽略,返回空回复 |

**结论:当前 vLLM 版本只支持 auto 模式,没有"强制"开关**——工具调不调全看模型心情,而 0.6B 恰恰不爱调。所以 5.3 的提示工程不是可选项,是必需品。

### 5.3 提升触发率:消息级 few-shot 示例注入(五种配置实测矩阵)

0.6B 直接上 auto 模式,7 条测试用例只有 1 条真正调用了工具,`738 × 216` 心算错成 159,768。**对策:在 system 之后手工插入一轮"标准示范"(Human → AI(tool_calls) → Tool → AI),让模型照猫画虎**。同一环境下五种配置的实测对比:

| 配置 | 大数 738×216 | 语义(弟弟比我大3岁,应 13) | 记忆回忆(我今年几岁,应 10) |
|------|-------------|--------------------------|---------------------------|
| auto + 无 few-shot + T=0.3 | 159,768 ❌(心算) | 11 ❌ | 10 ✅ |
| auto + 无 few-shot + T=0 | 文本退化 ❌ | 13 ✅ | 10 ✅ |
| **auto + few-shot + T=0.3(推荐)** | **调工具 → 159,408 ✅** | **"10 + 3 = 13 岁" ✅** | 14 ❌(被示例数字锚定) |
| auto + few-shot + T=0 | 调工具 → 159,408 ✅ | 10 ❌ | 拒答 ❌ |

解读:

1. **大数题是唯一有"机制保证"的指标**:模型心算 738×216 必然不可靠,只有走工具才可能对。few-shot 让它稳定走工具——这是推荐配置的核心价值。
2. few-shot 占了 4 条消息,对 0.6B 本就有限的注意力是负担,记忆回忆题偶有被示例数字(8+6=**14**)锚定的干扰。**这是模型能力边界,不是代码 Bug**;按 `docs/02-hardware-model-guide.md` 升级到 4B+ 底座后,这类问题会基本消失。
3. temperature 保持原脚本的 0.3 即可,实测 T=0 与 few-shot 组合反而更差。

### 5.4 完整代码(整份替换 `test_langchain_chatbot-mem.py`)

```python
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os


# -------------------------- 1. 环境与模型(与原脚本一致) --------------------------
load_dotenv()
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")
MODEL = os.getenv("MODEL")

if not API_KEY:
    raise ValueError("未检测到 API_KEY，请检查 .env 文件是否配置正确")
if not BASE_URL:
    raise ValueError("未检测到 BASE_URL，请检查 .env 文件是否配置正确")
if not MODEL:
    raise ValueError("未检测到 MODEL，请检查 .env 文件是否配置正确")

llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=MODEL,
    temperature=0.3,        # 实测 few-shot + T=0.3 组合最优，不必改 0
    max_tokens=1024,
    extra_body={            # 关闭 Qwen3 thinking 模式（与原脚本一致）
        "chat_template_kwargs": {"enable_thinking": False}
    }
)


# -------------------------- 2. 定义计算工具（Function Calling 核心） --------------------------
@tool
def calculator(expression: str) -> str:
    """计算一个数学表达式并返回结果。输入必须是合法的 Python 数学表达式，例如 3+5*2、(10+3)/2、25**2"""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"表达式 {expression!r} 无法计算: {e}"

# docstring 就是工具说明书：模型靠它决定"何时调、怎么填参数"，务必写清楚
llm_with_tools = llm.bind_tools([calculator])


# -------------------------- 3. 系统提示 + few-shot 示例 --------------------------
SYSTEM_PROMPT = "你是一名友好的个人助手。遇到任何数学计算，必须先调用 calculator 工具，拿到结果后用自然语言回答；非计算问题直接回答。"

# 消息级 few-shot：手工构造一轮"标准"工具调用示范。没有它 0.6B 触发率只有 1/7，加上后大数题稳定走工具
FEWSHOT_EXAMPLES = [
    HumanMessage(content="计算 8 加 6 等于多少"),
    AIMessage(content="", tool_calls=[{
        "name": "calculator",             # 与 @tool 函数名一致
        "args": {"expression": "8+6"},    # 模型要学会"填"的参数
        "id": "example_call_1",           # 调用 ID，ToolMessage 靠它配对
        "type": "tool_call",
    }]),
    ToolMessage(content="14", tool_call_id="example_call_1"),  # 工具结果回填
    AIMessage(content="8 加 6 等于 14。"),                      # 拿到结果后的最终话术
]


# -------------------------- 4. tool-calling 循环（单轮对话内） --------------------------
def run_turn(history_msgs, user_input, max_tool_rounds=3):
    """一次完整对话轮：模型要点工具 → 代码执行 → 回填 → 模型再生成……直到输出自然语言"""
    msgs = ([SystemMessage(content=SYSTEM_PROMPT)]
            + FEWSHOT_EXAMPLES
            + history_msgs
            + [HumanMessage(content=user_input)])

    response = llm_with_tools.invoke(msgs)
    for _ in range(max_tool_rounds):           # 圈数保护，防模型无限点菜
        if not response.tool_calls:            # 没有点菜 = 最终回答已生成
            break
        msgs.append(response)                  # 协议要求：带 tool_calls 的 AIMessage 必须先入列
        for tc in response.tool_calls:
            result = calculator.invoke(tc["args"])          # 真正执行工具
            print(f"    [调用工具] {tc['name']}({tc['args']}) -> {result}")
            msgs.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
        response = llm_with_tools.invoke(msgs)  # 拿着工具结果再问模型
    return response


# -------------------------- 5. 窗口记忆（只存"干净"的对话轮） --------------------------
WINDOW_SIZE = 2   # 保留最近 2 轮（每轮 = 用户 + 助手最终回答）
window_memory_store = {}

def get_window_history(session_id: str) -> list:
    if session_id not in window_memory_store:
        window_memory_store[session_id] = []
    return window_memory_store[session_id]

def append_to_window(session_id: str, human_msg, ai_msg):
    history = get_window_history(session_id)
    history.append(human_msg)
    history.append(ai_msg)
    if len(history) > 2 * WINDOW_SIZE:
        window_memory_store[session_id] = history[-2 * WINDOW_SIZE:]
    # 关键坑：中间过程的 AIMessage(tool_calls) / ToolMessage 不进窗口。
    # 若历史里出现"声明了 tool_calls 却没有对应 ToolMessage"的孤立消息，下次请求会直接报错


# -------------------------- 6. 多轮对话测试（与原脚本一致） --------------------------
if __name__ == "__main__":
    session_id = "student_001"
    print("===== Function Calling 版数学计算智能助手 =====")
    print("支持：多轮对话（窗口 2 轮）、模型自主调用计算工具")
    print("输入'退出'结束对话\n")

    while True:
        user_input = input("你：")
        if user_input in ["退出", "quit", "q"]:
            print("助手：再见！有问题随时问我～")
            break

        response = run_turn(get_window_history(session_id), user_input)
        answer = response.content or "（已完成工具调用，但没有生成文字回答）"
        print(f"助手：{answer}\n")

        append_to_window(session_id,
                         HumanMessage(content=user_input),
                         AIMessage(content=answer))
```

### 5.5 实现细节里值得学的三个点

1. **消息协议是硬约束**:`AIMessage(tool_calls)` 和 `ToolMessage(tool_call_id)` 必须严格配对成对出现,执行工具后必须回填再请求,这是 OpenAI tool calling 协议,vLLM/任何后端都一样。
2. **窗口记忆存"结论"而非"过程"**:只把最终问答存入窗口,中间的工具往返不存——既省 token,又避免截断出孤立 tool 消息。对比原脚本用 `RunnableWithMessageHistory` 自动存,这里手动管理反而把协议看得更清。
3. **`max_tool_rounds` 保护**:模型理论上可以连续点菜(先算 10+3 再算 ×2),循环允许它多轮,但必须设上限防止失控。

### 5.6 延伸:下一步往哪走

这份手写循环就是 ReAct Agent 的雏形。理解它之后,可以直接学 LangChain 1.x 的 `create_agent` / LangGraph 的 `create_react_agent`——框架帮你处理了循环、并行工具调用、checkpointer 持久化记忆。手写版的意义在于:出问题时你知道框架在帮你做什么。

---

## 6. 验证清单(手动改完后跑这几条)

```bash
printf '计算 3+5*2 等于多少\n3加5等于多少\n12÷4等于多少\n帮我算一下 738*216 等于多少\n我今年10岁\n我弟弟比我大3岁,弟弟今年多少岁?\n退出\n' | python3 test_langchain_chatbot-mem.py
```

方案 A 与方案 B 的对照期望(738×216 一题最能区分两版差异):

| 用例 | 方案 A 期望 | 方案 B 期望(实测) |
|------|------------|------------------|
| 3+5*2 | 13(工具算) | 13(可能走工具也可能心算对) |
| 3加5 | 8(不再变 35) | 8 |
| 12÷4 | 3(不再变 124) | 3 |
| **738*216** | **159,408(工具保证)** | **[调用工具] 738*216 → 159,408 ✅** |
| 我今年10岁 | 不误触发工具 | 直答,不调工具 |
| 弟弟比我大3岁 | 直答(可能仍错,模型边界) | "10 + 3 = 13 岁"(实测 ✅) |
| 我刚才说自己今年几岁 | 窗口记忆正常 | 可能被 few-shot 示例数字干扰(0.6B 边界,4B+ 可解) |

> 附:本文分析的复现/验证脚本在 `/tmp/diag_calc.py`(中间值诊断)、`/tmp/fix_verify.py`(方案A对比)、`/tmp/toolcall_test.py`(tool calling 探测)、`/tmp/toolchoice_diag.py`(tool_choice 取值)、`/tmp/planb_verify.py` ~ `planb5_verify.py`(方案B各配置对比),重启会丢失,如需留存可拷入本仓库。

## 7. 一句话总结

**不是模型"算错"了——是工具压根没算,模型一直在裸算替工具背锅。方案 A(`print()` 包装 + 归一化提取)把工具修活,方案 B(Function Calling)让语义理解回归模型;当前 vLLM 只支持 auto 模式,配一条 few-shot 示范消息就能把 0.6B 调教到"大数必走工具"——剩下的记忆偶错是 0.6B 的能力边界,升 4B 底座即可根治。**
