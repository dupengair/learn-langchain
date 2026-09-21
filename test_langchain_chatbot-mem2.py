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