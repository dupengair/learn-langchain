from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_openai import ChatOpenAI  # 补充LLM定义
from langchain_experimental.tools import PythonREPLTool    # 数学计算工具
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import re
import os


# -------------------------- 1. 环境 --------------------------
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
    temperature=0.3,        # 随机性：0-1，越小越严谨，越大越有创造力
    max_tokens=1024,        # 最大生成 tokens 数，避免生成过长内容
    extra_body={            # 关闭 Qwen3 的 thinking 模式：
        "chat_template_kwargs": {"enable_thinking": False}  # extra_body 里的字段会原样附加进 OpenAI 请求体
    }
)


# 初始化数学计算工具（PythonREPL）
calc_tool = PythonREPLTool()
# 窗口记忆大小：保留最近2轮对话（每轮=用户+助手消息）
WINDOW_SIZE = 2


# -------------------------- 2. 定义提示词模板（适配窗口记忆+工具调用） --------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一名友好的个人助手，规则如下：
    1. 能记住最近{window_size}轮对话内容，用简单语言解答问题；
    2. 如果问题包含数学计算（如加减乘除、公式、数值运算），先调用计算工具得到结果，再用自然语言解释；
    3. 非计算问题直接回答，记得结合历史对话上下文。"""),
    MessagesPlaceholder(variable_name="chat_history"),  # 窗口记忆注入点
    ("human", "{input}")  # 用户新问题
])


# -------------------------- 3. 工具调用逻辑（判断是否需要计算） --------------------------
def judge_and_calc(inputs):
    """
    核心逻辑：
    1. 检测用户问题是否包含数学计算需求
    2. 是：调用PythonREPLTool计算，再结合LLM生成回答
    3. 否：直接用LLM回答
    """
    user_input = inputs["input"]
    chat_history = inputs["chat_history"]
    
    # 简单的计算意图检测（可根据需求扩展）
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
        enhanced_input = f"""
            用户问题:{user_input}
            计算过程/结果:{calc_expr} = {calc_result}
            请结合计算结果,用简单易懂的语言回答用户问题。"""
    return inputs


# -------------------------- 4. 窗口记忆实现（仅保留最近N轮） --------------------------
# 会话存储：key=session_id，value=InMemoryChatMessageHistory
window_memory_store = {}

def get_window_session_history(session_id: str) -> BaseChatMessageHistory:
    """获取带窗口限制的会话历史，自动截断超出长度的消息"""
    # 初始化会话记忆（无则创建）
    if session_id not in window_memory_store:
        window_memory_store[session_id] = InMemoryChatMessageHistory()
    
    history = window_memory_store[session_id]
    # 截断逻辑：保留最近WINDOW_SIZE轮（每轮2条消息：Human+AI）
    total_messages = len(history.messages)
    if total_messages > 2 * WINDOW_SIZE:
        history.messages = history.messages[-2 * WINDOW_SIZE:]  # 只保留最后N轮
    
    return history


# -------------------------- 5. 构建完整的LCEL链（记忆+工具+LLM） --------------------------
# 核心链：参数传递 → 计算判断 → 提示词拼接 → LLM生成
chain = (
    RunnableLambda(judge_and_calc)
    | prompt
    | llm
)

# 注入窗口记忆功能
chain_with_window_memory = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history=get_window_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="output"
)


# -------------------------- 6. 多轮对话测试 --------------------------
if __name__ == "__main__":
    session_id = "student_001"  # 每个用户独立会话ID，记忆隔离
    print("===== 带窗口记忆的数学计算智能助手 =====")
    print("支持：多轮对话、仅保留最近2轮记忆、自动数学计算")
    print("输入'退出'结束对话\n")
    
    while True:
        user_input = input("你：")
        if user_input in ["退出", "quit", "q"]:
            print("助手：再见！有问题随时问我～")
            break
        
        # 调用带窗口记忆的智能体
        response = chain_with_window_memory.invoke(
            {"input": user_input, "window_size": WINDOW_SIZE},
            config={"configurable": {"session_id": session_id}}
        )
        
        # 输出回答（并将对话存入记忆）
        print(f"助手：{response.content}\n")