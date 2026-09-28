from langchain_openai import ChatOpenAI  # 补充LLM定义
from dotenv import load_dotenv
import os

# ================== 依赖 ==================
from typing import TypedDict, Optional
# 兼容低版本 Python 的 NotRequired 导入
try:
    from typing import NotRequired
except ImportError:
    from typing_extensions import NotRequired
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import PromptTemplate
from langgraph.checkpoint.memory import MemorySaver


# ================== LLM 初始化 ==================
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


# ------------------------------
# 2. 定义交互式状态（强类型，持久化存储所有流程数据）
# ------------------------------
class InteractiveOptState(TypedDict):
    user_input: str                # 固定：用户原始输入（全程不变）
    optimized_text: Optional[str]  # 每轮覆盖：AI优化后文本（只保留最新一版）
    optimize_suggest: Optional[str]# 每轮覆盖：优化建议/理由（只保留最新一版）
    user_feedback: Optional[str]   # 每轮覆盖：用户反馈（确认/修改/退出）
    final_result: Optional[str]    # 最终：流程结束时写入


# ------------------------------
# 3. 核心节点函数（无任何修改，保留你的原代码）
# ------------------------------
def optimize_node(state: InteractiveOptState) -> InteractiveOptState:
    """【机器节点】文本优化核心节点，使用管道符调用LLM"""
    user_input = state["user_input"]
    user_feedback = state["user_feedback"]

    if not user_feedback:
        prompt = PromptTemplate(
            input_variables=["text"],
            template="请优化以下文本，提升流畅度和专业度，严格保留核心信息：\n{text}\n优化完成后，单独一行以【优化理由：】开头给出1-2条简洁优化原因"
        )
        chain = prompt | llm
        result = chain.invoke({"text": user_input}).content
    else:
        prompt = PromptTemplate(
            input_variables=["text", "feedback"],
            template="根据用户反馈针对性优化文本，严格保留核心信息：\n原文本：{text}\n用户反馈：{feedback}\n优化完成后，单独一行以【优化理由：】开头给出1-2条简洁优化原因"
        )
        chain = prompt | llm
        result = chain.invoke({"text": user_input, "feedback": user_feedback}).content

    split_flag = "【优化理由：】"
    if split_flag in result:
        optimized_text, optimize_suggest = result.split(split_flag, 1)
    else:
        optimized_text = result
        optimize_suggest = "AI未生成明确优化理由，建议重新优化"

    return {
        "optimized_text": optimized_text.strip(),
        "optimize_suggest": optimize_suggest.strip()
    }

def feedback_node(state: InteractiveOptState) -> InteractiveOptState:
    """【人机交互节点】展示优化结果，接收用户操作和具体修改意见"""
    print("\n" + "-"*60)
    print("📝 AI优化后文本：")
    print(state["optimized_text"])
    print("\n💡 优化建议/理由：")
    print(state["optimize_suggest"])
    print("\n" + "-"*60)

    while True:
        action = input("请选择操作（确认/修改/退出）：").strip()
        if action == "确认" or action == "退出":
            return {"user_feedback": action}
        if action == "修改":
            detail = input("请输入具体修改意见：").strip()
            if detail:
                return {"user_feedback": detail}
            print("❌ 修改意见不能为空，请重新输入\n")
        else:
            print("❌ 请输入「确认」「修改」或「退出」\n")

def feedback_router(state: InteractiveOptState) -> str:
    """【条件路由】确认→结束，退出→终止，其他内容都是修改意见→回到优化节点"""
    feedback = state["user_feedback"]
    if feedback == "确认":
        return "final"
    if feedback == "退出":
        return "exit"
    return "optimize"

def final_node(state: InteractiveOptState) -> InteractiveOptState:
    """【机器节点】流程正常结束，生成格式化结果"""
    final_result = (
        "✅ 【多轮文本优化流程完成】\n"
        f"📌 最终优化文本：\n{state['optimized_text']}\n"
        f"💡 优化核心总结：\n{state['optimize_suggest']}"
    )
    return {"final_result": final_result}

def exit_node(state: InteractiveOptState) -> InteractiveOptState:
    """【机器节点】用户主动退出，生成终止提示"""
    return {"final_result": "🔚 【文本优化流程终止】\n你主动退出，本次无最终优化结果"}


# ------------------------------
# 4. 搭建循环交互图（无任何修改，保留你的原代码）
# ------------------------------
def build_interactive_graph():
    """构建LangGraph循环状态图，彻底适配最新API终极规范"""
    graph_builder = StateGraph(InteractiveOptState)

    # 添加节点（无修改）
    graph_builder.add_node("optimize", optimize_node)
    graph_builder.add_node("feedback", feedback_node)
    graph_builder.add_node("final", final_node)
    graph_builder.add_node("exit", exit_node)

    # 配置普通边（无修改）
    graph_builder.add_edge(START, "optimize")
    graph_builder.add_edge("optimize", "feedback")

    # 适配最新API：source + path
    graph_builder.add_conditional_edges(
        source="feedback",  # 分支起始节点
        path=feedback_router  # 路由函数（直接返回目标节点名）
    )

    # 配置结束边（无修改）
    graph_builder.add_edge("final", END)
    graph_builder.add_edge("exit", END)

    # 编译图：开启状态持久化（多轮交互必需）
    return graph_builder.compile(checkpointer=MemorySaver())


# ------------------------------
# 5. 运行交互测试（★仅修改初始输入部分★，改为用户手动输入+非空校验）
# ------------------------------
if __name__ == "__main__":
    # 构建循环图（彻底解决所有API报错）
    interactive_graph = build_interactive_graph()
    print("🔧 多轮交互式文本优化工具已启动（适配LangGraph最新API）...\n")

    # ★核心修改：用户手动输入待优化句子 + 非空校验★
    print("="*40 + " 输入待优化句子 " + "="*40)
    while True:
        user_input_text = input("请输入需要AI优化的句子：").strip()
        if user_input_text:  # 非空校验，避免用户输入空内容
            break
        print("❌ 输入不能为空，请重新输入需要优化的句子！\n")

    # 初始状态：使用用户输入的句子，其余字段保持默认
    initial_state: InteractiveOptState = {
        "user_input": user_input_text,  # 替换为用户输入的内容
        "optimized_text": None,
        "optimize_suggest": None,
        "user_feedback": None,
        "final_result": None
    }

    # 启动多轮交互流程（保留你的config配置）
    print(f"\n🚀 已接收你的句子，开始第一轮AI优化...")
    config = {"configurable": {"thread_id": "text_process_test_001"}}
    final_state = interactive_graph.invoke(initial_state, config=config)

    # 展示最终结果
    print("\n" + "="*60)
    print(final_state["final_result"])
    print("="*60)

    # 展示交互轮次（状态持久化验证）
    history = list(interactive_graph.get_state_history(config))
    interact_rounds = len(history) // 2  # 每轮=优化节点+反馈节点
    print("状态快照数量（超步骤）：", len(history))

    # 保存可视化流程图（保留你的原代码）
    png_data = interactive_graph.get_graph().draw_mermaid_png()  # 获取PNG字节流
    with open("interactive_optimize_graph.png", "wb") as file:  # wb=二进制写入
        file.write(png_data)
    print("📊 工作流可视化图已保存：interactive_optimize_graph.png\n")
