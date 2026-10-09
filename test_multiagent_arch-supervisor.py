# 1. 导入需要的模块
from llm_config import get_chat_model
from langchain_core.prompts import ChatPromptTemplate
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
#from langgraph.graph.state import CompiledStateGraph

# 2. 初始化大模型（和LangChain案例一样）
llm = get_chat_model(temperature=0.3, max_tokens=1024)

# ================== 状态定义 ==================
class TaskState(TypedDict):
    task: str
    research: Optional[str]
    draft: Optional[str]
    code: Optional[str]
    math: Optional[str]
    next_agent: Optional[str]
    result: Optional[str]
    round_count: int  # Supervisor 执行轮次
    supervisor_thoughts: Optional[str]  # 打印 LLM 思考过程

MAX_ROUNDS = 3

# ================== 员工智能体 ==================
research_agent = ChatPromptTemplate.from_messages([
    ("user", "请调研以下任务的背景信息，整理成条列要点，中文输出：{task}")
]) | llm

writer_agent = ChatPromptTemplate.from_messages([
    ("user", "根据以下信息撰写中文技术文章或说明文：{research}")
]) | llm

code_agent = ChatPromptTemplate.from_messages([
    ("user", "请根据以下任务生成 Python 示例代码：{task}")
]) | llm

math_agent = ChatPromptTemplate.from_messages([
    ("user", "请解决以下数学/逻辑问题，并详细说明过程：{task}")
]) | llm

# ================== 动态 Supervisor 节点 ==================
def supervisor_node(state: TaskState):
    new_round = state["round_count"] + 1

    # 超过最大轮次，触发兜底
    if new_round > MAX_ROUNDS:
        print(f"⚠️ 超过最大轮次 {MAX_ROUNDS}，触发兜底 → 结束任务")
        return {
            "round_count": new_round,
            "next_agent": "end",
            "supervisor_thoughts": "轮次数超过上限，直接结束任务"
        }

    # 中文提示词，严格约束 LLM
    prompt = f"""
你是多智能体系统的主管智能体（Supervisor），负责调度专家智能体，但你不执行任务。请阅读当前任务和已完成状态，并选择下一步最合适的智能体执行。

任务：
{state['task']}

已完成状态：
- 调研: {"已完成" if state.get("research") else "未完成"}
- 写作: {"已完成" if state.get("draft") else "未完成"}
- 编程: {"已完成" if state.get("code") else "未完成"}
- 数学: {"已完成" if state.get("math") else "未完成"}

可调度智能体：
- research_agent：负责调研和整理资料
- writer_agent：负责撰写中文文章或说明文
- code_agent：负责编写 Python 代码
- math_agent：负责数学/逻辑计算与推理

约束：
1. 不能选择已完成的智能体。
2. 必须选择与任务相关的智能体。
3. 如果所有任务完成，返回 "end"。
4. 请在回答中先写出你的“思考过程”，然后在最后一行返回下一步智能体名称（research_agent / writer_agent / code_agent / math_agent / end）。

请用中文完整回答：
"""
    res = llm.invoke(prompt)
    thoughts = res.content.strip()
    last_line = thoughts.splitlines()[-1]
    valid_agents = ("research_agent", "writer_agent", "code_agent", "math_agent", "end")
    next_agent = next((a for a in valid_agents if a in last_line), "end")
    print(f"🧠 主管思考过程：\n{thoughts}\n")
    print(f"🧠 主管调度 → {next_agent} (轮次 {new_round})")
    
    return {
        "round_count": new_round,
        "next_agent": next_agent,
        "supervisor_thoughts": thoughts
    }

# ================== 员工节点 ==================
def research_node(state: TaskState):
    print(">>> Research Agent 执行中...")
    try:
        res = research_agent.invoke({"task": state["task"]})
        result = res.content.strip()
    except Exception as e:
        result = f"调研失败：{str(e)[:50]}"
    
    # ✅ 正确写法：只返回更新字段
    return {
        "research": result,
        "result": result
    }

def writer_node(state: TaskState):
    print(">>> Writer Agent 执行中...")
    try:
        res = writer_agent.invoke({"research": state.get("research","")})
        result = res.content.strip()
    except Exception as e:
        result = f"写作失败：{str(e)[:50]}"
    
    # ✅ 正确写法
    return {
        "draft": result,
        "result": result
    }

def code_node(state: TaskState):
    print(">>> Code Agent 执行中...")
    try:
        res = code_agent.invoke({"task": state["task"]})
        result = res.content.strip()
    except Exception as e:
        result = f"代码生成失败：{str(e)[:50]}"
    
    # ✅ 正确写法
    return {
        "code": result,
        "result": result
    }

def math_node(state: TaskState):
    print(">>> Math Agent 执行中...")
    try:
        res = math_agent.invoke({"task": state["task"]})
        result = res.content.strip()
    except Exception as e:
        result = f"数学求解失败：{str(e)[:50]}"
    
    # ✅ 正确写法
    return {
        "math": result,
        "result": result
    }

# ================== 构建 LangGraph ==================
workflow = StateGraph(TaskState)

workflow.add_node("supervisor", supervisor_node)
workflow.add_node("research_agent", research_node)
workflow.add_node("writer_agent", writer_node)
workflow.add_node("code_agent", code_node)
workflow.add_node("math_agent", math_node)

workflow.add_edge(START, "supervisor")

workflow.add_conditional_edges(
    "supervisor",
    lambda s: s["next_agent"],
    {
        "research_agent": "research_agent",
        "writer_agent": "writer_agent",
        "code_agent": "code_agent",
        "math_agent": "math_agent",
        "end": END
    }
)

workflow.add_edge("research_agent", "supervisor")
workflow.add_edge("writer_agent", "supervisor")
workflow.add_edge("code_agent", "supervisor")
workflow.add_edge("math_agent", "supervisor")

app = workflow.compile()

# ================== 构建 LangGraph ==================
workflow = StateGraph(TaskState)

workflow.add_node("supervisor", supervisor_node)
workflow.add_node("research_agent", research_node)
workflow.add_node("writer_agent", writer_node)
workflow.add_node("code_agent", code_node)
workflow.add_node("math_agent", math_node)

workflow.add_edge(START, "supervisor")

workflow.add_conditional_edges(
    "supervisor",
    lambda s: s["next_agent"],
    {
        "research_agent": "research_agent",
        "writer_agent": "writer_agent",
        "code_agent": "code_agent",
        "math_agent": "math_agent",
        "end": END
    }
)

workflow.add_edge("research_agent", "supervisor")
workflow.add_edge("writer_agent", "supervisor")
workflow.add_edge("code_agent", "supervisor")
workflow.add_edge("math_agent", "supervisor")

app = workflow.compile()

# ================== 运行示例 ==================
if __name__ == "__main__":
    tasks = [
        "撰写一篇介绍 LangGraph 多智能体协作的中文文章，面向初学者",
    ]

    for t in tasks:
        print("\n" + "="*50)
        print(f"任务：{t}")
        init_state = {
            "task": t,
            "research": None,
            "draft": None,
            "code": None,
            "math": None,
            "next_agent": None,
            "result": None,
            "round_count": 0,
            "supervisor_thoughts": None
        }
        result = app.invoke(init_state)
        print("\n✅ 最终结果：\n", result["result"])
        png_data = app.get_graph().draw_mermaid_png()  # 获取PNG字节流
        with open("linear_text_process_graph.png", "wb") as file:  # wb=二进制写入
            file.write(png_data)
        print("📊 工作流可视化图已保存：linear_text_process_graph.png\n")