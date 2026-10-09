# 1. 导入需要的模块
from llm_config import get_chat_model
from langchain_core.prompts import ChatPromptTemplate
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# 2. 初始化大模型（和LangChain案例一样）
llm = get_chat_model(temperature=0.3, max_tokens=1024)

# ================== 定义状态 ==================
class NovelState(TypedDict):
    novel_name: str
    protagonist: str
    plot: Optional[str]

# ================== 节点：情节生成 Agent ==================
def plot_agent(state: NovelState):
    prompt = ChatPromptTemplate.from_template(
        """
请撰写小说《{novel_name}》的第一章情节，约200字。
主角名必须叫：{protagonist}
"""
    )
    msg = llm.invoke(
        prompt.format_messages(
            novel_name=state["novel_name"],
            protagonist=state["protagonist"]
        )
    )
    return {"plot": msg.content}

# ================== 构建工作流 ==================
graph = StateGraph(NovelState)

graph.add_node("plot", plot_agent)

graph.add_edge(START, "plot")
graph.add_edge("plot", END)

# ================== 启用 MemorySaver + 中断 ==================
memory = MemorySaver()

app = graph.compile(
    checkpointer=memory,
    interrupt_after=["plot"]  # ⭐ 情节生成后立刻中断
)

# ================== 第一次运行：生成情节并中断 ==================
thread_id = "dynamic_state_edit_demo"

print("\n=== 第一次运行：生成情节并中断 ===\n")

stream = app.stream(
    {
            "novel_name": "星际流浪记",
            "protagonist": "林启"
    },
    config={"configurable": {"thread_id": thread_id}}
)

for step in stream:
    if "plot" in step:
        print("【原始情节】\n")
        print(step["plot"]["plot"])
        print("\n⚠️ 工作流已中断，可进行【动态状态编辑】")

# ================== 动态状态编辑：只修改主角名字 ==================
print("\n=== 动态状态编辑：人工修改主角名字 ===\n")

new_name = input("请输入新的主角名字：").strip()

# 从 checkpoint 里取当前状态
checkpoint = memory.get({"configurable": {"thread_id": thread_id}})
state = checkpoint["channel_values"]

old_name = state["protagonist"]
old_plot = state["plot"]

# 只替换主角名字（教学最直观）
new_plot = old_plot.replace(old_name, new_name)

# ⭐ 正确写回方式：update_state
app.update_state(
    config={"configurable": {"thread_id": thread_id}},
    values={
        "protagonist": new_name,
        "plot": new_plot
    }
)

print("\n✅ 状态已更新，继续执行工作流...\n")

# ================== 第二次运行：从中断点继续 ==================
final_state = app.invoke(
    None,
    config={"configurable": {"thread_id": thread_id}}
)

print("\n=== 最终状态（已被人工修改）===\n")
print(final_state["plot"])