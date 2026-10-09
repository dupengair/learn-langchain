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
    interrupt_after=["plot"]  # 情节生成后中断
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
        print("\n⚠️ 工作流已中断，可演示【回退到 plot 节点】")

# ================== 工作流回退示例 ==================
print("\n=== 回退示例：将工作流回退到 plot 节点 ===\n")

# 获取 checkpoint 当前状态
checkpoint = memory.get({"configurable": {"thread_id": thread_id}})
state = checkpoint["channel_values"]

# 使用 update_state 设置 current_node，实现回退
app.update_state(
    config={"configurable": {"thread_id": thread_id, "current_node": "plot"}},
    values=state
)

print("✅ 工作流已回退到 plot 节点，准备重新执行该节点及后续流程...\n")

# ================== 继续执行回退后的工作流 ==================
final_state = app.invoke(
    None,
    config={"configurable": {"thread_id": thread_id}}
)

print("\n=== 最终状态（回退后重新执行 plot 节点完成）===\n")
print(final_state["plot"])
print("\n✅ 工作流回退并重新执行完成")