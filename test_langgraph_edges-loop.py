from typing import TypedDict, NotRequired
from langgraph.graph import StateGraph

# ====== 全局共享状态（黑板） ======
class TaskState(TypedDict):
    user_query: str #用户原始查询
    intent: NotRequired[str] #用户意图
    llm_answer: NotRequired[str] #LLM 生成的回答
    tool_result: NotRequired[str] #工具调用结果
    final_answer: NotRequired[str] #最终回答
    progress: NotRequired[int] #任务进度百分比

def final_node(state: TaskState):
    print("\n🔹 final_node")
    return {"progress": 100} 

#=======循环节点======
def loop_node(state: TaskState):
    progress = state.get("progress", 0)
    print("\n🔄 loop_node, progress =", progress)
    return {"progress": progress + 30}

#=======循环条件函数======
def loop_router(state: TaskState):
    if state["progress"] >= 100:
        return "final_node"
    return "loop_node"

#=======构建图=========
builder = StateGraph(TaskState)

builder.add_node("loop_node", loop_node)
builder.add_node("final_node", final_node)

builder.set_entry_point("loop_node")

builder.add_conditional_edges(
    "loop_node",
    loop_router,
    {
        "loop_node": "loop_node",     # 回环
        "final_node": "final_node"    # 终止
    }
)

graph = builder.compile()

print(graph.invoke(TaskState(user_query="test")))