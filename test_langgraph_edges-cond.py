from typing import TypedDict, NotRequired

# ====== 全局共享状态（黑板） ======
class TaskState(TypedDict):
    user_query: str #用户原始查询
    intent: NotRequired[str] #用户意图
    llm_answer: NotRequired[str] #LLM 生成的回答
    tool_result: NotRequired[str] #工具调用结果
    final_answer: NotRequired[str] #最终回答
    progress: NotRequired[int] #任务进度百分比

 #========固定边（线性流程）实践代码======
def parse_intent(state: TaskState):
    print("\n🔹 parse_intent")
    # 简单模拟意图识别
    intent = "summarize" if "总结" in state['user_query'] else "rewrite"
    return {"intent": intent, "progress": 30}


def summarize_node(state: TaskState):
    print("\n🔹 summarize_node")
    return {"llm_answer": f"总结结果: {state['user_query']}", "progress": 60}


def rewrite_node(state: TaskState):
    print("\n🔹 rewrite_node")
    return {"llm_answer": f"改写结果: {state['user_query']}", "progress": 60}


def final_node(state: TaskState):
    print("\n🔹 final_node")
    return {"progress": 100} 

# 条件边
def route_by_intent(state: TaskState):
    if state.get("intent") == "summarize":
        return "summarize_node"
    else:
        return "rewrite_node"  

#========构建 LangGraph 工作流======
from langgraph.graph import StateGraph

builder = StateGraph(TaskState)

builder.add_node("parse_intent", parse_intent)
builder.add_node("summarize_node", summarize_node)
builder.add_node("rewrite_node", rewrite_node)
builder.add_node("final_node", final_node)

builder.set_entry_point("parse_intent")
# 固定边
#builder.add_edge("parse_intent", "summarize_node")   # 固定走总结
#builder.add_edge("summarize_node", "final_node")
# 条件边
builder.add_conditional_edges(
    "parse_intent",
    route_by_intent,
    {
        "summarize_node": "summarize_node",
        "rewrite_node": "rewrite_node",
    }
)
builder.add_edge("rewrite_node", "final_node")
builder.add_edge("summarize_node", "final_node")

graph = builder.compile()

print("\n====== 测试总结 ======")
print(graph.invoke(TaskState(user_query="请总结这段话")))

print("\n====== 测试改写 ======")
print(graph.invoke(TaskState(user_query="请改写这段话")))

from IPython.display import Image, display
display(Image(graph.get_graph().draw_mermaid_png()))