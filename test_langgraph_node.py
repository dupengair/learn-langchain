from typing import TypedDict, NotRequired

class TaskState(TypedDict):
    user_query: str #用户原始查询
    intent: NotRequired[str] #用户意图
    llm_answer: NotRequired[str] #LLM 生成的回答
    tool_result: NotRequired[str] #工具调用结果
    final_answer: NotRequired[str] #最终回答
    progress: NotRequired[int] #任务进度百分比

#=======LLM 调用节点（模拟）========= 
#LLM 节点只负责“生成文本”，不负责工具、不负责格式化。
def llm_node(state: TaskState):
    print("\n🧠 [LLM Node] 输入状态:", state)

    # 模拟大模型生成
    llm_output = f"LangGraph 是一种用于构建可控AI工作流的框架，问题是：{state['user_query']}"

    return {
        "llm_answer": llm_output,
        "progress": 30
    } 

# ====== 工具调用节点（模拟检索） ======
#工具节点只做 I/O，不生成语言，不控制流程。
def tool_node(state: TaskState):
    print("\n🔧 [Tool Node] 输入状态:", state)

    # 模拟搜索工具
    tool_output = f"检索结果：LangGraph 支持 DAG、状态机、Agent workflow"

    return {
        "tool_result": tool_output,
        "progress": 70
    }

# ====== 节点3：数据处理节点 ======
#数据节点负责加工 state，而不是生成或检索。
def data_process_node(state: TaskState):
    print("\n🧹 [Data Node] 输入状态:", state)

    final = f"""
        【LLM回答】{state["llm_answer"]}
        【工具补充】{state["tool_result"]}
        """

    return {
        "final_answer": final.strip(),
        "progress": 100
    }

#========构建 LangGraph 工作流======
from langgraph.graph import StateGraph

# 1. 创建图
builder = StateGraph(TaskState)

# 2. 添加节点
builder.add_node("llm_node", llm_node)
builder.add_node("tool_node", tool_node)
builder.add_node("data_process_node", data_process_node)

# 3. 定义流程
builder.set_entry_point("llm_node")
builder.add_edge("llm_node", "tool_node")
builder.add_edge("tool_node", "data_process_node")

# 4. 编译
graph = builder.compile()

# 初始化状态
init_state = TaskState(user_query="什么是 LangGraph？")

# 执行
final_state = graph.invoke(init_state)

print("\n====== 最终状态 ======")
print(final_state)