# 1. 导入需要的模块
from llm_config import get_chat_model
from langchain_core.prompts import ChatPromptTemplate
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from langchain_core.output_parsers import StrOutputParser
from langgraph.checkpoint.memory import MemorySaver

# 4. 初始化大模型（和LangChain案例一样）
llm = get_chat_model(temperature=0.3, max_tokens=1024)

# ================== 定义状态 ==================
class TaskState(TypedDict):
    agent_name: str
    task_type: str
    task: str
    task_result: str
    approve_status: str
    retry: bool

# ================== 节点 1：任务分配 Agent ==================
def assign_task_agent(state: TaskState) -> dict:
    prompt = ChatPromptTemplate.from_template(
        "请给{agent_name}分配一个{task_type}任务，要求具体、可执行，篇幅50字以内。"
    )
    chain = prompt | llm
    result = chain.invoke({
        "agent_name": state["agent_name"],
        "task_type": state["task_type"]
    })
    return {"task": result.content}

# ================== 节点 2：任务执行 Agent ==================
def execute_task_agent(state: TaskState) -> dict:
    prompt = ChatPromptTemplate.from_template(
        "请执行以下任务：{task}，要求输出执行结果，简洁明了。"
    )
    chain = prompt | llm
    result = chain.invoke({"task": state["task"]})
    return {"task_result": result.content}

# ================== 节点 3：审批 Agent（关键节点，中断） ==================
def approve_task_agent(state: TaskState) -> dict:
    print("\n📝 【任务审批节点】")
    print(f"Agent：{state['agent_name']}")
    print(f"任务：{state['task']}")
    print(f"执行结果：{state['task_result']}")
    return {"approve_status": "passed"}

# ================== 构建工作流 ==================
graph = StateGraph(TaskState)

graph.add_node("assign_task", assign_task_agent)
graph.add_node("execute_task", execute_task_agent)
graph.add_node("approve_task", approve_task_agent)

graph.add_edge(START, "assign_task")
graph.add_edge("assign_task", "execute_task")
graph.add_edge("execute_task", "approve_task")
graph.add_edge("approve_task", END)

# ================== 启用 MemorySaver + 中断 ==================
memory = MemorySaver()

# 修改：增加 interrupt_after 演示
app = graph.compile(
    checkpointer=memory,
    interrupt_before=["approve_task"],  # 审批节点前中断
    interrupt_after=["execute_task"]    # 任务执行后中断
)

# ================== 第一次运行：执行到审批节点 ==================
thread_id = "multi_agent_task_001"
print("\n=== 第一次运行：任务分配与执行 ===")

stream = app.stream(
    {
        "agent_name": "情节设计Agent",
        "task_type": "小说章节情节撰写"
    },
    config={"configurable": {"thread_id": thread_id}}
)

for step in stream:
    if "assign_task" in step:
        print(f"\n📝 分配的任务：\n{step['assign_task']['task']}")
    if "execute_task" in step:
        print(f"\n✅ 任务执行结果：\n{step['execute_task']['task_result']}")
        print("\n⚠️ 系统已在任务执行后中断，请查看执行结果（输入任意内容继续）")
        input("按回车继续到审批节点...")

# ================== 人工审批 ==================
print("\n⚠️ 审批节点已中断，请进行人工审批（输入'审批通过'继续，其他内容驳回）")
user_input = input("请输入审批指令：")

if user_input.strip() == "审批通过":
    print("\n=== 审批通过，继续执行审批节点 ===")
    result = app.invoke(
        None,  # 从中断点继续，不需要传入新参数
        config={"configurable": {"thread_id": thread_id}}
    )
    print("\n✅ 工作流完成，审批状态：", result.get("approve_status"))

else:
    print("\n❌ 审批驳回，任务需重新执行")
    # 驳回后可重新执行任务执行节点（示例：传入 retry 标记）
    result = app.invoke(
        {"retry": True},
        config={"configurable": {"thread_id": thread_id}}
    )
    print("\n🔁 工作流已重新执行，状态：", result.get("approve_status"))