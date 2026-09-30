# 1. 导入需要的模块
from llm_config import get_chat_model
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

# 4. 初始化大模型（和LangChain案例一样）
llm = get_chat_model(temperature=0.3, max_tokens=1024)

# 5. 定义 State
class WorkflowState(TypedDict, total=False):
    user_role: str  # 存储用户角色
    original_advice: str  # 存储原始学习建议
    simplified_advice: str  # 存储精简后的建议

# 6. 定义节点
def generate_advice(state: WorkflowState):
    prompt = f"给{state['user_role']}写一段50字左右的 AI 学习建议。"
    result = llm.invoke(prompt)
    return {"original_advice": result.content}

def simplify_advice(state: WorkflowState):
    prompt = f"把下面的学习建议精简到30字以内：{state['original_advice']}"
    result = llm.invoke(prompt)
    return {"simplified_advice": result.content}

# 7. 构建工作流
workflow = StateGraph(WorkflowState)

workflow.add_node("generate", generate_advice)
workflow.add_node("simplify", simplify_advice)

workflow.add_edge(START, "generate")
workflow.add_edge("generate", "simplify")
workflow.add_edge("simplify", END)

app = workflow.compile()

# 8. 执行
result = app.invoke({"user_role": "高校学生"})

# 9. 输出
print("原始学习建议：")
print(result["original_advice"])
print("\n精简后学习建议：")
print(result["simplified_advice"])