# 1. 导入需要的模块
from llm_config import get_chat_model
from langchain_core.prompts import ChatPromptTemplate
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
#from langgraph.graph.state import CompiledStateGraph

# 2. 初始化大模型（和LangChain案例一样）
llm = get_chat_model(temperature=0.3, max_tokens=1024)

# ========== 3. 状态定义 ==========
class ChainState(TypedDict):
    task: str
    draft: Optional[str]
    corrected: Optional[str]
    polished: Optional[str]

# ========== 4. Agent Prompt ==========
write_agent = ChatPromptTemplate.from_messages([
    ("system", "你是写作智能体，只负责生成初稿，不要解释。"),
    ("user", "{task}")
]) | llm

correct_agent = ChatPromptTemplate.from_messages([
    ("system", "你是纠错智能体，只修正语法、逻辑和错别字，不扩写。"),
    ("user", "{draft}")
]) | llm

polish_agent = ChatPromptTemplate.from_messages([
    ("system", "你是润色智能体，只提升表达质量和专业度，不改变意思。"),
    ("user", "{corrected}")
]) | llm

# ========== 5. Agent Node ==========
def writer_node(state: ChainState):
    print("\n✍️【Writer Agent】生成初稿中...")
    res = write_agent.invoke({"task": state["task"]})
    return {"draft": res.content.strip()}

def correct_node(state: ChainState):
    print("\n🧹【Corrector Agent】纠错中...")
    res = correct_agent.invoke({"draft": state["draft"]})
    return {"corrected": res.content.strip()}

def polish_node(state: ChainState):
    print("\n✨【Polisher Agent】润色中...")
    res = polish_agent.invoke({"corrected": state["corrected"]})
    return {"polished": res.content.strip()}

# ========== 6. 构建链式 LangGraph ==========
workflow = StateGraph(ChainState)

workflow.add_node("writer", writer_node)
workflow.add_node("corrector", correct_node)
workflow.add_node("polisher", polish_node)

# 链式 Pipeline
workflow.add_edge(START, "writer")
workflow.add_edge("writer", "corrector")
workflow.add_edge("corrector", "polisher")
workflow.add_edge("polisher", END)

app = workflow.compile()

# ========== 7. 运行 ==========
if __name__ == "__main__":
    init_state = {
        "task": "撰写一篇150字左右的介绍文，说明LangGraph多智能体的核心优势，适合技术初学者阅读",
        "draft": None,
        "corrected": None,
        "polished": None,
    }

    result = app.invoke(init_state)

    print("\n" + "=" * 90)
    print("📊 链式多智能体 Pipeline 最终结果")
    print("=" * 90)
    print("\n📝 初稿：\n", result["draft"])
    print("\n✅ 纠错：\n", result["corrected"])
    print("\n✨ 润色：\n", result["polished"])
    print("=" * 90)