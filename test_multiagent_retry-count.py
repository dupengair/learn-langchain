# 1. 导入需要的模块
from llm_config import get_chat_model
from langchain_core.prompts import ChatPromptTemplate
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from langchain_core.output_parsers import StrOutputParser

# 2. 初始化大模型（和LangChain案例一样）
llm = get_chat_model(temperature=0.3, max_tokens=1024)

class NovelState(TypedDict):
    novel_name: str
    plot: Optional[str]
    retry_count: int
    review_result: Optional[str]
    failed: Optional[bool]  # 新增字段：超过重试次数时标记失败

# ================== 1. 定义核心节点 ==================
def plot_agent(state):
    prompt = ChatPromptTemplate.from_template(
        """请撰写小说《{novel_name}》的第一章情节，要求：
1. 引出主角；
2. 交代核心冲突；
3. 篇幅约200字。"""
    ) | llm

    plot_result = prompt.invoke({"novel_name": state["novel_name"]})
    retry_count = state.get("retry_count", 0)
    return {"plot": plot_result.content, "retry_count": retry_count}

def review_agent(state):
    """
    审核 Agent：判断情节是否达标
    严格返回 'pass' 或 'retry'
    """
    prompt = ChatPromptTemplate.from_template(
        """请审核以下小说情节是否达标，审核标准：
1. 引出主角；
2. 交代核心冲突；
3. 篇幅约200字。

情节：
{plot}

⚠️ 注意：只返回 'pass' 或 'retry'，不要输出其他内容，严格按照要求执行！"""
    ) | llm

    result = prompt.invoke({"plot": state["plot"]})
    review_result = result.content.strip().lower()
    
    # 在节点中处理计数更新（正确的不可变更新方式）
    retry_count = state.get("retry_count", 0)
    if review_result == "retry":
        retry_count = retry_count + 1
    
    return {
        "review_result": review_result,
        "retry_count": retry_count
    }

# ================== 2. 定义条件分支 ==================
MAX_RETRIES = 2  # 最大重试次数

def decide_next_node(state):
    """
    - 'pass' -> 结束
    - 'retry' + 未超限 -> 重新生成情节
    - 'retry' + 超限 -> 标记失败并结束
    """
    retry_count = state.get("retry_count", 0)
    review_result = state.get("review_result", "retry")
    
    if review_result == "pass":
        return "end"
    elif retry_count >= MAX_RETRIES:
        # 超过最大重试次数，标记失败并结束
        return "end"
    else:
        # 返回 plot 节点继续重试
        return "plot"

# ================== 3. 构建循环逻辑图 ==================
graph = StateGraph(NovelState)

# 添加节点
graph.add_node("plot", plot_agent)
graph.add_node("review", review_agent)
graph.add_node("end", lambda state: {
    **state,
    "failed": state.get("review_result") != "pass" and state.get("retry_count", 0) >= MAX_RETRIES
})

# 构建边
graph.add_edge(START, "plot")
graph.add_edge("plot", "review")

# 条件分支（审核结果决定下一步）
graph.add_conditional_edges(
    source="review",
    path=decide_next_node  # 直接返回 "end" 或 "plot"
)

# ================== 4. 编译 & 运行 ==================
app = graph.compile()

# 初始参数
input_state = {"novel_name": "星际流浪记", "retry_count": 0, "plot": None, "review_result": None, "failed": False}

result = app.invoke(input_state)

# ================== 5. 打印最终结果 ==================
if result.get("failed"):
    print(f"\n任务失败：超过最大重试次数 {MAX_RETRIES} 次，情节未达标。")
else:
    print(f"\n最终情节（重试 {result['retry_count']} 次）：\n")
    print(result["plot"])