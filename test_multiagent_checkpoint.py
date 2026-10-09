# 1. 导入需要的模块
from llm_config import get_chat_model
from langchain_core.prompts import ChatPromptTemplate
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from langchain_core.output_parsers import StrOutputParser
from langgraph.checkpoint.memory import MemorySaver

# 2. 初始化大模型（和LangChain案例一样）
llm = get_chat_model(temperature=0.3, max_tokens=1024)

class NovelState(TypedDict):
    novel_name: str
    plot: Optional[str]
    retry_count: int
    review_result: Optional[str]
    failed: Optional[bool]  # 新增字段：超过重试次数时标记失败

# ================== 1. 定义核心节点 ==================
def plot_agent(state: NovelState) -> dict:
    prompt = ChatPromptTemplate.from_template(
        """请撰写小说《{novel_name}》的第一章情节，要求：
1. 引出主角；
2. 交代核心冲突；
3. 篇幅约200字。"""
    ) 
    chain = prompt | llm
    result = chain.invoke({"novel_name": state["novel_name"]})

    return {
        "plot": result.content,
        "retry_count": state.get("retry_count", 0)
    }

MAX_RETRIES = 2
def review_agent(state: NovelState) -> dict:
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
    )
    chain = prompt | llm
    result = chain.invoke({"plot": state["plot"]})
    review_result = result.content.strip().lower()

    retry_count = state.get("retry_count", 0)
    
    # 如果审核结果为 retry，需要更新重试计数和失败标志
    if review_result == "retry":
        retry_count += 1
        if retry_count >= MAX_RETRIES:
            return {
                "review_result": review_result,
                "retry_count": retry_count,
                "failed": True
            }
        return {
            "review_result": review_result,
            "retry_count": retry_count
        }
    else:  # pass
        return {
            "review_result": review_result,
            "retry_count": retry_count
        }
# ================== 2. 定义条件分支 ==================
def decide_next_node(state: NovelState) -> str:
    # 若已标记失败或审核通过，则结束流程
    if state.get("failed", False):
        return END
    if state["review_result"] == "pass":
        return END
    # 否则返回 plot 节点重试
    return "plot"

# ================== 3. 构建循环逻辑图 ==================
graph = StateGraph(NovelState)

# 添加节点
graph.add_node("plot", plot_agent)
graph.add_node("review", review_agent)


# 构建边
graph.add_edge(START, "plot")
graph.add_edge("plot", "review")
graph.add_conditional_edges("review", decide_next_node)

# ================== 4. 编译 & 运行 ==================
checkpointer = MemorySaver()

app = graph.compile(checkpointer = checkpointer)


# =====================================================
# 第一阶段：执行到 plot 节点后中断
# =====================================================
print("\n=== 第一次运行（执行到 plot 后中断）===")

thread_id = "novel_session_001"

stream = app.stream(
    {
        "novel_name": "星际流浪记",
        "retry_count": 0,
        "plot": None,
        "review_result": None,
        "failed": False
    },
    config={
        "configurable": {
            "thread_id": thread_id
        }
    }
)

for step in stream:
    print("当前 step：", step)

    # 只要 plot 执行完，就人为中断
    if "plot" in step:
        plot_state = step["plot"]

        print("\n🛑 模拟程序中断（Ctrl+C 场景）")
        print(f"中断时版本：第 {plot_state['retry_count']} 版")
        print(f"中断时情节内容：\n{plot_state['plot']}")

        break   # ⛔ 中断执行


# =====================================================
# 第二阶段：从 checkpoint 恢复
# =====================================================
print("\n=== 第二次运行（从存档恢复）===")

result = app.invoke(
    None,  # 不传新输入
    config={
        "configurable": {
            "thread_id": thread_id
        }
    }
)

print("\n✅ 恢复后最终结果")
print(f"重试次数：{result['retry_count']}")
print(f"是否失败：{result.get('failed')}")
print("\n最终情节：\n")
print(result["plot"])