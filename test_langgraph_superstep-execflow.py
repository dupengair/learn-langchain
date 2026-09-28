from langgraph.graph import StateGraph, START, END
from typing import Optional
from typing import TypedDict, Optional, NotRequired
import time
import random


# ===== 1. 定义状态（共享黑板） =====

class TextProcessState(TypedDict):
    raw_text: str
    summary_text: NotRequired[str]
    keyword_text: NotRequired[str]
    has_sensitive: NotRequired[bool]
    final_text: NotRequired[str]

# ===== 2. 去重节点（入口节点） =====
def deduplicate_node(state: TextProcessState):
    print("\n【节点 deduplicate】执行中...")
    text = state["raw_text"]   # ✅ TypedDict 访问方式
    time.sleep(1)
    return {"raw_text": text}

# ===== 3. 摘要节点（并行节点1） =====
def summary_node(state: TextProcessState):
    print("⚡ 并行节点 summary 执行中...")
    time.sleep(random.uniform(1, 2))
    summary = "摘要：" + state["raw_text"][:10]
    return {"summary_text": summary}

# ===== 4. 关键词节点（并行节点2） =====
def keyword_node(state: TextProcessState):
    print("⚡ 并行节点 keyword 执行中...")
    time.sleep(random.uniform(1, 2))
    keywords = "、".join(state["raw_text"].split("，")[:3])
    return {"keyword_text": keywords}

# ===== 5. 并行汇合节点 =====
def sensitive_check_node(state: TextProcessState):
    print("\n【节点 sensitive_check】汇总并行结果")
    print(" summary_text =", state["summary_text"])
    print(" keyword_text =", state["keyword_text"])

    sensitive = "暴力" in state["raw_text"]
    return {
        "has_sensitive": sensitive,
        "final_text": f"最终输出 | 摘要={state['summary_text']} | 关键词={state['keyword_text']}"
    }

# ===== 6. 构建 LangGraph =====
builder = StateGraph(TextProcessState)

builder.add_node("deduplicate", deduplicate_node)
builder.add_node("summary", summary_node)
builder.add_node("keyword", keyword_node)
builder.add_node("sensitive_check", sensitive_check_node)

builder.add_edge(START, "deduplicate")

# 并行分叉
builder.add_edge("deduplicate", "summary")
builder.add_edge("deduplicate", "keyword")

# 并行汇合
builder.add_edge("summary", "sensitive_check")
builder.add_edge("keyword", "sensitive_check")

builder.add_edge("sensitive_check", END)

graph = builder.compile()

# ===== 7. 运行 =====
if __name__ == "__main__":
    init_state = TextProcessState(
        raw_text="LangGraph很强大，支持状态管理，支持动态分支，并行执行"
    )

    final_state = graph.invoke(init_state)

    print("\n====== 最终状态 ======")
    print(final_state)