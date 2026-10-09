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
class EmailState(TypedDict):
    sender: str
    recipient: str
    email_type: str
    subject: str
    email_content: str
    send_status: str

# ================== 节点 1：撰写邮件 ==================
def write_email_agent(state: EmailState) -> dict:
    prompt = ChatPromptTemplate.from_template(
        """请以{sender}的身份，给{recipient}写一封{email_type}邮件，
主题是《{subject}》，内容简洁、正式、符合邮件格式。"""
    )

    chain = prompt | llm
    result = chain.invoke({
        "sender": state["sender"],
        "recipient": state["recipient"],
        "email_type": state["email_type"],
        "subject": state["subject"]
    })

    return {
        "email_content": result.content
    }

# ================== 节点 2：发送邮件（模拟） ==================
def send_email_agent(state: EmailState) -> dict:
    print("\n📤 【邮件发送成功】")
    print(f"收件人：{state['recipient']}")
    print(f"主题：{state['subject']}")
    print(f"内容：\n{state['email_content']}\n")

    return {
        "send_status": "success"
    }

# ================== 构建 LangGraph ==================
graph = StateGraph(EmailState)

graph.add_node("write_email", write_email_agent)
graph.add_node("send_email", send_email_agent)

graph.add_edge(START, "write_email")
graph.add_edge("write_email", "send_email")
graph.add_edge("send_email", END)

# ================== 启用 MemorySaver + 中断配置 ==================
memory = MemorySaver()

app = graph.compile(
    checkpointer=memory,
    interrupt_before=["send_email"]  # ⭐ 关键：发送前中断
)

# =====================================================
# 第一次运行：执行到 send_email 前中断
# =====================================================
print("\n=== 第一次运行：生成邮件，等待人工确认 ===")

thread_id = "email_session_001"

stream = app.stream(
    {
        "sender": "学生张三",
        "recipient": "老师@xxx.edu.cn",
        "email_type": "请假",
        "subject": "请假申请（1天）"
    },
    config={
        "configurable": {
            "thread_id": thread_id
        }
    }
)

for step in stream:
    if "write_email" in step:
        print("\n✉️ 已生成邮件内容：\n")
        print(step["write_email"]["email_content"])
        print("\n⚠️ 系统已在【发送邮件】前中断")
        print("请输入：确认发送  → 继续执行\n")

# =====================================================
# 人工确认
# =====================================================
user_input = input("请输入授权指令：")

if user_input.strip() == "确认发送":
    print("\n=== 已确认，继续执行发送节点 ===")

    result = app.invoke(
        None,  # 不传新状态，直接从 checkpoint 恢复
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )

    print("✅ 工作流完成，发送状态：", result.get("send_status"))

else:
    print("\n❌ 已取消发送，工作流终止")