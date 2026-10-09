# 1. 导入需要的模块
from llm_config import get_chat_model
from langchain_core.prompts import ChatPromptTemplate
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from langchain_core.output_parsers import StrOutputParser
#from langgraph.graph.state import CompiledStateGraph

# 2. 初始化大模型（和LangChain案例一样）
llm = get_chat_model(temperature=0.3, max_tokens=1024)

output_parser = StrOutputParser()  # 统一解析为字符串，避免报错

# ================== 定义状态 ==================
class CandidateState(TypedDict):
    resume: str
    job_requirements: str
    skills: str
    interview_feedback: str
    resume_info: Optional[str]     # 扇出节点输出
    skill_match: Optional[str]     # 扇出节点输出
    interview_summary: Optional[str]  # 扇出节点输出
    summary: Optional[str]         # 汇总节点输出

# ================== 定义扇出智能体 ==================
resume_agent = (
    ChatPromptTemplate.from_template(
        "请阅读以下候选人简历内容，提取关键信息（姓名、学历、工作经历、技能清单）：\n{resume}"
    )
    | llm
    | output_parser
)

skill_agent = (
    ChatPromptTemplate.from_template(
        "根据岗位要求：{job_requirements}，请分析候选人技能匹配情况，并给出匹配分（0-10）：\n"
        "候选人技能：{skills}"
    )
    | llm
    | output_parser
)

interview_agent = (
    ChatPromptTemplate.from_template(
        "请根据以下面试评价内容，总结候选人的优点和潜在改进点，简明扼要：\n{interview_feedback}"
    )
    | llm
    | output_parser
)

summary_agent = (
    ChatPromptTemplate.from_template(
        "请整合以下候选人信息，生成一份完整的招聘推荐报告（150字以内）：\n"
        "简历关键信息：{resume_info}\n"
        "技能匹配分析：{skill_match}\n"
        "面试总结：{interview_summary}"
    )
    | llm
    | output_parser
)

# ================== 定义扇出节点（并行执行） ==================
def resume_node(state: CandidateState) -> dict:
    result = resume_agent.invoke({"resume": state["resume"]})
    print("\n[扇出节点] 简历信息:", result)
    return {"resume_info": result}

def skill_node(state: CandidateState) -> dict:
    result = skill_agent.invoke({
        "job_requirements": state["job_requirements"],
        "skills": state["skills"],
    })
    print("\n[扇出节点] 技能匹配:", result)
    return {"skill_match": result}

def interview_node(state: CandidateState) -> dict:
    result = interview_agent.invoke({"interview_feedback": state["interview_feedback"]})
    print("\n[扇出节点] 面试总结:", result)
    return {"interview_summary": result}

# ================== 定义扇入汇总节点 ==================
def summary_node(state: CandidateState) -> dict:
    result = summary_agent.invoke({
        "resume_info": state["resume_info"] or "未提取到简历信息",
        "skill_match": state["skill_match"] or "未分析技能匹配",
        "interview_summary": state["interview_summary"] or "未生成面试总结",
    })
    print("\n[汇总节点] 招聘推荐报告:", result)
    return {"summary": result}

# ================== 构建图 ==================
graph = StateGraph(CandidateState)

graph.add_node("resume_info", resume_node)
graph.add_node("skill_match", skill_node)
graph.add_node("interview_summary", interview_node)
graph.add_node("summary", summary_node)

# 扇出
graph.add_edge(START, "resume_info")
graph.add_edge(START, "skill_match")
graph.add_edge(START, "interview_summary")

# 扇入：等待三个扇出节点都完成后，再执行汇总节点
graph.add_edge(["resume_info", "skill_match", "interview_summary"], "summary")

# 汇总节点到结束
graph.add_edge("summary", END)

# ================== 编译并运行 ==================
app = graph.compile()

input_state = CandidateState(
    resume="张三，硕士学历，5年软件开发经验，熟悉Python、Java、SQL。",
    job_requirements="熟悉Python和数据分析，具有团队协作能力。",
    skills="Python, Java, SQL, 数据分析",
    interview_feedback="表达清晰，逻辑性强，但在团队管理经验方面稍弱。",
    resume_info=None,
    skill_match=None,
    interview_summary=None,
    summary=None
)

result = app.invoke(input_state)

print("\n=== 最终招聘推荐报告 ===")
print(result["summary"])