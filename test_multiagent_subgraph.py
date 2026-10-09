# 1. 导入需要的模块
from llm_config import get_chat_model
from langchain_core.prompts import ChatPromptTemplate
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from langchain_core.output_parsers import StrOutputParser
#from langgraph.graph.state import CompiledStateGraph

# 4. 初始化大模型（和LangChain案例一样）
llm = get_chat_model(temperature=0.3, max_tokens=1024)

output_parser = StrOutputParser()  # 统一解析为字符串，避免报错


# -------------------------- 第一步：定义“作业批改子图”（独立、可复用）--------------------------
# 子图独立状态：添加Optional，所有字段支持初始None/默认值
class CorrectionSubgraphState(TypedDict):
    homework_content: str                # 待批改作业（主图传递，必传）
    completion: Optional[str] = None     # 完成度：完成/未完成（初始None）
    accuracy: Optional[str] = None       # 正确率：正确率XX%（初始None）
    score: Optional[int] = 0             # 最终得分（初始默认0分，避免类型错误）

# 子图智能体（分工明确，LLM输出固定格式，无解析冗余）
# 智能体1：检查作业完成度（仅输出「完成」/「未完成」）
completion_check_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是作业完成度检查老师，仅输出「完成」或「未完成」，不添加任何额外文字！"),
    ("user", "作业内容：{homework_content}，判断是否完成（有具体内容、无空白即为完成）")
])
completion_check_agent = completion_check_prompt | llm | output_parser

# 智能体2：检查作业正确率（仅输出「正确率XX%」）
accuracy_check_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是作业正确率检查老师，仅输出「正确率XX%」，不添加任何额外文字！"),
    ("user", "作业内容：{homework_content}，假设是数学计算题，合理估算正确率")
])
accuracy_check_agent = accuracy_check_prompt | llm | output_parser

# 智能体3：计算最终得分（仅输出0-100整数）
score_calc_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是得分计算老师，仅输出0-100的整数，不添加任何额外文字！"),
    ("user", "完成度：{completion}，正确率：{accuracy}，计分规则：完成得60基础分，正确率每10%加4分，未完成得0分")
])
score_calc_agent = score_calc_prompt | llm | output_parser

# 子图节点函数（打印日志+更新状态，学生易观察）
def check_completion_node(state: CorrectionSubgraphState) -> CorrectionSubgraphState:
    """子图节点1：检查作业完成度"""
    print(f"🔍 子图执行 - 检查作业完成度")
    completion = completion_check_agent.invoke({"homework_content": state["homework_content"]})
    # 修复解包顺序：先解包原状态，再更新新字段（统一规范）
    return {**state, "completion": completion}

def check_accuracy_node(state: CorrectionSubgraphState) -> CorrectionSubgraphState:
    """子图节点2：检查作业正确率"""
    print(f"🔍 子图执行 - 检查作业正确率")
    accuracy = accuracy_check_agent.invoke({"homework_content": state["homework_content"]})
    return {**state, "accuracy": accuracy}

def calc_score_node(state: CorrectionSubgraphState) -> CorrectionSubgraphState:
    """子图节点3：计算最终得分"""
    print(f"🔍 子图执行 - 计算作业得分")
    # 子图内部空值校验：避免LLM输出异常导致报错
    completion = state["completion"] or "未完成"
    accuracy = state["accuracy"] or "正确率0%"
    # 调用得分智能体并转整数（增加异常捕获，适配LLM偶尔输出非数字的情况）
    try:
        score = int(score_calc_agent.invoke({"completion": completion, "accuracy": accuracy}))
    except:
        score = 0
    return {**state, "score": score}

# 构建并编译子图（独立流程，可复用）
correction_subgraph = StateGraph(CorrectionSubgraphState)
correction_subgraph.add_node("check_completion", check_completion_node)
correction_subgraph.add_node("check_accuracy", check_accuracy_node)
correction_subgraph.add_node("calc_score", calc_score_node)
# 子图线性流程：开始→完成度→正确率→计算得分→结束
correction_subgraph.add_edge(START, "check_completion")
correction_subgraph.add_edge("check_completion", "check_accuracy")
correction_subgraph.add_edge("check_accuracy", "calc_score")
correction_subgraph.add_edge("calc_score", END)
compiled_correction_subgraph = correction_subgraph.compile()


# -------------------------- 第二步：定义主图（作业处理主流程，调用子图）--------------------------
# 主图全局状态：添加Optional，适配初始None值，字段含义学生易理解
class HomeworkMainState(TypedDict):
    homework_content: str                          # 学生作业内容（必传）
    correction_result: Optional[CorrectionSubgraphState] = None  # 子图批改结果（初始None）
    feedback: Optional[str] = None                 # 最终批改反馈（初始None）

# 主图智能体：仅生成批改反馈，逻辑简单
feedback_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是班主任，根据批改结果给学生写1-2句亲切反馈，语气贴合得分情况。"),
    ("user", "作业内容：{homework_content}\n批改结果：完成度{completion}，正确率{accuracy}，得分{score}\n生成反馈：")
])
feedback_agent = feedback_prompt | llm | output_parser

# 主图节点函数（核心：修复子图调用节点的解包顺序，添加空值校验）
def receive_homework_node(state: HomeworkMainState) -> HomeworkMainState:
    """主图节点1：接收学生作业（模拟，日志可视化）"""
    print(f"\n📥 主图执行 - 接收学生作业：{state['homework_content']}")
    return state  # 接收作业，状态无变化

def correction_subgraph_node(state: HomeworkMainState) -> HomeworkMainState:
    """主图节点2：调用作业批改子图（教学核心！重点标注）"""
    print(f"\n📤 主图执行 - 调用作业批改子图")
    # 主图向子图传递参数：仅传子图需要的作业内容，其他用子图默认初始值
    subgraph_input = {"homework_content": state["homework_content"]}
    # 调用编译后的子图，获取完整批改结果
    subgraph_output = compiled_correction_subgraph.invoke(subgraph_input)
    # 🔥 核心修复：解包顺序反了导致的None覆盖问题！先解包原状态，再更新子图结果
    print(f"✅ 主图接收子图批改结果：完成度{subgraph_output['completion']}，正确率{subgraph_output['accuracy']}，得分{subgraph_output['score']}")
    return {**state, "correction_result": subgraph_output}

def generate_feedback_node(state: HomeworkMainState) -> HomeworkMainState:
    """主图节点3：生成批改反馈（添加空值校验，避免报错）"""
    print(f"\n📝 主图执行 - 生成学生批改反馈")
    # 空值校验：防止子图结果未正确更新（双重保险）
    if not state.get("correction_result"):
        return {**state, "feedback": "作业批改失败，无法生成反馈！"}
    # 提取子图批改结果（简化变量名，代码更清晰）
    corr = state["correction_result"]
    homework = state["homework_content"]
    # 调用反馈智能体生成结果
    feedback = feedback_agent.invoke({
        "homework_content": homework,
        "completion": corr["completion"] or "未完成",
        "accuracy": corr["accuracy"] or "正确率0%",
        "score": corr["score"] or 0
    })
    return {**state, "feedback": feedback}

# 构建并编译主图（嵌入子图，流程清晰）
main_graph = StateGraph(HomeworkMainState)
# 添加主图节点：接收作业 → 调用子图 → 生成反馈
main_graph.add_node("receive_homework", receive_homework_node)
main_graph.add_node("correction_subgraph", correction_subgraph_node)
main_graph.add_node("generate_feedback", generate_feedback_node)
# 主图线性流程：严格按教学需求设计，学生易观察
main_graph.add_edge(START, "receive_homework")
main_graph.add_edge("receive_homework", "correction_subgraph")
main_graph.add_edge("correction_subgraph", "generate_feedback")
main_graph.add_edge("generate_feedback", END)
compiled_main_graph = main_graph.compile()


# -------------------------- 第三步：测试运行（学生可直接观察全过程，无报错）--------------------------
if __name__ == "__main__":
    # 测试1：优秀作业（完成+正确率高，预期：正面反馈）
    print("="*60, "测试1：优秀作业（完成+正确率高）", "="*60)
    test1_input = {
        "homework_content": "2+3=5，4+6=10，7+8=15，9+11=20",
        # 初始值为None，无需手动赋值，由子图节点更新
        "correction_result": None,
        "feedback": None
    }
    result1 = compiled_main_graph.invoke(test1_input)
    print(f"\n🎉 最终结果 - 学生反馈：{result1['feedback']}\n")

    # 测试2：不合格作业（未完成+正确率低，预期：改进反馈）
    print("="*60, "测试2：不合格作业（未完成+正确率低）", "="*60)
    test2_input = {
        "homework_content": "2+3=6，4+6=（空白），7+8=（空白）",
        "correction_result": None,
        "feedback": None
    }
    result2 = compiled_main_graph.invoke(test2_input)
    print(f"\n🎉 最终结果 - 学生反馈：{result2['feedback']}")