# 1. 导入需要的模块
import os 
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

# 2. 加载API密钥
load_dotenv()

# 3. 配置 API Key
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")
MODEL = os.getenv("MODEL")

if not API_KEY:
    raise ValueError("未检测到 API_KEY，请检查 .env 文件是否配置正确")
if not BASE_URL:
    raise ValueError("未检测到 BASE_URL，请检查 .env 文件是否配置正确")
if not MODEL:
    raise ValueError("未检测到 MODEL，请检查 .env 文件是否配置正确")

# 4. 初始化大模型（和LangChain案例一样）
llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=MODEL,        # 注意：根据你使用的模型修改名称！！！！ 后面章节不再继续说明
    temperature=0.3,
    max_tokens=1024,        # 最大生成 tokens 数，避免生成过长内容
    extra_body={            # 关闭 Qwen3 的 thinking 模式：
        "chat_template_kwargs": {"enable_thinking": False}  # extra_body 里的字段会原样附加进 OpenAI 请求体
    }
)

# 5. 定义全局状态（所有智能体共享的数据，v1.0.0+ 推荐用TypedDict规范状态）
class AgentState(TypedDict):
    content: Optional[str]  # 短文内容
    error: Optional[str]    # 错误信息
    polished_content: Optional[str]  # 润色后内容

# 6. 初始化3个“专业智能体”（分工明确）
# 智能体1：写短文（只负责“写”，不考虑纠错和润色）
writer_prompt = ChatPromptTemplate.from_messages([
    ("user", "写一篇150字左右、关于“LangGraph多智能体”的短文，语言通俗，适合新手，不用纠错和润色。")
])
writer_agent = writer_prompt | llm

# 智能体2：纠错（只负责“找错+改错”，不修改文风）
corrector_prompt = ChatPromptTemplate.from_messages([
    ("user", "请检查以下短文，修正其中关于LangGraph的技术错误（比如接口、功能描述），只输出修正后的内容，不润色：\n{content}")
])
corrector_agent = corrector_prompt | llm

# 智能体3：润色（只负责“优化语言”，不修改核心内容）
polisher_prompt = ChatPromptTemplate.from_messages([
    ("user", "请润色以下短文，加入1个新手能理解的类比，语言更流畅，不改变核心内容和技术准确性：\n{content}")
])
polisher_agent = polisher_prompt | llm

# 7. 定义节点函数（v1.0.0+ 节点需是可调用函数，接收state，返回更新后的state）
def write_node(state: AgentState) -> AgentState:
    result = writer_agent.invoke({})
    return {"content": result.content, "error": None, "polished_content": None}

def correct_node(state: AgentState) -> AgentState:
    result = corrector_agent.invoke({"content": state["content"]})
    return {"content": result.content, "error": None, "polished_content": None}

def polish_node(state: AgentState) -> AgentState:
    result = polisher_agent.invoke({"content": state["content"]})
    return {"content": state["content"], "error": None, "polished_content": result.content}

# 8. 构建图（v1.0.0+ 用StateGraph构建，简化了旧版本的Graph接口）
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("writer", write_node)  # 写短文节点
workflow.add_node("corrector", correct_node)  # 纠错节点
workflow.add_node("polisher", polish_node)  # 润色节点

# 添加边（定义流程顺序：写→纠错→润色→结束）
workflow.add_edge(START, "writer")
workflow.add_edge("writer", "corrector")
workflow.add_edge("corrector", "polisher")
workflow.add_edge("polisher", END)

# 编译图（v1.0.0+ 必须编译后才能运行）
compiled_workflow: CompiledStateGraph = workflow.compile()

# 9. 运行流程
result = compiled_workflow.invoke({})  # 初始状态为空字典
print("多智能体输出（润色后）：")
print(result["polished_content"])