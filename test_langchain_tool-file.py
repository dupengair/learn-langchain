from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import FileManagementToolkit
from dotenv import load_dotenv
import langchain, langgraph
import importlib
import openai
import os


# ======================
# 1. 环境
# ======================
load_dotenv()
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")
MODEL = os.getenv("MODEL")

if not API_KEY:
    raise ValueError("未检测到 API_KEY，请检查 .env 文件是否配置正确")
if not BASE_URL:
    raise ValueError("未检测到 BASE_URL，请检查 .env 文件是否配置正确")
if not MODEL:
    raise ValueError("未检测到 MODEL，请检查 .env 文件是否配置正确")

llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=MODEL,         
    temperature=0.3,        # 随机性：0-1，越小越严谨，越大越有创造力
    max_tokens=1024,        # 最大生成 tokens 数，避免生成过长内容
    extra_body={            # 关闭 Qwen3 的 thinking 模式：
        "chat_template_kwargs": {"enable_thinking": False}  # extra_body 里的字段会原样附加进 OpenAI 请求体
    }
)


# -------------------
# 2. 创建文件管理工具
# -------------------
toolkit = FileManagementToolkit(root_dir="./docs")
tools = toolkit.get_tools()


# -------------------
# 3. 创建 Agent（最新版）
# -------------------
agent = create_agent(
    model=llm,
    tools=tools,
    debug=True,  # 打开调试，显示模型思考和工具调用过程
)


# -------------------
# 4. 执行任务
# -------------------
response = agent.invoke({
    "messages": [
        {"role": "user", "content": "列出当前目录下的文件"}
    ]
})

print("\n任务执行完成！")
print("Agent最终输出：\n", response["messages"][-1].content)

