from langchain.agents import create_agent
from langchain_core.tools import tool
from llm_config import get_chat_model
from langchain_community.agent_toolkits import FileManagementToolkit
import langchain, langgraph
import importlib
import openai
# ======================
# 1. 环境
# ======================

llm = get_chat_model(temperature=0.3, max_tokens=1024)


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

