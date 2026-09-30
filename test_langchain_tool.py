from langchain.agents import create_agent
from langchain_core.tools import tool
from llm_config import get_chat_model
import langchain, langgraph
import importlib
import openai
# ======================
# 1. 环境
# ======================

llm = get_chat_model(temperature=0.3, max_tokens=1024)


# ======================
# 2. 工具
# ======================
@tool
def weather_query(city: str) -> str:
    """查询指定城市天气"""
    weather_data = {
        "北京": "北京今日天气：晴，-2~8℃",
        "上海": "上海今日天气：多云，5~12℃",
        "广州": "广州今日天气：小雨，18~25℃",
    }
    return weather_data.get(city, f"暂无 {city} 数据")

tools = [weather_query]


# ======================
# 3. 创建 Agent（开启 debug）
# ======================
agent = create_agent(
    model=llm,
    tools=tools,
    debug=True,  # 👈 打开过程打印
)


# ======================
# 4. 运行
# ======================
response = agent.invoke({
    "messages": [
        {"role": "user", "content": "北京今天的天气怎么样？"}
    ]
})

print("\n最终回答：")
print(response["messages"][-1].content)