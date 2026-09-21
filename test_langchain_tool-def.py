from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
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


# ======================
# 2. 参数模型
# ======================
class TemperatureConvertInput(BaseModel):
    temperature: float = Field(description="需要转换的温度值，例如37.0")
    from_unit: str = Field(description="原始温度单位，只能是celsius或fahrenheit")


# ======================
# 3. 工具
# ======================
@tool(args_schema=TemperatureConvertInput)
def temperature_converter(temperature: float, from_unit: str) -> str:
    """温度单位转换工具"""
    
    if from_unit not in ["celsius", "fahrenheit"]:
        return f"错误：单位'{from_unit}'不合法"

    if from_unit == "celsius":
        fahrenheit = temperature * 9/5 + 32
        return f"{temperature}摄氏度 = {fahrenheit:.2f}华氏度"
    else:
        celsius = (temperature - 32) * 5/9
        return f"{temperature}华氏度 = {celsius:.2f}摄氏度"


tools = [temperature_converter]

system_prompt = """
你是一名专业温度转换助手，只能使用temperature_converter工具完成计算。
"""

# ======================
# 4. 创建 Agent
# ======================
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt,
    debug=True
)


query = "将37摄氏度转换为华氏度"

response = agent.invoke({
    "messages": [{"role": "user", "content": query}]
})

print("\n最终结果：")
print(response["messages"][-1].content)