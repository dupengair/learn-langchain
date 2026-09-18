import langchain
import langgraph
import openai
import importlib
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
# 如按本文前面步骤操作，此时项目中不存在 .env 文件，`load_dotenv()` 函数返回 `False` 是正常的
print("LangChain版本：", langchain.__version__)
print("LangGraph版本：", importlib.metadata.version("langgraph"))
print("OpenAI版本：", openai.__version__)


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
    model=MODEL,         # 注意填写服务商指定的模型名称（见下方说明）
)

response = llm.invoke("你好，请回复'配置成功'")
print(response.content)