from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import langchain, langgraph
import importlib
import openai
import os


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

# 手动实现全量记忆（无LangChain框架，理解核心逻辑）
chat_history = []  # 存储对话历史的列表

def chat_with_memory(user_input):
    # 1. 拼接历史+新问题
    prompt = "你是友好的助手，结合历史对话回答：\n"
    for msg in chat_history:
        prompt += f"{msg['role']}: {msg['content']}\n"
    prompt += f"用户：{user_input}"
    
    # 2. 调用LLM
    response = llm.invoke(prompt).content
    
    # 3. 保存新对话到历史
    chat_history.append({"role": "用户", "content": user_input})
    chat_history.append({"role": "AI", "content": response})
    
    return response

# 测试
print(chat_with_memory("我叫小明"))
print(chat_with_memory("我刚才叫什么名字？"))
print(chat_history)
