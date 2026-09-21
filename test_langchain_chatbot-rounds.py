# 1. 导入模块
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 2. 加载 .env 环境变量
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

# 4. 初始化大模型
llm_chat = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=MODEL,  
    temperature=0.3,        # 随机性：0-1，越小越严谨，越大越有创造力
    max_tokens=200          # 最大生成 tokens 数，避免生成过长内容
)

# 5.初始化对话历史（包含 system 设定）
history = [
    {"role": "system", "content": "你是一个耐心的AI学习助手，回复简洁易懂，适合高校学生理解。"}
]

# 6.第一轮对话
history.append({"role": "user", "content": "请用3句话解释什么是LangChain？"})

result = llm_chat.invoke(history)
print("【第一轮回复】：")
print(result.content)

# 7.将模型的回复添加到历史中（assistant 消息）
history.append({"role": "assistant", "content": result.content})

# 8.第二轮对话
# 追问，模型需要上下文才能理解"它"
history.append({"role": "user", "content": "它的核心组件有哪些？"})

result = llm_chat.invoke(history)
print("\n【第二轮回复】：")
print(result.content)

# 9.继续记录
history.append({"role": "assistant", "content": result.content})

# 10.第三轮对话
history.append({"role": "user", "content": "给我一个简单的使用场景"})

result = llm_chat.invoke(history)
print("\n【第三轮回复】：")
print(result.content)