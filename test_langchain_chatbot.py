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

# 5. 构造对话消息
# ChatModel需要接收的是“消息列表”，每个消息有角色（user/assistant/system）和内容
messages = [
    # system消息：给助手设定身份和行为准则，会影响后续所有回复
    {"role": "system", "content": "你是一个耐心的AI学习助手，回复简洁易懂，适合高校学生理解。"},
    # user消息：用户的问题
    {"role": "user", "content": "请用3句话解释什么是LangChain？"}
]

# 6. 调用模型生成结果
# 统一调用方法：invoke()，传入消息列表
result = llm_chat.invoke(messages)

# 7. 输出结果
# 结果是一个ChatMessage对象，content属性是回复内容
print("llm_chat 回复：")
print(result.content)