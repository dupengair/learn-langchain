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
llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=MODEL,  # 注意：根据你使用的模型修改名称！！！！ 后面章节不再继续说明
    temperature=0.3
)

# 5. 构造 Prompt（教学阶段用字符串更直观）
prompt = "请写一段50字左右的 AI 学习建议，语言简洁、实用，适合初学者。"

# 6. 调用模型
response = llm.invoke(prompt)

# 7. 输出结果
print("生成的学习建议：")
print(response.content)
