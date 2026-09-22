from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_openai import ChatOpenAI  # 补充LLM定义
from dotenv import load_dotenv
import os


# 1. 环境变量 & 模型
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

# 1️⃣ Prompt
summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "请简洁总结以下文本核心内容，不超过50字。"),
    ("human", "{text}")
])

# 2️⃣ 基础链
base_chain: Runnable = summary_prompt | llm | StrOutputParser()

# 3️⃣ 重试链（官方推荐：直接 with_retry）
retry_chain = base_chain.with_retry(
    stop_after_attempt=3,          # 最多重试 3 次
    wait_exponential_jitter=True,  # 指数退避 + 抖动（推荐）
    retry_if_exception_type=(
        ConnectionError,
        TimeoutError,
    ),
)

# 4️⃣ 调用
try:
    result = retry_chain.invoke({
        "text": "LangChain是一个用于构建大模型应用的框架，提供了丰富的Runnable组件，支持重试、降级等工程化能力。"
    })
    print("总结结果：", result)

except OutputParserException as e:
    # ❗ 解析错误通常是逻辑问题，不建议重试
    print("输出解析失败：", e)

except Exception as e:
    print("最终失败（已达到最大重试次数）：", e)