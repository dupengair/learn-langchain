from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableWithFallbacks
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

# 1️⃣ 核心链（性能高但可能不稳定）
core_llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=MODEL,         
    temperature=0.3,        # 随机性：0-1，越小越严谨，越大越有创造力
    max_tokens=1024,        # 最大生成 tokens 数，避免生成过长内容
    extra_body={            # 关闭 Qwen3 的 thinking 模式：
        "chat_template_kwargs": {"enable_thinking": False}  # extra_body 里的字段会原样附加进 OpenAI 请求体
    }
)

core_prompt = ChatPromptTemplate.from_messages([
    ("system", "用专业语言详细解答用户问题。"),
    ("human", "{query}")
])
core_chain = core_prompt | core_llm | StrOutputParser()

# 2️⃣ 降级链（稳定但精度略低）
fallback_llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=MODEL,         
    temperature=0.3,        # 随机性：0-1，越小越严谨，越大越有创造力
    max_tokens=1024,        # 最大生成 tokens 数，避免生成过长内容
    extra_body={            # 关闭 Qwen3 的 thinking 模式：
        "chat_template_kwargs": {"enable_thinking": False}  # extra_body 里的字段会原样附加进 OpenAI 请求体
    }
)

fallback_prompt = ChatPromptTemplate.from_messages([
    ("system", "用简洁语言解答用户问题，保证信息准确。"),
    ("human", "{query}")
])
fallback_chain = fallback_prompt | fallback_llm | StrOutputParser()

# 3️⃣ 构建带降级的链
chain_with_fallback: RunnableWithFallbacks = core_chain.with_fallbacks(
    fallbacks=[fallback_chain],
    exceptions_to_handle=(ConnectionError, TimeoutError),# ✅ 官方推荐：只捕获临时错误或网络错误
)

# 4️⃣ 调用链并捕获异常
try:
    result = chain_with_fallback.invoke({"query": "什么是RAG技术？"})
    print("解答：", result)
except OutputParserException as e:
    print(f"解析失败：{e}")
except Exception as e:
    print(f"最终失败：{e}")