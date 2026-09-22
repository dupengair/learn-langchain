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

# 1️⃣ 定义多变量 Prompt 链（营销话术生成示例）
marketing_prompt = ChatPromptTemplate.from_messages([
    ("system", "根据产品卖点和目标人群，撰写一句营销话术。"),
    ("human", "产品卖点：{sell_points}，目标人群：{target_audience}")
])

# 2️⃣ 构建链
marketing_chain: Runnable = marketing_prompt | llm | StrOutputParser()

# 3️⃣ 调用并捕获异常（官方推荐风格）
inputs = {
    "sell_points": "无线耳机续航30小时",
    # "target_audience" 故意缺失，用于演示 KeyError
}

try:
    result = marketing_chain.invoke(inputs)
    print("营销话术：", result)

except KeyError as e:
    # 抛出 KeyError 异常，直接提取缺失的变量名
    missing_var = str(e).strip("'\"")
    print(f"错误提示：缺少必要输入变量 [{missing_var}]，请检查输入数据是否完整。")
except OutputParserException as e:
    # 官方推荐：逻辑解析错误不重试
    print(f"解析失败：{e}，请确认 Prompt 与输出格式匹配。")

except Exception as e:
    # ❗ 兜底捕获未知异常
    print(f"未知错误：{type(e).__name__}: {e}，请联系开发者排查。")