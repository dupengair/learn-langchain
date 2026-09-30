from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain_core.exceptions import OutputParserException
from llm_config import get_chat_model
# 1. 环境变量 & 模型

llm = get_chat_model(temperature=0.3, max_tokens=1024)

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