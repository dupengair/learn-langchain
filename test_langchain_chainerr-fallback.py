from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableWithFallbacks
from langchain_core.output_parsers import StrOutputParser
from langchain_core.exceptions import OutputParserException
from llm_config import get_chat_model
# 1️⃣ 核心链（性能高但可能不稳定）
core_llm = get_chat_model(temperature=0.3, max_tokens=1024)

core_prompt = ChatPromptTemplate.from_messages([
    ("system", "用专业语言详细解答用户问题。"),
    ("human", "{query}")
])
core_chain = core_prompt | core_llm | StrOutputParser()

# 2️⃣ 降级链（稳定但精度略低）
fallback_llm = get_chat_model(temperature=0.3, max_tokens=1024)

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