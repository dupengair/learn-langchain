from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain_core.exceptions import OutputParserException
from llm_config import get_chat_model
# 1. 环境变量 & 模型

llm = get_chat_model(temperature=0.3, max_tokens=1024)

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