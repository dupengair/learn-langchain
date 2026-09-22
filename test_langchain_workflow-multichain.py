from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap, RunnablePassthrough
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


# 2. Prompt 定义
sell_point_prompt = PromptTemplate(
    input_variables=["product_intro"],
    template="从以下产品介绍中提取3个核心卖点，简洁列出：{product_intro}"
)

marketing_prompt = PromptTemplate(
    input_variables=["sell_points", "target_audience"],
    template="针对{target_audience}，结合以下核心卖点，写一段朋友圈营销话术：{sell_points}"
)


# 3. 多输入多输出线性链（教学标准版）
overall_chain = (
    # Step 1：生成卖点 + 透传原始输入
    RunnableMap({
        "sell_points": sell_point_prompt | llm | (lambda x: x.content),
        "target_audience": RunnablePassthrough(),
    })
    # Step 2：营销话术生成
    | marketing_prompt
    | llm
)


# 4. 执行
input_data = {
    "product_intro": "这款无线耳机采用蓝牙5.3芯片，连接稳定无延迟...",
    "target_audience": "大学生群体（喜欢运动、预算有限、注重性价比）"
}

result = overall_chain.invoke(input_data)

print("营销话术：")
print(result.content)