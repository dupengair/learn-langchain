# 1. 导入模块
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda


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
    temperature=0.3,
    max_tokens=1024,        # 最大生成 tokens 数，避免生成过长内容
    extra_body={            # 关闭 Qwen3 的 thinking 模式：
        "chat_template_kwargs": {"enable_thinking": False}  # extra_body 里的字段会原样附加进 OpenAI 请求体
    }
)

# 组件1：提取核心卖点
sell_point_prompt = PromptTemplate(
    input_variables=["product_intro"],
    template="请从以下产品介绍中提取3个核心卖点，用简洁的语言列出：{product_intro}"
)

sell_point_chain = sell_point_prompt | llm

extract_sell_points = RunnableLambda(
    lambda msg: {"sell_points": msg.content}
)

# 组件2：生成营销话术
marketing_prompt = PromptTemplate(
    input_variables=["sell_points"],
    template="请根据以下产品核心卖点，写一段吸引消费者的营销话术（适合朋友圈发布）：{sell_points}"
)

marketing_chain = marketing_prompt | llm

# 线性串联（Sequential Runnable）
overall_chain = (
    sell_point_chain
    | extract_sell_points
    | marketing_chain
)

# 执行
product_intro = """这款无线耳机采用蓝牙5.3芯片，连接稳定无延迟，支持高清通话；续航长达30小时，充电10分钟可使用2小时；机身采用亲肤硅胶材质，佩戴舒适，防水防汗，适合运动使用。"""
result = overall_chain.invoke({"product_intro": product_intro})

print("\n最终营销话术：")
print(result.content)