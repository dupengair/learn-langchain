# 导入必要的模板类
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
import json
from typing import Dict, List

# 1. 环境与模型初始化
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

chat_model = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=MODEL,
    temperature=0.3,        # 随机性：0-1，越小越严谨，越大越有创造力
    max_tokens=1024,         # 最大生成 tokens 数，避免生成过长内容
    extra_body={            # 关闭 Qwen3 的 thinking 模式：
        "chat_template_kwargs": {"enable_thinking": False}  # extra_body 里的字段会原样附加进 OpenAI 请求体
    }
)

# 自定义解析器
class CustomToolParser(BaseOutputParser):
    def parse(self, text: str) -> dict:
        """将模型输出按 '工具名@核心功能@学习难度' 解析为字典"""
        text = text.strip().replace("\n", "").replace(" ", "")
        parts = text.split("@")
        if len(parts) != 3:
            raise ValueError(f"输出格式错误！需满足「工具名@核心功能@学习难度」，当前输出：{text}")
        return {
            "tool_name": parts[0].strip(),
            "function": parts[1].strip(),
            "difficulty": parts[2].strip()
        }

    def get_format_instructions(self) -> str:
        """生成提示词，引导模型按自定义格式输出"""
        return "请严格按照「工具名@核心功能@学习难度」格式输出，不添加多余内容。示例：LangSmith@全链路调试监控@中等"

# 使用解析器
parser = CustomToolParser()
prompt = PromptTemplate(
    template="请介绍1个LangChain开发工具。{format_instructions}",
    input_variables=[],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)
chain = prompt | chat_model | parser
result = chain.invoke({})

print("自定义解析器解析结果：")
print(result)
print("解析结果类型：", type(result))