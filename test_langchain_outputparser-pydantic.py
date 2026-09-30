# 导入必要的模板类
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from llm_config import get_chat_model
from pydantic import BaseModel, Field
import json
from typing import Dict, List

# 1. 环境与模型初始化

chat_model = get_chat_model(temperature=0.3, max_tokens=1024)

# 2. 定义 Pydantic 数据模型
class ToolInfo(BaseModel):
    tool_name: str = Field(description="LangChain开发工具的名称，如 LangSmith")
    function: str = Field(description="工具的核心功能，30字以内")
    difficulty: str = Field(description="学习难度，仅可选：简单 / 中等 / 复杂")

# 3. 创建解析器
parser = PydanticOutputParser(pydantic_object=ToolInfo)

# 4. Prompt + Chain
prompt = PromptTemplate(
    template="{user_input}，严格按照要求输出。\n{format_instructions}",
    input_variables=["user_input"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)

chain = prompt | chat_model | parser
result = chain.invoke({"user_input": "请介绍1个 Python 开发工具"})

print("解析后的结构化数据（Pydantic 模型对象）：")
print(result)

print("字段校验 difficulty：", result.difficulty)

print("转化为字典：", result.model_dump())