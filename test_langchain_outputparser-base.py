# 导入必要的模板类
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate
from llm_config import get_chat_model
from pydantic import BaseModel, Field
import json
from typing import Dict, List

# 1. 环境与模型初始化

chat_model = get_chat_model(temperature=0.3, max_tokens=1024)

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