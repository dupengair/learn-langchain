# 导入必要的模板类
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from llm_config import get_chat_model
import json
from typing import Dict, List



chat_model = get_chat_model(temperature=0.3, max_tokens=1024)

# 2. 创建 JSON 解析器（无需额外配置，默认引导模型输出 JSON）
parser = JsonOutputParser()

# 3. 构建提示模板（无需手动嵌入格式指令，解析器自动关联）
prompt = PromptTemplate(
    template="请介绍1个LangChain开发工具，输出工具名和核心功能。{format_instructions}",
    input_variables=[],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# 4. 链式调用（LangChain ≥1.0.0 推荐方式，自动完成提示+调用+解析）
chain = prompt | chat_model | parser
result = chain.invoke({})  # 无输入参数，传入空字典

print("解析后的JSON（Python字典）：")
print(result)
print("获取单个字段：", result.get('tool_name', None))  # 可直接用于业务逻辑