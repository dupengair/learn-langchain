# 导入必要的模板类
from langchain_core.output_parsers import StrOutputParser
from llm_config import get_chat_model
import json
from typing import Dict, List



chat_model = get_chat_model(temperature=0.3, max_tokens=1024)

# 创建 StrOutputParser
# 核心作用：将 LLM 返回的 AIMessage 对象，统一转为纯字符串（str）
parser = StrOutputParser()

# 链式调用：模型 → 字符串解析
chain = chat_model | parser
result = chain.invoke("请简要介绍 LangChain 输出解析层的作用")

print("StrOutputParser 解析后的字符串：")
print(result)
print("\n解析结果类型：", type(result))  # str