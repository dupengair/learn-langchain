# 1. 导入模块
from llm_config import get_chat_model

# 4. 初始化大模型
llm = get_chat_model(temperature=0.3, max_tokens=1024)

# 5. 构造 Prompt（教学阶段用字符串更直观）
prompt = "请写一段50字左右的 AI 学习建议，语言简洁、实用，适合初学者。"

# 6. 调用模型
response = llm.invoke(prompt)

# 7. 输出结果
print("生成的学习建议：")
print(response.content)
