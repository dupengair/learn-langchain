# 导入PromptTemplate
from langchain_core.prompts import PromptTemplate
from llm_config import get_chat_model
chat_model = get_chat_model(temperature=0.3, max_tokens=1024)

# 1. 定义提示词模板
# input_variables：动态参数列表（这里是user_role和subject）
# template：提示词模板字符串，用{参数名}表示动态参数
prompt_template = PromptTemplate(
    input_variables=["user_role", "subject"],
    template="请给{user_role}写一段50字左右的{subject}学习建议，语言简洁实用，分2个小要点。"
)

# 2. 格式化模板（传入具体参数，生成完整提示词）
# 给“高校学生”生成“LangChain”学习建议
formatted_prompt = prompt_template.format(
    user_role="高校学生",
    subject="LangChain"
)
print("格式化后的提示词：")
print(formatted_prompt)

# 3. 调用模型生成结果
result = chat_model.invoke([{"role": "user", "content": formatted_prompt}])

print("\n生成的学习建议：")
print(result.content)