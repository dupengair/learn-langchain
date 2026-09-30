# 导入必要的模板类
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from llm_config import get_chat_model
chat_model = get_chat_model(temperature=0.3, max_tokens=1024)


# 1. 定义示例（少样本的核心：给模型看的参考案例）
examples = [
    {
        "subject": "Python编程",
        "method": "核心目标：掌握基础语法和常用库；学习步骤：1. 学习变量、函数等基础语法 2. 实操小项目（如计算器） 3. 学习Pandas、Matplotlib库；注意事项：多动手实操，遇到错误及时调试。"
    },
    {
        "subject": "机器学习",
        "method": "核心目标：理解基础算法原理和应用场景；学习步骤：1. 复习数学基础（线性代数、概率） 2. 学习经典算法（线性回归、决策树） 3. 用Scikit-learn实操；注意事项：先理解原理，再动手实现，避免死记硬背。"
    }
]

# 2. 定义示例模板（告诉模型如何解析示例）
example_template = """
学科：{subject}
学习方法：{method}
"""
example_prompt = PromptTemplate(
    input_variables=["subject", "method"],
    template=example_template
)

# 3. 定义最终的提示词模板（包含示例和用户需求）
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,                # 传入示例
    example_prompt=example_prompt,    # 示例模板
    suffix="学科：{new_subject}\n学习方法：",  # 最终给用户的提示（在示例之后）
    input_variables=["new_subject"]   # 动态参数：用户要查询的新学科
)

# 4. 格式化模板（传入新学科：LangChain）
formatted_prompt = few_shot_prompt.format(new_subject="LangChain")
print("少样本提示词：")
print(formatted_prompt)

# 5. 调用模型生成结果
result = chat_model.invoke([{"role": "user", "content": formatted_prompt}])

print("\n生成的LangChain学习方法：")
print(result.content)