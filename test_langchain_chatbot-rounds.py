# 1. 导入模块
from llm_config import get_chat_model

# 4. 初始化大模型
llm_chat = get_chat_model(temperature=0.3, max_tokens=200)

# 5.初始化对话历史（包含 system 设定）
history = [
    {"role": "system", "content": "你是一个耐心的AI学习助手，回复简洁易懂，适合高校学生理解。"}
]

# 6.第一轮对话
history.append({"role": "user", "content": "请用3句话解释什么是LangChain？"})

result = llm_chat.invoke(history)
print("【第一轮回复】：")
print(result.content)

# 7.将模型的回复添加到历史中（assistant 消息）
history.append({"role": "assistant", "content": result.content})

# 8.第二轮对话
# 追问，模型需要上下文才能理解"它"
history.append({"role": "user", "content": "它的核心组件有哪些？"})

result = llm_chat.invoke(history)
print("\n【第二轮回复】：")
print(result.content)

# 9.继续记录
history.append({"role": "assistant", "content": result.content})

# 10.第三轮对话
history.append({"role": "user", "content": "给我一个简单的使用场景"})

result = llm_chat.invoke(history)
print("\n【第三轮回复】：")
print(result.content)