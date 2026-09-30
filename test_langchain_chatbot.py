# 1. 导入模块
from llm_config import get_chat_model

# 4. 初始化大模型
llm_chat = get_chat_model(temperature=0.3, max_tokens=200)

# 5. 构造对话消息
# ChatModel需要接收的是“消息列表”，每个消息有角色（user/assistant/system）和内容
messages = [
    # system消息：给助手设定身份和行为准则，会影响后续所有回复
    {"role": "system", "content": "你是一个耐心的AI学习助手，回复简洁易懂，适合高校学生理解。"},
    # user消息：用户的问题
    {"role": "user", "content": "请用3句话解释什么是LangChain？"}
]

# 6. 调用模型生成结果
# 统一调用方法：invoke()，传入消息列表
result = llm_chat.invoke(messages)

# 7. 输出结果
# 结果是一个ChatMessage对象，content属性是回复内容
print("llm_chat 回复：")
print(result.content)