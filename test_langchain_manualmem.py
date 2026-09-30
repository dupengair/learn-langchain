from llm_config import get_chat_model
import langchain, langgraph
import importlib
import openai
llm = get_chat_model(temperature=0.3, max_tokens=1024)

# 手动实现全量记忆（无LangChain框架，理解核心逻辑）
chat_history = []  # 存储对话历史的列表

def chat_with_memory(user_input):
    # 1. 拼接历史+新问题
    prompt = "你是友好的助手，结合历史对话回答：\n"
    for msg in chat_history:
        prompt += f"{msg['role']}: {msg['content']}\n"
    prompt += f"用户：{user_input}"
    
    # 2. 调用LLM
    response = llm.invoke(prompt).content
    
    # 3. 保存新对话到历史
    chat_history.append({"role": "用户", "content": user_input})
    chat_history.append({"role": "AI", "content": response})
    
    return response

# 测试
print(chat_with_memory("我叫小明"))
print(chat_with_memory("我刚才叫什么名字？"))
print(chat_history)
