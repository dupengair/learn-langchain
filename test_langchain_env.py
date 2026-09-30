from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from llm_config import get_chat_model
import langchain, langgraph
import importlib
import openai
print("LangChain版本：", langchain.__version__)
print("LangGraph版本：", importlib.metadata.version("langgraph"))
print("OpenAI版本：", openai.__version__)



llm = get_chat_model()

response = llm.invoke("你好，请回复'配置成功'")
print(response.content)