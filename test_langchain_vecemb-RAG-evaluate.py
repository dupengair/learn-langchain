from datasets import Dataset
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

'''
from ragas import evaluate
from ragas.metrics.collections import (
    ContextPrecision,     # 检索精确率
    ContextRecall,        # 检索召回率
    Faithfulness,         # 事实一致性
    AnswerRelevancy       # 答案相关性
)
'''

# ★ 垫片补丁放最前（第 1 行之前）
import sys, types
_shim = types.ModuleType("langchain_community.chat_models.vertexai")
class ChatVertexAI: pass
_shim.ChatVertexAI = ChatVertexAI
sys.modules["langchain_community.chat_models.vertexai"] = _shim

# ---- 原 11-17 行改为：metrics 用 ragas.metrics（legacy 单例），不再 import collections ----
from ragas import evaluate
from ragas.metrics import (
    context_precision,   # 检索精确率
    context_recall,      # 检索召回率
    faithfulness,        # 事实一致性
    answer_relevancy     # 答案相关性
)
from ragas.run_config import RunConfig


# 1. 初始化大模型
load_dotenv()
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")
MODEL = os.getenv("MODEL")

if not API_KEY:
    raise ValueError("未检测到 API_KEY，请检查 .env 文件是否配置正确")
if not BASE_URL:
    raise ValueError("未检测到 BASE_URL，请检查 .env 文件是否配置正确")
if not MODEL:
    raise ValueError("未检测到 MODEL，请检查 .env 文件是否配置正确")

llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=MODEL,         
    temperature=0.3,        # 随机性：0-1，越小越严谨，越大越有创造力
    max_tokens=1024,        # 最大生成 tokens 数，避免生成过长内容
    extra_body={            # 关闭 Qwen3 的 thinking 模式：
        "chat_template_kwargs": {"enable_thinking": False}  # extra_body 里的字段会原样附加进 OpenAI 请求体
    }
)

# 2. 本地Qwen嵌入模型路径
main_path = "/home/dupengair/shared/LLM/Fine-tuning/"
model = "model/Qwen3-Embedding-0.6B"
embedding_model_name = main_path+model
# 验证模型路径有效性
if not os.path.exists(embedding_model_name):
    raise FileNotFoundError(f"Qwen3嵌入模型路径不存在：{embedding_model_name}")

# 初始化本地CPU运行的嵌入模型
embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs={
       # "device": "cpu"  # 强制使用CPU运行，无需GPU
    },
    encode_kwargs={
        "normalize_embeddings": True  # 归一化向量，提升检索效果
    }
)

# 3. 加载FAISS向量库
faiss_db_path = "./faiss_db"
if not os.path.exists(faiss_db_path):
    raise FileNotFoundError(f"FAISS向量数据库路径不存在：{faiss_db_path}")

vector_db = FAISS.load_local(
    folder_path=faiss_db_path,
    embeddings=embeddings,
    allow_dangerous_deserialization=True,
    index_name="local_cpu_faiss_index"  # 确保加载正确的索引文件
)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# 4. 构建检索-生成链
# 定义提示词
system_prompt = "基于以下上下文回答问题：{context}"
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}")
])

# 文档格式化函数
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

# 修复LCEL链语法（管道符连接）
rag_qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 5. 构建测试数据集
test_questions = [
    "RAG系统的核心价值是什么？",
    "SequentialChain的作用是什么？",
    "RAG系统的构建流程有哪些步骤？"
]

# 6. 采集RAG输出结果
test_data = []
for question in test_questions:
    answer = rag_qa_chain.invoke(question)
    retrieved_docs = retriever.invoke(question)
    contexts = [doc.page_content for doc in retrieved_docs]
    '''
    test_data.append({
        "question": question,
        "answer": answer,
        "contexts": contexts
    })
    '''
    # ---- 原 104-112 行：给每条样本补上参考答案 ground_truth ----
    test_data.append({
        "question": question,
        "answer": answer,
        "contexts": contexts,
        "ground_truth": "……人工/半自动撰写的标准答案……",   # ★ 新增
    })


# 7. 转换为RAGAS标准格式
dataset = Dataset.from_list(test_data)

# 5. 评估指标
#metrics = [ContextPrecision(), ContextRecall(), Faithfulness(), AnswerRelevancy()]
# ---- 原 118 行改为：单例指标无需（也不能）传 llm 构造 ----
metrics = [context_precision, context_recall, faithfulness, answer_relevancy]

# 8. 执行评估
'''
results = evaluate(
    dataset=dataset,
    metrics=metrics,
    llm=llm
)
'''
# ---- 原 121-125 行改为：补齐 base_url/model/变量名/embeddings ----
results = evaluate(
    dataset=dataset,
    metrics=metrics,
    llm=llm,                # 复用脚本开头那个配好 base_url 的 llm（★ 不是新建 ChatOpenAI）
    embeddings=embeddings,  # 复用已加载的 HuggingFaceEmbeddings（answer_relevancy 需要）
    run_config=RunConfig(max_retries=3, max_wait=5, timeout=60),  # 可选：控制超时重试
)

# 79. 输出结果
print("RAG系统自动化评估结果：")
print(results)