from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI  # 1.x推荐用ChatOpenAI，适配对话模型，功能更全
from langchain_core.prompts import ChatPromptTemplate  # 替代PromptTemplate，适配LCEL
from langchain_core.runnables import RunnablePassthrough  # LCEL核心组件，传递数据
from langchain_core.output_parsers import StrOutputParser  # 统一输出格式解析
from dotenv import load_dotenv
import os


# =========================
# 1. 环境准备与组件初始化
# =========================

# 1. 本地Qwen嵌入模型路径
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

# 2. 加载FAISS向量数据库（1.x版本用法完全兼容，保留原逻辑）
# 注意：首次构建时需用FAISS.from_documents(docs, embeddings)创建并save_local
vector_db = FAISS.load_local(
    folder_path="./faiss_db",
    embeddings=embeddings,
    allow_dangerous_deserialization=True,  # 本地开发可用，生产环境需谨慎（存在安全风险）
    index_name="local_cpu_faiss_index"  # 确保加载正确的索引文件
)

# 3. 初始化检索器（MMR策略，平衡相关性和多样性，参数无变化）
retriever = vector_db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.7}
)

# 4. 初始化大模型
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


# =========================
# 2. 用LCEL构建检索-生成链
# =========================

# 1. 自定义文档格式化函数（将检索到的多个文档拼接为统一文本，供提示词使用）
def format_docs(docs):
    """格式化检索到的文档片段，用空行分隔"""
    return "\n\n".join([doc.page_content for doc in docs])

# 2. 自定义提示词模板（1.x推荐用ChatPromptTemplate，通过from_messages创建）
# 保持原业务规则：基于参考资料、分点说明、带案例
system_prompt = """你是一个专业的RAG系统问答助手，必须基于以下提供的参考资料（context）回答用户问题。
规则：
1. 答案必须严格基于参考资料，不能编造未提及的信息；
2. 语言简洁明了，分点说明（如果有多个要点）；
3. 每个要点搭配1个简单案例，帮助理解。
参考资料：{context}"""

custom_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),  # 系统指令
    ("human", "{question}")     # 用户问题（1.x推荐用"question"键，语义更清晰）
])

# 3. 用LCEL构建完整检索-生成链（管道符串联组件，数据流式传递）
rag_qa_chain = (
    # 第一步：并行处理输入（传递用户问题+检索文档）
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    # 第二步：将格式化数据传入提示词模板
    | custom_prompt
    # 第三步：传入大模型生成答案
    | llm
    # 第四步：解析输出（统一为字符串格式）
    | StrOutputParser()
)

# 补充：如需返回检索的源文档（用于验证答案来源），可调整链结构：
rag_qa_chain_with_sources = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
        "source_documents": retriever  # 保留原始检索文档
    }
    | custom_prompt
    | llm
    | StrOutputParser()
)

# =========================
# 3. 测试RAG系统并验证结果
# =========================

# 测试问题列表（覆盖不同类型的查询）
test_questions = [
    "RAG系统的核心价值是什么？",
    "RAG和直接使用大模型相比，优势在哪里？",
    "企业内部知识库问答为什么适合用RAG？"
]

# 执行测试并打印结果
for i, question in enumerate(test_questions):
    print(f"\n===== 测试问题{i+1}：{question} =====")
    # 执行RAG链（1.x统一用invoke方法）
    result = rag_qa_chain_with_sources.invoke(question)  # 带源文档的链
    
    # 打印生成的答案
    print("\n生成答案：")
    print(result)
    
    # 打印参考资料（验证答案来源）
    print("\n参考资料：")
    # 注意：源文档从链的输入参数中获取（因链结构中保留了source_documents）
    sources = retriever.invoke(question)  # 重新调用检索器获取源文档（或在链中传递）
    for j, doc in enumerate(sources):
        print(f"\n参考片段{j+1}：")
        print(doc.page_content)
        if doc.metadata:  # 打印文档元数据（如文件名、页码等）
            print(f"元数据：{doc.metadata}")




