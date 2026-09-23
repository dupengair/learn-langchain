import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document


# 本地Qwen嵌入模型路径
main_path = "/home/dupengair/shared/LLM/Fine-tuning/"
model = "model/Qwen3-Embedding-0.6B"
embedding_model_name = main_path+model

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

# 加载已有的FAISS向量数据库
try:
    vector_db = FAISS.load_local(
        folder_path="./faiss_db",  # 之前的持久化路径
        embeddings=embeddings,
        allow_dangerous_deserialization=True, 
        index_name="local_cpu_faiss_index"
    )
    print("FAISS向量库加载成功！")
except FileNotFoundError:
    raise FileNotFoundError("未找到 ./faiss_db 文件夹，请确认向量库已正确保存")
except Exception as e:
    raise RuntimeError(f"加载FAISS向量库失败：{str(e)}")

# 创建不同类型的检索器（相似性+MMR对比）
# 相似性检索（默认）
retriever_similar: BaseRetriever = vector_db.as_retriever(
    search_type="similarity",  # 检索类型：纯相似性
    search_kwargs={"k": 3}     # 返回3个最相似的片段
)

# MMR检索（最大化边际相关性，兼顾相关性和多样性）
retriever_mmr: BaseRetriever = vector_db.as_retriever(
    search_type="mmr",  # 检索类型：MMR
    search_kwargs={
        "k": 3,         # 最终返回3个片段
        "fetch_k": 10,  # 先检索10个最相关的，再从中选多样化的
        "lambda_mult": 0.5  # 权重：0=只看多样性，1=只看相关性，0.5平衡
    }
)

# 测试对比两种检索方式（适配v0.1+的invoke方法）
query = "LangChain的链有哪些类型？"

# 相似性检索测试
print("=== 相似性检索结果 ===")
try:
    similar_docs: list[Document] = retriever_similar.invoke(query)  # 替换get_relevant_documents
    for i, doc in enumerate(similar_docs):
        print(f"\n片段{i+1}：{doc.page_content[:100]}...")
        print(f"来源文件：{doc.metadata.get('source', '未知')}")
except Exception as e:
    raise RuntimeError(f"相似性检索失败：{str(e)}")

# MMR检索测试
print("\n=== MMR检索结果 ===")
try:
    mmr_docs: list[Document] = retriever_mmr.invoke(query)  # 替换get_relevant_documents
    for i, doc in enumerate(mmr_docs):
        print(f"\n片段{i+1}：{doc.page_content[:100]}...")
        print(f"来源文件：{doc.metadata.get('source', '未知')}")
except Exception as e:
    raise RuntimeError(f"MMR检索失败：{str(e)}")

# MMR检索的评分对比（便于理解差异）
print("\n===== 相似性检索（带评分） =====")
similar_docs_with_score = vector_db.similarity_search_with_score(query, k=3)
for i, (doc, score) in enumerate(similar_docs_with_score):
    print(f"片段{i+1}（评分：{round(score,4)}）：{doc.page_content[:80]}...")