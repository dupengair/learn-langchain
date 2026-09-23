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

# 相似性检索（k=2）
retriever_similar_k2 = vector_db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}
)

# 相似性检索（k=5）
retriever_similar_k5 = vector_db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# MMR检索（偏向相关性，lambda_mult=0.8）
retriever_mmr_high_rel = vector_db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.8}
)

# MMR检索（偏向多样性，lambda_mult=0.3）
retriever_mmr_high_div = vector_db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.3}
)

# 定义测试查询并执行检索（适配v0.1+ invoke方法）
test_query = "RAG系统的核心价值是什么？"

def test_retriever(retriever: BaseRetriever, retriever_name: str):
    """测试检索器并打印结果"""
    try:
        # 核心适配：使用invoke()替代旧的get_relevant_documents()
        results: list[Document] = retriever.invoke(test_query)
        print(f"=== {retriever_name} 检索结果（共{len(results)}条） ===")
        for i, doc in enumerate(results):
            print(f"\n[{i+1}] 内容：{doc.page_content[:120]}...")
            print(f"   来源：{doc.metadata.get('source', '未知')}")
        print("\n" + "-"*80 + "\n")
    except Exception as e:
        raise RuntimeError(f"{retriever_name} 检索失败：{str(e)}")

# 执行所有检索并打印结果
test_retriever(retriever_similar_k2, "相似性检索（k=2）")
test_retriever(retriever_similar_k5, "相似性检索（k=5）")
test_retriever(retriever_mmr_high_rel, "MMR检索（偏向相关性 λ=0.8）")
test_retriever(retriever_mmr_high_div, "MMR检索（偏向多样性 λ=0.3）")

# 可选：补充相似性评分展示（FAISS特有）
print("=== 相似性检索（k=3）带评分结果 ===")
docs_with_scores = vector_db.similarity_search_with_score(test_query, k=3)
for i, (doc, score) in enumerate(docs_with_scores):
    print(f"\n[{i+1}] 评分：{round(score, 4)} | 内容：{doc.page_content[:80]}...")