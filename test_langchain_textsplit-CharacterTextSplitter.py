from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader 
from pathlib import Path

# 加载文档
txt_path = Path("knowledge_base") / "test.txt"
loader = TextLoader(txt_path, encoding="utf-8")
txt_docs = loader.load()

# 初始化（官方推荐设置自然分隔符）
text_splitter = CharacterTextSplitter(
    separator="\n\n",        # 优先按空行分割，减少语义破坏
    chunk_size=300,
    chunk_overlap=50,
    length_function=len,
    keep_separator=False     # 官方默认False，不保留分隔符（避免片段冗余）
)

split_docs = text_splitter.split_documents(txt_docs)

# 验证结果
print(f"分割后片段数：{len(split_docs)}")