# 方案1：轻量款（官方推荐，保留MD结构，优先选）
from langchain_community.document_loaders import UnstructuredMarkdownLoader
import os

# 定义MD路径
md_path = os.path.join("knowledge_base", "test.md")

# 加载文档（需提前安装python-markdown）
loader = UnstructuredMarkdownLoader(md_path)
md_docs = loader.load()

# 查看结果
print("\nMarkdown文档加载结果（轻量款）：")
print(f"文档数量：{len(md_docs)}")
print(f"文档内容（保留结构）：{md_docs[0].page_content[:200]}...")
print(f"元数据：{md_docs[0].metadata}")

# 方案2：通用款（适配多格式，含MD/TXT等）
from langchain_community.document_loaders import UnstructuredFileLoader

loader = UnstructuredFileLoader(md_path, mode="elements")  # mode="elements"保留结构
md_docs_univ = loader.load()
print("\nMarkdown文档加载结果（通用款）：")
print(f"内容预览：{md_docs_univ[0].page_content[:200]}...")