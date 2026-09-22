# 方案1：基础款（官方推荐，仅提取文本，轻量）
from langchain_community.document_loaders import PyPDFLoader
import os

# 定义PDF路径
pdf_path = os.path.join("knowledge_base", "test.pdf")

# 初始化加载器并加载（按页拆分）
loader = PyPDFLoader(pdf_path)
pdf_docs = loader.load_and_split()  # load_and_split()直接按页拆分，更易用

# 查看结果
print("\nPDF文档加载结果（基础款）：")
print(f"PDF总页数：{len(pdf_docs)}")
print(f"第1页内容：{pdf_docs[0].page_content[:200]}...")
print(f"第1页元数据：{pdf_docs[0].metadata}")  # 元数据包含页码、文档路径

# 方案2：复杂款（需保留表格/格式时用）
from langchain_community.document_loaders import PDFPlumberLoader

loader = PDFPlumberLoader(pdf_path)
pdf_docs_adv = loader.load()
print("\nPDF文档加载结果（复杂款）：")
print(f"第1页表格/格式保留情况：{pdf_docs_adv[0].page_content[:200]}...")