# 导入Word加载器（兼容 langchain_community 0.4.1）
from langchain_community.document_loaders import Docx2txtLoader
import os

# 定义Word路径
docx_path = os.path.join("knowledge_base", "test.docx")

# 加载文档（需提前安装 docx2txt：pip install docx2txt）
loader = Docx2txtLoader(docx_path)
docx_docs = loader.load()

# 查看结果
print("\nWord文档加载结果：")
print(f"文档数量：{len(docx_docs)}")
print(f"文档内容：{docx_docs[0].page_content[:200]}...")
print(f"元数据：{docx_docs[0].metadata}")