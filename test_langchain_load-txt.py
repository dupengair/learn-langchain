# 导入TXT加载器（新版路径：langchain_community.document_loaders）
from langchain_community.document_loaders import TextLoader
import os

# 定义文档路径（请替换成你自己的路径）
txt_path = os.path.join("knowledge_base", "test.txt")

# 初始化加载器并加载文档
loader = TextLoader(txt_path, encoding="utf-8")  # 指定编码，避免中文乱码
txt_docs = loader.load()  # load()返回Document对象列表（即使只有一个文档）

# 查看加载结果
print("TXT文档加载结果：")
print(f"文档数量：{len(txt_docs)}")
print(f"文档内容：{txt_docs[0].page_content[:200]}...")  # 打印前200个字符
print(f"文档元数据：{txt_docs[0].metadata}")  # 元数据包含文档路径等信息