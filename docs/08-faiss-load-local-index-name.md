# 08 - FAISS.load_local 报 "could not open faiss_db/index.faiss" 排查记录

> 对应脚本：`test_langchain_vecemb-RAG-evaluate.py`（RAG + RAGAS 评估）
> 日期：2026-09-23

## 1. 现象

运行脚本时在 **加载 FAISS 向量库** 一步直接崩溃：

```
$ python ./test_langchain_vecemb-RAG-evaluate.py
Traceback (most recent call last):
  File ".../test_langchain_vecemb-RAG-evaluate.py", line 87, in <module>
    vector_db = FAISS.load_local(
  File ".../langchain_community/vectorstores/faiss.py", line 1207, in load_local
    index = faiss.read_index(str(path / f"{index_name}.faiss"), io_flags)
  File ".../faiss/swigfaiss.py", line 14023, in read_index
    return _swigfaiss.read_index(*args)
RuntimeError: Error in faiss::FileIOReader::FileIOReader(const char*) at /project/faiss/impl/io.cpp:69:
Error: 'f' failed: could not open faiss_db/index.faiss for reading: No such file or directory
```

注意报错的最后一行：FAISS 想打开的文件是 **`faiss_db/index.faiss`**——文件名是 `index`。

## 2. 根因分析

### 2.1 证据一：磁盘上实际存的文件叫什么？

```
$ ls faiss_db/
local_cpu_faiss_index.faiss   ← 实际存在的索引文件
local_cpu_faiss_index.pkl     ← 实际存在的元数据文件
```

磁盘上**根本没有** `index.faiss`，只有 `local_cpu_faiss_index.faiss`。FAISS 按默认名去找，自然找不到。

### 2.2 证据二：这个库当初是怎么保存的？

查当初建库/加载的脚本 `test_langchain_vecemb-faissretrive.py` 和 `test_langchain_vecemb-RAG-qa.py`，两处都显式传了自定义索引名：

```python
vector_db = FAISS.load_local(
    folder_path="./faiss_db",
    embeddings=embeddings,
    allow_dangerous_deserialization=True,
    index_name="local_cpu_faiss_index"   # ★ 保存时用的就是这个名字
)
```

即当初 `save_local` 时是按 `index_name="local_cpu_faiss_index"` 保存的。

### 2.3 证据三：langchain_community 源码里 index_name 的默认值

`langchain_community/vectorstores/faiss.py`：

```python
def save_local(self, folder_path: str, index_name: str = "index") -> None:
    ...
    faiss.write_index(self.index, str(path / f"{index_name}.faiss"))

@classmethod
def load_local(
    cls,
    folder_path: str,
    embeddings: Embeddings,
    index_name: str = "index",          # ★ 默认值是 "index"
    *,
    allow_dangerous_deserialization: bool = False,
    ...
) -> FAISS:
    ...
    index = faiss.read_index(str(path / f"{index_name}.faiss"), io_flags)
```

### 2.4 结论

`test_langchain_vecemb-RAG-evaluate.py` 第 87–91 行的 `FAISS.load_local` **漏传了 `index_name` 参数**：

```python
vector_db = FAISS.load_local(
    folder_path=faiss_db_path,
    embeddings=embeddings,
    allow_dangerous_deserialization=True
    # ← 缺 index_name，于是用默认值 "index"，去找 faiss_db/index.faiss
)
```

一句话总结：**保存时用了自定义索引名 `local_cpu_faiss_index`，加载时却没传这个名字，默认去找 `index.faiss`，文件对不上号。**

## 3. 知识点：FAISS 持久化到底存了什么

`save_local(folder_path, index_name)` 会在目录里写**一对文件**，文件名都由 `index_name` 决定：

| 文件 | 内容 | 格式 |
|------|------|------|
| `<index_name>.faiss` | 向量索引本体（所有 embedding 向量 + FAISS 索引结构） | FAISS 原生二进制，由 C++ 的 `faiss::write_index/read_index` 读写 |
| `<index_name>.pkl` | docstore（原始文档文本+metadata）和 `index_to_docstore_id` 映射 | Python pickle（这也是加载时必须传 `allow_dangerous_deserialization=True` 的原因） |

`load_local` 必须用**和保存时相同的 `index_name`**，才能把这一对文件配对找齐。`index_name` 相当于"库名"——同一个 folder 下可以并存多套索引（比如不同 embedding 模型建的多套库），靠名字区分。

## 4. 修复方案（手动修改源码）

修改 `test_langchain_vecemb-RAG-evaluate.py` 第 87–91 行，给 `load_local` 补上 `index_name`：

```python
vector_db = FAISS.load_local(
    folder_path=faiss_db_path,
    embeddings=embeddings,
    index_name="local_cpu_faiss_index",   # ★ 新增：与 save_local 时一致
    allow_dangerous_deserialization=True
)
```

就这一处。改完即可通过文件加载这一步，进入后面的检索-生成与 RAGAS 评估流程。

### 经验教训

1. `FAISS.load_local` 的 `index_name` 默认是 `"index"`，**一旦 save 时用了自定义名字，load 时必须显式传同名**——这是最容易踩的默认值坑。
2. 同一目录下有 `.faiss`/`.pkl` 文件却仍报 "No such file or directory" 时，第一反应应该是**核对文件名前缀**，而不是怀疑路径。

## 5. 附：为什么脚本里的存在性检查没拦住？

脚本第 83–85 行有一层防护：

```python
faiss_db_path = "./faiss_db"
if not os.path.exists(faiss_db_path):
    raise FileNotFoundError(f"FAISS向量数据库路径不存在：{faiss_db_path}")
```

但它只检查了**目录**存在（`./faiss_db` 确实存在），没检查**索引文件**存在，所以防护被绕过。

另一个细节：最终抛出的是 faiss C++ 层的 `RuntimeError`（`faiss::FileIOReader` 打不开文件），而不是 Python 的 `FileNotFoundError`——因为文件打开发生在 C++ 代码里，Python 只是转译了错误消息。这也解释了为什么 traceback 里出现的是 `/project/faiss/impl/io.cpp:69` 这种 C++ 源码位置。

如果想加强防护，可以把检查改成针对具体文件（可选，非本 bug 必需）：

```python
if not os.path.exists(os.path.join(faiss_db_path, "local_cpu_faiss_index.faiss")):
    raise FileNotFoundError("未找到索引文件 local_cpu_faiss_index.faiss，请确认向量库已正确保存")
```
