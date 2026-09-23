# bert-base-uncased 嵌入报 UNEXPECTED 分析 —— 假警报背后藏着两个真问题

> 对应脚本:`test_langchain_vecemb.py`
> 现象:加载 bert-base-uncased 时 transformers 报告 7 个 `cls.predictions.* / cls.seq_relationship.*` 权重为 UNEXPECTED;检索"LangChain的链式工作流有哪些类型?"返回片段"大模型开发",评分 0.31
> 结论先行:**UNEXPECTED 本身是假警报(完全正常,可忽略);但 bert-base-uncased 做中文向量嵌入是真问题(60% token 变 [UNK],语义区分度失效);另外当前知识库只有 1 条文档,检索质量根本没被验证。该换模型,但首选轻量中文嵌入模型 bge-small-zh-v1.5,qwen3-embedding 是进阶选项而非必需。**
> 本文所有数据均来自本机实测(与脚本同配置:HuggingFaceEmbeddings + CPU + normalize_embeddings=True)。

## 1. 结论速览

| 问题 | 判定 | 依据 |
|------|------|------|
| UNEXPECTED 信息是错误吗? | ❌ 不是,**假警报** | missing=0(主干 199/199 完整),7 个 UNEXPECTED 全是 BertModel 用不上的预训练头 |
| bert-base-uncased 能做中文嵌入吗? | ❌ **不能,真问题** | "大模型开发"5 个 token 里 4 个 `[UNK]`;中文句对相似度 0.84-0.92 挤成一片,无关句反而比相关句"更相似" |
| 检索结果(score 0.31)可信吗? | ❌ 无法评判 | 知识库只有 1 条文档,向量库只有 1 个向量,top3 必然返回它 |
| 要换 qwen3-embedding 吗? | ⚠️ 该换模型,但不必是它 | 教学场景首选 **BAAI/bge-small-zh-v1.5**(约 100MB,CPU 秒嵌);qwen3-embedding 留给追求检索质量的进阶场景 |

## 2. UNEXPECTED 在说什么(假警报部分)

bert-base-uncased 官方 checkpoint 的原生身份是 **BertForPreTraining**,它 = BertModel 主干 + 两个预训练头:

```
BertForPreTraining checkpoint(检查点里的全部权重)
├── BertModel 主干(embeddings + 12 层 encoder + pooler)   ← BertModel 架构要加载的
├── cls.predictions.*      (MLM 遮盖语言模型头)             ← BertModel 没这个结构 → UNEXPECTED
└── cls.seq_relationship.* (NSP 下一句预测头)               ← BertModel 没这个结构 → UNEXPECTED
```

`BertModel.from_pretrained()` 只构建主干,checkpoint 里多出来的 7 个头权重"没地方放",于是被列入 UNEXPECTED 清单。实测确认:

```
BertModel 加载:missing_keys = 0 个      ← 该加载的权重全部就位,主干完整
BertModel 加载:unexpected_keys = 7 个   ← 全部是上述预训练头
```

**判读口诀:看 MISSING,别怕 UNEXPECTED。**

- `MISSING` = 模型结构需要的权重没加载到 → 真问题(比如随机初始化,效果会差)
- `UNEXPECTED` = checkpoint 里多余的权重 → 换架构加载的正常现象,transformers 文档原话就是 *"can be ignored when loading from different task/architecture"*

一个佐证:用 `AutoModelForMaskedLM` 加载**同一个** checkpoint,UNEXPECTED 清单会变成另外两个键(`bert.pooler.dense.*`,因为 MLM 任务不用 pooler)。同一个文件、不同架构、不同清单——这正说明 UNEXPECTED 只是"架构不匹配告知",不是损坏。

所以:这 7 行信息**不需要也无法修复**,放心忽略。

## 3. 真问题一:英文模型处理中文(致命,必须换模型)

### 3.1 词表证据:中文被切成了 [UNK]

bert-base-uncased 的词表(30522 个 WordPiece)是英文语料训出来的。实测 token 化:

| 输入 | token 结果 | [UNK] 占比 |
|------|-----------|-----------|
| `大模型开发` | `['大', '[UNK]', '[UNK]', '[UNK]', '[UNK]']` | **4/5** |
| `LangChain的链式工作流有哪些类型?` | 15 个 token,其中 9 个 `[UNK]` | **9/15** |
| `I love machine learning`(英文对照) | `['i', 'love', 'machine', 'learning']` | 0 |

`[UNK]` 是所有未登录字符共享的**同一个嵌入向量**——你的中文文本有六成字符在模型眼里是同一个"乱码",嵌入质量无从谈起。

### 3.2 区分度证据:中文语义空间"挤成一团"

用与脚本完全相同的配置算余弦相似度:

| 句对 | bert-base-uncased | 语义关系 |
|------|------------------|---------|
| 查询 vs "大模型开发" | 0.8433 | 相关(勉强) |
| 查询 vs "今天天气很好,适合出门散步" | **0.8794** | **完全无关** |
| "今天天气很好" vs "明天会下雨吗" | 0.8625 | 无关 |
| "大模型开发" vs "我喜欢吃苹果" | **0.9119** | 无关 |
| (英文) ML定义 vs "ML is a branch of AI" | 0.7547 | 相关 |
| (英文) ML定义 vs "I had pizza for lunch" | 0.5178 | 无关 |

两个致命信号:

1. **所有中文句对都挤在 0.84-0.92 的窄带里**——这是原生 BERT 嵌入空间的"各向异性"(所有句子向量挤在一个窄锥里),相似度分数失去区分能力;
2. **排序反直觉:完全无关句(0.879)的分数反而高于知识库片段(0.843)**,英文句对则正常(0.755 > 0.518)——模型本体没坏,**只是它的语义空间只为英文而建**。

### 3.3 你看到的 score 0.31 是什么

`similarity_search_with_score` 返回的是 FAISS 的**平方欧氏距离 L2²**(不是余弦)。向量已归一化时有个好用的换算:

```
cos = 1 − score/2
0.3135 → cos ≈ 0.843   (与上面实测完全吻合)
```

也就是说:你的查询和"大模型开发"这两个语义毫无关系的文本,相似度被评成了 0.84——**评分数字看起来"正常",实际上毫无意义**。

## 4. 真问题二:实验设计验证不了检索(知识库只有 1 条)

`knowledge_base/test.txt` 全文只有 6 个字("大模型开发"),分割后 1 个片段,**向量库里只有 1 个向量**:

- "最相关的 3 个片段"永远返回它——top-k 检索退化成"唯一答案";
- 无论嵌入模型好坏,score 高低都只反映"查询 vs 这一条"的距离,**没有任何排序可比**。

**修复建议:先扩充知识库**(4 条以上、主题各异,如 LangChain/FAISS/天气/菜谱),检索质量才有评判标准——这也是下面对照实验的做法。

## 5. 对照实验:换成中文嵌入模型后

`BAAI/bge-small-zh-v1.5`(约 100MB,对比学习训练的中文检索模型),同样的句对:

| 句对 | bert-base-uncased | bge-small-zh-v1.5 |
|------|------------------|-------------------|
| 查询 vs 知识库片段(相关) | 0.8433 | **0.3991** |
| 查询 vs 无关句 | 0.8794(反超相关句 ❌) | **0.2026**(正确低于相关句 ✅) |

扩充到 4 条不同主题文档后,检索"LangChain的链式工作流有哪些类型?":

```
排名1  L2²=0.2610  LangChain 的链式工作流主要分为两种:LCEL 声明式链...   ✅ 正确命中
排名2  L2²=1.1825  FAISS 是 Facebook 开发的高效向量相似度检索库...
排名3  L2²=1.5787  今天北京天气晴朗...
排名4  L2²=1.6015  红烧肉的做法...
```

第 1 名与第 2 名的分数差近 5 倍,**排序正确、区分度巨大**——这才是嵌入模型该有的样子。

## 6. 换哪个模型?——选型建议

| 模型 | 大小 | 中文 | 说明 |
|------|------|------|------|
| bert-base-uncased | 440MB | ❌ | 英文词表 + 原生 BERT 无对比学习,双杀 |
| bert-base-chinese | 400MB | ⚠️ | 词表对了,但仍是 MLM 预训练的原生 BERT,检索区分度一般 |
| **BAAI/bge-small-zh-v1.5** | **~100MB** | ✅ | **教学首选**:中文检索专用,CPU 秒嵌,区分度实测优秀 |
| BAAI/bge-base-zh-v1.5 | ~400MB | ✅ | 同系列更大版,质量更高 |
| shibing624/text2vec-base-chinese / moka-ai/m3e-base | ~400MB | ✅ | 同代可选 |
| **Qwen3-Embedding-0.6B** | ~1.2GB | ✅ | 多语言 SOTA、32K 上下文、指令感知;CPU 嵌入偏慢,**进阶/生产再上** |

**结论:不必一步到 qwen3-embedding。教学走通 RAG 流程,`bge-small-zh-v1.5` 性价比最高;等知识库规模和检索质量要求上来,再换 Qwen3-Embedding(0.6B 起步)。**

用 qwen3-embedding 的两个注意点(届时):
1. 官方 repo 自带 sentence-transformers 配置(last-token pooling),`HuggingFaceEmbeddings` 直接加载即可;查询侧建议加 instruction 以获得最佳效果;
2. 更优雅的用法是 `vllm serve` 加 `--task embed` 起一个嵌入服务,LangChain 换用 `OpenAIEmbeddings` 直连——与你仓库现有的"本地 vLLM + OpenAI 兼容端点"模式完全统一。

## 7. 修复方案(手动修改,对照学习)

### 改动 1:换嵌入模型(第 27-40 行区域,核心就一行)

```python
# 3. 初始化中文嵌入模型(bge-small-zh-v1.5,首次运行会从 HF 下载约 100MB;
#    若网络不通,可设置环境变量 HF_ENDPOINT=https://hf-mirror.com 走镜像,或手动下载后填本地路径)
embedding_model_name = "BAAI/bge-small-zh-v1.5"   # ← 原来是本地 bert-base-uncased 路径

embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs={
        "device": "cpu"
    },
    encode_kwargs={
        "normalize_embeddings": True   # 归一化后 L2² 与余弦单调等价:cos = 1 − score/2
    },
    query_instruction="为这个句子生成表示以用于检索相关文章："  # bge 官方查询指令,只加在查询侧,进一步提升检索效果
)
```

### 改动 2:扩充知识库(knowledge_base/test.txt)

替换成多主题文档(至少 4 条),例如 LangChain、FAISS、天气、菜谱各一段,让"检索排序"有对错可判。

### 不用改的

- `normalize_embeddings=True` 保留(写得对);
- UNEXPECTED 无需处理(第 2 节);
- 分割参数、FAISS 存取、检索代码全部不动。

## 8. 验证清单

换完模型、扩完知识库后重跑脚本,检查:

1. 加载日志不再出现 `cls.predictions.*` UNEXPECTED(bge checkpoint 没有 MLM/NSP 头);
2. 检索"LangChain的链式工作流有哪些类型?"第 1 名应为 LangChain 相关片段,且 score(即 L2²)显著小于第 2 名(实测约 0.26 vs 1.18);
3. 手动换算 `cos = 1 − score/2`,相关片段 cos 应明显高于所有无关片段。

## 9. 一句话总结

**UNEXPECTED 是"checkpoint 比模型架构多带了几个预训练头"的例行告知,看 MISSING 就行;真正的坑是拿英文 BERT 嵌中文——六成字符变 [UNK]、无关句比相关句还"像",再叠加知识库只有 1 条导致检索永远"对"。换 bge-small-zh-v1.5 + 扩充知识库,qwen3-embedding 留给进阶。**
