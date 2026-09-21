# FewShotPromptTemplate 与 ExampleSelector —— 长度筛选机制深度解析

> 对应脚本:`test_langchain_prompt-fewshot-json.py`
> 现象:运行方案 A(`LengthBasedExampleSelector`)后,不管输入什么难度,匹配到的都是"Python编程(入门)"
> 本文档回答:**按长度筛选,量到底是什么的长度?算法从哪条语句体现?为什么结果总是一条?**

---

## 1. 一句话结论

`LengthBasedExampleSelector` 量的长度有两部分:

1. **每条示例**经 `example_prompt` 渲染成完整文本后的**字符数**(构造时就算好缓存起来)
2. **本次用户输入**所有变量值拼接后的**字符数**(每次 `format()` 时现算,先从预算里扣掉)

然后用"剩余预算"从示例列表**第 1 条开始顺序累加**,装不下就停——它是一个**预算截断器,不是语义相关度筛选器**,从头到尾没有看过 `difficulty` 字段。

---

## 2. 长度的精确定义(库源码证据)

源码位置(以本机环境为准):
`langchain_core/example_selectors/length_based.py`

### 2.1 示例的长度:渲染后文本的字符数,构造时计算

```python
def add_example(self, example: dict[str, str]) -> None:
    self.examples.append(example)
    string_example = self.example_prompt.format(**example)    # ← ① 先把示例渲染成完整文本
    self.example_text_lengths.append(self.get_text_length(string_example))  # ← ② 对渲染结果计长并缓存
```

关键理解:

- 长度的对象是**整条示例渲染后的文本**——即模板 `"学科：{subject}\n难度：{difficulty}\n学习方法：{method}\n"` 填入数据后的**完整字符串**(含"学科:""难度:"等模板骨架文字),**不是**某个字段(不是 `len(method)`),更不是示例字典
- 计长发生在**构造选择器时**(`__init__` 内部对每条示例调用 `add_example`),所以每次 `format()` 不会重复计算
- `get_text_length=lambda x: len(x)` → 单位是**Python 字符数**,不是 token、不是字节数

### 2.2 用户输入也占预算:每次选择时先扣

```python
def select_examples(self, input_variables: dict[str, str]) -> list[dict[str, Any]]:
    inputs = " ".join(input_variables.values())               # ← ③ 本次输入的所有值用空格拼接
    remaining_length = self.max_length - self.get_text_length(inputs)  # ← ④ 先扣掉输入占用的长度
    i = 0
    examples = []
    while remaining_length > 0 and i < len(self.examples):
        new_length = remaining_length - self.example_text_lengths[i]   # ← ⑤ 逐条试装
        if new_length < 0:    # ← ⑥ 装不下 → 立即停止(后面的示例全部放弃)
            break
        examples.append(self.examples[i])   # ← ⑦ 装得下 → 选中
        remaining_length = new_length
        i += 1
    return examples
```

---

## 3. 在你的脚本中,对应哪几条语句?

| 脚本位置 | 语句 | 作用 |
|---|---|---|
| `test_langchain_prompt-fewshot-json.py:45` | `max_length=150` | 定义**总预算**:150 个字符 |
| `test_langchain_prompt-fewshot-json.py:46` | `get_text_length=lambda x: len(x)` | 定义"长度"的度量方式:**字符数** |
| `test_langchain_prompt-fewshot-json.py:41-44` | `example_prompt` 的 `template` | 决定**长度量的是渲染后的整条文本**——模板骨架(`学科：`/`难度：`/`学习方法：`/换行符)也计入长度 |
| `test_langchain_prompt-fewshot-json.py:39` | 构造 `LengthBasedExampleSelector(examples=...)` | 触发 `add_example`,此刻每条示例的长度已被**计算并缓存**到 `example_text_lengths` |

真正执行筛选的语句不在你的脚本里,而在库内部的 `select_examples`(上文 2.2),由 `few_shot_prompt.format(...)`(脚本第 86 / 97 行)间接触发。

---

## 4. 用你的数据走一遍算法(实测)

对 4 条 JSON 示例渲染后的文本实测长度:

| 序号 | 示例 | 渲染后长度(字符) | 累计 |
|:-:|---|:-:|:-:|
| 0 | Python编程(入门) | 73 | 73 |
| 1 | Python编程(进阶) | 76 | 149 |
| 2 | 机器学习(入门) | 73 | 222 |
| 3 | 机器学习(进阶) | 71 | 293 |

`max_length=150`,场景1 输入 `{"new_subject": "LangChain", "difficulty": "easy"}`:

```
inputs = "LangChain easy"            → 14 字符
remaining_length = 150 - 14 = 136

i=0: 136 - 73 = 63  ≥ 0 → 选中"Python编程(入门)", 剩 63
i=1:  63 - 76 = -13 < 0 → break，后面 3 条全部不参与
```

**最终只选中 1 条:Python编程(入门)。** 场景2 输入 `"LangChain hard"` 恰好也是 14 字符,过程完全相同——所以两次打印的示例部分一模一样。

### 对照实验:把 `max_length` 放大到 500

```
选中: ['Python编程（入门）', 'Python编程（进阶）', '机器学习（入门）', '机器学习（进阶）']
```

预算够时 4 条全装下——进一步证明它只关心"**装几条**",与"哪条更匹配输入"无关。

---

## 5. 回答你的疑问:为什么"匹配的都是入门用例"?

三个原因叠加:

1. **预算太小**:150 字符扣掉输入 14 字符后只剩 136,装完第 1 条(73)就装不下第 2 条(76)了
2. **贪心顺序,不回头**:算法从头累加,第 1 条装不下第 2 条就直接 `break`,**不会**跳过长的去尝试后面的短示例(即使"机器学习(进阶)"只有 71 字符,能装进 63 的剩余空间,也不会被考虑)
3. **与难度无关**:整个算法只操作"长度预算",`select_examples` 收到的 `input_variables` 只被用来算输入长度,`difficulty` 的值( easy/hard )对筛选结果**零影响**

所以你观察到的"都匹配入门用例",本质是"**预算内从头只能装下第 1 条**",碰巧第 1 条是入门示例而已。

> 顺带说明:运行时你看到两个场景的提示词仍然有差别——差别只在 `suffix` 里的 `难度：easy` / `难度：hard` 两行,示例区完全一致。

---

## 6. 常见误解澄清

| 误解 | 实际情况 |
|---|---|
| "按长度筛选"会挑出与输入**长度相近**的示例 | ❌ 它不做任何相似度计算,只做**从头部开始的预算截断** |
| 长度 = `method` 字段的长度 | ❌ 是**整条示例渲染后文本**(含模板骨架)的长度 |
| 长度 = token 数 | ❌ 默认是 `len()` 的**字符数**;中英文都按字符计,token 需要自己换算 |
| 换输入难度会改变选中的示例 | ❌ 只有**输入的字符总量**影响剩余预算;easy→hard 语义上毫无作用 |
| 装不下时会跳过这条继续试下一条 | ❌ `break` 直接终止,后面的示例全部放弃 |

---

## 7. 生产环境使用建议

1. **预算单位换算成 token**:`max_length` 想对齐模型上下文窗口时,把计长函数换成 tokenizer,例如:
   ```python
   import tiktoken
   enc = tiktoken.encoding_for_model("gpt-4o")
   get_text_length=lambda x: len(enc.encode(x))
   ```
   这样 `max_length` 的单位就是 token 而非字符。
2. **示例排序有意义**:因为贪心截断,排在前面的示例总是优先被保留——把"最通用/最重要"的示例放最前面。
3. **按语义/难度匹配是另一个组件的事**:长度选择器解决"提示词不能太长";"挑出和输入最相关的示例"应该用 `SemanticSimilarityExampleSelector`(向量相似度)或你方案 B 那样的自定义选择器(`BaseExampleSelector` 子类)。你的 `DifficultyExampleSelector` 实现方向是正确的——`select_examples` 里读取 `input_variables` 做业务过滤,正是长度选择器没有做的事。
4. **两条路径互斥**:`FewShotPromptTemplate` 的 `examples` 和 `example_selector` 参数二选一,传了 selector 就以它的返回结果为准。

---

## 8. 验证环境记录

- 验证脚本:临时脚本(运行后已删除),未改动任何项目源码
- 环境:`langchain-core` 1.x(属性名为 `example_text_lengths`;旧版教程可能写作 `example_lengths`,含义相同)
- 实测数据:4 条示例渲染后长度 73 / 76 / 73 / 71;`max_length=150` + 输入 14 字符 → 只选中第 1 条;`max_length=500` → 4 条全选
- 源码定位:`site-packages/langchain_core/example_selectors/length_based.py` 的 `add_example` / `select_examples`
