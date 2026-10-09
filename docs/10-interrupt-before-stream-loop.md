# 10. interrupt_before 中断机制三问：stream / for 循环 / 中断语义源码分析

对应脚本：`test_multiagent_interrupt-before.py`（邮件发送工作流，`send_email` 节点执行前中断，人工确认后继续）。

分析基于当前环境安装的 **langgraph 1.2.11** 源码（`site-packages/langgraph/pregel/`），关键结论均给出源码位置。

---

## 问题 1：正常执行用 `compiled_workflow.invoke({})`，这里是 line 90 的手动 for 循环在执行工作流，对吗？

**不完全对。for 循环不是工作流的"驱动者"，只是引擎执行过程的"旁观者"。**

### invoke 和 stream 是同一个引擎的两种消费方式

看 `Pregel.invoke` 的方法体（`langgraph/pregel/main.py:3819` 起）：

```python
# pregel/main.py —— invoke 内部（v1 分支，简化）
for chunk in self.stream(          # ← invoke 内部就是调用自己的 stream
    input, config, stream_mode=(["updates", "values"] if ... else stream_mode), ...
):
    ...
    elif mode == "values":
        latest = payload           # ← 只记住最后一个"全量状态"
...
return latest                      # ← 把它作为 invoke 的返回值
```

也就是说：

| 方式 | 内部执行 | 给调用方的东西 |
|---|---|---|
| `invoke({})` | 引擎跑完整条链 | 只给**最终状态**（中间步骤被丢弃） |
| `stream(...)` | **同一个引擎，同样跑完整条链** | 每执行完一步，把该步的状态更新**逐个 yield 出来** |

两种方式下，工作流的执行者都是 LangGraph 引擎（Pregel 循环：`tick()` 调度任务 → 节点函数执行 → `after_tick()` 合并状态、存 checkpoint）。`app.stream(...)` 这一行（line 76）拿到了运行入口（严格说由于 generator 的惰性，此刻引擎还没启动，真正"点燃"发生在 line 90 第一次迭代，见问题 6），`for step in stream` 则是在**排队领取**引擎每完成一步吐出的更新。

所以 line 90 的 for 循环的正确理解是：**消费流式输出的循环**，而不是"手动驱动工作流"。它和 `test_multiagent_multiLLM.py` 里 `invoke({})` 的区别不是"谁在执行工作流"（都是引擎），而是"中间步骤要不要暴露给你"——本脚本必须在 write_email 执行完、屏幕上展示邮件内容之后才能让用户输入确认，所以必须用 stream 逐步拿，这正是人工确认（human-in-the-loop）场景选 stream 的原因。

> 类比：`invoke` 是"做完一桌菜一次性端上来"；`stream` 是"每做好一道菜就端一道"。厨房（引擎）是同一个，区别只在出菜方式。

---

## 问题 2：line 91 的 `if "write_email" in step`，是 write_email 执行完了才会走到吗？

**是的，并且不止执行完——执行完、结果已合并进全局状态、checkpoint 已保存，这三件事都做完才会 yield 出这个 step。**

### updates 模式下 step 的产生时机

`app.stream(...)` 不指定 `stream_mode` 时默认为 `"updates"` 模式。引擎每一步（super-step）的时序在 `PregelLoop.after_tick()`（`langgraph/pregel/_loop.py`）：

```python
# pregel/_loop.py —— after_tick()（简化）
def after_tick(self) -> None:
    ...
    self.updated_channels = apply_writes(...)   # 1. 把节点返回的 dict 合并进全局状态
    ...
    self._put_checkpoint({"source": "loop"})    # 2. 保存 checkpoint（可恢复的存档点）
    ...
```

随后 `output_writes()` → `_emit("updates", map_output_updates, ...)` 把本步更新放进 stream 队列——**此时** `for` 循环才拿到一个新元素。每个元素（`step`）的格式是：

```python
{ "<节点名>": <该节点返回的部分状态更新 dict> }
```

本脚本第一次 stream 运行中，**带节点名的 `step` 只有一个**：

```python
{"write_email": {"email_content": "尊敬的老师：……"}}
```

> 严格说 stream 里还会多出一个**不含节点名**的中断信号条目 `{"__interrupt__": ()}`（引擎收尾时发出），line 91 的 `if "write_email" in step` 对它是 False，会被静默跳过——详见问题 5 的实验。

### `if "write_email" in step` 的语义

`step` 是 dict，`in` 判断的是 **key**。命中即表示"名为 write_email 的节点这一步执行完了，它返回的更新就在 step 里"。所以：

- 命中时，`write_email_agent` 里那条 `chain.invoke(...)`（真正的 LLM 调用）**早已返回**——生成邮件的耗时发生在这个 if 被走到**之前**，for 循环在这里等待是拿不到"生成中"的过程的；
- `step["write_email"]["email_content"]` 取到的就是节点返回值里的邮件内容；
- 这个 if 并不是为中断而写的——它只是"按节点名分发、分别处理每步更新"的常规写法（对比 `test_multiagent_interrupt-after.py` 里连续判断 `assign_task` / `execute_task` 就是同样模式）。没有 interrupt 配置时它照样工作，只是 step 会更多。

> 一个细节：line 94 打印"⚠️ 系统已在【发送邮件】前中断"这句话时，严格说此刻 send_email 还没轮到被"拦下"——真正的拦截发生在引擎下一步调度时（见问题 3）。但用户可见的时间线上，write_email 的 step 之后 stream 里**不会再有任何更新**，for 循环随即结束，所以教学脚本把提示写在这里没有问题。

---

## 问题 3：中断后，for 循环是继续执行，还是中断去执行 line 100？相当于对 for 循环做了 C 语言 `break` 吗？

**for 循环会"自然结束"（正常退出），然后顺序执行 line 100。效果上像 break，机制上完全不是 break。**

### 中断在引擎内部是怎么发生的

`PregelLoop.tick()` 在**每一步调度任务之前**检查中断配置（`langgraph/pregel/_loop.py:667`）：

```python
# pregel/_loop.py —— tick()（简化）
# before execution, check if we should interrupt
if self.interrupt_before and should_interrupt(
    self.checkpoint, self.interrupt_before, self.tasks.values()
):
    self.status = "interrupt_before"
    raise GraphInterrupt()          # ← 引擎内部抛出，也被引擎自己接住
```

`should_interrupt`（`pregel/_algo.py:155`）的判断条件：**自上次中断以来状态有更新，且本步待执行的任务名在 `interrupt_before` 列表里**。

套到本脚本的时间线：

| 步骤 | 引擎动作 | stream 产出 |
|---|---|---|
| 第 1 步 tick | 待执行任务 = write_email，不在 interrupt 列表 → 正常执行 | — |
| after_tick | 邮件内容合并进状态，存 checkpoint | yield `{"write_email": {...}}` → line 91 命中，打印邮件 |
| 第 2 步 tick | 待执行任务 = send_email（由 `write_email→send_email` 边触发），**命中 interrupt_before** → `raise GraphInterrupt()`，send_email **从未执行** | — |
| 引擎收尾 | GraphInterrupt 被引擎自己捕获（不算错误），存档完好，向 stream 发出中断信号后关闭队列 | yield `{"__interrupt__": ()}`（line 91 不命中，静默跳过）→ 队列关闭 |

### 于是 for 循环的结局是

stream 迭代器里没有更多元素了 → `for step in stream:` **循环体自然跑完、正常退出**（和遍历一个只有 1 个元素的列表结束的方式一模一样）——没有任何东西去"打断"这个 for 循环。控制流回到脚本顶层，接着顺序执行 line 100 的 `input()`。

### 和 C 语言 break 的区别（这是本题的关键）

| | C 的 `break` | `interrupt_before` |
|---|---|---|
| 谁在打断 | **你写的循环控制流**被 break 语句跳出 | 引擎在**调度下一个节点前**检查配置，决定"这一轮到此为止" |
| 打断的对象 | for 循环本身 | **不是循环**——是引擎的节点调度；循环从头到尾一次都没被打断 |
| 结束方式 | 异常退出循环体 | 迭代器耗尽，循环**自然结束** |
| 循环之后 | 直接跳出 | 本来就按顺序往下走 |
| 状态去哪 | break 不保存任何东西 | checkpoint 已保存，**可从断点恢复** |

更准确的类比是：interrupt 相当于引擎执行完 write_email 后说"下班了，进度已存档，剩下的活明天（用户确认后）接着干"，然后正常下班——for 循环只是等到"没有新包裹可领"，自己散了。

对比同目录 `test_multiagent_checkpoint.py` line 138 的 `break`：那才是真的 break——人主动跳出 for 循环（此时 stream 里可能还有数据没消费）。而本脚本没有 break，循环是自己走完的。

### 补充：为什么 line 105 的 `app.invoke(None, config)` 能跳过中断继续执行

`should_interrupt` 有一个"自上次中断以来是否有 channel 更新"的前置条件。恢复时传 `None`（不注入新输入）+ 同一个 `thread_id`（读回同一份 checkpoint）：状态没有产生新版本，`any_updates_since_prev_interrupt` 为 False → send_email 这次**不再被拦截**，从断点直接执行。这就是"中断—确认—恢复"闭环能成立的原因；如果恢复时传了新的输入 dict，状态版本变化，send_email 会被再次拦下。

---

## 问题 4：若 send_email 后面再挂一个节点 wait_reply，interrupt_before 是不是相当于"跳过 send_email、直接去执行 wait_reply"？

**不是。interrupt_before 拦停的是整个引擎，不是把某个节点从任务队列里剔除。到达中断点的那一刻，send_email 之后的所有节点（包括 wait_reply）全部卡在原地，一个都不会执行。**

### 推理误区在哪

问题 3 的推理"for 循环会执行完 → 引擎只是跳过了 send_email → 继续跑后面的节点"把因果搞反了。事实是：

- "for 循环自然结束"和"send_email 被中断"**不是两个独立的事件，而是同一件事的两个表现**：`GraphInterrupt` 终止了整轮 run → stream 队列随之关闭 → for 循环才因迭代器耗尽而结束。
- 引擎不是"跳过一个节点继续跑剩下的"，而是**整台机器停在了中断点**。for 循环收到 write_email 的更新和一条中断信号之后，就再也等不到 send_email / wait_reply 的更新了——这正是"引擎整体停摆"的证据：如果真是跳过 send_email 继续执行，stream 里还会 yield 出 wait_reply 的更新。

### 源码证据：GraphInterrupt 杀死的是整个引擎主循环

引擎主循环在 `Pregel.stream` 内部就是一行 `while`（`langgraph/pregel/main.py:2964`）：

```python
# pregel/main.py:2964 —— 引擎主循环
with SyncPregelLoop(...) as loop:
    ...
    while loop.tick():        # tick() 返回 True → 执行本步任务；返回 False → 正常收工
        ...
```

`tick()` 内部的中断检查（`pregel/_loop.py:667`）：

```python
# pregel/_loop.py —— tick() 内
if self.interrupt_before and should_interrupt(...):
    self.status = "interrupt_before"
    raise GraphInterrupt()    # ← 异常直接冲出 tick()，本步任务还没交给执行器
```

关键在异常的传播路径：`GraphInterrupt` 从 `tick()` 抛出 → `while loop.tick()` 立刻终止（`return True` 都没执行到，本步任务根本不会被执行）→ `with SyncPregelLoop` 块退出 → `__exit__`（`_loop.py:1336`）检测到是 GraphInterrupt → "suppress interrupt"（按正常中断收尾而非报错），保存 checkpoint、关闭 stream。

所以 `raise GraphInterrupt()` 不是"给 send_email 打个跳过标记然后 return True 继续跑"，而是**直接掀翻整个 while 循环**。引擎停了，后面自然没有任何节点会再被调度。

### 为什么 wait_reply 连"被跳过"的资格都没有

LangGraph 的执行模型是**按步触发**的：每一步的任务列表由**上一步的写入**触发（写入 channel → 触发订阅该 channel 的下游节点）。套到假想的图 `START → write_email → send_email → wait_reply → END`：

| 时刻 | 任务列表（tick 待执行任务） | 动作 |
|---|---|---|
| 第 1 步 | `[write_email]` | 执行 → 写入 email_content → yield `{"write_email": ...}` |
| 第 2 步 | `[send_email]`（**wait_reply 不在列表里**——它要等 send_email 写入后才会被触发进任务列表） | 命中 interrupt_before → `raise GraphInterrupt()` → **整轮 run 终止** |
| （确认前） | 引擎停摆，checkpoint 已存档 | wait_reply 从未进入过任务列表 |
| 确认后 `invoke(None)` | `[send_email]`（无新输入 → 不再拦截） | 执行 send_email → yield `{"send_email": ...}` |
| 下一步 | `[wait_reply]`（这次才被 send_email 的写入触发） | 执行 → yield `{"wait_reply": ...}` → END |

wait_reply 的"出生"依赖 send_email 的执行——上游被拦，下游连任务列表都进不去，谈不上被跳过或被执行。

### 反证：如果真是"跳过"，interrupt 就失去存在意义了

假设语义真的是"把被中断的节点摘掉、流程绕过它继续"：`interrupt_before=["send_email"]` 将等价于"免执行 send_email"，邮件没发出去流程却一路冲到 wait_reply 等回复——这在"邮件发送前等人工授权"的业务语义下是荒谬的。interrupt 的设计意图恰恰是**在危险节点门口把整条流程冻住**，等人类指令后决定"继续（从该节点接着跑）"或"终止/改道"。它是一个**暂停阀门**，不是一个**旁路开关**。

---

## 问题 5："for 循环自然结束"和"整轮 run 终止"矛盾吗？中断是不是相当于"把 send_email 从边上传删掉了"？

**不矛盾——它们是同一条因果链的上游和下游；"send_email 被删掉"的理解也不成立——图结构从未改变，send_email 只是被冻结在"待执行"状态。**

### 两层世界观：矛盾是表面上的，实为因果

问题 3 说"for 循环自然结束"，问题 4 说"整轮 run 终止"，两句话分属**两层**，靠因果链连着：

```text
【引擎层】第 2 步 tick 命中 interrupt_before
    → raise GraphInterrupt()
    → 引擎主循环 while loop.tick() 终止        ← 这就是"中断"的全部含义
    → 引擎收尾：存档 + 向 stream 发中断信号 + 关闭队列
─────────────────────────────────────────────
【脚本层】for 循环对引擎内部发生的一切毫不知情
    → 它只是一直向迭代器要下一个数据（next()）
    → 队列已关闭 → 迭代器抛 StopIteration
    → for 循环自然退出                          ← 这就是"自然结束"的全部含义
    → Python 顺序执行 line 100
```

所以正确的读法不是"for 循环自己发现后面没数据了"，而是：**引擎刹车（run 终止）导致 stream 队列关闭，队列关闭导致迭代器耗尽，迭代器耗尽导致 for 循环自然退出**。for 循环的"自然"是 Python 迭代协议层面的自然（StopIteration 触发正常退出，而不是 break 语句跳出），它的**原因**正是引擎层的"不自然"中断。

### 实验验证：不调 LLM 的三节点图（write_email → send_email → wait_reply）

用纯 Python 节点复现"send_email 后面再挂 wait_reply"的场景（`interrupt_before=["send_email"]`），真实输出：

```text
=== 第一次 stream ===
  [执行] write_email
  stream 产出 step 0: {'write_email': {'x': 'from_write_email'}}
  stream 产出 step 1: {'__interrupt__': ()}          ← 引擎终止前发出的中断信号条目

=== 引擎停在哪？（get_state().next）===
  next: ('send_email',)                              ← 冻结点被明确记录，而非节点被删

=== 确认后 stream 恢复 ===
  [执行] send_email                                  ← send_email 还在，照常执行
  stream 产出 step 0: {'send_email': {'x': 'from_send_email'}}
  [执行] wait_reply                                  ← 上游执行完，这才轮到它
  stream 产出 step 1: {'wait_reply': {'x': 'from_wait_reply'}}
```

这份输出同时钉死了三个事实：

1. **wait_reply 在第一次 run 里从未执行**——stream 里没有它的更新。若 interrupt 语义是"跳过 send_email 继续跑"，step 1 就该是 `{'wait_reply': ...}`，实际却是中断信号 `{'__interrupt__': ()}`。
2. **send_email 没有被删除**——`get_state().next` 明确记着 `('send_email',)`（"下一步该跑谁"），确认恢复后它第一个执行。如果它真被从边上删掉了，恢复后就不会再有 `from_send_email`，邮件永远发不出去。
3. **中断时 for 循环实际迭代了 2 次**——第二次拿到的是 `{"__interrupt__": ()}`。本脚本的 `if "write_email" in step` 对它是 False 所以无感，但需要写中断感知代码时，检查这个 key 就是官方入口（`state.next` / `__interrupt__` 是 LangGraph human-in-the-loop 的标准判断方式）。

### "for 循环遍历时发现后面没有节点了"——for 循环根本不遍历节点

这是新理解里需要纠正的关键点：**for 循环对图、节点、边一无所知**。它遍历的对象不是图结构，而是 stream 里的 update 数据流；它唯一会做的事是反复问迭代器"还有下一个吗"。它没有能力也没有必要去检查"图上 send_email 后面还有没有节点"。

所以"发现后面没有节点了，所以正常退出"应改为：**"发现队列里没有下一个数据了，所以正常退出"**——图上后面明明还有 send_email（和 wait_reply），它们处于冻结待执行状态，只是这一轮不再产出数据。

### 一个有用的推论：for 循环无法从"结束方式"上区分"跑完"和"被中断"

| 场景 | for 循环看到的 |
|---|---|
| 无中断，正常跑完 | 收到全部节点更新 → 迭代器耗尽 → **自然退出** |
| interrupt_before 中断 | 收到部分节点更新 + `__interrupt__` 信号 → 迭代器耗尽 → **自然退出** |
| 人为 break（checkpoint 脚本） | 收到部分更新 → **被 break 跳出**，迭代器里可能还有数据没消费 |

前两种在 for 循环眼里的结束方式**一模一样**（都是 StopIteration 正常退出），唯一区别是收到了几个 update、以及有没有 `__interrupt__` 条目。这再次印证：**中断是引擎层的事件，for 循环只是被动承受其结果**——它既不会被打断，也不会去"跳过"任何节点。

### 修正后的类比

- ~~"把 send_email 从流水线上拆走，打包员发现后面没包裹，就下班了"~~
- 正确版本："流水线开到 send_email 门口，哨声一响**全线停工**（引擎 run 终止），传送带停转（队列关闭），打包员（for 循环）发现传送带不会再送来新包裹，收工下班（自然退出，接着做 line 100 的事）。但 send_email 这一站**原地完好**（`next` 记着它），第二天（确认后）从这一站继续开工，干完才轮到 wait_reply。"

---

## 问题 6：for 循环前面没有 invoke，第一次执行（中断前的 LLM 调用）到底是被谁触发的？

**两句话：① 触发执行不需要 invoke——`stream` 和 `invoke` 是平级的两个运行入口；② 但本脚本里真正的"扳机"既不是 line 76 的 `app.stream(...)`，也不是某个 invoke，而是 line 90 的 for 循环第一次迭代。**

### 疑问的根源

按 `test_multiagent_multiLLM.py` 的经验，"`compile()` 之后要跟着 `invoke()` 才会执行"。本脚本 line 76 只是 `stream = app.stream(...)`，line 90 的 for 循环看起来只是"消费数据"——看起来没有任何一句"启动"代码，LLM 却实实在在执行了，它是怎么被触发？

### 先明确：stream 自己就是运行入口，不需要 invoke

LangGraph 的编译图实现了 Runnable 协议，`invoke` / `stream` / `batch` 是**平级的运行方法**，任何一个都会启动引擎。前面问题 1 已经从源码确认：`invoke` 内部就是调 `self.stream(...)` 并消费它——invoke 不是"比 stream 更正统的入口"，反而是 stream 在底下、invoke 在上面包了一层。所以"graph.compile 后面必须跟 invoke"这个心智模型可以放下了：跟 `stream(...)` 同样能执行，只是结果的交付方式不同。

### 但本脚本有个更反直觉的细节：line 76 那一刻，什么都没发生

`app.stream(...)` 返回的不是执行结果，而是一个 **Python generator**（生成器）。用纯图实验观察执行时机（节点带打印）：

```text
A: 即将调用 app.stream(...)
B: app.stream(...) 已返回，返回对象类型 = generator
   （注意：上面没有出现任何节点执行打印）
C: 即将进入 for 循环（第一次迭代 = 第一次 next()）
      >>> [执行] write_email 节点（LLM 调用发生在这里）   ← 现在才启动！
D: for 循环拿到 step 0: {'write_email': {...}}
D: for 循环拿到 step 1: {'__interrupt__': ()}
E: for 循环结束
```

B 行打印时没有任何 `[执行]` 输出——**调用 `app.stream(...)` 本身没有启动引擎**。引擎是在 C 之后、第一次迭代时才启动的。

### 为什么会这样：Python generator 的惰性求值

`Pregel.stream` 的函数体内含有 `yield`，这使它成为一个**生成器函数**。Python 对生成器函数的规则是：

1. **调用它不执行任何函数体代码**，只返回一个 generator 对象（line 76 发生的事）；
2. **第一次 `next()` 才开始执行函数体**，跑到第一个 `yield` 处暂停，把值交给调用方（line 90 第一次迭代发生的事）；
3. 之后的每次 `next()` 从暂停处恢复执行，直到函数体结束 → `StopIteration`。

所以本脚本的完整触发链是：

```text
line 76  stream = app.stream(...)   → 只是创建 generator，引擎未启动，LLM 未调用
line 90  for step in stream:        → 第一次 next()
              函数体开始执行 → 启动引擎（SyncPregelLoop 在后台线程 tick）
              → write_email 执行，LLM 被调用  ← 第一次执行的真正位置
              → 更新被推进队列 → yield 出来 → step 0 进入循环体
line 90  第二次迭代                  → 引擎第 2 步 tick 命中 interrupt_before
              → GraphInterrupt → run 终止 → 发出 __interrupt__ 条目 → 队列关闭
         第三次 next()               → StopIteration → for 自然退出
```

推论很硬：**如果删掉 line 90 的 for 循环，这个脚本第一次运行什么都不会发生**——没有 LLM 调用、没有 checkpoint、没有中断，line 100 的 `input()` 会直接出现。for 循环不只是流式输出的消费者，它同时是第一次执行的**启动器**。

### 和 invoke 的触发时机对照

| 写法 | 调用那一刻 | 何时真正执行 | 执行完拿到什么 |
|---|---|---|---|
| `result = app.invoke(input, cfg)` | **立即启动引擎**（invoke 是普通函数，内部立即把 stream 的 generator 喂进 for 循环完整消费） | 调用语句执行期间 | 最终状态 dict |
| `stream = app.stream(input, cfg)` | 只创建 generator，**不执行** | **第一次迭代时**（`for step in stream` 或 `next(stream)`） | 逐步的 update，由迭代节奏控制 |
| `stream = app.stream(...)` 后**从不迭代** | — | **永不执行**（无 LLM 调用、无副作用） | — |

一句话总结：`invoke` 是"调用即执行"，`stream` 是"迭代才执行"。本脚本把 line 76（拿 generator）和 line 90（开始迭代）拆成两行，让"何时启动引擎"的决定权落在了 for 循环上——这也是"for 循环消费数据"的表象下，它实际承担的第一个职责。

---

## 小结

| 问题 | 结论 |
|---|---|
| 1. 是手动 for 循环在执行工作流吗？ | 不是。引擎（Pregel 循环）在执行，`invoke` 与 `stream` 只是同一执行的两种消费方式；for 循环只负责逐个接收每步更新，这是为了在步骤之间插入人工交互 |
| 2. if 判断是 write_email 执行完才走到？ | 是。updates 模式下"执行完 → 合并状态 → 存 checkpoint"之后才 yield；step 格式为 `{节点名: 该节点返回的更新}`，`in` 判断的是 key |
| 3. 中断相当于对 for 循环 break？ | 效果相似（send_email 没执行、循环提前结束、接着跑 line 100），机制完全不同：引擎拦截的是**节点调度**，for 循环是因迭代器耗尽而**自然结束**，且进度已存档、可恢复——这是 break 没有的能力 |
| 4. send_email 后面再挂 wait_reply，中断相当于"跳过 send_email 执行 wait_reply"？ | 不是。`GraphInterrupt` 从 `tick()` 内部抛出，杀死的是引擎整个 `while loop.tick()` 主循环——是**整条流程停在中断点**，不是摘掉一个节点继续跑。wait_reply 依赖 send_email 的写入触发，上游被拦时它连任务列表都进不去；确认恢复后从 send_email 接着跑，执行完才轮到 wait_reply |
| 5. "for 循环自然结束"与"整轮 run 终止"矛盾吗？是节点被删了吗？ | 不矛盾：是同一条因果链的上下游——引擎刹车 → stream 队列关闭 → 迭代器耗尽 → for 循环才自然退出。send_email 也没被删除（`get_state().next` 记着它，恢复后照常执行），只是被冻结；for 循环遍历的是 update 数据流而非图结构，它"发现没有下一个数据了"，而不是"发现后面没有节点了" |
| 6. for 循环前面没有 invoke，第一次 LLM 执行是被谁触发的？ | `stream` 和 `invoke` 是平级的运行入口，不需要 invoke 触发；但 `app.stream(...)` 因 generator 惰性只创建对象不启动引擎，**真正的扳机是 line 90 的 for 循环第一次迭代**（第一次 `next()` 才启动引擎、调用 LLM）。删掉 for 循环则第一次运行什么都不会发生——"invoke 调用即执行，stream 迭代才执行" |
