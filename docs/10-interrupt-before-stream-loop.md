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

两种方式下，工作流的执行者都是 LangGraph 引擎（Pregel 循环：`tick()` 调度任务 → 节点函数执行 → `after_tick()` 合并状态、存 checkpoint）。`app.stream(...)` 这一行（line 76）就把整个工作流的执行"点燃"了，`for step in stream` 只是在**排队领取**引擎每完成一步吐出的更新。

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

本脚本第一次 stream 运行中，`step` 只有一个：

```python
{"write_email": {"email_content": "尊敬的老师：……"}}
```

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
| 引擎收尾 | GraphInterrupt 被引擎自己捕获（不算错误），状态/存档完好，本轮 run 正常结束，stream 队列关闭 | （无更多元素） |

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

## 小结

| 问题 | 结论 |
|---|---|
| 1. 是手动 for 循环在执行工作流吗？ | 不是。引擎（Pregel 循环）在执行，`invoke` 与 `stream` 只是同一执行的两种消费方式；for 循环只负责逐个接收每步更新，这是为了在步骤之间插入人工交互 |
| 2. if 判断是 write_email 执行完才走到？ | 是。updates 模式下"执行完 → 合并状态 → 存 checkpoint"之后才 yield；step 格式为 `{节点名: 该节点返回的更新}`，`in` 判断的是 key |
| 3. 中断相当于对 for 循环 break？ | 效果相似（send_email 没执行、循环提前结束、接着跑 line 100），机制完全不同：引擎拦截的是**节点调度**，for 循环是因迭代器耗尽而**自然结束**，且进度已存档、可恢复——这是 break 没有的能力 |
