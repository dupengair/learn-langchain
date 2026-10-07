# 1. 导入需要的模块
from llm_config import get_chat_model
# 导入LangGraph核心：StateGraph构建状态机、END表示流程终止节点
from langgraph.graph import StateGraph, END
# 导入TypedDict，用于定义强类型的全局状态字典（约束字段类型和名称）
from typing import TypedDict
# 导入ChatPromptTemplate，用于构建大模型的提示词模板
from langchain_core.prompts import ChatPromptTemplate
# 导入StrOutputParser，用于将大模型的ChatMessage输出解析为字符串
from langchain_core.output_parsers import StrOutputParser
# 导入类型注解：Annotated用于给字段加描述、Sequence表示序列类型、Literal表示字面量枚举
from typing import Annotated, Sequence, Literal


# 2. 初始化大模型（和LangChain案例一样）
llm = get_chat_model(temperature=0.3, max_tokens=1024)


# ========== 3. 定义全局状态（所有智能体的唯一信息源，核心！） ==========
# 继承TypedDict定义强类型的全局状态，约束所有字段的类型和含义
# 所有智能体仅基于该状态判断是否执行任务，修改也仅更新该状态，确保团队信息同步
class TeamState(TypedDict):
    # 项目核心目标：固定不变，作为所有智能体的行动最终指引
    project_goal: str
    # 待办任务列表：所有智能体共享，智能体自主认领执行，执行后从该列表移除
    todo_tasks: Annotated[Sequence[str], "待办任务列表，智能体自主认领执行，共享可见"]
    # 已完成任务列表：智能体执行任务后，从待办移入该列表，全局可见
    done_tasks: Annotated[Sequence[str], "已完成任务列表，所有智能体可查看，记录执行结果"]
    # 状态更新列表：智能体执行任务后添加该记录，让其他智能体感知全局状态变化（核心通信方式）
    status_updates: Annotated[Sequence[str], "状态更新记录，智能体执行后添加，用于团队信息同步"]
    # 项目完成标志：为True时流程终止，可手动置为True或由路由逻辑判定
    is_finished: Annotated[bool, "项目是否完成的标志，True则LangGraph流程终止"]


# ========== 4. 定义平等智能体（无主管、各有专属技能、自主判断执行） ==========
# 定义3个平等智能体的专属技能，无上下级、无主管，各智能体仅负责自身技能范围内的任务
# 可直接新增键值对扩展智能体（如设计、测试），无需修改核心逻辑
AGENT_SKILLS = {
    "产品智能体": "负责梳理产品需求、设计MVP功能、输出产品文档，确保产品方向匹配项目目标",
    "研发智能体": "负责根据产品文档实现MVP代码、解决技术问题、保证功能可运行，输出可测试的产品",
    "运营智能体": "负责根据MVP设计推广方案、撰写推广文案、初步落地推广，带来种子用户"
}

# 构建智能体决策的提示词模板（核心：让LLM基于全局状态自主决策，强化格式约束避免输出错误）
prompt = ChatPromptTemplate.from_messages([
    ("system", """
    你是创业团队的{agent_name}，核心技能是：{agent_skill}。
    团队无主管，所有人平等，你需基于全局项目状态自主判断是否执行任务，判断规则：
    1. 优先看「状态更新」：有新状态变化且需要你的技能衔接，必须主动干活；
    2. 再看「待办任务」：有待办且属于你的技能范围，主动认领执行；
    3. 最后看「项目目标」：无待办但目标未完成，主动提出待办并执行。

    ⚠️ 强制输出格式（必须严格遵守，缺一不可，用===分隔3个部分，不能合并、不能省略）：
    决策：执行/不执行
    ===
    原因：具体判断依据（基于全局状态的细节，不能简略）
    ===
    执行内容：执行则写具体做的事；不执行则严格写「无」，不能写其他内容

    ⚠️ 格式示例（必须按此结构输出）：
    决策：执行
    ===
    原因：状态更新显示产品完成了需求梳理，待办中有开发MVP代码的任务，属于我的研发技能范围
    ===
    执行内容：根据产品需求文档，实现AI智能体工具MVP的核心Python代码，完成本地功能测试
    """),
    # 用户输入部分：将全局状态的所有字段传入，让LLM基于完整状态决策
    ("user", "全局项目状态：\n项目目标：{project_goal}\n待办任务：{todo_tasks}\n已完成任务：{done_tasks}\n最新状态更新：{status_updates}\n项目是否完成：{is_finished}")
])

def agent_node(agent_name: str, agent_skill: str):
    """
    智能体节点**工厂函数**：根据智能体名称和技能，生成LangGraph要求的节点函数
    LangGraph节点函数规则：输入全局状态TeamState，返回更新后的全局状态TeamState
    :param agent_name: 智能体名称（如产品智能体）
    :param agent_skill: 智能体专属技能描述
    :return: 符合LangGraph要求的节点函数（输入state，输出new_state/state）
    """
    # 定义实际的LangGraph节点函数，嵌套函数可继承外部的agent_name和agent_skill
    def node(state: TeamState) -> TeamState:
        # 1. 构建LLM调用链，完成「提示词渲染→大模型推理→输出解析为字符串」
        chain = prompt | llm | StrOutputParser()
        # 调用链，传入智能体信息+全局状态，获取LLM的决策结果
        response = chain.invoke({
            "agent_name": agent_name,
            "agent_skill": agent_skill,** state  # 解包全局状态所有字段
        })

        # 打印LLM原始返回结果，方便排查格式错误（如未按===分隔、少部分等问题）
        print(f"\n===== {agent_name} 原始返回 =====")
        print(response)
        print(f"=========================\n")

        # 2. 分割LLM返回结果，并做**格式预处理**，解决空格/换行导致的识别问题
        # split("===")按分隔符分割，strip()去除每部分的首尾空格/换行/制表符
        parts = [p.strip() for p in response.split("===")]
        # 格式补全：不足3部分用默认值补（避免解包失败），多于3部分取前3个（忽略多余内容）
        if len(parts) < 3:
            parts += ["决策：不执行", "原因：模型返回格式错误，兜底判定", "执行内容：无"][len(parts):]
        if len(parts) > 3:
            parts = parts[:3]

        # 3. 解包分割结果+**异常兜底处理**，确保代码不会因格式错误崩溃
        try:
            # 解包为决策、原因、执行内容三部分
            decision_part, reason_part, action_part = parts
            # 提取核心内容：移除前缀（如决策：），处理模型可能的多余文字
            decision = decision_part.replace("决策：", "").strip() if "决策：" in decision_part else "不执行"
            reason = reason_part.replace("原因：", "").strip() if "原因：" in reason_part else "格式错误，兜底不执行"
            action = action_part.replace("执行内容：", "").strip() if "执行内容：" in action_part else "无"
            # 强制校验决策值：仅允许「执行/不执行」，其他值兜底为不执行（避免无效决策）
            if decision not in ["执行", "不执行"]:
                decision = "不执行"
                reason = f"决策值异常（{decision}），兜底判定不执行"
        except Exception as e:
            # 捕获所有解包/格式异常（如索引错误、类型错误），全部兜底为「不执行」
            decision = "不执行"
            reason = f"格式解析失败：{str(e)}，兜底判定不执行"
            action = "无"

        # 打印标准化后的决策信息，直观查看智能体最终判断结果
        print(f"===== {agent_name} 标准化决策 =====")
        print(f"是否执行：{decision}")
        print(f"判断原因：{reason}")
        print(f"执行内容：{action}\n")

        # 4. 若决策为「执行」，则更新全局状态；否则返回原状态（无任何修改）
        if decision == "执行" and action != "无" and action != "「无」":
            # 深拷贝原状态：LangGraph要求状态不可变，需生成新对象修改
            new_state = state.copy()
            # ① 执行内容加入「已完成任务列表」
            new_state["done_tasks"] = list(new_state["done_tasks"]) + [action]
            # ② 从「待办任务列表」移除已执行的任务（模糊匹配，避免文字完全一致的要求）
            new_state["todo_tasks"] = [t for t in new_state["todo_tasks"] if not any(k in t for k in action.split("：")[0].split("，"))]
            # ③ 添加状态更新记录：让其他智能体感知「该智能体完成了什么」，实现团队信息同步
            new_state["status_updates"] = list(new_state["status_updates"]) + [f"{agent_name}：{action}"]
            # 返回更新后的新状态，供其他智能体使用
            return new_state
        # 若不执行，直接返回原全局状态，无任何修改
        return state

    # 工厂函数返回定义好的节点函数
    return node

# 利用工厂函数，生成3个平等智能体的节点函数（无主管、无优先级，完全平等）
product_agent = agent_node("产品智能体", AGENT_SKILLS["产品智能体"])
dev_agent = agent_node("研发智能体", AGENT_SKILLS["研发智能体"])
ops_agent = agent_node("运营智能体", AGENT_SKILLS["运营智能体"])


# ========== 5. 构建LangGraph无主管状态机（核心：循环执行、自主响应） ==========
# 初始化状态机，绑定全局状态类型TeamState，确保所有节点遵循状态定义
graph_builder = StateGraph(TeamState)
# 向状态机添加3个智能体节点，节点名称与函数一一对应（平等添加，无顺序优先级）
graph_builder.add_node("product", product_agent)  # 产品智能体节点
graph_builder.add_node("dev", dev_agent)          # 研发智能体节点
graph_builder.add_node("ops", ops_agent)          # 运营智能体节点

def should_continue(state: TeamState) -> Literal["product", "dev", "ops", END]:
    """
    LangGraph条件路由函数：运营智能体执行后，判断流程**继续循环**还是**终止**
    返回值约束为字面量：继续则返回下一个节点（product），终止则返回END
    :param state: 当前的全局状态
    :return: 下一个节点名称 / END（终止）
    """
    # 终止条件：① 手动置为项目完成 ② 状态更新数>3（完成3个核心任务，模拟项目结束）
    if state["is_finished"] or len(state["status_updates"]) > 3:
        return END  # 返回END，流程终止
    return "product"  # 未终止则返回产品智能体，继续循环执行（产品→研发→运营）

# 设置状态机**入口点**：首次执行从产品智能体开始（无主管分配，固定入口）
graph_builder.set_entry_point("product")
# 定义节点间的**顺序边**：产品执行完→研发执行，研发执行完→运营执行
graph_builder.add_edge("product", "dev")
graph_builder.add_edge("dev", "ops")
# 定义**条件边**：运营执行完后，调用should_continue判断是继续循环还是终止
graph_builder.add_conditional_edges("ops", should_continue)

# 编译状态机，生成可运行的LangGraph图对象（编译后不可修改，可多次调用）
graph = graph_builder.compile()


# ========== 6. 测试运行：启动创业团队项目（无主管，智能体自主干活） ==========
if __name__ == "__main__":
    # 初始化项目**初始全局状态**：设定目标、初始待办、初始状态，项目未完成
    initial_state = TeamState(
        # 项目核心目标（固定不变）
        project_goal="开发一个AI智能体工具的MVP并完成初步推广，实现种子用户获取",
        # 初始待办任务（智能体自主认领执行，执行后自动移除）
        todo_tasks=[
            "梳理AI智能体工具MVP的核心需求",
            "实现MVP的核心功能代码",
            "撰写MVP推广文案并在小红书初步发布"
        ],
        done_tasks=[],  # 初始无已完成任务
        # 初始状态更新：标记项目启动，让所有智能体感知项目开始
        status_updates=["项目启动：开始推进AI智能体工具MVP开发与推广"],
        is_finished=False  # 初始项目未完成
    )

    # 打印项目启动信息，直观查看初始目标和待办
    print("===== 创业团队项目启动 =====")
    print(f"项目目标：{initial_state['project_goal']}")
    print(f"初始待办：{initial_state['todo_tasks']}\n")

    # 流式运行状态机：逐节点输出执行过程，直观看到智能体决策和状态更新
    # graph.stream()返回生成器，每次yield一个节点的执行结果（节点名称+更新后的状态）
    for step in graph.stream(initial_state):
        # 遍历每一步的节点和状态（单节点执行，故仅一个键值对）
        for node, state in step.items():
            print(f"===== 节点 {node} 执行后 - 全局状态 =====")
            print(f"✅ 已完成任务：{state['done_tasks']}")
            print(f"📋 剩余待办任务：{state['todo_tasks']}")
            print(f"📌 最新团队状态：{state['status_updates'][-1]}\n")

    # 调用graph.invoke()获取项目最终的全局状态，打印最终执行结果
    final_state = graph.invoke(initial_state)
    print("===== 项目执行完成 - 最终结果 =====")
    print(f"项目核心目标：{final_state['project_goal']}")
    print(f"✅ 团队全部已完成任务：{final_state['done_tasks']}")
    print(f"📌 团队完整状态更新记录：{final_state['status_updates']}")
