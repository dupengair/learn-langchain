# 1. 导入需要的模块
from llm_config import get_chat_model
from langchain_core.prompts import ChatPromptTemplate

# 4. 初始化大模型（和LangChain案例一样）
llm = get_chat_model(temperature=0.3, max_tokens=1024)

# 构建超长指令（模拟复杂任务）
prompt = ChatPromptTemplate.from_messages([
    ("user", """请完成3件事，按顺序来：
1. 写一篇300字左右、关于“LangGraph多智能体”的短文，语言通俗，适合新手；
2. 检查短文是否有错误（比如LangGraph的接口名称、功能描述），修正错误；
3. 润色短文，让语言更流畅，加入1个新手能理解的类比。""")
])

'''
prompt = ChatPromptTemplate.from_messages([
    ("user", """你是一个数据分析专家，请完成以下任务：：
1. 生成一份虚拟销售数据表（100行，字段包括日期、产品、地区、销量、收入）；
2. 清洗数据：处理缺失值和异常值；
3. 计算每个地区的总收入和平均销量；
4. 输出一个可视化分析方案（不需要画图，只给代码）；
5. 用通俗语言写一段商业分析报告；
6. 检查你的分析是否有统计学错误并修正；
7. 最后用类比解释给非技术人员听。
所有步骤请一次性完成。""")
])'''

# 执行单一LLM调用
chain = prompt | llm
result = chain.invoke({})
print("单一LLM输出：")
print(result.content)