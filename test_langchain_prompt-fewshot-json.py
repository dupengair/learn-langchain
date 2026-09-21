# 导入必要的模板类
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.example_selectors import BaseExampleSelector, LengthBasedExampleSelector
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import json
from typing import Dict, List


load_dotenv()
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")
MODEL = os.getenv("MODEL")

if not API_KEY:
    raise ValueError("未检测到 API_KEY，请检查 .env 文件是否配置正确")
if not BASE_URL:
    raise ValueError("未检测到 BASE_URL，请检查 .env 文件是否配置正确")
if not MODEL:
    raise ValueError("未检测到 MODEL，请检查 .env 文件是否配置正确")

chat_model = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=MODEL,
    temperature=0.3,        # 随机性：0-1，越小越严谨，越大越有创造力
    max_tokens=1024,         # 最大生成 tokens 数，避免生成过长内容
    extra_body={            # 关闭 Qwen3 的 thinking 模式：
        "chat_template_kwargs": {"enable_thinking": False}  # extra_body 里的字段会原样附加进 OpenAI 请求体
    }
)

# 2. 工程化示例管理：从JSON文件加载示例（避免硬编码，便于维护）
with open("learning_method_examples.json", "r", encoding="utf-8") as f:
    examples = json.load(f)  # 从JSON中直接提取示例数据列表

'''
# 3. 方案A：ExampleSelector按长度筛选示例（控制提示词总长度）
example_selector = LengthBasedExampleSelector(
     examples=examples,
     example_prompt=PromptTemplate(
         input_variables=["subject", "difficulty", "method"],
         template="学科：{subject}\n难度：{difficulty}\n学习方法：{method}\n"
     ),
     max_length=150,  # 控制示例总长度，避免提示词过长
     get_text_length=lambda x: len(x)  # 长度计算函数
 )
'''

# 4. 方案B（推荐）：自定义ExampleSelector按难度筛选示例
# 当需要根据用户输入的特征（如难度）精准匹配示例时，可自定义ExampleSelector
class DifficultyExampleSelector(BaseExampleSelector):
    """根据用户输入的 difficulty 字段筛选样本"""
    def __init__(self, examples: List[Dict[str, str]]):
        self.examples = examples

    def add_example(self, example: Dict[str, str]) -> None:
        self.examples.append(example)

    def select_examples(self, input_variables: Dict[str, str]) -> List[Dict]:
        # 获取用户输入的难度等级，如果没有提供则默认为 'easy'
        target_difficulty = input_variables.get("difficulty", "easy")
        # 过滤出匹配难度的所有示例
        return [ex for ex in self.examples if ex.get("difficulty") == target_difficulty]

# 本案例使用方案B（按难度筛选），如需使用方案A（按长度筛选），取消方案A的注释并注释掉方案B即可
example_selector = DifficultyExampleSelector(examples=examples)


# 5. 构建工程化少样本模板
few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,  # 替换固定examples为动态选择器
    example_prompt=PromptTemplate(
        input_variables=["subject", "difficulty", "method"],
        template="学科：{subject}\n难度：{difficulty}\n学习方法：{method}\n"
    ),
    example_separator="\n", # 控制examples示例之间的分隔方式
    prefix="少样本提示：",
    suffix="参考以上示例，回答：\n学科：{new_subject}\n难度：{difficulty}\n学习方法：",
    input_variables=["new_subject", "difficulty"]  # 新增难度参数
)

# 6. 动态生成不同难度的提示词
# 场景1：生成入门级LangChain学习方法
formatted_prompt_easy = few_shot_prompt.format(
    new_subject="LangChain",
    difficulty="easy"
)
print("入门级少样本提示词：")
print(formatted_prompt_easy)
result_easy = chat_model.invoke([{"role": "user", "content": formatted_prompt_easy}])
print("\n入门级学习方法：")
print(result_easy.content)

# 场景2：生成进阶级LangChain学习方法
formatted_prompt_hard = few_shot_prompt.format(
    new_subject="LangChain",
    difficulty="hard"
)
print("\n进阶级少样本提示词：")
print(formatted_prompt_hard)
result_hard = chat_model.invoke([{"role": "user", "content": formatted_prompt_hard}])
print("\n进阶级学习方法：")
print(result_hard.content)