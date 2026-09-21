# 导入Hugging Face的LLM接口
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

main_path = "/home/dupengair/shared/LLM/Fine-tuning/"
model_name = "model/Qwen3-0.6B"

# 1. 加载Hugging Face的模型和tokenizer
model = main_path + model_name
tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModelForCausalLM.from_pretrained(model)

# 2. 构建pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1024,
    temperature=0.3
)

# 3. 初始化LangChain的LLM接口（统一接口）
hf_llm = HuggingFacePipeline(pipeline=pipe)

# 4. 调用模型（和之前的LLM调用方式完全一样）
prompt = "请用3句话解释什么是LangChain？"
result = hf_llm.invoke(prompt)

print("Hugging Face模型回复：")
print(result)