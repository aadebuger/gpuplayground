import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
import os
mdir=os.getenv("mdir")
tokenizer = AutoTokenizer.from_pretrained(f"{mdir}/OrionStarAI/OrionStar-Yi-34B-Chat", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(f"{mdir}/OrionStarAI/OrionStar-Yi-34B-Chat", device_map="auto",
                                             torch_dtype=torch.bfloat16, trust_remote_code=True)

model.generation_config = GenerationConfig.from_pretrained("OrionStarAI/OrionStar-Yi-34B-Chat")
messages = [{"role": "user", "content": "你好! 你叫什么名字!"}]
response = model.chat(tokenizer, messages, streaming=False)
print(response)
messages = [{"role": "user", "content": "你有哪些功能"}]
response = model.chat(tokenizer, messages, streaming=False)
print(response)
messages = [{"role": "user", "content": "讲个笑话"}]
response = model.chat(tokenizer, messages, streaming=False)
print(response)
messages = [{"role": "user", "content": "鸡和兔在一个笼子里,共有26个头,68只脚,那么鸡有多少只,兔有多少只?"}]
response = model.chat(tokenizer, messages, streaming=False)
print(response)

