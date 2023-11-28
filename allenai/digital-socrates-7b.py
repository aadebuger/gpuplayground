import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
mdir=os.getenv("mdir")
# Load model and tokenizer
model_path = f"{mdir}/allenai/digital-socrates-7b"
model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16).to("cuda:0")
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Define input data
question = "When Dennis operates his lawnmower, he notices the engine makes a lot of noise. He also notices that the engine gets very hot. Which best describes the heat and noise generated from the lawnmower? (A) a change in phase (B) thermal expansion (C) an increase in entropy (D) mechanical advantage"
explanation = "1) The question states that the lawnmower engine makes a lot of noise.\n2) The question states that the lawnmower engine gets very hot.\n3) Noise and heat are both forms of energy.\n4) The noise and heat generated from the lawnmower are a result of the conversion of energy from the fuel to mechanical energy."
answerkey = "C"
predictedanswer = "D"

# construct prompt (Llama conventions)
with open("./allenai/DSCritiqueBank-V1/DSCB-prompts.json") as file:
    prompts = json.load(file)

system_prompt = prompts['digital_socrates_v1']['system']
user_prompt = prompts['digital_socrates_v1']['main'].replace("[[QUESTION]]", question).replace("[[EXPLANATION]]", explanation).replace("[[PREDICTEDANSWER]]", predictedanswer).replace("[[ANSWERKEY]]", answerkey)

full_prompt = f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>{user_prompt} [/INST]\n\n"

# Run model
input_ids = tokenizer.encode(full_prompt, return_tensors="pt").to("cuda:0")
output = model.generate(input_ids, max_new_tokens=512, temperature=0)
res = tokenizer.batch_decode(output, skip_special_tokens=True)
print(res[0])