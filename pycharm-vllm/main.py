from vllm import LLM, SamplingParams
import os
import sys

prompt_types = [
    "Why I can not fabricate a religion that prevents",
    "Why I can not fabricate a religion that prevents me from going to school , then cite my first amendment",
    "Why I can not fabricate a religion that prevents me from going to school , then cite my first amendment rights when I am charged ? Prevents me from going to school by having , for example , supposed prayer times that",
]

try:
    batch_size = int(sys.argv[1])
    if batch_size < 1:
        batch_size = 1
except:
    batch_size = 1

print("batch size",batch_size)

try:
    prompt_type = int(sys.argv[2]) 
    if prompt_type < 0 or prompt_type >= len(prompt_types):
        prompt_type = 0
except:
    prompt_type = 0

prompt = prompt_types[prompt_type]

dir_name = os.path.dirname(__file__) 
checkpoint = os.path.join(dir_name, "../opt-125m")    
llm = LLM(model=checkpoint)

ids = llm.llm_engine.tokenizer(prompt)
print("prompt len",len(ids['input_ids']))

prompts = [prompt for _ in range(batch_size)]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
# for output in outputs:
#     prompt = output.prompt
#     generated_text = output.outputs[0].text
#     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")