import requests
import json
import os
from transformers import AutoTokenizer
import transformers
import torch
import re
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer


def GPT4(prompt,key):
    #url = "https://api.openai.com/v1/chat/completions"
    #url = "https://xiaoai.plus/v1/chat/completions"
    #api_key = key
    #with open('template/template.txt', 'r') as f:
    #    template=f.readlines()
    #user_textprompt=f"Caption:{prompt} \n Let's think step by step:"
    
    #textprompt= f"{' '.join(template)} \n {user_textprompt}"
    
    #payload = json.dumps({
    #"model": "gpt-4o", # we suggest to use the latest version of GPT, you can also use gpt-4-vision-preivew, see https://platform.openai.com/docs/models/ for details. 
    #"model": "gpt-4-1106-preview",
    #"messages": [
    #    {
    #        "role": "user",
    #        "content": textprompt
    #    }
    #]
    #})
    #headers = {
    #'Accept': 'application/json',
    #'Authorization': f'Bearer {api_key}',
    #'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
    #'Content-Type': 'application/json'
    #}
    print('yuanming test 1119 waiting for GPT-4 response')
    #print("111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111")
    #print(type(prompt))
    #print(prompt)
    #print("222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222")
    #response = requests.request("POST", url, headers=headers, data=payload)
    #obj=response.json()
    #text=obj['choices'][0]['message']['content']
    # 假设 output.txt 文件已经位于当前工作目录中

    # 打开文件并读取内容
    #with open(f'lunwenzuoshuju/spatial{prompt:03d}.txt', 'r', encoding='utf-8') as file:
    with open(f'lunwenzuoshuju/spatialtst.txt', 'r', encoding='utf-8') as file:
    # 读取文件的所有内容，并将其赋值给字符串变量 text
        text = file.read()
    # 现在，text 是一个 <class 'str'> 的实例，并且包含了 output.txt 文件的内容
    #print("---------------------------------------------------------------------------------------------------")
    #print(text)  # 打印以验证内容是否正确读取
    #print("---------------------------------------------------------------------------------------------------")
    #print(type(text))  # 打印以验证 text 的类型是否为 <class 'str'>
    #print("---------------------------------------------------------------------------------------------------")
    # Extract the split ratio and regional prompt

    return get_params_dict(text)

#def local_llm(prompt,version,model_path=None):
#def local_llm(prompt,model_path=None):
def local_llm(prompt,mod=None,tok=None,mod_input=None,txtpro=None):
    #if model_path==None:
    #    model_id = "Llama-2-13b-chat-hf" 
    #else:
    #    model_id=model_path
    #print('Using model:',model_id)
    #tokenizer = LlamaTokenizer.from_pretrained(model_id)
    #print("tokenizer ok!!")
    #model = LlamaForCausalLM.from_pretrained(model_id, load_in_8bit=False, device_map='auto', torch_dtype=torch.float16)
    #print("model ok!!")
    #with open('template/template.txt', 'r') as f:
    #    template=f.readlines()
    #user_textprompt=f"Caption:{prompt} \n Let's think step by step:"
    #textprompt= f"{' '.join(template)} \n {user_textprompt}"
    #print("-------------------------------------------------------------------------------------------")
    #print(textprompt)
    #print("-------------------------------------------------------------------------------------------")
    #model_input = tokenizer(textprompt, return_tensors="pt").to("cuda")
    mod.eval()
    with torch.no_grad():
        print('waiting for LLM response')
        res = mod.generate(**mod_input, max_new_tokens=1024)[0]
        output=tok.decode(res, skip_special_tokens=True)
        print("output1-------------------------------------------------------")
        print(output)
        print("end-----------------------------------------------------------")
        output = output.replace(txtpro,'')
        print("output2-------------------------------------------------------")
        print(output)
        print("end-----------------------------------------------------------")
        # 假设你想要将张量移回CPU（尽管这通常不是必要的）
        #model_input_cpu = {key: value.to("cpu") for key, value in model_input.items()}
        # 然后你可以删除GPU上的张量引用
        #del model_input
        # 注意：model_input_cpu现在包含了CPU上的张量，如果你不再需要它们，也应该删除这个引用
        #del model_input_cpu
        #del model
        #torch.cuda.empty_cache()
        torch.cuda.empty_cache()
    return get_params_dict(output)

def get_params_dict(output_text):
    response = output_text
    # Find Final split ratio
    split_ratio_match = re.search(r"Final split ratio: ([\d.,;]+)", response)
    if split_ratio_match:
        final_split_ratio = split_ratio_match.group(1)
        #print("Final split ratio:", final_split_ratio)
    else:
        print("Final split ratio not found.")
    # Find Regioanl Prompt
    prompt_match = re.search(r"Regional Prompt: (.*?)(?=\n \n|\Z)", response, re.DOTALL)
    if prompt_match:
        regional_prompt = prompt_match.group(1).strip()
        #print("Regional Prompt:", regional_prompt)
    else:
        print("Regional Prompt not found.")

    image_region_dict = {'Final split ratio': final_split_ratio, 'Regional Prompt': regional_prompt}    
    return image_region_dict
