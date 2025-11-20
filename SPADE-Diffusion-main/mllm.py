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
    print('yuanming test 1119 waiting for GPT-4 response')
    with open(f'paras/spatialtst.txt', 'r', encoding='utf-8') as file:
        text = file.read()
    return get_params_dict(text)

def local_llm(prompt,mod=None,tok=None,mod_input=None,txtpro=None):
    mod.eval()
    with torch.no_grad():
        res = mod.generate(**mod_input, max_new_tokens=1024)[0]
        output=tok.decode(res, skip_special_tokens=True)
        output = output.replace(txtpro,'')
        torch.cuda.empty_cache()
    return get_params_dict(output)

def get_params_dict(output_text):
    response = output_text
    split_ratio_match = re.search(r"Final split ratio: ([\d.,;]+)", response)
    if split_ratio_match:
        final_split_ratio = split_ratio_match.group(1)
    else:
        print("Final split ratio not found.")
    #split_ratio_r_match = re.search(r"Final split ratio r: ([\d.,;]+)", response)
    #if split_ratio_r_match:
    #    final_split_ratio_r = split_ratio_r_match.group(1)
    #else:
    #    print("Final split ratio r not found.")
    split_ratio_all_match = re.search(r"Final split ratio all: ([\d.,;]+)", response)
    if split_ratio_all_match:
        final_split_ratio_all = split_ratio_all_match.group(1)
    else:
        print("Final split ratio r not found.")
    prompt_match = re.search(r"Regional Prompt:\s*([^\n]*)", response)
    if prompt_match:
        regional_prompt = prompt_match.group(1).strip()
    else:
        print("Regional Prompt not found.") 
    image_region_dict = {'Final split ratio': final_split_ratio, 'Final split ratio all': final_split_ratio_all, 'Regional Prompt': regional_prompt} 
    return image_region_dict
