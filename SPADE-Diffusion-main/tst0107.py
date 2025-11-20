from RegionalDiffusion_base import RegionalDiffusionPipeline
from RegionalDiffusion_xl import RegionalDiffusionXLPipeline
from diffusers.schedulers import KarrasDiffusionSchedulers,DPMSolverMultistepScheduler
from transformers import MllamaForConditionalGeneration, PreTrainedTokenizerFast
from mllm import local_llm,GPT4
import random
import torch

pipe = RegionalDiffusionXLPipeline.from_single_file("../mod/albedobaseXL_v20.safetensors", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
#print("\npipe的格式是：",type(pipe))
pipe.to("cuda")
#print("\n打印一下配置::",pipe.scheduler.config)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config,use_karras_sigmas=True)
#print("\n再打印一次scheduler::",pipe.scheduler)
pipe.enable_xformers_memory_efficient_attention()

prompt= 'a boy on the top of a wallet'

para_dict = GPT4(prompt,key='sk-UDZJFwAlNjn5bnQOOEJn8W7JXvm225kzvOXZARvVtuiQnPnR')

split_ratio = para_dict['Final split ratio']    #获取字典中的Final split ratio:0.5,0.5,0.5;0.5,0.5,0.5
#print(split_ratio)

regional_prompt = para_dict['Regional Prompt']      #获取字典中的Regional Prompt：每个区域的prompt
#print(regional_prompt)

negative_prompt = "NSFW,worst quality,low quality,normal quality,lowres,monochrome,grayscale,skin spots,acnes,skin blemishes,age spot,ugly,duplicate,morbid,mutilated,tranny,mutated hands,poorly drawn hands,blurry,bad anatomy,bad proportions,extra limbs,disfigured,missing arms,extra legs,fused fingers,too many fingers,unclear eyes,lowers,bad hands,missing fingers,extra digit,bad hands,missing fingers,extra arms and legs"    #没有negative

random_seed = random.randint(0, 100000)
images = pipe(
    prompt=regional_prompt,
    split_ratio=split_ratio, # The ratio of the regional prompt, the number of prompts is the same as the number of regions
    batch_size = 2, #batch size
    base_ratio = 0.5, # The ratio of the base prompt
    base_prompt= prompt,
    num_inference_steps=20, # sampling step
    #height = 1024,
    height = 1600,    #调个特殊值试试
    negative_prompt=negative_prompt, # negative prompt
    #width = 1024,
    width = 1600,    #调个特殊值试试
    seed = random_seed,# random seed
    guidance_scale = 7.0
).images[0]

images.save("test1221.png")
