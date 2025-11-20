from RegionalDiffusion_base import RegionalDiffusionPipeline
from RegionalDiffusion_xl import RegionalDiffusionXLPipeline
from diffusers.schedulers import KarrasDiffusionSchedulers,DPMSolverMultistepScheduler
from transformers import MllamaForConditionalGeneration, PreTrainedTokenizerFast
from mllm import local_llm,GPT4
import random
import torch
import os
pipe = RegionalDiffusionXLPipeline.from_single_file("../mod/albedobaseXL_v20.safetensors", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config,use_karras_sigmas=True)
pipe.enable_xformers_memory_efficient_attention()

negative_prompt = "NSFW,worst quality,low quality,normal quality,lowres,monochrome,grayscale,skin spots,acnes,skin blemishes,age spot,ugly,duplicate,morbid,mutilated,tranny,mutated hands,poorly drawn hands,blurry,bad anatomy,bad proportions,extra limbs,disfigured,missing arms,extra legs,fused fingers,too many fingers,unclear eyes,lowers,bad hands,missing fingers,extra digit,bad hands,missing fingers,extra arms and legs"
save_dir = '/root/autodl-tmp/SPADE-Diffusion-main/bakimg'

for i in range(1, 2):
    prompt = "A serene, balanced room filled with soft daylight. Smooth white walls reflect quiet brightness, creating a pure, spacious feeling of calm and understated warmth."
    para_dict = GPT4(i,key='sk-UDZJFwAlNjn5bnQOOEJn8W7JXvm225kzvOXZARvVtuiQnPnR')
    split_ratio = para_dict['Final split ratio']
    split_ratio_all = para_dict['Final split ratio all']
    regional_prompt = para_dict['Regional Prompt']
    for j in range(1, 2):
        random_seed = random.randint(1, 10000000)
        image_filename = f"{prompt}_{j}.png"
        images = pipe(
            prompt=regional_prompt,
            split_ratio=split_ratio,
            split_ratio_all=split_ratio_all,
            split_ratio_allcp=split_ratio_all,
            img_filename=image_filename,
            batch_size = 2,
            base_ratio = 0.2,
            base_prompt= prompt,
            num_inference_steps=20,
            height = 2048,
            negative_prompt=negative_prompt,
            width = 2048,
            seed = random_seed,
            guidance_scale = 7.0
        ).images[0]
        save_path = os.path.join(save_dir, image_filename)
        images.save(save_path)
